# Scheduled MLP hypergradients

`fortml_mlp_schedule_hypergradient` differentiates a deterministic full-batch
MLP trajectory with a typed learning-rate schedule.  The continuous outer
variable is packed as

```text
[ log(base_rate), log(l2), logit(min_rate_fraction), logit(decay_factor) ]
```

For `MLP_SCHEDULE_ONE_CYCLE`, the same four slots use positive logarithmic
peak/final-rate coordinates instead:

```text
[ log(base_rate), log(l2), log(peak_rate_fraction), log(final_rate_fraction) ]
```

The `MLP_SCHEDULE_LOG_PEAK_FRACTION` and
`MLP_SCHEDULE_LOG_FINAL_FRACTION` constants name these alternate slots, and
`metadata%one_cycle_coordinates` identifies the layout. The one-cycle domain
is `peak_rate_fraction >= 1` and `0 < final_rate_fraction <=
peak_rate_fraction`; invalid trial points return a typed domain error rather
than silently changing the schedule.

The schedule kind, warm-up count, and total update count are fixed discrete
configuration.  This keeps L-BFGS-B's search space smooth while still allowing
the schedule's continuous fields to be tuned.  Unused fields have an exact
zero derivative.

```fortran
use fortml_mlp_schedules, only: make_mlp_schedule_warmup_cosine
use fortml_mlp_schedule_hypergradient, only: &
    mlp_schedule_hypergradient_objective_t, &
    mlp_schedule_hypergradient_options_t

type(mlp_schedule_hypergradient_options_t) :: options
type(mlp_schedule_hypergradient_objective_t) :: objective
type(fortnum_status_t) :: status

options%steps = 32
options%base_rate = 1.0e-2_dp
options%l2 = 1.0e-4_dp
options%schedule = make_mlp_schedule_warmup_cosine(4, 32, 0.05_dp)
call objective%initialize(model, train_x, train_y, validation_x, validation_y, &
    options, status)
parameters = objective%parameters()
call objective%value_gradient(parameters, value, gradient, status)
call objective%jvp(parameters, direction, value, tangent, status)
call objective%vjp(parameters, output_bar, gradient_bar, status)
```

The reverse recurrence stores the parameter trajectory and applies the MLP
analytic HVP at each update.  The forward recurrence applies the same HVP to
the tangent state, including the exact chain rule through the schedule's
base-rate, minimum-fraction, and decay-factor products. For one-cycle
trajectories those final two products are the exact peak- and final-fraction
derivatives through the linear warm-up and cosine tail. Therefore the
`value_gradient`, `jvp`, and `vjp` paths share one objective and do not use
finite-difference or optimizer fallback code.

The type also exposes `hvp(parameters, direction, product, status)`.  For a
single affine layer and a constant schedule it is exact; see
`MLP_CONSTANT_SCHEDULE_HVP.md`.  Nonlinear networks return a typed
`FORTNUM_NOT_IMPLEMENTED` refusal because an outer hyper-HVP would require
third network derivatives.  Nonconstant schedules likewise refuse until their
rate second products are implemented.  Invalid shapes remain domain errors.
These boundaries are tested rather than replaced by hidden finite differences.

`mlp_optimize_schedule_hyperparameters` wraps the objective in FortOpt's
projected L-BFGS-B implementation.  Bounds are on the packed log/logit
coordinates, so every accepted point has positive learning rate and L2 and
schedule fractions in the open unit interval. For one-cycle schedules the
existing `lower_logit_min_fraction`/`upper_logit_min_fraction` and
`lower_logit_decay_factor`/`upper_logit_decay_factor` fields are interpreted
as bounds on the peak/final logarithms. The integer schedule shape is not
silently changed by the optimizer.

CUDA is intentionally a typed `FORTNUM_NOT_IMPLEMENTED` refusal until a
resident MLP trajectory kernel is linked.  The refusal is tested separately
from CPU products; it is never represented as a host timing labelled CUDA.

The release workload is `fortml_bench_mlp_schedule_hypergradient`; the
independent benchmark harness records exact values, gradient components, and a
directional JVP before retaining timing rows.  The affine constant-rate HVP
workload is `fortml_bench_mlp_constant_schedule_hvp`.  A second-order
hyper-HVP for nonlinear or nonconstant trajectories remains a separate
capability boundary rather than being approximated with finite differences.
