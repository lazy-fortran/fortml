# Scheduled MLP hypergradients

`fortml_mlp_schedule_hypergradient` differentiates a deterministic full-batch
MLP trajectory with a typed learning-rate schedule.  The continuous outer
variable is packed as

```text
[ log(base_rate), log(l2), logit(min_rate_fraction), logit(decay_factor) ]
```

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
base-rate, minimum-fraction, and decay-factor products.  Therefore the
`value_gradient`, `jvp`, and `vjp` paths share one objective and do not use
finite-difference or optimizer fallback code.

`mlp_optimize_schedule_hyperparameters` wraps the objective in FortOpt's
projected L-BFGS-B implementation.  Bounds are on the packed log/logit
coordinates, so every accepted point has positive learning rate and L2 and
schedule fractions in the open unit interval.  The integer schedule shape is
not silently changed by the optimizer.

CUDA is intentionally a typed `FORTNUM_NOT_IMPLEMENTED` refusal until a
resident MLP trajectory kernel is linked.  The refusal is tested separately
from CPU products; it is never represented as a host timing labelled CUDA.

The release workload is `fortml_bench_mlp_schedule_hypergradient`; the
independent benchmark harness records exact values, gradient components, and a
directional JVP before retaining timing rows.  A second-order hyper-HVP would
require third derivatives of the network loss and is kept as a separate
capability boundary rather than approximated with finite differences.
