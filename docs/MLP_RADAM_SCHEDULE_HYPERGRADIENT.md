# Scheduled RAdam trajectory hypergradients

`fortml_mlp_radam_schedule_hypergradient` differentiates a deterministic,
fixed full-batch RAdam trajectory while evaluating a typed learning-rate
schedule at every update. The packed outer vector is

```text
[ log(base_rate), log(l2), logit(beta1), logit(beta2), log(epsilon),
  logit(min_rate_fraction), logit(decay_factor) ]
```

Schedule kind and integer update counts are fixed. Constant, cosine,
warmup-cosine, and exponential-decay schedules are supported; inactive
continuous schedule fields have exact zero products. The value/gradient, JVP,
scalar VJP, and FortOpt L-BFGS-B products propagate RAdam parameter, moment,
bias-correction, rectification, schedule-rate, and schedule-parameter
sensitivities without finite-differencing an inner run.

The CPU contract is explicit. CUDA requests return `FORTNUM_NOT_IMPLEMENTED`
until resident MLP, schedule, and RAdam state are available. The `rho_t = 4`
branch, zero second-moment square root, and outer hyper-HVP (which requires
third network derivatives) return typed refusals rather than hidden
subgradients or finite-difference approximations. The independent
`test_mlp_radam_schedule_hypergradient` fixture checks cosine minimum-rate and
exponential decay-factor central differences, directional JVP, scalar VJP,
FortOpt integration, malformed/device refusals, and the outer-HVP refusal.

```fortran
use fortml_mlp_schedules, only: make_mlp_schedule_cosine_decay
use fortml_mlp_radam_schedule_hypergradient, only: &
    mlp_radam_schedule_hypergradient_options_t, &
    mlp_radam_schedule_hypergradient_objective_t

options%schedule = make_mlp_schedule_cosine_decay(100, 0.05_dp)
options%base_rate = 1.0e-2_dp
options%l2 = 1.0e-4_dp
call objective%initialize(model, train_x, train_target, validation_x, &
    validation_target, options, status)
call objective%value_gradient(objective%parameters(), value, gradient, status)
```
