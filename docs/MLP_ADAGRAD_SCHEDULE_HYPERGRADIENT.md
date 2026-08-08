# Scheduled Adagrad trajectory hypergradients

`fortml_mlp_adagrad_schedule_hypergradient` differentiates a deterministic,
fixed full-batch Adagrad trajectory while evaluating a typed learning-rate
schedule at every update. The packed outer variables are
`[log(base_rate), log(l2), log(epsilon), logit(min_rate_fraction),
logit(decay_factor)]`. Schedule kind and integer update counts are fixed;
inactive continuous schedule fields have exact zero products.

```fortran
use fortml_mlp_schedules, only: make_mlp_schedule_cosine_decay
use fortml_mlp_adagrad_schedule_hypergradient, only: &
    mlp_adagrad_schedule_hypergradient_options_t, &
    mlp_adagrad_schedule_hypergradient_objective_t

options%schedule = make_mlp_schedule_cosine_decay(100, 0.05_dp)
options%base_rate = 1.0e-2_dp
options%l2 = 1.0e-4_dp
options%epsilon = 1.0e-8_dp
call objective%initialize(model, train_x, train_target, validation_x, &
    validation_target, options, status)
call objective%value_gradient(objective%parameters(), value, gradient, status)
```

The objective provides exact value/gradient, JVP, VJP, and a FortOpt context
adapter. Its tangent recurrence differentiates the accumulated-square state,
the schedule rate, and the Adagrad update; it does not finite-difference an
optimizer trajectory. The implementation is CPU-only. CUDA requests return a
typed `FORTNUM_NOT_IMPLEMENTED` status until resident MLP, schedule, and
derivative state are available. `test_mlp_adagrad_schedule_hypergradient`
checks central-difference and directional products, scalar VJP adjointness,
FortOpt integration, malformed options, and the CUDA refusal.
