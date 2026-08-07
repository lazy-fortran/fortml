# Coupled-L2 Adam trajectory hypergradients

`fortml_mlp_adam_hypergradient` exposes an exact deterministic, fixed
full-batch Adam trajectory objective for outer hyperparameter search.  It uses
the coupled-L2 convention used by Adam: `l2*theta` is included in the loss
gradient before both moment recurrences, and no decoupled parameter shrinkage
is applied.

The packed outer vector is

```text
[ log(learning_rate), log(l2), logit(beta1), logit(beta2) ]
```

The logits map smoothly to `(0,1)`, so every bounded L-BFGS-B iterate has a
valid moment coefficient.  The initial MLP parameters are captured at
`initialize`; each evaluation starts from that state, performs exactly
`steps` full-batch updates, and returns unregularized validation MSE.

`value_gradient` and `jvp` propagate state tangents through the parameters,
first moments, second moments, bias corrections, and epsilon denominator.  The
derivative of each training gradient comes from `mlp_loss_hvp`, including the
mixed `log(l2)` direction.  `vjp` is the scalar adjoint identity and
`fortopt` adapts the objective directly to FortOpt's `objective_t` callback.
`mlp_optimize_adam_hyperparameters` supplies explicit bounds and reports both
the packed coordinates and their physical values.

```fortran
use fortml_mlp_adam_hypergradient, only: &
    mlp_adam_hypergradient_options_t, mlp_adam_hypergradient_result_t, &
    mlp_optimize_adam_hyperparameters

options%steps = 16
options%learning_rate = 1.0e-2_dp
options%l2 = 1.0e-4_dp
options%beta1 = 0.9_dp
options%beta2 = 0.999_dp
options%lower_log_learning_rate = -12.0_dp
options%upper_log_learning_rate = 1.0_dp
options%lower_log_l2 = -20.0_dp
options%upper_log_l2 = 0.0_dp
call mlp_optimize_adam_hyperparameters(model, train_x, train_target, &
    validation_x, validation_target, options, result, status)
```

The independent `test_mlp_adam_hypergradient` fixture checks all four
coordinates against central differences, a directional JVP, the scalar VJP
identity, a bounded FortOpt solve, and the explicit CUDA refusal.  Mini-batch,
scheduled, stochastic, mixed-precision, and resident-device Adam trajectories
remain separate contracts until their state and reproducibility derivatives are
available; this path never silently falls back to finite differences.
