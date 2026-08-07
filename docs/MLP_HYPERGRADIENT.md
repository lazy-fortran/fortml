# MLP trajectory hypergradients

`fortml_mlp_hypergradient` provides a small, exact outer-search contract for
MLP training. It is intentionally narrower than the general MLP trainer: the
inner optimizer is deterministic, full-batch gradient descent with no momentum
and a fixed number of steps.

The packed outer vector has a stable two-entry layout:

```text
[ log_learning_rate, log_l2 ]
```

The model parameters present when `initialize` is called are the fixed inner
initial state. Each value evaluation starts from that state, applies exactly
`steps` updates on the training arrays, and leaves the model at the resulting
final state. The scalar objective is the unregularized validation MSE. Thus
validation rows affect neither an inner update nor the L2 penalty.

## Exact products

The forward `jvp` propagates a parameter tangent through every update. It uses
the MLP analytic Hessian-vector product to differentiate the training gradient;
the log transform contributes `d learning_rate = learning_rate*d log_rate` and
`d l2 = l2*d log_l2`.

`value_gradient` and `vjp` use a reverse adjoint through the same stored
trajectory. They are exact products of the implemented MLP and loss equations;
there is no finite-difference fallback. The independent test
`test_mlp_hypergradient` compares both products with central differences and
checks the scalar VJP adjoint identity.

## FortOpt search

```fortran
use fortml_mlp_hypergradient, only: &
    mlp_hypergradient_options_t, mlp_hypergradient_result_t, &
    mlp_optimize_hyperparameters

type(mlp_hypergradient_options_t) :: options
type(mlp_hypergradient_result_t) :: result

options%steps = 16
options%learning_rate = 0.01_dp
options%l2 = 1.0e-4_dp
options%lower_log_learning_rate = -12.0_dp
options%upper_log_learning_rate = 1.0_dp
options%lower_log_l2 = -20.0_dp
options%upper_log_l2 = 0.0_dp
call mlp_optimize_hyperparameters(model, train_x, train_target, &
    validation_x, validation_target, options, result, status)
```

The adapter hands the analytic value/gradient callback directly to FortOpt
L-BFGS-B and reports the optimized log parameters and their exponentiated
values. Bounds are in log space and the initial positive values must lie inside
them.

Adam, momentum/Nesterov SGD, and CUDA/device-resident trajectories are refused
with `FORTNUM_NOT_IMPLEMENTED`; silently differentiating a different optimizer
would invalidate the hypergradient. Extending this contract to schedules,
mini-batches, stochastic state, and resident device buffers requires separate
derivative and reproducibility products.

## AdamW trajectory contract

`mlp_adamw_hypergradient_objective_t` applies the same fixed full-batch
validation objective through bias-corrected AdamW. Its packed outer vector is

```text
[ log_learning_rate, log_l2, log_weight_decay ]
```

The first and second moment recurrences, decoupled weight decay, and each
log-parameter sensitivity are propagated analytically using the MLP HVP. The
object exposes exact `value_gradient`, `jvp`, and scalar `vjp` products;
`mlp_optimize_adamw_hyperparameters` sends them to FortOpt L-BFGS-B with
independent log bounds. The behavioral test
`test_mlp_adamw_hypergradient` checks all three components against central
differences, the JVP, the scalar adjoint, and an L-BFGS-B solve. Mini-batch,
schedule, beta, and CUDA AdamW trajectories remain explicit follow-up work.
