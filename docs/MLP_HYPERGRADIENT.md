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
validation objective through bias-corrected AdamW. Its legacy packed outer vector is

```text
[ log_learning_rate, log_l2, log_weight_decay ]
```

The first and second moment recurrences, decoupled weight decay, and each
log-parameter sensitivity are propagated analytically using the MLP HVP. The
object exposes exact `value_gradient`, `jvp`, and scalar `vjp` products;
`mlp_optimize_adamw_hyperparameters` sends them to FortOpt L-BFGS-B with
independent log bounds.

For differentiable moment coefficients use
`mlp_adamw_full_hypergradient_objective_t`, whose packed vector is

```text
[ log_learning_rate, log_l2, log_weight_decay,
  logit_beta1, logit_beta2 ]
```

The full objective propagates beta-logit sensitivities through bias correction
and also exposes an analytic outer `hvp` on a one-layer linear MLP. That HVP
uses the constant affine loss Hessian and returns
`FORTNUM_NOT_IMPLEMENTED` for nonlinear or multilayer networks rather than
approximating third derivatives. `test_mlp_adamw_hypergradient` checks the
HVP against an independent central-difference derivative of the exact
trajectory gradient, while `fortml-bench/results/ADAMW_BETA_HYPERGRADIENT.md`
records the NumPy value, gradient, JVP, and HVP oracle. CUDA and lower
precision AdamW hypergradient trajectories remain explicit typed refusals.

## Scheduled AdamW trajectory contract

`mlp_adamw_schedule_hypergradient_objective_t` extends the fixed AdamW
objective with constant, cosine, warmup-cosine, and exponential-decay typed
schedules. The eight packed coordinates are
`[log(base_rate), log(l2), log(weight_decay), logit(beta1), logit(beta2),
log(epsilon), logit(min_rate_fraction), logit(decay_factor)]`. Exact CPU
value/gradient, JVP, scalar VJP, and FortOpt L-BFGS-B products propagate the
complete AdamW moment and decoupled-decay state. The fixture and benchmark use
an independent central-difference/NumPy affine oracle. CUDA, lower precision,
zero-square-root derivatives, and outer hyper-HVP requests are explicit typed
refusals; no host fallback or hidden finite difference is used.

## RMSprop trajectory contract

`mlp_rmsprop_hypergradient_objective_t` differentiates a fixed full-batch
RMSprop trajectory with the FortOpt recurrence. Its packed outer vector is

```text
[ log_learning_rate, log_l2, decay, log_epsilon, momentum ]
```

The square-average, optional centered gradient-average, and momentum buffer are
propagated together with the network tangent. Consequently `value_gradient`,
`jvp`, and scalar `vjp` are exact for both centered and uncentered modes, with
the MLP analytic HVP supplying the derivative of each training gradient. The
`centered` option is a fixed discrete branch; it is intentionally not packed
or differentiated. `mlp_optimize_rmsprop_hyperparameters` sends the analytic
products to FortOpt L-BFGS-B using bounds for all five packed variables.

The independent `test_mlp_rmsprop_hypergradient` fixture checks every packed
component against central differences, the directional JVP, scalar VJP, both
centered branches, the L-BFGS-B adapter, and typed refusal for unsupported
optimizer/device choices. Mini-batch, schedules, clipping, and CUDA-resident
RMSprop state remain separate contracts until their state and reproducibility
derivatives are implemented.

## Adagrad trajectory contract

`mlp_adagrad_hypergradient_objective_t` differentiates a fixed full-batch
Adagrad trajectory. Its packed outer vector is

```text
[ log_learning_rate, log_l2, log_epsilon ]
```

The accumulated-square recurrence and epsilon-stabilized diagonal step are
propagated with the MLP analytic HVP. `value_gradient`, `jvp`, and scalar `vjp`
are exact products, and `mlp_optimize_adagrad_hyperparameters` sends the same
callback to FortOpt L-BFGS-B under explicit log bounds. The independent
`test_mlp_adagrad_hypergradient` fixture checks central differences, a
directional JVP, the scalar adjoint, optimizer convergence, and typed refusal
for unsupported optimizer/device choices. Mini-batch, schedules, clipping,
and CUDA-resident Adagrad state remain separate contracts until their state and
reproducibility derivatives are implemented.

## Unfactored Adafactor trajectory contract

`mlp_adafactor_hypergradient_objective_t` differentiates a fixed full-batch
unfactored Adafactor trajectory. Its packed outer vector is

```text
[ log_learning_rate, log_l2, decay, log_epsilon, log_clip_threshold ]
```

The squared-gradient state, update-RMS clipping, and stabilized denominator are
propagated with the MLP analytic HVP. Setting `relative_step` enables the
deterministic `min(learning_rate, 1/sqrt(step))` rate branch; setting
`scale_parameter` multiplies that rate by `max(parameter_rms, epsilon)`. Both
branches include exact smooth-state JVP/VJP products and are accepted by the
FortOpt L-BFGS-B adapter. A trajectory that lands on a clip, relative-rate, or
parameter-scale transition returns `FORTNUM_NOT_IMPLEMENTED`, preserving the
fixed active-set contract. Matrix-factored state and CUDA remain explicit
refusals. `test_mlp_adafactor_hypergradient` independently checks central
differences for the relative-step and parameter-scale branches, as well as
the CUDA refusal.
