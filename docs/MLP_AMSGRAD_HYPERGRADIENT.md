# AMSGrad trajectory hypergradients

`fortml_mlp_amsgrad_hypergradient` exposes a deterministic, fixed full-batch
AMSGrad training trajectory as a FortOpt objective.  The packed search vector is

```text
[ log(learning_rate), log(l2), logit(beta1), logit(beta2), log(epsilon) ]
```

The forward product propagates parameter, first-moment, second-moment, and
max-second-moment tangents through every update, including bias correction.
`value_gradient` returns the validation MSE and its
five-coordinate gradient; `jvp` and scalar `vjp` are the corresponding exact
products.  `fortopt` supplies the same objective registry to FortOpt
L-BFGS-B, and `mlp_optimize_amsgrad_hyperparameters` is a bounded convenience
wrapper.

The implementation is intentionally explicit about its derivative boundary.
The max-second-moment active-set boundary and a zero second-moment square root
are nonsmooth. Products at those points return `FORTNUM_NOT_IMPLEMENTED`.
Zero bias-correction or update denominators use the same typed refusal. CUDA
requests are also a typed
`FORTNUM_NOT_IMPLEMENTED` until a resident AMSGrad trajectory kernel is linked.
The independent `test_mlp_amsgrad_hypergradient` fixture checks coordinate
central differences, a directional finite difference, and the scalar VJP
adjoint identity, then exercises the FortOpt adapter and both refusal paths.

```fortran
use fortml_mlp_amsgrad_hypergradient, only: &
    mlp_amsgrad_hypergradient_objective_t, &
    mlp_amsgrad_hypergradient_options_t

type(mlp_amsgrad_hypergradient_options_t) :: options
type(mlp_amsgrad_hypergradient_objective_t) :: objective
call objective%initialize(model, train_x, train_y, valid_x, valid_y, options, status)
parameters = objective%parameters()
call objective%value_gradient(parameters, value, gradient, status)
```

This slice is CPU-only by contract and uses the existing exact MLP loss HVP;
no finite differences or hidden optimizer fallback are used in production
code.
