# Piecewise-smooth Lion trajectory hypergradients

`fortml_mlp_lion_hypergradient` provides a deterministic, fixed full-batch
Lion trajectory objective for tuning an MLP's optimizer coordinates with
FortOpt. The packed outer vector is

```text
[ log(learning_rate), log(l2), logit(beta1), logit(beta2) ]
```

Each evaluation restores the parameters captured by `initialize`, performs
exactly `steps` Lion updates, and returns unregularized validation MSE. The
recurrence is the usual Lion state update: the interpolated first-moment
candidate is signed for the parameter step, then the second momentum
coefficient updates the state. `value_gradient` and `jvp` propagate analytic
MLP Hessian-vector products through the parameters and both momentum
coefficients; `vjp` is the scalar adjoint identity. The FortOpt adapter consumes
the same callback and searches bounded log/logit coordinates.

Lion's sign map is nonsmooth. Products are therefore defined only on a fixed
nonzero sign branch. A configurable `sign_margin` detects an update near zero
and returns `FORTNUM_NOT_IMPLEMENTED` rather than reporting a false
derivative. Crossing a sign boundary requires a new discrete branch and is
not silently finite-differenced. CUDA and resident optimizer state are also
explicit typed refusals until the complete trajectory is lowered to a device.

```fortran
use fortml_mlp_lion_hypergradient, only: &
    mlp_lion_hypergradient_options_t, &
    mlp_optimize_lion_hyperparameters

options%steps = 16
options%learning_rate = 1.0e-3_dp
options%l2 = 1.0e-4_dp
options%beta1 = 0.9_dp
options%beta2 = 0.99_dp
call mlp_optimize_lion_hyperparameters(model, train_x, train_target, &
    validation_x, validation_target, options, result, status)
```

`test_mlp_lion_hypergradient` is an independent behavioral oracle. It checks
central differences away from sign boundaries, JVP contraction, scalar VJP
scaling, bounded FortOpt result coordinates, CUDA refusal, and the explicit
nondifferentiable-branch refusal.
