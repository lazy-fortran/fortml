# Poisson MLP objective

`fortml_mlp_poisson` provides a smooth Poisson negative log likelihood for a
one-output MLP. The network output is a log rate. Targets are finite and
non-negative, and optional row weights use the positive-weight mean reduction.

```fortran
use fortml_mlp_poisson, only: mlp_poisson_training_objective_t

type(mlp_poisson_training_objective_t) :: objective
call objective%initialize(model, x, counts, 1.0e-3_dp, status, &
    optimize_l2=.true., sample_weight=weights)
parameters = objective%parameters()
call objective%value_gradient(parameters, value, gradient, status)
call objective%hvp(parameters, direction, product, status)
```

The optional final packed coordinate is the L2 coefficient. Its gradient is
half the squared network norm, and the HVP contains the mixed network/L2
block. The Poisson output curvature and MLP reverse-over-forward product are
analytic. `mlp_poisson_optimize_lbfgsb` sends the same value and gradient to
FortOpt's bounded L-BFGS-B implementation.

The objective is CPU-only. A CUDA request returns `FORTNUM_NOT_IMPLEMENTED`
before model state is installed. `test_mlp_poisson_objective` checks value and
HVP central differences, JVP contraction, VJP scaling, optimizer convergence,
and the typed device boundary.
