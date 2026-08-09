# FortOpt training for multi-output GPs

`fortml_multi_output_gp_training` exposes exact hyperparameter optimization for
the intrinsic-coregionalization model (`multi_output_gp_t`).  It is the
multi-output counterpart of `fortml_gp_training`: each FortOpt L-BFGS-B trial
updates the packed model state transactionally, refactors the joint covariance,
and evaluates the analytic negative log marginal likelihood and gradient.

```fortran
use fortml_multi_output_gp_training, only: &
    multi_output_gp_hyperparameter_options_t, &
    multi_output_gp_hyperparameter_result_t, &
    multi_output_gp_optimize_hyperparameters

type(multi_output_gp_hyperparameter_options_t) :: options
type(multi_output_gp_hyperparameter_result_t) :: result
call multi_output_gp_optimize_hyperparameters(model, options, result, status)
```

The packed coordinates are `[kernel parameters, log noise variance, W,
independent variances]`.  Bounds are typed by block.  Kernel and noise
coordinates are log parameters, latent-output weights are unconstrained, and
independent variances have a non-negative lower bound.  `starts`, `seed`, and
`include_current` provide deterministic multistart selection.  Only finite
converged runs compete for retention.  The best state is restored at the
end.

The adapter uses the same likelihood gradient and Hessian-vector products as
the model, so an outer HPO implementation does not finite-difference a
coregionalization parameter.  The public model products remain available for
JVP/VJP/HVP composition around the retained state.

Exact ICM factorization and the optimizer are CPU reference implementations.
Passing a selected CUDA device returns `FORTNUM_NOT_IMPLEMENTED` before any
state mutation.  There is no hidden host fallback.  A resident CUDA path still
requires device covariance assembly, factorization, reductions, and their
derivative kernels.

`test_multi_output_gp_training.f90` assembles the joint covariance and LML with
an independent dense Gaussian-elimination oracle, checks the optimizer result
against the fitted likelihood, and verifies the CUDA capability boundary.
