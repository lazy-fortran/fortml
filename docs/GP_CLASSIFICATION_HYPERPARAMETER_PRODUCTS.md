# Laplace GP-classification hyperparameter products

`gp_classification_t` exposes objective products for the converged binary
Laplace classifier.  The packed coordinate order is the kernel's logarithmic
parameter order returned by `model%parameters()`.

```fortran
real(dp) :: direction(2), value, tangent, cotangent
real(dp) :: parameter_bar(2)
call model%hyperparameter_jvp(direction, value, tangent, status)
call model%hyperparameter_vjp(cotangent, parameter_bar, status)
```

The scalar value is the mode log-posterior at the current fitted state.  Its
JVP is the envelope-gradient contraction
`dot_product(direction, model%hyperparameter_gradient())`; the Newton mode is
not differentiated for this scalar objective.  This is the same objective
consumed by the FortOpt L-BFGS-B adapter in
`fortml_gp_classification_training`.  For a prediction whose fitted mode,
posterior curvature, and variance all need to be differentiated, use
`predict_latent_hyperparameter_jvp` (and the probability wrapper) instead.
The existing `hyperparameter_hvp` differentiates this envelope through the
implicit mode equation and is suitable for second-order outer products.

Both products have explicit device dispatch:

- CPU dispatch executes the analytic products.
- CUDA returns `FORTNUM_NOT_IMPLEMENTED` until the Cholesky factorization,
  Newton mode, and their derivative graph are resident.  No hidden host
  fallback is performed.

The independent behavioral oracle is
`test/test_gp_classification_hyperparameter_products.f90`.  It compares the
analytic JVP with a central finite difference of two independently refitted
logistic and probit classifiers, checks the JVP/VJP contraction identities,
and verifies the CPU dispatch plus typed CUDA refusals.
