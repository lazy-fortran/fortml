# Matérn mixed-observation likelihood HVPs

`gp_derivative_regression_t%hyperparameter_hvp` now supports mixed value and
first-derivative observations for the Matérn 3/2 and 5/2 leaves. The packed
coordinates are `[log(variance), log(lengthscale), log(noise_variance)]`, and
the result is the exact Hessian-vector product of the dense GP log marginal
likelihood in those coordinates.

The implementation uses the generated Matérn value/HVP kernels together with
closed-form radial products for `f(r)`, `f'(r)`, and `f''(r)`. Value/derivative
blocks are assembled from `f'(r)/r` and `(f''(r)-f'(r)/r)/r**2`; coincident
blocks use the finite Matérn limit `f''(0)`. No finite difference is present
in the production path. Sum/product dispatch remains exact when every child
supports the requested product.

The independent `test_derivative_gp_matern_hvp` fixture reconstructs the dense
mixed covariance from the Matérn closed forms and checks likelihood values,
packed gradients, and HVP projections against central differences for both
leaves. It also checks that selected CUDA prediction refuses with
`FORTNUM_NOT_IMPLEMENTED` before modifying caller outputs. Resident CUDA
covariance, factorization, and derivative-query kernels remain an explicit
boundary until a device graph and transfer-inclusive oracle are available.
