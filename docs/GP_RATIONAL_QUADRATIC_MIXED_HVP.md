# Rational-quadratic mixed-observation GP HVPs

`gp_derivative_regression_t%hyperparameter_hvp` now supports mixed
value/first-derivative observations for `make_rational_quadratic_kernel`.
The packed coordinates are

```text
[log(variance), log(lengthscale), log(alpha), log(noise_variance)]
```

The CPU path differentiates the dense covariance and Cholesky solve exactly.
For `s = ||x1-x2||²`, `D = 1 + s/(2 alpha lengthscale²)`, and
`F = variance D**(-alpha)`, the radial products are

```text
F_s   = -F/(2 lengthscale² D)
F_ss  = F (alpha + 1)/(4 alpha lengthscale⁴ D²)
```

The value/gradient/Hessian observation blocks are assembled from these
products, and their parameter-direction products are analytic in all three
logarithmic kernel coordinates. No finite-difference fallback is present in
the production path. `test_derivative_gp_products` independently assembles
the one-dimensional covariance and checks the likelihood gradient and HVP by
central differences; it also checks query products and the selected-CUDA
typed refusal.

Resident CUDA covariance, factorization, and derivative-query kernels remain
an explicit `FORTNUM_NOT_IMPLEMENTED` boundary until a device graph and
transfer-inclusive oracle are available. The release benchmark is
[`DERIVATIVE_GP_RATIONAL_QUADRATIC_HVP.md`](../../fortml-bench/results/DERIVATIVE_GP_RATIONAL_QUADRATIC_HVP.md).
