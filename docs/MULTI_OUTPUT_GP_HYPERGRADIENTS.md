# ICM multi-output GP likelihood products

`multi_output_gp_t` now exposes the exact hyperparameter-product contract
needed to optimize an intrinsic-coregionalization model (ICM). The packed
parameter vector is

```text
[ kernel parameters,
  log observation-noise variance,
  output-major coregionalization weights W,
  independent output variances ]
```

The model stores the training targets after `fit`, so `set_parameters` can
refactor the joint Cholesky state without asking the caller to repeat the
data. Candidate updates are transactional: an invalid variance, unsupported
kernel product, or failed factorization restores the previous parameters and
posterior state.

## Products

The implementation forms the exact dense ICM covariance

```text
A = B (x) K + sigma^2 I,       B = W W^T + diag(independent)
```

and contracts the usual Gaussian-process identity

```text
d log p(y) / d theta = 1/2 trace((alpha alpha^T - A^-1) dA/dtheta).
```

`hyperparameter_gradient` covers kernel, noise, weights, and independent
coordinates. `log_marginal_likelihood_jvp` and
`log_marginal_likelihood_vjp` provide scalar composition. The analytic
`hyperparameter_hvp` propagates the covariance directional solve and uses the
kernel's generated parameter/mixed products. Kernels without generated second
products return `FORTNUM_NOT_IMPLEMENTED`; no finite-difference fallback is
hidden in the production path.

`sample_count`, `feature_count`, and `output_count` expose the state shape
explicitly for pipeline and benchmark code.

## Devices and benchmark

CPU is the reference implementation. The device wrappers preserve output
clearing and return a typed `FORTNUM_NOT_IMPLEMENTED` for CUDA until a
resident ICM covariance and factorization path is linked. This is an explicit
capability boundary, not a silent host fallback.

The release app `fortml_bench_multi_output_gp_hypergradients` measures the
gradient and HVP on a three-output, two-latent RBF ICM state. The independent
finite-difference oracle is
`test_multi_output_gp_hypergradients.f90`; it checks the gradient, HVP,
transaction rollback, scalar JVP/VJP composition, shape metadata, and typed
CUDA refusals.
