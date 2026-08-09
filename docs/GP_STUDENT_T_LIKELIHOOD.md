# Fixed-latent Student-t likelihood products

`student_t_likelihood_t` is the scalar observation-likelihood seam for a GP
latent batch. The latent locations are held fixed by the inference method; the
likelihood coordinates are the unconstrained pair

```text
theta = [ log(scale), log(nu) ]
```

so every optimizer iterate has positive scale and degrees of freedom. The
public free procedures return the normalized, summed log density for paired
`observations` and `locations`:

```text
log p(y | f, scale, nu) =
  lgamma((nu + 1)/2) - lgamma(nu/2) - log(scale)
  - 1/2 log(nu*pi)
  - (nu + 1)/2 log(1 + (y - f)^2/(nu*scale^2)).
```

`student_t_log_likelihood_value`, `..._gradient`, `..._jvp`, `..._vjp`, and
`..._hvp` provide analytic products with respect to `theta`. The object method
`value_gradient` returns the negative log likelihood, so its
`fortopt(objective,status)` context can be passed directly to FortOpt
L-BFGS-B. `set_parameters` validates before committing, which keeps rejected
line-search candidates from changing the fitted likelihood state.

This is a fixed-latent-state contract. It does not differentiate a GP mode,
kernel factorization, or covariance hyperparameter; those belong to the
inference-specific GP adapters. The implementation is a dense CPU reference
path and uses analytic digamma/trigamma asymptotics for the degrees-of-freedom
products. CUDA initialization is an explicit `FORTNUM_NOT_IMPLEMENTED`
refusal until a resident batch and special-function implementation is linked;
there is no hidden host fallback.

`test_student_t_likelihood` compares the implementation against an
independent scalar density with central differences, checks the JVP/VJP
adjoint identity and directional HVP, verifies FortOpt callback parity and
transactional setters, and checks the typed CUDA and malformed-parameter
boundaries. The release probe is
`fortml_bench_student_t_likelihood_products`.
