# Gamma GP likelihood

`gamma_likelihood_t` models a positive observation with a Gamma density. The GP
latent is the log mean, `mean = exp(latent)`, and the trainable likelihood
coordinate is `log_shape`. This parameterization keeps the mean and shape
positive for every finite optimizer iterate.

For observation `y`, latent `f`, and shape `a = exp(log_shape)`, the implemented
log density is

```text
a log(a) - log_gamma(a) + (a - 1) log(y) - a f - a y exp(-f).
```

`gamma_log_likelihood_value_gradient` returns the weighted sum, one derivative
per latent, and the derivative with respect to `log_shape`. The JVP, VJP, and
HVP procedures use the same joint coordinate space. The HVP includes the mixed
latent-shape block, so outer optimization can propagate a likelihood
hyperparameter direction into the GP latent coordinates. Sample weights must
be finite and nonnegative, with positive total mass.

The object holds observations and latents fixed and exposes the transformed
shape through `parameters`, `set_parameters`, `value_gradient`, `jvp`, `vjp`,
and `hvp`. `fortopt` returns the same negative-log-likelihood objective to any
FortOpt solver. `optimize_lbfgsb` applies explicit bounds to `log_shape` and
uses FortOpt L-BFGS-B. Invalid setters and failed optimizer runs preserve the
last accepted shape.

CPU execution is supported. A CUDA initialization returns
`FORTNUM_NOT_IMPLEMENTED` and allocates no model state. Resident CUDA support
requires device implementations of log-gamma, digamma, and trigamma together
with the weighted batch reductions.

`test_gamma_likelihood` checks the density and every derivative product against
an independent scalar formula and central differences. It also checks the
JVP/VJP adjoint identity, weighted observations, FortOpt adaptation, the
L-BFGS-B result against a SciPy reference optimum, rollback, and CUDA refusal.
`fortml_bench_gamma_likelihood` emits product throughput, fitted shape,
optimizer diagnostics, and the CUDA status for release evidence.
