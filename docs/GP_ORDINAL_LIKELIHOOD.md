# Native ordered-logit and ordered-probit likelihood products

`fortml_gp_ordinal_classification` exposes a backend-independent likelihood
primitive for native ordinal GP inference. It is deliberately separate from
`gp_ordinal_classification_t`, whose current fitted model is a latent-Gaussian
rank surrogate with fixed mid-rank cut points.

For latent scores `eta(i)`, rank labels `y(i) in {1,...,K}`, and strictly
increasing cut points `t(1:K-1)`, define `t(0)=-infinity` and `t(K)=+infinity`.
The selected likelihood has category probability

```
p(i) = F(t(y(i)) - eta(i)) - F(t(y(i)-1) - eta(i)),
```

where `F` is the logistic CDF for
`GP_ORDINAL_LIKELIHOOD_LOGISTIC` or the standard-normal CDF for
`GP_ORDINAL_LIKELIHOOD_PROBIT`. The value procedure returns
`sum(log(p(i)))`.

The JVP and VJP include both `eta` and cut-point coordinates. The HVP is an
analytic directional product of the full gradient, including the logistic or
normal density derivative. It is suitable for native likelihood objectives and
outer hyperparameter products without finite differences. Invalid ranks,
non-finite inputs, non-increasing thresholds, malformed tangents, and
probability underflow return `FORTNUM_DOMAIN_ERROR` or
`FORTNUM_CONVERGENCE_ERROR` without partial output state.

```fortran
call gp_ordinal_log_likelihood_value(eta, labels, thresholds, &
    GP_ORDINAL_LIKELIHOOD_PROBIT, value, status)
call gp_ordinal_log_likelihood_hvp(eta, labels, thresholds, &
    GP_ORDINAL_LIKELIHOOD_PROBIT, value_bar, eta_dot, thresholds_dot, &
    eta_hvp, thresholds_hvp, status)
```

The primitive is CPU-only until a resident CUDA likelihood/reduction kernel is
linked. Query `gp_ordinal_likelihood_device_supported` before selecting a
backend; CUDA returns false rather than hiding a host fallback. The independent
value, JVP/VJP adjoint, HVP finite-difference, transaction, and capability
oracle is `test_gp_ordinal_likelihood`. The release benchmark is
`fortml-bench/results/GP_ORDINAL_LIKELIHOOD.md`.
