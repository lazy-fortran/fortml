# Categorical variational-GP likelihood products

`gp_variational_categorical_classification_t` now exposes the positive scalar
softmax likelihood temperature as a separate, transformed hyperparameter.  The
stored coordinate is `logit_scale`; its physical scale is
`likelihood_scale() = exp(logit_scale)`, with the default value one.  Existing
packed variational parameters remain unchanged: `parameters()` still contains
only inducing means and log-Cholesky entries, while
`likelihood_parameters()` returns the one-element likelihood vector.

The coupled categorical logits are

```
z[i, j] = exp(logit_scale) * m[i, j] / sqrt(1 + c * v[i, j])
```

where `c = pi/8` for the logistic approximation and `c = 1` for probit.  A
stable row-wise softmax produces the simplex probabilities.  The scalar
coordinate has exact CPU products:

- `predict_proba_likelihood_parameter_jvp` and
  `predict_proba_likelihood_parameter_vjp` differentiate the complete
  temperature-scaled softmax;
- `elbo_likelihood_parameter_gradient` and
  `elbo_likelihood_parameter_jvp` differentiate the weighted categorical ELBO,
  including the optional likelihood `scale` and sample weights; and
- `fit_likelihood` sends the same analytic value/gradient callback through
  bounded FortOpt L-BFGS-B without changing the inducing posterior.

`fit_likelihood` is transactional: malformed data/options, non-finite
products, or a non-converged solve restore the original log-scale.  The fit
state records convergence, iteration and line-search counts, ELBO, and the
final gradient.  Bounds are on the log coordinate, so the physical scale is
always positive.

The CPU implementation is the reference path.  The explicit device wrappers
for likelihood JVP/VJP return `FORTNUM_NOT_IMPLEMENTED` for CUDA until the
inducing solves and softmax reduction are resident; no host fallback is hidden
behind a CUDA request.  The independent finite-difference and adjoint oracle
is [`test_gp_variational_categorical_likelihood`](../test/test_gp_variational_categorical_likelihood.f90).
The release probe is [`fortml_bench_gp_categorical_likelihood.f90`](../app/fortml_bench_gp_categorical_likelihood.f90),
with the NumPy comparison recorded in
`fortml-bench/results/GP_CATEGORICAL_LIKELIHOOD.md`.
