# Ordered latent-Gaussian GP classification

`gp_ordinal_classification_t` is the bounded ordinal GP baseline. It fits one
zero-mean Gaussian process to the integer rank of each sorted class, then maps
the predictive Gaussian to class probabilities with adjacent normal-CDF
differences. The cut points are the fixed mid-ranks `1.5, 2.5, ...`; actual
integer labels are preserved by `classes()` and `predict()`.

```fortran
use fortml_gp_ordinal_classification, only: &
    gp_ordinal_classification_t, gp_ordinal_classification_options_t

type(gp_ordinal_classification_t) :: model
type(gp_ordinal_classification_options_t) :: options
type(fortnum_status_t) :: status

options%noise_variance = 0.04_dp
call model%fit(x, labels, kernel, status, options)
call model%predict_proba(query, probabilities, status)
```

For a query with latent mean `m` and variance `v`, the scale is
`sqrt(1+v)`. For cut point `t_j`, `F_j = Phi((t_j-m)/sqrt(1+v))` and
`P(class=j) = F_j-F_(j-1)`, with zero and one tails. This keeps every row on
the probability simplex and makes the uncertainty contribution explicit.

The fitted state delegates packed kernel/noise parameters to
`gp_regression_t`. `predict_latent_parameter_jvp/vjp` and
`predict_proba_parameter_jvp/vjp` include the complete Cholesky solve and
normal-CDF chain. `predict_latent_input_jvp/vjp` and the probability products
differentiate the kernel cross-covariance and posterior variance analytically;
unsupported kernel input derivatives return their typed status.

The evidence surface uses the same packed order as `parameters()`:
`[kernel parameters..., log(noise variance)]`. `hyperparameter_gradient` is
the analytic exact-GP log-marginal-likelihood gradient and
`hyperparameter_hvp(direction, product)` is its directional Hessian product;
both delegate to the latent exact GP, so kernel-specific generated parameter
products and the differentiated Cholesky solve are reused without finite
differences. `log_marginal_likelihood_jvp` is the corresponding directional
scalar product. The aliases `hyperparameters()`, `hyperparameter_count()`, and
`set_hyperparameters()` make the optimized coordinate block explicit while
retaining transactional shape and finite-value checks.

`gp_ordinal_optimize_hyperparameters` in
`fortml_gp_ordinal_classification_training` minimizes negative latent evidence
with FortOpt L-BFGS-B. Bounds are applied to every packed coordinate;
converged iterations report objective evaluations and the final gradient norm,
while failed or non-converged runs restore the initial packed state. The
adapter is CPU-only until exact factorization and the optimizer are resident
on CUDA. Device gradient/HVP and optimizer calls return
`FORTNUM_NOT_IMPLEMENTED` for a selected CUDA device rather than staging
through the host.

This is a latent-Gaussian ordered surrogate, not a Laplace approximation to a
cumulative-logit/probit likelihood. Optimized cut points, ordinal likelihood
hyperparameters, natural gradients, and resident CUDA solves remain separate
roadmap work. CPU is the reference path; all CUDA prediction, reverse, and
evidence products return `FORTNUM_NOT_IMPLEMENTED` rather than silently
staging through the host. The original prediction oracle is
`test_gp_ordinal_classification`; the evidence gradient/HVP and optimizer
oracle is `test_gp_ordinal_classification_hyperparameters`.
