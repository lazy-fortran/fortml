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

This is a latent-Gaussian ordered surrogate, not a Laplace approximation to a
cumulative-logit/probit likelihood. Optimized cut points, ordinal likelihood
hyperparameters, natural gradients, and resident CUDA solves remain separate
roadmap work. CPU is the reference path; all CUDA prediction and reverse
products return `FORTNUM_NOT_IMPLEMENTED` rather than silently staging through
the host. The independent central-difference and adjoint oracle is
`test_gp_ordinal_classification`.
