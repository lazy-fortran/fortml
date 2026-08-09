# Ordinal-GP cut-point training

`fortml_gp_ordinal_cutpoint_training` calibrates the ordered cut points of a
fitted `gp_ordinal_classification_t`. The latent Gaussian-process fit stays
fixed. The objective evaluates its posterior mean at the supplied calibration
rows, then minimizes the weighted ordered-logit or ordered-probit negative log
likelihood. This is a calibration step. It does not differentiate through the
latent GP fit or perform joint Laplace inference.

The optimizer uses one location coordinate and one log gap for each remaining
cut point. For parameters `q`, minimum gap `g`, and thresholds `b`, the mapping
is

```text
b(1) = q(1)
b(j) = b(j-1) + g + exp(q(j)),  j = 2,...,K-1
```

Every FortOpt L-BFGS-B trial therefore has strictly increasing thresholds.
Separate bounds control the first threshold and the log gaps. The model is
updated once, after convergence. Invalid data, invalid bounds, iteration-limit
failure, and device refusal leave the original thresholds unchanged.

## API

`gp_ordinal_optimize_cutpoints` accepts the fitted model, calibration inputs,
the original integer labels, options, and optional nonnegative sample weights.
Each fitted class must have positive effective weight. The result records the
initial and final normalized negative log likelihood, the threshold-gradient
norm, the FortOpt iteration count, and line-search evaluations.

`gp_ordinal_cutpoint_value_gradient` and `gp_ordinal_cutpoint_hvp` expose the
fixed-latent objective in the physical threshold coordinates. The HVP is
analytic for ordered-logit and ordered-probit likelihoods. The classifier also
provides threshold-coordinate JVP/VJP products for `predict_proba` and
`predict_log_proba`, plus a transactional `set_thresholds` method.

CPU calls execute the dense latent prediction and likelihood reduction. CUDA
objective, product, and optimizer calls return `FORTNUM_NOT_IMPLEMENTED` until
the latent prediction, ordered likelihood, and FortOpt state share a resident
device graph. No CUDA request falls back to the host.

## Example

```fortran
use fortml_gp_ordinal_classification, only: gp_ordinal_classification_t
use fortml_gp_ordinal_cutpoint_training, only: gp_ordinal_cutpoint_options_t, &
    gp_ordinal_cutpoint_result_t, gp_ordinal_optimize_cutpoints

type(gp_ordinal_classification_t) :: model
type(gp_ordinal_cutpoint_options_t) :: options
type(gp_ordinal_cutpoint_result_t) :: result

! Fit model before calibration.
options%location_lower = -5.0_dp
options%location_upper = 5.0_dp
options%log_gap_lower = -8.0_dp
options%log_gap_upper = 4.0_dp
call gp_ordinal_optimize_cutpoints(model, x_calibration, labels_calibration, &
    options, result, status, sample_weight=weights)
```

`test_gp_ordinal_cutpoint_training` checks the objective against a separately
coded normal-CDF oracle. It central-differences the gradient, HVP, probability
products, and log-probability products. It also checks adjoint identities,
strict transformed gaps, bounded convergence, rollback, and CUDA refusals.
The release application is `fortml_bench_gp_ordinal_cutpoints`.
