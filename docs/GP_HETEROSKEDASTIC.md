# Heteroskedastic Gaussian-process regression

`heteroskedastic_gp_t` is a dense scalar GP for observations whose supplied
measurement variance changes with the input. `fit` accepts a positive variance
for each training row, a signal kernel, and a separate log-noise kernel. The
signal posterior is conditioned with the row-specific diagonal
`K + diag(noise_variance) + jitter I`; `predict` returns latent signal mean and
variance, while `noise_at` interpolates the log variances and exponentiates
them so the reported observation noise stays positive.

The noise levels are treated as known. This keeps the posterior exact and makes
the contract auditable; jointly inferring a latent noise process requires
variational or EM machinery and remains separate roadmap work. Constant supplied
noise is an exact ordinary-GP special case. `test_heteroskedastic_gp` checks
that reduction against `gp_regression_t`, confidence and mean tracking across
quiet/noisy regions, positive log-noise interpolation with geometric-mean
reversion away from data, and typed shape/zero-noise/pre-fit refusals.

This is a CPU dense reference path. Derivatives, sparse/variational inference,
noise hyperparameter optimization, and resident CUDA execution remain open.
