# Robust and count Gaussian-process regression

`robust_gp_t` fits a dense scalar GP with a Laplace approximation for either a
Poisson count likelihood or a Student-t observation likelihood. Poisson uses a
log-rate latent mode, so response predictions are positive and count variance
follows the rate. Student-t uses a bounded-curvature Newton step: extreme
residuals contribute no negative confidence and therefore cannot pull the mode
without bound.

`fit(x,y,kernel,likelihood,status[,nu,scale,jitter,max_iterations,tolerance])`
returns convergence metadata and keeps the latent mode, curvature, and
stationary `alpha` in the fitted state. `predict_latent` returns the Laplace
latent mean/variance; `predict_response` returns the Poisson rate or the
Student-t location/variance approximation. Invalid counts, degrees of freedom,
unknown likelihoods, and pre-fit predictions return typed status codes.

The independent `test_robust_gp` oracle checks Poisson stationarity and rate
tracking, Student-t resistance to a synthetic outlier, convergence, and all
refusal boundaries. This is a CPU dense reference path: derivatives, exact
non-Gaussian evidence, sparse/variational inference, and resident CUDA remain
roadmap work.
