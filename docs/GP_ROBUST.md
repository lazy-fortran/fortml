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
refusal boundaries. The Poisson path now also exposes the normalised count
likelihood and exact value/gradient/JVP/VJP/HVP products in log-rate
coordinates. A fixed-covariance latent posterior objective is available
through `robust_gp_poisson_objective_t`; its L-BFGS-B adapter uses FortOpt and
updates the Laplace state only after a valid candidate factorisation. Query
latent and response input JVP/VJP products are analytic and hold the fitted
state fixed. `predict_response_device`, `log_posterior_device`, and the
optimizer's CUDA option return `FORTNUM_NOT_IMPLEMENTED` rather than staging a
host fallback. These products are a CPU dense reference path: kernel
hyperparameter-through-refit products, sparse/variational count inference,
and resident CUDA remain roadmap work.

The focused `test_robust_gp_poisson_products` fixture checks the direct
Poisson formula against an independent hand oracle, posterior HVP central
differences, query products, transactional latent updates, FortOpt
convergence, and the CUDA refusal. The release lane is
`scripts/bench_robust_gp_poisson_products.py` in `fortml-bench`.
