# Student-t process regression

`student_t_process_t` is a dense, scalar Student-t process (TP) regression
baseline. It uses the covariance parameterization of Shah, Wilson, and
Ghahramani: the kernel matrix is the marginal covariance, rather than a scale
matrix. Consequently the predictive mean is the usual GP mean, while the
predictive variance is multiplied by the data-dependent factor

```text
(nu + beta - 2) / (nu + n - 2),
beta = y' (K + noise + jitter I)^(-1) y.
```

Here `nu > 2` is required for a finite covariance. `fit(x,y,kernel,nu,
noise_variance,status[,jitter])` validates the inputs, factors the noisy
kernel matrix, and records the observed Mahalanobis distance. `predict` returns
the posterior mean and marginal variance; `posterior_dof` and
`covariance_scale` expose `nu+n` and the data-dependent scale. The fitted
state also provides `log_marginal_likelihood`.

The one-coordinate fixed-state likelihood contract uses
`theta = log(nu - 2)`. `likelihood_parameters` and
`set_likelihood_parameters` expose that positive-offset coordinate, while
`log_marginal_likelihood_likelihood_parameter_jvp`, `..._vjp`, and `..._hvp`
differentiate the fitted Student-t marginal density with the covariance,
factorization, and observed Mahalanobis statistic fixed. The products are
analytic in `theta`; the test independently checks them with central finite
differences of the public likelihood and the VJP/JVP adjoint identity.
Likelihood-only optimization, kernel/noise cross-products, and derivatives
through refitting are not implied by this fixed-state contract.

The independent `test_student_t_process` oracle checks the large-`nu` limit
against `gp_regression_t`, exact equality of the predictive mean across
degrees of freedom, variance widening for surprising observations, the
posterior degrees-of-freedom formula, and typed refusals for invalid `nu`,
shapes, and prediction before fitting. This is a CPU dense reference path.
The likelihood products have explicit CPU dispatch; CUDA JVP/VJP/HVP calls
return `FORTNUM_NOT_IMPLEMENTED` until a resident factorization is linked.
Sparse/variational inference, derivative observations, likelihood-only
optimization, and derivatives through fitting remain explicit roadmap work.

The implementation follows *Student-t Processes as Alternatives to Gaussian
Processes* (Shah, Wilson, and Ghahramani, arXiv:1402.4306); the companion
benchmark keeps a local provenance record under `fortml-bench/literature/`.
