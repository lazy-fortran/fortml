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

The independent `test_student_t_process` oracle checks the large-`nu` limit
against `gp_regression_t`, exact equality of the predictive mean across
degrees of freedom, variance widening for surprising observations, the
posterior degrees-of-freedom formula, and typed refusals for invalid `nu`,
shapes, and prediction before fitting. This is a CPU dense reference path:
derivatives, sparse/variational inference, derivative observations, and
resident CUDA execution remain explicit roadmap work.

The implementation follows *Student-t Processes as Alternatives to Gaussian
Processes* (Shah, Wilson, and Ghahramani, arXiv:1402.4306); the companion
benchmark keeps a local provenance record under `fortml-bench/literature/`.
