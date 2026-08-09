# Poisson GP likelihood

`fortml_poisson_likelihood` provides the count-observation likelihood used by
Poisson-process and count-regression GP adapters.  A latent value is a log
rate; an optional shared `log_rate_offset` represents exposure or an intercept:

\[
  \eta_i = f_i + o, \qquad
  \log p(y_i\mid f_i,o) = y_i\eta_i - \exp(\eta_i) - \log\Gamma(y_i+1).
\]

Observations are validated as finite, non-negative integer counts.  Optional
sample weights may be zero, but their total mass must be positive.  The public
free procedures return the weighted log likelihood and analytic latent/offset
gradients, JVPs, VJPs, and Hessian-vector products.  `poisson_likelihood_t`
wraps the same products as a one-coordinate FortOpt objective and offers
transactional L-BFGS-B fitting of the log-rate offset.

The CPU path is complete.  CUDA initialization is rejected with
`FORTNUM_NOT_IMPLEMENTED` before state allocation because the resident
exponential and reduction kernels are not yet linked; there is no hidden host
fallback.  The independent gate is `test_poisson_likelihood`, which compares
the value and every product with central finite differences, checks the
JVP/VJP adjoint identity, verifies the optimizer optimum, malformed counts,
non-finite parameters, and the typed CUDA refusal.
