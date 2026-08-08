# Variational Bernoulli GP classification

`gp_variational_classification_t` provides a deterministic CPU reference
objective for an inducing-point Bernoulli GP. It is intentionally separate
from the existing Laplace classifier and Gaussian-likelihood `sparse_gp_t`:

```fortran
use fortml_gp_variational_classification, only: &
    gp_variational_classification_t

type(gp_variational_classification_t) :: model
real(dp) :: value, tangent
real(dp), allocatable :: gradient(:), parameters(:)

call model%initialize(inducing_points, kernel, 32, 20260807, status)
parameters = model%parameters()
call model%elbo(x, labels, value, status)
allocate(gradient(model%parameter_count()))
call model%elbo_gradient(x, labels, value, gradient, status)
call model%elbo_jvp(x, labels, direction, value, tangent, status)
```

The variational state is `q(u)=N(m,L Lᵀ)` at the fixed inducing points. The
packed parameter vector is

```text
[ m(1:M), log(L(1,1)), L(2,1), log(L(2,2)), ..., log(L(M,M)) ]
```

`elbo` computes a deterministic Monte Carlo expectation using the seeded
reparameterization of the training marginals and subtracts the analytic
`KL(q(u)||N(0,K_uu))`. `elbo_gradient` differentiates both terms in packed
coordinates. `elbo_jvp` propagates an arbitrary packed direction directly
through the latent mean and variance; its result agrees with a centered
finite difference of `elbo`. An optional `scale` multiplies only the expected
log likelihood, making minibatch scaling explicit without scaling the KL.

Fixed-state predictive kernel products are available through
`predict_latent_kernel_parameter_jvp` and
`predict_proba_kernel_parameter_jvp`, with matching
`predict_latent_kernel_parameter_vjp` and
`predict_proba_kernel_parameter_vjp` reverse products. A direction is in
the kernel's log-hyperparameter coordinates; `kernel_parameter_count()`
reports its length. The products differentiate the inducing solve and all
`K_uu`, `K_ux`, and diagonal `K_xx` terms while holding inducing points and
the variational state fixed. CPU dispatch is exact; the device VJP returns
`FORTNUM_NOT_IMPLEMENTED` for CUDA until the resident projection graph is
available.

The likelihood can be logistic (the default,
`GP_VARIATIONAL_LOGISTIC`) or probit (`GP_VARIATIONAL_PROBIT`). Labels must be
integer zero/one values. The model exposes
`device_supported(FORTML_DEVICE_CPU)` and `elbo_device`; CUDA returns
`FORTNUM_NOT_IMPLEMENTED` until the full inducing solve, likelihood, and
reduction are resident. No implicit host fallback is performed. Natural-
gradient updates and resident GPU inference remain planned extensions.

The independent behavioral oracles are `test_gp_variational_classification`
and `test_gp_variational_kernel_products`. The latter checks centered finite
differences, JVP/VJP dot-product identities, both likelihood paths, and the
typed CUDA refusal.
It checks the prior KL identity, ELBO decomposition, every packed gradient
coordinate against finite differences, a directional JVP, CPU dispatch, and
the typed CUDA and invalid-label refusals.
