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

The likelihood can be logistic (the default,
`GP_VARIATIONAL_LOGISTIC`) or probit (`GP_VARIATIONAL_PROBIT`). Labels must be
integer zero/one values. The model exposes
`device_supported(FORTML_DEVICE_CPU)` and `elbo_device`; CUDA returns
`FORTNUM_NOT_IMPLEMENTED` until the full inducing solve, likelihood, and
reduction are resident. No implicit host fallback is performed. Multiclass
coupling, kernel/inducing hyperparameter products, natural-gradient updates,
posterior prediction, and resident GPU inference remain planned extensions.

The independent behavioral oracle is `test_gp_variational_classification`.
It checks the prior KL identity, ELBO decomposition, every packed gradient
coordinate against finite differences, a directional JVP, CPU dispatch, and
the typed CUDA and invalid-label refusals.
