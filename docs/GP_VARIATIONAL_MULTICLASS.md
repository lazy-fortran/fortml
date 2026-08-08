# One-vs-rest variational GP classification

`gp_variational_multiclass_classification_t` composes one
`gp_variational_classification_t` per integer class.  `initialize` sorts and
validates the caller's class labels, copies the inducing points into each
binary model, and uses deterministic seed offsets (`seed + class_index - 1`)
for independent Monte Carlo tables.  Class order is stable and available from
`classes()`.

The wrapper deliberately exposes a caller-owned objective.  `elbo` sums the
Bernoulli logistic or probit ELBO for the one-vs-rest targets;
`elbo_gradient` and `elbo_jvp` return matching packed products.  The packed
vector is the concatenation of each binary model's inducing mean and
log-Cholesky covariance coordinates, in sorted class order.  It can be passed
directly to FortOpt L-BFGS-B or the generic FortML trainer.  `scale` scales
only each likelihood term, which allows minibatch callers to apply their own
sample-count correction.

`predict_latent` returns one latent mean and variance column per class.
`predict_proba` takes the positive Bernoulli column from each class and
normalizes rows onto the probability simplex; `predict` uses stable
class-order ties.  `predict_proba_parameter_jvp` differentiates this complete
normalization with respect to the packed per-class variational state.  Shared
kernel hyperparameters have matching fixed-state products:
`predict_latent_kernel_parameter_jvp`,
`predict_proba_kernel_parameter_jvp`,
`predict_latent_kernel_parameter_vjp`, and
`predict_proba_kernel_parameter_vjp`.  The direction is applied to each
copied binary kernel, while reverse products accumulate class contributions;
the inducing solve and `K_uu`, `K_ux`, and diagonal `K_xx` terms are included.
`kernel_parameter_count()` reports the shared direction length.

The CPU path is the deterministic reference.  `elbo_device` and
`predict_proba_device` dispatch a selected CPU context exactly and return
`FORTNUM_NOT_IMPLEMENTED` for CUDA until the per-class inducing solves,
likelihood tables, and reductions are resident.  There is no hidden host
fallback.  The independent finite-difference, simplex, packed-JVP, label-order,
and device-refusal oracle is
[`test_gp_variational_multiclass_classification`](../test/test_gp_variational_multiclass_classification.f90).
The kernel-product finite-difference and adjoint oracle is
[`test_gp_variational_kernel_products`](../test/test_gp_variational_kernel_products.f90);
its CUDA device wrapper is an explicit typed refusal until resident OVR
projection kernels are linked.
