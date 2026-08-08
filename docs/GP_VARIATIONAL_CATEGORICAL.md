# Coupled categorical variational GP classification

`gp_variational_categorical_classification_t` provides a multiclass variational
GP with one inducing posterior per class and one shared categorical likelihood.
Classes are sorted during initialization. The order returned by `classes()` is
the column order of every prediction and the order of packed variational
parameters.

## Initialization and fitting

```fortran
use fortml_gp_variational_categorical_classification, only: &
    gp_variational_categorical_classification_t, &
    gp_variational_categorical_options_t

type(gp_variational_categorical_classification_t) :: model
type(gp_variational_categorical_options_t) :: options
type(fortnum_status_t) :: status

call model%fit(x, labels, inducing_points, kernel, status, options)
```

`fit` initializes a mean-field inducing posterior for each sorted class and
maximizes the deterministic categorical ELBO with FortOpt L-BFGS-B. The
variational vector for each class is the binary model's inducing mean followed
by log-Cholesky diagonal and strict-lower entries. `options` controls the
logistic or probit correction, deterministic seed, jitter, optimizer bounds,
and stopping tolerances. `initialize` exposes the same prior state without
running the optimizer. Callers can then update `parameters()` through their own
optimizer or trainer.

The objective is

```
sum_i w_i log softmax(mu_i / sqrt(1 + c variance_i))[label_i] - sum_k KL(q_k || p_k),
```

where `c = pi/8` for logistic and `c = 1` for probit. `scale` multiplies only
the weighted likelihood, which leaves each inducing KL unchanged for minibatch
corrections. Sample weights are finite, nonnegative, and must have positive
mass. Unknown or duplicate labels are rejected.

## Predictions and products

`predict_latent` returns one mean and variance column per sorted class.
`predict_proba` applies a stable coupled softmax to the corrected logits and
returns rows that sum to one. `predict` uses the first class in sorted order
for exact ties.

The CPU reference exposes analytic products for the complete coupled path:

* `elbo_gradient` and `elbo_jvp` include categorical softmax and inducing KL
  terms.
* `predict_proba_parameter_jvp` and `predict_proba_parameter_vjp` differentiate
  all packed inducing states.
* `predict_proba_input_jvp` and `predict_proba_input_vjp` differentiate query
  coordinates through the kernel projection, predictive variance, correction,
  and softmax.

An HVP entry point is not declared for this slice. A caller that needs a
second product can differentiate the analytic JVP with FortAD, or use a
finite-difference oracle around `elbo_gradient`. Kernel hyperparameter and
inducing-point products are also outside this bounded state contract.

The device methods dispatch CPU exactly. CUDA returns `FORTNUM_NOT_IMPLEMENTED`
until resident inducing solves, categorical reductions, and reverse products
are linked. No host fallback is hidden behind a CUDA request.

`test_gp_variational_categorical_classification` checks sorted labels, the
hand-computed softmax, ELBO and prediction JVP/VJP finite differences, JVP/VJP
duality, input products, fitting, and typed CUDA refusals.
