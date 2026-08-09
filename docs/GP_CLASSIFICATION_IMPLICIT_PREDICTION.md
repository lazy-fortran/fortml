# GP classifier prediction through fit

`gp_classification_t` has two distinct kernel-parameter prediction contracts.
The existing `predict_*_parameter_jvp` and `predict_*_parameter_vjp` methods
hold the fitted Laplace mode and likelihood curvature fixed.  The new
`predict_latent_hyperparameter_jvp` and
`predict_proba_hyperparameter_jvp` methods differentiate through the
converged fit.

```fortran
call model%predict_latent_hyperparameter_jvp(x_query, direction, &
    mean, mean_dot, variance, variance_dot, status)
call model%predict_proba_hyperparameter_jvp(x_query, direction, &
    probabilities, probabilities_dot, status)
```

The direction follows the packed logarithmic kernel layout returned by
`model%parameters()`.  Training inputs, labels, sample weights, likelihood
kind, and the converged Newton branch are fixed.  The product includes the
implicit mode tangent, the prior solve, the likelihood-curvature tangent, the
posterior factorization solve, the predictive mean and variance, and the
logistic or probit probability map.  A row with zero sample weight contributes
zero curvature and zero curvature tangent.  The numerical curvature floor has
a fixed-active-set derivative of zero.

`predict_proba_hyperparameter_jvp_device` dispatches a selected CPU context to
the same implementation.  CUDA returns `FORTNUM_NOT_IMPLEMENTED` before
touching the output arrays because a resident Laplace factorization graph is
not linked.

`test_gp_classification_implicit_prediction` checks logistic and probit models
with non-uniform sample weights.  Its independent oracle perturbs the RBF
kernel coordinates, refits both models at each central probe, and differences
latent means, latent variances, and probabilities.  It also covers unfitted
state, direction shape, CPU dispatch, CUDA refusal, and output preservation.
