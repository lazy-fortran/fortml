# Ordinal GP log probabilities

`gp_ordinal_classification_t` now exposes the same stable log-probability
surface as the other FortML classifiers:

```fortran
call model%predict_log_proba(x_query, log_probabilities, status)
call model%predict_log_proba_parameter_jvp(x_query, direction, &
    log_probabilities, log_probabilities_dot, status)
call model%predict_log_proba_parameter_vjp(x_query, log_probabilities_bar, &
    parameter_bar, status)
call model%predict_log_proba_input_jvp(x_query, x_dot, log_probabilities, &
    log_probabilities_dot, status)
call model%predict_log_proba_input_vjp(x_query, log_probabilities_bar, &
    x_bar, status)
```

The output columns retain `classes()` order.  The implementation evaluates
the existing adjacent normal-CDF probability graph, then applies
`log(max(p, tiny(real64)))`.  This keeps tails finite without changing the
probability simplex.  Product routines use the exact chain rule for positive
probabilities and return a zero contribution for a probability clipped at the
finite floor.

The fixed-state parameter products use the packed kernel/noise coordinates
returned by `parameters()`.  Input products differentiate both the posterior
mean and posterior variance, including the normal-CDF scale.  The independent
`test_gp_ordinal_log_proba` oracle checks probability/log-probability
equivalence, central-difference JVPs, VJP duality, and output-clearing typed
CUDA refusal.  CUDA remains unsupported until the ordinal covariance and
normal-CDF graph is resident; no host fallback is used.
