# Stable log probabilities for linear classifiers

`logistic_regression_t` and `softmax_regression_t` expose `predict_log_proba`
alongside their probability methods. The implementation evaluates the
log-sigmoid and log-sum-exp forms directly. Saturated logits therefore retain
finite log values when the corresponding probability would underflow in a
separate `log(p)` operation.

The two classifiers also expose exact fixed-state products:

- `predict_log_proba_jvp` differentiates with respect to both packed model
  parameters and input features.
- `predict_log_proba_vjp` returns the packed-parameter and input cotangents.

The binary model uses the sorted two-label convention already used by
`predict_proba`. The multinomial model preserves its sorted class array and
uses the row-wise log-sum-exp normalization. Product calls validate shapes,
finite tangents, and fitted state before writing outputs.

`test_logistic_regression` and `test_softmax_regression` check normalization,
central-difference JVPs, and the binary log-probability adjoint identity.
Fitting remains CPU-owned. Device adapters must report a typed refusal until a
resident classifier and reduction kernel exists.
