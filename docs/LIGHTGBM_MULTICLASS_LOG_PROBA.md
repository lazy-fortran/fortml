# LightGBM multiclass log-probability products

`lightgbm_multiclass_t` is a deterministic sorted-label one-vs-rest adapter
around the numeric LightGBM-style binary estimator. This slice adds the same
stable log-probability and fixed-state derivative contract available for the
other boosted-tree multiclass adapters.

For child raw margins (m_c(x)), the positive-class scores are

```text
ell_c = log(sigmoid(m_c))
log p_c = ell_c - logsumexp(ell_1, ..., ell_C)
```

`predict_log_proba` evaluates both operations in stable form. It does not
take `log(predict_proba(...))`, so large negative margins do not first
underflow a probability. The returned columns follow `classes()` and each row
is a normalized OVR simplex after exponentiation (subject to floating-point
underflow at the final conversion).

The fixed topology has two differentiable coordinate families:

- input products (`predict_log_proba_jvp`/`vjp`) hold routing fixed and return
  a typed split-boundary error when a query lies on a learned threshold;
- packed leaf products (`predict_log_proba_parameter_jvp`/`vjp`) use the
  concatenated `[base_score, leaf weights]` vector returned by `parameters()`.

For a parameter direction (\dot m_c), the log-product is

```text
d log p_c = (1 - sigmoid(m_c)) d m_c
            - sum_j p_j (1 - sigmoid(m_j)) d m_j.
```

The reverse products apply the adjoint of this expression, then dispatch the
result to each binary child's leaf-coordinate VJP. The probability variants
use the equivalent sigmoid and quotient rule. All products validate finite
values and exact output shapes.

The selected CPU device executes the host implementation. Selected CUDA
requests return `FORTNUM_NOT_IMPLEMENTED` transactionally until resident
LightGBM tree and reduction kernels are linked; no hidden host fallback is
used. `test_lightgbm_multiclass_log_proba` checks the independent margin
normalization oracle, simplex/log round trip, input finite-difference and
adjoint products, packed-parameter adjoints, CPU dispatch, and CUDA refusal.

