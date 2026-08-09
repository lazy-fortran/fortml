# Multiclass XGBoost log probabilities and fixed-tree products

`xgboost_multiclass_t` now exposes a stable `predict_log_proba` contract for
arbitrary sorted integer labels. Each binary child margin is transformed with
a branch-stable log-sigmoid, and the positive links are normalized with
log-sum-exp. This avoids taking `log` after a tail probability has underflowed.
`exp(predict_log_proba(x))` agrees with `predict_proba(x)` to floating-point
roundoff on ordinary fixtures.

The fixed fitted structure has a packed parameter layout in sorted class order:

```fortran
parameters = model%parameters(status)
call model%predict_log_proba_parameter_jvp(x, parameter_dot, log_p, log_p_dot, status)
call model%predict_log_proba_parameter_vjp(x, log_p_bar, parameter_bar, status)
```

The aliases `leaf_parameter_count` and `leaf_parameters` make the relationship
to the binary XGBoost leaf-coordinate products explicit. Probability and
log-probability JVP/VJP methods apply the complete OVR normalization chain rule.
input products retain the fitted-tree split-boundary refusal. Parameter
products are smooth for fixed routing, including queries on a split surface,
because only base and leaf values vary.

CPU device dispatch is equivalent to the ordinary methods. CUDA methods return
`FORTNUM_NOT_IMPLEMENTED` until a resident multiclass tree/reduction kernel is
linked, and the device wrappers do not modify output arrays on refusal. The
independent `test_xgboost_multiclass_log_proba` oracle checks sorted labels,
simplex/log round trips, central-difference input products, parameter
JVP/VJP adjoint identities, malformed-query refusal, CPU dispatch, and the
typed CUDA boundary.

The release benchmark is
[`XGBOOST_MULTICLASS_LOG_PROBA.md`](../fortml-bench/results/XGBOOST_MULTICLASS_LOG_PROBA.md)
in the companion benchmark repository.
