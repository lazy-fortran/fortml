# XGBoost interaction constraints

`xgboost_options_t%interaction_groups` provides a compact, deterministic
interaction-constraint policy for numeric exact and histogram trees. Allocate
one integer per feature. Features with the same positive group label may be
used on the same root-to-leaf path; a zero label leaves that feature
unconstrained. Once a positive-group feature is selected, descendant splits on
that path are restricted to the same group. The policy is checked before fit,
copied by slicing and warm starts, and preserved by the versioned text
snapshot.

```fortran
type(xgboost_options_t) :: options
options%max_depth = 3
options%interaction_groups = [1, 1, 2, 0]
call model%fit_regression(x, y, status, options)
```

The fitted label is available through `model%interaction_group(feature)`. A
malformed vector, a negative label, or a changed policy during warm start is a
typed domain error and leaves the fitted state unchanged. Tree fitting remains
piecewise: `predict_jvp` and `predict_vjp` retain their existing split-boundary
refusal. There is no implicit CUDA fallback; device prediction reports the
existing typed CUDA refusal because resident tree kernels are not linked.

The independent fixture `test_xgboost_interaction_constraints` verifies a
depth-two four-leaf oracle, the constrained group means, invalid metadata, and
save/load round trips.
