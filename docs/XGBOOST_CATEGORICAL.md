# Bounded categorical XGBoost splits

`fortml_xgboost` supports integer-coded categorical columns through one
deterministic policy: ordered-gradient partitions.  Set the one-based feature
indices in `xgboost_options_t%categorical_features`, select
`categorical_policy="ordered"`, and set an explicit
`categorical_max_categories` (2 through 64):

```fortran
type(xgboost_options_t) :: options
options%categorical_policy = "ordered"
options%categorical_max_categories = 8
options%categorical_features = [1, 3]
```

At each node, finite integer codes are grouped by their accumulated gradient
over Hessian score.  Prefix partitions in ascending score order are evaluated
with the same weighted second-order gain, regularisation, missing-value
direction, and minimum-child-Hessian rules as numeric splits.  Score ties are
resolved by the integer code, so repeated fits are deterministic.  The
category count is checked at fit time and a typed `FORTNUM_NOT_IMPLEMENTED`
status is returned when a feature has more categories than the explicit bound;
non-integer values are refused as well.

The learned feature list, policy, and bound are retained by warm starts,
prefix slicing, and version-3 text snapshots.  Each categorical node stores
its selected integer-code prefix, so save/load predictions are bitwise stable.
CPU prediction and the ordinary fixed-tree value path are supported.  Tree
growth has no resident CUDA kernel in this release, so CUDA dispatch returns a
typed `FORTNUM_NOT_IMPLEMENTED` status.  Categorical inputs have no canonical
continuous tangent space; `predict_jvp` and `predict_vjp` therefore refuse a
categorical model explicitly rather than returning a misleading zero product.

The independent behavioral fixture is `test_xgboost_categorical`.  The
release benchmark `xgboost_categorical.csv` compares the fitted predictions
with an independent NumPy ordered-partition tree oracle and records a typed
CUDA-unavailable row.
