# Random-forest regression

`random_forest_regressor_t` is a deterministic bootstrap ensemble of weighted
numeric CART regression trees.  The estimator keeps rows as samples and target
columns as outputs, so one fit covers scalar and multi-output regression with
the same seeded tree topology.

```fortran
type(random_forest_regressor_t) :: forest
type(fortnum_status_t) :: status
real(dp) :: prediction(n_query, n_outputs)

call forest%fit(x, targets, status, n_trees=128, max_depth=8, &
    min_samples_leaf=2, seed=1729, sample_weight=weights)
call forest%predict(query, prediction, status)
```

Fit is transactional: malformed dimensions, non-finite values, non-positive
weights, or unsupported hyperparameters return `FORTNUM_DOMAIN_ERROR` without
replacing an existing fitted forest.  Bootstrap draws use the Park--Miller
stream seeded by `seed`; `bootstrap_inclusion()` exposes the resulting
sample-by-tree audit matrix.  Metadata accessors report feature/output/sample
counts, tree count, depth, minimum leaf size, seed, and schema version `1`.

`predict_staged` returns every prefix average with shape
`(n_query,n_trees,n_outputs)`.  `feature_importances` is a normalized
split-frequency diagnostic over all output trees.  It intentionally differs
from gain-based XGBoost/LightGBM importance: CART stores no differentiable
split gain after fitting, while split membership remains fully auditable.

Tree routing is piecewise constant.  `predict_jvp` and `predict_vjp` return
exact zero products away from every visited split threshold and return
`FORTNUM_DOMAIN_ERROR` when any query lies exactly on a threshold.  No
derivative through bootstrap membership or split selection is claimed.  CPU
dispatch is selected explicitly; a selected CUDA context returns
`FORTNUM_NOT_IMPLEMENTED` and preserves caller output until a resident
regression-forest kernel is linked.

The independent release benchmark in
[`fortml-bench/results/RANDOM_FOREST_REGRESSION.md`](../fortml-bench/results/RANDOM_FOREST_REGRESSION.md)
replays the bootstrap stream and weighted exhaustive CART policy in NumPy,
checking scalar/multi-output predictions, staged prefixes, feature-importance
normalization, fixed-state JVPs, timings, and the typed device boundary.
