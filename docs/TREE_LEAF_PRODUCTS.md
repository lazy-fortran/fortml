# Fixed-structure tree leaf products

`xgboost_t` and `lightgbm_t` expose a continuous derivative surface for a
fitted tree ensemble without pretending that split selection is smooth. The
packed coordinate order is:

```
[base_score, leaf weights in estimator order and node-array order]
```

`leaf_parameter_count()` returns the packed length and
`leaf_parameters(status)` returns the current values. Fixed routing,
thresholds, categorical partitions, DART/GOSS choices, and per-tree scales
are held constant as fitted structure, not differentiated coordinates.

`predict_leaf_jvp(x, parameter_dot, margin, margin_dot, status)` computes the
raw-margin value and forward product. `predict_leaf_vjp(x, output_bar,
parameter_bar, status)` computes the reverse product. Tree contributions are
multiplied by the fitted learning rate and tree scale, while the base
coordinate contributes once per row. For binary objectives these are raw-link
margin products; differentiate the sigmoid separately for probability
products.

These products remain defined on split surfaces because no input tangent is
taken. Input `predict_jvp`/`predict_vjp` retain their split-boundary refusal.
Shape, nonfinite, uninitialized-model, and invalid categorical-query checks
return `FORTNUM_DOMAIN_ERROR` before output mutation. The ordinary build is
CPU-only; tree device prediction remains a typed refusal until model and
derivative state can stay resident.

The explicit `predict_leaf_jvp_device` and `predict_leaf_vjp_device` wrappers
dispatch the same products on CPU and return `FORTNUM_NOT_IMPLEMENTED` for a
selected CUDA context. This preserves the no-hidden-host-fallback contract
and lets benchmark rows assert the exact refusal status.

The independent hand oracle is `test_tree_leaf_products`: it fits a single
stump, checks the two routed leaf coordinates, verifies the JVP/VJP adjoint
identity, and checks malformed tangent transactional behavior without reading
private node arrays. The release workload and raw records are in
`fortml-bench/results/TREE_LEAF_PRODUCTS.md` and
`fortml-bench/results/tree_leaf_products.csv`.
