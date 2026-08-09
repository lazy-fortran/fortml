# Resident CUDA boosted-tree plan

`fortml_cuda_boosted_tree_api` is the explicit device boundary for a fixed,
flattened additive tree ensemble.  It is useful for a fitted XGBoost or
LightGBM-style model that has already committed its split topology; fitting and
split search remain discrete CPU operations.  The plan copies the topology,
leaf weights, per-tree scales, base score, and learning rate to the selected
CUDA device at `create` time.  Subsequent calls transfer only a query batch and
the output margin.

The flattened representation uses zero-based indices.  `tree_offset` has
`n_trees+1` entries, `node_feature=-1` marks a leaf, and child indices stay
inside the corresponding tree's half-open range.  A NaN query follows the
node's `node_missing_left` bit; infinities are rejected.  Prediction is

```
margin = base_score + sum_t(learning_rate * tree_scale(t) * leaf_weight(t, x))
```

The plan also exposes `predict_jvp`.  It returns the exact zero input tangent
for a fixed routing topology.  NaN queries and exact split-boundary queries
are rejected because their input derivative is undefined.  Candidate outputs
are copied into caller buffers only after the native kernel and device
synchronization succeed, so failed calls preserve the supplied outputs.

```fortran
use fortml_cuda_boosted_tree_api, only: cuda_boosted_tree_plan_t
type(cuda_boosted_tree_plan_t) :: plan
call plan%create(tree_offset, node_feature, node_left, node_right, &
    node_threshold, node_weight, node_missing_left, tree_scale, base, rate, &
    device_index=0, status=status, n_inputs=n_features)
call plan%predict(query_x, margin, status)
call plan%predict_jvp(query_x, query_x_dot, margin, margin_dot, status)
call plan%transfer_stats(host_to_device_bytes, device_to_host_bytes, &
    resident_bytes, status)
call plan%destroy(status)
```

`transfer_stats` is cumulative for the lifetime of a plan.  The immutable
tree topology, weights, scales, base score, and learning rate contribute to
`host_to_device_bytes` during `create` and to `resident_bytes` once.  A value
prediction then adds exactly `8*n_query*n_inputs` host-to-device bytes for the
Fortran-column-major query and `8*n_query` device-to-host bytes for the margin.
A JVP adds the same query-sized tangent and second output transfer.  No model
array is re-uploaded between repeated predictions, which makes the counters a
direct guard against hidden host fallback or per-query model transfers.

The ordinary build links a typed `FORTNUM_NOT_IMPLEMENTED` stub; it never
silently routes a CUDA request through the host tree walker.  The native
correctness gate is `test/run_cuda_boosted_tree_plan.sh`, which compares a
separate CPU leaf-walk oracle, learned-NaN routing, resident repeated batches,
and value/JVP boundary behavior.  `test_cuda_boosted_tree_api` checks shape,
invalid-device, typed-refusal, output-preservation, and scalar-oracle behavior
without requiring a GPU.
