# Resident CUDA tree prediction boundary

The deterministic CART and random-forest classifiers currently have a CPU
prediction implementation only. Their `predict_device` and
`predict_proba_device` entry points deliberately return
`FORTNUM_NOT_IMPLEMENTED` for a selected CUDA context. They do not copy a
query to the host implementation, and they leave the output and device
transfer counters unchanged. This is a typed capability boundary, not a
claim of GPU support.

## Resident no-autodiff kernel

A first native CUDA lowering is now available as a small C ABI in
`src/classification/fortml_cuda_forest.{h,cu}`. It is deliberately
prediction-only: fitting, split selection, class metadata, and every
derivative product remain on the CPU. The plan uploads the flattened model
once and never invokes the CPU tree implementation on a CUDA request.

The ABI uses zero-based C indices and explicit storage conventions:

| argument | shape/layout | meaning |
| --- | --- | --- |
| `tree_offset` | `n_trees + 1`, half-open | global node range for each tree |
| `node_feature` | `n_nodes` | zero-based feature; `-1` marks a leaf |
| `node_left`, `node_right` | `n_nodes` | children within the owning tree |
| `node_threshold` | `n_nodes` | strict-left split threshold |
| `node_probability` | `n_nodes*n_classes`, node-major | empirical leaf probabilities |
| `class_label` | `n_classes` | output labels in sorted class order |

Queries and probability outputs retain the Fortran column-major convention
(`feature*n_query+query` and `class*n_query+query`). One CUDA thread routes
one query through every tree, accumulates the leaf probabilities, and applies
the stable first-maximum class rule for label prediction. Model arrays remain
resident across repeated calls; only a query batch and its requested result
cross the host/device boundary. Non-finite inputs and malformed tree ranges
are rejected before allocation.

`test/run_cuda_forest_plan.sh` compiles the native kernel when `nvcc` and a
CUDA device are present. `test/test_cuda_forest_plan.cu` computes a separate
CPU tree-walk oracle, checks strict-threshold boundaries, labels and
probabilities, and repeats prediction on a second batch. Machines without a
CUDA compiler/device report `skipped` rather than turning a CPU execution into
GPU evidence.

The higher-level Fortran `random_forest_cuda_plan_t` still returns its typed
`FORTNUM_NOT_IMPLEMENTED` boundary: exposing private CART node storage and
binding this generic C ABI to that plan is a separate integration step. This
keeps the current Fortran API honest and prevents an implicit host fallback.

## Planned Fortran integration

The first tree GPU product should be a resident, prediction-only plan. Model
fitting, split selection, class metadata construction, and all derivative
products remain on the CPU. The plan ABI should upload the following
column-major arrays once:

| array | shape | meaning |
| --- | --- | --- |
| `tree_offset` | `n_trees + 1` | half-open node ranges in the flat tree arrays |
| `node_feature` | `n_nodes` | one-based feature index for internal nodes; zero for leaves |
| `node_left`, `node_right` | `n_nodes` | one-based child indices within each tree |
| `node_threshold` | `n_nodes` | strict-left split threshold |
| `node_probability` | `n_nodes * n_classes` | empirical leaf probabilities |
| `class_label` | `n_classes` | sorted external integer labels |

Each CUDA thread handles one query row. It walks every tree from its offset,
routes with the exact CPU rule `x(feature) < threshold`, accumulates the leaf
probability vector, and divides by `n_trees`. A second label kernel can take
the stable first-maximum class index. The plan owns all model arrays and a
scratch probability buffer on the selected device; repeated prediction calls
copy only query batches in and results out. No autodiff, training, or host
fallback is part of this first kernel.

Before implementation, the flattening ABI must be added to
`cart_classifier_t` without exposing mutable internal storage, and an
independent CPU oracle must compare probability and label outputs for exact
thresholds, leaves, class alignment, and repeated resident calls. The CUDA
gate must compile and run only when `nvcc` and a device are available; on
other machines the gate is reported as skipped rather than treating a CPU
run as GPU evidence.

The current contract is covered by
`test/test_random_forest_classifier.f90`: selected CUDA contexts receive a
typed refusal, output sentinels are preserved, and no transfer or residency
metadata is recorded. `random_forest_cuda_plan_t` exposes ABI version `1`,
records the fitted forest's feature/class/tree counts and selected device
index, and keeps `create`, `predict[_proba]`, and `destroy` lifecycle methods
typed even while the native plan is unavailable. Plan creation records shape
metadata but returns `FORTNUM_NOT_IMPLEMENTED`; it never allocates or copies
host trees. The sibling benchmark records both prediction and plan creation
as unavailable CUDA rows until the resident plan is linked.
