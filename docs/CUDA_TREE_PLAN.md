# Resident CUDA tree prediction boundary

The deterministic CART and random-forest classifiers currently have a CPU
prediction implementation only. Their `predict_device` and
`predict_proba_device` entry points deliberately return
`FORTNUM_NOT_IMPLEMENTED` for a selected CUDA context. They do not copy a
query to the host implementation, and they leave the output and device
transfer counters unchanged. This is a typed capability boundary, not a
claim of GPU support.

## Planned no-autodiff kernel

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
metadata is recorded. The sibling benchmark records this as an unavailable
CUDA row until the resident plan is linked.
