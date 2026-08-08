# Bounded tree SHAP-like attributions

`xgboost_t%predict_shap` and `lightgbm_t%predict_shap` expose a deterministic
raw-margin explanation with output shape `(n_samples, n_features + 1)`.
Column one is the path-dependent expected margin. The remaining columns are
per-feature Shapley attributions; summing each row reproduces
`predict_margin` to floating-point roundoff. Logistic and count objectives
remain on their raw link, so a probability link is applied only after the
columns are summed.

The implementation evaluates every feature subset for each fitted tree. At a
split whose feature is absent from a subset, XGBoost uses the stored child
cover proportions and LightGBM uses the fitted child row-count proportions.
When the feature is present, normal fitted routing—including learned NaN and
ordered categorical routing for XGBoost—is used. This gives an independent,
background-data-free path contract and preserves exact additivity for DART
tree scales, fitted prefixes, and warm-start models.

To keep latency and memory explicit, the exact subset path is bounded to
`XGB_MAX_SHAP_FEATURES = 12` and `LIGHTGBM_MAX_SHAP_FEATURES = 12`.
Wider models return `FORTNUM_NOT_IMPLEMENTED`; they do not silently switch to
an approximation. CPU execution is implemented. The device wrappers return a
typed CUDA refusal until resident tree explanation kernels are linked; they
never copy data to a hidden host fallback.

`test_tree_shap` checks the one-stump baseline and leaf-value oracle,
additivity, unused-feature zeros, and both CUDA refusal codes. The release
benchmark [`TREE_SHAP.csv`](../fortml-bench/results/TREE_SHAP.csv) and
[`TREE_SHAP.md`](../fortml-bench/results/TREE_SHAP.md) replay the same fixture
with an independent NumPy oracle.
