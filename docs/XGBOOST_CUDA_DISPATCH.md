# Numeric XGBoost CUDA dispatch

`xgboost_t%predict_device` now uses the resident additive-tree CUDA plan when
the native `fortml_cuda_boosted_tree` kernel is linked. The fitted numeric
tree topology and leaf values are flattened into the plan, transferred once
at plan creation, and evaluated on the selected CUDA device. Objective links
(logistic, Poisson/Tweedie/Gamma, and squared-log) are applied after the
resident margin is returned.

Categorical nodes, an unavailable CUDA kernel, invalid device selection, and
split-boundary derivative cases remain explicit typed refusals; FortML never
labels a host fallback as GPU work. Ordinary builds therefore return
`FORTNUM_NOT_IMPLEMENTED` for CUDA while CPU dispatch remains unchanged.

The release gate is `test_xgboost_cuda_dispatch`, which checks CPU dispatch
parity and accepts either a successful resident CUDA result or the typed
unavailable capability. `test/run_cuda_boosted_tree_plan.sh` supplies the
independent flattened-tree value/JVP oracle for native CUDA builds.
