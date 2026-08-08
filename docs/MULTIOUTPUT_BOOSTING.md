# Multi-output XGBoost and LightGBM regression

`fortml_xgboost_multioutput` provides `xgboost_multioutput_t` and
`lightgbm_multioutput_t`.  Both adapters keep the established row-oriented
contract: `x(n_samples,n_features)`, target and prediction matrices with
shape `(...,n_outputs)`, and one deterministic CPU child booster per output.
The XGBoost child uses the exact or bounded histogram policy selected in
`xgboost_options_t`; the LightGBM child uses its weighted-quantile,
best-first leaf-wise policy from `lightgbm_options_t`.

## Fit and prediction

`fit(x,targets,status,options[,sample_weight,validation_x,validation_targets,
validation_weight])` validates every matrix and weight before constructing a
temporary child ensemble.  A child failure therefore leaves a previously
fitted adapter unchanged.  Validation columns are routed to the corresponding
child, so weighted early stopping and restore-best state retain each backend's
normal semantics.  `predict` and `predict_margin` commit their matrix only
after all child predictions succeed.  `predict_staged_margin` returns
`(n_samples,n_estimators,n_outputs)` and requires every child to retain the same
number of stages.

`feature_count`, `output_count`, `estimator_count`, `parameter_count`, and
`fitted` expose shape and fitted-state metadata.  `leaf_parameters` concatenates
each child's `[base_score, leaf weights in tree/node order]` vector, in output
order.  The packed parameter count is the sum of the child counts; split
thresholds, categorical partitions, and other topology decisions remain
discrete.

## Derivative products and devices

`predict_jvp(x,x_dot,values,values_dot,status)` and
`predict_vjp(x,output_bar,x_bar,status)` route input products to every child and
sum output reverse products.  They retain the child contracts: XGBoost and
LightGBM products are zero away from split boundaries and return a typed
domain refusal on a learned threshold.  `predict_leaf_jvp` and
`predict_leaf_vjp` expose fixed-structure products over the concatenated leaf
coordinates, including queries exactly on split surfaces.  All output and
reverse buffers are transactional on malformed arguments.

`predict_device` and `predict_device_margin` accept the explicit device control
plane.  CPU dispatch reuses validated child prediction.  A selected CUDA
device returns `FORTNUM_NOT_IMPLEMENTED`; no host fallback is timed or hidden.
`device_supported(FORTML_DEVICE_CPU/CUDA)` reports this boundary.  Resident
CUDA tree growth, staged execution, and multi-output reductions remain open
work rather than being implied by the adapter.

The independent behavioral oracle is `test_xgboost_multioutput`: it fits a
one-tree, two-output fixture, checks the closed-form Newton stump values,
staged margins, input and leaf-coordinate JVP/VJP adjoint behavior, metadata,
transactional malformed fit, and typed CUDA refusals for both backends.
