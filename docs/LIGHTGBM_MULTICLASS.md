# LightGBM-style multiclass classification

`lightgbm_multiclass_t` is a deterministic one-vs-rest classifier built from
the existing `lightgbm_t` binary logistic estimator. It keeps the LightGBM
leaf-wise growth policy and exposes a classifier-shaped API with sorted integer
labels.

```fortran
use fortml_lightgbm, only: lightgbm_options_t
use fortml_lightgbm_multiclass, only: lightgbm_multiclass_t

type(lightgbm_multiclass_t) :: model
type(lightgbm_options_t) :: options
type(fortnum_status_t) :: status
real(dp) :: probabilities(n_query, n_class)
integer :: labels(n_sample), predicted(n_query)

options%n_estimators = 32
options%num_leaves = 8
options%min_data_in_leaf = 4
call model%fit(x, labels, status, options)
call model%predict_proba(x_query, probabilities, status)
call model%predict(x_query, predicted, status)
```

The class metadata is the sorted unique set of training labels. Each child
fits a binary target for one class. Positive child probabilities are normalized
row by row, so `sum(probabilities(i,:))` is one and `predict` selects the first
class in sorted order on a tie. `predict_proba_staged` returns
`(sample,class,tree)` values. `decision_function` and
`decision_function_staged` return the unnormalized child raw margins.

Validation data uses the same class set. Child early stopping is disabled so
all children have one common prefix. The multiclass weighted log loss then
selects the best prefix using `early_stopping_rounds` and
`early_stopping_min_delta`. With `restore_best=.true.` the fitted children are
sliced to that prefix. A malformed validation label or child fit leaves the
destination model unchanged.

The fixed-tree input products are explicit. `predict_proba_jvp` and
`predict_proba_vjp` differentiate the smooth sigmoid and probability
normalization while holding every split fixed. They return the child
LightGBM boundary status when a query lies on a split. Away from boundaries the
tree contribution is locally constant, so the input products are zero. These
products are CPU products. `predict_proba_device` and `predict_device` dispatch
CPU explicitly and return `FORTNUM_NOT_IMPLEMENTED` for selected CUDA devices;
there is no hidden host fallback.

The independent behavioral oracle is `test_lightgbm_multiclass`. The release
workload is `fortml_bench_lightgbm_multiclass`, with a NumPy OVR/stage
normalization replay in `fortml-bench/scripts/bench_lightgbm_multiclass.py`.
