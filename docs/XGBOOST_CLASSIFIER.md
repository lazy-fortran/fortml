# Binary XGBoost classifier

`fortml_xgboost_classifier` provides a classifier-shaped facade over the
deterministic second-order logistic trees in `fortml_xgboost`.  It is the
binary counterpart to `xgboost_multiclass_t`; use it when the public model
should expose integer labels rather than a regression-style real vector.

```fortran
use fortml_xgboost, only: xgboost_options_t
use fortml_xgboost_classifier, only: xgboost_classifier_t
use fortnum_kinds, only: dp
use fortnum_status, only: fortnum_status_t

type(xgboost_classifier_t) :: model
type(xgboost_options_t) :: options
type(fortnum_status_t) :: status
real(dp) :: x(n_samples, n_features), probabilities(n_query, 2)
integer :: labels(n_samples), prediction(n_query)

call model%fit(x, labels, status, options, sample_weight=weights)
call model%predict_proba(x_query, probabilities, status)
call model%predict(x_query, prediction, status)
```

The two fitted classes are sorted and retained by `classes()`.  Column one of
`predict_proba` is the first class and column two is the second class.  Ties
in the two probabilities select the first class, matching the deterministic
argmax convention used by the multiclass adapter.  `decision_function`
returns the positive-class logit, and `predict_proba_staged` and
`decision_function_staged` expose all cumulative boosting rounds.  The final
staged slice is equal to the ordinary prediction.

`fit` accepts positive finite `sample_weight`, optional validation features
and integer labels, and `validation_weight`.  Validation labels must belong
to the two training classes.  The wrapped `xgboost_options_t` retains exact or
weighted-histogram growth, depth/leaf and regularization controls,
without-replacement sampling, monotone constraints, and `missing_policy`.
The policy defaults to `error`; `learn`, `left`, and `right` preserve NaN
routing in every tree while infinities remain refused.

`feature_importance(kind,normalize)` forwards gain, split-count (`weight`),
and cover diagnostics.  `predict_proba_jvp` and `predict_proba_vjp` provide
the fixed-tree input products: they are zero away from split boundaries and
return the same structured domain refusal as the underlying tree at a finite
threshold.  Labels are discrete and have no derivative product.

Device dispatch is explicit.  CPU methods call the validated host path;
`predict_device` and `predict_proba_device` return
`FORTNUM_NOT_IMPLEMENTED` for selected CUDA until a resident tree/histogram
kernel is linked.  No host fallback is hidden in a CUDA timing row.

The independent behavioral oracle is `test_xgboost_classifier`.  The release
workload `fortml_bench_xgboost_classifier` reports weighted CPU fit/predict
timings, log loss, accuracy, staged-probability consistency, and the typed
CUDA refusal.
