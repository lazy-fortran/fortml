# Leakage-safe multiclass calibration

`fortml_calibrated_softmax_classifier` combines a weighted `softmax_regression_t`
with the shared multiclass temperature calibrator.  Fitting is leakage-safe:
each deterministic stratified fold trains a fresh softmax model, writes its
held-out logits into an out-of-fold (OOF) matrix, and only then fits the
positive temperature.  A final softmax model is fitted on all samples for
deployment.  The fitted state reports both raw and calibrated OOF log loss.

```fortran
use fortml_calibrated_softmax_classifier, only: &
    calibrated_softmax_classifier_t, calibrated_softmax_classifier_options_t

type(calibrated_softmax_classifier_t) :: model
type(calibrated_softmax_classifier_options_t) :: options
type(fortnum_status_t) :: status

options%cv_folds = 5
options%cv_shuffle = .true.
options%cv_seed = 31
call model%fit(x, labels, status, options=options)
call model%predict_proba(x_query, probabilities, status)
call model%predict(x_query, labels_query, status)
```

The options default to positive temperature scaling.  Labels are sorted and retained as arbitrary integer values.  Every class must
have at least `cv_folds` positive-weight rows; this prevents a training fold
from losing a class and makes OOF calibration well-defined.  Sample weights
are nonnegative and class weights follow the sorted class order.  The
multiclass calibration policy is currently positive temperature scaling;
sigmoid and isotonic policies return `FORTNUM_NOT_IMPLEMENTED` explicitly.

The packed parameter vector contains the fitted softmax coefficients and
intercepts followed by the positive temperature.  `predict_proba_jvp` and
`predict_proba_vjp` differentiate both model and query inputs; the parameter
products include the calibration temperature.  `set_parameters` preserves
the same layout and validates finite values and positivity.  `decision_function`
returns uncalibrated logits, while `predict_proba` applies the fitted
temperature.

The fit and derivative paths are host/CPU paths.  `predict_proba_device` and
`decision_function_device` accept a selected CPU context and return
`FORTNUM_NOT_IMPLEMENTED` for CUDA until a resident softmax-plus-calibration
kernel is linked; no implicit host fallback is used.

The independent test `test_calibrated_softmax_classifier` checks sorted-label
metadata, OOF convergence and determinism, the probability simplex, central
finite-difference JVPs, the JVP/VJP adjoint identity, weighted-fold support
refusals, unsupported calibration policies, and the typed CUDA boundary.
