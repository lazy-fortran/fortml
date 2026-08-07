# Calibrated neural classifier

`fortml_mlp_calibrated_classifier` composes a fitted
`mlp_classifier_t` with a deterministic probability-calibration head.  The
network is fit first; calibration then consumes the fitted training logits
with the same sample and sorted-class weights.  The packed estimator
parameters are

```
[ network parameters, calibration parameters ]
```

Inputs are row-major samples in `x(n_samples,n_features)` and labels are
arbitrary integers.  Classes are sorted ascending and prediction ties choose
the first sorted class.

## Calibration methods

Set `options%calibration%method` to one of the public constants
`MLP_CALIBRATION_TEMPERATURE`, `MLP_CALIBRATION_SIGMOID`, or
`MLP_CALIBRATION_ISOTONIC`.

- Binary temperature scaling uses the existing positive-temperature
  `probability_calibrator_t` on the oriented margin `logit(2)-logit(1)`.
- Binary sigmoid calibration uses the existing Platt fit on that same margin.
- Binary isotonic calibration uses weighted PAVA.  Its fitted prediction is
  available, but all estimator JVP/VJP products return
  `FORTNUM_NOT_IMPLEMENTED`: the PAVA active set is a discrete fit-time
  object and cannot be treated as a smooth parameter.
- Multiclass temperature scaling fits one positive scalar `T` and returns
  `softmax(logits/T)`.  Multiclass sigmoid/isotonic calibration is refused
  with `FORTNUM_NOT_IMPLEMENTED` rather than silently fitting independent
  binary maps and changing the probability policy.

`options%classifier` is passed unchanged to the base MLP fit and
`options%calibration` controls Newton tolerances, damping, regularization, and
iteration limits.  Calibration is deterministic for a fixed classifier seed
and data.  The state object retains both base-classifier and calibration
diagnostics.

## Products and devices

`decision_function`, `predict_proba`, and `predict` expose stable logits,
probabilities, and sorted labels.  `decision_function_jvp/vjp` include the
network parameter slice and return zeros for calibration-only logits.  The
probability `predict_proba_jvp/vjp` methods propagate exact joint derivatives
through the MLP, softmax/sigmoid/temperature head, and smooth calibration
parameters.  Their parameter slice includes the calibrated temperature or
Platt slope/intercept.  Isotonic products refuse as described above.

`predict_proba_device`, `decision_function_device`, and `predict_device` run
on an explicitly selected CPU context.  CUDA requests return
`FORTNUM_NOT_IMPLEMENTED` until resident MLP and calibration kernels are
linked; there is no hidden host fallback.  `device_supported(CUDA)` is false
in the current release.

The independent behavioral oracle is `test_mlp_calibrated_classifier`.  It
checks sorted labels, probability normalization, deterministic fits, binary
temperature joint finite-difference and adjoint products, multiclass
temperature probabilities, isotonic active-set refusal, and typed CUDA
refusal.
