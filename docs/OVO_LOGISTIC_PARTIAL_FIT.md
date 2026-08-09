# OVO logistic partial fitting

`ovo_logistic_classifier_t` exposes a deterministic, transactional
`partial_fit` contract for weighted one-vs-one logistic classification.  The
classifier stores one sorted integer class vocabulary and replays accepted
batches through every pair model.  Consequently, a weighted stream is
numerically equivalent to fitting once on its concatenated batches (up to the
configured optimizer tolerance), while preserving the fixed-state probability
and derivative products.

```fortran
type(ovo_logistic_classifier_t) :: model
type(fortnum_status_t) :: status

call model%partial_fit(x_first, labels_first, status, classes=[-7, 10, 42], &
    sample_weight=w_first, class_weight=[1.5_dp, 0.8_dp, 2.0_dp])
call model%partial_fit(x_second, labels_second, status, sample_weight=w_second)
```

The optional `classes` argument is used on the first batch to declare the
complete sorted vocabulary.  A batch may omit classes at this stage; the
model remains pending and is fitted as soon as every declared class has
appeared.  Later batches must use the initialized vocabulary.  Class weights
are positive finite values in sorted-class order and are retained from the
first batch unless replaced explicitly.  Sample weights are finite,
nonnegative, and must have positive mass for every accepted batch.

`metadata()` returns a copy of `classification_state_t`, exposing the sorted
classes, feature count, accepted sample count, and batch count.  Input shape,
unknown-label, class-vocabulary, and weight failures are validated before the
candidate model is assigned, so a failed call leaves predictions, parameters,
history, and metadata unchanged.  `warm_start` is an explicit alias for
`partial_fit`.

Fitting and replay are host operations.  `predict_proba_device` and
`predict_device` dispatch selected CPU devices and return a typed
`FORTNUM_NOT_IMPLEMENTED` refusal for CUDA until a resident pairwise kernel is
linked; no hidden host fallback is used.  The independent behavioral oracle is
`test/test_ovo_logistic_partial_fit.f90`.
