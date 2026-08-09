# OVR logistic partial fitting

ovr_logistic_classifier_t now exposes a deterministic, transactional
partial_fit contract for multiclass one-vs-rest logistic models. The
classifier keeps one sorted integer vocabulary in a shared
classification_state_t metadata object and replays all accepted batches
when a new batch arrives. This gives a stable warm-start boundary without
changing the fixed-state prediction products.

    type(ovr_logistic_classifier_t) :: model
    type(fortnum_status_t) :: status

    call model%partial_fit(x_first, labels_first, status, &
        classes=[-7, 10, 42], l2=0.1_dp)
    call model%partial_fit(x_second, labels_second, status)
    if (model%fitted()) then
        call model%predict_proba(x_query, probabilities, status)
    end if

The warm_start binding is an explicit alias for partial_fit when an
application names the continuation operation rather than the batch API.

The optional classes argument is required only when the first batch does
not contain at least two distinct labels. It must be strictly increasing.
Every later batch must use the initialized vocabulary without changes. A first batch that does
not yet cover every declared class is accepted and retained as pending
history. Once all classes have appeared, the estimator is fitted over the
complete history. If the model was already fitted, a batch containing only a
subset of classes is still accepted because the previous history supplies the
missing classes.

metadata() returns a copy of classification_state_t, whose accessors report
sorted classes(), class_count(), feature_count(), sample_count(), batch_count(),
and initialized(). Sample weights are stored per accepted batch. The
class-weight and optimizer controls from the first batch are retained unless
explicitly replaced on a later call.

Validation occurs before a candidate is assigned to the live model. Unknown
labels, mismatched class metadata, invalid weights, shape errors, and failed
optimizer fits therefore leave predictions, parameters, history, and metadata
unchanged. Accepted replay is deterministic and matches a one-shot fit on the
concatenated data up to optimizer tolerance.

The fixed-state input and packed-parameter JVP/VJP interfaces remain the same
as the one-shot classifier. Fitting is a CPU operation. A CUDA
predict_proba_device request returns a typed FORTNUM_NOT_IMPLEMENTED status
until a resident OVR multi-head kernel is available. The API does not use a
hidden host fallback.

Independent behavioral checks are in
test/test_ovr_logistic_partial_fit.f90, covering delayed class completion,
sorted arbitrary labels, deterministic replay, malformed-batch rollback,
input JVP finite differences, and changed-vocabulary refusal. The release
workload is fortml_bench_ovr_logistic_partial_fit.
