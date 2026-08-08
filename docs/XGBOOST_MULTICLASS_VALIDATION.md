# Multiclass XGBoost validation and early stopping

`xgboost_multiclass_t` is a deterministic one-vs-rest classifier over the
exact or bounded-histogram logistic XGBoost tree builder.  A fit may provide a
held-out matrix, sorted-label integer targets, and positive validation weights:

```fortran
call model%fit(x, labels, status, options, sample_weight=weights, &
    validation_x=x_valid, validation_labels=labels_valid, &
    validation_weight=weights_valid)
```

The adapter grows every class child to the requested tree count, evaluates the
normalized multiclass probabilities after each common stage, and applies the
same `early_stopping_rounds`, `early_stopping_min_delta`, and `restore_best`
policy as the binary estimator.  Validation loss is weighted multiclass
log-loss.  `best_iteration()` reports its one-based minimum, while
`best_validation_loss()` reports the weighted value.  `early_stopped()` is
true only when the patience threshold stopped the common stage loop.

`requested_estimator_count()` preserves the configured tree budget.  When
`restore_best` is true, all children are transactionally sliced to the best
common prefix, so `predict_proba_staged` exposes exactly the retained stages
and its final slice equals `predict_proba`.  With `restore_best=false`, an
early-stopped fit retains the completed prefix while still reporting the best
round.

Training and validation labels are arbitrary integers.  Validation labels must
belong to the training class set; positive finite sample weights are checked
before any child fit.  Any invalid option, shape, label, weight, or child
failure leaves an already fitted destination unchanged.  The text snapshot
schema records the requested count and validation diagnostics.  CUDA prediction
remains an explicit `FORTNUM_NOT_IMPLEMENTED` refusal until resident tree
kernels are available; no host fallback is hidden.

`test_xgboost_multiclass` contains an independent weighted log-loss replay,
best-prefix, metadata, and transactional-refusal oracle.  The release workload
`fortml_bench_xgboost_multiclass_validation` records CPU timing, weighted
validation loss, staged-prefix consistency, and the typed CUDA refusal.
