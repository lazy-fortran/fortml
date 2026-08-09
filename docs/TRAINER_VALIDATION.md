# Generic trainer validation state

`fortml_trainer` can evaluate a validation metric after every full-batch
update. The metric is supplied by a process-local
`trainer_validation_callback_proc`. It receives the step number and the
current packed parameter vector and returns one finite scalar. The trainer
records the metric history, best value, best step, number of consecutive
non-improving steps, and the parameter vector at the best step.

`trainer_options_t%validation_min_delta` controls the smallest improvement
that counts. By default the callback is a loss and lower values are better.
set `trainer_options_t%validation_higher_is_better` for scores such as
accuracy or R2, where larger values are better. A positive
`validation_patience` stops training after that many consecutive
non-improving evaluations. With
`validation_restore_best`, the packed parameters are restored transactionally
to the best validation point before the step reports the stop. A patience of
zero records diagnostics without early stopping.

Validation state is part of the schema-8 formatted trainer checkpoint. The
checkpoint stores options, metric history, best parameters, counters, and
stop flags. Callback procedures are process-local and are never serialized.
Loading a checkpoint that contains validation state therefore requires the
caller to attach a validation callback with the same presence. Missing or
unexpected callbacks are typed transactional refusals. The generic trainer
does not claim a CUDA path because the callback and its data remain host-owned.

The independent quadratic oracle in `test/test_trainer.f90` checks the
known-answer metric sequence, best-state restoration, schema round-trip, and
continuation after a split checkpoint. The release workload is
`fortml_bench_trainer_validation`.
