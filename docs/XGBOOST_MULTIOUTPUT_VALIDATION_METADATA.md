# Multi-output booster validation metadata

`xgboost_multioutput_t` and `lightgbm_multioutput_t` now expose validation
state for every output model. `best_iteration()` returns one-based selected
rounds, `best_validation_loss()` returns the corresponding loss values, and
`early_stopped()` reports the patience boundary for each output. The arrays
are output-major and retain their shape when output models stop at different
rounds.

The accessors read child state after a transactional multi-output fit. An
invalid fit leaves the destination model unchanged, so metadata cannot mix a
new output prefix with an old sibling. The tree topology remains discrete and
its input derivative boundary is unchanged.

The independent test `test_xgboost_multioutput_validation_metadata` fits each
scalar output without validation, computes every staged validation loss from
the returned margins, selects the expected round, and compares that oracle
with the multi-output metadata for both XGBoost-style depth growth and
LightGBM-style leaf growth. It also checks the early-stop flags and loss
vectors.

The multi-output metadata is CPU state. Existing device prediction methods
continue to return a typed CUDA refusal until resident multi-output tree state
is linked.
