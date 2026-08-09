# Classifier-chain clone contract

`classifier_chain_t%clone` provides the host-side reset seam for fitted
classifier chains. It copies the trained logistic heads, sorted integer class
pairs, per-output thresholds, and packed parameter-size metadata in one
transaction. A malformed or unfitted source returns `FORTNUM_DOMAIN_ERROR`
without changing an existing destination, which makes the operation safe for
validation and model-selection candidates.

The copied heads are independent allocatable state. Updating a clone's packed
parameters therefore changes only the clone's probabilities; the source keeps
its original chain prediction. `test_classifier_chain` checks this mutation
behavior against an independent sequential sigmoid oracle rather than merely
comparing internal fields.

`clone_device` is explicit about placement. A selected CPU device calls the
same exact deep copy. A selected CUDA device returns
`FORTNUM_NOT_IMPLEMENTED` until resident chain state is linked, with no hidden
host fallback or transfer. The source and destination remain unchanged on
that CUDA refusal.
