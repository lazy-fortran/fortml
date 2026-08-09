# Cross-validation scoring and search

`fortml_cross_validation` combines the deterministic index-only splitters from
`fortml_validation` with a metadata-driven scorer callback.  A callback is
called once per fold with disjoint one-based train and test indices and returns
the natural-orientation fold score, its parameter gradient, and a positive fold
weight.  The callback owns fitting a fresh clone or resetting the estimator;
the evaluator refuses metadata that declares neither operation, so fitted state
cannot leak between folds by accident.

```fortran
use fortml_cross_validation, only: cross_validation_evaluate, &
    cross_validation_result_t

call cross_validation_evaluate(splitter, metadata, theta, estimator_state, &
    fold_score, result, status)
```

`cross_validation_result_t` stores every fold score, fold weight, fold
gradient, the weighted mean or weighted sum, scorer-oriented value, and the
minimization value/gradient consumed by FortOpt.  Select
`FORTML_CV_WEIGHTED_MEAN` (the default) or `FORTML_CV_WEIGHTED_SUM` through
`cross_validation_options_t`.  A lower-is-better scorer is oriented by the
metadata record; its objective value is therefore `-oriented_value` for
FortOpt.

For differentiable hyperparameter search, initialize a borrowed
`cross_validation_objective_t` with any supported K-fold, stratified, grouped,
or chronological splitter and call `as_objective`.  The resulting
`fortopt_objective::objective_t` can be passed directly to the existing grid,
random, or bounded L-BFGS-B search drivers.  Parameters, scorer metadata, and
clone/reset declarations are checked on every evaluation.  Search and split
control is CPU-only; a CUDA request returns `FORTNUM_NOT_IMPLEMENTED` rather
than introducing a hidden host callback.

The independent oracle is `test_cross_validation`: it checks weighted fold
aggregation, orientation and parameter products, FortOpt callback parity,
callback/fold accounting, clone/reset leakage refusal, and typed CUDA refusal.
