# Random-forest out-of-bag products

`random_forest_classifier_t` stores one logical bootstrap-inclusion column per
tree.  `oob_decision_function(x, probabilities, status)` averages only trees
for which each training row was absent from the bootstrap sample.  `x` must
be the same row set used by `fit`; the output columns follow `classes()` and
are aligned even when a CART tree did not observe every class.

`oob_score(x, labels, score, status)` computes accuracy from the same OOB
probabilities.  `oob_coverage()` reports the fraction of fitted training rows
with at least one OOB tree, and `bootstrap_inclusion()` returns a defensive
copy of the `(n_samples,n_trees)` inclusion matrix for audit and reproducible
oracles.

The OOB methods are transactional: they never substitute an in-bag prediction.
If any row has no OOB tree, they leave the caller's output unchanged and return
`RANDOM_FOREST_OOB_INSUFFICIENT` (the generic
`FORTNUM_CONVERGENCE_ERROR` code).  Invalid shapes and non-finite inputs return
`FORTNUM_DOMAIN_ERROR`.  CPU device dispatch is supported; CUDA requests
return `FORTNUM_NOT_IMPLEMENTED` and preserve the output buffer or score.

```fortran
real(dp) :: probabilities(n_samples, n_classes), score
type(fortnum_status_t) :: status

call forest%oob_decision_function(x_train, probabilities, status)
call forest%oob_score(x_train, y_train, score, status)
```

The seeded bootstrap stream chooses one deterministic member from each class
for every tree and then fills the remaining draws with the same bounded linear
congruential stream.  Unlike a fixed first-row stratification this preserves
the possibility that every training row is OOB.  The benchmark
`fortml-bench/results/RANDOM_FOREST_OOB.md` checks this contract against an
independent threshold-label oracle and records the typed CUDA boundary.
