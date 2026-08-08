# Random-forest permutation importance

`random_forest_classifier_t%permutation_importance` reports the mean decrease
in classification accuracy after independently permuting each query feature.
The fitted CART trees, bootstrap inclusion matrix, and all split thresholds are
read-only.  The operation is therefore a fixed-state diagnostic: it does not
refit trees, change routing, or expose an input/parameter derivative.  A
permutation on a split boundary is not treated as a differentiable surrogate.

```fortran
real(real64) :: importance(3), importance_std(3), baseline
call forest%permutation_importance(x, labels, importance, status, &
    n_repeats=24, seed=991, importance_std=importance_std, &
    baseline_score=baseline)
```

The baseline is the fitted forest accuracy on the supplied finite query rows.
For each feature and repeat, a deterministic Park--Miller
Fisher--Yates permutation is applied to that column only.  The returned
importance is `baseline - permuted_accuracy`; `importance_std` is the
population standard deviation over repeats.  Defaults are five repeats and a
fixed positive seed (`RANDOM_FOREST_DEFAULT_SEED + 7919`).  Repeats are bounded
by `RANDOM_FOREST_MAX_PERMUTATION_REPEATS` (1024).

All output arguments are transactional.  Invalid fitted state, dimensions,
labels outside the fitted classes, non-finite inputs, options, or a prediction
failure leave every supplied output untouched and return
`FORTNUM_DOMAIN_ERROR`.  The device overload executes on a selected CPU and
returns `FORTNUM_NOT_IMPLEMENTED` for CUDA, preserving `importance`, optional
standard deviations, and the baseline scalar; there is no hidden host
fallback.  The release workload and independent NumPy replay are in
`../fortml-bench/results/random_forest_permutation.csv` and
`../fortml-bench/scripts/bench_random_forest_permutation.py`.
