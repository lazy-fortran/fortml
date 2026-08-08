# Bagging classifier

`fortml_bagging_classifier` provides a deterministic binary or multiclass
bagging estimator around the existing weighted CART classifier. It supports
seeded bootstrap and without-replacement subsets, aligned class probabilities,
and an explicit CPU/CUDA boundary.

## Fit and predict

Rows are samples and columns are features. Integer labels may have arbitrary
values; the fitted `classes()` array is sorted and defines probability-column
order:

```fortran
use fortml_bagging_classifier, only: bagging_classifier_t

type(bagging_classifier_t) :: model
type(fortnum_status_t) :: status
real(dp) :: x(n_samples,n_features), probabilities(n_query,n_classes)
integer :: labels(n_samples), prediction(n_query)

call model%fit(x, labels, status, n_trees=32, max_samples=180, &
    bootstrap=.true., seed=1729)
call model%predict_proba(x_query, probabilities, status)
call model%predict(x_query, prediction, status)
```

`n_trees` is bounded by `BAGGING_MAX_ESTIMATORS` (256). The defaults are ten
trees, depth three, one observation per leaf, and all training rows per
subset. `max_samples` is an integer row count. The sampler is seeded and
reproducible; one first occurrence of every sorted class is forced into each
subset, then remaining rows are drawn with replacement when `bootstrap=.true.`
or without replacement otherwise. This class-coverage rule makes every CART
tree expose the same probability-column layout. `criterion` accepts Gini or
entropy, and `missing_policy` is forwarded to each CART fit. Positive finite
sample weights are copied with sampled rows.

`predict_proba` averages per-tree leaf probabilities and `predict` uses the
first maximum to map back to the original labels. Metadata accessors expose
tree count, subset size, depth, leaf size, criterion, seed, bootstrap policy,
feature/class counts, and fitted state.

## Derivatives and devices

CART routing is discrete and the bagging wrapper does not pretend that the
piecewise map is globally smooth. `predict_proba_jvp` and
`predict_proba_vjp` return `FORTNUM_NOT_IMPLEMENTED` with zeroed products. A
selected CPU `fortml_device_t` executes the host prediction. A selected CUDA
device returns `FORTNUM_NOT_IMPLEMENTED` without silently copying through the
host until a resident ensemble kernel is available; `device_supported` reports
this boundary explicitly.

The independent `test_bagging_classifier` fixture checks depth-zero empirical
class probabilities, sample weighting, seeded bootstrap determinism, subset
class coverage, malformed options, discrete derivative refusals, and output
preservation for CUDA requests. `fortml_bench_bagging_classifier` records fit
and prediction timings, simplex error, query accuracy, and the CUDA refusal.
