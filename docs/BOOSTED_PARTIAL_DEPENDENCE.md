# Boosted-tree partial dependence

`fortml_boosted_partial_dependence` computes one-feature partial dependence
and individual conditional expectation (ICE) values for fitted `xgboost_t`
and `lightgbm_t` models. The implementation replaces one input column with
each supplied grid value, predicts every resulting row, and averages those
predictions. Optional nonnegative sample weights define the average.

```fortran
use fortml_boosted_partial_dependence, only: boosted_partial_dependence

real(dp) :: average(size(grid)), ice(size(x, 1), size(grid))

call boosted_partial_dependence(model, x, feature_index, grid, average, &
    status, sample_weight=weight, individual=ice)
```

The default response is the estimator's transformed prediction. Pass
`FORTML_TREE_RESPONSE_MARGIN` to average raw margins. This distinction matters
for binary logistic models because the average probability generally differs
from the logistic transform of the average margin.

The operation costs `size(grid)` model predictions and stores one working copy
of `x`. Supplying `individual` additionally stores an
`size(x, 1) * size(grid)` ICE matrix. Outputs are transactional: invalid
features, grids, weights, output shapes, models, or devices leave caller-owned
arrays unchanged.

CPU execution is available. `boosted_partial_dependence_device_supported`
reports the capability by device kind. A selected CUDA device returns
`FORTNUM_NOT_IMPLEMENTED` because the boosted-tree predictor and intervention
matrix do not yet have a resident CUDA implementation. No host replay is
reported as GPU work.

`test_boosted_partial_dependence` checks a fitted one-split model against
hand-computed leaf values. It also checks weighted aggregation, ICE ordering,
the binary response link, transactional errors, and CUDA refusal for both tree
families. `fortml_bench_boosted_partial_dependence` exposes the same fixed
fixture to the independent NumPy benchmark oracle.
