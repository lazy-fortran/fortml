# Multi-output radius-neighbors regression

`fortml_radius_neighbors_multioutput_regression` provides a dense,
multi-output `RadiusNeighborsRegressor`-style estimator. Training rows are
stored as `x(n_samples,n_features)` and `targets(n_samples,n_outputs)`. A
query includes every training row whose squared Euclidean distance is at most
`radius**2`; one weighted neighborhood is shared by every target column.

```fortran
use fortml_radius_neighbors_multioutput_regression, only: &
    radius_neighbors_multioutput_regressor_t

type(radius_neighbors_multioutput_regressor_t) :: model
type(fortnum_status_t) :: status
real(dp) :: x(n_samples,n_features), y(n_samples,n_outputs)
real(dp) :: y_query(n_query,n_outputs)

call model%fit(x, y, status, radius=0.75_dp, &
    weights=RADIUS_MULTI_REGRESSION_WEIGHTS_DISTANCE)
call model%predict(x_query, y_query, status)
```

The `weights` option is either
`RADIUS_MULTI_REGRESSION_WEIGHTS_UNIFORM` or
`RADIUS_MULTI_REGRESSION_WEIGHTS_DISTANCE`. Inverse-distance weighting uses
sample weight divided by Euclidean distance; when one or more selected rows
coincide exactly with a query, only those exact rows receive weight. Optional
nonnegative `sample_weight` values scale all output columns together and must
have positive total mass. An optional `outlier_value(n_outputs)` supplies the
vector returned for an empty neighborhood; without it, prediction returns a
typed domain error.

## Derivatives and devices

Radius membership and inverse-distance neighbor sets are piecewise. For a
query that is not on a radius boundary, `predict_jvp` and `predict_vjp` return
the exact zero product of the fixed-state map. A query at a boundary returns
`FORTNUM_DOMAIN_ERROR`, rather than silently treating the discrete selection
as smooth. The methods include shape and finite-value checks and preserve the
same multi-output ordering as `predict`.

The ordinary estimator is CPU-only. A selected CPU `fortml_device_t` delegates
to the host implementation. A selected CUDA device returns
`FORTNUM_NOT_IMPLEMENTED` until a resident radius-search reduction kernel is
available, with no hidden host fallback; `device_supported` reports this
boundary explicitly.

The independent `test_radius_neighbors_multioutput_regression` fixture checks
uniform and inverse-distance averages, sample-weighted columns, exact-neighbor
handling, vector outliers, empty-neighborhood errors, boundary derivative
refusals, and the CUDA output-preservation contract.
