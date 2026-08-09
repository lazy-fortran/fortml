# Deterministic linear SGD

`fortml_linear_sgd` provides two CPU reference estimators:
`linear_sgd_regression_t` for dense vector or multi-output least squares and
`linear_sgd_classifier_t` for binary logistic classification. They are
deliberately separate from the exact `linear_regression_t`,
`logistic_regression_t`, and `softmax_regression_t` objectives: the update
order is part of the stochastic contract.

## Update contract

Inputs use `x(n_samples,n_features)`. A `linear_sgd_options_t` records
`batch_size`, `epochs`, an optional seeded shuffle, `learning_rate`, and either
the constant or inverse-scaling schedule. Each non-empty batch computes a
sample-weighted mean gradient, adds L2 to the coefficient block, applies one
gradient step, then applies the L1 proximal soft-threshold. The intercept is
never penalized. Zero-mass batches are skipped. `average=.true.` accumulates
post-update parameters and prediction uses their Polyak average.

```fortran
use fortml_linear_sgd, only: linear_sgd_options_t, linear_sgd_regression_t
type(linear_sgd_options_t) :: options
type(linear_sgd_regression_t) :: model
real(dp) :: x(n, p), y(n), prediction(n)
type(fortnum_status_t) :: status

options%epochs = 20
options%batch_size = 16
options%learning_rate = 5.0e-2_dp
options%shuffle = .true.
options%shuffle_seed = 29
options%average = .true.
call model%fit(x, y, status, options)
call model%predict(x, prediction, status)
```

`partial_fit` consumes exactly one epoch, regardless of `options%epochs`, and
retains the fitted coefficients, update count, Polyak sums, and shuffle stream.
The first classifier call may provide `classes=[negative,positive]`; labels
must remain in that sorted pair on later calls. Supplying changed options or
classes after initialization is a domain error. `parameters` and
`set_parameters` use `[intercept, coefficient...]` for one output and
Fortran column-major `[intercept, coefficient...]` blocks for multi-output
regression.

The CPU path is complete for this bounded contract. `predict_device` and
`device_supported(FORTML_DEVICE_CUDA)` return a typed refusal because there is
no resident stochastic loader or optimizer state; no host fallback is hidden.
Stochastic-path JVP/VJP/HVP products and derivative-through-fit remain open.

The independent recurrence, continuation, averaging, label, and device tests
are in [`test_linear_sgd.f90`](../test/test_linear_sgd.f90).
