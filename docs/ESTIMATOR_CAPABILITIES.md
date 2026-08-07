# Estimator capability contract

`fortml_estimator_capabilities` is the common control-plane record used by
generic model-selection and pipeline code. It describes what an estimator
accepts and which products it can provide; it never turns an unavailable
implementation into a fallback.

```fortran
use fortnum_status, only: fortnum_status_t, status_ok
use fortml_estimator_capabilities, only: estimator_capability_t, &
    make_classifier_capabilities, FORTML_DERIVATIVE_PARAMETER_HVP

type(estimator_capability_t) :: tags
type(fortnum_status_t) :: status

tags = make_classifier_capabilities("my_classifier", 8, 3, status, &
    fitted=.true.)
tags%supports_parameter_hvp = .true.
if (.not. status_ok(status)) error stop "invalid capability"
if (tags%supports_derivative(FORTML_DERIVATIVE_PARAMETER_HVP)) then
    ! The caller may route an exact parameter HVP request here.
end if
```

The role mask uses `FORTML_ROLE_TRANSFORMER`, `FORTML_ROLE_PREDICTOR`,
`FORTML_ROLE_REGRESSOR`, and `FORTML_ROLE_CLASSIFIER`. A classifier and
regressor constructor also set the predictor role. Input tags are queried with
`supports_input` for dense, sparse, missing, sample-weight, and partial-fit
support. Derivative tags distinguish input, parameter, and hyperparameter
JVP/VJP/HVP products. Device tags distinguish CPU, CUDA, and OpenACC; the
separate `supports_resident` flag means that model and batch state stay on the
device rather than merely allowing a transfer-based call.

Requirements are ordinary records. Initialize one with the desired role and
shape, set only the boolean tags that must be true, and call
`require_estimator_capability(actual, requirement, status)`. The check is
monotone: false requirement tags are ignored, while every true tag must be
provided by the actual estimator. Invalid records or unmet requirements return
`FORTNUM_DOMAIN_ERROR`.

The horizontal `basis_pipeline_t`, sequential
`sequential_basis_pipeline_t`, and column-selecting
`column_basis_pipeline_t` expose `%capabilities(report,status)`. Their reports
declare fitted state and feature shape, exact input/parameter products, dense
CPU support, and explicit sparse/missing/sample-weight/CUDA refusals. The
validation module provides `validate_estimator_capability` and the same
requirement check so cross-validation and search can reject incompatible
workflows before consuming a split.

The contract is intentionally additive. Models that do not yet expose a
capability method remain usable through their existing APIs; model-wide
adoption, clone/reset metadata, sparse views, and metadata routing are tracked
separately in `ROADMAP.md`.
