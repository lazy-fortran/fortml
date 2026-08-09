# Conditional basis feature unions

`fortml_conditional_pipeline` adds a bounded piecewise-routing layer on top of
the existing `column_basis_pipeline_t` primitive.  It is useful for
mixture-of-experts features, regime-dependent Fourier/polynomial expansions,
and explicit fallback branches without introducing a second basis-map or
scatter implementation.

```fortran
use fortml_basis, only: basis_map_t, make_fourier_basis
use fortml_conditional_pipeline, only: conditional_basis_pipeline_t, &
    make_conditional_basis_pipeline

type(conditional_basis_pipeline_t) :: union
type(basis_map_t) :: low, high
type(fortnum_status_t) :: status

low = make_fourier_basis(1, reshape([0.8_dp], [1, 1]), status)
high = make_fourier_basis(1, reshape([1.2_dp], [1, 1]), status)
union = make_conditional_basis_pipeline(2, status)
call union%append(low, [2], 1, -huge(1.0_dp), 0.0_dp, status, name="low")
call union%append(high, [2], 1, 0.0_dp, huge(1.0_dp), status, name="high")
call union%fit(x, status)
call union%transform(x, features, status)
```

Each branch is active for rows whose route-column value satisfies the
half-open interval `[lower_bound, upper_bound)`.  Branches may overlap or
leave gaps; inactive feature rows are zero.  The selected columns and basis
derivatives are delegated to `column_basis_pipeline_t`, so the packed layout
is deterministic and parameter blocks remain branch-major.  Use
`branch_feature_offset` and `branch_parameter_offset` for one-based offsets,
and `branch_name`, `feature_name`, `parameter_name`,
`branch_route_column`, `branch_lower_bound`, and `branch_upper_bound` for
stable metadata.

The value map is defined at an interval endpoint by the half-open rule.  A
JVP, VJP, or HVP at an endpoint returns `FORTNUM_DOMAIN_ERROR`, because the
route mask is discontinuous there.  Nonfinite inputs, malformed intervals,
duplicate names, and shape mismatches are rejected before output or packed
state is changed.  `set_parameters` and `set_input_schema` use the same
transactional contract.  Input schemas are dense and named through
`set_input_schema`, `input_schema_name`, and `validate_input_schema`.

CPU `transform`, `jvp`, `vjp`, and `hvp` are exact products assembled from the
underlying basis maps.  Their device-dispatch counterparts delegate only to a
selected available CPU and return `FORTNUM_NOT_IMPLEMENTED` for CUDA without
touching caller-owned output arrays; a resident route-mask CUDA executor is a
separate roadmap item.

The independent finite-difference and adjoint oracle is
`test_conditional_pipeline`.  The release workload is
`fortml-bench/results/CONDITIONAL_PIPELINE.md`, generated from
`app/fortml_bench_conditional_pipeline.f90` and the companion NumPy reference.
