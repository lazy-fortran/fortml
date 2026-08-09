# Learned basis fan-in

`basis_blend_pipeline_t` combines named sequential basis branches that have
the same input and output shapes. For branches `f_i` and scalar weights `w_i`,
the transform is

```text
y(x) = sum_i w_i f_i(x).
```

The weights are trainable coordinates. They share the packed parameter vector
with each branch's basis parameters. The layout is branch-major: one weight,
then that branch's parameters, followed by the next branch. The
`branch_parameter_offset`, `parameter_name`, `branch_name`, and
`branch_weight` accessors expose the layout without duplicating offset logic in
an optimizer or persistence layer.

```fortran
use fortml_basis_blend_pipeline, only: basis_blend_pipeline_t, &
    make_basis_blend_pipeline

type(basis_blend_pipeline_t) :: blend

blend = make_basis_blend_pipeline(2, status)
call blend%append(polynomial_branch, 1.0_dp, status, name="trend")
call blend%append(fourier_branch, 0.1_dp, status, name="correction")
call blend%fit(x, status)
call blend%transform(x, features, status)
```

`append` requires an initialized, valid `sequential_basis_pipeline_t`. Every
branch must consume `input_count()` columns and return the same
`feature_count()`. Names must be unique and weights must be finite. Failed
appends leave the branch list, weights, fit state, and packed layout unchanged.
`fit` and `set_parameters` validate a deep candidate before committing it, so
a failed branch update cannot leave part of the graph changed. Dense input
names use the same transactional `basis_input_schema_t` contract as the other
basis pipelines.

The CPU API includes value, JVP, VJP, and scalar-contraction HVP products. All
three derivative products cover the mixing weights, branch parameters, and
inputs. In particular, the weight tangent contributes `w_dot_i f_i(x)` to a
JVP, and the weight cotangent is the contraction of the output cotangent with
the branch output. HVPs include the cross terms between a weight direction and
the branch parameter or input gradient.

`capabilities` reports dense CPU transform and input/parameter JVP, VJP, and
HVP support. CUDA residency and OpenACC execution are false capability rows.
The device methods execute for a selected CPU. Selected CUDA and OpenACC
requests return `FORTNUM_NOT_IMPLEMENTED` before changing caller-owned output
arrays. This boundary prevents a requested accelerator operation from becoming
an implicit host copy.

`test_basis_blend_pipeline` checks values against direct polynomial and Fourier
maps. It also checks the JVP/VJP adjoint identity, HVPs against central
differences, weight gradients against an analytic contraction, transactional
refusals, metadata, and CPU/CUDA/OpenACC dispatch. The release workload is
`fortml-bench/results/BASIS_BLEND_PIPELINE.md`.
