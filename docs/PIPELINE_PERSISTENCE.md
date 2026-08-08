# Basis-pipeline persistence

`fortml_pipeline_persistence` provides a small, compiler-independent state
dictionary for a fitted horizontal `basis_pipeline_t`. The dictionary is
versioned (`FORTML_PIPELINE_STATE_VERSION = 1`) and carries the complete
routing contract: input schema names, stage names, derived feature and
parameter names, one-based stage offsets, fit state, dimensions, and the
packed trainable parameter vector.

The persistence boundary is deliberately separate from basis construction.
Loaders restore values onto a preconfigured pipeline, so a file cannot invent
executable callback procedures or silently choose a different polynomial,
Fourier, radial, or spline topology. Callback maps therefore require an
application-owned registration layer and are rejected if the target topology
or metadata does not match.

## API

```fortran
type(basis_pipeline_state_t) :: state
call capture_basis_pipeline_state(pipeline, state, status)
call save_basis_pipeline_text(pipeline, "model.fml", status)
call load_basis_pipeline_text(preconfigured_fitted_pipeline, "model.fml", status)
```

`capture_basis_pipeline_state` is read-only. `state%valid()` checks the schema,
dimensions, unique names, offsets, and finite parameters. `restore_...` first
validates every name and offset and applies parameters to a deep candidate copy;
the candidate replaces the caller's pipeline only after all updates succeed.
Malformed files, version mismatches, dimension changes, metadata changes, and
nonfinite values leave the target untouched with `FORTNUM_DOMAIN_ERROR`.

The target must already contain the same basis-map topology and fit state. This
is intentional: structural descriptors (knots, frequencies, callback addresses)
are executable configuration rather than untrusted file data. A caller can
construct and fit the topology, then restore its packed state and schema.

The text format is line-oriented and begins with
`FORTML_BASIS_PIPELINE_STATE 1`. It has explicit `counts`, `fitted`, name,
offset, parameter, and `end` records. Unknown or truncated records are refused;
there is no partial publication of a checkpoint. Future schema versions must
add a migration before they are accepted.

## Device boundary

`save_basis_pipeline_device` and `load_basis_pipeline_device` accept an
explicit `FORTML_DEVICE_CPU` or `FORTML_DEVICE_CUDA` kind. CPU dispatch uses
the exact host text path. CUDA and resident serialization return
`FORTNUM_NOT_IMPLEMENTED` without changing the model or pretending that a
host file operation is a resident-device checkpoint. The independent
`test_pipeline_persistence` oracle checks this refusal as well as value,
JVP/VJP/HVP, metadata, and malformed-input behavior.

The current slice persists horizontal fitted feature unions. Sequential,
fan-out, residual, linear-regression coefficient, trainer, sparse, and
device-resident graph schemas remain separate versioned contracts rather than
being encoded ambiguously in this format.
