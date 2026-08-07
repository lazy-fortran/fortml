# Transform-aware hyperparameter registry

`fortml_hyperparameter_registry` is the optimizer-facing companion to
`fortml_parameter_registry`. A `hyperparameter_block_t` names one contiguous
parameter block and records its physical bounds, coordinate transform,
trainability, provenance, device label, and whether an objective supplies an
analytic HVP for the block. The registry does not approximate derivatives or
own model callbacks; it only provides a deterministic coordinate contract for
an outer objective and FortOpt L-BFGS-B.

## Blocks

Use `initialize` to adapt an existing model callback, or
`initialize_values` for a small owned block (useful for schedules, likelihood
parameters, and tests):

```fortran
type(hyperparameter_block_t) :: rate
type(fortnum_status_t) :: status

call rate%initialize_values("optimizer.learning_rate", [1.0e-2_dp], status, &
    transform=HP_TRANSFORM_LOG, lower=[1.0e-6_dp], upper=[1.0_dp], &
    provenance="training.schedule", device="cpu", hvp_available=.true.)
```

The identity transform leaves coordinates unchanged. The log transform maps
positive physical values to `log(value)`. The logit transform maps a physical
value strictly inside each explicit `(lower, upper)` interval to
`log((value-lower)/(upper-value))`; finite, distinct bounds are mandatory.
Bounds are checked on every physical update. Invalid shapes, non-finite
values, duplicate names, and transform/bound combinations that cannot be
represented are rejected with `FORTNUM_DOMAIN_ERROR`.

`get_physical`/`set_physical` operate in model coordinates. The corresponding
`get_unconstrained`/`set_unconstrained` methods operate in optimizer
coordinates. `unconstrained_bounds` returns the bounds to pass to a bounded
optimizer; `project_unconstrained` clips one block to those bounds.

For a smooth separable transform, `physical_derivatives` returns `p(u)`,
`dp/du`, and `d2p/du2`. The registry-level `unconstrained_gradient` and
`unconstrained_hvp` methods pull physical objective products into the exact
trainable optimizer coordinates. In particular, for a physical gradient `g`,
physical HVP evaluated along `p'*v`, and optimizer direction `v`, the returned
product is `p'*(H*(p'*v)) + g*p''*v` blockwise. This keeps transform curvature in bounded
L-BFGS-B hyperparameter optimization without finite differences.

## Registry and L-BFGS-B vectors

```fortran
type(hyperparameter_registry_t) :: registry
real(dp) :: x(n), lower(n), upper(n)

call registry%add(rate, status)
call registry%pack_trainable(x, status)
call registry%optimizer_bounds(lower, upper, status)
call registry%project(x, status)
call registry%unpack_trainable(x, status)
```

`pack` and `pack_unconstrained` include every block. The optimizer variants
skip blocks marked `trainable=.false.` and preserve insertion order. This makes
the same vector suitable for FortOpt's projected L-BFGS-B callback while frozen
parameters remain in the model. `range(name, ..., trainable_only=.true.)`
reports optimizer-vector offsets. Provenance/device labels are metadata for
the surrounding trainer and are intentionally not interpreted by this layer;
a device-specific objective must refuse unsupported lowering explicitly.

The registry records HVP availability as metadata (`hvp_available`). A caller
must only route a second-order callback through FortOpt when every participating
block and objective has an analytic HVP; no finite-difference fallback is
silently introduced.
