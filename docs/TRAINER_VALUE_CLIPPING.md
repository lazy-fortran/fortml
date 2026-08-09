# Generic trainer gradient value clipping

`fortml_trainer` now supports both global-norm and per-coordinate gradient
clipping. Set `trainer_options_t%gradient_clip_value` to a finite positive
bound to clamp every objective-gradient coordinate to
`[-gradient_clip_value, gradient_clip_value]` before the optimizer update.
The default zero disables value clipping. Global norm clipping, when enabled,
is applied after this coordinate-wise bound.

The operation is transactional. Objective products are checked for finite
values before clipping, and optimizer failures restore the pre-update
parameters. `trainer_state_t%value_clipped_steps` records the number of
accepted updates in which at least one coordinate was clamped. The existing
`clipped_steps` counter remains the independent global-norm diagnostic.

The value bound and diagnostic counter are persisted in the formatted trainer
checkpoint (schema 8). Loading validates the bound, counters, optimizer state,
and callback presence before replacing the destination. This feature is a
CPU objective operation; `partial_fit_device(FORTML_DEVICE_CUDA, ...)` retains
the explicit resident-CUDA typed refusal rather than copying through the host.

An independent quadratic oracle in `test_trainer` checks the exact SGD update,
counter semantics, and checkpoint round trip.
