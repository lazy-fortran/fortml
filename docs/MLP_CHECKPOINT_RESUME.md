# MLP checkpoint and resume contract

`mlp_training_checkpoint_t` is the complete deterministic trainer snapshot:
model parameters, optimizer moments, schedule counters, EMA/best state, the
seeded batch permutation and cursor, and pending gradient-accumulation state.
The formatted text writer and reader use schema 12.  Schema 12 persists
`accumulated_weight_mass` as a binary64 scalar; this matters when a checkpoint
is taken between weighted microbatches because replacing it with the integer
sample count changes the next normalized update.

Save/load is transactional.  A malformed, truncated, stale, or incompatible
snapshot leaves the destination unchanged and returns a typed domain error.
Resume still validates the static data/training policy before restoring the
dynamic trajectory; callers must install custom schedule and callback
procedures again because procedure pointers are not serialized.

The formatted file is host-owned.  `mlp_checkpoint_require_device` returns
`FORTNUM_OK` for `FORTML_DEVICE_CPU` and a typed `FORTNUM_NOT_IMPLEMENTED` for
`FORTML_DEVICE_CUDA`; no CUDA request silently copies through a host fallback.
A resident CUDA trainer must provide an explicit device-to-host snapshot before
using this format.  `mlp_checkpoint_device_supported` exposes the same
capability predicate for dispatch code.

`test_mlp_checkpoint_io` supplies independent behavioral oracles for exact
non-integral pending weight-mass round trips, resume state, malformed input,
and the CPU/CUDA capability boundary.
