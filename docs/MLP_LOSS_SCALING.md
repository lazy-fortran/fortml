# MLP automatic loss scaling

`mlp_loss_scale_state_t` is the explicit numerical state used by the dense
MLP trainer for automatic mixed-precision loss scaling.  It keeps the scale,
growth/backoff policy, good-update streak, overflow count, and skipped-update
count in one validated value type:

```fortran
type(mlp_training_options_t) :: options
type(fortnum_status_t) :: status

call options%loss_scale%initialize(status, enabled=.true., &
    initial_scale=256.0_dp, growth_factor=2.0_dp, backoff_factor=0.5_dp, &
    growth_interval=2000, minimum_scale=1.0_dp, maximum_scale=16777216.0_dp)
```

The recurrence is deterministic.  A finite update increments `good_steps`;
when the configured interval is reached, the scale is multiplied by
`growth_factor` and the streak is reset.  An overflow clears the streak,
increments `overflow_count` and `skipped_updates`, and multiplies the scale by
`backoff_factor`, bounded by `minimum_scale`.  All transitions are validated
for finite values and configured bounds.

The FP64 trainer may enable the state as a CPU recurrence/reference lane.  It
checks the scaled gradient for IEEE overflow and skips the optimizer update on
an overflow; with the default disabled state, existing FP64 trajectories are
bit-for-bit unchanged.  FP32, FP16, and BF16 training still return
`FORTNUM_NOT_IMPLEMENTED`: master weights, resident lower-precision kernels,
and device loss-scaling reductions are not claimed until independently gated.
No CUDA path is inferred from the option.

The dynamic state is captured in `mlp_training_checkpoint_t` and in the
versioned formatted checkpoint schema 11.  Resume validates the static policy
(`initial_scale`, bounds, factors, and interval) while restoring the dynamic
scale and counters transactionally.  A malformed or stale scale state is a
typed domain error and cannot partially replace a checkpoint.

`test_mlp_loss_scaling` is an independent recurrence oracle, tests overflow
backoff and growth boundaries, verifies formatted checkpoint round-tripping,
and checks that unsupported lower precision refuses without mutating model
parameters.
