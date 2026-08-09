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

The state also owns the explicit gradient products used by a trainer:

```fortran
call options%loss_scale%scale_gradient(gradient, scaled_gradient, status)
if (options%loss_scale%scaled_gradient_finite(scaled_gradient)) then
    call options%loss_scale%unscale_gradient(scaled_gradient, gradient, status)
    ! Commit the optimizer update only after the status is successful.
else
    call options%loss_scale%observe(.false., .false., status)
    ! Skip the update and retry with the backed-off scale.
end if
```

`scale_gradient` and `unscale_gradient` are shape-checked, allocation-free
products.  `scaled_gradient_finite` catches both a non-finite source gradient
and overflow introduced by multiplication by the scale.  The FP64 reference
trainer now executes this scale/check/unscale sequence before every optimizer
commit, so finite trajectories are exactly invariant to enabling loss scaling
while overflow updates are transactionally skipped.

The trainer checks the scaled gradient for IEEE overflow, unscales it, and
skips the optimizer update on an overflow; with the default disabled state,
existing FP64 trajectories are bit-for-bit unchanged.  CPU FP32 uses binary64
master parameters with a rounded FP32 forward/gradient boundary and the same
transactional loss-scale state.  FP16 and BF16 storage, plus resident CUDA
loss-scaling reductions, still return `FORTNUM_NOT_IMPLEMENTED` until their
kernels are independently gated.  No CUDA path is inferred from the option.

When the scale-induced overflow branch is taken, the typed event callback
receives `MLP_EVENT_UPDATE_SKIPPED` (`update_skipped`).  Its update counter and
parameter vector remain unchanged; the loss-scale backoff and overflow/skipped
counters advance.  This event makes a replay observable without pretending
that a discarded optimizer step succeeded.

Growth and overflow branches are discrete policy decisions.  They are not
advertised as smooth hyperparameter JVPs or HVPs, and an outer derivative
objective must keep the same fixed branch or return its own typed active-set
refusal.

The dynamic state is captured in `mlp_training_checkpoint_t` and in the
versioned formatted checkpoint schema 11.  Resume validates the static policy
(`initial_scale`, bounds, factors, and interval) while restoring the dynamic
scale and counters transactionally.  A malformed or stale scale state is a
typed domain error and cannot partially replace a checkpoint.

`test_mlp_loss_scaling` is an independent recurrence oracle, tests overflow
backoff and growth boundaries, verifies formatted checkpoint round-tripping,
and checks that unsupported lower precision refuses without mutating model
parameters.  `test_mlp_training` additionally exercises the production FP32
overflow branch, its zero-update oracle, and the `update_skipped` event.
