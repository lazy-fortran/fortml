# MLP learning-rate schedules

`fortml_mlp_schedules` provides stateless, validated schedules that can be
selected with `mlp_training_options_t%use_typed_schedule`, routed through the
existing `mlp_learning_rate_schedule_proc` callback, or used from a
differentiable optimizer trajectory. The schedule never owns a mutable epoch
cursor: callers pass the one-based update number explicitly, which keeps
replay and checkpoint behavior deterministic.

```fortran
use fortml_mlp_schedules, only: &
    mlp_learning_rate_schedule_t, make_mlp_schedule_warmup_cosine

type(mlp_learning_rate_schedule_t) :: schedule
schedule = make_mlp_schedule_warmup_cosine(100, 10_000, 0.05_dp)
```

The constructors are:

- `make_mlp_schedule_constant()`;
- `make_mlp_schedule_linear_warmup(warmup_updates)`;
- `make_mlp_schedule_cosine_decay(total_updates, min_rate_fraction)`;
- `make_mlp_schedule_warmup_cosine(warmup_updates, total_updates, min_rate_fraction)`;
- `make_mlp_schedule_exponential_decay(warmup_updates, decay_factor)`;
- `make_mlp_schedule_one_cycle(warmup_updates, total_updates,
  peak_rate_fraction, final_rate_fraction)`;
- `make_mlp_schedule_plateau(patience_updates, min_delta, factor,
  metric_mode)`.

`rate(update, base_rate, rate, status)` returns the effective positive rate.
The cosine schedules clamp after `total_updates`; exponential decay holds the
base rate through warm-up and then multiplies by `decay_factor` once per
update. One-cycle starts at `base_rate`, rises linearly to
`base_rate*peak_rate_fraction`, and follows a cosine tail to
`base_rate*final_rate_fraction`; updates after `total_updates` stay at the
final rate. `valid()` rejects invalid update counts, non-finite values,
fractions outside their family domains, and decay factors outside `(0,1)`.

For hyperparameter optimization, call
`rate_with_derivatives(update, base_rate, rate, d_base, d_min_fraction,
d_decay_factor, status)`. The products are analytic: no finite differences are
used. Derivatives for parameters unused by a schedule are exactly zero. This
makes a schedule safe to include in a JVP/VJP recurrence while the trainer's
existing custom callback remains available for application-specific policies.

One-cycle exposes the additional `peak_rate_fraction` and
`final_rate_fraction` products through
`rate_with_full_derivatives(update, base_rate, rate, d_base, d_min_fraction,
d_decay_factor, d_peak_fraction, d_final_fraction, status)`. The two added
products are exact and are zero for the other schedule families. Integer
warm-up and total-update counts are structural controls, not differentiable
coordinates.

`rate_with_second_derivatives` adds the exact five-slot raw rate Hessian in
the order `(base_rate, min_rate_fraction, decay_factor, peak_rate_fraction,
final_rate_fraction)`.  Cosine and warm-up factors are affine in their minimum
fraction, one-cycle factors are affine in their peak/final fractions, and the
exponential factor contributes its analytic second derivative in the decay
factor.  This product is used by the affine scheduled MLP outer-HVP objective;
it is not a finite-difference approximation.  The schedule kind and integer
update counts remain fixed structural controls.

## Metric-aware plateau schedule

`MLP_SCHEDULE_PLATEAU` is stateless. The caller supplies the current metric,
the best metric so far, the consecutive non-improvement count, and the number
of reductions already applied. `rate_with_metric` returns the effective rate
and the next values of all four state variables. A minimizing schedule marks
an improvement when `metric < best_metric-min_delta`. A maximizing schedule
uses `metric > best_metric+min_delta`. An improvement updates the best value
and clears the bad counter. Otherwise the counter increases. Once it reaches
`patience_updates`, one reduction is applied, the counter is cleared, and the
reduction count increases. The effective rate is
`base_rate*factor**next_reductions`.

`rate_with_metric_derivatives` also returns exact products for the base rate
and factor. Products with respect to metric, best metric, and `min_delta` are
zero on the selected comparison branch. The comparison and integer patience
decisions are discrete controls. Their branch-boundary convention is therefore
the documented zero product, not a hidden finite-difference approximation.
The ordinary `rate` surface returns a typed refusal for a plateau schedule
because it does not own a metric state channel. `mlp_train` is validation-aware:
it observes validation loss at each completed epoch when a held-out stream is
present (training loss otherwise), owns the four state variables, and carries
them in the version-11 checkpoint. A split/resumed run therefore uses the same
reduction sequence as an uninterrupted run. Custom trainer adapters can still
call the metric-aware method directly.

## Trainer integration

Pass a built-in schedule directly to `mlp_train` to keep the schedule in the
training contract rather than routing it through a procedure pointer:

```fortran
schedule = make_mlp_schedule_warmup_cosine(100, 10_000, 0.05_dp)
options%use_typed_schedule = .true.
options%typed_schedule = schedule
call mlp_train(model, x, target, status, options, state, checkpoint=checkpoint)
```

The trainer validates the schedule once and evaluates its exact rate at every
optimizer update. A typed schedule and a custom callback are mutually
exclusive. Its structural and continuous fields are copied into the portable
checkpoint; resumed options must provide the same typed schedule, otherwise
the trainer returns a domain error instead of changing the optimization path.
Typed schedules are currently CPU-only and have no hidden CUDA fallback.

```fortran
subroutine scheduled_rate(epoch, update, base_rate, rate)
    integer, intent(in) :: epoch, update
    real(dp), intent(in) :: base_rate
    real(dp), intent(out) :: rate
    type(fortnum_status_t) :: status
    real(dp) :: d_base, d_min, d_decay

    call schedule%rate_with_derivatives(update, base_rate, rate, d_base, &
        d_min, d_decay, status)
    if (.not. status_ok(status)) rate = -1.0_dp
end subroutine scheduled_rate
```

For example, a one-cycle callback can retain the full tangent products for an
outer FortOpt objective:

```fortran
schedule = make_mlp_schedule_one_cycle(100, 10_000, 6.0_dp, 0.05_dp)
call schedule%rate_with_full_derivatives(update, base_rate, rate, d_base, &
    d_min, d_decay, d_peak, d_final, status)
```

The independent `test_mlp_schedules` fixture checks every recurrence,
transition and terminal value, finite-difference oracles for each continuous
field, and typed refusal of malformed schedules.
`test_mlp_plateau_schedule` checks minimizing and maximizing transitions,
patience reset semantics, reduction compounding, active-branch derivatives,
malformed configurations, checkpoint round trips, invalid checkpoint refusal,
and the trainer capability boundary.
