# MLP learning-rate schedules

`fortml_mlp_schedules` provides stateless, validated schedules that can be
used directly from the existing `mlp_learning_rate_schedule_proc` callback or
from a differentiable optimizer trajectory. The schedule never owns a mutable
epoch cursor: callers pass the one-based update number explicitly, which keeps
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
  peak_rate_fraction, final_rate_fraction)`.

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
