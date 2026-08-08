# Metric-aware plateau training

`mlp_learning_rate_schedule_t` now supports a typed `MLP_SCHEDULE_PLATEAU`
schedule directly in `mlp_train`.  The schedule observes the completed epoch
metric (validation loss when a validation set is supplied, otherwise training
loss), compares it with the best metric using `min_delta`, and applies
`plateau_factor` after `patience_updates` non-improving observations.

The transition is deterministic and explicit:

```text
(best_metric, bad_updates, reductions)
  -> (next_best_metric, next_bad_updates, next_reductions, learning_rate)
```

The comparison is an active-set decision.  Products with respect to the
continuous base rate and plateau factor are analytic on the selected branch;
metric, best-value, `min_delta`, and integer counters have zero products at the
comparison boundary.  The standalone schedule API retains its typed refusal
for the stateless `rate` method because a metric state is required.

`mlp_train` owns the metric state and keeps it in the version-10 in-memory and
formatted checkpoints.  A resumed run therefore reproduces the uninterrupted
learning-rate trajectory, including reductions that occur at an epoch boundary.
The returned `mlp_training_state_t` exposes the current best metric, bad
observation count, and reduction count for diagnostics and manifests.
Malformed or incomplete plateau state is rejected transactionally.  The
independent `test_mlp_plateau_schedule` fixture checks the recurrence,
continuous-factor finite difference, trainer integration, checkpoint
round-trip, split/resumed equality, and malformed schedule refusal.

Plateau state is host-owned today.  CUDA schedule execution returns the
existing typed unavailable capability; there is no hidden host fallback in a
resident-device claim.  A resident metric reduction and optimizer-state plan
must land with its own CPU oracle and transfer counters before this boundary is
changed.
