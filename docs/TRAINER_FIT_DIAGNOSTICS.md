# Generic trainer fit diagnostics

`fortml_trainer::trainer_state_t` now records the outcome of the most recent
successful `fit` call alongside the parameter and optimizer state. The fields
are intentionally small, deterministic, and serializable:

| field | meaning |
| --- | --- |
| `fit_calls` | Number of successful `trainer_t%fit` calls since initialization or checkpoint load. |
| `optimizer_iterations` | Accepted streaming updates, or FortOpt L-BFGS-B convergence iterations, from the latest fit. |
| `line_search_evaluations` | FortOpt L-BFGS-B trial evaluations from the latest bounded fit; zero for streaming optimizers. |
| `curvature_updates` | Accepted L-BFGS-B `(s,y)` history updates from the latest bounded fit; zero for streaming optimizers. |

The fields are diagnostics, not a second optimizer state. `partial_fit` does
not increment `fit_calls`; its moments, schedules, and histories remain the
resumable state. A failed `fit` leaves the previous diagnostic snapshot in
place. The bounded L-BFGS-B path still has no resumable quasi-Newton history,
so `save_checkpoint` returns a typed domain error for that optimizer.

The formatted trainer checkpoint is schema 7. Loading validates all four
integer counters as non-negative before replacing the destination, preserving
the existing transactional load contract. Unknown or older schemas are
rejected rather than silently discarding diagnostics.

The independent `test_trainer_fit_diagnostics` quadratic fixture checks the
FortOpt iteration, line-search, and curvature counters against the analytic
minimum, then checks schema-7 persistence of the streaming fit counters.
CUDA remains a typed refusal for the generic trainer because the objective and
optimizer state are host-owned.
