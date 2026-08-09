# Multi-fidelity hyperparameter search

`fortml_hyperparameter_search` now provides a resource-aware search contract
for basis, model, and pipeline objectives. A
`hyperparameter_resource_objective_t` callback receives a parameter vector and
an integer resource level. It returns the objective value and the full
parameter gradient at that level.

`hyperparameter_successive_halving_search` samples a deterministic parameter
box from a caller-provided seed. It evaluates every candidate at the minimum
resource, sorts by objective value, and retains the best integer fraction at
each rung. The schedule reaches the requested maximum resource even when the
maximum is not an exact power of the reduction factor. The result reports rung
count, evaluation count, survivor count, best resource, and the best packed
parameter vector.

The callback can wrap a differentiable basis-pipeline objective. Its gradient
is the same product consumed by FortOpt. After pruning, call
`hyperparameter_lbfgsb_resource_search` with the surviving vector and the
maximum resource. The adapter routes every value and gradient request through
FortOpt L-BFGS-B and preserves box bounds.

The implementation is CPU-only. A selected CUDA device returns
`FORTNUM_NOT_IMPLEMENTED` before invoking the callback, so host execution is
never reported as resident GPU work. The independent release test
`test_hyperparameter_successive_halving` checks the analytic quadratic
resource objective, deterministic rung accounting, bound preservation,
L-BFGS-B convergence, invalid schedule refusal, and the CUDA status boundary.
