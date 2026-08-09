# Global-norm clipping hypergradients

`fortml_mlp_clip_hypergradient` differentiates a fixed full-batch SGD
trajectory with respect to the learning rate, coupled L2 coefficient, and
global gradient-clip threshold. The packed outer vector is
`[log(learning_rate), log(l2), log(gradient_clip_norm)]`.

At each update, the objective computes the same MSE and coupled L2 gradient as
`mlp_train`. When the raw gradient norm exceeds the threshold, it applies

```text
clipped_gradient = gradient * clip_norm / norm(gradient)
```

and differentiates the norm, threshold, scale, and network state. The network
state product uses `mlp_loss_hvp`, so `value_gradient`, `jvp`, and `vjp` do not
use finite differences. An inactive threshold has an exact zero derivative.
The exact active-set boundary returns `FORTNUM_NOT_IMPLEMENTED` because the
clipping map has no unique derivative there.

`mlp_clip_hypergradient_objective_t%fortopt` exposes the same value and gradient
callback to FortOpt. `mlp_optimize_clip_hyperparameters` supplies independent
bounds for all three logarithmic coordinates and runs L-BFGS-B. The integer
step count stays fixed during one outer solve.

The current trajectory is CPU-only and uses full-batch SGD without momentum.
CUDA returns `FORTNUM_NOT_IMPLEMENTED`. Outer HVPs have the same typed status
until the MLP loss API provides the required third network derivatives.

`test_mlp_clip_hypergradient` compares the analytic products with central
differences through the separate production `mlp_train` path. It also checks
the inactive branch, clipping kink, scalar adjoint, L-BFGS-B result bounds, and
CUDA/HVP status contracts. `fortml_bench_mlp_clip_hypergradient` emits the
complete value, gradient, JVP, and HVP-status array used by the release
benchmark.
