# MLP optimizer-group trajectory hypergradients

`fortml_mlp_optimizer_group_hypergradient` exposes the production trainer's
contiguous optimizer-group update rule to FortOpt hyperparameter search. The
fixed full-batch SGD trajectory packs

```text
[ log(learning_rate), log(l2), log(multiplier_1), ..., log(multiplier_g) ]
```

Group names and ranges are discrete metadata captured at initialization.
Every update uses the same post-optimizer scaling as `mlp_train`. Parameters in
group `i` receive `multiplier_i` times the shared SGD delta, while uncovered
parameters retain multiplier one. The outer coordinates are differentiable
through the MLP analytic HVP, the learning rate, L2, and every group
multiplier. `value_gradient`, `jvp`, `vjp`, and the FortOpt bounded L-BFGS-B
adapter share one deterministic objective. No finite-difference optimizer
fallback is used. The public `hvp` entry point is equally explicit: it
returns a zero product with `FORTNUM_NOT_IMPLEMENTED` until third network
derivatives are available. Shape or non-finite inputs return
`FORTNUM_DOMAIN_ERROR`. No numerical hyper-HVP is hidden behind the API.

The adapter also accepts a fixed `gradient_clip_norm` and applies the same
global-norm clipping order as `mlp_train`, after the L2 term and before grouped
scaling. The packed derivatives propagate through the clipped branch for a
fixed active set. A trajectory that lands on the clipping boundary returns a
typed `FORTNUM_NOT_IMPLEMENTED` instead of assigning a false derivative. The
clip norm itself is not an outer coordinate. Momentum, Adam-family state,
schedules, minibatch cursors, and resident CUDA group state remain separate
capability boundaries. CUDA requests return typed
`FORTNUM_NOT_IMPLEMENTED`. Invalid or overlapping ranges are domain errors.

```fortran
use fortml_mlp_training, only: mlp_optimizer_group_t
use fortml_mlp_optimizer_group_hypergradient, only: &
    mlp_optimizer_group_hypergradient_options_t, &
    mlp_optimize_optimizer_group_hyperparameters

call weight_group%initialize("weights", 1, 4, 0.5_dp, status)
call bias_group%initialize("bias", 5, 5, 1.0_dp, status)
allocate(options%groups(2))
options%groups = [weight_group, bias_group]
call mlp_optimize_optimizer_group_hyperparameters(model, train_x, train_y, &
    validation_x, validation_y, options, result, status)
```

`test_mlp_optimizer_group_hypergradient` checks an independent central-
finite-difference oracle for every packed coordinate, JVP contraction, scalar
VJP scaling, exact parity with `mlp_train`'s group update, FortOpt result
coordinates, overlap validation, the typed outer-HVP refusal, and the typed
CUDA refusal.
