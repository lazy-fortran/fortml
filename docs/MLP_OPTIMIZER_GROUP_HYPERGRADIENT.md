# MLP optimizer-group trajectory hypergradients

`fortml_mlp_optimizer_group_hypergradient` exposes the production trainer's
contiguous optimizer-group update rule to FortOpt hyperparameter search. The
fixed full-batch SGD trajectory (and the coupled-L2 Adam variant described in
[`MLP_ADAM_OPTIMIZER_GROUP_HYPERGRADIENT.md`](MLP_ADAM_OPTIMIZER_GROUP_HYPERGRADIENT.md)) packs

```text
[ log(learning_rate), log(l2), log(multiplier_1), ..., log(multiplier_g) ]
```

For a nonconstant stateless schedule, two schedule coordinates are inserted
before the group multipliers. Cosine and warm-up/cosine schedules use
`[logit(min_rate_fraction), logit(decay_factor)]`; one-cycle uses
`[log(peak_rate_fraction), log(final_rate_fraction)]`. Inactive fields have
exact zero products. The constant-schedule ABI is unchanged, while
`metadata%first_log_multiplier_index` and
`metadata%schedule_parameter_count` describe an extended layout. Schedule
kind and integer warm-up/total-update counts are fixed metadata.

Group names and ranges are discrete metadata captured at initialization.
Every update uses the same post-optimizer scaling as `mlp_train`. Parameters in
group `i` receive `multiplier_i` times the shared SGD delta, while uncovered
parameters retain multiplier one. The Adam variant applies the same group
scales after bias-corrected moment updates. The outer coordinates are
differentiable through the MLP analytic HVP, the learning rate, L2, and every
group multiplier. `value_gradient`, `jvp`, `vjp`, and the FortOpt bounded
L-BFGS-B adapter share one deterministic objective. No finite-difference
optimizer fallback is used. The public `hvp` entry point also has an exact
production slice: affine MLPs, constant-rate full-batch SGD, no clipping, and
fixed group ranges propagate a second trajectory tangent analytically. The
affine training Hessian is constant, so this path uses the existing loss HVP
and includes the mixed log-L2 term without third derivatives. Adam,
nonconstant schedules, clipping, and nonlinear MLPs return a zero product with
`FORTNUM_NOT_IMPLEMENTED` until their complete second-state derivative
contracts are available. Shape or non-finite inputs return
`FORTNUM_DOMAIN_ERROR`. No numerical hyper-HVP is hidden behind the API.

The adapter also accepts a fixed `gradient_clip_norm` and applies the same
global-norm clipping order as `mlp_train`, after the L2 term and before grouped
scaling. The packed derivatives propagate through the clipped branch for a
fixed active set. A trajectory that lands on the clipping boundary returns a
typed `FORTNUM_NOT_IMPLEMENTED` instead of assigning a false derivative. The
clip norm itself is not an outer coordinate. Momentum beyond Adam, other
Adam-family state, and minibatch cursors remain separate capability boundaries.
Cosine, linear
warm-up, warm-up/cosine, and one-cycle rates use the schedule's analytic
`rate_with_full_derivatives` path on CPU. Plateau schedules and resident CUDA
group state return typed
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
coordinates, overlap validation, and the typed CUDA/refusal boundaries.
`test_mlp_optimizer_group_affine_hvp` supplies an independent scalar affine
recurrence and checks the full outer HVP against its central difference, plus
the nonlinear typed refusal.
