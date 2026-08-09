# Grouped coupled-L2 Adam trajectory products

`fortml_mlp_optimizer_group_hypergradient` can now replay the production
trainer's contiguous optimizer groups with either plain SGD or coupled-L2
Adam.  Set `options%optimizer = MLP_OPTIMIZER_GROUP_ADAM` and provide the
fixed Adam coefficients (`beta1`, `beta2`, and `epsilon`) to select the Adam
trajectory.  The packed outer coordinates remain

```text
[ log(learning_rate), log(l2), log(multiplier_1), ..., log(multiplier_g) ]
```

and schedule coordinates, when active, are inserted before the group
multipliers as described in
[`MLP_OPTIMIZER_GROUP_HYPERGRADIENT.md`](MLP_OPTIMIZER_GROUP_HYPERGRADIENT.md).
The Adam moment coefficients are explicit trajectory metadata rather than
packed coordinates in this ABI. Use `fortml_mlp_adam_hypergradient` when
optimizing `beta1` and `beta2` themselves.

Each update computes the regularized MSE gradient, updates Adam's first and
second moments with bias correction, and then applies the group's multiplier to
the complete Adam delta.  The value, gradient, JVP, VJP, and FortOpt callback
propagate exact tangents through the parameter and moment state. No numerical
optimizer fallback is used.  A zero second-moment square-root is a typed
`FORTNUM_NOT_IMPLEMENTED` boundary because its derivative is singular.
Gradient clipping and stateless learning-rate schedules retain the existing
fixed-active-set contracts.  Plateau schedules and all resident CUDA paths
remain typed refusals until their device state and derivative kernels are
available.

```fortran
use fortml_mlp_optimizer_group_hypergradient, only: &
    mlp_optimizer_group_hypergradient_options_t, &
    mlp_optimizer_group_hypergradient_objective_t, MLP_OPTIMIZER_GROUP_ADAM

options%optimizer = MLP_OPTIMIZER_GROUP_ADAM
options%beta1 = 0.9_dp
options%beta2 = 0.999_dp
options%epsilon = 1.0e-8_dp
call objective%initialize(model, train_x, train_target, validation_x, &
    validation_target, options, status)
call objective%value_gradient(objective%parameters(), value, gradient, status)
```

`test_mlp_adam_optimizer_group_hypergradient` is an independent behavioral
oracle: it implements the two-parameter linear Adam recurrence directly,
checks the validation value, and compares every packed derivative with a
central difference.  `test_mlp_optimizer_group_hypergradient` continues to
cover the SGD/schedule/clip and FortOpt contracts.
