# Production MLP Lion training

`mlp_train` supports Lion as a stateful optimizer through
`MLP_OPTIMIZER_LION`. The trainer uses the two coefficients already present in
`mlp_training_options_t`:

```fortran
options%optimizer = MLP_OPTIMIZER_LION
options%learning_rate = 5.0e-4_dp
options%beta1 = 0.9_dp
options%beta2 = 0.99_dp
options%weight_decay = 1.0e-2_dp
call mlp_train(model, train_x, train_target, status, options, state, &
    validation_x=valid_x, validation_target=valid_target, checkpoint=checkpoint)
```

For a gradient `g_k` and the stored momentum `m_k`, one update is

\[
u_k = \operatorname{sign}(\beta_1m_k+(1-\beta_1)g_k),\qquad
\theta_{k+1}=\theta_k-\eta(u_k+\lambda\theta_k),
\]

followed by

\[
m_{k+1}=\beta_2m_k+(1-\beta_2)g_k.
\]

The sign of an exactly zero component is zero. `l2` remains the differentiable
loss penalty. `weight_decay` is the decoupled Lion term. Norm clipping is
applied to `g_k` before the Lion interpolation. Typed learning-rate schedules,
optimizer groups, EMA, validation patience, callbacks, and in-memory or text
checkpoints use the same state machine as the other MLP optimizers.

The checkpoint stores the single Lion momentum vector in the existing
`first_moment` record, together with beta values, step metadata, iterator
state, EMA, validation history, and the best model. A resumed run requires the
same data shape and optimizer controls. It follows the uninterrupted trajectory
when `max_epochs` is increased to the final target.

`test_mlp_lion_training` computes the linear-model gradient and Lion recurrence
independently, then checks parameters, EMA, momentum, and a split checkpoint
resume against the trainer. The release benchmark reports full-batch training,
EMA and checkpoint replay as separate rows. The current implementation uses
the FP64 CPU trainer. Resident CUDA Lion state and differentiable Lion
sign-branch products remain typed follow-up capabilities. The existing fixed
trajectory Lion hypergradient also refuses a sign-margin crossing instead of
silently differentiating the discontinuity.
