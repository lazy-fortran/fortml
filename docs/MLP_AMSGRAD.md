# AMSGrad MLP training

`mlp_train` supports `MLP_OPTIMIZER_AMSGRAD` as a deterministic CPU trainer
optimizer. It shares the MLP trainer's Adam coordinates (`learning_rate`,
`beta1`, `beta2`, and `epsilon`) and updates the flat parameter vector with

```text
m_t = beta1*m_(t-1) + (1-beta1)*g_t
v_t = beta2*v_(t-1) + (1-beta2)*g_t^2
vhat_t = max(vhat_(t-1), v_t)
theta_t = theta_(t-1) - rate*(m_t/(1-beta1^t)) /
          (sqrt(vhat_t/(1-beta2^t)) + epsilon)
```

The elementwise maximum is kept in `mlp_training_checkpoint_t%max_second_moment`.
It is validated for shape and finite values, included in the version-7
in-memory checkpoint, and serialized by schema 5. Loading a malformed or
incompatible snapshot returns `FORTNUM_DOMAIN_ERROR` before replacing the
caller-owned checkpoint. Split/resume therefore reproduces an uninterrupted
AMSGrad trajectory.

AMSGrad has no resident CUDA implementation yet. The trainer is CPU-only and
does not emulate a device path through host copies; a future device adapter
must provide a resident state recurrence and an independent CPU oracle before
claiming CUDA support. Fixed-trajectory AMSGrad hypergradients are also open:
the `max` active-set boundaries require a declared smooth branch or a typed
refusal rather than hidden finite differences.

`test_mlp_amsgrad` compares the production trajectory with an independent
recurrence, checks in-memory and formatted checkpoint continuation, and
verifies the serialized maximum-second-moment state.
