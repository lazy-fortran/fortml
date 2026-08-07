# MLP SGD momentum trajectory hypergradients

`fortml_mlp_sgd_momentum_hypergradient` exposes an exact fixed full-batch
trajectory objective for classical SGD momentum and Nesterov acceleration.  A
packed outer vector is

```text
[ log(learning_rate), log(l2), momentum ]
```

The model parameters are initialized from the state at
`objective%initialize`.  Each value evaluation starts from that same state,
performs exactly `options%steps` full-batch MSE + L2 updates, and returns the
held-out validation MSE after the final update.  Momentum is constrained to
`[0, 1)` and Nesterov mode is selected as a fixed discrete option; Nesterov
requires a positive lower momentum bound.

The recurrence is the one used by FortOpt's `sgd_t`:

```text
v_next = momentum*v + gradient(theta)
direction = v_next                         (classical)
direction = gradient(theta) + momentum*v_next  (Nesterov)
theta_next = theta - learning_rate*direction
```

The implementation propagates the model's analytic Hessian-vector product
through the parameter and velocity states.  `value_gradient`, `jvp`, and
scalar `vjp` therefore share one exact derivative path; no finite-difference
or optimizer fallback is hidden in the API.  `fortopt` and
`mlp_optimize_sgd_momentum_hyperparameters` provide bounded L-BFGS-B search
over the same packed objective.  CUDA, mini-batch, schedule, clipping, and
stochastic trajectories return `FORTNUM_NOT_IMPLEMENTED` until their resident
state derivatives are implemented.

Minimal use:

```fortran
use fortml_mlp_sgd_momentum_hypergradient, only: &
    mlp_sgd_momentum_hypergradient_objective_t, &
    mlp_sgd_momentum_hypergradient_options_t

type(mlp_sgd_momentum_hypergradient_objective_t) :: objective
type(mlp_sgd_momentum_hypergradient_options_t) :: options
real(dp) :: p(3), value, gradient(3)

options%steps = 8
options%momentum = 0.9_dp
call objective%initialize(model, train_x, train_target, validation_x, &
    validation_target, options, status)
p = objective%parameters()
call objective%value_gradient(p, value, gradient, status)
```

Independent finite-difference, JVP, VJP, Nesterov, FortOpt, and typed refusal
oracles are in `test/test_mlp_sgd_momentum_hypergradient.f90`.
