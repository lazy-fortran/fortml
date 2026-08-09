# MLP SGD momentum trajectory hypergradients

`fortml_mlp_sgd_momentum_hypergradient` exposes an exact fixed-trajectory
objective for classical SGD momentum and Nesterov acceleration.  A
packed outer vector is

```text
[ log(learning_rate), log(l2), momentum ]
```

The model parameters are initialized from the state at
`objective%initialize`.  Each value evaluation starts from that same state,
performs exactly `options%steps` MSE + L2 updates, and returns the held-out
validation MSE after the final update.  Set `microbatch_size` and
`accumulation_steps` to select a deterministic contiguous reduction; each
update covers every training row exactly once, weights a short final batch by
its row mass, and only then updates the velocity.  Momentum is constrained to
`[0, 1)` and Nesterov mode is selected as a fixed discrete option; Nesterov
requires a positive lower momentum bound.

`initialize` and `mlp_optimize_sgd_momentum_hyperparameters` accept an optional
`validation_weight(:)` vector.  It must have one finite non-negative entry per
validation row and positive total mass; the held-out objective then uses that
weighted mean.  The weights are copied into the objective, so subsequent caller
mutation cannot change an evaluation.  Value, JVP, VJP, and FortOpt products
use the same weighted measure.  The affine outer HVP remains certified for a
uniform measure; a non-uniform vector returns a typed
`FORTNUM_NOT_IMPLEMENTED` HVP status until residual-weighted second products
are generalized.

The recurrence is the one used by FortOpt's `sgd_t` after the accumulated
gradient has been formed:

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
over the same packed objective.  Schedules, clipping, stochastic loaders, and
CUDA-resident trajectories return `FORTNUM_NOT_IMPLEMENTED` until their
resident state derivatives are implemented. Invalid accumulation layouts are
rejected transactionally.

The `hvp(parameters, direction, product, status)` entry point additionally
provides an exact outer hyper-HVP for the one-layer affine branch (one dense
layer with linear output). That branch has a parameter-independent MSE
Hessian, so mixed second tangents through both classical and Nesterov velocity
states are analytic and are suitable for second-order FortOpt callers. A
nonlinear or multilayer model returns `FORTNUM_NOT_IMPLEMENTED`, preserving the
third-network-derivative boundary rather than finite-differencing an inner
trajectory.

Minimal use:

```fortran
use fortml_mlp_sgd_momentum_hypergradient, only: &
    mlp_sgd_momentum_hypergradient_objective_t, &
    mlp_sgd_momentum_hypergradient_options_t

type(mlp_sgd_momentum_hypergradient_objective_t) :: objective
type(mlp_sgd_momentum_hypergradient_options_t) :: options
real(dp) :: p(3), value, gradient(3)

options%steps = 8
options%microbatch_size = 32
options%accumulation_steps = 4
options%momentum = 0.9_dp
call objective%initialize(model, train_x, train_target, validation_x, &
    validation_target, options, status)
p = objective%parameters()
call objective%value_gradient(p, value, gradient, status)
```

Independent finite-difference, JVP, VJP, affine outer-HVP, Nesterov, FortOpt,
and typed refusal oracles are in
`test/test_mlp_sgd_momentum_hypergradient.f90`.
