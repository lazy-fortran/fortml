# Mini-batch Adam hypergradients

`fortml_mlp_minibatch_adam_hypergradient` provides a deterministic, CPU
reference objective for tuning the learning rate, coupled L2 coefficient,
both moment coefficients, and epsilon through a fixed mini-batch Adam
trajectory. The batch order is recorded at initialization. The outer objective
is validation MSE after the final update.

The packed search coordinates are

```text
[ log(learning_rate), log(l2), logit(beta1), logit(beta2), log(epsilon) ]
```

Every evaluation replays the same private batch cursor. The implementation
propagates forward tangents through the MLP parameters, coupled-L2 gradients,
first and second moments, beta-dependent bias correction, stabilized
square-root denominator, and parameter update. `value_gradient`, `jvp`, and
scalar `vjp` are analytic. The FortOpt callback and
`mlp_optimize_minibatch_adam_hyperparameters` use those products with explicit
log and logit bounds.

```fortran
use fortml_mlp_minibatch_adam_hypergradient

type(mlp_minibatch_adam_hypergradient_options_t) :: options
type(mlp_minibatch_adam_hypergradient_objective_t) :: objective
real(dp) :: p(5), g(5), value

options%epochs = 3
options%batch_size = 32
options%shuffle = .true.
options%shuffle_seed = 43
options%learning_rate = 8.0e-2_dp
options%l2 = 3.5e-2_dp
options%beta1 = 0.84_dp
options%beta2 = 0.93_dp
options%epsilon = 2.5e-2_dp
call objective%initialize(model, train_x, train_y, valid_x, valid_y, options, status)
call objective%value_gradient(objective%parameters(), value, g, status)
```

The product refuses a zero bias-corrected second moment because the square-root
branch has no derivative there. The trajectory is CPU-only. A selected CUDA
context returns `FORTNUM_NOT_IMPLEMENTED` without an implicit host copy. The
outer hyper-HVP also returns a typed refusal because it requires third network
derivatives.

`test_mlp_minibatch_adam_hypergradient` replays the shuffled affine trajectory
in code independent of the production recurrence, then checks all five
components and a directional product. It also checks a nonlinear MLP against
central differences. The companion release workload is
`fortml_bench_mlp_minibatch_adam_hypergradient`.
