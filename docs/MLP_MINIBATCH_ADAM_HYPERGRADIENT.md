# Mini-batch Adam hypergradients

`fortml_mlp_minibatch_adam_hypergradient` provides a deterministic, CPU
reference objective for tuning the learning rate and coupled L2 coefficient
through a fixed mini-batch Adam trajectory. It is deliberately bounded: the
batch order is recorded at initialization, beta1/beta2/epsilon are fixed, and
the outer objective is validation MSE after the final update.

The packed search coordinates are

```text
[ log(learning_rate), log(l2) ]
```

Every evaluation replays the same private batch cursor. The implementation
propagates forward tangents through the MLP parameters, coupled-L2 gradients,
first and second moments, bias correction, stabilized square-root denominator,
and parameter update. Consequently `value_gradient`, `jvp`, and scalar `vjp`
are analytic products rather than finite differences. The FortOpt callback and
`mlp_optimize_minibatch_adam_hyperparameters` use those same products with
explicit log-coordinate bounds.

```fortran
use fortml_mlp_minibatch_adam_hypergradient

type(mlp_minibatch_adam_hypergradient_options_t) :: options
type(mlp_minibatch_adam_hypergradient_objective_t) :: objective
real(dp) :: p(2), g(2), value

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

The trajectory is CPU-only. A selected CUDA context returns
`FORTNUM_NOT_IMPLEMENTED` without mutating the model; there is no implicit host
copy. The outer hyper-HVP also returns a typed refusal because it would require
third network derivatives. This boundary is intentional until a resident Adam
state and a complete third-derivative contract are available.

The independent behavioral oracle is
`test_mlp_minibatch_adam_hypergradient`. The release workload is
`fortml_bench_mlp_minibatch_adam_hypergradient` in the companion benchmark
repository.
