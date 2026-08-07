# Deterministic mini-batch MLP hypergradients

`fortml_mlp_minibatch_hypergradient` differentiates a fixed mini-batch SGD
trajectory with respect to its learning rate and L2 coefficient. The packed
outer vector is

```text
[ log(learning_rate), log(l2) ]
```

The inner model starts from the parameters present at `initialize`. It runs a
fixed number of epochs, constructs an explicit batch cursor for every update,
and evaluates validation MSE after the final update. With `shuffle=.true.`,
each epoch uses a private Park-Miller stream seeded by `shuffle_seed`. Objective
evaluations reuse the recorded order, so L-BFGS-B sees a deterministic scalar
function.

The objective exposes `value_gradient`, scalar `jvp`, scalar `vjp`, and a
FortOpt context adapter. The trajectory derivative uses the MLP's analytic
loss HVP for each recorded batch. An outer HVP is refused with
`FORTNUM_NOT_IMPLEMENTED` because it would require third network derivatives.
CUDA requests are refused until the model, batches, optimizer state, and
products can remain resident on the device.

```fortran
use fortml_mlp_minibatch_hypergradient

type(mlp_minibatch_hypergradient_options_t) :: options
type(mlp_minibatch_hypergradient_result_t) :: result

options%epochs = 8
options%batch_size = 32
options%shuffle = .true.
options%shuffle_seed = 991
options%learning_rate = 2.0e-2_dp
options%l2 = 1.0e-4_dp
call mlp_optimize_minibatch_hyperparameters(model, train_x, train_y, &
    validation_x, validation_y, options, result, status)
```

This is a fixed-trajectory hyperparameter objective. It does not claim an
unbiased derivative of a random data-loader stream, early stopping, clipping,
or a resident GPU trajectory. Those contracts remain separate roadmap items.

The independent behavioral oracle is `test_mlp_minibatch_hypergradient`.
