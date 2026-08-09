# Weighted MLP training

`mlp_train` and `mlp_regressor_t%fit` accept optional row weights for the MLP
MSE objective. A weight vector has one entry per input row. Entries must be
finite and non-negative, and the total mass must be positive. A zero-weight
row remains in the deterministic batch order and contributes no loss or
gradient.

```fortran
use fortml_mlp_training, only: mlp_train, mlp_training_options_t

real(dp) :: sample_weight(n_train), validation_weight(n_validation)
type(mlp_training_options_t) :: options

call mlp_train(model, train_x, train_target, status, options, state, &
    validation_x=validation_x, validation_target=validation_target, &
    sample_weight=sample_weight, validation_weight=validation_weight)
```

The mean data term is

\[
 L_{data} = \frac{1}{\sum_i w_i}\sum_i w_i\,\frac12\|f(x_i)-y_i\|_2^2,
\]

and the regularized objective is

\[
 L = L_{data} + \frac{\lambda}{2}\|\theta\|_2^2.
\]

The parameter gradient is the exact weighted mean residual VJP plus
`lambda*theta`. The L2 term is added once at the optimizer boundary, so its
scale is independent of the number of minibatches. The loss and HVP products
use the same optional weights and positive-mass validation rule.

With `batch_size` and `accumulation_steps`, each microbatch contributes its
weighted-mean data gradient multiplied by its weight mass. The accumulated
gradient is divided by the accumulated mass before the optimizer update. This
gives the same gradient as one weighted full-batch evaluation, including an
uneven final batch and batches containing only zero-weight rows. Checkpoints
carry the accumulated mass when an active accumulation group is captured.

`validation_weight` is accepted only with `validation_x` and
`validation_target`. It controls validation loss, patience, best-state
selection, callbacks, and the final validation diagnostic. Training and
validation weight validation happens before the candidate model is installed,
so malformed or zero-support vectors leave the fitted object unchanged.

The optional arguments were appended to existing calls. Existing positional
calls without weights retain their previous behavior. The FortOpt L-BFGS-B
adapter used by `mlp_regressor_t` accepts `sample_weight` as well. The current
implementation is the CPU FP64 reference path. CUDA MLP training remains an
explicit typed boundary until resident network, loss, and optimizer state are
available.

`test_mlp_weighted_training` compares the value, gradient, L2 derivative, and
one-step minibatch recurrence with an independent affine hand calculation. It
also checks all current CPU optimizer dispatches, transactional weight errors,
and weighted validation through the public regressor facade. The release app
is `app/fortml_bench_mlp_weighted_training.f90`.
