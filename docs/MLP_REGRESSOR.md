# MLP regressor

`fortml_mlp_regressor` is the estimator-facing facade for dense multi-output
MLP regression. It keeps row-oriented samples and columns as outputs, owns
the fitted topology and trainer/checkpoint state, and delegates all products
to the same `mlp_t` and `fortml_mlp_training` kernels used by lower-level
objectives.

```fortran
type(mlp_regressor_t) :: model
type(mlp_regressor_options_t) :: options
type(fortnum_status_t) :: status
allocate(options%layer_sizes(3))
options%layer_sizes = [n_features, 16, n_outputs]
options%training%max_epochs = 200
options%training%learning_rate = 1.0e-3_dp
call model%fit(x, target, status, options, validation_x=xv, validation_target=yv)
call model%predict(xq, mean, status)
```

`options%use_lbfgsb=.true.` selects the exact full-batch MSE+L2 objective and
FortOpt L-BFGS-B bounds in `options%lbfgsb`; the optimizer consumes the
analytic network gradient and HVP-compatible loss seam. The default path uses
the deterministic Adam-family trainer with validation, patience, best-state
restoration, and optional in-memory checkpoints.

`predict_jvp` and `predict_vjp` differentiate the fixed fitted network with
respect to packed parameters and inputs. `loss_gradient` and `loss_hvp` expose
the MSE/L2 parameter and scalar-L2 hyperparameter products, including optional
row weights. The fit transaction commits the candidate only after the selected
training path succeeds. CPU is the reference device; CUDA returns a typed
`FORTNUM_NOT_IMPLEMENTED` refusal until resident optimizer and model state are
available. The independent release evidence is
`fortml-bench/results/MLP_REGRESSOR.md`.
