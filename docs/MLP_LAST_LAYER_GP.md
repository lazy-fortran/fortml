# Finite-feature GP/NTK last-layer initialization

`fortml_mlp_last_layer_gp` provides a deterministic closed-form warm start for
an existing `mlp_t`. It freezes all hidden layers, obtains the activation
immediately before the final dense layer, appends an intercept, and solves

\[
  (Z^T Z + \lambda I)C = Z^T Y,
\]

where `lambda` is the positive `regularization` hyperparameter. The first rows
of `C` initialize the final-layer weight and its last row initializes the bias.
`fit_apply` performs the fit and commits that final-layer replacement after
validation.

This is a finite-feature kernel-ridge posterior mean (equivalently, a
fixed-feature last-layer NTK approximation). It is not an exact NNGP, NTK, or
infinite-width GP equivalence. The metadata returned by `metadata()` records
`exact_infinite_width=.false.`, the sample/feature/output dimensions, and the
fixed-feature derivative scope so callers cannot mistake the warm start for a
full Bayesian or infinite-width calculation.

```fortran
use fortml_mlp, only: mlp_t, MLP_TANH, MLP_LINEAR
use fortml_mlp_last_layer_gp, only: mlp_last_layer_gp_initializer_t

type(mlp_t) :: model
type(mlp_last_layer_gp_initializer_t) :: initializer
type(fortnum_status_t) :: status

call model%initialize([n_input, n_hidden, n_output], status, &
    hidden_activation=MLP_TANH, output_activation=MLP_LINEAR)
call initializer%fit_apply(model, x, target, status, regularization=1.0e-3_dp)
call initializer%predict(model, x_query, mean, status)
```

The CPU `jvp` differentiates the posterior mean with respect to
`regularization` while holding the finite feature map and training targets
fixed. Its analytic solve is checked against a central finite difference in
`test_mlp_last_layer_gp`. `parameters()` and `parameter_metadata()` expose the
named positive regularization coordinate for FortOpt/search adapters.

The CUDA entry points intentionally return `FORTNUM_NOT_IMPLEMENTED` without
mutating model or output state: a CPU feature-map evaluation is never relabeled
as resident GPU execution. The companion benchmark reports the independent
NumPy normal-equation oracle, FortML fit/predict timing, approximation error,
and an explicit CUDA-unavailable row in
`fortml-bench/results/mlp_last_layer_gp.csv`.
