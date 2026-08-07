# fortml

`fortml` is a machine-learning library for modern Fortran. It provides dense
regression, binary classification, neural models, exact and approximate
Gaussian processes, lazy linear operators, and optimizer-facing derivative
products. Numerical kernels
come from `fortnum`, optimization from `fortopt`, source-generated derivatives
from `fortad`, and sparse storage from `fortsparse`.

The current public surface concentrates on regression, binary and multinomial
classification, Laplace binary GP classification, shared classification
metrics, differentiable preprocessing and basis pipelines, a deterministic MLP
trainer and classifier, exact second-order depth-one XGBoost-style boosting,
neural and variational model primitives, Gaussian processes, and structured
linear algebra. Full estimator-family parity, deeper/histogram tree growth,
checkpoint APIs, model serialization, and distributed execution remain parity
work packages. [ROADMAP.md](ROADMAP.md)
records their acceptance criteria and delivery order.

The library uses separate Fortran modules instead of an umbrella `fortml`
module. For example, exact GP regression uses `fortml_kernels` and
`fortml_gaussian_process`.

## Build and test

The development manifest resolves four sibling checkouts:

```text
parent/
  fortad/
  fortml/
  fortnum/
  fortopt/
  fortsparse/
```

From the `fortml` directory, run the complete static, build, and test lane with
`fo`:

```sh
fo
```

The direct fpm lane is:

```sh
fpm test
```

The default `fo` profile enables runtime checks and no optimization. Use an
explicit release build for timings:

```sh
fo build --flag "-O3 -funroll-loops"
```

NVIDIA builds select `nvfortran` through `FO_FC`. The accelerator-specific
drivers and flags are documented in [benchmark/README.md](benchmark/README.md).

## Array and status conventions

For matrix-oriented models, samples occupy rows, features occupy columns, and
outputs occupy columns. A batch of `n` samples with `d` features and `p`
outputs therefore uses `x(n,d)` and `y(n,p)`. Recurrent arrays use
`(time,batch,feature)` order.

Procedures that return a `fortnum_status_t` report success through
`status_ok(status)`. Constructors and fits can fail without producing a usable
object, so check the status before the next call. Sparse operator procedures
use the corresponding `fortsparse_status_t`. Krylov solves return integer
`info` values from `fortnum_krylov`.

## Exact GP example

This program fits two output columns with one shared RBF covariance. Predictive
variance is the latent-function variance and does not include observation
noise.

```fortran
program exact_gp_example
    use, intrinsic :: iso_fortran_env, only: dp => real64
    use fortml_gaussian_process, only: gp_regression_t
    use fortml_kernels, only: kernel_t, make_rbf_kernel
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    real(dp) :: x(4, 1), y(4, 2), query(2, 1)
    real(dp) :: mean(2, 2), variance(2)
    type(kernel_t) :: kernel
    type(gp_regression_t) :: model
    type(fortnum_status_t) :: status

    x(:, 1) = [-1.0_dp, -0.25_dp, 0.5_dp, 1.25_dp]
    y(:, 1) = x(:, 1)**2
    y(:, 2) = sin(x(:, 1))
    query(:, 1) = [0.0_dp, 1.0_dp]

    kernel = make_rbf_kernel(1, 1.0_dp, 0.8_dp, status)
    if (.not. status_ok(status)) error stop "kernel construction failed"
    call model%fit(x, y, kernel, 1.0e-4_dp, status)
    if (.not. status_ok(status)) error stop "GP fit failed"
    call model%predict(query, mean, variance, status)
    if (.not. status_ok(status)) error stop "GP prediction failed"
end program exact_gp_example
```

## Implemented surface

| Area | Public modules | Main limits |
| --- | --- | --- |
| Regression and features | `fortml_linear_regression`, `fortml_preprocessing`, `fortml_basis`, `fortml_pipeline` | Dense SVD fit plus fitted standard/min-max scalers with exact input JVPs. Pipelines currently form horizontal unions of fixed basis stages. |
| Classification | `fortml_logistic_regression`, `fortml_softmax_regression`, `fortml_mlp_classifier`, `fortml_gp_classification`, `fortml_gp_multiclass_classification`, `fortml_classification_metrics`, `fortml_losses` | Linear, neural, binary Laplace GP, and one-vs-rest multiclass GP classifiers use stable probabilities and arbitrary integer labels. Linear fits accept nonnegative sample weights. Shared accuracy, balanced accuracy, confusion, precision/recall/F1, weighted accuracy, and log-loss metrics are implemented. Class weights and variational categorical GP likelihoods remain planned. |
| Neural models | `fortml_mlp`, `fortml_mlp_training`, `fortml_mlp_classifier`, `fortml_bnn`, `fortml_vae`, `fortml_rnn` | MLPs use deterministic Xavier/He initialization, Adam training, exact MSE+L2 parameter/hyperparameter HVPs, and a multiclass logits classifier. The recurrent model is one vanilla `tanh` RNN with a zero initial state |
| Trees and boosting | `fortml_tree`, `fortml_xgboost` | Deterministic regression stumps, squared-loss residual boosting, and exact depth-one second-order squared/logistic boosting with L1/L2/gamma/min-child-Hessian regularization. Deeper/histogram/CART/missing-value/ranking/constraint policies remain planned. |
| Variational inference | `fortml_variational`, `fortml_sparse_gp` | `sparse_gp_t` has scalar targets and caller-supplied variational parameters |
| Exact GPs | `fortml_kernels`, `fortml_gaussian_process`, `fortml_gp_training`, `fortml_derivative_gaussian_process`, `fortml_multi_output_gp` | Exact-GP hyperparameters can be optimized with FortOpt L-BFGS-B. Derivative observations cover function values and first input derivatives |
| Approximate GPs | `fortml_sparse_prior_gp`, `fortml_local_experts`, `fortml_ski_gp` | Multidimensional SKI requires one isotropic RBF leaf. Local experts support contiguous or deterministic clustered partitions. |
| Lazy inference | `fortml_linear_operator`, `fortml_kernel_operator`, `fortml_sparse_operator`, `fortml_structured_operator`, `fortml_toeplitz_operator`, `fortml_banded_precision` | Toeplitz products are host-resident |
| Supporting contracts | `fortml_kernel_formula`, `fortml_lanczos`, `fortml_multilevel_grid`, `fortml_inference_policy`, `fortml_parameter_registry`, `fortml_parameter_products` | Product availability depends on the wrapped model |

Validated user kernel formulas are lowered into the same postfix program as
built-in kernels. The generic kernel operator can run that program through its
OpenACC path or native CUDA plan. Arbitrary basis callbacks remain host-only.

See [docs/EXAMPLES.md](docs/EXAMPLES.md) for executable examples and
[docs/API.md](docs/API.md) for the public module reference. The implementation
boundaries are recorded in [docs/DESIGN.md](docs/DESIGN.md) and
[docs/ML_ARCHITECTURE.md](docs/ML_ARCHITECTURE.md). Benchmark evidence and parity
work packages are maintained in [ROADMAP.md](ROADMAP.md). The package is
distributed under the [MIT license](LICENSE).
