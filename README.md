# fortml

`fortml` is a machine-learning library for modern Fortran. It provides dense
regression, binary classification, neural models, exact and approximate
Gaussian processes, lazy linear operators, and optimizer-facing derivative
products. Numerical kernels
come from `fortnum`, optimization from `fortopt`, source-generated derivatives
from `fortad`, proof-backed symbolic kernel derivations from `fortsym`, and
sparse storage from `fortsparse`.

The current public surface concentrates on regression, binary and multinomial
classification, Laplace binary GP classification, shared classification
metrics, differentiable preprocessing and basis pipelines, a deterministic MLP
trainer and classifier, exact depth-limited second-order XGBoost-style boosting,
neural and variational model primitives, Gaussian processes, and structured
linear algebra. Full estimator-family parity, histogram tree growth, model
serialization, and distributed execution remain parity work packages. The MLP
trainer now has an in-memory resumable checkpoint API, and binary logistic
objectives have a bounded FortOpt L-BFGS-B adapter. [ROADMAP.md](ROADMAP.md)
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

## Derivative and initialization status

Derivative products are capability-specific. An absent product is refused or
listed in [ROADMAP.md](ROADMAP.md). It is never inferred to be zero. The
current surface is:

| Surface | Products | Boundary |
| --- | --- | --- |
| Linear regression, scalers, basis maps, and basis pipelines | Value, input/parameter JVPs and VJPs where the public API declares them | General DAG transforms and second-order products remain open |
| Dense MLP and MSE training objective | Parameter/input JVPs, VJPs, exact MSE+L2 HVPs, and the L2 hyperparameter derivative | Other losses, optimizers, and neural module families are partial |
| Exact GP regression | Kernel-parameter products, input derivatives, prediction products, and differentiated-solve HVPs | Approximate and matrix-free training products are partial |
| Derivative-observation GP | Mixed value/first-derivative observations, parameter products, and query-input JVP/VJP products. Validated user-formula kernels carry analytic value/gradient/Hessian products | Query-input products and the likelihood HVP use documented deterministic finite differences. Analytic third-order kernels and joint posterior covariance remain open |
| Trees, boosting, and classifiers | Piecewise JVPs where declared, with split-boundary refusals | Classifier HVPs and smooth split derivatives remain open |
| BNN, VAE, RNN, and most approximate GP paths | Value or model-specific gradient surfaces | Full JVP/VJP/HVP coverage is a roadmap item |

`fortsym` is used when it proves a smaller or more stable closed form for a
kernel or derivative leaf. `fortad` remains the general source-transformation
baseline. A generated product enters the implementation only after an
independent numerical oracle and a symbolic identity check agree. Accepted
generated artifacts must record the FortSym and FortAD revisions, operation
count, source hash, and any fallback or refusal. The release benchmarks compare
the two routes where both are available.

The same rule applies to model initialization. Xavier/He, PCA, NNGP/NTK, and
GP-posterior starts are separate contracts with recorded seeds, design sets,
and tolerances. A finite network is not declared identical to its infinite-
width GP. Physics-informed, Hamiltonian, symplectic, and autoencoder starts
must report the residual, invariant defect, reconstruction, or covariance
error at initialization.

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
| Regression and features | `fortml_linear_regression`, `fortml_basis_linear_regression`, `fortml_regression_metrics`, `fortml_preprocessing`, `fortml_basis`, `fortml_pipeline`, `fortml_column_pipeline`, `fortml_validation` | Dense SVD fit plus core weighted regression metrics, fitted standard/min-max scalers with exact input JVPs, horizontal, sequential, and explicit column-selecting basis pipelines, and a fitted basis-to-linear estimator with one packed parameter vector and chained JVP/VJP products. Deterministic K-fold/stratified split iterators are public. |
| Classification | `fortml_logistic_regression`, `fortml_logistic_training`, `fortml_ovr_logistic_classifier`, `fortml_softmax_regression`, `fortml_gaussian_naive_bayes`, `fortml_cart_classifier`, `fortml_mlp_classifier`, `fortml_gp_classification`, `fortml_gp_multiclass_classification`, `fortml_classification_metrics`, `fortml_losses` | Linear, OVR, GaussianNB, CART, and neural classifiers support nonnegative sample weights and positive sorted-class weights. The logistic training objective adds exact packed gradients, mixed L2 HVPs, and bounded FortOpt L-BFGS-B. The OVR logistic and GaussianNB wrappers expose normalized probabilities, arbitrary integer labels, packed parameter metadata, and input/parameter JVP/VJP products. GaussianNB adds weighted class moments, variance smoothing, explicit or empirical priors, and stable log-probability products. CART classification provides deterministic weighted Gini/entropy splits and piecewise-constant leaf probabilities. Binary Laplace GP and one-vs-rest multiclass GP classifiers use stable probabilities and arbitrary integer labels. Shared accuracy, top-k, balanced accuracy, confusion, precision/recall/F1, Brier, binary Matthews, weighted accuracy, log-loss, expected calibration error, and maximum calibration error metrics are implemented. Sigmoid/isotonic calibrator estimators and variational categorical GP likelihoods remain planned. |
| Neural models | `fortml_mlp`, `fortml_mlp_training`, `fortml_mlp_classifier`, `fortml_hamiltonian_mlp`, `fortml_bnn`, `fortml_vae`, `fortml_rnn` | MLPs use deterministic Xavier/He initialization, Adam training, explicit seeded batch cursors, sample-weighted microbatch accumulation, held-out validation with interval monitoring and best-state restoration, user schedules, norm clipping, explicit mean/sum weighted MSE diagnostics, exact MSE+L2 parameter/hyperparameter HVPs, and a multiclass logits classifier. The separable Hamiltonian MLP exposes energy/vector-field products and a symplectic leapfrog map. The recurrent model is one vanilla `tanh` RNN with a zero initial state |
| Trees and boosting | `fortml_tree`, `fortml_cart_classifier`, `fortml_xgboost`, `fortml_xgboost_multiclass` | Deterministic finite-only regression stumps, a weighted exhaustive-split depth-limited CART regressor with zero-away-from-boundary JVPs, weighted Gini/entropy CART classification with leaf probabilities, squared-loss residual boosting, exact depth-limited second-order squared/logistic boosting with recursive Newton leaves and L1/L2/gamma/min-child-Hessian regularization, and one-vs-rest multiclass probabilities with quotient-rule JVPs. Histograms, missing-value/ranking/constraint policies remain planned. |
| Variational inference | `fortml_variational`, `fortml_sparse_gp` | `sparse_gp_t` has scalar targets and caller-supplied variational parameters |
| Exact GPs | `fortml_kernels`, `fortml_gaussian_process`, `fortml_gp_training`, `fortml_derivative_gaussian_process`, `fortml_derivative_gp_training`, `fortml_multi_output_gp` | Exact-GP hyperparameters and mixed value/first-derivative GP states expose bounded FortOpt adapters. Derivative-GP prediction parameter and deterministic finite-difference query-input JVP/VJP products cover fixed value/first-derivative queries. Analytic third-order kernel products and joint posterior covariance remain roadmap work. |
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
