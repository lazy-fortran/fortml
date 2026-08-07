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
metrics, differentiable preprocessing and basis pipelines with analytic HVPs,
a deterministic MLP trainer and classifier, deterministic dense k-nearest-neighbor
and random-forest classification,
weighted LDA/QDA discriminant analysis, exact depth-limited second-order
XGBoost-style boosting with squared/logistic/Poisson/Huber/quantile objectives
and staged diagnostics,
neural and variational model primitives, Gaussian processes, and structured
linear algebra. Full estimator-family parity, histogram tree growth, model
serialization, and distributed execution remain parity work packages. The MLP
trainer now has an in-memory resumable checkpoint API, AdamW, Adagrad, RMSprop, and
FortOpt-backed SGD, and binary logistic objectives have a bounded FortOpt
L-BFGS-B adapters for logistic objectives and binary/shared-kernel Laplace GP
classification. Adagrad's accumulated-square state is checkpointed and
resumed exactly. A fixed full-batch MLP trajectory also exposes exact learning-rate/L2
hypergradients; see [docs/MLP_HYPERGRADIENT.md](docs/MLP_HYPERGRADIENT.md).
[ROADMAP.md](ROADMAP.md) records their acceptance criteria and delivery order.

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
| Linear regression, ridge/elastic-net estimators, scalers, basis maps, and basis pipelines | Value, input/parameter JVPs and VJPs; analytic HVPs for polynomial/Fourier/radial/spline maps and horizontal/column/sequential pipelines | General DAG transforms, callback HVPs, and fit-time derivatives remain open |
| Poisson/Gamma GLM regression | Analytic weighted log-link objective/gradient, alpha/dispersion hypergradients, prediction, input/parameter JVPs and VJPs | Fit-time optimizer differentiation remains an explicit boundary; resident CUDA kernels remain planned |
| Dense MLP and MSE training objective | Parameter/input JVPs, VJPs, exact MSE+L2 HVPs, L2 hyperparameter derivative, Adam/AdamW/Adagrad/RMSprop/SGD momentum/Nesterov training, typed constant/warmup/cosine/exponential schedules with analytic rate products, exact fixed-trajectory learning-rate/L2, AdamW (including beta logits), and RMSprop hypergradients, and exact in-memory optimizer checkpoints | Mini-batch/schedule optimizer-trajectory hypergradients and neural module families are partial |
| Exact GP regression | Kernel-parameter products, input derivatives, prediction products, and differentiated-solve HVPs; RBF, Matérn 1/2 and 3/2, periodic, and rational-quadratic products use analytic leaves, with the RBF and Matérn products cross-checked against FortSym-generated forms | Approximate and matrix-free training products are partial; Matérn 5/2 HVPs retain the FortAD product |
| Derivative-observation GP | Mixed value/first-derivative observations, parameter products, and query-input JVP/VJP products. Validated user-formula kernels carry analytic value/gradient/Hessian products | Query-input products and the likelihood HVP use documented deterministic finite differences. Analytic third-order kernels and joint posterior covariance remain open |
| Trees, boosting, and classifiers | Piecewise JVPs where declared, with split-boundary refusals; weighted LDA/QDA exposes smooth Gaussian probability products | Classifier HVPs, smooth split derivatives, and resident classifier GPU kernels remain open |
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
| Regression and features | `fortml_linear_regression`, `fortml_ridge_regression`, `fortml_elastic_net_regression`, `fortml_linear_svr`, `fortml_glm_regression`, `fortml_basis_linear_regression`, `fortml_regression_metrics`, `fortml_preprocessing`, `fortml_simple_imputer`, `fortml_one_hot_encoder`, `fortml_basis`, `fortml_pipeline`, `fortml_column_pipeline`, `fortml_validation` | Dense SVD, weighted ridge and deterministic weighted elastic-net fits, weighted dense linear SVR with squared or ordinary epsilon-insensitive losses, bounded FortOpt Poisson/Gamma log-link GLMs, core weighted regression metrics, fitted mean/median/constant imputation, deterministic integer-category one-hot encoding with packed metadata and explicit unknown/missing policies, standard/min-max scalers with exact input JVPs, horizontal, sequential, and explicit column-selecting basis pipelines with named stage/feature/parameter metadata and routed offsets, and fitted basis-to-linear estimators with packed parameters and chained JVP/VJP products. Deterministic K-fold/stratified split iterators are public. |
| Unsupervised decomposition | `fortml_pca`, `fortml_linear_autoencoder` | Centered dense PCA uses a thin LAPACK SVD with deterministic component signs, rank selection, explained-variance metadata, optional sample-variance whitening, reconstruction, and fixed-state input JVP/VJP products. `linear_autoencoder_t` copies fitted PCA into a tied linear encoder/decoder, with exact encode/reconstruct JVPs and an explicit CPU-only device contract. Incremental, randomized, sparse, kernel PCA, ICA, and NMF remain roadmap items. |
| Classification | `fortml_logistic_regression`, `fortml_linear_svm_classifier`, `fortml_logistic_training`, `fortml_ovr_logistic_classifier`, `fortml_ovo_logistic_classifier`, `fortml_multilabel_logistic_classifier`, `fortml_ordinal_logistic_classifier`, `fortml_softmax_regression`, `fortml_gaussian_naive_bayes`, `fortml_bernoulli_naive_bayes`, `fortml_multinomial_naive_bayes`, `fortml_complement_naive_bayes`, `fortml_categorical_naive_bayes`, `fortml_probability_calibration`, `fortml_cart_classifier`, `fortml_random_forest_classifier`, `fortml_extra_trees_classifier`, `fortml_mlp_classifier`, `fortml_gp_classification`, `fortml_gp_multiclass_classification`, `fortml_classification_metrics`, `fortml_losses` | Linear logistic and squared/ordinary-hinge SVM, OVR, OVO, independent multilabel logistic, weighted ordinal cumulative-logit, GaussianNB, BernoulliNB, MultinomialNB, ComplementNB, CategoricalNB, CART, deterministic random forest and randomized-threshold Extra-Trees, neural, and binary calibration estimators support nonnegative sample weights and positive sorted-class weights where declared. MLP classifiers expose exact prediction JVP/VJP products; tree ensembles expose deterministic aligned probabilities and explicit CPU/CUDA refusals. Linear SVM accepts arbitrary binary integer labels, uses deterministic FortOpt L-BFGS-B, exposes packed affine JVP/VJP products, and returns explicit split-boundary and CUDA refusals. The ordinal head stores ordered integer labels, strictly increasing cut points, packed parameter products, and explicit CPU/CUDA capability boundaries. Multilabel heads return a dense positive-probability matrix, per-label thresholds, packed parameter products, and explicit CPU/CUDA capability boundaries. OVO uses an explicit normalized pairwise-vote policy and exposes pair/input/parameter derivative products. The logistic training objective adds exact packed gradients, mixed L2 HVPs, and bounded FortOpt L-BFGS-B. The OVR and Naive Bayes wrappers expose normalized probabilities, arbitrary integer labels, packed parameter metadata, and input/parameter JVP/VJP products where inputs are continuous. GaussianNB adds weighted Gaussian moments and variance smoothing; BernoulliNB adds relaxed-[0,1] features and positive smoothing; MultinomialNB adds relaxed nonnegative counts and token-mass smoothing; ComplementNB adds complement distributions, optional second weight normalization, and the same differentiable products. CategoricalNB stores sorted per-feature category metadata, weighted smoothed likelihoods, unknown-category error/ignore policies, and explicit discrete-JVP refusal. CART classification provides deterministic weighted Gini/entropy splits and piecewise-constant leaf probabilities. Binary and one-vs-rest Laplace GP classifiers use stable probabilities, input JVPs, and exact envelope gradients for their mode-log-posterior kernel objective. `probability_calibrator_t` implements Platt sigmoid and weighted PAVA isotonic calibration with score/parameter products and explicit knot active-set boundaries. Shared accuracy, top-k, balanced accuracy, confusion, precision/recall/F1, Brier, binary Matthews, weighted accuracy, log-loss, expected calibration error, and maximum calibration error metrics are implemented. Variational categorical GP likelihoods remain planned. |
| Neighbors | `fortml_knn_classifier`, `fortml_radius_neighbors_classifier` | Dense exact kNN and closed-radius classification with uniform or inverse-distance weighting, arbitrary integer labels, optional sample weights, deterministic ties, an in-training outlier-label policy, and a resident native-CUDA kNN training-set plan in CUDA builds. Both estimators explicitly refuse derivatives through discrete neighbor selection; KD/ball trees, sparse inputs, and differentiable soft-neighbor relaxation remain planned. |
| Neural models | `fortml_mlp`, `fortml_mlp_chain`, `fortml_mlp_training`, `fortml_mlp_grouped_training`, `fortml_mlp_schedules`, `fortml_mlp_hypergradient`, `fortml_mlp_schedule_hypergradient`, `fortml_mlp_classifier`, `fortml_hamiltonian_mlp`, `fortml_bnn`, `fortml_vae`, `fortml_rnn` | MLPs use deterministic Xavier/He initialization, linear/`tanh`/ReLU/GELU/SiLU/ELU/softplus/leaky-ReLU activations, Adam, AdamW with decoupled weight decay, Adagrad, RMSprop (centered or uncentered, optional momentum), or FortOpt SGD with momentum/Nesterov, explicit seeded batch cursors, sample-weighted microbatch accumulation, held-out validation with interval monitoring and best-state restoration, typed constant/warmup/cosine/exponential/one-cycle schedules with analytic rate products, norm clipping, resumable optimizer state, explicit mean/sum weighted MSE diagnostics, exact MSE+L2 parameter/hyperparameter HVPs, named non-overlapping log-L2 parameter groups with exact mixed HVPs, fixed full-batch trajectory hypergradients for SGD over log learning rate/L2, AdamW over log learning rate/L2/weight decay, RMSprop over log learning rate/L2/decay/log epsilon/momentum, and typed schedule trajectories over log/logit rate, L2, and schedule fields, each with JVP/VJP products and a FortOpt L-BFGS-B adapter. The shared loss facade adds MAE products with exact-kink refusals, stable focal BCE-with-logits value/JVP/VJP products, heteroscedastic Gaussian and Poisson/count NLL value/JVP/VJP/HVP products, exact BCE/logistic and softmax cross-entropy HVPs, weighted-MSE value/JVP/VJP/HVP products, and typed Huber-kink refusals. The named sequential MLP chain adds stable stage ranges, exact composed JVP/VJP/HVP products, and a bounded all-stage L-BFGS-B objective. Outer schedule HVPs return a typed third-derivative refusal; grouped objectives and CUDA trajectory requests return explicit refusals until resident derivative kernels exist. The separable Hamiltonian MLP exposes energy/vector-field products and a symplectic leapfrog map. The recurrent model is one vanilla `tanh` RNN with a zero initial state |
| Trees and boosting | `fortml_tree`, `fortml_cart_classifier`, `fortml_random_forest_classifier`, `fortml_extra_trees_classifier`, `fortml_xgboost`, `fortml_xgboost_multiclass` | Deterministic finite-only regression stumps, weighted exhaustive-split CART regression/classification, seeded bootstrap random-forest and randomized-threshold Extra-Trees probabilities, squared-loss residual boosting, exact and bounded-histogram depth-limited second-order squared/logistic boosting with recursive Newton leaves and L1/L2/gamma/min-child-Hessian regularization, deterministic NaN default routing (`error`, learned, forced-left, or forced-right), binary and one-vs-rest multiclass staged predictions, decision margins, gain/weight/cover feature importance, per-feature monotonic constraints (`-1/0/+1`) with recursive leaf bounds, and explicit CPU/CUDA device dispatch/refusal. Ranking, categorical, and interaction policies remain planned. |
| Variational inference | `fortml_variational`, `fortml_sparse_gp` | `sparse_gp_t` has scalar targets and caller-supplied variational parameters |
| Exact GPs | `fortml_kernels`, `fortml_gaussian_process`, `fortml_gp_training`, `fortml_derivative_gaussian_process`, `fortml_derivative_gp_training`, `fortml_multi_output_gp` | Exact-GP hyperparameters and mixed value/first-derivative GP states expose bounded FortOpt adapters. Derivative-GP prediction parameter and exact third-input query JVP/VJP products cover fixed value/first-derivative queries for the supported smooth kernels. Joint posterior covariance and broader derivative-kernel coverage remain roadmap work. |
| Approximate GPs | `fortml_sparse_prior_gp`, `fortml_local_experts`, `fortml_ski_gp` | Multidimensional SKI requires one isotropic RBF leaf. Local experts support contiguous or deterministic clustered partitions. |
| Lazy inference | `fortml_linear_operator`, `fortml_kernel_operator`, `fortml_sparse_operator`, `fortml_structured_operator`, `fortml_toeplitz_operator`, `fortml_banded_precision` | Toeplitz products are host-resident |
| Supporting contracts | `fortml_kernel_formula`, `fortml_lanczos`, `fortml_multilevel_grid`, `fortml_inference_policy`, `fortml_parameter_registry`, `fortml_parameter_products`, `fortml_hyperparameter_search`, `fortml_device`, `fortml_cuda_metrics` | Product availability depends on the wrapped model. `fortml_hyperparameter_search` provides deterministic grid/random enumeration, bounded single-start and seeded multistart FortOpt L-BFGS-B over shared analytic objectives. `fortml_device` records explicit CPU/CUDA capability, residency ownership, and transfer events. `fortml_cuda_metrics` adds a transfer-inclusive native-CUDA weighted MSE reduction with an unavailable stub and no host fallback. Neither module claims complete GPU execution. |

The newly added elastic-net, OVO, multilabel logistic, Laplace-GP
classification, probability calibration, typed schedule, and derivative-GP
APIs expose explicit CPU/CUDA capability/refusal methods. They are CPU-only
until resident model kernels exist: selected CUDA contexts return
`FORTNUM_NOT_IMPLEMENTED`, and no benchmark may report GPU timing for these
paths. The transfer-inclusive weighted-MSE reduction and resident optimizer,
kNN, and kernel micro-kernels documented in
[`docs/DEVICE.md`](docs/DEVICE.md) remain separate no-autodiff building blocks.

Validated user kernel formulas are lowered into the same postfix program as
built-in kernels. The generic kernel operator can run that program through its
OpenACC path or native CUDA plan. Arbitrary basis callbacks remain host-only.

See [docs/EXAMPLES.md](docs/EXAMPLES.md) for executable examples and
[docs/API.md](docs/API.md) for the public module reference. The implementation
boundaries are recorded in [docs/DESIGN.md](docs/DESIGN.md) and
[docs/ML_ARCHITECTURE.md](docs/ML_ARCHITECTURE.md). Benchmark evidence and parity
work packages are maintained in [ROADMAP.md](ROADMAP.md). The package is
distributed under the [MIT license](LICENSE).
