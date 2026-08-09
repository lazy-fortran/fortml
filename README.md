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
weighted LDA/QDA discriminant analysis, leakage-safe calibrated logistic
cross-validation, exact depth-limited second-order XGBoost-style boosting with
squared/logistic/Poisson/Gamma/Tweedie/Huber/quantile/absolute objectives
and staged diagnostics,
neural and variational model primitives, Gaussian processes, and structured
linear algebra. Full estimator-family parity, histogram tree growth, model
serialization, and distributed execution remain parity work packages. The MLP
trainer now has an in-memory resumable checkpoint API, AdamW, Adagrad, RMSprop,
AMSGrad, RAdam, and
FortOpt-backed SGD, and binary logistic objectives have a bounded FortOpt
L-BFGS-B adapter for logistic objectives and binary/shared-kernel Laplace GP
classification. Adagrad's accumulated-square state is checkpointed and
resumed exactly. The multiclass MLP classifier also exposes weighted softmax
cross-entropy value/JVP/VJP/HVP products and a bounded FortOpt L-BFGS-B
objective. Fixed full-batch MLP trajectories expose exact learning-rate/L2
hypergradients for SGD, Adam, AdamW, RMSprop, Adagrad, and unfactored Adafactor;
AMSGrad has validated CPU recurrence/checkpoint state but its max-active-set
trajectory derivatives remain open.
Adafactor's relative-step and parameter-scale branches carry exact smooth-state
products as well; clip, rate, and scale active-set boundaries return typed
refusals. See
[docs/MLP_HYPERGRADIENT.md](docs/MLP_HYPERGRADIENT.md).

The trainer also exposes an explicit `mlp_loss_scale_state_t` recurrence for
automatic mixed-precision policy testing: finite-update growth, overflow
backoff, skipped-update counters, and transactional schema-11 checkpoint
state. The CPU FP64 reference can exercise the recurrence without changing
the default trajectory; FP32/FP16/BF16 and resident CUDA training remain typed
refusals until master-weight and lower-precision kernels have independent
gates. See [docs/MLP_LOSS_SCALING.md](docs/MLP_LOSS_SCALING.md).

The production trainer also exposes layout-aware matrix-factored Adafactor
state through `fortml_adafactor_factored`. Matrix blocks use row and column
second-moment factors, vectors use the unfactored fallback, and checkpoint
migration plus resident CUDA execution remain typed refusals. The independent
lane is `fortml-bench/results/ADAFACTOR_FACTORED.md`.
[ROADMAP.md](ROADMAP.md) records their acceptance criteria and delivery order.
Fitted horizontal basis unions have a versioned, transactional host state
dictionary and text round-trip in
[`docs/PIPELINE_PERSISTENCE.md`](docs/PIPELINE_PERSISTENCE.md); resident CUDA
serialization remains an explicit typed refusal.

Deterministic K-fold, stratified, grouped, and chronological splitters can now
feed [`fortml_cross_validation`](docs/CROSS_VALIDATION.md), which aggregates
weighted fold scores and parameter gradients with explicit clone/reset leakage
guards. Differentiable scorers expose the same oriented objective to FortOpt's
grid, random, and bounded L-BFGS-B search drivers; the index/control plane
returns a typed CUDA refusal until resident estimator callbacks exist.

The GP surface also includes a latent-Gaussian ordinal classification baseline
and dense Student-t, heteroskedastic, and robust Poisson/Student-t process
regression references. The ordinal adapter exposes fixed-cut probability and
input/parameter products plus exact kernel/log-noise evidence gradients and
directional HVPs. Its bounded FortOpt L-BFGS-B companion optimizes those same
analytic products and reports/restores state transactionally; the Student-t process keeps
the GP mean but scales posterior variance by the observed Mahalanobis distance,
and the heteroskedastic process conditions on supplied input-dependent noise.
The robust Laplace adapter covers positive Poisson rates and bounded Student-t
outlier influence.
Both CPU contracts have independent tests and explicit typed CUDA boundaries;
native ordinal likelihood optimization and resident Student-t inference remain
roadmap work.

The exact-GP kernel catalog also includes the gated change-point kernel. Its
input, mixed, and packed parameter products are covered by an independent
NumPy lane, while static-operator and resident CUDA requests return typed
refusals.

The library uses separate Fortran modules instead of an umbrella `fortml`
module. For example, exact GP regression uses `fortml_kernels` and
`fortml_gaussian_process`.

Recent closure slices also expose query-coordinate JVP/VJP products for
Bernoulli variational GPs (logistic and probit), exact unfactored-Adafactor
trajectory hypergradients for FortOpt L-BFGS-B, fixed-tree XGBoost derivative
oracles and transactional fitted-prefix slicing. Sparse variational GPs now
also expose fixed-state kernel-log-parameter JVP/VJP products; MLP training
accepts validated contiguous optimizer groups with checkpointed metadata; and
classifier chains provide smooth probability-chain prediction plus exact
input/parameter products. These paths retain explicit
CPU/CUDA capability metadata and typed CUDA refusals until their complete
operation graphs are resident; no host fallback is presented as GPU support.

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
| Linear regression, ridge/elastic-net estimators, scalers, basis maps, and basis pipelines | Value, input/parameter JVPs and VJPs; analytic HVPs for polynomial/Chebyshev/Fourier/radial/spline maps and horizontal/column/sequential/fan-out/residual-sum/conditional interval-routed pipelines; versioned transactional state dictionaries for fitted horizontal unions | Cyclic DAG transforms, callback HVPs, fit-time derivatives, graph/estimator-wide and resident-device serialization remain open |
| Poisson/Gamma GLM regression | Analytic weighted log-link objective/gradient, alpha/dispersion hypergradients, prediction, input/parameter JVPs and VJPs | Fit-time optimizer differentiation remains an explicit boundary; resident CUDA kernels remain planned |
| Dense MLP and MSE training objective | Parameter/input JVPs, VJPs, exact MSE+L2 HVPs, L2 hyperparameter derivative, Adam/AdamW/Adagrad/RMSprop/AMSGrad/RAdam/SGD momentum/Nesterov training, typed constant/warmup/cosine/exponential/one-cycle schedules with analytic rate products, exact fixed-trajectory learning-rate/L2, one-cycle peak/final schedule products, exact affine outer HVPs for constant/linear-warmup/cosine/warm-up-plus-cosine/exponential/one-cycle schedules, scheduled RAdam base-rate/beta/epsilon/schedule hypergradients, AdamW (including beta logits), RMSprop hypergradients, AMSGrad/RAdam recurrence/checkpoints, exact in-memory optimizer checkpoints, exact outer SGD momentum/Nesterov HVPs for one-layer affine MLPs, and copied positive-support validation weights with exact value/JVP/VJP products | Mini-batch/schedule optimizer-group/validation-policy/device-state hypergradients and neural module families are partial; nonlinear/plateau/device scheduled outer HVPs return typed refusals |
| Exact GP regression | Kernel-parameter products, input derivatives, prediction products, and differentiated-solve HVPs; RBF, Matérn 1/2, 3/2, and 5/2, periodic, and rational-quadratic products use analytic leaves, with RBF and Matérn products cross-checked against FortSym-generated forms; `deep_kernel_gp_t` composes an MLP feature map with a dense exact GP and exposes feature/weight likelihood gradients | Approximate and matrix-free training products are partial; joint feature/kernel FortOpt training, KISS-GP/SKI, and resident CUDA execution remain open |
| Derivative-observation GP | Mixed value/first-derivative observations, dense joint latent covariance, analytic parameter JVP/VJP products, exact query-input JVP/VJP products for RBF, Matérn 3/2 and 5/2, periodic, rational-quadratic, cosine, linear, constant, and supported sum/product leaves; periodic mixed-observation likelihood HVPs now cover all logarithmic kernel/noise coordinates; validated user formulas carry analytic value/gradient/Hessian products; bounded scalar 1-D `second_derivative_gp_t` supports RBF value/gradient/Hessian/third-derivative rows with order-six covariance, order-seven query products, and analytic likelihood gradient/HVPs, while Matérn-5/2 remains at orders `0:2` | Matérn 1/2 coincident derivative blocks, rational-quadratic/cosine mixed HVPs, user-formula third-input products, higher-order observations beyond RBF order three, and resident CUDA covariance/solve kernels remain typed refusals |
| Trees, boosting, and classifiers | Piecewise JVPs where declared, with split-boundary refusals; weighted LDA/QDA exposes smooth Gaussian probability products | Classifier HVPs, smooth split derivatives, and resident classifier GPU kernels remain open |
| BNN, VAE, RNN, and most approximate GP paths | Value or model-specific gradient surfaces | Full JVP/VJP/HVP coverage is a roadmap item |

`basis_fanout_pipeline_t` now supplies a bounded named DAG composition: each
branch is an arbitrary sequential basis pipeline, forward features concatenate
in branch order, and reverse input cotangents are summed. Its packed branch
parameter layout and analytic JVP/VJP/HVP products are CPU exact; CUDA dispatch
returns a typed refusal until resident branch executors exist.

`conditional_basis_pipeline_t` adds interval-routed parallel feature branches
on top of the existing column pipeline. Each named branch selects original
columns, is active on a half-open route interval, and contributes a stable
feature/parameter block with exact CPU value/JVP/VJP/HVP products. Endpoint
derivatives, malformed intervals, and nonfinite routes are typed refusals;
parameter/schema updates are transactional. CPU device dispatch is explicit,
while CUDA remains `FORTNUM_NOT_IMPLEMENTED` until a resident route executor
exists. See [`docs/CONDITIONAL_PIPELINE.md`](docs/CONDITIONAL_PIPELINE.md).

Horizontal, sequential, and fan-out pipelines also carry a transactional dense
`basis_input_schema_t`. Install unique input names with `set_input_schema`, use
`input_schema_name` for routing, and call `validate_input_schema` before a batch
crosses the transform boundary. Duplicate, mismatched, empty, overlong, and
wrong-count names are refused without mutating the previous schema. Dtype,
sparse-layout, and estimator-wide metadata routing remain explicit roadmap
boundaries; the CPU contract and typed device boundary are benchmarked in
`fortml-bench/results/PIPELINE_SCHEMA.md`.

`fortsym` is used when it proves a smaller or more stable closed form for a
kernel or derivative leaf. `fortad` remains the general source-transformation
baseline. A generated product enters the implementation only after an
independent numerical oracle and a symbolic identity check agree. Accepted
generated artifacts must record the FortSym and FortAD revisions, operation
count, source hash, and any fallback or refusal. The release benchmarks compare
the two routes where both are available.

The current classification surface also includes positive binary temperature
scaling (`sigmoid(score/T)`) with exact score/temperature JVP/VJP products, and
the GP surface includes a bounded Bernoulli variational objective with seeded
logistic/probit ELBO samples, analytic KL, packed gradients, and exact JVPs.
Both retain explicit CUDA refusals until their complete graphs are resident.

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
| Regression and features | `fortml_linear_regression`, `fortml_ridge_regression`, `fortml_elastic_net_regression`, `fortml_linear_svr`, `fortml_glm_regression`, `fortml_basis_linear_regression`, `fortml_regression_metrics`, `fortml_preprocessing`, `fortml_sparse_preprocessing`, `fortml_simple_imputer`, `fortml_missing_indicator`, `fortml_one_hot_encoder`, `fortml_basis`, `fortml_pipeline`, `fortml_pipeline_persistence`, `fortml_column_pipeline`, `fortml_conditional_pipeline`, `fortml_validation`, `fortml_estimator_capabilities` | Dense SVD, weighted ridge and deterministic weighted elastic-net fits, weighted dense linear SVR with squared or ordinary epsilon-insensitive losses, bounded FortOpt Poisson/Gamma log-link GLMs, core weighted regression metrics, fitted mean/median/constant imputation, dense all/missing-only NaN indicators with exact zero mask products, deterministic integer-category one-hot encoding with packed metadata and explicit unknown/missing policies, standard/min-max/median-IQR scalers with exact input JVPs, sparse-safe CSC standard scaling with implicit-zero statistics and exact sparse JVP/VJP products (centering is a typed refusal), horizontal, sequential, explicit column-selecting, named residual-sum, and interval-routed conditional basis pipelines with transactional dense input schemas plus named stage/feature/parameter metadata and routed offsets, versioned transactional host persistence for fitted horizontal pipelines, and fitted basis-to-linear estimators with packed parameters and chained JVP/VJP products. Deterministic K-fold/stratified/grouped and chronological expanding or rolling time-series split iterators are public. `estimator_capability_t` gives generic fitted-state, role, shape, input-kind, derivative, probability, and device tags; `estimator_score_metadata_t` records scorer orientation and input representation; `estimator_validation_metadata_t` records explicit clone/reset declarations. Each basis pipeline exposes a capability query and validation can enforce requirements before a split is consumed. |
| Unsupervised decomposition | `fortml_pca`, `fortml_linear_autoencoder`, `fortml_kmeans` | Centered dense PCA uses a thin LAPACK SVD with deterministic component signs, rank selection, explained-variance metadata, optional sample-variance whitening, reconstruction, and fixed-state input JVP/VJP products. `linear_autoencoder_t` copies fitted PCA into a tied linear encoder/decoder, with exact encode/reconstruct JVPs and an explicit CPU-only device contract. `kmeans_t` provides deterministic seeded dense Lloyd fit/predict/transform, inertia, fixed-center input JVP/VJP products, and typed empty-cluster/device refusals. Incremental, randomized, sparse, kernel PCA, minibatch k-means, ICA, and NMF remain roadmap items. |
| Neural initialization | `fortml_mlp_last_layer_gp` | Freezes an existing MLP feature map and solves a deterministic regularized finite-feature kernel-ridge last layer, with transactional application, regularization metadata, fixed-feature JVPs, independent NumPy oracle, and typed CUDA refusal. It is a finite-feature warm start; NNGP covariance and exact infinite-width claims remain roadmap items. |
| Classification | `fortml_logistic_regression`, `fortml_linear_svm_classifier`, `fortml_rbf_svm_classifier`, `fortml_one_class_svm`, `fortml_logistic_training`, `fortml_ovr_logistic_classifier`, `fortml_ovo_logistic_classifier`, `fortml_multilabel_logistic_classifier`, `fortml_ordinal_logistic_classifier`, `fortml_softmax_regression`, `fortml_gaussian_naive_bayes`, `fortml_bernoulli_naive_bayes`, `fortml_multinomial_naive_bayes`, `fortml_complement_naive_bayes`, `fortml_categorical_naive_bayes`, `fortml_probability_calibration`, `fortml_cart_classifier`, `fortml_random_forest_classifier`, `fortml_extra_trees_classifier`, `fortml_bagging_classifier`, `fortml_adaboost_classifier`, `fortml_mlp_classifier`, `fortml_mlp_ordinal_classifier`, `fortml_gp_classification`, `fortml_gp_multiclass_classification`, `fortml_classification_metrics`, `fortml_losses` | Linear logistic and squared/ordinary-hinge SVM, dense finite-basis RBF squared-hinge SVM, dense RBF one-class nu-SVM, OVR, OVO, independent multilabel logistic, weighted ordinal cumulative-logit, GaussianNB, BernoulliNB, MultinomialNB, ComplementNB, CategoricalNB, CART, deterministic random forest and randomized-threshold Extra-Trees, seeded bagging, binary/multiclass SAMME and multiclass SAMME.R AdaBoost, neural, and binary/multiclass calibration estimators support nonnegative sample weights and positive sorted-class weights where declared. MLP classifiers expose exact prediction JVP/VJP products; tree ensembles expose deterministic aligned probabilities and explicit CPU/CUDA refusals. Linear and finite-basis RBF SVMs accept arbitrary binary integer labels, use deterministic FortOpt L-BFGS-B, expose packed score/probability JVP/VJP products, and return explicit split-boundary and CUDA refusals. The one-class SVM solves a deterministic capped-simplex RBF dual, exposes signed anomaly scores and fixed-state query JVP/VJP products, and keeps fit active-set and CUDA boundaries explicit. The ordinal linear and neural heads store ordered integer labels, strictly increasing cut points, packed parameter products, and explicit CPU/CUDA capability boundaries; the neural head trains a scalar MLP score with FortOpt L-BFGS-B. Multilabel heads return a dense positive-probability matrix, per-label thresholds, packed parameter products, and explicit CPU/CUDA capability boundaries. OVO uses an explicit normalized pairwise-vote policy and exposes pair/input/parameter derivative products. The logistic training objective adds exact packed gradients, mixed L2 HVPs, and bounded FortOpt L-BFGS-B. The OVR and Naive Bayes wrappers expose normalized probabilities, arbitrary integer labels, packed parameter metadata, and input/parameter JVP/VJP products where inputs are continuous. GaussianNB adds weighted Gaussian moments and variance smoothing; BernoulliNB adds relaxed-[0,1] features and positive smoothing; MultinomialNB adds relaxed nonnegative counts and token-mass smoothing; ComplementNB adds complement distributions, optional second weight normalization, and the same differentiable products. CategoricalNB stores sorted per-feature category metadata, weighted smoothed likelihoods, unknown-category error/ignore policies, and explicit discrete-JVP refusal. CART classification provides deterministic weighted Gini/entropy splits and piecewise-constant leaf probabilities. Binary and one-vs-rest Laplace GP classifiers use stable probabilities, input JVPs, and exact envelope gradients for their mode-log-posterior kernel objective. `probability_calibrator_t` implements Platt sigmoid and weighted PAVA isotonic calibration, while `multiclass_probability_calibrator_t` adds weighted one-vs-rest isotonic maps and positive softmax temperature with sorted-class simplex probabilities; temperature products are exact and isotonic active-set products return typed refusals; all calibration device boundaries are explicit. Shared accuracy, top-k, balanced accuracy, confusion, precision/recall/F1, Brier, binary Matthews, weighted accuracy, log-loss, expected calibration error, and maximum calibration error metrics are implemented. Coupled categorical variational GP likelihood temperature products and likelihood-only FortOpt fitting are implemented; resident CUDA inference remains typed refusal. |
The shared loss facade also provides multiclass focal-softmax value/JVP/VJP/HVP
products through `focal_softmax_cross_entropy_*` (and the
`multiclass_focal_cross_entropy_*` aliases), with weighted mean/sum reductions,
positive class factors, stable underflow refusal, and typed CUDA refusal. Set
`mlp_classifier_options_t%focal_gamma` or the FortOpt
`mlp_classifier_lbfgsb_options_t%focal_gamma` to train the same objective
through the multiclass MLP classifier; zero preserves ordinary cross-entropy.
See [`docs/NEURAL_LOSS_PRODUCTS.md`](docs/NEURAL_LOSS_PRODUCTS.md).

Pairwise metric-learning heads can use the shared contrastive loss directly:
`contrastive_loss_value`, `contrastive_loss_jvp`, `contrastive_loss_vjp`, and
`contrastive_loss_hvp` implement weighted mean/sum Euclidean pair products for
matching/non-matching embeddings. Exact zero-distance non-match and margin-kink
derivative requests return typed domain errors; CUDA value requests remain an
explicit `FORTNUM_NOT_IMPLEMENTED` boundary. See
[`docs/CONTRASTIVE_LOSS.md`](docs/CONTRASTIVE_LOSS.md).

The classification surface also includes `fortml_gp_multilabel_classification`,
which fits independent weighted logistic/probit Laplace-GP heads for dense
indicator targets and exposes packed/input JVP/VJP products plus explicit CPU/
CUDA capability metadata. It also includes `fortml_calibrated_softmax_classifier`
for leakage-safe stratified OOF softmax calibration with positive temperature,
one-vs-rest Platt, and weighted isotonic policies. Temperature and Platt have
packed smooth products; isotonic active-set products and every CUDA path are
typed refusals, and malformed refits preserve a deployed candidate. It also includes
`fortml_gp_variational_categorical_classification`:
it fits a variance-corrected shared-softmax ELBO with bounded FortOpt and exposes
packed/input JVP/VJP products plus typed CUDA refusal. Its positive softmax
likelihood temperature is a separate log-scale coordinate with exact
probability/ELBO JVP/VJP products, fixed-state probability and ELBO HVPs, and a
transactional FortOpt likelihood-only fit; CUDA likelihood products remain
typed refusals until the inducing graph is resident. The independent release lane is
`results/GP_CATEGORICAL_LIKELIHOOD.md` in `../fortml-bench`.

`mlp_multilabel_training_objective_t` exposes the shared multilabel MLP BCE
through FortOpt. The copied sample-by-label weights support nonnegative sample
weights and positive per-label class factors. Its packed vector can append a
direct L2 coefficient or `log(l2)`, with exact network/L2 JVP, VJP, and mixed
HVP products. `mlp_multilabel_optimize_lbfgsb` supplies bounded network and
L2 coordinates. Invalid indicators, weights, or coordinate modes fail before
the fitted model is changed. See
[`docs/MLP_MULTILABEL_OBJECTIVE.md`](docs/MLP_MULTILABEL_OBJECTIVE.md).

The standalone `multiclass_probability_calibrator_t` also supports weighted
one-vs-rest Platt sigmoid maps over stable softmax columns.  Its interleaved
`[slope, intercept]` parameter vector has exact smooth input and parameter
JVP/VJP products, while weighted isotonic active-set products and all CUDA
calibration requests remain typed refusals.

| Neighbors | `fortml_knn_classifier`, `fortml_radius_neighbors_classifier`, `fortml_radius_neighbors_regression` | Dense exact kNN, closed-radius classification, and scalar closed-radius regression with uniform or inverse-distance weighting, optional sample weights, deterministic boundaries, explicit empty-neighborhood policies, and a resident native-CUDA kNN training-set plan in CUDA builds. Neighbor selection explicitly refuses derivatives; KD/ball trees, multi-output regression, sparse inputs, and differentiable soft-neighbor relaxation remain planned. |
| Neural and physics models | `fortml_mlp`, `fortml_mlp_chain`, `fortml_mlp_training`, `fortml_mlp_grouped_training`, `fortml_mlp_schedules`, `fortml_mlp_hypergradient`, `fortml_mlp_sgd_momentum_hypergradient`, `fortml_mlp_schedule_hypergradient`, `fortml_mlp_adagrad_schedule_hypergradient`, `fortml_mlp_radam_hypergradient`, `fortml_mlp_radam_schedule_hypergradient`, `fortml_mlp_minibatch_hypergradient`, `fortml_mlp_minibatch_adam_hypergradient`, `fortml_mlp_classifier`, `fortml_mlp_ordinal_classifier`, `fortml_hamiltonian_mlp`, `fortml_symplectic`, `fortml_physics_objective`, `fortml_pinn`, `fortml_bnn`, `fortml_vae`, `fortml_rnn` | MLPs use deterministic Xavier/He initialization, linear/`tanh`/ReLU/GELU/SiLU/ELU/softplus/leaky-ReLU/sigmoid/Mish activations, Adam, AdamW with decoupled weight decay, Adagrad, RMSprop (centered or uncentered, optional momentum), RAdam, or FortOpt SGD with momentum/Nesterov, explicit seeded batch cursors, sample-weighted microbatch accumulation, held-out validation with interval monitoring and best-state restoration, typed constant/warmup/cosine/exponential/one-cycle schedules with analytic rate products, norm clipping, resumable optimizer state, explicit mean/sum weighted MSE diagnostics, exact MSE+L2 parameter/hyperparameter HVPs, named non-overlapping log-L2 parameter groups with exact mixed HVPs, fixed full-batch trajectory hypergradients for classical/Nesterov SGD momentum over log learning rate/L2/momentum, with an exact outer HVP on one-layer affine MLPs, AdamW over log learning rate/L2/weight decay, RMSprop over log learning rate/L2/decay/log epsilon/momentum, RAdam over log learning rate/L2/logit betas/log epsilon including bias-correction and rectification, typed schedule trajectories over log/logit rate, L2, and schedule fields, exact affine outer HVPs for constant/warm-up/cosine/one-cycle paths, exact scheduled Adagrad trajectories over log base rate/L2/epsilon and schedule logits, exact scheduled RAdam trajectories over log base rate/L2/betas/epsilon and schedule logits, and deterministic seeded mini-batch SGD and coupled-L2 Adam trajectories over log learning rate/L2, each with JVP/VJP products and a FortOpt L-BFGS-B adapter. The shared loss facade adds MAE products with exact-kink refusals, stable softmax/log-softmax value/JVP/VJP/HVP products, weighted softmax cross-entropy products, focal binary cross-entropy-with-logits value/JVP/VJP/HVP products, stable focal products, heteroscedastic Gaussian and Poisson/count NLL value/JVP/VJP/HVP products, exact BCE/logistic and softmax cross-entropy HVPs, weighted-MSE value/JVP/VJP/HVP products, and typed Huber-kink/CUDA-device refusals. The named sequential MLP chain adds stable stage ranges, exact composed JVP/VJP/HVP products, and a bounded all-stage L-BFGS-B objective. The multiclass MLP classifier additionally exposes fixed-input probability parameter JVP/VJP products with explicit CPU dispatch and typed CUDA refusals. Nonlinear/plateau/device scheduled outer HVPs retain typed boundaries. The Hamiltonian MLP supports separable and general nonseparable scalar H, canonical value/JVP/VJP/HVP products, and a symplectic leapfrog map only for the separable mode; `fortml_symplectic` adds canonical-form residual/value/JVP/VJP diagnostics and a reusable physics-constraint bridge with typed CUDA refusal. General leapfrog requests are typed refusals. `pinn_training_adapter_t` composes named data/residual/boundary/conservation terms, forwards exact products where callbacks provide them, exposes bounded L-BFGS-B fitting, and returns a typed CUDA refusal. The recurrent model is one vanilla `tanh` RNN with a zero initial state |
| Trees and boosting | `fortml_tree`, `fortml_cart_classifier`, `fortml_random_forest_classifier`, `fortml_extra_trees_classifier`, `fortml_bagging_classifier`, `fortml_adaboost_classifier`, `fortml_xgboost`, `fortml_xgboost_classifier`, `fortml_xgboost_multiclass`, `fortml_lightgbm`, `fortml_lightgbm_multiclass` | Deterministic finite-only regression stumps, weighted exhaustive-split CART regression/classification, seeded bootstrap random-forest and randomized-threshold Extra-Trees probabilities, seeded bagging, binary/multiclass SAMME AdaBoost, squared-loss residual boosting, exact and bounded-histogram depth-limited second-order squared/logistic/Poisson/fixed-shape Gamma/Tweedie/squared-log/Huber/quantile/absolute boosting with recursive Newton leaves and L1/L2/gamma/min-child-Hessian regularization, deterministic NaN default routing (`error`, learned, forced-left, or forced-right), bounded ordered-gradient integer categorical partitions, binary and one-vs-rest multiclass staged predictions, sorted integer labels, decision margins, gain/weight/cover feature importance, bounded exact-subset SHAP-like per-feature raw-margin attributions, per-feature monotonic constraints (`-1/0/+1`) with recursive leaf bounds, seeded XGBoost DART with persisted per-tree normalisation, and explicit CPU/CUDA device dispatch/refusal. `lightgbm_t` adds a separately named weighted regression/binary-logistic weighted-quantile histogram path with deterministic globally best-leaf growth up to `num_leaves`, held-out weighted validation loss with patience/min-delta, best-round diagnostics, restore-best or retain-all ensembles, transactional GOSS top/other-rate sampling and bounded seeded DART/dropout with persisted tree-normalisation scales, and versioned persistence. `lightgbm_multiclass_t` wraps those binary children with normalized staged probabilities, sorted labels, common validation prefixes, and fixed-tree input JVP/VJP products. Split-boundary derivatives and resident CUDA remain explicit refusals. Ranking, EFB, distributed, and resident histogram policies remain planned. |
| Multi-output boosting | `fortml_xgboost_multioutput`, `fortml_lightgbm_multioutput` | Transactional one-child-per-target regression adapters expose matrix/staged predictions, input and fixed-leaf JVP/VJP products, packed target metadata, and typed CUDA refusals. See [`docs/MULTIOUTPUT_BOOSTING.md`](docs/MULTIOUTPUT_BOOSTING.md). |
| Multiclass boosting validation | `fortml_xgboost_multiclass` | Weighted validation log-loss over common boosting stages, deterministic patience/min-delta early stopping, best-prefix restoration, requested/best-round diagnostics, arbitrary validation labels and transactional refusal semantics. See [`docs/XGBOOST_MULTICLASS_VALIDATION.md`](docs/XGBOOST_MULTICLASS_VALIDATION.md). |
| Multi-output boosting | `fortml_xgboost_multioutput`, `fortml_lightgbm_multioutput` | Transactional one-child-per-target regression adapters expose matrix/staged predictions, input and fixed-leaf JVP/VJP products, packed target metadata, and typed CUDA refusals. See [`docs/MULTIOUTPUT_BOOSTING.md`](docs/MULTIOUTPUT_BOOSTING.md). |
| Variational inference | `fortml_variational`, `fortml_sparse_gp`, `fortml_gp_variational_classification` | `sparse_gp_t` has scalar Gaussian targets, packed mean/log-Cholesky variational parameters, a separate transformed `[log(noise_variance)]` likelihood block with analytic fixed-state ELBO JVP/VJP/HVP products, transactional updates, and typed CPU/CUDA dispatch; `gp_variational_classification_t` adds bounded Bernoulli logistic/probit ELBOs, inducing `q(u)`, packed gradients, deterministic JVPs, minibatch scaling, and an explicit CUDA refusal |
| Exact GPs | `fortml_kernels`, `fortml_gaussian_process`, `fortml_gp_training`, `fortml_deep_kernel_gp`, `fortml_derivative_gaussian_process`, `fortml_derivative_gp_training`, `fortml_second_derivative_gaussian_process`, `fortml_multi_output_gp` | Exact-GP hyperparameters and mixed value/first-derivative GP states expose bounded FortOpt adapters. `deep_kernel_gp_t` provides a CPU MLP feature-map composition with exact feature/weight likelihood gradients. Derivative-GP prediction parameter and exact third-input query JVP/VJP products cover fixed value/first-derivative queries for the supported smooth kernels; `second_derivative_gp_t` adds RBF order-three observations and analytic likelihood gradient/HVP products; dense joint latent posterior covariance is available on CPU. See [GP_DERIVATIVES.md](docs/GP_DERIVATIVES.md) and [SECOND_DERIVATIVE_GP.md](docs/SECOND_DERIVATIVE_GP.md) for the kernel/refusal matrices. Resident CUDA covariance, factorization, and derivative-query kernels remain open. |
| Approximate GPs | `fortml_sparse_prior_gp`, `fortml_local_experts`, `fortml_ski_gp` | Multidimensional SKI requires one isotropic RBF leaf. Local experts support contiguous or deterministic clustered partitions. |
| Lazy inference | `fortml_linear_operator`, `fortml_kernel_operator`, `fortml_sparse_operator`, `fortml_structured_operator`, `fortml_toeplitz_operator`, `fortml_banded_precision` | Toeplitz products are host-resident |
| Supporting contracts | `fortml_kernel_formula`, `fortml_lanczos`, `fortml_multilevel_grid`, `fortml_inference_policy`, `fortml_parameter_registry`, `fortml_parameter_products`, `fortml_hyperparameter_search`, `fortml_device`, `fortml_cuda_metrics` | Product availability depends on the wrapped model. `fortml_hyperparameter_search` provides deterministic grid/random enumeration, bounded single-start and seeded multistart FortOpt L-BFGS-B over shared analytic objectives. `fortml_device` records explicit CPU/CUDA capability, residency ownership, and transfer events. `fortml_cuda_metrics` adds a transfer-inclusive native-CUDA weighted MSE reduction with an unavailable stub and no host fallback. Neither module claims complete GPU execution. |

The LightGBM path also exposes staged margins/predictions, additive
base-plus-tree contributions, and transactional fitted-prefix slicing. The
independent tree-walk oracles are `test_lightgbm_staged_slice` and
`test_lightgbm_multiclass`. The multiclass API is documented in
[`docs/LIGHTGBM_MULTICLASS.md`](docs/LIGHTGBM_MULTICLASS.md).

The multiclass MLP classifier also exposes a weighted softmax cross-entropy
objective with an optional L2 coordinate. Its parameter/L2 value, JVP, VJP,
and HVP products are analytic, and bounded FortOpt L-BFGS-B uses the same
callback. Resident CUDA classifier training remains a typed refusal. The
independent gate and benchmark are documented in `docs/API.md` and the
companion `fortml-bench/results/MLP_CLASSIFIER_OBJECTIVE.md` report.

`fortml_mlp_calibrated_classifier` adds the calibrated neural-head slice to
this surface.  Binary models support deterministic sigmoid, positive
temperature, and weighted isotonic calibration; multiclass models support a
single positive softmax temperature.  Smooth temperature/sigmoid network,
input, and calibration-parameter JVP/VJP products are exact, while isotonic
active-set products and all CUDA requests return explicit typed refusals.  See
[`docs/MLP_CALIBRATED_CLASSIFIER.md`](docs/MLP_CALIBRATED_CLASSIFIER.md).

`fortml_mlp_last_layer_gp` adds a deterministic finite-feature GP/NTK
last-layer warm start. It solves a regularized closed-form posterior for an
existing MLP's final affine layer, exposes named regularization metadata and a
CPU hyperparameter JVP, and returns a typed CUDA refusal. This is explicitly a
finite-width approximation, not an exact NNGP or infinite-width equivalence;
see [`docs/MLP_LAST_LAYER_GP.md`](docs/MLP_LAST_LAYER_GP.md).

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
work packages are maintained in [ROADMAP.md](ROADMAP.md). The canonical
derivative/device/HPO row schema is in
[docs/DERIVATIVE_CAPABILITY_MATRIX.md](docs/DERIVATIVE_CAPABILITY_MATRIX.md).
The classification-specific acceptance matrix is in
[docs/CLASSIFICATION_MATRIX.md](docs/CLASSIFICATION_MATRIX.md).
The package is
distributed under the [MIT license](LICENSE).
