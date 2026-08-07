# fortml roadmap

Verified on 2026-08-06. Interfaces are documented in
[`docs/API.md`](docs/API.md), examples in [`docs/EXAMPLES.md`](docs/EXAMPLES.md),
and implementation limits in [`docs/DESIGN.md`](docs/DESIGN.md) and
[`docs/ML_ARCHITECTURE.md`](docs/ML_ARCHITECTURE.md).

## Verification

| Compiler | Command | Result |
| --- | --- | --- |
| GNU Fortran | `fo` | Static and lint checks passed. Build and 30 of 30 tests passed. See [`verification/fortml-gfortran.txt`](verification/fortml-gfortran.txt). |
| NVIDIA HPC SDK | `FO_FC=nvfortran fo` | Static and lint checks passed. Build and 30 of 30 tests passed. See [`verification/fortml-nvfortran.txt`](verification/fortml-nvfortran.txt). |
| Intel LLVM Fortran | `ifx` | Compiler unavailable in the verification environment. Not tested. |

Behavioral oracles include dense or analytic references, finite differences,
adjoint identities, convergence checks, and seeded known-answer cases.
Repository-state checks do not count.

## Parity objective

FortML targets workflow parity with the parts of the Python machine-learning
stack used for supervised tabular models and differentiable neural and Gaussian
process models. The reference behavior is:

- scikit-learn for estimator, transformer, pipeline, model-selection, metric,
  linear-classifier, and tree-boosting workflows.
- PyTorch and JAX for batches, train state, optimizer steps, device selection,
  and checkpoint resumption.
- GPyTorch and GPflow for Gaussian-process hyperparameter training, derivative
  observations, and approximate inference.

Parity means that a Fortran program can prepare data, fit, validate, serialize,
reload, and deploy the listed model families without a Python runtime. Public
names and array layouts remain Fortran-native. Numerical results must agree with
an analytic, dense, finite-difference, or pinned external reference within a
documented tolerance. API naming and floating-point instruction order need not
match the reference library.

The work packages below define the parity target. An item is complete when its
API, implementation, documentation, independent oracle, and refusal tests are
present.

## Scope, architecture, and release rules

The target is a clean, Fortran-native estimator stack rather than a thin
wrapper around Python. The pre-1.0 API may change without a compatibility
shim: when a contract is wrong, replace it and update all in-tree callers,
examples, and benchmark fixtures in the same change. Once the first production
release is cut, schema and public-interface changes require an explicit
versioned migration. Every new public type must have one owning module, one
documented parameter layout, and one independent behavioral oracle.

The implementation is layered as follows:

1. **Data and array layer.** Typed row-major-at-the-API data views, masks,
   sample weights, sparse views, deterministic RNG streams, and host/device
   residency descriptors. A view never silently copies or changes the sample
   axis.
2. **Transform and basis layer.** Stateless and fitted transformers, basis
   maps, feature-name/category metadata, and composable sequential, parallel,
   and column-wise graphs. Each transform exposes value and, where supported,
   parameter/input JVP, VJP, and HVP products.
3. **Estimator layer.** Regressors, classifiers, density estimators,
   decompositions, clusterers, trees, GPs, and neural modules implement shared
   `fit`, `partial_fit`, `predict`/`transform`, `score`, `get/set_parameters`,
   clone/reset, fitted-state, and status contracts. The contract records
   whether a method is exact, stochastic, approximate, or refused.
4. **Objective and derivative layer.** A scalar objective owns reduction,
   weights, regularization, constraints, diagnostics, and a flat parameter
   registry. A derivative product is selected by capability (analytic,
   `fortsym`-generated, `fortad`, or an explicitly refused path), never by an
   accidental finite-difference fallback in production.
5. **Training and search layer.** Trainers own batches, optimizer state,
   schedules, validation, callbacks, checkpoints, and distributed reduction.
   Hyperparameter search treats model, preprocessing, and optimizer settings as
   named differentiable or nondifferentiable blocks and can call bounded
   L-BFGS-B when a complete gradient/HVP contract exists.
6. **Backend and deployment layer.** CPU, OpenACC, CUDA, and later accelerator
   backends share the same numerical contract. A device path must cover the
  complete operation graph or return a precise refusal. Hidden host transfers
   and silent precision changes are errors.

The target is intentionally broader than the currently implemented subset.
The capability matrix below is the source of truth for what is implemented,
partial, planned, or explicitly refused. A checked box never means that
one happy-path method exists: it means the public contract, independent oracle,
documentation, refusal behavior, and benchmark evidence are all present.

### Capability matrix (target versus current state)

| Area | Current state | Production target |
| --- | --- | --- |
| Linear regression and generalized linear models | Linear regression is implemented. Logistic and softmax are partial | OLS, weighted/ridge/lasso/elastic-net, robust, quantile, Poisson/Gamma/Tweedie, multinomial, calibrated and regularized classifiers with shared solver and derivative contracts |
| Feature transforms and basis maps | Polynomial, Fourier, radial, B-spline, callback bases are implemented | Fitted preprocessing, sparse/categorical features, feature names, unions, DAG pipelines, leakage-safe cross-validation, differentiable basis hyperparameters |
| Nearest-neighbor and margin methods | Missing | kNN/KD-tree or ball-tree search, kernel/radius neighbors, linear/kernel SVM and SVR, calibrated probabilities, deterministic tie and missing-value policies |
| Trees and ensembles | Missing | CART, random/extra trees, random forests, histogram gradient boosting, XGBoost/LightGBM-style boosting, ranking, monotonic and interaction constraints |
| Clustering and unsupervised learning | Missing | k-means/minibatch k-means, Gaussian mixtures/EM, density and graph clustering, manifold methods, outlier detection, decomposition, matrix factorization, and density metrics |
| Neural networks | MLP/BNN/VAE/RNN primitives and selected products exist | A production module/parameter tree, all common activations and losses, convolution/attention/sequence/graph extensions, mixed precision, distributed training, compile/fusion, and resumable trainers |
| Gaussian processes | Exact, derivative, sparse, structured and local variants are partial-to-implemented | GPyTorch/GPflow-style kernels, likelihoods, multitask/batch shapes, exact/variational/lazy inference, derivative operators, constraints, calibration, and trainable hyperparameters |
| Derivatives | Exact GP and selected neural/kernel products exist | Value/JVP/VJP/HVP and implicit/hypergradients for every declared parameter/input path, including preprocessing, likelihood, optimizer/search variables, and device kernels |
| Model selection and metrics | Benchmark-specific checks exist | Shared metrics, splitters, cross-validation, calibration, grid/random/Bayesian/differentiable search, nested validation, and leakage/refusal checks |
| Persistence and serving | Missing public contract | Versioned state dictionaries, safe model/trainer serialization, compiler-independent metadata, streaming inference, batching, and reproducible deployment manifests |
| GPU and scale-out | Operator-specific OpenACC/CUDA paths | Complete resident CPU/CUDA/OpenACC training and inference for supported estimators, mixed precision, multi-GPU/MPI sharding, transfer accounting, and deterministic reductions |
| Performance evidence | Several model/GP lanes exist | Matched correctness-gated comparisons with scikit-learn, XGBoost/LightGBM, PyTorch/JAX, GPyTorch/GPflow, and published hardware/toolchain provenance |

## Current baseline

### Package and public contracts

- [x] Establish the MIT-licensed package with sibling `fortnum`, `fortopt`,
  `fortad`, and `fortsparse` dependencies and separate public Fortran modules.
- [x] Store samples in array rows at the API level, return explicit status
  objects, use flat column-major parameter packing, and provide one
  optimizer-facing registry for live model parameter blocks.
- [x] Add registry-routed value/JVP/VJP/HVP products for the models that
  declare them, with checks over every packed parameter.
- [x] Document the public module surface, array shapes, parameter layouts,
  examples, refusal behavior, and backend limits.

### Regression, features, and neural models

- [x] Implement multi-output linear regression with intercepts, ridge
  regularization, an SVD least-squares fit, and prediction JVP/VJP products.
- [x] Implement polynomial, Fourier, radial, B-spline, and callback basis maps
  behind one facade, with value/JVP/VJP products and differentiable radial and
  Fourier parameters.
- [x] Implement dense MLPs with linear, `tanh`, and ReLU activations, batched
  prediction, parameter and input JVP/VJP products, and weighted-output HVPs.
- [x] Check the MLP optimizer seam with `fortopt_adam` and compare static scalar
  fixtures with `fortad`-generated JVP, VJP, and HVP kernels.
- [x] Implement Bayesian neural networks with deterministic seeded Monte Carlo,
  Gaussian variational posteriors, analytic KL terms, and ELBO
  value/JVP/VJP/HVP products.
- [x] Implement reusable diagonal and full-covariance Gaussian variational
  families, seeded reparameterization, minibatch scaling, KL products, and
  natural-gradient or `fortopt` update seams.
- [x] Implement a Gaussian VAE and a vanilla `tanh` RNN with explicit parameter
  packing, reconstruction or forward evaluation, and ELBO or BPTT gradients.

### Kernels and Gaussian-process models

- [x] Implement RBF, Matérn 1/2, Matérn 3/2, Matérn 5/2, linear, constant, and
  white-noise kernels, plus recursive sum and product kernels.
- [x] Implement validated bounded postfix formulas for user kernels and lower
  eligible formulas into the generic kernel-operator program.
- [x] Supply kernel matrix products, parameter JVP/VJP/HVP products, input
  derivatives, generated radial derivative kernels, and independent analytic,
  finite-difference, adjoint, and `fortsym` checks.
- [x] Implement exact multi-output-column GP regression with Cholesky inference,
  latent predictive variance, log marginal likelihood, hyperparameter
  gradients, prediction products, and differentiated-solve HVPs.
- [x] Implement mixed function-value and first-derivative observations with
  explicit Matérn smoothness and white-noise refusal rules.
- [x] Implement correlated multi-output GPs and scalar-target inducing-point
  variational GPs with caller-supplied variational parameters.

### Lazy and structured inference

- [x] Define the abstract linear-operator contract for vector and multi-right-
  hand-side products, diagonals, sample counts, CG, and independent batched CG
  recurrences.
- [x] Implement specialized RBF and generic composable-kernel operators with
  cached data residency, fused products, diagonal preconditioning, and dense
  solve oracles.
- [x] Add block and Nyström preconditioners, stochastic Lanczos log
  determinants, and LOVE-style predictive-variance products.
- [x] Implement compact-support sparse operators through `fortsparse`, including
  CSC construction, host products, nonzero diagnostics, and a resident
  OpenACC CSR view.
- [x] Implement separable tensor-product operators with vector, batched,
  derivative, CG, and resident OpenACC products.
- [x] Implement cached one-dimensional Toeplitz products, multilevel-grid
  prolongation and restriction, and banded Markov precision factorization,
  solves, and log determinants.
- [x] Add a declared-structure inference policy for dense Cholesky, tensor
  grids, compact support, banded precision, matrix-free Krylov, and inducing
  points.

### Accelerator backends

- [x] Add correctness-gated tiled RBF vector and batched products with OpenACC
  residency and `nvfortran` compilation.
- [x] Add device CG paths that retain lowered kernel programs, data, right-hand
  sides, and reusable workspaces across products and solves.
- [x] Define a backend-neutral opaque C ABI for resident matrix-free plans and
  implement the CUDA backend for generic postfix products.
- [x] Add optional native CUDA shared-neighbor tiles for the fixed
  eight-feature RBF vector and batched paths, with OpenACC fallbacks.
- [x] Consume a `fortsym`-generated RBF CUDA leaf through the resident postfix
  plan and check the resident-plan path against independent dense products.

### Approximate Gaussian processes

- [x] Implement subset-of-data selection and the SoR/DTC/FITC/PITC prior
  approximations with independent dense posterior and likelihood oracles.
- [x] Implement the variational inducing-point path used by the VFE benchmark
  driver.
- [x] Implement one-dimensional Toeplitz SKI and multidimensional Kronecker SKI
  with local multilinear interpolation, matrix-free CG, diagonals, batched
  products, and interpolated train-to-query cross products.
- [x] Implement NLE/PoE/GPoE/BCM/RBCM/GRBCM and entropy-gated MoE
  aggregation with balanced contiguous or deterministic Lloyd partitions.
- [x] Implement the GRBCM communication-set semantics from Liu et al., *When
  Gaussian Process Meets Big Data*, IEEE TNNLS 31(11), Eq. (29): seeded
  selection without replacement, disjoint remainder partitions, enhanced
  experts on `D_c` union `D_i`, the first unit coefficient, later unclipped
  entropy coefficients, and the communication-expert correction.
- [x] Add the review toy problem and one benchmark driver for exact, inducing,
  SKI, local-expert, clustered local-expert, and matrix-free comparison lanes.

### Benchmark applications

- [x] Add correctness-gated applications for linear regression, MLP, exact GP,
  GP feature products, and approximate GP workloads.
- [x] Have the applications emit correctness results and timed phases in
  machine-readable output. The approximate GP driver emits explicit NaN rows
  for preflight refusals.
- [x] Use the sibling `../fortml-bench` harnesses to add peak RSS, build and
  toolchain provenance, release measurements, and external Python comparisons.

## Supported boundaries

Multidimensional SKI accepts one isotropic RBF leaf. Its `n_grid` argument is a
maximum total grid budget. Every axis receives the largest common extent `q`
such that `q**d <= n_grid`, and each interpolation row has `2**d` corners.
Queries outside the training box clamp to its nearest face. The benchmark driver
reports an SKI mean and marks predictive variance undefined.

Local fits use balanced contiguous blocks by default or deterministic Lloyd
clusters through `fit_clustered`. GRBCM requires at least two reported experts.
Its fixed default communication seed is reproducible, and callers may supply a
different seed. MoE uses an entropy-score softmax gate. The gate and inducing
locations are not learned jointly with model parameters.

The benchmark VFE lane constructs its variational distribution with dense
linear algebra. It is not a minibatch stochastic-variational implementation.
Toeplitz FFT products remain host-resident.

Accelerator support is operator-specific. OpenACC and native CUDA paths cover
the kernel, structured, and sparse operations documented above. The
`nvfortran` verification establishes compiler compatibility for all 30 tests.
It does not establish that every model trains or predicts entirely on a GPU.

## Parity work packages

The source inventory is dated 2026-08-07.

| Work package | State | Implemented baseline | Package exit |
| --- | --- | --- | --- |
| Classification | Partial | `fortml_logistic_regression` and `fortml_softmax_regression` provide binary and multinomial integer-label fitting, stable probabilities, deterministic class order, and FortOpt L-BFGS-B optimization. Shared metrics and weighting are open. | Binary and multiclass linear, neural, GP, and boosted-tree classifiers share label, probability, weighting, and metric conventions. |
| Estimator contracts, pipelines, and bases | Partial | `basis_map_t`, row-oriented sample conventions, status objects, and the parameter registry are public. | Fitted transformers and estimators compose without data leakage, expose routed parameters, and run through cross-validation. |
| Tree boosting | Missing | No tree, split finder, histogram builder, or ensemble module exists. | Regression and classification trees support deterministic histogram boosting, validation-based stopping, missing values, and model persistence. |
| Training infrastructure | Partial | Model-specific gradients, `fortopt_adam` integration, natural-gradient seams, and seeded variational draws exist. | One trainer owns batches, optimizer state, schedules, clipping, validation, early stopping, callbacks, and resumable state for every model with a completed trainer adapter. |
| GP derivatives and hyperparameters | Partial | Exact GP likelihood and prediction products include parameter gradients and HVPs. Mixed value and first-derivative observations can be fitted and predicted. | Exact, derivative, multi-output, sparse, and matrix-free GP families expose documented trainable parameters, scalar objectives, parameter gradients, and train-state adapters. |
| GPU and device execution | Partial | Kernel, structured, and sparse operator products have selected OpenACC or CUDA paths, including resident CG for kernel operators. | Supported training and prediction workflows keep model, optimizer, and batch state resident on a selected device and have CPU parity tests. |
| Serialization and distributed execution | Missing | No public model-file or distributed-execution contract exists. | Versioned model and trainer files round-trip across supported compilers, and MPI training or inference agrees with a one-rank oracle. |
| Benchmark coverage | Partial | Correctness-gated model and GP applications feed release harnesses in `../fortml-bench`. | Every completed parity package has a pinned external oracle, release timings, memory measurements, provenance, raw data, and a maintained report. |

### WP1: classification

- [x] Define one public class-label contract. Classes have a deterministic order,
  predicted labels use that order to break ties, and probability matrices have
  one column per class.
- [x] Add binary logistic regression with an intercept, L2 regularization,
  `fit`, `decision_function`, `predict_proba`, and `predict`.
- [ ] Add sample weights and class weights to binary logistic regression while
  preserving the documented reduction and class-label contract.
- [x] Add multinomial softmax regression with a numerically stable log-sum-exp
  objective. Its class-label and weighting contract still needs the shared
  metrics and sample/class-weight extension below.
- [ ] Add classifier adapters for `mlp_t` and variational GP classification.
  Each adapter owns its likelihood and training objective instead of treating a
  raw network or GP mean as a probability.
- [ ] Add one-vs-rest and one-vs-one wrappers with deterministic class
  ordering, tie handling, shared preprocessing, and a clear distinction
  between independent binary probabilities and a normalized multiclass
  probability simplex.
- [ ] Add multilabel-indicator and multioutput classification contracts,
  including per-output class metadata, sparse target support, threshold
  policies, micro/macro/samples averaging, and refusal of ambiguous target
  shapes.
- [ ] Add ordinal classification and ranking losses as separate contracts.
  do not reinterpret nominal softmax probabilities as ordered outcomes.
- [ ] Add tree and boosted-tree classifier adapters once WP3 establishes the
  shared class, missing-value, monotonic-constraint, and probability contract.
- [ ] Add Bernoulli, probit, robust, and multiclass variational GP likelihoods,
  quadrature or variational objectives, predictive probability products, and
  calibrated latent-to-observed uncertainty. All GP classifiers share the
  same label and metric machinery as linear and neural classifiers.
- [ ] Add neural classifier heads for binary, multinomial, multilabel, ordinal,
  and calibrated outputs. Heads expose loss, logits, probabilities, and
  derivative products separately so a user can train on logits without losing
  a stable deployment probability path.
- [ ] Add accuracy, balanced accuracy, confusion matrix, log loss, precision,
  recall, F1, and binary ROC AUC with explicit zero-division and sample-weight
  behavior.
- [ ] Add probability calibration by sigmoid and isotonic fits after the base
  classifier API is stable.

Acceptance: hand-computed separable and nonseparable fixtures cover labels,
weights, ties, probabilities, multilabel thresholds, and ordinal cut points.
Objective gradients agree with central finite differences and, where available,
`fortad` or `fortsym` products. A pinned scikit-learn harness compares
coefficients or decision scores where the objectives match and compares
probabilities and metrics for all public classifiers. A separate XGBoost or
LightGBM fixture covers boosted probabilities. GP and neural fixtures compare
latent and observed probabilities under matched likelihoods. Invalid labels,
nonfinite data, empty classes, ambiguous target shapes, and mismatched weights
return status errors.

### WP2: estimator contracts, pipelines, and feature bases

- [x] Provide polynomial, Fourier, radial, B-spline, and callback basis maps with
  value, JVP, and VJP products.
- [x] Provide row-oriented sample conventions, explicit status results, and a
  registry for packed model parameters.
- [ ] Define fitted transformer, predictor, regressor, and classifier contracts.
  The contracts cover feature counts, fitted state, reset or clone behavior,
  parameter names, and status propagation.
- [ ] Define estimator tags and capability queries for supervised versus
  unsupervised targets, sparse/dense inputs, missing values, sample weights,
  partial fitting, derivatives, device support, and probabilistic outputs.
  Search and pipeline code must query capabilities instead of guessing from a
  dynamic type.
- [ ] Add standard and min-max scaling, constant and mean imputation, one-hot
  encoding with a stored category order, and column selection.
- [ ] Add robust scaling, quantile and power transforms, normalization,
  missing-indicator features, ordinal encoding, target encoding with leakage
  guards, polynomial interactions, hashing, and sparse CSR/CSC feature views.
  Every fitted transform records statistics, feature names, dtypes, and the
  treatment of unseen or missing categories.
- [ ] Add sequential pipelines, parallel feature unions, and column-wise
  transformers. Basis maps must work as fitted or fixed pipeline stages.
- [ ] Add named DAG composition with fan-out/fan-in, residual branches,
  conditional stages, and a cycle refusal. Pipeline nodes expose local
  parameter blocks so hyperparameter derivatives and optimizer routing remain
  composable through the graph.
- [ ] Add feature-name propagation, schema validation, metadata routing,
  train-only fitting, `fit_transform`, `transform`, `inverse_transform` where
  mathematically defined, and partial-fit propagation for streaming data.
- [ ] Add deterministic train/test, K-fold, stratified K-fold, and grouped split
  iterators plus cross-validation scoring and routed parameters.
- [ ] Add grid and seeded random parameter search after estimator cloning and
  scoring are stable.
- [ ] Add successive-halving, Bayesian, and differentiable hyperparameter
  search. Differentiable search must distinguish validation objectives from
  training objectives and expose implicit differentiation through the fitted
  estimator only when its linear solves and stopping policy are differentiable.

Acceptance: pipeline predictions equal a manually composed reference for each
stage. Cross-validation tests prove that transformer statistics use training
folds only. Split indices have seeded known answers. Parameter routing reaches
exactly one named stage, and a stage failure preserves its original status.
Examples cover regression and classification pipelines with mixed numeric and
categorical columns, sparse features, basis maps, and one differentiable
hyperparameter block. A deliberate train/validation leakage fixture must fail.

### WP3: trees and histogram boosting

- [ ] Add deterministic CART regression and classification trees with weighted
  squared-error, Gini, and entropy criteria, depth and leaf constraints, and a
  specified tie rule.
- [ ] Add weighted quantile binning and per-feature histograms. Store missing
  values in a dedicated bin and learn a default branch at every split.
- [ ] Add gradient-boosted regression for squared, absolute, and Huber losses.
- [ ] Add binary and multiclass gradient-boosted classification with stable
  logistic and softmax objectives.
- [ ] Add learning-rate shrinkage, row and feature subsampling, L1 and L2 leaf
  penalties, validation-based early stopping, warm starts, and deterministic
  feature importance.
- [ ] Add random forest, extra-trees, bagging, random-subspace, and
  isolation-forest ensembles with seeded bootstrap semantics, out-of-bag
  diagnostics, class-balanced sampling, and permutation or split importance.
- [ ] Add XGBoost-style second-order boosting: per-leaf gradient/Hessian
  aggregation, regularized split gain, weighted quantile sketch, exact and
  histogram algorithms, sparse-aware default directions, monotonic and
  interaction constraints, column blocks, and deterministic feature ordering.
- [ ] Add LightGBM-style histogram and leaf-wise growth as a separately named
  policy. Cover exclusive-feature bundling, gradient-based one-sided sampling,
  categorical target statistics with leakage guards, and a bounded-memory
  external-data iterator only after independent small-data oracles exist.
- [ ] Add DART/dropout boosting, class- and query-weighted ranking objectives
  (pairwise logistic and Lambda-style NDCG), survival/count objectives,
  quantile and Tweedie losses, custom objective callbacks, and custom
  evaluation metrics. Unsupported objectives must return a structured refusal.
- [ ] Add warm-start continuation, monotone prediction checks, staged
  predictions, partial dependence, SHAP-like additive contribution products,
  and model-size/tree export diagnostics.
- [ ] Add categorical split support only after numeric and missing-value
  behavior has independent oracles and benchmark evidence.

Acceptance: small trees reproduce exhaustive hand-enumerated split searches,
including weighted and missing-value cases. Training loss is nonincreasing for
the exact line-search fixtures. Fixed seeds reproduce tree structures and
predictions. Pinned scikit-learn or XGBoost workloads compare probabilities and
regression predictions under matched objectives. Pinned XGBoost and LightGBM
workloads additionally compare growth policy, quantile approximation, missing
value routing, constraints, and staged predictions. Release benchmarks record
fit and predict time, peak memory, tree count, depth, histogram size, split
count, early-stopping iteration, feature importance, and (where applicable)
ranking quality. Differences caused by regularization, quantile approximation,
or tie ordering are reported rather than hidden.

### WP3a: scikit-learn estimator-family parity

The following families are the concrete scikit-learn parity inventory. They are
separate from the differentiable GP and neural work because each has different
state, scoring, and refusal rules.

- [ ] Add weighted OLS, ridge, lasso, elastic-net, positive-constrained,
  Bayesian ridge, ARD regression, Huber, Theil-Sen, RANSAC, quantile,
  Poisson, Gamma, and Tweedie regression. Solvers expose convergence status,
  regularization scaling, warm starts, and coefficient covariance where it is
  defined.
- [ ] Add linear SGD/regression and classification with deterministic minibatch
  schedules, averaging, penalties, and `partial_fit` semantics. Keep its
  stochastic objective separate from the exact logistic/softmax objective.
- [ ] Add nearest-neighbor regression/classification, radius neighbors,
  kernel-density estimation, and large-data exact/brute, KD-tree, and ball-tree
  search with deterministic ties, metric callbacks, and missing-value policy.
- [ ] Add linear and kernel SVM/SVR, one-class SVM, and kernel approximation
  (Nyström and random Fourier features), with probability calibration and
  explicit solver/feature-memory limits.
- [ ] Add Gaussian/Multinomial/Bernoulli/Complement/Categorical naive Bayes,
  LDA/QDA, and discriminant shrinkage with stable log-probability products.
- [ ] Add k-means/minibatch k-means, Gaussian mixtures, Bayesian mixtures,
  spectral and agglomerative clustering, DBSCAN/OPTICS, affinity propagation,
  BIRCH, and graph-connected components where dependencies and memory limits
  are explicit.
- [ ] Add PCA, incremental/randomized PCA, sparse PCA, kernel PCA, ICA, NMF,
  dictionary learning, truncated SVD, random projection, and covariance
  estimators with reconstruction, whitening, rank, and sign conventions.
- [ ] Add manifold and embedding methods (t-SNE/UMAP-like experimental lanes),
  novelty/outlier detectors (isolation forest, local outlier factor, robust
  covariance, one-class methods), and density metrics only after reproducible
  seeded behavior is specified.
- [ ] Add multioutput, multiclass, multilabel, regressor/classifier chains,
  voting, stacking, bagging, and calibrated meta-estimators with nested
  parameter routing and leakage-safe fitting.

Acceptance: every estimator family has a hand-computable fixture, a refusal
matrix, and a pinned scikit-learn comparison for values, shapes, fitted state,
and metrics. Approximate or stochastic methods additionally compare seeded
distributions and report algorithmic differences. `partial_fit`, warm-start,
clone, reset, and sample-weight behavior are tested independently of the
mathematical objective.

### WP3b: metrics, validation, and model selection

- [ ] Implement regression metrics (R2, explained variance, MSE, RMSE, MAE,
  median absolute error, max error, MSLE, MAPE, pinball, Poisson/Gamma/Tweedie
  deviance), classification metrics (accuracy, top-k, balanced accuracy,
  precision/recall/F-beta, Jaccard, Hamming, log loss, ROC/PR AUC, Brier,
  calibration error), ranking metrics (DCG/NDCG, MAP, MRR), clustering metrics
  (silhouette, Calinski-Harabasz, Davies-Bouldin, adjusted rand, mutual
  information), and probabilistic metrics (NLL, CRPS, interval coverage,
  sharpness, calibration).
- [ ] Define finite, NaN, masked, zero-support, zero-division, multiclass,
  multilabel, and sample-weight behavior for every metric. Metrics return a
  value plus diagnostics rather than silently dropping invalid rows.
- [ ] Add train/test, K-fold, repeated K-fold, stratified, grouped,
  time-series/blocked, and Monte Carlo splitters. Index generation is seeded,
  independent of estimator state, and safe for empty or uneven folds.
- [ ] Add cross-validation prediction, learning curves, validation curves,
  permutation tests, bootstrap confidence intervals, calibration curves, and
  statistical comparison reports. Every transform is fitted inside each fold.
- [ ] Add grid, random, successive-halving, Bayesian, multi-fidelity, and
  differentiable search with parallel trials, pruning, seeded resume, failure
  recording, and nested CV. Search results include all parameter blocks,
  training state, resource budget, and validation split provenance.
- [ ] Add an Optuna-like trial interface without requiring Python, plus a
  bounded adapter for FortOpt L-BFGS-B when the model provides a complete
  hyperparameter gradient. The adapter must never finite-difference a noisy or
  early-stopped objective without an explicit user opt-in and warning status.

Acceptance: metrics agree with direct formulas and pinned scikit-learn or
specialist references on dense, sparse, weighted, masked, and degenerate
fixtures. Cross-validation catches a deliberately leaky transformer. Search
resumption reproduces trial order and best state, while failed or refused
trials remain visible in the result schema.

### WP4: training infrastructure

- [x] Expose packed parameters and model-specific JVP, VJP, HVP, or gradient
  products for the current trainable neural and exact GP models.
- [x] Check one MLP update seam with `fortopt_adam`, and expose natural-gradient
  or `fortopt` update seams for Gaussian variational families.
- [ ] Define objective and loss contracts with sum and mean reductions, sample
  weights, regularization terms, and named scalar diagnostics.
- [ ] Define a module tree and parameter-tree API for nested neural networks.
  Parameters, buffers, frozen blocks, tied/shared weights, masks, and stateful
  layers have stable paths and can be flattened without losing aliases.
- [ ] Add common neural losses and metrics: MSE/MAE/Huber/quantile,
  cross-entropy and focal losses, multilabel and ordinal losses, Gaussian and
  count likelihoods, contrastive and triplet losses, KL terms, and sequence
  masking. Every loss has explicit reduction, weighting, logits/probability,
  and empty-batch semantics.
- [ ] Add a deterministic batch iterator with seeded shuffling, final-batch
  behavior, and separate training and validation streams.
- [ ] Add a trainer that owns optimizer state, learning-rate schedules, gradient
  clipping, accumulation, validation intervals, early stopping, and callbacks.
- [ ] Add production optimizers and schedules: SGD with momentum/Nesterov,
  Adam/AdamW, RMSprop, Adagrad, L-BFGS/L-BFGS-B, natural gradient, cosine,
  one-cycle, warmup/decay, plateau, and user callbacks. Optimizer state is
  dtype/device aware and rejects incompatible parameter trees.
- [ ] Add automatic mixed precision with loss scaling, overflow detection,
  master weights, deterministic accumulation modes, and explicit fp16/bf16/fp32
  capability reports. A mixed-precision result must pass a full-precision
  accuracy oracle before it can enter a performance report.
- [ ] Add gradient accumulation, microbatching, activation checkpointing,
  truncated BPTT, gradient centralization/noise, norm/value clipping, and
  anomaly detection with parameter-path diagnostics.
- [ ] Add callback and event contracts for logging, progress, validation,
  checkpoint, early stop, learning-rate changes, and recoverable failures.
  Callback order and failure propagation are deterministic and testable.
- [ ] Add data-loader workers or asynchronous prefetch only when the ownership
  and RNG contract is explicit. Worker count must not silently change the
  sampled batches for a deterministic run.
- [ ] Add trainer adapters for linear classifiers, MLPs, BNNs, VAEs, RNNs, exact
  GPs, derivative GPs, and sparse variational GPs. Each adapter requires a scalar
  objective, parameter gradient, reduction rule, and complete train-state update.
- [ ] Add adapters for convolutional, recurrent/attention, graph, autoencoder,
  probabilistic, and tree/boosting objectives as their model contracts land.
- [ ] Add compile/fusion planning for static expression graphs and batched
  kernels, with a cache key containing architecture, dtype, device, and shape.
  Compilation may be optional, but a stale or incompatible plan must refuse
  rather than execute with wrong strides.
- [ ] Define in-memory train state independently of file serialization. It must
  include parameters, optimizer accumulators, epoch and batch positions, RNG
  streams, schedules, and early-stopping state.
- [ ] Add distributed data/model parallel state, all-reduce precision policy,
  gradient bucketing, elastic rank refusal, and deterministic checkpoint
  barriers. A single-rank path remains the reference implementation.

Acceptance: each adapter has an independent gradient oracle and a fixture whose
objective decreases under a documented optimizer configuration. Two runs with
the same seeds produce the same batches and parameter history. Saving train
state in memory at a batch boundary and resuming it reproduces the uninterrupted
CPU run. Callback order, early stopping, clipping, and failed optimizer steps
have known-answer tests. Pinned PyTorch and JAX fixtures compare loss curves,
gradient norms, parameter updates, checkpoint-resumed outputs, and throughput
under matched dtype, batch, seed, and compiler settings. A result is a
performance claim only when compilation, warmup, input transfer, and steady-
state phases are reported separately.

### WP5: GP derivatives and hyperparameter training

- [x] Expose exact GP log marginal likelihood gradients, JVPs, HVPs, prediction
  JVPs, VJPs, and differentiated-solve HVPs for kernel and noise parameters.
- [x] Fit and predict mixed function values and first input derivatives with
  kernel smoothness and white-noise refusal rules.
- [x] Expose kernel parameter products and input gradients plus mixed input
  Hessians for the supported analytic kernels.
- [ ] Add trainable constant and linear mean functions and automatic relevance
  determination length scales. Parameter packing must include mean, kernel,
  likelihood, and optional inducing-location blocks in a documented order.
- [ ] Add bounded hyperparameter optimization with multiple seeded restarts,
  priors, jitter escalation, convergence diagnostics, and retained best state.
- [ ] Define a derivative capability table for every estimator, transform,
  objective, and backend. It must list supported value, input JVP/VJP/HVP,
  parameter JVP/VJP/HVP, stochastic-path derivative, and refusal conditions.
  an absent product is never inferred to be zero.
- [ ] Add a unified hyperparameter registry covering preprocessing statistics,
  basis frequencies/knots, model weights, kernel and likelihood parameters,
  inducing locations, regularization, optimizer schedules, and differentiable
  validation weights. Registry entries carry bounds, transforms, priors,
  trainability, device residency, and provenance.
- [ ] Route complete hyperparameter gradients and HVPs through bounded
  FortOpt L-BFGS-B, with projected-gradient stopping, active-bound diagnostics,
  line-search status, seeded multistart, and best-finite-state retention.
  Optimization must use the same derivative products as training and expose
  an independent finite-difference or dense oracle for each objective.
- [ ] Add implicit differentiation through linear solves, fixed-point
  iterations, variational optima, early-stopped training, and cross-validation
  only when convergence and solver tolerances are part of the declared
  contract. Otherwise return a refusal rather than differentiate an unstated
  approximation.
- [ ] Add mixed-partial and symmetry checks for Hessians, adjoint identities
  for VJPs, directional finite differences for JVPs, and randomized property
  tests over parameter blocks. Check input, parameter, hyperparameter, and
  pipeline derivatives independently so a shared packing bug cannot pass all
  tests.
- [ ] Generate analytic kernels with `fortsym` when it proves a smaller
  expression, preserve the proof/operation-count/source hash, and compare the
  generated product against current FortAD `main` and an independent oracle.
  Generated code is accepted only with a fallback or a documented structured
  refusal for unsupported shapes and smoothness.
- [ ] Add likelihood gradients and HVPs for derivative-observation GPs, including
  noise parameters for each observation type.
- [ ] Add joint posterior covariance for value and derivative queries, plus
  cross-covariances between requested components.
- [ ] Extend derivative observations to second derivatives only for kernels with
  the required smoothness, with explicit refusal at singular coincident cases.
- [ ] Add scalar objectives and parameter gradients for multi-output, sparse
  variational, local, SKI, Lanczos, and matrix-free GP paths. Inducing-point and
  local-gate training remain separate parameter blocks.
- [ ] Add Bernoulli and multiclass variational GP classification after the shared
  classifier likelihood and metric contracts are complete.

Acceptance: every new derivative agrees with central finite differences and an
independently assembled dense covariance on small fixtures. Hyperparameter fits
reproduce seeded trajectories and retain the best finite objective. Pinned
GPyTorch or GPflow comparisons cover posterior means, variances, derivative
covariances, likelihoods, and optimized parameters under matched kernels and
jitter. Boundary tests cover smoothness, duplicate inputs, nonfinite parameters,
and failed factorizations. Hyperparameter-search fixtures compare the selected
state and validation objective against a high-accuracy dense oracle, not merely
against FortOpt's own callback output.

### WP5a: GPyTorch and GPflow parity matrix

The GP target is a capability-for-capability comparison, not a claim that every
approximation has identical floating-point instruction order. A parity fixture
pins the kernel, likelihood, mean, batch shape, train/eval mode, jitter, solver
tolerance, quadrature rule, and random stream before comparing results.

- [ ] Match the common kernel families: RBF/SE, Matérn, linear, constant,
  polynomial, periodic, spectral mixture, cosine, piecewise-polynomial,
  locally periodic, additive, product, and user-composed kernels. Include
  priors, constraints, ARD, batch-shaped parameters, and active dimensions.
- [ ] Match Gaussian, Bernoulli/probit, categorical, Student-t, Poisson,
  negative-binomial, heteroskedastic, multitask, and likelihood-noise models
  that have a stable FortML objective. Record quadrature or variational
  approximations instead of silently substituting a different likelihood.
- [ ] Match exact Cholesky, conjugate-gradient lazy inference, LOVE variance,
  SKI/KISS-GP, inducing-point variational inference, stochastic variational
  inference, deep-kernel and multi-output/LMC paths. Deep GP and unsupported
  non-Gaussian approximations remain explicit experimental/refusal lanes.
- [ ] Match module state, priors, constraints, batch shapes, train/eval mode,
  fantasy or online updates, posterior sampling, and state-dict round trips.
  Fortran-native serialization may use a different file format, but it must
  preserve the same semantic state and prediction.
- [ ] Match derivative information for function values, input derivatives,
  mixed derivatives, parameter derivatives, and derivative covariance blocks.
  Compare both latent and observed predictive distributions, including noise
  on each derivative-observation type.
- [ ] Add cross-library fixtures for exact small problems, variational small
  problems, and matrix-free large problems. Compare posterior mean, variance,
  covariance slices, log likelihood, gradients, HVPs, optimizer trajectories,
  and calibrated intervals within declared tolerance bands.

GPyTorch-style lazy operators and batched shapes are considered complete only
when the same operation graph can run without materializing a dense covariance
on the target workload. A dense fallback is useful as an oracle but does not
count as a production lazy implementation.

### WP6: GPU and device execution

- [x] Provide correctness-gated OpenACC kernel, structured, and sparse products,
  resident kernel-operator CG, and an opaque CUDA plan for postfix kernels.
- [x] Verify the full host test suite with `nvfortran`. This is compiler coverage,
  not end-to-end GPU coverage.
- [ ] Define a public device selector and ownership contract for host and CUDA
  allocations, streams, synchronization, and recoverable backend refusal.
- [ ] Keep batches, parameters, gradients, optimizer accumulators, and workspaces
  resident through complete MLP and variational training steps.
- [ ] Extend residency to basis/pipeline transforms, tree histograms, classifier
  likelihoods, neural forward/backward products, GP solves, derivative
  operators, and L-BFGS-B objective/gradient evaluations. A mixed CPU/GPU graph
  must expose every transfer and cannot claim full-device execution.
- [ ] Add CUDA kernels for common dense primitives, reductions, activations,
  normalization, scatter/gather, segmented histogramming, sparse products, and
  batched factorizations. Each kernel has a scalar CPU oracle and a noncontiguous
  stride test.
- [ ] Add OpenACC implementations or structured refusals for the same operation
  graph. Backend selection is explicit and does not infer CUDA from compiler
  identity alone.
- [ ] Add resident exact or matrix-free GP prediction and hyperparameter-gradient
  paths, including preconditioned CG and batched LOVE work.
- [ ] Add device Toeplitz transforms, SKI interpolation, sparse variational
  products, and histogram construction one path at a time. Retain a path when a
  correctness-gated release workload reduces median time beyond its recorded
  run-to-run dispersion or reduces peak host memory.
- [ ] Extend the backend ABI to one non-CUDA accelerator runtime after the CUDA
  ownership and error contracts are stable.
- [ ] Add device memory accounting, leak checks, and transfer counters to the
  benchmark harness.

Acceptance: each device kernel agrees with an independent CPU oracle across
boundary shapes and noncontiguous batch sizes. A timed training step performs no
implicit host transfer after residency begins. Repeated create, train, predict,
and destroy cycles return device memory to the baseline. Unsupported devices or
features return a status and leave host state usable. Release reports include
backend, compiler, driver, device, precision, transfer count, and peak device
memory.

### WP7: serialization and distributed execution

- [ ] Define a versioned model schema with type tags, dimensions, dtypes,
  parameter layout versions, kernel trees, basis definitions, preprocessing
  statistics, class order, and optional fitted state.
- [ ] Add save and load procedures for each completed estimator and pipeline.
  Callback bases and user kernel callbacks require registered stable names or an
  explicit serialization refusal.
- [ ] Add trainer checkpoints containing optimizer, schedule, RNG, batch,
  validation, and early-stopping state.
- [ ] Specify compatibility rules for newer readers, reject unknown required
  fields, and provide migrations for every released schema change.
- [ ] Add MPI data-parallel training with deterministic gradient reduction for
  trainer-compatible dense models.
- [ ] Add sharded prediction and matrix-free products for data sets that do not
  fit one rank. Collective failure must return the same status on every rank.
- [ ] Add distributed checkpoint coordination and rank-local temporary files
  with atomic publication of the completed checkpoint.

Acceptance: model round trips preserve parameters, metadata, and predictions
across GNU and NVIDIA compiler builds. Golden files from every supported schema
version remain readable. An interrupted write never replaces the last complete
checkpoint. One-rank and two-rank runs agree within a documented reduction
tolerance, and fixed process counts reproduce results. MPI tests cover empty
shards, uneven final batches, and one-rank failure propagation.

### WP7a: interoperability, serving, and operations

- [ ] Define a stable C ABI for prediction, probability, transformation,
  derivative products, error/status retrieval, and explicit buffer ownership.
  The ABI supports row-major callers without changing the Fortran-native
  internal sample convention.
- [ ] Add import/export adapters for interoperable linear, tree, and neural
  models (ONNX or a documented subset) with a refusal for unsupported operators,
  dtypes, dynamic shapes, custom callbacks, or stochastic state.
- [ ] Add streaming and online inference with bounded workspaces, batch-size
  independent outputs, input schema/version validation, and backpressure or
  refusal when a request cannot fit the configured device memory.
- [ ] Add model cards and training manifests containing data schema hashes,
  feature statistics, class order, objective, seed streams, compiler/toolchain,
  dependency revisions, precision, hardware, and known refusal boundaries.
- [ ] Add reproducibility audit tooling that rebuilds a manifest, replays a
  saved seed/checkpoint, and reports the first divergent parameter, batch,
  derivative, or prediction rather than only a final mismatch.
- [ ] Add security limits for model files, callback registration, dimensions,
  allocation sizes, and integer overflow. Untrusted files are data-only and
  never execute arbitrary Fortran callbacks.

Acceptance: C-ABI and imported models agree with native predictions and
derivatives on independent fixtures. A versioned model can be loaded by a
different supported compiler, and malformed or oversized files fail before
allocation. A serving smoke test measures cold start, warm latency, throughput,
peak memory, and batch-size scaling with the same correctness gate as training.

### WP8: benchmark and parity evidence

- [x] Provide correctness-gated applications for linear regression, MLP, exact
  GP, GP feature products, and approximate or matrix-free GP methods.
- [x] Record release timings, peak RSS, build provenance, external Python
  comparisons, raw CSV files, and plots through `../fortml-bench`.
- [ ] Define one versioned result schema for correctness, train time, predict
  time, peak host and device memory, compiler, flags, dependency revisions,
  hardware, seed, warmup, repetitions, and refusal reason.
- [ ] Add pinned external oracle harnesses for every completed classifier,
  transformer pipeline, boosted tree, trainer, GP derivative, and serialization
  package.
- [ ] Add matched benchmark lanes for scikit-learn preprocessing and
  estimators, XGBoost and LightGBM trees, PyTorch and JAX neural training, and
  GPyTorch/GPflow exact and variational GPs. Each lane uses the same data,
  objective, precision, initialization, stopping rule, and correctness gate.
- [ ] Cover small analytic fixtures, dense tabular data, sparse/categorical
  data, wide features, long sequences, image-like tensors, graph batches,
  derivative observations, and physics trajectories. Record the shape and
  memory footprint rather than comparing only one convenient workload.
- [ ] Add scaling sweeps over samples, features, output count, network width and
  depth, GP inducing points, tree count/depth, batch size, device, and MPI rank.
  Report throughput, latency percentiles, peak host/device memory, transfers,
  compile/warmup time, and energy where a reliable counter is available.
- [ ] Add derivative and hyperparameter-search lanes measuring value, gradient,
  HVP, optimizer evaluation count, convergence, and selected-state quality.
  Compare generated `fortsym`, FortAD, and reference-framework paths with
  operation count and numerical error.
- [ ] Add workload tiers for unit-size correctness, CI smoke runs, single-node
  release measurements, accelerator runs, and multi-rank scaling.
- [ ] Store median and dispersion across repetitions, separate compile and warmup
  costs, and prevent debug profiles from entering performance reports.
- [ ] Define regression thresholds only after two release baselines on the same
  hardware. A threshold failure must retain the raw result and identify the
  changed code and toolchain revisions.
- [ ] Publish one maintained report per parity package with workload definitions,
  correctness tolerances, refused cases, raw-data links, and plots or tables.

Acceptance: a report can be regenerated from raw artifacts without editing the
data. Every timed row has passed its independent correctness gate in the same
build. Missing hardware or infeasible workloads emit parseable refusal records.
The harness records enough provenance to rebuild the tested FortML and sibling
dependencies.

### WP9: physics-consistent, Hamiltonian, and symplectic models

FortML should support scientific models in which the differential equation,
conservation law, or geometric structure is part of the model contract. This
work package is a research track. It becomes an implementation claim only when
the residual, derivative, and long-horizon behavior have independent tests.

The literature establishes several complementary directions:

- [Hamiltonian Neural Networks](https://papers.nips.cc/paper/9672-hamiltonian-neural-networks.pdf)
  parameterize a scalar Hamiltonian and obtain the vector field from the
  canonical symplectic gradient.
- [Symplectic learning for Hamiltonian neural networks](https://arxiv.org/abs/2106.11753)
  analyzes the discretization error introduced by the training integrator and
  motivates training through a symplectic map.
- [Physics-informed neural networks](https://doi.org/10.1016/j.jcp.2018.10.045)
  combine data and differential-equation residuals in one objective. The
  [PIML review](https://arxiv.org/abs/2201.05624) surveys physics-guided,
  physics-informed, and physics-encoded architectures and the different ways
  equations and domain knowledge enter a model.
- [Physics consistency of infinite neural networks](https://ml4physicalsciences.github.io/2023/files/NeurIPS_ML4PS_2023_9.pdf)
  by Sascha Ranftl connects kernels satisfying linear differential constraints
  to the infinite-width neural-network limit. Its finite-width construction is
  an approximation to the limiting GP, so FortML will test seeded ensembles
  against that covariance instead of claiming an exact initializer.
- A forthcoming TU Graz DocDay abstract by Johanna Moser describes the
  [Ghosttasking and Monge-GP direction](https://www.tugraz.at/sites/dsp/docdays/past-docdays/september-2026)
  for physics-informed GPs for linear differential equations, including
  parameter inference outside the constant-coefficient and controllable cases.
- [Symplectic Neural Gaussian Processes](https://www.ijcai.org/proceedings/2024/465)
  combines a GP Hamiltonian with a learned system representation for
  data-efficient Hamiltonian dynamics.
- [Lagrangian Neural Networks](https://arxiv.org/abs/2003.04630),
  [SympNets](https://arxiv.org/abs/2001.03750), and
  [symplectic recurrent neural networks](https://arxiv.org/abs/1909.13334)
  provide complementary structure-preserving architectures.
- [Direct Poisson neural networks](https://arxiv.org/abs/2305.05540) extend the
  target beyond nondegenerate canonical symplectic systems to Poisson systems.

The project-specific symplectic-GP and Hamiltonian/ANN benchmark results from
the FortML authors and Katharina Rath are a required pinned reference set. The
roadmap records the interface and reproduction work without treating private
results as an external literature claim.

#### WP9a: physics contracts and autodiff products

- [ ] Define a `physics_constraint_t` callback with residual value, JVP, VJP,
  and HVP products. Callbacks declare state, parameter, coordinate, and time
  layouts, units, boundary masks, and reduction weights.
- [ ] Add data, PDE/ODE residual, initial or boundary, conservation, and
  symplectic-form terms to one composable objective. Include nondimensionalizing
  transforms and named diagnostics for every term.
- [ ] Add collocation and trajectory samplers with seeded random, adaptive
  residual, boundary, and event-aware policies. A sampler records the points it
  emitted so a run can be reproduced exactly.
- [ ] Add residual derivatives through `fortad` or an equivalent analytic
  product path. The current `fortad` `main` checkout is the baseline. Use
  `fortsym` to derive and emit a kernel when symbolic differentiation produces a
  smaller proven expression, then verify the emitted code against the symbolic
  identity and an independent numerical oracle. A physics objective must not
  rely on finite differences in its production training path.
- [ ] Record the `fortad` and `fortsym` revisions, proof strength, operation
  count, and fallback reason in generated-kernel provenance.

The repository snapshot used for this roadmap resolves `fortad` `main` at
`8f46509` and `fortsym` `main` at `58a0e06`. Future derivative work must refresh
both checkouts before deciding that a product is unavailable.

#### WP9b: Hamiltonian, Lagrangian, and symplectic networks

- [ ] Add `hamiltonian_mlp_t` with scalar H(q,p), canonical J, symplectic
  gradient, optional learned skew structure matrix, and parameter/input
  products. Promote that matrix to a Poisson structure only after skew
  symmetry and the Jacobi identity have independent tests.
- [ ] Add `lagrangian_mlp_t` with Euler-Lagrange residuals, mass-matrix checks,
  and a refusal for a singular velocity Hessian. Positive definiteness is an
  additional requirement only for a separable mechanical mass metric.
- [ ] Add SympNet and symplectic recurrent map variants with architecture-
  specific composition certificates and a testable symplectic Jacobian. A
  generating-function certificate is required only for an architecture that
  explicitly uses one.
- [ ] Add differentiable symplectic Euler, Verlet, and higher-order splitting
  integrators for separable or otherwise splittable Hamiltonians. General
  Hamiltonians require an applicable implicit symplectic method or an explicit
  refusal. Training can differentiate through the map, while inference reports
  the integrator and step size used.
- [ ] Add gauge handling for additive constants in H and for the Lagrangian
  equivalence `L -> L + dF(q,t)/dt`, canonical versus noncanonical coordinates,
  and optional noisy derivative observations.
- [ ] Add conservation, reversibility, volume or symplectic-form error, and
  long-horizon trajectory metrics to the benchmark schema.

#### WP9c: physics-consistent and symplectic GPs

- [ ] Generalize derivative observations from coordinate gradients to registered
  linear differential operators. The operator registry provides value, adjoint,
  and mixed-operator covariance products and rejects unsupported smoothness.
- [ ] Add physics-consistent kernels and mean functions for linear ODE/PDE
  constraints, boundary conditions, and Green-function constructions. Include
  Ghosttasking and Monge-GP prototypes behind explicit experimental modules.
- [ ] Add symplectic GP priors for scalar Hamiltonians and vector fields.
  Construct `f = J grad(H)` for canonical systems or `f = P grad(H)` for
  Poisson systems, where the structure tensor is the antisymmetric object.
  Expose joint covariance for values, derivatives, and cross-components, and
  test symplectic, divergence, or Jacobi properties as applicable.
- [ ] Add trainable equation parameters, operator hyperparameters, and noise
  blocks to the optimizer-ready parameter registry. FortOpt L-BFGS-B and
  bounded multi-start diagnostics are the reference training path.
- [ ] Compare physics-consistent GP posterior means, covariances, and recovered
  parameters against an independently assembled dense operator GP on small
  fixtures.

#### WP9d: GP-limit, PCA, and linear-optimum initialization

- [ ] Implement NNGP covariance propagation for the supported MLP depth,
  activations, weight and bias priors, and widths, following the [deep-network
  GP correspondence](https://arxiv.org/abs/1711.00165). On a user design set,
  estimate covariance from a seeded finite-width ensemble and report its error,
  mean, and variance calibration against the limiting kernel.
- [ ] Add three separate MLP initialization contracts: a sampled prior draw,
  deterministic fitting of a GP posterior mean, and last-layer kernel-ridge
  initialization. Each records the kernel, architecture, width, seed, design
  set, and solve tolerance, and states whether it promises a mean fit or a
  covariance approximation.
- [ ] Add PCA initialization for linear autoencoders, following the
  [principal-component initialization proposal](https://doi.org/10.1007/978-3-030-30484-3_14).
  The encoder and decoder
  use the selected principal subspace, with centering, whitening, rank, and
  sign conventions recorded. The reconstruction oracle must match the PCA
  projection to numerical tolerance.
- [ ] Add GP or basis-map initialization for nonlinear autoencoders and VAEs.
  The linear optimum is the starting point, while nonlinear layers begin with
  an identity or contractive perturbation whose reconstruction and Jacobian
  products are tested.
- [ ] Add physics-consistent and symplectic initializers that preserve the
  declared operator or form at initialization. Compare convergence from random,
  Xavier/He, PCA, NNGP, and GP-posterior starts under identical seeds and
  optimizer budgets.

#### WP9e: scientific benchmark matrix

- [ ] Add analytic harmonic oscillator, pendulum, Kepler/two-body, and
  Hénon-Heiles workloads. Compare standard MLP, HNN, LNN, SympNet/SRNN, exact
  GP, symplectic GP, and GP-initialized networks.
- [ ] Add graph differential-operator workloads for irregular domains, using
  the [physics-informed graph-network construction](https://arxiv.org/abs/2205.08332)
  and a manufactured graph oracle for node and edge residuals.
- [ ] Add Poisson, heat, Burgers, and wave equation workloads with manufactured
  solutions. Compare PINN residual training, physics-consistent GP priors,
  derivative observations, and a numerical solver reference.
- [ ] Add Ghosttasking and Monge-GP inverse-problem fixtures once the public
  equation definitions and reference implementations are pinned.
- [ ] Record short- and long-horizon trajectory error, energy drift, symplectic
  defect, residual and boundary norms, parameter recovery, posterior calibration,
  wall time, memory, and optimizer evaluations. Refused combinations remain
  parseable benchmark rows.
- [ ] Publish a reproducible comparison against the author's symplectic-GP and
  Hamiltonian/ANN benchmark results, including the exact data generation,
  integrator, step size, model width, seed, and stopping criteria.

Acceptance: every model has an analytic or manufactured-solution oracle,
finite-difference checks for the public products, and a structure check for
each declared invariant. Symplectic tests measure the Jacobian form defect.
Energy tests distinguish true-system energy error from learned-Hamiltonian
drift. Forced, dissipative, and time-dependent systems are not required to
conserve energy. GP and finite-network initializations reproduce their declared
design-set mean, covariance approximation, or projection. Long-horizon
conclusions use the same integrator and sampling budget for every baseline, and
every release row links to raw data and the pinned reference.

## Benchmark evidence

The maintained reports and their raw artifacts are in `../fortml-bench/results`:

- [`MODEL_WORKLOADS.md`](../fortml-bench/results/MODEL_WORKLOADS.md), backed by
  `model_workloads.csv`, `exact_gp_workloads.png`, and `mlp_workloads.png`.
- [`GP_FEATURES.md`](../fortml-bench/results/GP_FEATURES.md), backed by
  `gp_features.csv` and `gp_features.png`.
- [`CLASSIFICATION.md`](../fortml-bench/results/CLASSIFICATION.md), backed by
  `classification_workloads.csv` with FortML and scikit-learn provenance.
- Corrected GRBCM evidence in `scalable_gp_grbcm_corrected.csv` and
  `scalable_gp_grbcm_corrected_train_seconds.png`.
- Current partition and dimension evidence in `scalable_gp_clustered.csv` and
  `scalable_gp_dimension_current.csv`.

Those reports contain the workload definitions, compiler flags, hardware records,
correctness gates, raw timings, and plots. This roadmap contains no copied
timing numbers.

GRBCM results produced before the communication-set and enhanced-expert
correction are superseded. Use the corrected CSV and plot above for GRBCM
claims.

## Completion gate for later changes

A code item is complete when its implementation, public documentation, focused
tests, and an independent behavioral oracle agree. Run bare `fo` before
handoff. Compiler-sensitive changes also require the relevant compiler lane.

A performance claim additionally needs a release build, a recorded workload and
toolchain, a correctness result, raw data, and a plot or table in
`../fortml-bench/results`. Timings from the checked debug profile are invalid as
performance evidence.

## Delivery order

Benchmark and documentation work ships with each implementation slice. The
dependency order for the remaining code is:

1. Freeze the clean-break data, status, parameter-tree, estimator, transform,
   objective, and derivative contracts. Update all in-tree callers together.
   do not create compatibility aliases for a superseded layout.
2. Finish shared classification labels, weights, metrics, probabilities, and
   validation. Add binary/multinomial linear, neural, GP, tree, multilabel, and
   ordinal adapters against those contracts.
3. Implement fitted preprocessing, basis/pipeline DAGs, sparse/categorical
   views, leakage-safe splitters, and metrics/search. Adapt linear models before
   using the pipeline in every later benchmark.
4. Build the production trainer and parameter-tree infrastructure: deterministic
   batches, complete optimizer/schedule set, derivatives, mixed precision,
   callbacks, checkpoints, and one MLP/GP adapter with a resumption oracle.
5. Implement scikit-learn estimator families in small independent slices, then
   exact CART, random/extra forests, histogram boosting, and XGBoost/LightGBM-
   style growth policies with external correctness and performance lanes.
6. Add GP mean/ARD/likelihood families, full hyperparameter and implicit
   derivatives, GPyTorch/GPflow parity, and bounded FortOpt L-BFGS-B training.
   Extend every completed objective to resident GPU products before claiming
   device parity.
7. Add model schemas, C ABI, serving, MPI/sharded execution, and reproducibility
   manifests once in-memory trainer state and ownership rules are stable.
8. Add physics constraints, Hamiltonian/Lagrangian/symplectic models, operator
   GPs, and Ghosttasking/Monge-GP prototypes behind residual and structure
   oracles. Use current FortAD main and FortSym-generated kernels where proven.
9. Add NNGP, PCA, autoencoder, and physics-consistent initializers as explicit
   experiments before making any initializer a default. Expand release
   benchmarks after every slice, and run the `ifx` compiler lane when available.
