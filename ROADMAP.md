# fortml roadmap

Verified on 2026-08-07. Interfaces are documented in
[`docs/API.md`](docs/API.md), examples in [`docs/EXAMPLES.md`](docs/EXAMPLES.md),
and implementation limits in [`docs/DESIGN.md`](docs/DESIGN.md) and
[`docs/ML_ARCHITECTURE.md`](docs/ML_ARCHITECTURE.md).

## Verification

| Compiler | Command | Result |
| --- | --- | --- |
| GNU Fortran | `fo` | Static, build, and lint checks passed. The fresh 2026-08-07 run passed all 79 tests. See [`verification/fortml-gfortran.txt`](verification/fortml-gfortran.txt). |
| NVIDIA HPC SDK | `FO_FC=nvfortran fo` | Static and lint checks passed in the recorded compiler lane. The checked-in NVIDIA log predates the latest 79-test GNU run. See [`verification/fortml-nvfortran.txt`](verification/fortml-nvfortran.txt). |
| Intel LLVM Fortran | `ifx` | Compiler unavailable in the verification environment. Not tested. |

The checked-in compiler logs above are older compiler snapshots. A fresh GNU
Fortran `fo` run on 2026-08-07 passed static analysis, the build, all 79 tests,
and lint. The fresh run includes implementation work whose compiler logs have
not yet replaced those snapshots. NVIDIA compiler coverage therefore remains
an explicit older-build result.

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
| Linear regression and generalized linear models | Linear regression is implemented. Logistic and softmax support sample and positive sorted-class weights | OLS, weighted/ridge/lasso/elastic-net, robust, quantile, Poisson/Gamma/Tweedie, multinomial, calibrated and regularized classifiers with shared solver and derivative contracts |
| Feature transforms and basis maps | Polynomial, Fourier, radial, B-spline, callback bases, standard/min-max scalers, integer categorical one-hot encoding, horizontal/sequential basis pipelines, and a fitted basis-to-linear estimator are implemented | Sparse/categorical feature views, feature names, DAG pipelines, leakage-safe cross-validation, differentiable basis hyperparameters |
| Nearest-neighbor and margin methods | Dense exact kNN classifier | KD-tree or ball-tree search, radius neighbors, sparse inputs, linear/kernel SVM and SVR, calibrated probabilities, and differentiable soft-neighbor policies |
| Trees and ensembles | Partial | Deterministic finite-only regression stumps, weighted depth-limited CART regression and classification, squared-loss stump boosting, and exact depth-limited second-order squared/logistic boosting are implemented. Exact XGBoost-style trees now support explicit NaN rejection, learned default directions, and forced-left/right routing; forests, histograms, ranking, monotonic and interaction constraints remain planned |
| Clustering and unsupervised learning | Centered dense `pca_t` is implemented with deterministic SVD signs, rank selection, whitening, reconstruction, variance metadata, and fixed-state input products | Incremental/randomized/sparse/kernel PCA, ICA, NMF, k-means/minibatch k-means, Gaussian mixtures/EM, density and graph clustering, manifold methods, outlier detection, matrix factorization, and density metrics |
| Neural networks | MLP/BNN/VAE/RNN primitives, a separable Hamiltonian MLP, selected products, deterministic MLP Adam/AdamW/Adagrad/RMSprop/SGD training, a bounded full-batch MLP L-BFGS-B path, and an in-memory resumable optimizer checkpoint exist | A production module/parameter tree, all common activations and losses, convolution/attention/sequence/graph extensions, mixed precision, distributed training, compile/fusion, and serialized/distributed trainers |
| Gaussian processes | Exact, derivative, sparse, structured and local variants are partial-to-implemented. Exact fitted GPs and binary/shared-kernel one-vs-rest Laplace classifiers have bounded FortOpt L-BFGS-B adapters | GPyTorch/GPflow-style kernels, likelihoods, multitask/batch shapes, exact/variational/lazy inference, derivative operators, constraints, calibration, coupled multiclass GP classification, evidence-corrected and likelihood-parameter training |
| Derivatives | Exact GP and selected neural/kernel products exist | Value/JVP/VJP/HVP and implicit/hypergradients for every declared parameter/input path, including preprocessing, likelihood, optimizer/search variables, and device kernels |
| Model selection and metrics | Benchmark-specific checks exist | Shared metrics, splitters, cross-validation, calibration, grid/random/Bayesian/differentiable search, nested validation, and leakage/refusal checks |
| Persistence and serving | Missing public contract | Versioned state dictionaries, safe model/trainer serialization, compiler-independent metadata, streaming inference, batching, and reproducible deployment manifests |
| GPU and scale-out | Operator-specific OpenACC/CUDA paths | Complete resident CPU/CUDA/OpenACC training and inference for supported estimators, mixed precision, multi-GPU/MPI sharding, transfer accounting, and deterministic reductions |
| Performance evidence | Several model/GP lanes exist | Matched correctness-gated comparisons with scikit-learn, XGBoost/LightGBM, PyTorch/JAX, GPyTorch/GPflow, and published hardware/toolchain provenance |

### Parity inventory and release gates

This inventory prevents a name-only implementation from being counted as
parity. A family moves to **implemented** only when fit/predict (or
transform), weighting, arbitrary labels/categories, packed state, declared
derivatives, refusal behavior, and an independent benchmark oracle all exist.

| Reference family | FortML coverage today | Remaining release gate |
| --- | --- | --- |
| scikit-learn linear/GLM | Dense linear regression, logistic/softmax, bounded logistic and MLP L-BFGS-B | Ridge/lasso/elastic-net/robust/quantile/Poisson/Gamma/Tweedie, SGD estimators, solver parity, calibration, multioutput and partial-fit contracts |
| scikit-learn Naive Bayes | GaussianNB, BernoulliNB, MultinomialNB, ComplementNB, CategoricalNB with weighted sorted categories and unknown-category policy | sparse counts, calibrated and incremental variants |
| scikit-learn neighbors/margins | Dense exact `fortml_knn_classifier` with deterministic ties and explicit discrete-derivative refusal | Radius search, KD/ball trees, sparse inputs, linear/kernel SVM/SVR, one-class SVM, and smooth input/parameter products |
| scikit-learn trees/ensembles | Stumps, weighted CART, squared boosting, XGBoost-style second-order binary/multiclass lanes | Random/extra forests, bagging, AdaBoost, histogram/leaf-wise growth, missing/categorical/ranking/monotonic/interaction constraints, staged and warm-start APIs |
| scikit-learn unsupervised | Basis maps, centered dense PCA, validation splitters, variational primitives | Incremental/randomized/sparse/kernel PCA, ICA/NMF/TruncatedSVD, k-means/GMM/density/manifold/outlier methods, sparse/categorical preprocessing, model persistence |
| PyTorch/JAX neural core | Dense MLP, classifier, BNN, VAE, RNN, Hamiltonian MLP; Adam, AdamW, Adagrad, RMSprop, and SGD momentum/Nesterov; exact fixed full-batch SGD learning-rate/L2 and AdamW learning-rate/L2/weight-decay trajectory hypergradients | Optimizer-trajectory schedules and stochastic/device hypergradients, complete loss/activation/module tree, convolution/attention/sequence/graph models, AMP, compile/fusion, distributed and device-resident train state |
| GPyTorch/GPflow | Exact, derivative-observation, sparse/local/SKI/structured GP primitives; Laplace binary/OVR GP classification | Kernel/likelihood/constraint/batch-shape parity, variational categorical/count likelihoods, multitask, operator-valued derivatives, implicit hypergradients, serialization and resident GPU training |
| XGBoost/LightGBM | Exact depth-limited squared/logistic Newton trees, binary/OVR multiclass staged predictions, margins, and gain/weight/cover diagnostics with explicit NaN policies (`error`, learned, forced-left/right) | Histogram quantiles, categorical/quantile/ranking objectives, DART, monotonic/interaction constraints, and distributed training |
| Differentiability and search | Capability-specific JVP/VJP/HVP products, FortOpt L-BFGS-B for selected objectives, and an exact fixed-trajectory MLP hypergradient over log learning rate/L2 | Complete derivative matrix for every declared parameter/input/hyperparameter, stochastic/device optimizer hypergradients, implicit differentiation, and refusal rather than hidden finite differences |
| Device and performance | OpenACC/native CUDA operator lanes plus explicit device control-plane contract | Resident model/optimizer/batch state for every supported estimator, CPU parity, transfer/memory accounting, mixed precision, and matched PyTorch/JAX/GPyTorch/XGBoost evidence |

The inventory deliberately distinguishes an implemented algorithm from an
implemented *workflow*. For example, an XGBoost-style Newton tree without
histograms, missing-value routing, constraints, staged prediction, and a
release benchmark is a useful exact baseline but not XGBoost parity. The same
rule applies to a GP kernel without likelihood constraints, batch shapes,
train-state serialization, and derivative/hyperparameter products.

## Complete parity gap register

This register is the implementation index for the long-term target. A checked
item has a public contract, an independent behavioral oracle, refusal tests,
documentation, and a benchmark record. An unchecked item remains open even
when a lower-level primitive already exists.

### Estimators and supervised workflows

- [x] Linear, logistic, softmax, one-vs-rest, GaussianNB, BernoulliNB,
  MultinomialNB, ComplementNB, CategoricalNB, CART, MLP, binary Laplace GP,
  one-vs-rest Laplace GP, and exact XGBoost-style squared/logistic estimators.
- [ ] Ridge, lasso, elastic-net, Huber, quantile, Poisson, Gamma, Tweedie,
  ordinal, multilabel, multioutput, and partial-fit estimators with the shared
  parameter registry and sample-weight contract.
- [x] Dense k-nearest-neighbor classification with deterministic ties, uniform
  or inverse-distance voting, optional sample weights, and explicit
  nondifferentiable neighbor-selection boundaries.
- [ ] Radius-neighbor search with deterministic ties,
  brute/KD-tree/ball-tree backends, sparse inputs, leave-one-out scoring, and
  explicit nondifferentiable neighbor-selection boundaries.
- [ ] Linear, kernel, one-class, and ranking SVM/SVR estimators with bounded
  solvers, probability calibration, support-vector metadata, and input/parameter
  products for smooth regions.
- [ ] Calibration workflows: Platt/sigmoid, isotonic, temperature scaling,
  reliability diagrams, class weighting, and calibration-aware cross-validation.
- [ ] Random forests, extra trees, bagging, AdaBoost, random patches, and
  histogram gradient boosting with staged prediction, warm starts, feature
  importance, missing/categorical values, monotonic constraints, and permutation
  importance.
- [x] Binary and one-vs-rest XGBoost-style staged predictions, cumulative
  margins, and gain/weight/cover feature-importance diagnostics.
- [ ] XGBoost and LightGBM parity beyond exact growth: quantile sketches,
  histogram bins, leaf-wise growth, ranking objectives, DART/GOSS/EFB,
  categorical partitions, monotonic and interaction constraints, distributed
  workers, early stopping, model dumps, and staged/tree-contribution APIs.

### Gaussian processes and probabilistic models

- [x] RBF, Matérn 1/2, 3/2, and 5/2, linear, constant, white-noise, user-formula,
  sum, and product kernels with exact value and selected derivative products.
- [ ] Complete kernel catalog: periodic, spectral mixture, rational quadratic,
  polynomial, cosine, locally periodic, change-point, neural-network, graph,
  string, and operator-valued kernels with compositional parameter metadata.
- [ ] Likelihood catalog: Gaussian, Bernoulli, categorical, multinomial,
  Poisson, count, heteroscedastic, censored, ordinal, Student-t, and warped
  likelihoods with stable links, constraints, and declared derivative modes.
- [ ] Exact GP workflow parity: batched/multitask shapes, mean functions,
  priors, constraints, lazy solves, preconditioners, stochastic log-determinants,
  predictive roots, fantasy updates, online updates, and serialized train state.
- [ ] Variational and scalable inference: whitened/unwhitened SVGP, natural
  gradients, stochastic ELBO minibatches, interdomain features, SKI/KISS-GP,
  local experts, deep GPs, variational classification, and distributed inducing
  points.
- [ ] Derivative observations for every supported smooth kernel, mixed orders,
  vector fields, Hessian observations, operator-valued outputs, analytic
  third-order query products, and covariance products over value/derivative
  blocks.
- [ ] GP classification optimization beyond the implemented binary/shared-kernel
  adapters: likelihood parameters, independent per-class blocks, multiclass
  likelihoods beyond one-vs-rest, calibration, Laplace evidence corrections,
  variational classification, and exact JVP/VJP/HVP products through the
  selected inference state.

### Neural networks and training state

- [x] Dense MLP, classifier, BNN, VAE, vanilla RNN, and separable Hamiltonian
  MLP primitives with selected value/JVP/VJP/HVP products and deterministic
  checkpointable Adam, AdamW, Adagrad, RMSprop, and SGD training.
- [ ] A general module tree with explicit parameter/state buffers, named
  submodules, cloning, freezing, tied weights, parameter groups, buffers,
  hooks, train/eval modes, and a stable flat-registry projection.
- [ ] Convolution, transposed convolution, pooling, normalization, dropout,
  embeddings, attention, transformers, residual blocks, recurrent LSTM/GRU,
  temporal convolutions, graph message passing, and neural operators.
- [ ] Full activation and loss catalog: sigmoid, softmax, GELU, SiLU, ELU,
  softplus, leaky/parametric ReLU, log-softmax, cross-entropy, focal,
  multilabel BCE, Poisson, Huber, quantile, contrastive, triplet, CTC, and
  physics residual losses, each with explicit derivative and refusal contracts.
- [x] Production RMSprop with centered/uncentered running statistics, optional
  momentum, exact checkpoint/resume state, and an independent recurrence oracle.
- [ ] Production optimizers and schedules: L-BFGS/L-BFGS-B, Adafactor,
  Lion, RAdam, AMSGrad, cosine/one-cycle/warmup/plateau schedules, gradient
  accumulation, clipping, EMA, decoupled regularization, and parameter groups.
- [ ] Exact fixed-trajectory and implicit hypergradients through all supported
  optimizers, schedules, batch cursors, clipping, weight decay, validation,
  early stopping, and optimizer state. Every unsupported stochastic or device
  path must return a typed refusal.
- [ ] Mixed precision with master weights, loss scaling, overflow recovery,
  deterministic reduction modes, activation checkpointing, truncated BPTT,
  compile/fusion, distributed data/model parallelism, and resumable serialized
  checkpoints with schema migration.
- [ ] Physics-informed and physics-consistent neural models: PINNs, HNNs,
  Lagrangian and symplectic networks, constrained neural ODEs, Hamiltonian
  residuals, conservation/invariant diagnostics, Ghost Tasking workloads, and
  manufactured-PDE training with long-horizon trajectory gates.

### Differentiation, initialization, and composition

- [x] Capability-specific analytic, FortAD-generated, FortSym-generated,
  JVP, VJP, HVP, and explicit-refusal paths for the current supported models.
- [ ] A complete derivative capability matrix over every model parameter,
  input, basis hyperparameter, kernel hyperparameter, likelihood parameter,
  optimizer variable, schedule variable, validation variable, and device
  transfer counter. Products must include adjoint identities and finite-
  difference checks where a trusted analytic oracle is unavailable.
- [ ] Implicit differentiation through linear solves, fixed points, Laplace
  modes, variational optima, constrained tree policies, and optimizer fixed
  points, with FortOpt L-BFGS-B consuming the same parameter registry.
- [ ] Basis and pipeline DAGs with named features, sparse views, missing-value
  policies, fit/transform leakage guards, cross-validation cloning, graph
  serialization, static device lowering, and parameter-group hypergradients.
- [ ] Polynomial, spline, Fourier, radial, and GP bases as interchangeable
  initializers for linear models, autoencoders, NNGP/NTK finite networks,
  physics-consistent networks, symplectic networks, and GP posterior starts.
- [ ] PCA, linear autoencoder, NNGP, NTK, Xavier/He, GP-posterior, and
  physics-informed initialization benchmarks with identical seeds, residual or
  reconstruction oracles, and convergence comparisons.

### Devices, persistence, and benchmark gates

- [ ] Resident CPU/CUDA/OpenACC execution for every supported estimator and
  optimizer, including model/optimizer/batch state, true transfer accounting,
  deterministic reductions, and matched float32/float64 behavior.
- [ ] CUDA kernels and generated FortSym/FortAD products for all declared
  derivative paths, with device-side JVP/VJP/HVP and no hidden host callbacks.
- [ ] Versioned state dictionaries, compiler-independent serialization,
  checksums, safe loading, streaming/batched inference, model cards, and
  deployment manifests for models, pipelines, kernels, and trainer state.
- [ ] Correctness-gated benchmark matrices against scikit-learn, XGBoost,
  LightGBM, PyTorch, JAX, GPyTorch, GPflow, Flux, and Lux over matched data,
  precision, initialization, stopping, device, memory, compile, transfer,
  latency, throughput, and energy measurements.
- [ ] Public benchmark fixtures for analytic toy problems, dense tabular data,
  sparse/categorical inputs, wide data, long sequences, image-like tensors,
  graph batches, derivative observations, and physics trajectories. Every
  unavailable dependency or unsupported backend remains a parseable refusal
  row with a reason.

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
- [x] Extend validated user-formula kernels with a forward value/gradient/mixed
  Hessian stack, so mixed value/first-derivative GP observations and their
  kernel-parameter JVPs use an analytic rule rather than an implicit refusal.
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

### Derivative and initialization policy

The derivative capability is a matrix of declared products, not a promise that
every estimator is fully differentiable. Current products cover the linear,
basis, scaler, dense MLP, exact GP, and mixed value/first-derivative GP paths
listed in [README.md](README.md). CART and boosting products are piecewise and
refuse split-boundary directions. BNN, VAE, RNN, classifier HVP, and most
approximate-GP paths remain partial. Query-input derivative-GP products and
the derivative-GP likelihood HVP are documented deterministic finite
differences. They are not presented as analytic or generated third-order
products.

Derivative selection follows this order:

1. Use an analytic implementation or a `fortsym`-derived expression when its
   identity proof and operation count pass the independent oracle.
2. Use `fortad` source transformation for general differentiable model code and
   retain the explicit reference product beside the generated source.
3. Expose a deterministic finite-difference product only when the public API
   labels it as such and the production objective does not silently depend on
   it.
4. Return a structured refusal when the smoothness, shape, device, or operator
   contract is missing.

Accepted generated derivative artifacts will record the FortSym and FortAD
revisions, proof strength, operation count, source hash, and fallback reason.
The benchmark harness compares generated and reference products wherever both
exist.

The same provenance rule applies to initialization. Standard Xavier/He,
PCA/linear-autoencoder, NNGP/NTK, and GP-posterior starts are separate
contracts. Physics-informed, Hamiltonian, symplectic, and autoencoder
initializers must report their residual, invariant defect, reconstruction, or
covariance error on the declared design set. A finite-width network is not
claimed to equal its infinite-width GP.

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
the kernel, structured, and sparse operations documented above. The historical
`nvfortran` verification establishes compiler compatibility for the older
30-test snapshot. The newer tests, including Bernoulli Naive Bayes, are covered
by the GNU run only in this environment.
It does not establish that every model trains or predicts entirely on a GPU.

## Parity work packages

The source inventory is dated 2026-08-07.

| Work package | State | Implemented baseline | Package exit |
| --- | --- | --- | --- |
| Classification | Partial | `fortml_logistic_regression` and `fortml_softmax_regression` provide binary and multinomial integer-label fitting with sample-weighted reductions, `fortml_ovr_logistic_classifier` adds deterministic one-vs-rest binary logistic fits with normalized probabilities and parameter products, `fortml_gaussian_naive_bayes` adds weighted Gaussian class moments and input/parameter probability products, `fortml_bernoulli_naive_bayes` adds weighted relaxed-[0,1] Bernoulli features, positive smoothing, packed state, and input/parameter probability products, `fortml_multinomial_naive_bayes` adds weighted relaxed nonnegative counts, token-mass smoothing, packed state, and input/parameter probability products, `fortml_complement_naive_bayes` adds weighted complement distributions, optional weight normalization, packed state, and input/parameter probability products, `fortml_knn_classifier` adds deterministic dense uniform or inverse-distance neighbors with explicit discrete derivative refusals, `fortml_logistic_training` adds an exact weighted logistic objective with parameter/L2 gradients and HVPs plus bounded FortOpt L-BFGS-B, `fortml_cart_classifier` adds deterministic weighted Gini/entropy trees and leaf probabilities, `fortml_mlp_classifier` adds deterministic multiclass logits training with Adam and sample/class weights, `fortml_gp_classification` adds binary Laplace logistic/probit inference plus an exact mode-envelope kernel gradient, `fortml_gp_multiclass_classification` adds packed one-vs-rest envelope gradients, and shared metrics cover accuracy, top-k, balanced accuracy, confusion, precision/recall/F1, Brier, binary Matthews, weighted accuracy, log loss, and expected/maximum calibration error. | Binary and multiclass linear, tree, neural, GP, and boosted-tree classifiers share label, probability, weighting, and metric conventions. |
| Estimator contracts, pipelines, and bases | Partial | `basis_map_t`, horizontal and sequential basis pipelines, fitted standard/min-max scalers with input JVPs, `basis_linear_regression_t`, row-oriented sample conventions, status objects, and the parameter registry are public. | Fitted transformers and estimators compose without data leakage, expose routed parameters, and run through cross-validation. |
| Tree boosting | Partial | `decision_stump_t`, weighted depth-limited `cart_regressor_t` and `cart_classifier_t`, squared-loss `gradient_boosting_regressor_t`, `xgboost_t`, and `xgboost_multiclass_t` provide deterministic exhaustive split products. The CART lanes have weighted squared-error or Gini/entropy criteria, depth and leaf constraints, fixed feature/threshold tie ordering, piecewise input JVP/refusal for regression, and finite-only probability/prediction paths for classification. The XGBoost-style lane has exact depth-limited squared/logistic gradients, Hessians, regularized gains, recursive Newton leaves, tree-depth/node diagnostics, binary and one-vs-rest multiclass probabilities, staged predictions/margins, and gain/weight/cover feature importance. | Regression and classification trees support deterministic histogram boosting, validation-based stopping, missing values, deeper growth, and model persistence. |
| Training infrastructure | Partial | Model-specific gradients, exact MSE+L2 neural HVPs including the L2 mixed hyperparameter block, `fortopt_adam` and FortOpt SGD momentum/Nesterov integration, AdamW with decoupled decay, Adagrad with an explicit accumulated-square state, RMSprop with centered/uncentered statistics and optional momentum, deterministic seeded batch cursors, per-update learning-rate callbacks, norm clipping, sample-weighted gradient accumulation, validation/early stopping, resumable optimizer state, a fixed-trajectory learning-rate/L2 hypergradient, natural-gradient seams, and seeded variational draws exist. | One trainer owns batches, optimizer state, schedules, clipping, validation, early stopping, callbacks, and resumable state for every model with a completed trainer adapter; stochastic and device-resident optimizer hypergradients remain open. |
| GP derivatives and hyperparameters | Partial | Exact GP likelihood and prediction products include parameter gradients and HVPs. Mixed value and first-derivative observations can be fitted and predicted. | Exact, derivative, multi-output, sparse, and matrix-free GP families expose documented trainable parameters, scalar objectives, parameter gradients, and train-state adapters. |
| GPU and device execution | Partial | Kernel, structured, and sparse operator products have selected OpenACC or CUDA paths, including resident CG for kernel operators. `fortml_device` now gives callers an explicit CPU/CUDA selector, runtime capability probe, backend identity, residency ownership metadata, transfer counters, and structured CUDA refusal. | Supported training and prediction workflows keep model, optimizer, and batch state resident on a selected device and have CPU parity tests. |
| Serialization and distributed execution | Missing | No public model-file or distributed-execution contract exists. | Versioned model and trainer files round-trip across supported compilers, and MPI training or inference agrees with a one-rank oracle. |
| Benchmark coverage | Partial | Correctness-gated model and GP applications feed release harnesses in `../fortml-bench`; current release lanes include Bernoulli/Multinomial/ComplementNB, integer one-hot encoding, MLP SGD/Nesterov/AdamW, fixed-trajectory MLP hypergradients, differentiable imputation, basis/pipeline, exact/approximate GP, and boosting workloads. | Every completed parity package has a pinned external oracle, release timings, memory measurements, provenance, raw data, and a maintained report. |

### WP1: classification

- [x] Define one public class-label contract. Classes have a deterministic order,
  predicted labels use that order to break ties, and probability matrices have
  one column per class.
- [x] Add binary logistic regression with an intercept, L2 regularization,
  `fit`, `decision_function`, `predict_proba`, and `predict`.
- [x] Add nonnegative sample weights to binary logistic and multinomial softmax
  fits while preserving the documented positive-weight-mass reduction and
  class-label contract. Add positive sorted-class weights and combine them
  with sample weights in linear and MLP classifiers.
- [x] Add multinomial softmax regression with a numerically stable log-sum-exp
  objective, sorted integer labels, and the shared sample-weight reduction.
- [x] Add packed coefficient/intercept parameters plus input-and-parameter JVP
  and VJP products for binary logistic and multinomial softmax scores and
  probabilities. Products validate finite tangents/cotangents, preserve the
  sorted class convention, and refuse unfitted, malformed, or nonsmooth paths.
  Classifier HVPs remain a separate second-order contract.
- [x] Add `logistic_training_objective_t` and
  `logistic_optimize_lbfgsb` for fitted binary logistic models. The weighted
  objective exposes exact packed-parameter gradients, an optional L2
  hyperparameter block, mixed parameter/L2 HVPs, and explicit FortOpt bounds.
  independent value, gradient, HVP, optimizer, and refusal oracles cover the
  contract. Softmax and OVR objective adapters remain separate follow-up work.
- [x] Add a multiclass `mlp_classifier_t` adapter with a logits layer, stable
  softmax cross-entropy, deterministic Adam, sorted integer labels, probability
  normalization, sample/class weighting, and a packed parameter-gradient
  product. Binary, multilabel,
  ordinal, and variational GP adapters remain separate contracts.
- [x] Add a deterministic one-vs-rest logistic wrapper with sorted class
  ordering, first-max tie handling, shared sample/class weights, normalized
  positive probabilities, packed parameter metadata, and input/parameter
  JVP/VJP products. One-vs-one wrappers and shared preprocessing ownership
  remain open.
- [x] Add deterministic Gaussian Naive Bayes with weighted class moments,
  sorted arbitrary integer labels, empirical or explicit priors, global
  variance smoothing, packed means/variances/priors, stable log-probabilities,
  and independent input/parameter JVP/VJP/refusal oracles.
- [x] Add differentiable Bernoulli Naive Bayes with relaxed `[0,1]` features,
  sorted arbitrary integer labels, positive Laplace/Beta smoothing, empirical
  or explicit priors, sample/class weights, packed parameters, stable
  log-probabilities, input/parameter JVP/VJPs, and independent analytic,
  finite-difference, adjoint, and refusal oracles.
- [x] Add differentiable Multinomial Naive Bayes with relaxed nonnegative real
  counts, sorted arbitrary integer labels, positive token-mass smoothing,
  empirical or explicit priors, sample/class weights, packed parameters, stable
  log-probabilities, input/parameter JVP/VJPs, and independent analytic,
  finite-difference, adjoint, and refusal oracles.
- [x] Add differentiable Complement Naive Bayes with relaxed nonnegative real
  counts, sorted arbitrary integer labels, positive complement smoothing,
  empirical or explicit priors, sample/class weights, optional second weight
  normalization, packed parameters, stable log-probabilities, input/parameter
  JVP/VJPs, and independent analytic, finite-difference, adjoint, and refusal
  oracles.
- [ ] Add multilabel-indicator and multioutput classification contracts,
  including per-output class metadata, sparse target support, threshold
  policies, micro/macro/samples averaging, and refusal of ambiguous target
  shapes.
- [ ] Add ordinal classification and ranking losses as separate contracts.
  do not reinterpret nominal softmax probabilities as ordered outcomes.
- [x] Add the deterministic finite-only `cart_classifier_t` adapter with
  sorted integer classes, weighted Gini/entropy probabilities, and explicit
  depth, leaf-size, tie, and nonfinite-input contracts.
- [ ] Add shared tree and boosted-tree classifier adapters for missing-value,
  monotonic-constraint, and probability-policy variants.
- [x] Add a binary Laplace GP classifier with Bernoulli logistic and probit
  likelihoods, Newton convergence state, predictive latent moments, observed
  probabilities, and input JVPs over the supported kernel derivative contract.
- [x] Add a deterministic one-vs-rest multiclass GP wrapper with sorted integer
  classes, independent binary Laplace fits, normalized probability simplex,
  deterministic prediction ties, and refusal propagation.
- [ ] Add robust and multiclass variational GP likelihoods, quadrature or
  variational objectives, predictive probability products, and calibrated
  latent-to-observed uncertainty. All GP classifiers share the same label and
  metric machinery as linear and neural classifiers.
- [ ] Extend neural classifier heads to binary, multilabel, ordinal, and
  calibrated outputs. Heads expose loss, logits, probabilities, and derivative
  products separately so a user can train on logits without losing a stable
  deployment probability path.
- [x] Add accuracy, top-k accuracy, balanced accuracy, confusion matrix, log
  loss, Brier score, binary Matthews correlation, precision,
  recall, and F1 with explicit class ordering, zero-support behavior, and
  weighted accuracy/log-loss semantics. Binary ROC AUC remains open.
- [x] Add deterministic multiclass expected and maximum calibration error
  metrics with row normalization, first-maximum tie handling, equal-width
  bins, optional sample weights, and explicit empty-bin and confidence-one
  semantics. Sigmoid and isotonic calibrator estimators remain open.
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
- [x] Provide a horizontal `basis_pipeline_t` that concatenates fixed basis
  stages, packs stage parameters, and routes value/JVP/VJP products with shape
  and stage-initialization refusal tests.
- [x] Provide `sequential_basis_pipeline_t` with explicit stage feature-count
  contracts, flattened parameters, chained JVP/VJP products, and independent
  finite-difference and adjoint tests. Column-wise and DAG composition remain
  open.
- [x] Add fitted `standard_scaler_t` and `minmax_scaler_t` transformers with
  explicit row-oriented fit/transform/inverse contracts, constant-column
  policies, and exact input JVPs. Fitted statistics are state rather than
  silently promoted differentiable parameters.
- [x] Provide row-oriented sample conventions, explicit status results, and a
  registry for packed model parameters.
- [x] Add index-only deterministic K-fold and stratified-K-fold splitters with
  seeded shuffling, balanced test folds, replayable cursors, and explicit
  invalid-fold refusals. Cross-validation scoring and separate train/validation
  batch streams remain open.
- [ ] Define fitted transformer, predictor, regressor, and classifier contracts.
  The contracts cover feature counts, fitted state, reset or clone behavior,
  parameter names, and status propagation.
- [ ] Define estimator tags and capability queries for supervised versus
  unsupervised targets, sparse/dense inputs, missing values, sample weights,
  partial fitting, derivatives, device support, and probabilistic outputs.
  Search and pipeline code must query capabilities instead of guessing from a
  dynamic type.
- [x] Add `simple_imputer_t` with mean, median, and constant strategies,
  explicit IEEE-NaN missingness, all-missing-column policy, fitted statistics,
  and independent transform/JVP/VJP tests.
- [x] Add `one_hot_encoder_t` for integer categorical columns with deterministic
  per-feature sorted categories, packed category/output offsets, optional
  reference-category dropping, explicit unknown error/ignore behavior, and
  missing error/ignore/category policies. Its categorical JVP/VJP methods
  validate shapes and return an explicit `FORTNUM_NOT_IMPLEMENTED` boundary
  rather than claiming a derivative that has no canonical meaning. Independent
  value, metadata, policy, and refusal tests are in `test_one_hot_encoder`.
- [ ] Add missing-indicator features and sparse CSR/CSC one-hot views. Column
  selection is implemented for basis unions; general transformer columns remain
  open. The dense one-hot correctness lane is now in
  `../fortml-bench/results/ONE_HOT_ENCODER.md`; sparse throughput and memory
  comparisons remain before this family is performance-complete.
- [ ] Add robust scaling, quantile and power transforms, normalization,
  missing-indicator features, ordinal encoding, target encoding with leakage
  guards, polynomial interactions, hashing, and sparse CSR/CSC feature views.
  Every fitted transform records statistics, feature names, dtypes, and the
  treatment of unseen or missing categories.
- [x] Add sequential basis pipelines. Basis maps work as fitted or fixed
  pipeline stages and propagate chained JVP/VJP products.
- [x] Add column-selecting basis feature unions. `column_basis_pipeline_t`
  validates one-based per-stage column lists, gathers selected inputs, packs
  stage parameters, and routes exact JVP/VJP products with scatter-add input
  cotangents. Cross-stage column reuse is deterministic. Duplicate indices
  within a stage are rejected.
- [x] Add `basis_linear_regression_t` as an estimator-level composition seam.
  It fits a multi-output linear model on a fitted basis pipeline, packs basis
  parameters before column-major coefficients, and chains exact input and
  parameter JVP/VJP products. Independent finite-difference and adjoint tests
  cover the complete composition.
- [ ] Add parallel feature-union execution, device-resident transforms, and
  general column-wise transformer graphs beyond fixed basis maps.
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

- [x] Implement a deterministic exhaustive-split regression stump with
  piecewise-constant prediction and an input-JVP refusal at split boundaries.
- [x] Define the finite-input refusal contract for exact stumps and residual
  boosting. NaN and infinite fit values, prediction inputs, and JVP tangents
  return a domain status. No missing value is silently routed to a branch.
- [x] Implement squared-loss gradient boosting over regression stumps with
  staged predictions and deterministic tree order. Weighted, missing-value,
  histogram, classifier, and second-order boosting remain open.
- [x] Add the exact depth-limited `xgboost_t` second-order lane for squared and
  binary-logistic objectives. It aggregates per-leaf gradients and Hessians,
  applies L1/L2/gamma/min-child-Hessian regularization and shrinkage, exposes
  margins/probabilities, split gains and leaf weights, and has a piecewise
  input-JVP/refusal contract. Recursive growth and tree depth/node diagnostics
  are implemented. Explicit NaN policies reject, learn, or force a default
  branch per split; infinities remain refused and NaN queries have zero local
  input JVP. Histograms and constraint policies remain open.
- [x] Add deterministic one-vs-rest `xgboost_multiclass_t` classification over
  the binary logistic lane. Sorted integer labels, normalized probabilities,
  argmax prediction, decision margins, quotient-rule probability JVPs, and
  split-boundary refusals have independent behavioral and finite-difference
  tests.
- [x] Add explicit XGBoost-compatible NaN handling to binary and one-vs-rest
  trees. `missing_policy="error"` is the default refusal; `"learn"` compares
  both default directions in every exact threshold and stores the strict-best
  route (left wins exact ties), while `"left"` and `"right"` force a route.
  Prediction and multiclass normalization use the stored route, infinities are
  rejected, and binary NaN routing plus boundary/JVP behavior have independent
  analytic tests.
- [x] Add a deterministic depth-limited CART regression tree with weighted
  squared-error splits, `max_depth` and `min_samples_leaf` constraints, fixed
  feature/threshold tie ordering, finite-only/refusal behavior, and independent
  prediction/JVP oracles.
- [x] Add deterministic CART classification with weighted Gini and entropy
  criteria, sorted integer classes, class probabilities, finite-only/refusal
  behavior, depth and leaf constraints, and the same feature/threshold tie
  rule. Independent pure-leaf, weighted-frequency, criterion, tie-order, and
  refusal oracles cover the public `fortml_cart_classifier` contract.
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
- [x] Add weighted Multinomial, Bernoulli, and Complement naive Bayes with
  stable log-probability products and declared input/parameter derivative
  boundaries.
- [x] Add Categorical naive Bayes with sorted per-feature category offsets,
  weighted class priors and likelihood smoothing, explicit unknown-category
  error/ignore policies, and a discrete-input JVP refusal test.
- [ ] Add LDA/QDA and discriminant shrinkage with stable log-probability
  products.
- [ ] Add k-means/minibatch k-means, Gaussian mixtures, Bayesian mixtures,
  spectral and agglomerative clustering, DBSCAN/OPTICS, affinity propagation,
  BIRCH, and graph-connected components where dependencies and memory limits
  are explicit.
- [x] Add centered dense PCA with a thin SVD, deterministic signs, rank selection,
  whitening, reconstruction, explained-variance metadata, fixed-state input
  JVP/VJP products, and independent closed-form covariance/refusal tests.
- [ ] Add incremental/randomized PCA, sparse PCA, kernel PCA, ICA, NMF,
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

- [x] Implement core regression metrics (R2, explained variance, MSE, RMSE,
  MAE, weighted median absolute error, max error, MSLE, MAPE, and pinball)
  with explicit finite, shape, weight, constant-target, and quantile refusal
  contracts and independent hand-formula oracles. Poisson/Gamma/Tweedie
  deviance remains open. Classification metrics (accuracy, top-k, balanced accuracy,
  precision/recall/F-beta, Jaccard, Hamming, log loss, ROC/PR AUC, Brier,
  calibration error), ranking metrics (DCG/NDCG, MAP, MRR), clustering metrics
  (silhouette, Calinski-Harabasz, Davies-Bouldin, adjusted rand, mutual
  information), and probabilistic metrics (NLL, CRPS, interval coverage,
  sharpness, calibration).
- [x] Define multiclass expected and maximum calibration error as weighted
  equal-width confidence-bin metrics with deterministic tie handling and
  refusal of invalid rows, labels, weights, or bin counts.
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
- [x] Add deterministic full-batch and mini-batch MLP Adam training with seeded
  shuffling, early stopping, callbacks, best-state restoration, loss history,
  and an analytic L2 hyperparameter derivative. The adapter currently targets
  mean-squared-error objectives.
- [x] Add the multiclass MLP cross-entropy trainer adapter with deterministic
  Adam state, sorted labels, probability products, and a packed parameter
  gradient. Other likelihoods and shared parameter-tree routing remain open.
- [x] Add the exact MSE+L2 MLP joint HVP product, including the mixed
  parameter/L2 hyperparameter block. Independent linear and nonlinear finite-
  difference tests cover the product used by outer FortOpt objectives. Adam
  trajectory and schedule hypergradients remain open.
- [x] Add the exact fixed full-batch MLP trajectory hypergradient objective over
  `[log(learning_rate), log(l2)]`, including forward JVP, reverse value/VJP,
  validation-MSE objective, and a FortOpt L-BFGS-B adapter. Independent
  central-difference and adjoint tests cover the products. Adam,
  momentum/Nesterov, mini-batch, schedule, and CUDA trajectories explicitly
  refuse until their optimizer state and reproducibility derivatives are
  implemented.
- [x] Add the exact fixed full-batch AdamW trajectory hypergradient objective
  over `[log(learning_rate), log(l2), log(weight_decay)]`, including analytic
  moment/decoupled-decay sensitivities, JVP/VJP products, independent central
  differences, and a FortOpt L-BFGS-B adapter. Mini-batch, schedule, beta, and
  CUDA AdamW hypergradients remain explicit follow-up contracts.
- [x] Add `mlp_training_objective_t`, a FortOpt context adapter exposing the
  packed MLP objective, analytic gradient, and exact HVP. Its optional final L2
  component makes bounded L-BFGS-B regularization search use the same products
  as training. Central-difference tests cover value gradients, HVPs, and the
  FortOpt callback path.
- [x] Add `mlp_optimize_lbfgsb`, a deterministic full-batch FortOpt adapter
  that optimizes the packed network parameters under explicit bounds and can
  append a bounded L2 hyperparameter. It consumes the analytic MLP value and
  gradient products, reports optimizer diagnostics, and refuses malformed or
  non-finite bounds. An independent closed-form ridge fixture checks the
  fitted weight and a refusal fixture checks inverted bounds.
- [x] Add `mlp_batch_iterator_t` with explicit seeded Fisher--Yates epochs,
  reproducible copied cursors, and unpadded uneven final batches. `mlp_train`
  consumes this cursor rather than maintaining a second hidden batching
  implementation.
- [x] Add per-update learning-rate schedule callbacks and global gradient-norm
  clipping to the MLP Adam trainer. The resulting state records the effective
  per-epoch rates and clipping count, with independent schedule, clipping, and
  iterator oracles.
- [x] Add sample-weighted MLP gradient accumulation. `accumulation_steps`
  flushes a configurable number of consecutive microbatches into one Adam
  update, adds L2 exactly once, clips only the accumulated gradient, and flushes
  an uneven final group. Independent tests compare one accumulated update with
  the equivalent full-batch Adam update and check update/microbatch accounting.
- [ ] Define objective and loss contracts with sum and mean reductions, sample
  weights, regularization terms, and named scalar diagnostics.
- [x] Define the MLP MSE objective's mean and sum reductions, finite
  non-negative sample weights, L2 regularization component, and named scalar
  diagnostics. The general loss and likelihood contract for other models and
  reductions remains open.
- [ ] Define a module tree and parameter-tree API for nested neural networks.
  Parameters, buffers, frozen blocks, tied/shared weights, masks, and stateful
  layers have stable paths and can be flattened without losing aliases.
- [ ] Add common neural losses and metrics: MSE/MAE/Huber/quantile,
  cross-entropy and focal losses, multilabel and ordinal losses, Gaussian and
  count likelihoods, contrastive and triplet losses, KL terms, and sequence
  masking. Every loss has explicit reduction, weighting, logits/probability,
  and empty-batch semantics.
- [x] Add a deterministic batch iterator with seeded shuffling and final-batch
  behavior. Separate training and validation streams remain open.
- [ ] Add a trainer that owns optimizer state, learning-rate schedules, gradient
  clipping, accumulation, validation intervals, early stopping, and callbacks.
  The current MLP trainer now covers deterministic accumulation, schedules,
  clipping, patience, callbacks, a finite held-out validation stream with
  interval-based monitoring and best-state restoration, and an in-memory
  resumable `mlp_training_checkpoint_t` containing Adam/AdamW/Adagrad/RMSprop/SGD,
  iterator/schedule, and
  validation state. Event typing and serialized/distributed checkpoint
  coordination remain open.
- [ ] Add production optimizers and schedules: SGD with momentum/Nesterov,
  Adam/AdamW, L-BFGS/L-BFGS-B, natural gradient, cosine,
  one-cycle, warmup/decay, plateau, and user callbacks. Optimizer state is
  dtype/device aware and rejects incompatible parameter trees.
- [x] Add FortOpt-backed SGD with momentum and Nesterov acceleration to the
  dense MLP trainer. Its velocity, optimizer kind, and step counter are
  checkpointed and resumed exactly; independent one-step and trajectory
  oracles cover the update and state contract. Schedule families,
  device-aware optimizer state, and optimizer-trajectory
  hypergradients remain open.
- [x] Add FortOpt-backed AdamW with decoupled weight decay to the dense MLP
  trainer. Its first/second moments, decay coefficient, optimizer kind, and
  step counter are checkpointed and resumed exactly; independent full-batch,
  shuffled-minibatch, refusal, and resume oracles cover the update/state
  contract. Schedule families, device-aware optimizer state, and
  optimizer-trajectory hypergradients remain open.
- [x] Add FortOpt-backed Adagrad to the dense MLP trainer. Its accumulated
  squares, epsilon, optimizer kind, and step counter are checkpointed and
  resumed exactly; independent two-step recurrence, refusal, and interrupted
  versus uninterrupted trajectory oracles cover the update/state contract.
  Schedule families, device-aware optimizer state, and
  optimizer-trajectory Adagrad hypergradients remain open.
- [x] Add FortOpt-backed RMSprop to the dense MLP trainer. Centered and
  uncentered running statistics, optional momentum, optimizer configuration,
  and step state are checkpointed and resumed exactly; an independent
  recurrence oracle covers both variants. Optimizer-trajectory RMSprop
  derivatives and schedule families remain open.
- [ ] Add automatic mixed precision with loss scaling, overflow detection,
  master weights, deterministic accumulation modes, and explicit fp16/bf16/fp32
  capability reports. A mixed-precision result must pass a full-precision
  accuracy oracle before it can enter a performance report.
- [x] Add sample-weighted MLP microbatch accumulation with an explicit flush
  boundary and exact full-batch equivalence for the MSE+L2 objective.
- [ ] Add activation checkpointing, truncated BPTT, gradient
  centralization/noise, value clipping, and anomaly detection with
  parameter-path diagnostics.
- [ ] Add callback and event contracts for logging, progress, validation,
  checkpoint, early stop, learning-rate changes, and recoverable failures.
  Callback order and failure propagation are deterministic and testable.
- [ ] Add data-loader workers or asynchronous prefetch only when the ownership
  and RNG contract is explicit. Worker count must not silently change the
  sampled batches for a deterministic run.
- [ ] Add trainer adapters for linear classifiers, MLPs, BNNs, VAEs, RNNs, exact
  GPs, derivative GPs, and sparse variational GPs. The MLP MSE and multiclass
  classifier adapters are complete for their current objectives. Each new
  adapter requires a scalar objective, parameter gradient, reduction rule, and
  complete train-state update.
- [ ] Add adapters for convolutional, recurrent/attention, graph, autoencoder,
  probabilistic, and tree/boosting objectives as their model contracts land.
- [ ] Add compile/fusion planning for static expression graphs and batched
  kernels, with a cache key containing architecture, dtype, device, and shape.
  Compilation may be optional, but a stale or incompatible plan must refuse
  rather than execute with wrong strides.
- [x] Define in-memory train state independently of file serialization. The MLP
  `mlp_training_checkpoint_t` includes parameters, Adam accumulators and bias
  step, epoch and microbatch positions, the exact iterator permutation/RNG
  stream, schedule metadata/history, validation and best-state counters, and
  deterministic resume validation. Procedure pointers remain caller-owned and
  best-state restoration marks a snapshot non-resumable when optimizer and
  model state no longer align.
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
- [x] Add a bounded single-start exact-GP hyperparameter adapter using FortOpt
  L-BFGS-B, the public analytic likelihood gradient, explicit log-parameter
  bounds, convergence diagnostics, and a final gradient norm.
- [x] Add derivative-GP parameter packing, likelihood/JVP/HVP entry points, and
  a bounded FortOpt adapter for mixed value/first-derivative observations.
  The likelihood gradient uses analytic parameter tangents for the supported
  RBF, Matérn, linear, constant, and composed kernels and is checked against
  an independently assembled dense oracle. The public HVP is currently a
  deterministic directional finite difference of that gradient. Generated
  second-order derivative products remain a separate capability gate.
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
- [x] Generate and ship the FortSym RBF primal and first-order natural-parameter
  leaf, retaining the generated Fortran source, independent dense and
  finite-difference checks, generator revision, IR-node count, and compound-op
  count. It feeds the RBF parameter JVP/VJP paths. Generate and ship the
  FortSym Matérn 1/2 HVP leaf as well (`9482261`, 37 IR nodes, 28 compound
  operations), with an independent analytic and directional finite-difference
  test in `test_fortsym_matern12`. The Matérn 3/2 HVP is now also emitted by
  FortSym `b72a23a` (60 IR nodes, 48 compound operations), with an independent
  oracle in `test_fortsym_matern32`; Matérn 5/2 remains FortAD-generated;
  complete proof and source-hash sidecars remain a release task, and the
  general kernel family matrix still follows the capability/refusal policy.
- [ ] Generate analytic kernels with `fortsym` when it proves a smaller
  expression, preserve the proof/operation-count/source hash, and compare the
  generated product against current FortAD `main` and an independent oracle.
  This includes RBF/Matérn parameter JVP/VJP/HVP leaves when the symbolic
  common-subexpression count beats the current FortAD product. Generated code
  is accepted only with a fallback or a documented structured refusal for
  unsupported shapes and smoothness.
- [ ] Add likelihood gradients and HVPs for derivative-observation GPs, including
  noise parameters for each observation type.
- [x] Add parameter JVP and VJP products for derivative-observation GP
  prediction means and variances, with independent dense finite-difference and
  reverse-product oracles over value/first-derivative query components.
- [x] Add deterministic finite-difference query-input JVP and VJP products for
  derivative-observation GP means and variances, with independent directional
  and adjoint finite-difference oracles. The products hold parameters and
  training inputs fixed. An analytic third-order kernel contract and joint
  posterior covariance remain open.
- [ ] Add joint posterior covariance for value and derivative queries, plus
  cross-covariances between requested components.
- [ ] Extend derivative observations to second derivatives only for kernels with
  the required smoothness, with explicit refusal at singular coincident cases.
- [ ] Add scalar objectives and parameter gradients for multi-output, sparse
  variational, local, SKI, Lanczos, and matrix-free GP paths. Inducing-point and
  local-gate training remain separate parameter blocks.
- [x] Add binary Laplace GP classification for logistic and probit likelihoods,
  with damped Newton state, latent/probability prediction, input JVPs over the
  kernel derivative contract, and an exact envelope gradient for the converged
  mode log posterior (without evidence correction).
- [x] Publish fitted-kernel parameter metadata and exact mode-envelope
  hyperparameter gradients for binary and one-vs-rest GP classifiers. The
  multiclass wrapper packs independent binary gradients; a shared categorical
  Laplace evidence gradient remains open.
- [x] Add one-vs-rest multiclass GP classification as a deterministic wrapper
  over the binary Laplace contract, with sorted labels and normalized positive
  probabilities. Variational categorical likelihoods remain open.
- [x] Add bounded FortOpt L-BFGS-B adapters for binary and shared-kernel
  one-vs-rest GP classification. Each trial refits the Laplace mode and uses
  the analytic envelope gradient; invalid bounds, failed mode solves, and
  nonfinite objectives are refused. Full evidence, likelihood-parameter,
  independent per-class, and implicit/HVP training remain open.
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
- [x] Define a public CPU/CUDA device selector and ownership contract for host
  and CUDA allocations, with capability probes, backend identity, residency
  byte/event counters, and recoverable refusal for unavailable CUDA kernels or
  unsupported streams. The metadata layer does not allocate buffers or claim
  complete GPU execution; operator data regions remain explicit.
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
  GP, GP feature products, approximate or matrix-free GP methods, and the
  matched multinomial/neural classifier lane in `../fortml-bench`.
- [x] Record release timings, peak RSS, build provenance, external Python
  comparisons, raw CSV files, and plots through `../fortml-bench`.
- [x] Add the MLP-training, basis-pipeline, decision-stump, depth-limited CART
  regression, core regression-metrics, and residual-stump boosting lanes with
  independent NumPy oracles, contextual scikit-learn rows, and explicit
  PyTorch/JAX/XGBoost availability or refusal rows. The release record is
  [`results/FEATURES.md`](../fortml-bench/results/FEATURES.md).
- [x] Add the exact depth-limited recursive XGBoost-style squared/logistic lane
  (including depth/node diagnostics), explicit learned/forced NaN routing, and the
  fitted-scaler plus binary and one-vs-rest multiclass Laplace GP
  logistic/probit lanes with independent NumPy oracles. The release records are
  [`results/XGBOOST.md`](../fortml-bench/results/XGBOOST.md) and
  [`results/CLASSIFICATION_EXTENSIONS.md`](../fortml-bench/results/CLASSIFICATION_EXTENSIONS.md).
- [x] Add ComplementNB and integer one-hot benchmark lanes with independent
  NumPy oracles, contextual scikit-learn rows, explicit categorical derivative
  refusals, and parseable unavailable FortML release-target rows. The release
  records are [`results/COMPLEMENT_NB.md`](../fortml-bench/results/COMPLEMENT_NB.md)
  and [`results/ONE_HOT_ENCODER.md`](../fortml-bench/results/ONE_HOT_ENCODER.md).
- [x] Add a CategoricalNB release app and independent category-count oracle;
  the report is [`results/CATEGORICAL_NB.md`](../fortml-bench/results/CATEGORICAL_NB.md).
- [x] Add AdamW training and fixed full-batch MLP hypergradient lanes with
  independent NumPy recurrences/finite differences, passing FortML release apps,
  explicit CPU-only and CUDA refusal rows, and clean revision provenance. The release record is
  [`results/ADAMW_HYPERGRADIENT.md`](../fortml-bench/results/ADAMW_HYPERGRADIENT.md).
- [x] Add a centered dense PCA lane with an independent NumPy thin-SVD oracle,
  scikit-learn context rows, deterministic-sign/orthogonality guards, and a
  FortML release-app timing. The raw record is
  [`results/pca.csv`](../fortml-bench/results/pca.csv); complete fitted-array
  export remains explicitly open in
  [`results/PCA.md`](../fortml-bench/results/PCA.md).
- [x] Add an Adagrad accumulated-square lane with independent recurrence and
  split/resume checks plus a FortOpt release-app norm/timing gate. The raw
  record is [`results/adagrad.csv`](../fortml-bench/results/adagrad.csv), with
  the contract documented in
  [`results/ADAGRAD.md`](../fortml-bench/results/ADAGRAD.md).
- [x] Add independent kNN uniform/inverse-distance, RMSprop direct/MLP, and
  binary/multiclass staged-XGBoost benchmark lanes. Their reports and raw
  records are [`results/KNN.md`](../fortml-bench/results/KNN.md),
  [`results/RMSPROP.md`](../fortml-bench/results/RMSPROP.md), and
  [`results/XGBOOST.md`](../fortml-bench/results/XGBOOST.md).
- [x] Add a bounded binary/shared-kernel GP-classification hyperparameter lane
  with a NumPy mode/envelope-gradient oracle. The evidence is explicitly for
  mode log posterior rather than full Laplace evidence:
  [`results/GP_CLASSIFICATION_TRAINING.md`](../fortml-bench/results/GP_CLASSIFICATION_TRAINING.md).
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
- [Symplectic learning for Hamiltonian neural networks](https://doi.org/10.1016/j.jcp.2023.112495)
  analyzes the discretization error introduced by the training integrator and
  motivates training through a symplectic map.
- [Physics-informed neural networks](https://doi.org/10.1016/j.jcp.2018.10.045)
  combine data and differential-equation residuals in one objective. The
  [PIML review](https://doi.org/10.1038/s42254-021-00314-5) by Karniadakis et
  al. surveys physics-guided, physics-informed, and physics-encoded
  architectures and the different ways equations and domain knowledge enter a
  model.
- [A connection between probability, physics and neural networks](https://doi.org/10.3390/psf2022005011)
  by Sascha Ranftl connects kernels satisfying linear differential constraints
  to the infinite-width neural-network limit (arXiv:2209.12737). Its
  finite-width construction is an approximation to the limiting GP, so FortML
  will test seeded ensembles against that covariance instead of claiming an
  exact initializer.
- [Symplectic Gaussian Process Regression of Hamiltonian Flow Maps](https://arxiv.org/abs/2009.05569)
  by Katharina Rath, Christopher Albert, Bernd Bischl, and Udo von Toussaint
  provides the project-specific symplectic-GP reference. Product and sum
  kernels correspond to implicit and explicit symplectic-Euler constructions,
  which gives FortML a concrete covariance and long-horizon oracle.
- [Boundary constrained Gaussian processes for robust physics-informed machine learning](https://www.jmlr.org/papers/v25/23-1508.html)
  provides exact Dirichlet, Neumann, Robin, and mixed boundary-condition GP
  priors for linear PDEs. [Physics-informed Kernel Learning](https://www.jmlr.org/papers/v26/24-1536.html)
  gives a recent Fourier-approximated kernel-regression alternative to a PINN.
  Both are candidates for a native operator-kernel lane.
- The official TU Graz DocDay program [abstract by Johanna
  Moser](https://www.tugraz.at/sites/dsp/docdays/past-docdays/september-2026)
  describes the Ghosttasking and Monge-GP direction for physics-informed GPs
  for linear differential equations, including parameter inference outside the
  constant-coefficient and controllable cases. This is an experimental,
  non-peer-reviewed reference, not a completed FortML feature claim.
- [Symplectic Neural Gaussian Processes](https://www.ijcai.org/proceedings/2024/465)
  combines a GP Hamiltonian with a learned system representation for
  data-efficient Hamiltonian dynamics.
- [Lagrangian Neural Networks](https://arxiv.org/abs/2003.04630),
  [SympNets](https://arxiv.org/abs/2001.03750), and
  [symplectic recurrent neural networks](https://arxiv.org/abs/1909.13334)
  provide complementary structure-preserving architectures.
- [Direct Poisson neural networks](https://arxiv.org/abs/2305.05540) extend the
  target beyond nondegenerate canonical symplectic systems to Poisson systems.
- [Deep Neural Networks as Gaussian Processes](https://arxiv.org/abs/1711.00165)
  and the [Neural Tangent Kernel](https://arxiv.org/abs/1806.07572) separate
  prior-function covariance from the linearized training kernel. They motivate
  distinct NNGP and NTK initializers and benchmarks rather than treating a
  finite MLP as exactly equivalent to one GP.
- [Neural networks and principal component analysis](https://doi.org/10.1016/0893-6080(89)90014-2)
  proves the linear autoencoder/PCA optimum and its saddle structure. FortML's
  PCA initializer should therefore be a deterministic linear optimum with
  explicit centering, rank, whitening, and sign conventions, not a random
  pretraining shortcut.

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
- [ ] Add a PINN and physics-informed GP training adapter over the shared
  objective. It must keep data, residual, initial/boundary, and conservation
  terms separately addressable for weighting, derivatives, and diagnostics.
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
`72f8fe8` and `fortsym` `main` at `b72a23a`. The checked-in RBF HVP/product
module was generated by FortAD `5e1bfe0`; the RBF primal and first-order leaf
was generated by FortSym `f71a1aa` (the earlier primal-only leaf remains
generator revision `16fc3a8`). The RBF log-length tangent used
by the derivative-GP products was independently checked with a temporary
FortSym proof against the dense numerical oracle. The Matérn 1/2 HVP leaf is
also generated by FortSym `9482261`; its independent oracle is the
`test_fortsym_matern12` target. The Matérn 3/2 HVP leaf is generated by the
FortSym `b72a23a` and checked by `test_fortsym_matern32`; Matérn 5/2 still
uses FortAD. Future derivative work must refresh both
checkouts before deciding that a product is unavailable.

#### WP9b: Hamiltonian, Lagrangian, and symplectic networks

- [x] Add the separable `hamiltonian_mlp_t` prototype with scalar `V(q)` and
  `T(p)` MLPs, packed energy/state JVP and VJP products, canonical vector-field
  products, and an explicit leapfrog map. Independent finite-difference,
  adjoint, reversibility, and symplectic-form oracles cover the contract.
  General nonseparable Hamiltonians, learned Poisson structures, and implicit
  integrators remain open.
- [ ] Extend the Hamiltonian MLP to a general nonseparable scalar H(q,p),
  canonical J, optional learned skew structure matrix, and parameter/input
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
  Compare with Raissi, Perdikaris, and Karniadakis, [machine learning of
  linear differential equations using Gaussian processes](https://doi.org/10.1016/j.jcp.2017.07.050),
  and [numerical GPs for time-dependent and nonlinear PDEs](https://doi.org/10.1137/17M1120762).
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
- [ ] Add structure-aware GP-posterior initialization for ordinary MLPs,
  Hamiltonian and symplectic networks, and PINN residual networks. The mapping
  from the infinite-width GP or NNGP/NTK feature representation to finite
  weights must record its mean, covariance, and structure-defect error instead
  of claiming an exact finite-width equivalence.
- [ ] Add PCA initialization for linear autoencoders, following the exact
  Baldi--Hornik optimum above. The empirical [principal-component
  initialization proposal](https://doi.org/10.1007/978-3-030-30484-3_14)
  (Suzuki and Sakanashi, 2019) is a separate deep-autoencoder warm start, not
  a claim about the exact linear optimum. The encoder and decoder use the
  selected principal subspace, with centering, whitening, rank, and sign
  conventions recorded. The reconstruction oracle must match the PCA
  projection to numerical tolerance.
- [ ] Reuse the public `pca_t` centered-SVD state for a linear autoencoder
  initializer. Tied and untied decoder choices, rank
  truncation, whitening, and sign conventions must produce the same projected
  reconstruction oracle.
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
- [`CLASSIFICATION_EXTENSIONS.md`](../fortml-bench/results/CLASSIFICATION_EXTENSIONS.md),
  backed by `classification_extensions.csv` for binary and one-vs-rest Laplace
  GP classification and fitted preprocessing.
- [`CLASSIFICATION_MODELS.md`](../fortml-bench/results/CLASSIFICATION_MODELS.md),
  backed by `classification_models.csv` for multinomial softmax and multiclass
  neural classifiers.
- [`BERNOULLI_NB.md`](../fortml-bench/results/BERNOULLI_NB.md), backed by
  `bernoulli_naive_bayes.csv` for relaxed Bernoulli Naive Bayes, its input JVP,
  and the native FortML release-app protocol.
- [`MULTINOMIAL_NB.md`](../fortml-bench/results/MULTINOMIAL_NB.md), backed by
  `multinomial_naive_bayes.csv` for token-mass smoothing, stable probabilities,
  predictions, and complete input-JVP output arrays.
- [`COMPLEMENT_NB.md`](../fortml-bench/results/COMPLEMENT_NB.md), backed by
  `complement_naive_bayes.csv` for complement counts, stable probabilities,
  predictions, and complete input-JVP oracle checks.
- [`CATEGORICAL_NB.md`](../fortml-bench/results/CATEGORICAL_NB.md), backed by
  `categorical_naive_bayes.csv` for packed category metadata, smoothed
  probabilities, predictions, and the discrete-derivative boundary.
- [`ONE_HOT_ENCODER.md`](../fortml-bench/results/ONE_HOT_ENCODER.md), backed by
  `one_hot_encoder.csv` for sorted categories, packed one-based offsets,
  dense transforms, and explicit categorical derivative refusals.
- [`ADAMW_HYPERGRADIENT.md`](../fortml-bench/results/ADAMW_HYPERGRADIENT.md),
  backed by `adamw_training.csv` and `mlp_hypergradient.csv` for independent
  AdamW recurrence and fixed-trajectory hypergradient finite-difference oracles.
- [`TRAINING_IMPUTER.md`](../fortml-bench/results/TRAINING_IMPUTER.md), backed
  by `training_imputer.csv` for Adam-independent momentum-SGD/Nesterov MLP
  trajectories and mean/median/constant imputer transform/JVP/VJP products.
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
