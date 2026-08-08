# fortml roadmap

Verified on 2026-08-08. Interfaces are documented in
[`docs/API.md`](docs/API.md), examples in [`docs/EXAMPLES.md`](docs/EXAMPLES.md),
and implementation limits in [`docs/DESIGN.md`](docs/DESIGN.md) and
[`docs/ML_ARCHITECTURE.md`](docs/ML_ARCHITECTURE.md).
The cross-library acceptance table is maintained in
[`docs/PARITY_MATRIX.md`](docs/PARITY_MATRIX.md).

## Verification

The GitHub `v0.1.0` tag currently points to the earlier release-verification
commit `a387cc5`; the trainer, calibration, variational-GP, transform, and CUDA
VJP closure slices documented below are post-tag additions. The broad parity
gate is still open, so this work does not move or recreate that tag.

| Compiler | Command | Result |
| --- | --- | --- |
| GNU Fortran | `fo` | Static build, all 193 behavioral tests, and lint passed at the current FortML/FortAD-main revisions. The compiler still emits non-fatal array-temporary warnings; see [`verification/fortml-gfortran.txt`](verification/fortml-gfortran.txt). |
| NVIDIA HPC SDK | `FO_FC=nvfortran fo` | Static and lint checks passed in the recorded older compiler lane. The checked-in NVIDIA log predates the current 193-test GNU run. See [`verification/fortml-nvfortran.txt`](verification/fortml-nvfortran.txt). |
| Intel LLVM Fortran | `ifx` | Compiler unavailable in the verification environment. Not tested. |

The checked-in GNU compiler log is the fresh 2026-08-08 run against FortML code
revision `176aac62f2e488b7ee96a93be16a1ae61dffa697`, FortAD `origin/main` at
`5f879a6021eb1f0637a7cc7de7713756f7f09ae1`, and FortNum at
`38bc0e578ec5c6c0e636e8fdd3844f54f9e3e473`, run from the clean checkout
under `/mnt/storage/code/lazy-fortran/fortml`. The run includes the
  kernel-catalog, weighted LDA/QDA, robust/absolute XGBoost, neural NLL, random-forest,
Extra-Trees, grouped MLP HVP and L-BFGS-B, basis/pipeline HVP, cosine
derivative-GP, multilabel/ROC-AUC/PR-AUC/F-beta ranking, derivative-GP
capability refusals, resident-MSE and dense-affine CUDA contracts, resident
forest plan boundary, PCA-initialized linear autoencoder, seeded exact-GP
multistart, multilabel/ordinal neural losses, squared-log XGBoost, named MLP
parameter layout, softmax objective products, and validation-stopping XGBoost
slices, binary MLP loss products, trainable exact-GP mean products, ARD GP
 products, XGBoost sampling, XGBoost serialization, the bounded Bernoulli
variational-GP objective, multiclass variational-GP prediction/JVPs/VJPs, positive
 positive multiclass temperature calibration, PINN objective fitting, unfactored
 Adafactor recurrence/checkpoint and relative-step/parameter-scale products, the portable trainer checkpoint, pairwise XGBoost
 ranking, physics residual value/JVP/VJP products, and the
 transform-aware hyperparameter registry, sparse variational-GP ELBO products,
 XGBoost warm-start continuation, sparse-GP kernel hyperparameter products,
 classifier-chain logistic products, MLP optimizer groups/checkpoint metadata,
 and differentiable basis-pipeline
 training objective. The build emits non-fatal GNU
array-temporary warnings in FortFront query/generator calls, existing GP
benchmark boundaries, variational-GP batch conversions, and basis-pipeline
shape conversions. They are isolated to array construction; all 193 behavioral
tests pass. Lint has zero unused-import findings and the full `fo` lint stage
passes despite the non-fatal compiler warning corpus. The independent CUDA gate additionally covers the
resident dense-affine value/JVP/VJP path and its single-layer MSE update with
parameter snapshots and transfer counters. NVIDIA
compiler coverage remains an
explicit older-build result.

The companion benchmark harness is clean at FortML-bench revision
`58fc29433c29d43c3f78dfe43a7f76c561b39c58`;
the trainer-checkpoint, unfactored-Adafactor, binary-objective,
multiclass-calibration, variational-multiclass-GP, PINN/physics-objective,
physics HVP, grouped K-fold, spectral-mixture, XGBoost-ranking,
resident dense-MSE CUDA, binary XGBoost classifier, calibrated neural classifier,
SGD-momentum, sparse preprocessing, derivative-GP covariance and polynomial HVP,
weighted multiclass MLP objective, resident Adagrad, and mini-batch hypergradient
CSV rows record their FortML source revisions and independent NumPy or analytic
behavioral oracles. The basis-pipeline lane now includes the optimized-ridge
coordinate/mixed-HVP case, and the binary Laplace-GP parameter-product test has
an independent fixed-state finite-difference oracle. The current closure slices
also cover packed OVR Laplace-GP parameter products, weighted LightGBM validation
early stopping, and scheduled-Adagrad trajectory hypergradients. The latter has
its own CPU-product and CUDA-refusal rows in
`results/mlp_adagrad_schedule_hypergradient.csv`. CUDA rows are explicit `unavailable`/typed-refusal records
rather than host timings. The XGBoost warm-start lane adds an independent
Newton-stump replay, transactional refusal rows, and an explicit CUDA-
unavailable record in `results/xgboost_warm_start.csv`. The classifier-chain
lane adds packed-head NumPy replay, integer-label prediction checks, fit/predict
timing, and an explicit CUDA-unavailable row in
`results/classifier_chain.csv`. The XGBoost interaction-constraint lane adds
an independent NumPy group-mean/path-mask oracle for unconstrained and
separated-group depth-two trees; its CPU rows pass with zero prediction error
and its CUDA row is an explicit unavailable capability record in
`results/xgboost_interaction.csv`.

### 2026-08-08 parity and provenance slice

This verification records the current bounded production contracts without
changing the broader parity claim. `classifier_chain_t` fits one binary logistic head per
output, accepts arbitrary sorted integer label pairs, supports shared sample
weights, per-output class weights and thresholds, and uses observed positive
indicators during fitting followed by smooth positive probabilities at
prediction. Its packed head parameters have exact input and parameter JVP/VJP
products; hard labels and fit-time optimizer decisions remain discrete. CPU
dispatch is tested and selected CUDA calls return `FORTNUM_NOT_IMPLEMENTED`.
The independent fixture is `test_classifier_chain`, with the release evidence
in `fortml-bench/results/CLASSIFIER_CHAIN.md`.

Sparse variational-GP ELBOs now expose fixed-state kernel-log-parameter JVP and
VJP products. The reverse product includes the inducing solve, cross-covariance,
diagonal predictive variance, and KL terms, and is checked against central
finite differences and scalar adjoint duality in `test_sparse_gp`. CPU dispatch
is complete; the CUDA entry points are typed refusals. The contract is limited
to the current Gaussian variational state and does not claim natural-gradient,
minibatch, interdomain, or resident-GPU SVGP training.

MLP training now accepts non-overlapping contiguous `optimizer_groups` with
named positive learning-rate multipliers. Every existing CPU optimizer applies
the multiplier to that block's deterministic update, while shared moment state
remains canonical. Group ranges, names, multipliers, checkpoint compatibility,
and formatted schema metadata are validated transactionally by
`test_mlp_optimizer_groups`; the schema is now format 6/text schema 4. CUDA
optimizer-group execution, mixed precision, distributed state, and migration
remain open. The source and benchmark pins for this earlier optimizer-group
slice were FortML `05632ce8fa95268417c7a2d979fa1461a202abaa` and
FortML-bench `0fb8ac7`; the current aggregate verification is the newer
`bf3810d`/`58fc294` pair recorded above.

The variational-GP classification and OVR wrappers now expose fixed-state
kernel-log-parameter JVP/VJP products for latent margins and normalized
probabilities, including the simplex adjoint. The independent
`test_gp_variational_kernel_products` fixture checks central differences,
scalar duality, simplex preservation, and typed CUDA refusal. This is a
fixed-state prediction contract; inducing-state, natural-gradient, likelihood
hyperparameter, and resident-GPU products remain open.

The MLP training surface now includes a weighted one-output Poisson log-rate
objective with exact value, JVP, VJP, HVP, optional L2 coordinate, and a
bounded FortOpt L-BFGS-B adapter. `test_mlp_poisson_objective` independently
checks finite differences, adjoint scaling, convergence, and the typed CUDA
boundary. It does not close the remaining loss/module catalog or resident
neural training gates; the API contract is in `docs/MLP_POISSON.md`.

The variational-GP binary and sorted-label OVR objectives now accept finite
nonnegative sample weights with positive total mass. Uniform scaling leaves the
normalized likelihood and KL behavior unchanged, while nonuniform weights
differentiate through the ELBO, prediction products, and bounded FortOpt
training adapter. `test_gp_variational_classification_weights` covers the
weighted finite-difference, OVR-composition, malformed-weight, and CPU/CUDA
contracts.

MLP training now includes a deterministic fixed full-batch Lion trajectory
hypergradient. The packed outer coordinates are log learning rate, log L2, and
two logit betas. Analytic JVP/VJP/HVP products feed FortOpt L-BFGS-B directly;
the sign-margin branch is a named nonsmooth refusal and CUDA remains typed
until resident trajectory state is linked. The independent fixture is
`test_mlp_lion_hypergradient`, with the API contract in
`docs/MLP_LION_HYPERGRADIENT.md`.

The current estimator slice also records deterministic binary AdaBoost over
weighted CART weak learners and dense multi-output closed-radius regression.
`test_adaboost_classifier` and `test_radius_neighbors_multioutput_regression`
provide independent hand-oracle fixtures, while the companion release rows in
`fortml-bench/results/ADABOOST_CLASSIFIER.md` and
`fortml-bench/results/RADIUS_NEIGHBORS_MULTIOUTPUT.md` retain complete CPU
prediction checks, provenance, and typed CUDA refusals. These slices do not
claim multiclass SAMME, tree-search backends, or resident radius/boosting
kernels.

### 2026-08-07 objective-trainer and tree-contribution slice

The model-agnostic `fortml_trainer` core is now a shared full-batch state
machine for any FortOpt objective. It owns explicit SGD, Adam, AdamW,
Adagrad, RMSprop, and bounded L-BFGS-B state, gradient clipping, optional
projection bounds, EMA parameters, convergence/history records, typed step
callbacks, and cloneable in-memory checkpoints. The independent quadratic
oracle is `test_trainer`; this closes the reusable objective-training seam but
does not claim mini-batch data loading, validation streams, distributed state,
mixed precision, or resident GPU execution.

The XGBoost tree contribution API now exposes additive base-margin and
per-tree contributions for regression and logistic models, with staged-margin
equivalence, shape/refusal checks, and typed CUDA refusal. Its independent
oracle is `test_xgboost_contributions`; the contribution vector is fixed-tree
and therefore remains nondifferentiable through split routing.

`trainer_t` now also has a versioned, compiler-independent formatted-text
checkpoint boundary. `save_checkpoint` records optimizer options, bounds,
parameters, EMA values, objective/history state, and complete SGD/Adam/AdamW/
Adagrad/RMSprop recurrence state; `load_checkpoint` validates schema, order,
counts, finite values, and EOF transactionally. The interrupted-versus-
uninterrupted Adam continuation oracle and malformed/truncated/extra-record
refusals are covered by `test_trainer`. Procedure callbacks and objective
closures remain process-local, and L-BFGS-B has no resumable streaming state in
this format.

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

### Production parity acceptance matrix

The release gate is capability-based. A model name is not a closed feature when
only its simplest fit path exists. Each row below needs the same state,
derivative, device, persistence, and benchmark treatment as its reference
workflow. Unsupported coordinates and nonsmooth boundaries are explicit typed
refusals. They are never implemented with hidden finite differences or a host
fallback from a CUDA request.

| Surface | Required production variants |
| --- | --- |
| Classification | Binary and multinomial linear logits; OVR and OVO coupling policies; multilabel and multioutput heads; ordinal cumulative-logit heads; Gaussian, Bernoulli, Multinomial, Complement, and Categorical Naive Bayes; LDA/QDA; linear, kernel, and one-class SVM; exact kNN and radius neighbors; CART, random/extra forests, bagging, AdaBoost, histogram boosting, XGBoost/LightGBM policies; Platt, temperature, isotonic, calibration curves; Laplace and variational GP classifiers; neural heads for binary, multiclass, multilabel, ordinal, calibrated, and physics-aware predictions. |
| Regression and transforms | OLS, weighted/ridge/lasso/elastic-net, linear and kernel SVR, robust/Huber, quantile, Gaussian/Poisson/Gamma/Tweedie GLMs, multioutput and partial-fit streams; polynomial, spline, Fourier, radial, GP, categorical, missing-value, scaling, and DAG basis/pipeline transforms with routed parameters. |
| Gaussian processes | Every shipped kernel must define value, input JVP/VJP, parameter JVP/VJP/HVP, and derivative-observation products where mathematically defined. The inference matrix covers exact dense, derivative observations, sparse/inducing, variational whitened and unwhitened, SKI/structured/lazy, local experts, batch/multitask, operator-valued and physics/symplectic GPs. Likelihoods cover Gaussian, Student-t, Bernoulli, categorical, Poisson/count, heteroskedastic, and derivative-noise blocks. |
| Neural training | Functional nested module and buffer trees; train/eval state; dense, convolutional, recurrent/GRU/LSTM, attention, graph, autoencoder/VAE, BNN, HNN/LNN, symplectic, PINN, and physics-consistent modules; complete smooth and nonsmooth loss/activation catalog; SGD, momentum/Nesterov, Adam/AdamW, Adagrad, RMSprop, Lion-like research optimizers where specified, schedules, clipping, EMA, validation, early stopping, checkpoint/resume, mixed precision, distributed reduction, and exact stochastic/device hypergradients. |
| Boosting and ensembles | Exact and histogram growth; squared, logistic, Poisson, Gamma/Tweedie, squared-log, Huber, quantile, pairwise/listwise ranking, survival/AFT where supported; monotonic, categorical, interaction, missing-value, DART/GOSS/EFB, warm-start, staged/sliced model, feature sampling, row sampling, model dumps, SHAP-compatible contributions, and resident GPU histogram paths. |
| Cross-cutting contracts | Parameter registries and transforms, FortOpt bounded L-BFGS-B and multistart, implicit differentiation through solves and optima, deterministic seeded RNG/cursors, compiler-independent state dictionaries, streaming/partial-fit data, metric/splitter/model-selection APIs, CPU/OpenACC/CUDA residency and transfer accounting, and matched scikit-learn/PyTorch/JAX/GPyTorch/Flux/Lux/XGBoost benchmark matrices. |

The matrix is a planning contract, not an implementation claim. The capability
tables and closure ledger below record which rows have evidence today and which
remain open. New work must select one bounded row, add its behavioral oracle,
and update the corresponding benchmark and provenance record in the same
change.

The external feature inventory is maintained in
[`docs/PARITY_REFERENCE.md`](docs/PARITY_REFERENCE.md). It tracks the current
scikit-learn, PyTorch, JAX, GPyTorch, GPflow, XGBoost, LightGBM, Flux, and Lux
surfaces that drive the gap register. The reference links are part of the
roadmap evidence and should be refreshed when a package changes its public
workflow or device contract.

### 2026-08-08 production-parity reset

The implementation work now follows one capability ledger rather than a list
of estimator names. A classifier, GP, neural module, booster, or basis is not
counted as production-complete until its state, weighted fit/predict workflow,
derivative products, typed device behavior, persistence seam, independent
oracle, and benchmark row agree. The clean-break pre-1.0 policy permits
replacing an incomplete interface; callers and fixtures are updated in the
same change instead of preserving a misleading compatibility layer.

The current parallel closure order is:

1. **Classification:** finish kernel-SVM/SVR and sparse or multi-output target
   contracts, then close coupled/categorical variational-GP likelihoods and
   calibration-aware model selection. Existing binary, multinomial, OVR, OVO,
   multilabel, ordinal, Naive Bayes, LDA/QDA, tree, neural, Laplace-GP, and
   Bernoulli-variational-GP implementations remain independently tested, but
   their shared preprocessing, persistence, and GPU training gates are still
   open.
2. **GPs:** make the kernel/likelihood/mean/observation derivative matrix
   explicit for exact, derivative-observation, sparse, variational, SKI,
   local, multitask, and physics/symplectic paths. Query and packed-parameter
   JVP/VJP/HVP products must either be generated by FortSym/FortAD and checked
   against a dense oracle or return a named refusal. Dense CPU products are
   reference paths, not evidence of resident GPU inference.
3. **Neural training:** unify MLP, chain, physics, recurrent, autoencoder, and
   future convolution/attention modules under one parameter-tree and train
   state. Optimizer, schedule, validation, early-stopping, clipping, EMA,
   checkpoint, and HPO variables are first-class differentiable blocks. Every
   fixed trajectory that is exposed to FortOpt gets value/JVP/VJP products and
   an explicit HVP/third-derivative boundary; stochastic and device paths do
   not silently finite-difference.
4. **Boosting and ensembles:** extend the existing deterministic XGBoost and
   LightGBM-style cores with categorical/interaction policies, warm-start and
   slicing state, DART/GOSS/EFB or typed refusals, and resident histogram
   kernels only after an independent CPU oracle and transfer-inclusive timing
   are recorded.
5. **Composition and evidence:** route every estimator through parameter,
   capability, pipeline, serialization, and benchmark contracts. A benchmark
   reports correctness, build/warm-up/steady-state timings, memory and
   transfers, precision, compiler, device, and an explicit unavailable row;
   a raw timing without the correctness gate is not a performance claim.

This reset is deliberately honest about the remaining scope: current
classification and GP surfaces are broad but partial, selected MLP optimizer
trajectories are exact but not yet universal, and GPU support is resident for
specific no-autodiff primitives rather than the whole estimator catalog. New
agents should select one bounded row from the gap register, implement behavior
plus oracle and refusal, and commit it before taking another row.

### External parity expansion

The gap register covers the following release surfaces. Each row is a separate
implementation, derivative, refusal, and benchmark contract. A wrapper around
an existing CPU estimator does not close a row when the reference package has a
different state, likelihood, split policy, or device graph.

| Reference surface | Closure work |
| --- | --- |
| scikit-learn | Estimator and transformer cloning, fitted-state tags, sparse CSR/CSC views, metadata routing, partial-fit and online updates, calibration-aware cross-validation, kernel and one-class SVMs, clustering and mixtures, decomposition, outlier detection, inspection, and model-selection reports. |
| PyTorch | Nested module and buffer trees, train/eval state, parameter selection, complete loss and activation catalog, AMP with loss scaling, compiled static graphs, distributed reduction, deterministic data loading, and portable optimizer checkpoints. |
| JAX | Functional pytrees, `jit`-eligible static graphs, `vmap` batching, forward/reverse products, donation and ownership rules, explicit sharding and collectives, and deterministic multi-device reductions. |
| GPyTorch and GPflow | Batch and multitask shapes, ARD and active dimensions, priors and constraints, likelihood families, exact lazy inference, LOVE/CIQ variance, inducing-point and stochastic variational objectives, natural gradients, deep-kernel adapters, posterior sampling, and semantic state-dict round trips. |
| XGBoost and LightGBM | Quantile sketches, validation and early stopping, warm starts and model slicing, ranking objectives, categorical and interaction constraints, DART, leaf-wise growth, GOSS, EFB, distributed histogram reduction, model dumps, and resident GPU histograms. |
| Flux and Lux | Composable nested module trees, named parameter and buffer selectors, immutable or functional training state, callback/checkpoint protocols, optimizer-state routing, and GPU array execution with the same derivative and residency evidence as the MLP trainer. |

The first implementation slices prioritize contracts that unlock several rows:
softmax and OVR objective adapters feed classification and differentiable
search, named MLP parameter blocks feed Flux/Lux-style selection and grouped
hypergradients, and booster validation state feeds early stopping, staged
prediction, model slicing, and reproducible benchmarks. Physics-informed,
Hamiltonian, symplectic, and GP-initialized models remain separate work
packages because their residual and structure certificates require additional
oracles.

### Full parity inventory

The following inventory expands the acceptance matrix into the concrete
surfaces that a production release must either implement or refuse explicitly.
It is intentionally broader than the current source tree. Each entry becomes
an API, state layout, derivative matrix, device contract, independent oracle,
and benchmark row before it can move into the implemented column.

| Reference family | Required production surface | Evidence required for closure |
| --- | --- | --- |
| scikit-learn classifiers | Ridge/SGD/perceptron/passive-aggressive linear heads; logistic, softmax, OVR, OVO, classifier chains, multilabel and ordinal heads; Gaussian, Bernoulli, Multinomial, Complement and Categorical NB; LDA/QDA; linear, kernel and one-class SVM; kNN/radius neighbors; trees, forests, extra trees, bagging, AdaBoost and histogram boosting; calibration and class-weight policies | Weighted dense and sparse fixtures, sorted-label and tie behavior, fit/predict/proba/decision/score, parameter products, nonsmooth-boundary refusals, persistence, and CPU/CUDA rows |
| scikit-learn regressors | OLS, ridge, lasso, elastic-net, Huber, Theil-Sen, RANSAC, quantile, Tweedie, Poisson/Gamma GLMs, linear/kernel SVR, kernel ridge, nearest-neighbor and multioutput/partial-fit variants | Closed-form or pinned numerical references, sample-weight and missing-data policies, fit-time and fixed-state derivatives, streaming state, convergence diagnostics, and device/refusal records |
| scikit-learn transforms and selection | Imputation, missing indicators, scaling/normalization, quantile/power transforms, one-hot/ordinal/target encoding, polynomial/spline/Fourier/radial features, sparse CSR/CSC views, feature unions and DAG pipelines; train/test, repeated/grouped/time-series CV; grid/random/halving/Bayesian/differentiable search; scorers, learning curves, calibration curves, inspection and permutation importance | Leakage guards, clone/reset and metadata routing, feature-name/schema state, routed parameter offsets, nested validation, deterministic seeds, search-variable JVP/VJP/HVP products, and serialized fold/search state |
| PyTorch, JAX, Flux, and Lux | Nested module and buffer trees, aliases/tied parameters, train/eval and freeze state, functional pytrees, batching/vmap, static graph capture, data-loader ownership, AMP/loss scaling, gradient accumulation/clipping, optimizer groups, callbacks, EMA, compiled execution, checkpoint migration, donation/partitioning, collectives, and distributed data/model parallelism | Exact value/JVP/VJP/HVP products, deterministic CPU reference, explicit ownership and transfer counters, float32/float64 parity, resumable state hashes, and resident-device tests for every claimed path |
| Neural and physics modules | Dense and convolutional blocks, normalization/dropout/embeddings, residual and attention/transformer blocks, RNN/GRU/LSTM/ODE cells, graph message passing, autoencoder/VAE/BNN, HNN/LNN/symplectic/PINN and neural-operator modules, smooth and nonsmooth losses, and physics residual/conservation diagnostics | Module-tree parameter names and buffers, train/eval semantics, derivative products through inputs/parameters/hyperparameters, seeded trajectory and long-horizon tests, manufactured-PDE or invariant oracles, and typed unsupported-boundary behavior |
| GPyTorch and GPflow | ARD/active-dimension and batch/multitask kernels, priors/constraints/mean functions, Gaussian/Student-t/Bernoulli/categorical/count/Poisson/Gamma/heteroskedastic/warped likelihoods, exact lazy and matrix-free inference, derivative/operator observations, whitened/unwhitened SVGP, natural gradients, SKI/KISS-GP, local/deep GPs, posterior sampling/fantasy updates, and semantic state dictionaries | Dense and operator-valued covariance oracles, derivative-observation finite differences, implicit solve products, stochastic ELBO seeds, batch-shape and multitask reductions, serialization round trips, and resident factorization/solve benchmarks |
| XGBoost and LightGBM | Exact and histogram/quantile-sketch growth, all supported regression/classification/ranking/survival objectives, missing defaults, monotone/categorical/interaction constraints, row/feature subsampling, early stopping and validation history, warm starts and slicing, staged margins/contributions/SHAP-compatible values, DART/GOSS/EFB, leaf-wise growth, model dumps, distributed workers, and resident GPU histograms | Independent split/gain/tree-walk references, deterministic tie and missing policies, fixed-state input products with split-boundary refusals, model/state persistence, validation/early-stop equivalence, and transfer-inclusive CPU/GPU timing rows |
| Differentiation and HPO | Value, gradient, JVP, VJP, HVP, implicit and hypergradient products over model parameters, bases, preprocessing, kernels, likelihoods, optimizer state, schedules, validation/early-stopping decisions, and search coordinates; FortOpt bounded L-BFGS-B and multistart consume the same registry | Adjoint identities, central differences against an independent oracle, declared third-derivative and nonsmooth limits, transactional optimizer refusal, and a capability matrix generated from the registry rather than hand-maintained claims |

The inventory also governs the benchmark repository. Every row records build,
compile/warm-up, steady-state, memory, transfer, precision, compiler, device,
and source revisions. CUDA/OpenACC measurements are split into resident and
transfer-inclusive classes. A typed unavailable row is evidence of the
boundary, never a performance result.

### 2026-08-07 closure slice

The current release adds several cross-package contracts that were previously
only listed as gaps:

- `softmax_training_objective_t` supplies weighted multinomial cross-entropy,
  feature-only L2, packed-parameter gradients, scalar JVP/VJP products, and
  mixed parameter/L2 HVPs. Direct L2 or bounded positive log-L2 coordinates
  share the FortOpt callback and bounded L-BFGS-B path; the independent
  transformed-product fixture is `test_softmax_log_hyperparameter`.
  The release lane is `results/softmax_training.csv` in `fortml-bench`.
- MLPs expose stable named parameter blocks with ranges, shapes, and
  trainable/buffer roles. This is the selector seam required by Flux/Lux-style
  functional training state; aliases, tied parameters, and full buffer routing
  remain open.
- `mlp_binary_classifier_t` adds a one-logit sigmoid head with weighted BCE,
  deterministic Adam minibatches, early stopping, packed input/parameter
  JVP/VJP products, exact loss HVPs, and an explicit CUDA refusal. The shared
  `mlp_multilabel_classifier_t` now emits all indicator logits from one MLP,
  supports per-label thresholds, mean-reduced BCE products, exact parameter
  HVPs, and the same typed CUDA refusal. The calibrated neural head is now a
  separate deterministic composition contract; resident-GPU neural heads
  remain open.
  The release evidence is `results/MLP_BINARY_CLASSIFIER.md` in `fortml-bench`.
- The multilabel MLP wrapper validates indicator targets, exposes concatenated
  parameter/input JVP/VJP products and exact mean-reduced loss HVPs, and keeps
  shared-head fitting deterministic. Its independent finite-difference/adjoint test
  is `test_mlp_multilabel_classifier`; release evidence is
  `results/MLP_MULTILABEL_CLASSIFIER.md` in `fortml-bench`.
- Exact GP regression accepts zero, constant, and linear mean templates. Mean
  coefficients are packed per output after kernel and log-noise parameters, and
  prediction and likelihood products include their analytic JVP/VJP/HVP terms.
  ARD, priors, and sparse/multitask mean routing remain open. A dedicated
  release benchmark is `results/GP_MEAN.md` in `fortml-bench`.
- XGBoost validation monitoring accepts typed validation arrays, computes
  objective-native weighted validation loss, records best iteration and loss,
  and supports restore-best or retain-all ensembles. Warm starts, serialized
  tree state, categorical/interaction policies, and resident GPU histograms remain
  open. The release lane is
  `results/XGBOOST_EARLY_STOPPING.md` in `fortml-bench`.
- XGBoost also supports deterministic without-replacement row and feature
  subsampling with positive `int64` seeds. Full fractions preserve the exact
  historical tree path. Subsampling is covered by a seed and structure oracle,
  while warm starts, serialized trees, and distributed histogram reduction stay
  open. The release evidence is `results/XGBOOST_SAMPLING.md` in
  `fortml-bench`.
- XGBoost now has a bounded `rank:pairwise` objective with explicit positive
  query/group IDs, group-isolated logistic pair loss, stable gradients and
  positive Hessians, optional pair weights, validation-group support, and
  public loss/derivative products. The independent oracle
  `test_xgboost_ranking` covers finite differences, two-item ordering, group
  isolation, and singleton refusal. Categorical/interaction policies, DART,
  GOSS/EFB, distributed growth, and resident GPU histograms remain open.
- The exact GP kernel catalog now includes an ARD squared-exponential kernel
  with one log length scale per input feature. Scalar, matrix, input-derivative,
  parameter-product, and exact-GP likelihood paths share the same packed
  parameter contract. The independent kernel/GP oracle is
  `test_gp_ard_kernel`; resident CUDA remains a typed refusal. The release
  evidence is `results/GP_ARD.md` in `fortml-bench`.
- XGBoost models now have versioned `FORTML_XGBOOST_TEXT` save/load with strict
  schema, finite-value, topology, EOF, and unknown-record validation. Round-trip
  prediction, staged margins, missing routing, and validation diagnostics are
  covered by `test_xgboost_serialization`; distributed model state remains open.
  The release evidence is `results/XGBOOST_SERIALIZATION.md` in `fortml-bench`.

- The resident CUDA dense-affine plan now exposes forward-mode `jvp` and
  reverse-mode `vjp` products for feature, weight, bias, and output cotangents.
  Its native kernels cover the eight resident CUDA MLP activation codes (the
  CPU path additionally covers stable sigmoid and Mish), keep the resident
  layer resident, and have independent CPU value/JVP/VJP oracles. It now also has a
  resident full-batch mean-squared-error update with parameter snapshots and
  explicit successful host/device byte counters; ordinary builds retain typed
  `FORTNUM_NOT_IMPLEMENTED` refusals. Multi-layer optimizer state, HVP, and
  resident general-MLP training remain open.
  The gate is `test/run_cuda_dense_plan.sh`; the contract is documented in
  `docs/CUDA_DENSE_PLAN.md` and `docs/API.md`.
- `basis_pipeline_training_objective_t` now optimizes a composable basis map
  and multi-output linear coefficients in one packed CPU objective. It exposes
  analytic value/gradient/JVP/VJP/HVP products, ridge regularization, a FortOpt
  callback, and a typed CUDA refusal. An optional final nonnegative packed
  coordinate now makes the ridge coefficient differentiable, including exact
  mixed ridge/coefficient HVP blocks. The independent fixtures are
  `test_basis_pipeline_training`; release evidence is
  `results/BASIS_PIPELINE_TRAINING.md` in `fortml-bench`.
- `gp_classification_t` now exposes fixed-state kernel-parameter JVP/VJP
  products for latent predictions and observed probabilities for both logistic
  and probit Laplace models. The covariance solve, cross-covariance, and prior
  blocks are differentiated analytically while the fitted Newton mode is held
  fixed; an independent parameter-product oracle covers the adjoint and shape
  refusal contracts.
- `gp_multiclass_classification_t` now packs one fixed-state kernel-parameter
  block per sorted OVR class. Latent margins and simplex-normalized observed
  probabilities expose parameter JVP/VJP products, with the quotient adjoint
  applied before per-class accumulation. CPU dispatch, shape validation, and
  CUDA refusal are covered by an independent cross-kernel finite-difference
  and adjoint oracle in `test_gp_multiclass_classification`.
- `lightgbm_t` now accepts weighted validation data and reports validation loss,
  best iteration, best loss, and early-stopping state for regression and binary
  objectives. Patience, minimum improvement, restore-best versus retain-all
  behavior, and malformed validation input are independently checked by
  `test_lightgbm_early_stopping`; the release app is
  `app/fortml_bench_lightgbm_early_stopping.f90`. Resident histogram CUDA
  execution remains an explicit refusal.
- `fortml_mlp_adagrad_schedule_hypergradient` differentiates a fixed full-batch
  unfactored Adagrad trajectory through the accumulator, denominator, typed
  schedule, and validation objective. The packed outer vector is
  `[log(base_rate), log(l2), log(epsilon), logit(min_rate_fraction),
  logit(decay_factor)]`; JVP, scalar VJP, central-difference, FortOpt
  L-BFGS-B, malformed-option, and CUDA-refusal behavior are covered by
  `test_mlp_adagrad_schedule_hypergradient`.
- `mlp_binary_training_objective_t` now packages weighted BCE, optional
  sample/class weights, exact parameter/L2 JVP/VJP/HVP products, and a bounded
  `mlp_binary_optimize_lbfgsb` FortOpt path. The independent fixture
  `test_mlp_binary_objective` covers contractions, finite-difference products,
  and convergence/refusal behavior; resident CUDA training remains refused.
- `mlp_classifier_training_objective_t` now provides the corresponding
  multiclass weighted softmax cross-entropy contract. It packs logits-network
  parameters with an optional L2 coordinate, differentiates the softmax and
  MLP second-order terms analytically, and routes the same value/gradient
  callback through bounded FortOpt L-BFGS-B. The independent
  `test_mlp_classifier_objective` checks weighted value, JVP/VJP duality, HVP
  central differences, and bounded optimization; resident CUDA classifier
  training remains an explicit typed refusal. Release evidence is
  `results/MLP_CLASSIFIER_OBJECTIVE.md` in `fortml-bench`.
- `gp_variational_classification_t` now exposes latent/probability prediction
  and packed parameter JVPs, and
  `gp_variational_multiclass_classification_t` composes sorted-label OVR
  logistic/probit posteriors with simplex probabilities, deterministic ties,
  summed ELBO products, and explicit CUDA refusal. Independent fixtures are
  `test_gp_variational_classification` and
  `test_gp_variational_multiclass_classification`; coupled categorical,
  natural-gradient, and resident-GPU inference remain open.
- `mlp_train` now accepts a stateless typed schedule through
  `options%use_typed_schedule` and `options%typed_schedule`, in addition to
  the existing callback seam. Built-in schedules are validated once and
  evaluated analytically at each optimizer update, with callback/typed
  conflicts refused. The schedule kind, structural counts, and continuous
  fractions are captured by the version-5 in-memory checkpoint and schema-3
  formatted checkpoint, so native and serialized resumes reject a changed
  schedule instead of silently changing the trajectory. The independent
  `test_mlp_typed_schedule` fixture checks exact rate-history values, native
  and serialized replay, malformed schedules, and callback conflicts. This
  closes the CPU typed-schedule trainer boundary; CUDA-resident schedule
  execution and stochastic/device hypergradients remain open.

The FortBO and FortMC companion pins were rechecked against their remote
`main` branches on 2026-08-08: FortBO
`95495344e109b52e8562ebe7c36329e30a754688` and FortMC
`4dde0ccdc37b4c331126605406b08e1f3bda4f59`. Their roadmaps remain authoritative
for acquisition and sampling algorithms; FortML owns the posterior/log-density
protocols and does not embed sampler or acquisition state. FortBO additionally
provides analytic EI/PI/UCB/log-EI products and marginal Monte-Carlo EI/PI with
common random numbers, antithetic draws, and pathwise gradients, Sobol TuRBO
candidate generation with deterministic Thompson selection, and a FortML
adapter that chooses value-only versus derivative-observation GPs from the
history contract. FortBO now also runs its gradient-based DTuRBO in-region
acquisition search end to end with Sobol multistarts and FortOpt L-BFGS-B,
beating random search on the Branin fixture at equal budget. The current pin
also supplies exact posterior mean Hessians from derivative predictions, an
indefinite-curvature bound-constrained quadratic subproblem, multi-objective
Pareto archives with exact hypervolume and scalarizations, and stopping rules
that report a machine-readable reason. Posterior-
derivative DTuRBO mode 2, full batch and knowledge/entropy/noisy acquisitions,
device execution, and wider sparse, variational, and multi-output adapters
remain open. Any future adapter must add
a focused oracle, typed GPU/refusal row, and a benchmark record in the companion
harness.

## Bayesian ecosystem split

FortML remains the owner of probabilistic ML objects. The probability layer is
not a fourth repository: distributions, constraints, transforms, priors,
likelihoods, posterior objects, predictive sampling, variational objectives,
and generic differentiable log-density/model protocols live inside FortML and
are shared by its GP, neural, regression, and classification models.

Two companion repositories own algorithms that should not be coupled to the
full estimator catalog:

| Repository | Responsibility | Dependencies |
| --- | --- | --- |
| [`fortmc`](https://github.com/lazy-fortran/fortmc) | MCMC and Monte Carlo inference: Metropolis, slice, HMC, NUTS, SMC, chain state, diagnostics, warmup, and checkpoint/resume | FortML probability/model protocols, FortNum, FortAD, FortSym, FortOpt |
| [`fortbo`](https://github.com/lazy-fortran/fortbo) | Bayesian optimization: posterior-aware acquisitions, sequential design, constraints, batch/fantasy policies, and candidate search | FortML posterior protocols, FortNum, FortAD, FortSym, FortOpt; optional FortMC integration |

The dependency direction is deliberately one-way:

```text
FortNum + FortAD + FortSym + FortOpt
                 ↓
              FortML
          ↙              ↘
       FortMC            FortBO
```

FortML must not depend on FortMC or FortBO. The current FortMC boundary accepts
only a position-valued `value` and a position-gradient `gradient`; packed
parameter registries, transforms, HVPs, samplers, and chain state remain in its
roadmap. FortBO now exposes a capability-gated posterior protocol, durable
gradient-aware history, and normalized continuous/integer/categorical/mixed/
conditional search spaces, analytic EI/PI/UCB/log-EI, and marginal Monte-Carlo
EI/PI with CRN, antithetic, and pathwise-gradient products. It also supplies
Sobol TuRBO candidates, discrete Thompson selection, a tested FortML
value-only/derivative-observation GP adapter, and a tested gradient-based
DTuRBO in-region optimizer. Posterior-derivative DTuRBO mode 2, full batch and
knowledge/entropy/noisy acquisitions, device execution, and sparse/variational/
multi-output adapters remain in its roadmap. Adapters map FortML posterior
moments and derivative observations into those contracts without adding sampler
or acquisition state to every estimator.

All three repositories use MIT licensing. FortAD is the default source of
general derivatives. FortSym is preferred for compact fixed transition,
acquisition, likelihood, and reduction kernels when it can prove the identity
and generate a smaller implementation. FortNum owns numerical kernels, RNG,
linear algebra, reductions, and device-safe array primitives. FortOpt owns
local optimization, including L-BFGS-B, line searches, multistart, and bounded
inner solves. Every differentiable path must expose an independent analytic,
finite-difference, adjoint, or pinned external oracle. Every GPU path must keep
the complete operation graph resident or return a typed refusal; OpenACC is the
first choice when it preserves semantics, and native CUDA is reserved for
fixed no-autodiff hot loops where OpenACC cannot.

The dependency pins used by the current GNU verification are FortAD `5f879a6`,
FortSym `1477a6d`, and FortOpt `bfbf1fc`, all checked against their remote
`main` branches on 2026-08-08. Generated derivatives record the exact FortSym
revision and source hash; model-level autodiff uses the same FortAD `main` pin.

The companion implementation roadmaps are authoritative for their sampler and
acquisition work packages:

- [`fortmc/ROADMAP.md`](https://github.com/lazy-fortran/fortmc/blob/main/ROADMAP.md)
- [`fortbo/ROADMAP.md`](https://github.com/lazy-fortran/fortbo/blob/main/ROADMAP.md)

The companion repositories were checked on 2026-08-08 at FortMC
`4dde0ccdc37b4c331126605406b08e1f3bda4f59` and FortBO
`95495344e109b52e8562ebe7c36329e30a754688`, both on their `main` branches. The
FortBO pin now includes a versioned capability-gated posterior contract,
gradient-aware observation history/checkpointing, normalized continuous/integer/
categorical/mixed/conditional search spaces, a differentiable-coordinate mask,
analytic EI/PI/UCB/log-EI, and marginal Monte-Carlo EI/PI with CRN, antithetic
draws and pathwise gradients, Sobol TuRBO candidates, Thompson selection,
gradient-based DTuRBO in-region acquisition search, exact posterior mean
Hessians, an indefinite-curvature quadratic subproblem, Pareto archives with
exact hypervolume, scalarizations, machine-readable stopping reasons, and FortML
value/derivative-GP adapters; refresh these pins when
their protocol or device contracts change.

The current FortBO checkout builds and runs all 14 registered tests with `fo
test`. FortMC's current checkout builds cleanly and reports zero registered
tests, so its sampler and diagnostics claims remain roadmap items rather than
FortML verification evidence.

Both pinned companion revisions also clarify that FortSym is a generation-time
dependency: generated/proven kernels may be checked into or consumed by the
runtime packages, but FortBO and FortMC do not require the FortSym executable
or module at runtime. FortML follows the same boundary for fixed kernels and
records the FortSym revision in the generated-artifact provenance. This keeps
the three packages buildable independently while preserving a proof trail for
CUDA and other no-autodiff kernels.

FortML work packages that depend on these projects must add a focused adapter,
an independent oracle, and a benchmark row rather than embedding a second
MCMC or Bayesian-optimization implementation.

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

### Production architecture contract

The clean-break API is organized around five composable contracts. Every new
estimator, basis, likelihood, or network must implement the contracts that its
algorithm supports and must expose a typed refusal for the rest.

| Contract | Required state | Required products | Device rule |
| --- | --- | --- | --- |
| Data and preprocessing | row-oriented arrays, masks, weights, category metadata, fitted statistics | `fit`, `transform`, inverse transform where defined, input JVP/VJP, leakage and shape checks | host callbacks remain host-only; static maps lower through the device plan |
| Estimator and parameter tree | immutable topology, flat trainable registry, buffers, fitted state, clone/reset metadata | fit or partial-fit, predict/score, packed parameters, status, deterministic serialization seam | model, buffers, and batches must be resident for a GPU claim |
| Objective and derivatives | reduction, regularization, constraints, active-set/split boundary, state snapshot | value, gradient, JVP, VJP, HVP or a named refusal; adjoint and finite-difference oracles | derivative kernels use FortSym/FortAD products or a typed device refusal |
| Training and search | batches, RNG cursor, optimizer moments, schedules, validation, callbacks, checkpoints | resumable step, hyperparameter products, bounded FortOpt L-BFGS-B when gradients are complete | optimizer state and transfer counters are part of the device contract |
| Backend and evidence | CPU reference, OpenACC/native-CUDA plan, compiler/device metadata | resident and transfer-inclusive execution, memory counters, reproducible benchmark record | OpenACC is preferred when it preserves semantics; fixed no-autodiff hot loops may use CUDA C++ |

The parameter registry is the only route between models, pipelines, derivative
products, and FortOpt. A pipeline owns named child registries and maps their
offsets into one deterministic vector. A trainer owns optimizer state but never
reaches into private model arrays. A device plan owns residency and exposes
every transfer. This prevents a benchmark from measuring a hidden host copy
and prevents a hypergradient from silently omitting a schedule or validation
variable.

### Variant coverage matrix

The parity target includes the following independent model variants. The matrix
is intentionally explicit so a binary implementation cannot be mistaken for
multiclass, weighted, probabilistic, derivative, or GPU coverage.

Calibration update (2026-08-07): `CALIBRATION_TEMPERATURE` now fits a positive
scalar temperature for binary, pre-oriented logits and exposes exact score and
temperature-parameter JVP/VJP products. Multiclass temperature scaling,
reliability diagrams, and calibration-aware cross-validation remain open.

| Family | Required variants | Current FortML baseline | Missing production gates |
| --- | --- | --- | --- |
| Classification | binary, multinomial/softmax, OVR, OVO, multilabel, Naive Bayes, LDA/QDA, tree, neural, dense RBF one-class SVM, Laplace GP, variational GP, calibrated, ordinal | binary/softmax/OVR/OVO/multilabel and weighted ordinal cumulative-logit heads, weighted Gaussian/Bernoulli/Multinomial/Complement/Categorical NB, weighted LDA/QDA, CART, MLP, dense RBF nu-SVM, binary/OVR Laplace GP, bounded Bernoulli variational GP (logistic/probit) including sorted-label OVR multiclass, positive temperature, Platt sigmoid, and weighted PAVA isotonic calibration, exact and histogram boosted trees | sparse/multioutput multilabel, ordinal GP and coupled/categorical variational GP likelihoods, kernel-SVM parity, resident GPU training, shared preprocessing/search |
| Regression | OLS, weighted/ridge/lasso/elastic-net, robust, quantile, GLM, multi-output, partial-fit | dense OLS, weighted ridge, weighted elastic-net/lasso, weighted linear SVR, weighted Poisson/Gamma log-link GLM, exact/histogram XGBoost-style squared/squared-log (RMSLE)/Huber/quantile/absolute regression with fixed-state products, multi-output fixed-fit products | Tweedie, positive/Bayesian/ARD, partial-fit, fit-time/hyperparameter products, resident GPU kernels |
| Ensembles | CART, random/extra forests, bagging, AdaBoost, histogram boosting, XGBoost/LightGBM ranking/categorical/DART | weighted CART, deterministic seeded random-forest and randomized-threshold Extra-Trees classification, binary AdaBoost over weighted CART, squared/squared-log/Huber/quantile/absolute boosting, exact and bounded histogram second-order XGBoost-style binary/OVR, bounded `rank:pairwise`, exact/histogram per-feature monotonic and interaction-group constraints, and a bounded weighted LightGBM-style leaf-wise regression/binary path with validation loss, patience/min-delta early stopping, best-iteration metadata, restore-best slicing, and weighted validation objectives | bagging, multiclass AdaBoost/SAMME, categorical partitions, DART/GOSS/EFB, distributed and resident GPU histograms |
| Gaussian processes | exact, derivative observations, multitask, sparse/variational, SKI/lazy, local experts, classification | exact and derivative GPs with RBF, Matérn, periodic, rational-quadratic, cosine, polynomial, linear, constant, white-noise, user, sum, and product leaves, sparse/local/SKI/structured operators, binary/OVR Laplace classification, bounded Bernoulli variational classification with logistic/probit ELBOs, packed sparse-GP mean/log-Cholesky ELBO gradients/JVPs/VJPs, fixed-state binary and OVR Laplace latent/probability kernel-parameter JVP/VJP products, and fixed-state binary/OVR variational kernel-log products | full likelihood/kernel catalog, batch/multitask/coupled variational classification, inducing-state and likelihood hyperparameter products, implicit derivatives, natural gradients, resident GPU solves |
| Neural and physics models | MLP, CNN, RNN/GRU/LSTM, attention, autoencoder/VAE, BNN, HNN/LNN/symplectic/PINN | MLP/MLP classifier, named sequential `mlp_chain_t`, BNN, VAE, vanilla RNN, Hamiltonian MLP, composable four-slot physics residual objective, weighted Poisson objective with exact HVP/L-BFGS-B, Lion fixed-trajectory hypergradients, selected optimizer hypergradients, and an exact fixed-trajectory scheduled-Adagrad hyperparameter objective | broader module tree, recurrent/attention/convolution families, full products, complete loss catalog, GP/linear initialization, physics samplers/adapters, and long-horizon GPU gates |

### Capability matrix (target versus current state)

| Area | Current state | Production target |
| --- | --- | --- |
| Linear regression and generalized linear models | Linear regression, weighted ridge and weighted elastic-net/lasso coordinate descent, weighted Poisson/Gamma log-link GLMs with bounded FortOpt L-BFGS-B, and logistic/softmax sample and positive sorted-class weights are implemented | Robust, quantile/Tweedie, multinomial, calibrated and regularized classifiers with shared solver and derivative contracts; resident GLM GPU kernels |
| Feature transforms and basis maps | Polynomial, Fourier, radial, B-spline, callback bases, standard/min-max/median-IQR scalers, sparse-safe CSC standard scaling, integer categorical one-hot encoding, horizontal/sequential/column pipelines, analytic basis/pipeline HVPs, a fitted basis-to-linear estimator, and a joint differentiable basis-pipeline training objective are implemented. The objective can also pack a nonnegative ridge coordinate with exact gradient and mixed HVP blocks | CSR/CSC categorical and indicator views, DAG pipelines, leakage-safe cross-validation, callback second derivatives, and resident GPU transforms |
| Nearest-neighbor and margin methods | Dense exact kNN and closed-radius classification plus scalar and multi-output regression, weighted linear SVM/SVR, and dense RBF one-class SVM | KD-tree or ball-tree search, sparse inputs, kernel SVM/SVR, calibrated probabilities, resident GPU kernels, and differentiable soft-neighbor policies |
| Trees and ensembles | Partial | Deterministic finite-only regression stumps, weighted depth-limited CART regression and classification, seeded bootstrap random-forest classification, seeded randomized-threshold Extra-Trees classification, binary AdaBoost over weighted CART, squared-loss stump boosting, exact/histogram depth-limited second-order squared/logistic/Poisson/squared-log/Huber/quantile boosting, and bounded `rank:pairwise` boosting are implemented. XGBoost-style trees support weighted quantile cuts, bounded histograms, explicit NaN rejection, learned default directions, forced-left/right routing, per-feature monotonic and interaction-group constraints with recursive leaf bounds/masks, staged predictions, contributions, serialization, and transactional fitted-prefix slicing; bagging, multiclass AdaBoost/SAMME, categorical, DART/GOSS/EFB, distributed growth, and resident GPU histograms remain planned |
| Clustering and unsupervised learning | Centered dense `pca_t` is implemented with deterministic SVD signs, rank selection, whitening, reconstruction, variance metadata, and fixed-state input products; `linear_autoencoder_t` reuses fitted PCA as the tied linear optimum with exact encode/reconstruction JVPs; deterministic dense seeded `kmeans_t` provides fit/predict/transform, inertia, and fixed-center input products with explicit empty-cluster and device refusals | Incremental/randomized/sparse/kernel PCA, ICA, NMF, minibatch k-means, Gaussian mixtures/EM, density and graph clustering, manifold methods, outlier detection, matrix factorization, and density metrics |
| Neural networks | MLP/BNN/VAE/RNN primitives, a separable Hamiltonian MLP, a named sequential `mlp_chain_t` parameter tree, dense MLP linear/`tanh`/ReLU/GELU/SiLU/ELU/softplus/leaky-ReLU/sigmoid/Mish products, deterministic MLP Adam/AdamW/Adagrad/RMSprop/SGD/Adafactor training, exact fixed full-batch SGD momentum/Nesterov/AdamW/Adam/RMSprop/Adagrad/Adafactor trajectory hypergradients including Adafactor relative-step and parameter-scaling smooth branches, weighted binary BCE, multiclass cross-entropy, and Poisson log-rate FortOpt/L-BFGS-B objectives with exact mixed HVPs, bounded full-batch MLP and composed-chain L-BFGS-B paths, named group-wise log-L2 hyperparameters with exact mixed HVPs, portable trainer checkpoints, resident dense-affine CUDA value/JVP/VJP plus single-layer MSE-update primitives, and a resident no-autodiff CUDA Adagrad state plan exist | Alias-aware module/buffer tree, the remaining activation/loss/module catalog, convolution/attention/sequence/graph extensions, mixed precision, distributed training, compile/fusion, serialized/distributed trainers, and resident multi-layer neural training |
| Gaussian processes | Exact, derivative, sparse, structured and local variants are partial-to-implemented. Exact fitted GPs and binary/shared-kernel one-vs-rest Laplace classifiers have bounded FortOpt L-BFGS-B adapters; bounded Bernoulli variational classification has deterministic logistic/probit ELBO, packed gradients, a bounded FortOpt L-BFGS-B adapter, prediction JVPs/VJPs, minibatch scaling, and typed CUDA refusal. Sparse variational GPs now expose packed mean/log-Cholesky ELBO gradients/JVPs/VJPs with central-difference and adjoint oracles. Binary Laplace prediction additionally exposes fixed-state kernel-parameter JVP/VJP products for latent and observed probabilities | GPyTorch/GPflow-style kernels, likelihoods, multitask/batch shapes, exact/variational/lazy inference, derivative operators, kernel/inducing hyperparameter products, constraints, calibration, coupled multiclass GP classification, natural gradients, evidence-corrected and likelihood-parameter training |
| Derivatives | Exact GP, analytic polynomial/Fourier/radial/spline basis and pipeline HVPs, and selected neural/kernel products exist | Value/JVP/VJP/HVP and implicit/hypergradients for every declared parameter/input path, including preprocessing, likelihood, optimizer/search variables, and device kernels |
| Model selection and metrics | Benchmark-specific checks exist; `fortml_validation` now accepts the shared `estimator_capability_t` contract for pre-flight validation | Shared metrics, splitters, cross-validation, calibration, grid/random/Bayesian/differentiable search, nested validation, and leakage/refusal checks |
| Persistence and serving | Missing public contract | Versioned state dictionaries, safe model/trainer serialization, compiler-independent metadata, streaming inference, batching, and reproducible deployment manifests |
| GPU and scale-out | Operator-specific OpenACC/CUDA paths; kNN has a resident native-CUDA plan, dense-affine value/JVP/VJP and single-layer MSE update have resident CUDA C plans, and direct RMSprop/AdamW/Adagrad state have resident CUDA C plans. Elastic-net, OVO, LDA/QDA, random forest, MLP-classifier prediction products, basis/pipeline HVPs, Laplace-GP (binary and OVR multiclass), probability calibration, neural losses, XGBoost (binary/OVR and robust objectives), and typed schedules expose explicit CPU/CUDA capability and typed CUDA refusals; complete RMSprop/Adagrad training, staged XGBoost, robust/discriminant/forest training, basis transforms, and GP-classification-training release rows remain CPU-only | Complete resident CPU/CUDA/OpenACC training and inference for supported estimators, mixed precision, multi-GPU/MPI sharding, transfer accounting, and deterministic reductions |
| Performance evidence | Several model/GP lanes exist | Matched correctness-gated comparisons with scikit-learn, XGBoost/LightGBM, PyTorch/JAX, GPyTorch/GPflow, and published hardware/toolchain provenance |

### Production closure ledger

This ledger is the release gate for the broad parity objective. A row moves to
**implemented** only after the API, independent behavioral oracle, refusal
tests, documentation, and benchmark record land together. A row marked
**partial** describes code that is useful in production workflows while one or
more required variants remain open.

| Subsystem | Implemented surface | Closure evidence still required |
| --- | --- | --- |
| Classification | Binary, softmax, OVR, OVO, multilabel, ordinal, five Naive Bayes variants, weighted LDA/QDA, CART, deterministic random forest, MLP, linear and dense RBF one-class SVM, temperature/sigmoid/isotonic calibration, binary or OVR Laplace GP classification, and bounded Bernoulli variational GP classification including sorted-label OVR multiclass | Sparse and multioutput labels, ordinal and coupled/categorical variational GP likelihoods, shared preprocessing and search, resident GPU training, and kernel SVM/margin parity |
| Regression and bases | OLS, ridge, elastic-net, weighted linear SVR, scalar and multi-output closed-radius neighbors with uniform or distance weighting, Poisson/Gamma GLM, PCA, polynomial/Fourier/radial/spline maps, analytic basis/pipeline HVPs, sequential or column pipelines, joint differentiable basis-pipeline training, and robust/squared-log/quantile XGBoost-style objectives | KD/ball-tree and approximate neighbors, Tweedie, partial fit, sparse views, graph serialization, callback/pipeline second derivatives, and resident GPU execution |
| Trees and boosting | Weighted CART, deterministic seeded random-forest classification, binary AdaBoost over weighted CART, exact and bounded-histogram second-order squared/logistic/Poisson/squared-log/Huber/quantile and `rank:pairwise` boosting, staged binary or OVR predictions, diagnostics, monotonic and interaction-group constraints, and transactional fitted-prefix slicing | Extra-trees, bagging, multiclass AdaBoost/SAMME, categorical partitions, DART/GOSS/EFB, distributed workers, and complete GPU histograms |
| Gaussian processes | Exact, derivative-observation, sparse, local, SKI, structured operators, periodic/rational-quadratic/cosine/polynomial leaves, binary or OVR Laplace classification, bounded Bernoulli variational classification, packed sparse-GP mean/log-Cholesky ELBO products, and fixed-state binary/OVR variational kernel-log products | Full likelihood and kernel catalog, batch or multitask shapes, coupled variational inference, inducing-state and likelihood hyperparameter products, operator-valued derivatives, implicit products, serialization, and resident GPU solves |
| Neural training | Dense MLPs, named sequential chains, BNN/VAE/RNN/Hamiltonian primitives, weighted binary BCE, multiclass cross-entropy, and Poisson log-rate FortOpt/L-BFGS-B objectives with exact mixed HVPs, six trainer optimizers including unfactored Adafactor, Lion fixed-trajectory products, schedules, portable checkpoints, fixed-trajectory hypergradients through SGD/Adam/AdamW/Adagrad/RMSprop/Adafactor including relative-step and parameter-scaling smooth branches, and contiguous optimizer groups | Complete loss and module catalog, stochastic and device hypergradients, parameter-group routing beyond the bounded CPU slice, AMP, distributed state, serialization migration, and resident GPU training |
| Differentiation | Analytic, FortSym, FortAD, JVP, VJP, HVP, and typed refusal paths for selected models | A generated capability matrix for every model, input, basis, likelihood, optimizer, schedule, validation, and transfer variable, plus implicit differentiation through solves and optima |
| Device backends | CPU reference, OpenACC operator paths, native CUDA weighted metrics, kNN, AdamW state, RMSprop state, and explicit CUDA refusals | Resident model, batch, optimizer, and derivative state for every supported estimator, deterministic reductions, mixed precision, transfer accounting, and matched accelerator benchmarks |
| Persistence and evidence | MLP checkpoint format and correctness-gated workload CSVs with source and benchmark revisions | Versioned state dictionaries for every estimator and pipeline, streaming serving manifests, cross-library performance matrices, and published memory or energy measurements |

### Parity inventory and release gates

This inventory prevents a name-only implementation from being counted as
parity. A family moves to **implemented** only when fit/predict (or
transform), weighting, arbitrary labels/categories, packed state, declared
derivatives, refusal behavior, and an independent benchmark oracle all exist.

| Reference family | FortML coverage today | Remaining release gate |
| --- | --- | --- |
| scikit-learn linear/GLM | Dense linear regression, weighted ridge/lasso/elastic-net, logistic/softmax, bounded logistic and MLP L-BFGS-B | Robust/quantile/Gamma/Tweedie, SGD estimators, solver parity, calibration, complete multioutput and partial-fit contracts |
| scikit-learn Naive Bayes | GaussianNB, BernoulliNB, MultinomialNB, ComplementNB, CategoricalNB with weighted sorted categories and unknown-category policy | sparse counts, calibrated and incremental variants |
| scikit-learn neighbors/margins | Dense exact `fortml_knn_classifier`, closed-radius `fortml_radius_neighbors_classifier`, scalar and multi-output `fortml_radius_neighbors_regression`, weighted linear `linear_svm_classifier_t`, weighted dense `linear_svr_regression_t`, dense RBF `one_class_svm_t`, and dense finite-basis `rbf_svm_classifier_t` with deterministic boundaries, fixed-state products, and explicit derivative/device refusals | KD/ball trees, approximate and sparse inputs, kernel SVR, calibrated support-vector workflows, resident GPU kernels, and smooth fit/hyperparameter products |
| scikit-learn trees/ensembles | Stumps, weighted CART with deterministic NaN routing, binary AdaBoost over weighted CART, squared boosting, exact and histogram XGBoost-style second-order squared/binary-logistic/Poisson/squared-log/absolute lanes, bounded `rank:pairwise`, per-feature monotonic and interaction-group constraints, transactional XGBoost warm-start continuation, and bounded LightGBM-style weighted leaf-wise regression/binary growth | Random/extra forests, bagging, multiclass AdaBoost/SAMME, categorical partitions, DART/GOSS/EFB, distributed growth, and staged/warm-start LightGBM APIs |
| scikit-learn unsupervised | Basis maps, centered dense PCA, deterministic seeded dense k-means, validation splitters, variational primitives | Incremental/randomized/sparse/kernel PCA, ICA/NMF/TruncatedSVD, minibatch k-means/GMM/density/manifold/outlier methods, sparse/categorical preprocessing, model persistence |
| PyTorch/JAX neural core | Dense MLP, classifier, named sequential MLP chain, BNN, VAE, RNN, Hamiltonian MLP; Adam, AdamW, Adagrad, RMSprop, Adafactor, and SGD momentum/Nesterov; exact fixed full-batch SGD learning-rate/L2/momentum including classical and Nesterov velocity state, Adam/AdamW beta and decay products, RMSprop, Adagrad, unfactored Adafactor, and typed schedule trajectory hypergradients | Stochastic/device optimizer hypergradients, complete loss/activation/module tree, convolution/attention/sequence/graph models, AMP, compile/fusion, distributed and device-resident train state |
| GPyTorch/GPflow | Exact, derivative-observation, sparse/local/SKI/structured GP primitives; Laplace binary/OVR and bounded Bernoulli variational GP classification with sorted-label OVR multiclass prediction and fixed-state kernel-log JVPs/VJPs | Kernel/likelihood/constraint/batch-shape parity, variational categorical/count likelihoods, inducing-state products, multitask, operator-valued derivatives, implicit hypergradients, serialization and resident GPU training |
| XGBoost/LightGBM | Exact and bounded-histogram depth-limited XGBoost-style squared/logistic/Poisson/squared-log (RMSLE)/Huber/quantile/absolute Newton trees, weighted binary/OVR multiclass staged predictions, bounded `rank:pairwise`, margins, gain/weight/cover diagnostics, per-feature monotonic and interaction-group constraints, serialization, fitted-prefix slicing, transactional XGBoost warm starts, and a separately named `lightgbm_t` weighted regression/binary-logistic path with shared weighted-quantile cuts and deterministic globally best-leaf growth up to `num_leaves` | Tweedie, categorical splits, DART, GOSS/EFB, distributed training, warm-start LightGBM state, and resident GPU histograms |
| Differentiability and search | Capability-specific JVP/VJP/HVP products, FortOpt L-BFGS-B for selected objectives, exact group-wise log-L2 mixed HVPs, and exact fixed-trajectory MLP hypergradients for SGD momentum/Nesterov, AdamW (including beta logits), and RMSprop | Complete derivative matrix for every declared parameter/input/hyperparameter, stochastic/device optimizer hypergradients, implicit differentiation, and refusal rather than hidden finite differences |
| Device and performance | OpenACC/native CUDA operator lanes plus explicit device control-plane contract; kNN, dense-affine value/JVP/VJP inference, and direct RMSprop state have correctness-gated native-CUDA oracles, while complete RMSprop training, staged boosting, variational GP classification, and calibration still report CPU-only rows or typed refusals | Resident model/optimizer/batch state for every supported estimator, CPU parity, transfer/memory accounting, mixed precision, and matched PyTorch/JAX/GPyTorch/XGBoost evidence |

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
- [x] Weighted multi-output/vector ridge regression with nonnegative
  regularization, optional intercept, positive-mass sample weights, packed
  coefficient state, and fixed-fit input/parameter JVP/VJP products. The SVD
  fit and rank decisions are an explicit nondifferentiable boundary.
- [x] Lasso and elastic-net estimators with deterministic weighted coordinate
  descent, multi-output packed state, fixed-fit input/parameter JVP/VJP
  products, convergence/refusal statuses, and an independent fixture oracle.
- [x] Add weighted ordinal cumulative-logit classification with arbitrary
  sorted integer labels, strictly increasing thresholds, packed coefficient /
  intercept / threshold parameters, analytic input and parameter JVP/VJP
  products, and explicit CPU/CUDA capability metadata. CUDA requests return
  `FORTNUM_NOT_IMPLEMENTED` until a resident ordinal kernel is linked.
- [ ] Huber, quantile, Poisson, Gamma, ordinal-GP, multilabel, multioutput,
  and partial-fit estimators with the shared parameter registry and
  sample-weight contract.
- [x] Dense k-nearest-neighbor classification with deterministic ties, uniform
  or inverse-distance voting, optional sample weights, and explicit
  nondifferentiable neighbor-selection boundaries.
- [x] Add dense radius-neighbor classification with a closed squared-Euclidean
  radius, uniform or inverse-distance votes, nonnegative sample weights,
  arbitrary sorted integer labels, deterministic probability ties, an
  in-training outlier-label policy, and explicit nondifferentiable
  selection-boundary products. CPU behavior has an independent hand oracle;
  CUDA returns `FORTNUM_NOT_IMPLEMENTED` until a resident radius kernel is
  linked.
- [x] Add multi-output radius-neighbor regression with uniform or inverse-
  distance weighting, nonnegative sample weights, vector outlier values,
  output-count metadata, fixed-fit input JVP/VJP products, exact-radius
  derivative refusals, and typed CUDA refusal. The independent fixture is
  `test_radius_neighbors_multioutput_regression`; release evidence is
  `fortml-bench/results/RADIUS_NEIGHBORS_MULTIOUTPUT.md`.
- [ ] Add brute/KD-tree/ball-tree backends, sparse inputs, leave-one-out
  scoring, and resident GPU radius search.
- [x] Linear SVM classification with arbitrary binary integer labels,
  nonnegative sample weights, feature-only L2 regularization, deterministic
  FortOpt L-BFGS-B fitting, signed decision/prediction APIs, packed affine
  parameter/input JVP/VJP products, ordinary and squared hinge objectives, and
  explicit exact-margin split and CUDA refusals. Kernel SVM, ranking,
  probability calibration, and support-vector metadata remain open.
- [x] Add a dense RBF `one_class_svm_t` with the standard nu-SVM capped-simplex
  dual, deterministic projected-gradient fitting, KKT offset selection,
  support-weight metadata, signed anomaly scores, fixed-state query JVP/VJP
  products, an independent RBF/dual-constraint oracle, and explicit CPU/CUDA
  capability boundaries. Fit active-set and hyperparameter derivatives remain
  typed roadmap gaps rather than hidden finite differences.
- [x] Weighted dense linear SVR with arbitrary real targets, nonnegative sample
  weights, feature-only L2 regularization, squared and ordinary
  epsilon-insensitive losses, deterministic FortOpt L-BFGS-B fitting, packed
  affine prediction JVP/VJP products, exact objective hypergradients, ordinary
  epsilon-kink refusal, and explicit CUDA refusal. Kernel SVR, one-class SVM,
  and support-vector metadata remain open.
- [ ] Calibration workflows: reliability diagrams, class weighting, and
  calibration-aware cross-validation. Positive-temperature, Platt/sigmoid,
  and weighted isotonic fitting now have independent derivative/refusal tests;
  multiclass temperature maps, calibration curves, and cross-validation remain.
- [x] Deterministic seeded random-forest and randomized-threshold Extra-Trees
  classifiers provide aligned probabilities, arbitrary integer labels, Gini or
  entropy criteria, depth/leaf controls, positive sample weights, seeded
  reproducibility, and explicit CPU/CUDA device contracts. Bagging, AdaBoost,
  random patches, histogram gradient boosting, staged/warm-start APIs,
  feature importance, missing/categorical values, monotonic constraints, and
  permutation importance remain open.
- [x] Add deterministic binary AdaBoost over weighted CART weak learners.
  `adaboost_classifier_t` retains arbitrary sorted integer labels, fits
  weighted depth-limited trees, exposes signed margins and stable probabilities,
  and stops on a perfect learner or rejects a first learner at chance. The
  independent `test_adaboost_classifier` fixture checks the one-stump
  weighted-error/alpha oracle, simplex, labels, split-routing JVP refusal, and
  typed CUDA refusal. Multiclass SAMME/SAMME.R, staged explanations, and
  resident GPU boosting remain open.
- [x] Binary and one-vs-rest XGBoost-style staged predictions, cumulative
  margins, gain/weight/cover feature-importance diagnostics, and fixed-tree
  input JVP/VJP products with split-boundary refusals.
- [x] Add additive XGBoost base-margin and per-tree contribution predictions
  for regression and logistic objectives, staged-margin equivalence, explicit
  CPU/CUDA dispatch, and malformed-shape refusals. Split-routing and tree
  structure derivatives remain the declared nondifferentiable boundary.
- [x] Add transactional XGBoost warm-start continuation. `fit_warm_start`
  retains a fitted prefix, validates objective and control metadata, grows only
  the requested suffix, records requested versus retained tree counts, and
  refuses changed controls, invalid targets, and unfitted sources without
  mutating the prefix. `test_xgboost_warm_start` and the
  `xgboost_warm_start.csv` release lane provide staged-margin and independent
  Newton-stump oracles. Distributed reduction and resident CUDA remain open.
- [x] Add XGBoost interaction groups. `xgboost_options_t%interaction_groups`
  records positive per-feature groups, recursive path masks prevent features
  from different groups sharing a root-to-leaf path, and fitted metadata is
  preserved across warm starts, slices, and version-2 text save/load. The
  independent `test_xgboost_interaction_constraints` oracle and
  `fortml-bench/results/xgboost_interaction.csv` release lane cover the CPU
  path and typed CUDA refusal.
- [ ] XGBoost and LightGBM parity beyond the completed exact/histogram growth,
  validation, ranking, monotonic, sampling, serialization, warm-start, and
  interaction-group slices: quantile-sketch equivalence, categorical
  partitions, DART/GOSS/EFB, distributed workers, warm-start LightGBM state,
  and resident GPU histograms.

### Gaussian processes and probabilistic models

- [x] RBF, Matérn 1/2, 3/2, and 5/2, periodic, rational-quadratic, cosine,
  polynomial, linear, constant, white-noise, user-formula, sum, and product
  kernels with exact value, input derivatives, and parameter JVP/VJP/HVP
  products. The new cosine and polynomial leaves have independent
  finite-difference and adjoint tests; their dense device ABI remains a typed
  refusal until resident kernels are linked.
- [x] Add a GPyTorch-compatible spectral-mixture kernel with positive
  log-weight/log-scale coordinates and signed frequency means. The packed
  per-mixture metadata is `[log_weight,log_scale(:),mean(:)]`; dense values,
  input gradients/mixed Hessians, parameter JVP/VJP/HVP products, composition,
  and exact-GP fit/predict integration are analytic and independently checked
  by `test_gp_spectral_mixture_kernel`. CUDA remains an explicit typed refusal
  until a resident spectral-mixture kernel is linked; see
  `docs/GP_SPECTRAL_MIXTURE.md`.
- [ ] Complete kernel catalog: locally
  periodic, change-point, neural-network, graph,
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
- [x] Add a bounded Bernoulli variational-classification slice:
  `gp_variational_classification_t` owns inducing-point `q(u)`, deterministic
  seeded logistic/probit ELBO samples, analytic KL and packed variational
  gradients, exact deterministic prediction JVPs/VJPs, minibatch likelihood scaling, and an
  explicit CUDA refusal until the inducing solve and reductions are resident.
  The independent finite-difference/JVP/VJP oracle is
  `test_gp_variational_classification`. A sorted-label one-vs-rest wrapper,
  `gp_variational_multiclass_classification_t`, now packs one independent
  inducing posterior per class, sums the OVR ELBO and gradients, exposes
  latent margins, simplex-normalized probabilities, packed-parameter JVPs/VJPs,
  deterministic ties, and explicit CUDA refusal; its independent behavioral
  oracle is `test_gp_variational_multiclass_classification`. Kernel/inducing
  hyperparameter products, natural gradients, coupled categorical likelihoods,
  and resident GPU inference stay open.
- [x] Extend the Bernoulli variational-GP binary contract with exact query-input
  JVP/VJP products for latent marginals and corrected probabilities. The
  products differentiate the cross-kernel and Schur-complement variance with
  analytic kernel input derivatives, and
  `test_gp_variational_classification_input` checks central differences,
  adjoints, logistic/probit branches, CPU dispatch, and typed CUDA refusal.
  Hyperparameter products and resident GPU inference remain open.
- [x] Add a bounded FortOpt L-BFGS-B adapter over the packed Bernoulli
  variational-GP ELBO. `gp_variational_classification_optimize` exposes
  explicit bounds/tolerances, commits the packed state on convergence, reports
  ELBO/gradient/iteration diagnostics, restores the initial packed state on
  optimizer refusal or nonconvergence, and refuses CUDA until resident
  inducing solves and reductions exist. The independent convergence,
  finite-difference-gradient, and typed-refusal oracle is
  `test_gp_variational_classification_training`.
- [x] Add packed mean/log-Cholesky variational parameters to `sparse_gp_t`.
  Its Gaussian ELBO exposes analytic value, gradient, directional JVP, and
  scalar VJP products, with central-difference and adjoint checks in
  `test_sparse_gp`; CPU dispatch is implemented and resident CUDA returns a
  typed refusal. Kernel/inducing-point hyperparameter products, natural
  gradients, and coupled likelihoods remain open.
- [x] Add fixed-state variational-classification kernel-log-parameter JVP/VJP
  products for binary and sorted-label OVR models. Latent margins and
  simplex-normalized probabilities include the inducing solve, cross-kernel,
  predictive variance, KL, and quotient-rule adjoint terms. The independent
  `test_gp_variational_kernel_products` fixture covers central differences,
  duality, simplex preservation, and typed CUDA refusal; inducing-state,
  likelihood, natural-gradient, and resident-GPU products remain open.
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
  checkpointable Adam, AdamW, Adagrad, RMSprop, unfactored Adafactor, and SGD
  training.
- [x] Add the first production module-tree slice: `fortml_mlp_chain` owns
  named sequential `mlp_t` children, validates interface widths, exposes stable
  stage parameter ranges, and routes exact composed value/JVP/VJP/HVP products
  through one flat parameter vector. `mlp_chain_objective_t` and its bounded
  FortOpt L-BFGS-B adapter consume the same all-stage gradient and optional L2
  hyperparameter block. The broader tree contract (buffers, cloning, freezing,
  tied weights, parameter groups, hooks, train/eval modes, and resident GPU
  lowering) remains open.
- [ ] Convolution, transposed convolution, pooling, normalization, dropout,
  embeddings, attention, transformers, residual blocks, recurrent LSTM/GRU,
  temporal convolutions, graph message passing, and neural operators.
- [x] Add a checked `mlp_t%initialize_linear`/`set_linear_parameters` seam for
  finite two-layer affine or PCA-seeded states. It derives the topology from
  the supplied `(weight,bias)` blocks, requires linear activations and exact
  dimensions, preserves packed column-major layout, and has an independent
  prediction/transaction/refusal oracle. This is not an NNGP, NTK, or
  GP-posterior initializer; those structure-aware mappings remain open in
  WP9d.
- [x] Extend dense MLP value/JVP/VJP/HVP products with linear, `tanh`, ReLU,
  tanh-approximate GELU, SiLU, ELU, softplus, fixed-slope leaky ReLU, stable
  sigmoid, and Mish activations. Independent value and central-difference
  first/second-product tests cover every activation; the complete CUDA path
  remains explicitly refused until resident activation and dense-gradient
  kernels are linked.
- [ ] Complete the remaining activation and loss catalog: softmax, log-softmax,
  focal, multilabel BCE variants, Poisson count/dispersion variants, Huber,
  quantile, contrastive, triplet, CTC, and
  physics residual losses, each with explicit derivative and refusal contracts.
- [x] Add a weighted one-output Poisson log-rate objective with exact value,
  JVP, VJP, HVP, optional L2 coordinate, and a bounded FortOpt L-BFGS-B
  adapter. `test_mlp_poisson_objective` checks finite differences, adjoint
  scaling, convergence, and typed CUDA refusal; the API contract is in
  `docs/MLP_POISSON.md`.
- [x] Production RMSprop with centered/uncentered running statistics, optional
  momentum, exact checkpoint/resume state, and an independent recurrence oracle.
- [ ] Production optimizers and schedules: matrix-factored
  Adafactor,
  Lion, RAdam, AMSGrad, cosine/one-cycle/warmup/plateau schedules, gradient
  accumulation, clipping, EMA, decoupled regularization, and parameter groups.
  The deterministic mini-batch SGD trajectory objective now records a private
  batch cursor (including seeded epoch shuffles), exposes exact learning-rate
  and L2 hypergradients through validation MSE, and is consumable by FortOpt
  L-BFGS-B. The bounded CPU optimizer-group slice validates contiguous ranges,
  names, multipliers, and checkpoint metadata; general stochastic-loader,
  clipping, validation-policy, migration, and resident-device products remain
  open.
- [x] Add the fixed full-batch Lion trajectory hypergradient. The packed
  coordinates are log learning rate, log L2, and beta logits; analytic
  MLP-HVP products feed a bounded FortOpt L-BFGS-B adapter. Sign-margin and
  CUDA requests return typed refusals. `test_mlp_lion_hypergradient` and
  `fortml-bench/results/mlp_lion_hypergradient.csv` cover the CPU oracle.
- [x] Add deterministic unfactored Adafactor to the model-agnostic trainer and
  dense MLP trainer. The explicit squared-gradient state, update-RMS clipping,
  optional relative-step/parameter scaling, schema-versioned checkpoint state,
  and independent quadratic/MLP recurrence and resume oracles are complete.
  Matrix-layout factored state, optimizer-trajectory hypergradients, and
  resident CUDA Adafactor remain separate capability gates.
- [ ] Exact fixed-trajectory and implicit hypergradients through all supported
  optimizers, schedules, batch cursors, clipping, weight decay, validation,
  early stopping, and optimizer state. Fixed full-batch SGD/AdamW/Adagrad/
  RMSprop and deterministic mini-batch SGD now have analytic trajectory
  products. Every unsupported stochastic or device path must return a typed
  refusal.
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
- [ ] Versioned state dictionaries, checksums, safe loading, streaming/batched
  inference, model cards, and deployment manifests for models, pipelines,
  kernels, and estimator state. A compiler-independent versioned text format
  now covers the complete in-memory MLP training checkpoint, including all
  optimizer/iterator/history state and malformed-file refusals.
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
  regularization, an SVD least-squares fit, and prediction JVP/VJP products;
  weighted ridge and elastic-net/lasso coordinate fits share the packed
  prediction derivative contract.
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
approximate-GP paths remain partial. Derivative-GP query-input JVP/VJP
products are analytic third-input rules for the supported smooth leaves;
value-only derivative-GP likelihood HVPs use the analytic kernel-HVP path,
while mixed-observation HVPs retain a documented deterministic central
difference until input-parameter second products are complete. The full
kernel/refusal matrix is maintained in
[docs/GP_DERIVATIVES.md](docs/GP_DERIVATIVES.md).

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

Accelerator support is operator-specific, and CPU-only rows in the benchmark
reports are provisional rather than parity evidence. Every estimator and
trainer accepted into the production matrix must have a resident GPU path for
its model, optimizer, batches, and derivative products, or a machine-readable
device refusal. OpenACC is the first implementation route when it preserves
the numerical contract. For operations where OpenACC cannot express the
required launch, memory residency, synchronization, or performance contract,
and no autodiff-generated code is required, the implementation moves to a
native CUDA kernel with the same host oracle and explicit transfer accounting.
Autodiff-sensitive paths retain a FortAD/FortSym-compatible reference and are
not silently replaced by a nondifferentiable CUDA shortcut. The historical
`nvfortran` verification establishes compiler compatibility for the older
30-test snapshot. The newer tests, including Bernoulli Naive Bayes, are covered
by the GNU run only in this environment.
It does not establish that every model trains or predicts entirely on a GPU.

## Parity work packages

The source inventory is dated 2026-08-08.

| Work package | State | Implemented baseline | Package exit |
| --- | --- | --- | --- |
| Classification | Partial | `fortml_logistic_regression` and `fortml_softmax_regression` provide binary and multinomial integer-label fitting with sample-weighted reductions, `fortml_ovr_logistic_classifier` adds deterministic one-vs-rest binary logistic fits with normalized probabilities and parameter products, `fortml_ovo_logistic_classifier` adds deterministic one-vs-one pairwise-vote probabilities and input/parameter products, `fortml_multilabel_logistic_classifier` adds independent dense indicator heads, `fortml_ordinal_logistic_classifier` adds weighted cumulative-logit fitting with ordered cut points and input/packed-parameter JVP/VJPs, `fortml_gaussian_naive_bayes` adds weighted Gaussian class moments and input/parameter probability products, `fortml_bernoulli_naive_bayes` adds weighted relaxed-[0,1] Bernoulli features, positive smoothing, packed state, and input/parameter probability products, `fortml_multinomial_naive_bayes` adds weighted relaxed nonnegative counts, token-mass smoothing, packed state, and input/parameter probability products, `fortml_complement_naive_bayes` adds weighted complement distributions, optional weight normalization, packed state, and input/parameter probability products, `fortml_lda_classifier`/`fortml_qda_classifier` add weighted Cholesky Gaussian discriminants with input/parameter products, `fortml_probability_calibration` adds Platt sigmoid and weighted PAVA isotonic calibration with score/parameter products and active-set refusals, `fortml_knn_classifier` adds deterministic dense uniform or inverse-distance neighbors with explicit discrete derivative refusals, `fortml_logistic_training` adds an exact weighted logistic objective with parameter/L2 gradients and HVPs plus bounded FortOpt L-BFGS-B, `fortml_linear_svm_classifier` and `fortml_rbf_svm_classifier` add weighted hinge/squared-hinge binary margins with fixed-state probability and product contracts, `fortml_cart_classifier` adds deterministic weighted Gini/entropy trees and leaf probabilities, `fortml_random_forest_classifier` adds seeded bootstrap CART probabilities, `fortml_extra_trees_classifier` adds seeded randomized-threshold/feature ensembles, `fortml_mlp_classifier` adds deterministic multiclass logits training with Adam and sample/class weights, `fortml_mlp_ordinal_classifier` adds a scalar-score neural cumulative-logit head with ordered cut points and deterministic FortOpt L-BFGS-B training, `fortml_gp_classification` adds binary Laplace logistic/probit inference plus an exact mode-envelope kernel gradient and query-feature JVP/VJP products, `fortml_gp_multiclass_classification` adds packed one-vs-rest envelope gradients, latent margins, and query-feature JVP/VJP products, and shared metrics cover accuracy, top-k, balanced accuracy, confusion, precision/recall/F1, Brier, binary Matthews, weighted accuracy, log loss, and expected/maximum calibration error. | Binary and multiclass linear, multilabel, ordinal, tree, neural, GP, and boosted-tree classifiers share label, probability, weighting, metric, and calibration conventions. |

| Estimator contracts, pipelines, and bases | Partial | `basis_map_t`, horizontal and sequential basis pipelines, fitted standard/min-max scalers with input JVPs, `basis_linear_regression_t`, joint `basis_pipeline_training_objective_t`, weighted multi-output `ridge_regression_t` and `elastic_net_regression_t`, weighted `linear_svr_regression_t`, row-oriented sample conventions, status objects, and the parameter registry are public. | Fitted transformers and estimators compose without data leakage, expose routed parameters, and run through cross-validation. |
| Tree boosting | Partial | `decision_stump_t`, weighted depth-limited `cart_regressor_t` and `cart_classifier_t`, squared-loss `gradient_boosting_regressor_t`, `xgboost_t`, `xgboost_multiclass_t`, and separately named `lightgbm_t` provide deterministic exhaustive, bounded-histogram, and best-first leaf-wise split products. The CART lanes have weighted squared-error or Gini/entropy criteria, depth and leaf constraints, fixed feature/threshold tie ordering, piecewise input JVP/refusal for regression, and finite-only probability/prediction paths for classification. The XGBoost-style lane has exact and weighted-histogram squared/logistic/Poisson/Huber/quantile/absolute and bounded pairwise-ranking gradients, Hessians, regularized gains, recursive Newton leaves, per-feature monotonic bounds, tree-depth/node diagnostics, binary and one-vs-rest multiclass probabilities, staged predictions/margins, and gain/weight/cover feature importance. The LightGBM-style lane supports weighted regression/binary logistic objectives, shared weighted-quantile cuts, deterministic global best-leaf growth up to `num_leaves`, validation monitoring with patience/min-delta, best-round metadata, restore-best or retain-all ensembles, and typed split-boundary/CUDA refusals. | Regression and classification trees support validation-based stopping, categorical objectives, interaction constraints, deeper growth, and model persistence. |
| Training infrastructure | Partial | Model-specific gradients, exact MSE+L2 neural HVPs including the L2 mixed hyperparameter block, weighted multiclass MLP cross-entropy value/JVP/VJP/HVP products with bounded FortOpt L-BFGS-B, `mlp_training_objective_t` scalar JVP/VJP products (including the optional optimized L2 coordinate), named group-wise log-L2 objective/HVP products, differentiable Huber/quantile loss products, a joint basis-pipeline value/gradient/JVP/VJP/HVP objective, `fortopt_adam` and FortOpt SGD momentum/Nesterov integration, AdamW with decoupled decay, coupled-L2 Adam with exact fixed full-batch trajectory hypergradients, Adagrad with an explicit accumulated-square state, RMSprop with centered/uncentered statistics and optional momentum, deterministic seeded batch cursors, per-update callback and typed learning-rate schedules, norm clipping, sample-weighted gradient accumulation, validation/early stopping, resumable optimizer state, generic versioned portable trainer checkpoints, versioned portable MLP checkpoint files, fixed-trajectory full-batch SGD/Adam/AdamW/RMSprop/Adagrad and deterministic mini-batch SGD hypergradients, unfactored Adafactor rate/accumulator/validation hypergradients including smooth relative-step and parameter-scaling branches, scheduled-Adagrad rate/accumulator/validation hypergradients with FortOpt L-BFGS-B, natural-gradient seams, and seeded variational draws exist. | One trainer owns batches, optimizer state, schedules, clipping, validation, early stopping, callbacks, and resumable state for every model with a completed trainer adapter; stochastic and device-resident optimizer hypergradients remain open. |
| GP derivatives and hyperparameters | Partial | Exact GP likelihood and prediction products include parameter gradients and HVPs. Mixed value and first-derivative observations can be fitted and predicted; value-only HVPs use the analytic kernel-HVP/differentiated-solve path, while RBF/linear/constant/polynomial and their sum/product mixed-observation HVPs now use analytic covariance-block second products. Polynomial HVPs cover all four log kernel coordinates, including the degree-one limit, and have an independent likelihood finite-difference oracle. Spectral-mixture value/first-derivative parameter gradients and query JVP/VJP products are analytic; its mixed parameter HVP remains a typed refusal. Correlated multi-output GPs now expose packed coregionalization/noise/kernel parameters plus posterior-mean query and parameter JVP/VJP products through a differentiated Cholesky solve, with independent finite-difference and adjoint oracles. Binary and OVR Laplace classifiers now expose fixed-state kernel-parameter JVP/VJP products for latent and normalized observed predictions. Other mixed leaves return an explicit `FORTNUM_NOT_IMPLEMENTED` refusal rather than finite-differencing. Scalar likelihood VJPs include the packed noise block. Dense joint posterior covariance now also exposes exact parameter JVP/VJP products through the same solve, with independent finite-difference and adjoint oracles. | Exact, derivative, multi-output, sparse, and matrix-free GP families expose documented trainable parameters, scalar objectives, parameter gradients, and train-state adapters. |
| GPU and device execution | Partial | Kernel, structured, and sparse operator products have selected OpenACC or CUDA paths, including resident CG for kernel operators. The kNN classifier has a resident native-CUDA training-set plan with a direct kernel oracle; no-autodiff RMSprop, AdamW, and Adagrad state recurrences have resident CUDA C plans with independent recurrence tests; weighted MSE has both a transfer-inclusive CUDA reduction and a resident C-ABI plan; the dense-affine plan now includes resident value/JVP/VJP and single-layer mean-squared-error updates with parameter snapshots and transfer counters; and a prediction-only resident CUDA forest C ABI now retains flattened trees across repeated query batches with an independent CPU tree-walk oracle. Elastic-net, OVO, binary/OVR-multiclass GP classification, bounded pairwise XGBoost ranking, probability calibration, multilabel/Jaccard/Hamming/ROC/PR ranking, derivative-GP covariance, and typed schedule trajectories expose CPU dispatch plus explicit CUDA refusal until resident kernels are linked. `fortml_device` gives callers an explicit CPU/CUDA selector, runtime capability probe, backend identity, residency ownership metadata, transfer counters, and typed refusal when a CUDA object is not linked. The sibling benchmark harness records independent-NumPy CUDA gates for the resident micro-kernels and all currently implemented CPU metric lanes. | Supported training and prediction workflows keep model, optimizer, and batch state resident on a selected device and have CPU parity tests. |
| Serialization and distributed execution | Partial | `fortml_mlp_checkpoint` provides a versioned compiler-independent formatted-text representation with schema magic/version, exact optimizer/iterator/history state, validated temporary loading, and malformed/truncated/extra-record refusals. Other model/pipeline files and distributed execution remain open. | Versioned model and trainer files round-trip across supported compilers, and MPI training or inference agrees with a one-rank oracle. |
| Benchmark coverage | Partial | Correctness-gated model and GP applications feed release harnesses in `../fortml-bench`; current release lanes include Bernoulli/Multinomial/ComplementNB, integer one-hot encoding, weighted ridge and elastic-net derivative products, scalar and multi-output radius neighbors, binary AdaBoost, OVR/OVO/multilabel/ordinal/RBF-SVM classification, multilabel precision/recall/F1/Jaccard/Hamming and ROC/PR-AUC metrics, temperature/sigmoid/isotonic probability calibration, the shared objective trainer and portable checkpoint, weighted binary and multiclass MLP objectives with bounded L-BFGS-B, Lion hypergradients, transformed softmax log-L2 products, additive XGBoost contributions, pairwise ranking, and interaction constraints, bounded LightGBM leaf-wise boosting with validation early stopping, XGBoost warm-start continuation, MLP activation products, MLP SGD/Nesterov/Adam/AdamW, coupled-Adam and other fixed-trajectory hypergradients, typed schedules including one-cycle, fixed-trajectory and resident-state Adagrad, scheduled-Adagrad hypergradients, differentiable imputation, basis/pipeline, exact/approximate and correlated multi-output GP products including weighted OVR variational and packed Laplace multiclass prediction products, analytic GP likelihood products, derivative-GP spectral-mixture products, polynomial mixed-observation GP HVPs, exact and weighted-histogram squared/logistic/Poisson boosting, monotonic-constraint query grids, general Hamiltonian MLP products, generic grid/L-BFGS-B search, resident CUDA dense-affine value/JVP/VJP and MSE-update device-contract gates, and resident forest prediction. | Every completed parity package has a pinned external oracle, release timings, memory measurements, provenance, raw data, and a maintained report. |

The ledger now also records contiguous CPU MLP optimizer groups with checkpoint
metadata, fixed-state sparse-GP and variational-classification kernel-log-
parameter products, the classifier-chain logistic adapter, XGBoost interaction
groups, and the weighted Poisson MLP objective. These slices are production evidence for
their stated contracts. They do not close the broader GPU, stochastic,
distributed, multitask, or architecture-family gates listed above.

The benchmark matrix includes the weighted multiclass MLP objective and
bounded L-BFGS-B gate in `fortml-bench/results/mlp_classifier_objective.csv`;
its CUDA row is an explicit unavailable contract. The current classification
baseline also includes positive temperature scaling,
the current GP baseline includes bounded Bernoulli variational classification
with sorted-label OVR multiclass prediction/JVPs, and the current physics
baseline includes the four-slot residual objective seam;
the detailed closure slices and their independent benchmark reports below are
authoritative when this compact work-package table is read alongside older
package prose.

The calibrated neural classification slice adds
`mlp_calibrated_classifier_t`: binary MLP logits can be calibrated with the
shared sigmoid, positive-temperature, or weighted-PAVA isotonic maps, while
multiclass logits use one positive softmax temperature. Smooth temperature and
sigmoid network/input/calibration products are exact; isotonic active-set
products and CUDA execution return typed refusals. Its independent behavioral
oracle is `test_mlp_calibrated_classifier` and its API contract is documented
in `docs/MLP_CALIBRATED_CLASSIFIER.md`.

The companion release application `app/fortml_bench_mlp_calibrated_classifier`
emits a complete-array temperature-calibration workload for the benchmark
harness.  It records fit/predict timings and a row-oriented CSV of labels,
predictions, and probability columns; the external NumPy/scikit-learn oracle
and provenance row remain a `fortml-bench` integration task.

The sibling device-contract report now also includes the resident forest
prediction gate; higher-level Fortran forest integration remains an explicit
CUDA refusal until private CART storage is safely bound to the C ABI.

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
  contract. The matching softmax adapter now also exposes scalar JVP/VJP
  products and an optional bounded positive log-L2 coordinate; OVR objective
  adapters remain a separate follow-up contract.
- [x] Add a multiclass `mlp_classifier_t` adapter with a logits layer, stable
  softmax cross-entropy, deterministic Adam, sorted integer labels, probability
  normalization, sample/class weighting, and a packed parameter-gradient
  product. Binary, multilabel,
  ordinal, and variational GP adapters remain separate contracts.
- [x] Add `mlp_binary_classifier_t` with a one-logit sigmoid head, weighted
  BCE, deterministic Adam minibatches, early stopping, packed input/parameter
  JVP/VJP products, exact parameter loss HVPs, and typed CUDA refusal. Multilabel,
  ordinal, calibrated, and resident-GPU neural heads remain open.
- [x] Add `mlp_binary_training_objective_t` and
  `mlp_binary_optimize_lbfgsb` as the weighted BCE FortOpt seam. The adapter
  exposes analytic scalar JVP/VJP products and joint network/L2 HVPs, accepts
  sample/class weights, and enforces explicit network/L2 bounds; generic
  trainer and direct bounded L-BFGS-B calls share the same objective.
- [x] Add deterministic one-vs-rest and one-vs-one logistic wrappers with
  sorted class/pair ordering, first-max tie handling, shared sample/class
  weights, normalized probabilities, packed parameter metadata, and
  input/parameter JVP/VJP products. OVO uses an explicit pairwise-vote
  probability policy; scikit-learn's separate pairwise coupling and shared
  preprocessing ownership remain open.
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
- [x] Add multilabel-indicator classification metrics and reductions. The
  implemented dense-indicator lane covers threshold policies,
  micro/macro/samples averaging, precision/recall/F1, Jaccard, Hamming,
  ROC-AUC, and PR-AUC with typed zero-support and CUDA refusal behavior.
  Multioutput targets, sparse target support, and refusal of ambiguous target
  shapes remain a separate contract.
- [x] Add ordinal cumulative-logit classification as a separate ordered-label
  contract; nominal softmax probabilities are not reinterpreted as ordered
  outcomes. Ranking losses remain a separate open contract.
- [x] Add the deterministic finite-only `cart_classifier_t` adapter with
  sorted integer classes, weighted Gini/entropy probabilities, and explicit
  depth, leaf-size, tie, and nonfinite-input contracts.
- [x] Extend `cart_classifier_t` with explicit NaN routing. The default
  `missing_policy="error"` preserves finite behavior; `"learn"` compares both
  default branches per split with deterministic left-on-tie selection, and
  `"left"`/`"right"` force a branch. Independent fit/predict, finite-baseline,
  unsupported-policy, and shape-refusal oracles cover the public contract.
- [ ] Add shared tree and boosted-tree classifier adapters for missing-value,
  monotonic-constraint, and probability-policy variants.
- [x] Add a binary Laplace GP classifier with Bernoulli logistic and probit
  likelihoods, Newton convergence state, predictive latent moments, observed
  probabilities, and input JVPs over the supported kernel derivative contract.
- [x] Add a deterministic one-vs-rest multiclass GP wrapper with sorted integer
  classes, independent binary Laplace fits, normalized probability simplex,
  deterministic prediction ties, and refusal propagation.
- [x] Add bounded one-vs-rest multiclass variational GP likelihood plumbing:
  sorted integer classes, per-class packed inducing parameters, deterministic
  logistic/probit ELBO sums, analytic packed gradients and JVPs, latent
  margins, simplex-normalized predictive probabilities, parameter JVPs/VJPs,
  and explicit CPU/CUDA device contracts. The independent oracle is
  `test_gp_variational_multiclass_classification`; coupled categorical,
  quadrature, calibrated uncertainty, and resident GPU inference remain open.
- [x] Add a shared-head multilabel neural classifier with indicator validation,
  configurable per-label thresholds, deterministic full-batch Adam, exact
  logits/probability input and parameter JVP/VJP products, BCE parameter HVPs,
  and typed CUDA refusal. Binary and ordinal heads have separate contracts;
  calibrated neural heads are covered by the next checked slice.
- [x] Add a scalar-score ordinal neural classifier with sorted arbitrary integer
  labels, strictly increasing cumulative-logit thresholds, deterministic
  full-batch FortOpt L-BFGS-B training, positive sample weights, packed
  network/threshold parameters, exact input and parameter probability JVP/VJP
  products, and typed CPU/CUDA dispatch. The independent behavioral oracle is
  `test_mlp_ordinal_classifier`; resident GPU training and calibrated neural
  ordinal heads remain open.
- [x] Add `mlp_calibrated_classifier_t` as a deterministic composition of the
  MLP logits head and shared probability calibration primitives. Binary heads
  support sigmoid, positive temperature, and weighted isotonic calibration;
  multiclass heads support one positive softmax temperature. Packed parameters
  include the network and smooth calibration coordinates, with exact joint
  input/parameter JVP/VJP products for sigmoid/temperature and explicit
  `FORTNUM_NOT_IMPLEMENTED` refusal for isotonic PAVA active-set products.
  `test_mlp_calibrated_classifier` covers sorted labels, normalization,
  deterministic fit, finite-difference/adjoint products, and typed CUDA
  refusal. Resident-GPU neural outputs remain a separate open exit.
- [x] Add accuracy, top-k accuracy, balanced accuracy, confusion matrix, log
  loss, Brier score, binary Matthews correlation, precision, recall, and F1
  with explicit class ordering, zero-support behavior, and weighted
  accuracy/log-loss semantics. Add multilabel indicator precision/recall/F1
  with micro, macro, and sample averaging, explicit zero-division policy, and
  probability threshold (`>=` is positive) semantics. Add pairwise binary and
  one-vs-rest ROC AUC with half-credit ties, sample weights, and typed CUDA
  refusals. Add weighted binary/OVR PR AUC with average-precision step
  semantics, and multilabel Jaccard and Hamming reductions with micro, macro,
  and samples averaging. Add weighted multilabel F-beta (`beta>0`) and its
  probability-threshold wrapper with per-label/per-row reduction semantics;
  the beta=2 CPU lane is covered by an independent NumPy oracle in
  `fortml-bench`. Independent indicator and ranking oracles cover all CPU
  paths; malformed and zero-support behavior is explicit, and ranking device
  entry points return typed CUDA refusals until resident kernels exist.
- [x] Add deterministic multiclass expected and maximum calibration error
  metrics with row normalization, first-maximum tie handling, equal-width
  bins, optional sample weights, and explicit empty-bin and confidence-one
  semantics. The standalone sigmoid and isotonic calibrator estimators below
  provide fitted probability maps; multiclass coupling remains a separate
  contract.
- [x] Add probability calibration by positive temperature scaling, Platt sigmoid, and weighted PAVA isotonic
  fits. `probability_calibrator_t` validates arbitrary integer classes and
  weights, exposes smooth temperature and sigmoid score/parameter JVP/VJP products and
  linearly interpolated isotonic score products away from knots, and refuses
  active-set knot/PAVA-parameter derivatives explicitly. CPU complete-array
  behavior is benchmarked; selected CUDA contexts return a typed refusal until
  a resident calibration kernel is linked.
- [x] Add `multiclass_probability_calibrator_t` for sorted arbitrary integer
  classes and one positive softmax temperature. The weighted NLL fit, stable
  probabilities, first-class tie policy, logit/temperature JVP/VJP products,
  and explicit CUDA refusal are independently covered by
  `test_multiclass_probability_calibration` and the multiclass calibration
  benchmark lane. Multiclass Platt and isotonic maps remain typed
  `FORTNUM_NOT_IMPLEMENTED` policies rather than independent binary fits.

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

- [x] Provide polynomial, Fourier, radial, and B-spline basis maps with value,
  JVP, VJP, and analytic scalar-contraction HVP products. Callback maps retain
  value/JVP/VJP products and return a typed HVP refusal.
- [x] Provide a horizontal `basis_pipeline_t` that concatenates fixed basis
  stages, packs stage parameters, and routes value/JVP/VJP/HVP products with
  shape and stage-initialization refusal tests.
- [x] Provide `sequential_basis_pipeline_t` with explicit stage feature-count
  contracts, flattened parameters, chained JVP/VJP/HVP products, and
  independent finite-difference and adjoint tests. Column-wise HVPs are also
  covered; general DAG composition remains open.
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
- [x] Add a deterministic group-K-fold splitter. `group_kfold_splitter_t`
  keeps every integer group in one fold and greedily balances uneven group
  sizes; seeded shuffling only changes equal-size tie order. Independent
  isolation, balance, replay, and invalid-input oracles are in
  `test_group_kfold`. The iterator is deliberately index-only CPU behavior;
  derivative/device routing and cross-validation scoring remain separate
  contracts.
- [ ] Define fitted transformer, predictor, regressor, and classifier contracts.
  The contracts cover feature counts, fitted state, reset or clone behavior,
  parameter names, and status propagation.
- [x] Add the first shared estimator capability contract in
  `fortml_estimator_capabilities`. `estimator_capability_t` records role,
  fitted state, feature/target/class counts, dense/sparse/missing/sample-weight
  and partial-fit tags, transform/predict/probability methods,
  input/parameter/hyperparameter JVP/VJP/HVP products, and CPU/OpenACC/CUDA/
  resident-device support. It has typed role/input/derivative/device queries,
  validation and requirement checks. Horizontal, sequential, and
  column-selecting basis pipelines expose `%capabilities`; validation exposes
  `validate_estimator_capability` and `require_estimator_capability`, so generic
  split/search code can reject an incompatible estimator before consuming a
  fold. Reset/clone, parameter-name, metadata-routing, and full model-wide
  adoption remain open follow-up contracts.
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
- [x] Add dense `missing_indicator_t` features with `all` and `missing-only`
  fit-time column policies, explicit NaN/Infinity handling, schema metadata,
  exact zero input JVP/VJP products, and independent refusal tests. The
  sparse CSR/CSC one-hot views and device-resident indicator kernels remain
  open; the dense one-hot correctness lane is in
  `../fortml-bench/results/ONE_HOT_ENCODER.md`.
- [x] Add sparse-safe `sparse_standard_scaler_t` over `fortsparse` real CSC
  matrices. Fit counts implicit zeros, refuses centering rather than silently
  densifying, preserves the sparse structure for transform/inverse, and
  exposes exact value JVP/VJP products. `test_sparse_preprocessing` compares
  against an independent dense expansion oracle and checks the typed centering
  refusal. CSR conversion, sparse one-hot/indicator views, and resident
  device kernels remain separate follow-up work.
- [x] Add `robust_scaler_t` with median centers, configurable finite quantile
  ranges (default IQR), constant-feature unit scales, inverse transforms, and
  exact input JVPs. `test_preprocessing` checks an independent percentile
  interpolation oracle, reconstruction, constant-column behavior, and the
  unfitted/nonfinite refusal boundary. Sparse and device variants remain open.
- [ ] Add quantile and power transforms, normalization,
  ordinal encoding, target encoding with leakage guards, polynomial
  interactions, hashing, and sparse CSR/CSC feature views.
  Every fitted transform records statistics, feature names, dtypes, and the
  treatment of unseen or missing categories.
- [x] Add sequential basis pipelines. Basis maps work as fitted or fixed
  pipeline stages and propagate chained JVP/VJP products.
- [x] Add column-selecting basis feature unions. `column_basis_pipeline_t`
  validates one-based per-stage column lists, gathers selected inputs, packs
  stage parameters, and routes exact JVP/VJP/HVP products with scatter-add input
  cotangents. Cross-stage column reuse is deterministic. Duplicate indices
  within a stage are rejected.
- [x] Add `basis_linear_regression_t` as an estimator-level composition seam.
  It fits a multi-output linear model on a fitted basis pipeline, packs basis
  parameters before column-major coefficients, and chains exact input and
  parameter JVP/VJP products. Independent finite-difference and adjoint tests
  cover the complete composition.
- [x] Add `basis_pipeline_training_objective_t` for joint basis-frequency or
  knot and linear-coefficient optimization. The packed CPU objective exposes
  analytic value/gradient/JVP/VJP/HVP products, an L2 ridge block, a FortOpt
  callback, and an explicit CUDA refusal. Its independent test checks a
  Fourier basis against coordinate and directional finite differences.
- [ ] Add parallel feature-union execution, device-resident transforms, and
  general column-wise transformer graphs beyond fixed basis maps.
- [ ] Add named DAG composition with fan-out/fan-in, residual branches,
  conditional stages, and a cycle refusal. Pipeline nodes expose local
  parameter blocks so hyperparameter derivatives and optimizer routing remain
  composable through the graph.
- [ ] Add feature-name propagation, schema validation, metadata routing,
  train-only fitting, `fit_transform`, `transform`, `inverse_transform` where
  mathematically defined, and partial-fit propagation for streaming data.
- [ ] Add deterministic train/test, repeated K-fold, time-series/blocked, and
  Monte Carlo split iterators plus cross-validation scoring and routed
  parameters. The ordinary, stratified, and group-K-fold iterators now exist
  as independent index-only contracts.
- [x] Add `fortml_hyperparameter_search` orchestration for deterministic bounded
  Cartesian grids, seeded random candidates, and FortOpt L-BFGS-B. The grid
  path guards product overflow, the random path uses a caller-provided
  Fortnum RNG seed and finite-value budget, and both require complete finite
  value/gradient objectives. Each path records every evaluation; the L-BFGS-B
  path consumes the same analytic callback without hidden finite differences.
  A seeded multistart L-BFGS-B entry point retains the best converged state and
  reports start and success counts with the same refusal policy. Independent
  quadratic tests and release benchmarks cover the CPU optima and
  typed CUDA refusal until resident search state and objective kernels are
  linked.
- [ ] Add estimator cloning/scoring, successive-halving, Bayesian, and
  differentiable hyperparameter search. Differentiable search must distinguish
  validation objectives from training objectives and expose implicit
  differentiation through the fitted estimator only when its linear solves and
  stopping policy are differentiable.
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
  histogram, classifier, and second-order variants have separate contracts.
- [x] Add the exact depth-limited `xgboost_t` second-order lane for squared and
  binary-logistic objectives. It aggregates per-leaf gradients and Hessians,
  applies L1/L2/gamma/min-child-Hessian regularization and shrinkage, exposes
  margins/probabilities, split gains and leaf weights, and has a piecewise
  input-JVP/refusal contract. Recursive growth and tree depth/node diagnostics
  are implemented. Explicit NaN policies reject, learn, or force a default
  branch per split; infinities remain refused and NaN queries have zero local
  input JVP. Fixed-tree input VJP products now mirror the zero-away-from-split
  contract and refusal boundary. Per-feature monotonic constraints (`-1/0/+1`)
  propagate recursive leaf bounds through exact and histogram trees, with
  independent query-grid oracles and a typed CUDA refusal contract.
- [x] Add deterministic one-vs-rest `xgboost_multiclass_t` classification over
  the binary logistic lane. Sorted integer labels, normalized probabilities,
  argmax prediction, decision margins, quotient-rule probability JVP/VJP
  products, and split-boundary refusals have independent behavioral,
  finite-difference, and adjoint tests.
- [x] Add the classifier-shaped binary `xgboost_classifier_t` facade over the
  logistic tree lane. It retains arbitrary sorted integer classes and exposes
  weighted fit/validation, integer `predict`, `(n,2)` probabilities, raw
  decision margins, staged probabilities/margins, gain/weight/cover feature
  importance, monotone and missing-policy metadata, and fixed-tree probability
  JVP/VJP products. CPU dispatch is ordinary execution and selected CUDA
  dispatch is an explicit `FORTNUM_NOT_IMPLEMENTED` refusal. The independent
  `test_xgboost_classifier` oracle covers class ordering, simplex/staged
  invariants, weights, validation, NaN routing, derivatives, feature
  diagnostics, and device behavior; `fortml_bench_xgboost_classifier` records
  weighted CPU timing, accuracy/log loss, and the CUDA refusal.
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
- [x] Add deterministic weighted quantile binning and bounded per-feature
  histograms for the XGBoost binary/regression and one-vs-rest lanes. Store
  missing values in an explicit bin and learn a default branch at every split;
  exact-tree and histogram-tree methods remain separately selectable.
- [x] Add per-feature XGBoost monotonic constraints. Validate one constraint
  per input feature, reject values outside `{-1,0,+1}`, propagate shared leaf
  bounds through constrained split branches, expose fitted constraints, and
  test exact and weighted-histogram monotonicity on independent query grids.
  `predict_device` returns a typed CUDA refusal until a resident tree kernel is
  linked; no CPU fallback is counted as GPU execution.
- [x] Add the production Poisson log-link objective to `xgboost_t`.
  `fit_poisson` validates nonnegative finite targets, initializes a guarded log
  weighted mean, applies stable `exp(margin)-target` gradients and positive
  Hessians, and returns expected counts from vector, matrix, and staged
  prediction APIs. Exact and weighted-histogram fits, zero-count behavior,
  negative-target refusal, input products, and the explicit CUDA refusal have
  independent behavioral tests and a release benchmark lane. Resident GPU
  tree growth remains open; CPU timing is never relabeled as CUDA evidence.
- [x] Add Huber and quantile/pinball objectives to `xgboost_t`. `fit_huber`
  and `fit_quantile` expose positive parameter validation, weighted base
  margins, exact piecewise gradients, explicit Hessian floors, objective
  metadata, and independent one-tree CPU oracles for exact and histogram
  dispatch. CUDA prediction/training remains a typed refusal until a resident
  robust-tree kernel is linked.
- [x] Add XGBoost `reg:squaredlogerror` (RMSLE) to `xgboost_t` through
  `fit_squared_log` and the generic objective aliases `squaredlog`,
  `squaredlogerror`, and `rmsle`. The implementation uses the weighted
  geometric base margin in `log(1+y)` coordinates, stable analytic gradient
  and positive-clipped Hessian, guarded `expm1` prediction/staged links,
  exact/histogram parity, nonnegative-target validation, and an independent
  one-tree oracle. The inverse link has the mathematical lower bound `-1`;
  CUDA prediction remains an explicit typed refusal until resident trees are
  linked.
- [x] Add gradient-boosted absolute-deviation regression. `fit_absolute` uses a
  weighted-median identity-link base margin, an exact sign subgradient with a
  positive Hessian floor, exact/histogram split parity, fixed-tree products,
  and explicit CUDA refusal; the independent oracle is `test_xgboost_absolute`.
- [ ] Add gradient-boosted regression for Tweedie losses.
- [x] Add binary and deterministic one-vs-rest multiclass gradient-boosted
  classification with stable logistic objectives, staged margins, normalized
  probabilities, feature diagnostics, and typed CUDA refusals.
- [x] Add weighted validation objectives, patience, minimum improvement,
  best-round accounting, ensemble trimming, and restore-best behavior to all
  current XGBoost objectives. Independent early-stopping and malformed-validation
  tests cover the lifecycle.
- [x] Add deterministic without-replacement row and feature subsampling with
  positive `int64` seed, stable ascending selected-index order, exact full-data
  defaults, and invalid-fraction refusals. Independent seed and structure tests
  cover the contract.
- [x] Add transactional fitted-ensemble slicing with
  `xgboost_t%slice(n_trees,destination,status)`. The prefix copy preserves the
  objective/link, base margin, missing-value routing, monotonic constraints,
  regularization, and tree diagnostics; staged-prediction and malformed-prefix
  tests prove that the result is a valid standalone model without refitting.
- [x] Add transactional warm-start continuation to `xgboost_t`. The fitted
  prefix is retained byte-for-byte while a larger requested suffix is grown;
  objective, tree, sampling, regularization, and monotone-control changes are
  refused transactionally. Independent staged-prefix and target-domain tests
  cover the API. Deterministic distributed feature reduction remains open.
  Learning-rate shrinkage and L1/L2 leaf penalties are implemented in the
  current core. Versioned serialized tree state is implemented by
  `xgboost_t%save_text/%load_text` and covered by an independent round-trip
  oracle.
- [x] Add a deterministic seeded random-forest classifier built from weighted
  Gini/entropy CART trees. It aligns bootstrap-tree probability columns,
  exposes class/tree/depth metadata, and has independent cluster, simplex,
  determinism, invalid-option, and CUDA-refusal tests plus a NumPy benchmark.
  Extra-trees, bagging, random-subspace, isolation forests, out-of-bag and
  permutation importance remain open.
- [x] Add XGBoost-style second-order boosting: per-leaf gradient/Hessian
  aggregation, regularized split gain, weighted quantile cuts, exact and
  histogram algorithms, sparse-aware default directions, monotonic constraints,
  recursive depth-limited growth, and deterministic feature ordering.
- [ ] Add interaction constraints, column blocks, categorical partitions, and
  distributed histogram reduction with independent oracles.
- [x] Add a bounded, separately named LightGBM-style numeric histogram and
  leaf-wise policy. `lightgbm_t` reuses the XGBoost weighted-quantile cut
  primitive, supports weighted squared regression and binary logistic loss,
  and splits the globally best current leaf until `num_leaves` (or
  `max_depth`) is reached. `test_lightgbm` is an independent hand oracle for
  weighted leaves, logistic probabilities, deterministic tree shape, fixed-tree
  JVP/VJP products, split-boundary refusals, and CPU/CUDA capability. The
  release lane is `lightgbm_leafwise.csv`; GOSS, EFB, categorical statistics,
  external-data iterators, and resident GPU histograms remain separate gaps.
- [ ] Add DART/dropout boosting, class- and query-weighted ranking objectives
  (pairwise logistic and Lambda-style NDCG), survival/count objectives,
  quantile and Tweedie losses, custom objective callbacks, and custom
  evaluation metrics. Unsupported objectives must return a structured refusal.
- [ ] Add monotone prediction checks, partial dependence, SHAP-like additive
  contribution products, and model-size/tree export diagnostics. Warm-start
  continuation and staged predictions are covered above.
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

- [x] Add weighted ridge, lasso, and elastic-net regression with deterministic
  coordinate/SVD solvers, nonnegative sample weights, multi-output state, and
  fixed-fit coefficient/input JVP/VJP products. Fit-time nonsmooth solver
  decisions remain explicit derivative boundaries.
- [ ] Add weighted OLS, positive-constrained, Bayesian ridge, ARD regression,
  Huber, Theil-Sen, RANSAC, quantile, and Tweedie regression. Solvers expose
  convergence status, regularization scaling, warm starts, and coefficient
  covariance where it is defined.
- [x] Add weighted Poisson and Gamma GLM regression with a shared stable log-link
  estimator, strict response-domain checks, finite coefficient bounds, sample
  weights, dispersion, analytic objective gradients, prediction/input/parameter
  JVP/VJP products, alpha/dispersion hypergradients, and bounded FortOpt
  L-BFGS-B fitting. CPU dispatch is
  complete and a selected CUDA context returns `FORTNUM_NOT_IMPLEMENTED` until
  a resident positive-link kernel is available; release evidence is recorded by
  `fortml_bench_glm_regression` and the sibling NumPy lane.
- [ ] Add linear SGD/regression and classification with deterministic minibatch
  schedules, averaging, penalties, and `partial_fit` semantics. Keep its
  stochastic objective separate from the exact logistic/softmax objective.
- [x] Add dense closed-radius nearest-neighbor scalar regression with uniform
  or inverse-distance averaging, nonnegative sample weights, an explicit
  empty-neighborhood value, deterministic distance boundaries, independent
  hand-oracle tests, and typed CPU/CUDA/derivative contracts through
  `radius_neighbors_regressor_t`. The selection operation is intentionally
  nondifferentiable and returns `FORTNUM_NOT_IMPLEMENTED` for input JVP/VJP.
- [x] Add multi-output nearest-neighbor regression with dense closed-radius
  uniform or inverse-distance reductions, vector outliers, and typed
  derivative/device contracts. The dedicated release oracle is recorded in
  `fortml-bench/results/RADIUS_NEIGHBORS_MULTIOUTPUT.md`; kernel-density,
  exact/brute, KD-tree, ball-tree, metric-callback, and missing-value-policy
  extensions remain open.
- [x] Add weighted dense linear SVM/SVR estimators with arbitrary labels or
  real targets, FortOpt L-BFGS-B fitting, packed affine products, and typed
  nonsmooth/CUDA boundaries. Kernel SVM/SVR and kernel approximation (Nyström
  and random Fourier features) remain open, with explicit solver/feature-memory
  limits.
- [x] Add weighted Multinomial, Bernoulli, and Complement naive Bayes with
  stable log-probability products and declared input/parameter derivative
  boundaries.
- [x] Add Categorical naive Bayes with sorted per-feature category offsets,
  weighted class priors and likelihood smoothing, explicit unknown-category
  error/ignore policies, and a discrete-input JVP refusal test.
- [x] Add weighted LDA/QDA and discriminant shrinkage with sorted arbitrary
  integer labels, stable Gaussian log probabilities, Cholesky factors,
  packed mean/covariance/prior products, input and parameter JVP/VJP tests,
  an independent NumPy benchmark, and an explicit CUDA refusal until resident
  discriminant kernels are linked.
- [x] Add deterministic dense seeded k-means with fit/predict/transform,
  inertia, fixed-center input JVP/VJP products, an independent oracle, and
  explicit empty-cluster, nonfinite-input, and CUDA refusal contracts.
- [ ] Add minibatch k-means, Gaussian mixtures, Bayesian mixtures,
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
- [x] Add a sequential binary classifier-chain adapter with arbitrary sorted
  integer labels, observed-label training features, smooth probability-chain
  prediction, packed-head input/parameter JVP/VJPs, thresholds, weights, and a
  typed CUDA refusal. The bounded release evidence is `test_classifier_chain`
  and `fortml-bench/results/classifier_chain.csv`.
- [ ] Add general multioutput, multiclass, multilabel, regressor chains,
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
  multilabel, PR/ROC-AUC, Jaccard, Hamming, and sample-weight behavior for
  every metric. Metrics return a
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
  gradient. The weighted `mlp_classifier_training_objective_t` now adds
  optional L2, analytic parameter/L2 JVP/VJP/HVP products, and bounded FortOpt
  L-BFGS-B. Its independent gate is `test_mlp_classifier_objective`, with
  release evidence in `fortml-bench/results/MLP_CLASSIFIER_OBJECTIVE.md`.
  Other likelihoods and shared parameter-tree routing remain open.
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
- [x] Add the exact fixed full-batch SGD momentum/Nesterov trajectory
  hypergradient objective over `[log(learning_rate), log(l2), momentum]`.
  Classical velocity and Nesterov look-ahead state sensitivities use the same
  analytic MLP HVP recurrence as `fortopt_sgd`; value/gradient, JVP, scalar VJP,
  and bounded FortOpt L-BFGS-B products are covered by independent central-
  difference, adjoint, Nesterov, and typed CPU/CUDA/optimizer-refusal oracles.
  Mini-batch, schedules, clipping, stochastic, and CUDA-resident state
  derivatives remain explicit follow-up contracts.
- [x] Add the exact fixed full-batch AdamW trajectory hypergradient objective
  over `[log(learning_rate), log(l2), log(weight_decay)]`, including analytic
  moment/decoupled-decay sensitivities, JVP/VJP products, independent central
  differences, and a FortOpt L-BFGS-B adapter.
- [x] Extend the AdamW trajectory contract with exact beta1/beta2 sensitivities
  over unconstrained logits, bias-correction derivatives, independent central
  differences, and the same FortOpt L-BFGS-B adapter. Mini-batch, schedule, and
  CUDA AdamW hypergradients remain explicit follow-up contracts.
- [x] Add the exact fixed full-batch coupled-L2 Adam trajectory hypergradient
  objective over `[log(learning_rate), log(l2), logit(beta1), logit(beta2)]`.
  The regularized loss gradient feeds both moment states without AdamW's
  decoupled shrinkage; parameter, moment, bias-correction, value-gradient,
  JVP, scalar-VJP, and FortOpt L-BFGS-B products are covered by independent
  central-difference/adjoint tests. Mini-batch, schedules, stochastic state,
  and resident CUDA Adam remain explicit refusals until their state derivatives
  and reproducibility contracts land.
- [x] Add the exact fixed full-batch RMSprop trajectory hypergradient objective
  over `[log(learning_rate), log(l2), decay, log(epsilon), momentum]`, including
  centered and uncentered square/mean/momentum state sensitivities, JVP/VJP
  products, independent finite-difference tests, and a FortOpt L-BFGS-B
  adapter. The centered branch is fixed discrete state; mini-batch, schedule,
  clipping, and CUDA-resident RMSprop hypergradients remain open.
- [x] Add the exact fixed full-batch Adagrad trajectory hypergradient objective
  over `[log(learning_rate), log(l2), log(epsilon)]`, including accumulated-
  square and epsilon-stabilized diagonal-step sensitivities, JVP/VJP products,
  independent finite-difference and adjoint tests, and a FortOpt L-BFGS-B
  adapter. Mini-batch, schedules, clipping, and CUDA-resident Adagrad
  hypergradients remain explicit refusals until their state derivatives land.
- [x] Add the exact fixed full-batch unfactored Adafactor trajectory
  hypergradient objective over `[log(learning_rate), log(l2), decay,
  log(epsilon), log(clip_threshold)]`. The analytic product differentiates the
  second-moment, update-RMS clipping, and stabilized denominator states and is
  consumable by FortOpt L-BFGS-B. Central-difference, directional JVP, scalar
  VJP, active-set, discrete-branch, and CUDA-refusal oracles are in
  `test_mlp_adafactor_hypergradient`. Relative-step and parameter-scaling
  smooth branches now carry the same products and central-difference oracles;
  matrix-factored state and resident-CUDA branches remain explicit refusals.
- [x] Add `mlp_training_objective_t`, a FortOpt context adapter exposing the
  packed MLP objective, analytic gradient, scalar JVP/VJP, and exact HVP. Its
  optional final L2 component makes bounded L-BFGS-B regularization search use
  the same products as training. Central-difference and scalar adjoint tests
  cover value gradients, JVP/VJP, HVPs, and the FortOpt callback path.
- [x] Add `mlp_optimize_lbfgsb`, a deterministic full-batch FortOpt adapter
  that optimizes the packed network parameters under explicit bounds and can
  append a bounded L2 hyperparameter. It consumes the analytic MLP value and
  gradient products, reports optimizer diagnostics, and refuses malformed or
  non-finite bounds. An independent closed-form ridge fixture checks the
  fitted weight and a refusal fixture checks inverted bounds.
- [x] Add `mlp_grouped_training_objective_t` and
  `mlp_parameter_group_t` for named, non-overlapping parameter slices with
  independent positive log-L2 hyperparameters. The packed network/log-L2
  objective exposes exact value/JVP/VJP/HVP products, including the mixed
  network/log-coefficient blocks, and installs directly into FortOpt
  L-BFGS-B. `test_mlp_grouped_training` checks the products against a
  hand-derived linear ridge oracle, the scalar FortOpt callback, and the
  explicit CUDA refusal. A resident CUDA graph for grouped MLP training is
  still open; the API never hides a host fallback.
- [x] Add `mlp_grouped_optimize_lbfgsb` with shared network bounds and
  independently bounded log-L2 coordinates. The convenience adapter passes
  the exact grouped gradient callback to FortOpt, reports optimizer
  diagnostics and final group coefficients, allows equal bounds to freeze a
  hyperparameter, and returns the same typed CUDA derivative refusal. The
  independent grouped test checks a closed-form linear ridge optimum; the
  grouped benchmark reports objective, gradient norm, iterations, and each
  optimized log-L2 coordinate.
- [x] Add `mlp_batch_iterator_t` with explicit seeded Fisher--Yates epochs,
  reproducible copied cursors, and unpadded uneven final batches. `mlp_train`
  consumes this cursor rather than maintaining a second hidden batching
  implementation.
- [x] Add per-update learning-rate schedule callbacks and global gradient-norm
  clipping to the MLP Adam trainer. The resulting state records the effective
  per-epoch rates and clipping count, with independent schedule, clipping, and
  iterator oracles.
- [x] Add a stateless typed schedule contract with constant, linear warm-up,
  cosine, warm-up-plus-cosine, exponential-decay, and one-cycle
  (linear-warmup/cosine-tail) families. Each schedule validates update/rate
  domains and returns analytic products with respect to the base rate,
  minimum-rate fraction, and decay factor; one-cycle additionally returns
  exact peak-rate and final-rate fraction products through
  `rate_with_full_derivatives`. Independent formula and central-difference
  oracles cover transitions, terminal clamping, and malformed schedules. The
  separate scheduled trajectory adapter now exposes
  exact fixed full-batch JVP/VJP products over base rate, L2, minimum fraction,
  and decay logits and a FortOpt L-BFGS-B integration; its CPU/CUDA boundary is
  independently benchmarked and CUDA is an explicit refusal until a resident
  MLP trajectory kernel is linked.
- [x] Route the stateless typed schedules through `mlp_train` using
  `options%use_typed_schedule` and `options%typed_schedule`. The trainer
  validates and evaluates the schedule without a process-local callback,
  captures its fields in version-5 trainer checkpoints and schema-3 text
  files, and refuses callback conflicts or changed schedules on resume.
  `test_mlp_typed_schedule` supplies exact learning-rate, replay,
  malformed-input, and serialization oracles; CUDA schedule lowering and
  device-resident optimizer state remain open.
- [x] Add `mlp_schedule_hypergradient_objective_t` for exact differentiable
  schedule/optimizer trajectories. The packed log/logit layout has independent
  finite-difference and scalar-adjoint tests, a complete-array release app and
  NumPy benchmark, and explicit second-order hyper-HVP scope (third network
  derivatives are not approximated). L-BFGS-B consumes the same reverse
  products; unsupported device trajectories return typed refusal.
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
- [x] Add differentiable mean Huber and quantile losses to the shared loss
  facade. Huber has a continuous first derivative at its transition; quantile
  JVP/VJP products refuse exact zero residuals. Independent formula,
  finite-difference, adjoint, and kink-refusal tests cover both products.
- [x] Complete the smooth neural-loss derivative slice: stable BCE/logistic and
  softmax cross-entropy Hessian-vector products, weighted MSE value/JVP/VJP/HVP
  products with explicit mean/sum reductions, and a Huber HVP that refuses its
  exact transition kink. Route the existing weighted MLP objective and its
  HVP through the shared weighted-MSE kernels; independent behavioral tests
  cover finite-difference curvature, adjoint identities, reductions, and the
  MLP integration. Resident CUDA loss kernels remain open; unsupported device
  requests must return a typed refusal rather than copying through the host.
- [x] Add production MAE and focal BCE-with-logits products to the shared loss
  facade. MAE has weighted mean/sum value products and explicitly refuses exact
  zero-residual JVP/VJP calls; focal BCE accepts stable logits, relaxed binary
  targets, `alpha`/`gamma`, row weights, and both reductions with analytic
  value/JVP/VJP products. Independent finite-difference, adjoint, reduction,
  extreme-logit, and parameter-refusal tests cover the slice. Resident CUDA
  loss kernels remain open and never fall back silently to host execution.
- [x] Add Gaussian and Poisson/count negative-log-likelihood products to the
  shared neural-loss facade. Gaussian NLL uses mean/log-variance coordinates
  and includes the normalizing constant; Poisson NLL uses log-rate coordinates
  and `log_gamma(count+1)`. Both expose weighted mean/sum value/JVP/VJP/HVP
  products, aliases, finite-scale and nonnegative-count validation, and
  independent finite-difference/adjoint/curvature tests. Resident CUDA NLL
  kernels remain open; CUDA requests are typed refusals with no host fallback.
- [x] Define a sequential nested-MLP parameter-tree seam with stable named
  stage paths, contiguous offsets, exact chain-rule products, and an analytic
  FortOpt L-BFGS-B objective. Independent JVP finite-difference, VJP adjoint,
  HVP differentiated-VJP, optimizer, and CUDA-refusal tests cover the current
  scope. Buffers, frozen/tied blocks, masks, stateful layers, and alias-aware
  flattening remain open extensions of the general tree.
- [x] Add weighted multilabel BCE-with-logits and ordered cumulative-logit
  ordinal negative-log-likelihood products to the shared neural-loss facade.
  Both expose explicit mean/sum reductions, finite row weights, value/JVP/VJP
  products, and exact logits HVPs with independent formula, finite-difference,
  and adjoint tests. Multilabel targets are relaxed indicators; ordinal rows
  require strictly ordered cumulative logits and one-based class labels.
  Resident CUDA loss kernels remain open and unsupported device requests are
  typed refusals. Contrastive/triplet losses, KL terms, and sequence masking
  still need the same logits/probability and empty-batch contracts.
- [x] Add a deterministic batch iterator with seeded shuffling and final-batch
  behavior. Separate training and validation streams remain open.
- [x] Add the model-agnostic `trainer_t` objective seam with explicit
  full-batch SGD, Adam, AdamW, Adagrad, RMSprop, unfactored Adafactor, and
  bounded L-BFGS-B state,
  clipping, projection bounds, EMA, convergence histories, callbacks, and
  cloneable in-memory checkpoints. The independent quadratic oracle is
  `test_trainer`; data-loader, validation, distributed, mixed-precision, and
  resident-device adapters remain separate contracts.
- [ ] Add a trainer that owns optimizer state, learning-rate schedules, gradient
  clipping, accumulation, validation intervals, early stopping, and callbacks.
  The current MLP trainer now covers deterministic accumulation, schedules,
  clipping, patience, callbacks, a finite held-out validation stream with
  interval-based monitoring and best-state restoration, and an in-memory
  resumable `mlp_training_checkpoint_t` containing Adam/AdamW/Adagrad/RMSprop/
  unfactored Adafactor/SGD,
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
- [x] Add deterministic parameter exponential moving averages to the MLP
  trainer. `ema_decay` validates the closed-open decay domain, starts from
  the initial packed parameters, updates after every optimizer step, and
  persists the averaged vector through in-memory and versioned file
  checkpoints. Independent recurrence and interrupted/serialized-resume
  tests cover the state; EMA is an explicit export surface and never hides a
  model-parameter replacement.
- [x] Add validated contiguous MLP optimizer groups with named positive
  learning-rate multipliers and deterministic per-block update scaling across
  the CPU optimizers. Group ranges and metadata round-trip through in-memory
  and formatted checkpoints; mixed precision, distributed groups, and resident
  device execution remain open.
- [ ] Add activation checkpointing, truncated BPTT, gradient
  centralization/noise, value clipping, and anomaly detection with
  parameter-path diagnostics.
- [x] Add a typed MLP event contract for train begin, optimizer update,
  validation, epoch end, checkpoint, and train end. Events carry counters,
  losses, gradient norm, effective learning rate, and a stop/status channel;
  callback order and failure propagation are deterministic and independently
  tested. Distributed callback coordination remains open.
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
- [x] Add compiler-independent versioned MLP checkpoint save/load. The text
  schema round-trips all optimizer variants, iterator/RNG cursor, histories,
  validation state, and resume metadata, and validates into a temporary
  destination before replacement. Unknown, truncated, extra, malformed, and
  invalid records have independent refusal tests.
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
  RBF, Matérn, periodic, rational-quadratic, cosine, linear, constant, polynomial,
  and composed kernels and is checked against
  an independently assembled dense oracle. Value-only observation lists use
  the analytic kernel parameter-HVP and differentiated Cholesky solve; mixed
  observation lists retain a deterministic directional finite-difference
  fallback until generated input-parameter second products cover every
  supported kernel. `test_derivative_gp_products` checks both branches.
- [x] Add trainable constant and linear mean templates to exact GP regression.
  Per-output mean coefficients follow kernel and log-noise parameters, and
  prediction/LML JVP, VJP, and HVP products include the mean block. Automatic
  relevance determination length scales, priors, and inducing-location blocks
  remain open.
- [x] Add an ARD RBF kernel with per-feature log length scales, scalar and dense
  matrix products, input gradients/mixed Hessians, parameter VJP/HVP products,
  composed-kernel compatibility, and exact-GP likelihood integration. The
  isotropic RBF default remains unchanged and CUDA returns a typed refusal.
- [x] Add bounded exact-GP hyperparameter optimization with deterministic
  seeded restarts, explicit first-start retention, convergence accounting, and
  restoration of the best finite converged state. The API reports start and
  success counts, best-start index, objective evaluations, and refuses a
  selected CUDA device until exact factorization and optimizer state are
  resident. Priors, jitter escalation, and derivative-GP multistart remain
  separate follow-up contracts.
- [ ] Define a derivative capability table for every estimator, transform,
  objective, and backend. It must list supported value, input JVP/VJP/HVP,
  parameter JVP/VJP/HVP, stochastic-path derivative, and refusal conditions.
  an absent product is never inferred to be zero.
- [x] Add the transform-aware hyperparameter registry foundation. Named blocks
  now expose physical/unconstrained identity, log, and bounded-logit
  coordinates, finite lower/upper bounds, trainability filtering, provenance
  and device metadata, HVP-availability metadata, and projected optimizer
  vectors with deterministic ranges. Owned-value and live callback adapters
  share the existing parameter-block contract; model-specific priors,
  inducing locations, validation-weight derivatives, and complete FortOpt
  objective adapters remain follow-up work.
- [x] Add exact transform Jacobian and curvature pullbacks. Registry blocks
  expose `physical_derivatives`; trainable registries expose physical-gradient
  and physical-HVP pullbacks into identity, log, and bounded-logit optimizer
  coordinates. An independent analytic transform oracle covers the chain rule;
  model objective adapters and implicit solve derivatives remain open.
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
- [x] Add analytic likelihood values, parameter gradients, JVPs, scalar VJPs,
  and noise-parameter products for derivative-observation GPs. Mixed
  value/first-derivative HVPs are analytic for RBF, linear, constant, and
  sum/product compositions made solely from those leaves; the dense
  differentiated solve includes the log-noise block. Matérn, periodic,
  rational-quadratic, cosine, polynomial, user-formula, and other leaves return
  `FORTNUM_NOT_IMPLEMENTED` for mixed HVPs until their second
  input/parameter products are generated and independently checked. Polynomial
  mixed HVPs are now closed-form for all four log parameters, including the
  degree-one limit, and are checked by `test_derivative_gp_polynomial` and the
  derivative-GP benchmark. There is no hidden finite-difference fallback.
  `test_derivative_gp_products` checks the scalar VJP, independent likelihood
  oracle, analytic RBF mixed HVP, and the typed unsupported-leaf refusal.
- [x] Add parameter JVP and VJP products for derivative-observation GP
  prediction means and variances, with independent dense finite-difference and
  reverse-product oracles over value/first-derivative query components.
- [x] Add exact query-input JVP and VJP products for derivative-observation GP
  means and variances. RBF, Matérn 3/2, Matérn 5/2, periodic,
  rational-quadratic, polynomial, linear, constant, and sum/product kernels propagate the
  third-input derivative analytically through value, gradient, and mixed
  Hessian covariance blocks. Independent directional finite-difference and
  adjoint tests cover the smooth leaves. Matérn 1/2 coincident derivatives,
  user formulas, and other unsupported leaves return typed refusals; no hidden
  finite-difference fallback is used. See [docs/GP_DERIVATIVES.md](docs/GP_DERIVATIVES.md)
  for the complete public capability matrix.
- [x] Add an explicit device capability and prediction dispatch contract for
  mixed value/first-derivative GPs. CPU dispatch is reference-equivalent;
  CUDA refuses with `FORTNUM_NOT_IMPLEMENTED` until a resident covariance,
  factorization, and derivative-query graph is linked. No hidden host fallback
  is permitted, and the refusal is covered by an independent test.
- [ ] Lower derivative-GP covariance assembly, solves, parameter products, and
  query JVP/VJP products to resident CUDA kernels; only then promote the CUDA
  capability flag and add timed GPU benchmark rows.
- [x] Add dense joint latent posterior covariance for arbitrary value and
  first-derivative query sets, including cross-covariances between requested
  components. The CPU implementation reuses the exact derivative covariance
  blocks and differentiated solve path, symmetrizes roundoff, clamps only tiny
  negative diagonals, and exposes an explicit CUDA refusal until the resident
  graph is linked. Independent dense covariance and device-dispatch oracles
  cover the contract; observation noise remains excluded from latent posterior
  covariance.
- [x] Add exact parameter JVP and VJP products for dense derivative-GP joint
  posterior covariance. `joint_covariance_jvp` differentiates the prior,
  train/query cross-covariance, and Cholesky solve in packed log-kernel/log-
  noise coordinates; `joint_covariance_vjp` propagates a symmetric cotangent
  through the same blocks and is checked by an independent finite-difference
  and adjoint oracle in `test_derivative_gp_products`. The products are CPU
  reference paths with no finite-difference fallback; CUDA remains an explicit
  refusal until the resident covariance graph is linked.
- [x] Extend derivative-observation covariance, hyperparameter-gradient, and
  query-input JVP/VJP products to the GPyTorch-compatible spectral-mixture
  kernel. The independent dense oracle in
  `test_derivative_gp_spectral_mixture` covers value/first-derivative blocks,
  posterior covariance, packed parameter gradients, and query adjoints;
  mixed parameter HVPs remain a typed `FORTNUM_NOT_IMPLEMENTED` refusal until
  fourth input/parameter products are generated.
- [ ] Extend derivative observations to second derivatives only for kernels with
  the required smoothness, with explicit refusal at singular coincident cases.
- [ ] Add scalar objectives and parameter gradients for multi-output, sparse
  variational, local, SKI, Lanczos, and matrix-free GP paths. Inducing-point and
  local-gate training remain separate parameter blocks.
- [x] Add binary Laplace GP classification for logistic and probit likelihoods,
  with damped Newton state, latent/probability prediction, input JVP/VJP
  products over the kernel derivative contract, and an exact envelope gradient
  for the converged mode log posterior (without evidence correction).
- [x] Expose the shared signed-margin binary GP likelihood as analytic
  `value`/`JVP`/`VJP` products for logistic and probit links, with a stable
  negative-tail log-CDF and independent scalar/finite-difference/adjoint
  tests. This is a backend-independent building block for future variational
  and minibatch objectives; it does not claim resident GPU GP training.
- [x] Publish fitted-kernel parameter metadata and exact mode-envelope
  hyperparameter gradients for binary and one-vs-rest GP classifiers. The
  multiclass wrapper packs independent binary gradients; a shared categorical
  Laplace evidence gradient remains open.
- [x] Add one-vs-rest multiclass GP classification as a deterministic wrapper
  over the binary Laplace contract, with sorted labels, latent margins,
  normalized positive probabilities, and chained query-feature JVP/VJP
  products. Add explicit `predict_proba_device`/`predict_device` dispatch and
  `device_supported` capability metadata: CPU dispatch is exact and selected
  CUDA contexts return `FORTNUM_NOT_IMPLEMENTED` until independent per-class
  covariance/Laplace state is resident. Variational categorical likelihoods
  remain open.
- [x] Add bounded FortOpt L-BFGS-B adapters for binary and shared-kernel
  one-vs-rest GP classification. Each trial refits the Laplace mode and uses
  the analytic envelope gradient; invalid bounds, failed mode solves, and
  nonfinite objectives are refused. Full evidence, likelihood-parameter,
  independent per-class, and implicit/HVP training remain open.
- [x] Add bounded Bernoulli variational GP classification for logistic and probit
  likelihoods. `gp_variational_classification_t` owns inducing `q(u)`, a seeded
  deterministic ELBO table, analytic KL, packed gradients, exact prediction
  JVPs/VJPs, variable-batch likelihood scaling, and an explicit CUDA refusal.
  The independent finite-difference/JVP/VJP/device oracle is
  `test_gp_variational_classification`. Multiclass coupling, kernel/inducing
  hyperparameter products, natural gradients, and resident GPU inference remain
  open.
- [x] Add fixed-state sparse variational-GP ELBO JVP/VJP products for all
  currently supported kernel log parameters. The products differentiate the
  inducing solve, cross-covariance, predictive diagonal, and KL terms and are
  independently checked by `test_sparse_gp`; general inducing-state,
  likelihood, natural-gradient, and resident-GPU products remain open.

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
- [x] Add a resident native-CUDA kNN training-set plan with deterministic
  stable ties and one-based class-index parity, plus a resident no-autodiff
  RMSprop state plan. Both have independent host/NumPy recurrence oracles,
  explicit create/step/download/destroy lifecycles, and typed unavailable
  behavior when CUDA is not linked.
- [x] Add the matching resident no-autodiff AdamW state plan with explicit
  device-resident gradients, bias-corrected moments, decoupled weight decay,
  lifecycle operations, and an independent multi-step recurrence oracle.
- [x] Add the matching resident no-autodiff Adagrad state plan with explicit
  device-resident parameters, accumulated-square state, and gradients,
  lifecycle operations, an independent eight-step recurrence oracle, and a
  typed unavailable path when CUDA is not linked. The plan is state-only and
  does not claim resident MLP autodiff or end-to-end training.
- [x] Add the resident no-autodiff weighted-MSE C-ABI plan. It uploads target,
  prediction, and optional weights once, executes five repeated reductions,
  preserves the scalar across invalid-size refusal, and is checked against an
  independent CPU oracle. This primitive does not claim resident estimator
  training or autodiff support.
- [x] Keep backend selection explicit: when OpenACC cannot preserve residency
  or deterministic semantics and no autodiff product is required, use a native
  CUDA kernel with a CPU oracle; autodiff-bearing trajectories remain on the
  CPU until generated FortAD/FortSym device products and transfer contracts
  are available.
- [x] Add explicit device capability/refusal methods for elastic-net
  prediction, OVO probabilities/labels, Laplace-GP latent/probability
  prediction, the shared GP likelihood, and typed MLP schedules. Their CPU
  dispatches retain the reference behavior; selected CUDA contexts return
  `FORTNUM_NOT_IMPLEMENTED` until resident kernels and transfer accounting
  exist. Independent synthetic-device tests cover the no-hidden-host-fallback
  boundary, and the sibling benchmark records untimed CUDA refusal rows.
- [x] Record the resident micro-kernel device contracts in the sibling benchmark
  harness with
  machine-readable pass/skipped/failed rows, hardware and revision provenance,
  and no claim of end-to-end MLP/GP/XGBoost GPU residency.
- [x] Add explicit CPU/CUDA capability contracts for the new cosine/polynomial
  kernels, weighted LDA/QDA, robust XGBoost objectives, and Gaussian/Poisson
  neural NLL products. Their independent tests and benchmark rows return a
  typed CUDA refusal with no host fallback; this is a correctness boundary,
  not a claim of GPU support.
- [x] Add explicit CPU/CUDA capability contracts for the seeded random-forest
  classifier, MLP-classifier prediction JVP/VJP products, and analytic basis/
  pipeline HVPs. CPU behavior is independently oracle-tested; selected CUDA
  contexts return typed `FORTNUM_NOT_IMPLEMENTED` until resident ensemble,
  neural, and derivative kernels exist.
- [x] Define the versioned `random_forest_cuda_plan_t` ABI boundary (version 1)
  with fitted shape/device metadata, lifecycle methods, sentinel-preserving
  typed refusals, and a benchmark plan-creation row. It does not allocate or
  copy host trees; a resident no-autodiff CUDA tree kernel remains open.
- [x] Add the first resident no-autodiff CUDA forest prediction C ABI. The
  flattened model remains on the selected device across repeated query batches;
  strict-threshold routing, sorted-class ties, probabilities, and malformed
  model refusal are checked against an independent CPU tree-walk oracle. The
  higher-level Fortran random-forest adapter remains an explicit refusal until
  its private CART storage is safely bound to this ABI.
- [x] Add a Fortran-facing `cuda_forest_plan_t` wrapper for explicit flattened
  models. The ordinary build links a typed unavailable stub and preserves
  caller buffers on refusal; native CUDA applications can bind the C plan
  without exposing private CART storage or adding an autodiff path.
- [x] Add the first resident no-autodiff dense-neural primitive. The native
  `cuda_dense_plan_t`/C ABI keeps one affine layer's weights and bias on the
  selected CUDA device, supports every current MLP activation, and copies only
  query batches and outputs. A typed ordinary-build refusal, independent CPU
  activation oracle, repeated resident batches, finite-input validation, and
  `fortml_device` capability probe prevent this inference kernel from being
  mistaken for complete MLP training or a device-side FortAD/FortSym graph.
- [ ] Keep batches, parameters, gradients, optimizer accumulators, and workspaces
  resident through complete MLP and variational training steps.
- [ ] Extend residency to basis/pipeline transforms, tree histograms, classifier
  likelihoods, neural forward/backward products, GP solves, derivative
  operators, and L-BFGS-B objective/gradient evaluations. A mixed CPU/GPU graph
  must expose every transfer and cannot claim full-device execution.
- [ ] Lower the fixed no-autodiff portions of robust/random-forest tree
  prediction/training, discriminant Gaussian scoring, and common NLL/reduction products to resident
  CUDA kernels when OpenACC cannot preserve the declared residency or
  determinism. Keep differentiable paths on generated FortAD/FortSym products
  until matching device JVP/VJP/HVP kernels and transfer accounting exist.
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
- [x] Extend the feature lane with an independent central-difference-of-VJP
  oracle and timing row for polynomial/Fourier basis-pipeline HVPs. The raw
  record is [`results/features_workloads.csv`](../fortml-bench/results/features_workloads.csv)
  and the contract is documented in [`results/FEATURES.md`](../fortml-bench/results/FEATURES.md).
- [ ] Add a release-app benchmark for joint basis-pipeline training, including
  linear and Fourier initializations, FortOpt convergence, and a typed CUDA
  refusal row.
- [x] Add the deterministic seeded random-forest classifier benchmark with a
  direct NumPy threshold oracle, aligned probability-simplex checks, CPU fit
  and prediction timings, and an explicit CUDA refusal in
  [`results/RANDOM_FOREST.md`](../fortml-bench/results/RANDOM_FOREST.md).
- [x] Add the deterministic randomized-threshold Extra-Trees classifier
  benchmark with an independent direct NumPy threshold oracle, aligned
  probability-simplex checks, CPU fit and prediction timings, and an explicit
  CUDA refusal in [`results/EXTRA_TREES.md`](../fortml-bench/results/EXTRA_TREES.md).
- [x] Add the named grouped-MLP regularization benchmark with an independent
  linear-ridge value/gradient/JVP/HVP oracle, FortOpt-ready packed products,
  and an explicit CUDA derivative-graph refusal in
  [`results/MLP_GROUPED_TRAINING.md`](../fortml-bench/results/MLP_GROUPED_TRAINING.md).
- [x] Add the PCA-initialized tied linear-autoencoder lane with an independent
  centered thin-SVD reconstruction oracle, exact RMSE agreement, CPU timing,
  and an explicit CUDA refusal in
  [`results/LINEAR_AUTOENCODER.md`](../fortml-bench/results/LINEAR_AUTOENCODER.md).
- [x] Add the exact depth-limited recursive XGBoost-style squared/logistic lane
  (including depth/node diagnostics), explicit learned/forced NaN routing, and the
  fitted-scaler plus binary and one-vs-rest multiclass Laplace GP
  logistic/probit lanes with independent NumPy oracles. The release records are
  [`results/XGBOOST.md`](../fortml-bench/results/XGBOOST.md) and
  [`results/CLASSIFICATION_EXTENSIONS.md`](../fortml-bench/results/CLASSIFICATION_EXTENSIONS.md).
- [x] Extend the derivative-observation GP benchmark and independent covariance
  oracle to the cosine kernel and polynomial mixed-observation HVP. Query
  JVP/VJP, joint covariance, hyperparameter HVP, and explicit CUDA refusal rows
  are recorded in
  [`results/DERIVATIVE_GP.md`](../fortml-bench/results/DERIVATIVE_GP.md).
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
- [x] Add the deterministic mini-batch SGD trajectory hypergradient objective
  over `[log(learning_rate), log(l2)]`. Its private seeded batch cursor is
  replayed for every FortOpt evaluation, and exact per-batch MLP HVPs provide
  value/gradient, JVP, and scalar VJP products for validation MSE. The
  independent `test_mlp_minibatch_hypergradient` fixture covers central
  differences, adjoints, optimizer convergence, and the CUDA refusal; the
  release workload is `fortml_bench_mlp_minibatch_hypergradient`.
- [x] Add independent kNN uniform/inverse-distance, RMSprop direct/MLP, and
  binary/multiclass staged-XGBoost benchmark lanes. Their reports and raw
  records are [`results/KNN.md`](../fortml-bench/results/KNN.md),
  [`results/RMSPROP.md`](../fortml-bench/results/RMSPROP.md), and
  [`results/XGBOOST.md`](../fortml-bench/results/XGBOOST.md).
- [x] Add correctness-gated deterministic k-means and robust median-IQR
  preprocessing benchmark lanes. The NumPy oracles, FortML timings, and typed
  CUDA refusals are recorded in [`results/KMEANS.md`](../fortml-bench/results/KMEANS.md)
  and [`results/ROBUST_SCALER.md`](../fortml-bench/results/ROBUST_SCALER.md).
- [x] Extend the dense RBF-SVM lane with CPU device dispatch for decision and
  probability JVP/VJP products plus typed CUDA derivative refusals. The
  independent score/label oracle and refreshed release record are in
  [`results/RBF_SVM.md`](../fortml-bench/results/RBF_SVM.md).
- [x] Add exact and weighted-histogram XGBoost monotonic-constraint benchmark
  rows. The independent NumPy harness parses complete query vectors, checks
  adjacent monotonicity, and records CPU fit/predict timings plus explicit
  CUDA resident-kernel refusals in
  [`results/XGBOOST_MONOTONIC_CONSTRAINTS.md`](../fortml-bench/results/XGBOOST_MONOTONIC_CONSTRAINTS.md).
- [x] Add the fixed full-batch RMSprop hypergradient release app and
  correctness-gated NumPy central-difference lane. The packed five-component
  product, centered branch, and explicit CPU/CUDA capability rows are recorded
  in [`results/RMSPROP_HYPERGRADIENT.md`](../fortml-bench/results/RMSPROP_HYPERGRADIENT.md).
- [x] Add resident CUDA device-contract gates for kNN prediction, the
  no-autodiff RMSprop state recurrence, and dense-affine value/JVP/VJP across
  the eight resident CUDA MLP activation codes. Stable sigmoid and Mish have
  explicit typed CUDA-refusal rows. Independent NumPy fixtures, concise pass/skipped/failed
  CSV rows, and hardware/revision provenance are recorded in
  [`results/DEVICE_CONTRACTS.md`](../fortml-bench/results/DEVICE_CONTRACTS.md);
  resident timing and end-to-end model GPU parity remain open.
- [x] Add the resident no-autodiff Adagrad state gate with an independent
  CPU recurrence and explicit unavailable CUDA row. The release record is
  [`results/CUDA_ADAGRAD.md`](../fortml-bench/results/CUDA_ADAGRAD.md); it
  records state residency only, not a complete GPU trainer or derivative graph.
- [x] Add the transfer-inclusive native CUDA weighted-MSE reduction with an
  independent scalar oracle, an unavailable stub, and a real-toolchain gate.
  The device-contract benchmark records pass/skipped/failed status and keeps
  transfer-inclusive metric evidence separate from resident estimator claims.
- [x] Add a bounded binary/shared-kernel GP-classification hyperparameter lane
  with a NumPy mode/envelope-gradient oracle. The evidence is explicitly for
  mode log posterior rather than full Laplace evidence:
  [`results/GP_CLASSIFICATION_TRAINING.md`](../fortml-bench/results/GP_CLASSIFICATION_TRAINING.md).
- [x] Extend the generic hyperparameter-search benchmark with an eight-start
  seeded bounded L-BFGS-B row. The independent quadratic oracle checks the
  retained best state, start count, evaluation budget, and typed CUDA refusal
  in [`results/HYPERPARAMETER_SEARCH.md`](../fortml-bench/results/HYPERPARAMETER_SEARCH.md).
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

The current API/evidence boundary is summarized in
[`docs/PHYSICS_MODELS.md`](docs/PHYSICS_MODELS.md); the current benchmark slice
is [`../fortml-bench/results/PHYSICS_MODELS.md`](../fortml-bench/results/PHYSICS_MODELS.md).

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

- [x] Define a bounded `physics_constraint_t` callback with normalized residual
  value, JVP, VJP, and an optional exact reverse-over-forward HVP product.
  `physics_objective_t` composes data, PDE/ODE residual, initial/boundary, and
  conservation slots and adapts its value/gradient path to FortOpt. Providers
  without the optional HVP callback retain a typed refusal; providers with it
  supply the exact weighted least-squares HVP without forming a Jacobian or
  Hessian. Independent affine and nonlinear residual oracles are in
  `test_physics_objective`; callbacks retain ownership of state, coordinates,
  units, and device residency.
- [x] Add the CPU `pinn_training_adapter_t` facade over
  `physics_objective_t`. It forwards the four named data/residual/boundary/
  conservation terms and value/gradient/JVP/VJP/HVP products, exposes bounded
  FortOpt L-BFGS-B fitting, and returns a typed CUDA refusal without a host
  fallback. `test_pinn` is an independent manufactured-solution gate covering
  all products, a nonlinear HVP, fitting, and shape/device boundaries.
- [ ] Extend the seam with symplectic-form terms, nondimensionalizing
  transforms, coordinate/time layout metadata, and richer named diagnostics
  for every term. The bounded `physics_objective_t%term_values` diagnostic now
  exposes normalized data/residual/boundary/conservation contributions (zero
  for inactive slots, summing to the objective value), with an independent
  affine oracle in `test_physics_objective`; dedicated PINN and
  physics-informed GP training adapters remain future work while preserving
  separate data/residual/boundary/conservation weights.
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
`19e8cda7ad71990339f9ed254cc40128fcbff364` and `fortsym` `main` at
`1477a6d1d9fc659e0a0b537af8a292bb971d4992`. The checked-in RBF HVP/product
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
- [x] Extend the Hamiltonian MLP to a general nonseparable scalar H(q,p) with
  canonical `J`: `initialize_general` stores one full-state scalar MLP and
  exposes exact energy, canonical vector-field, JVP, VJP, and HVP-backed
  vector-field-JVP products. The independent test covers parameter/state
  finite differences and the adjoint identity. The explicit split leapfrog
  method returns `FORTNUM_NOT_IMPLEMENTED` in general mode; an implicit
  symplectic integrator is still required. Learned skew/Poisson structures
  remain open and require independent skew-symmetry and Jacobi tests.
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
- [x] Add PCA initialization for linear autoencoders, following the exact
  Baldi--Hornik optimum above. The empirical [principal-component
  initialization proposal](https://doi.org/10.1007/978-3-030-30484-3_14)
  (Suzuki and Sakanashi, 2019) is a separate deep-autoencoder warm start, not
  a claim about the exact linear optimum. The encoder and decoder use the
  selected principal subspace, with centering, whitening, rank, and sign
  conventions recorded. The reconstruction oracle must match the PCA
  projection to numerical tolerance.
- [x] Reuse the public `pca_t` centered-SVD state for a linear autoencoder
  initializer. Tied and untied decoder choices, rank
  truncation, whitening, and sign conventions must produce the same projected
  reconstruction oracle. The current `linear_autoencoder_t` implements the
  tied, centered, unwhitened reconstruction and exact input JVPs; untied
  decoders and nonlinear starts remain separate work.
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

- [`PARITY_MATRIX.md`](../fortml-bench/results/PARITY_MATRIX.md) is the
  family-level status index. It separates CPU correctness, resident CUDA
  correctness, transfer-inclusive measurements, and typed refusals.

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
- [`ROBUST_SCALER.md`](../fortml-bench/results/ROBUST_SCALER.md), backed by
  `robust_scaler.csv` for median-IQR fit/transform/inverse/JVP products and
  the typed CUDA refusal.
- [`KMEANS.md`](../fortml-bench/results/KMEANS.md), backed by `kmeans.csv` for
  deterministic seeded Lloyd fit/transform, inertia, and the typed CUDA
  refusal.
- [`RBF_SVM.md`](../fortml-bench/results/RBF_SVM.md), backed by `rbf_svm.csv`
  for the independent dense score/label oracle and CPU/CUDA derivative-device
  contract.
- [`LINEAR_SVM.md`](../fortml-bench/results/LINEAR_SVM.md), backed by
  `linear_svm.csv` for weighted arbitrary-label primal SVM fit/predict,
  signed-margin oracle checks, and the explicit CUDA refusal.
- [`LINEAR_SVR.md`](../fortml-bench/results/LINEAR_SVR.md), backed by
  `linear_svr.csv` for weighted arbitrary-target primal SVR fit/predict,
  packed-parameter and prediction oracle checks, and the explicit CUDA
  refusal.
- [`NEURAL_LOSSES.md`](../fortml-bench/results/NEURAL_LOSSES.md), backed by
  `neural_losses.csv` for BCE, weighted multilabel BCE, ordered cumulative-
  logit ordinal loss, softmax cross-entropy, weighted-MSE, Huber, and
  weighted-MLP HVP products.
- [`KERNEL_CATALOG.md`](../fortml-bench/results/KERNEL_CATALOG.md), backed by
  `kernel_catalog.csv` for periodic, rational-quadratic, cosine, and
  polynomial value/input/parameter products plus typed CUDA refusals.
- [`DISCRIMINANT_ANALYSIS.md`](../fortml-bench/results/DISCRIMINANT_ANALYSIS.md),
  backed by `discriminant_analysis.csv` for weighted LDA/QDA probabilities,
  predictions, fitted-state diagnostics, input JVPs, and CUDA refusals.
- [`BASIS_PIPELINE_TRAINING.md`](../fortml-bench/results/BASIS_PIPELINE_TRAINING.md),
  backed by `basis_pipeline_training.csv` for the joint Fourier/linear
  objective, derivative gate, and typed CUDA refusal.
- [`DEVICE_CONTRACTS.md`](../fortml-bench/results/DEVICE_CONTRACTS.md), backed by
  `device_contracts.csv` for resident kNN, forest, MSE, optimizer-state, and
  dense-affine value/JVP/VJP CUDA correctness gates.
- [`MLP_CLASSIFIER_OBJECTIVE.md`](../fortml-bench/results/MLP_CLASSIFIER_OBJECTIVE.md),
  backed by `mlp_classifier_objective.csv` for weighted softmax objective
  products, bounded FortOpt L-BFGS-B, and the typed CUDA refusal.
- [`CUDA_ADAGRAD.md`](../fortml-bench/results/CUDA_ADAGRAD.md), backed by
  `cuda_adagrad.csv` for the resident no-autodiff Adagrad recurrence and
  explicit unavailable-CUDA contract.
- [`XGBOOST_ROBUST.md`](../fortml-bench/results/XGBOOST_ROBUST.md), backed by
  `xgboost_robust.csv` for independent Huber and quantile objective oracles.
- [`DERIVATIVE_GP.md`](../fortml-bench/results/DERIVATIVE_GP.md), backed by
  `derivative_gp.csv` for exact periodic and rational-quadratic mixed-query
  JVP/VJP products and typed CUDA refusals.
- [`SPECTRAL_MIXTURE.md`](../fortml-bench/results/SPECTRAL_MIXTURE.md), backed by
  `spectral_mixture.csv` for the GPyTorch-compatible spectral-mixture value,
  input derivative, parameter JVP/VJP/HVP products, and typed CUDA refusal.
- [`PHYSICS_OBJECTIVE.md`](../fortml-bench/results/PHYSICS_OBJECTIVE.md), backed
  by `physics_objective.csv` for composable residual products, exact nonlinear
  reverse-over-forward HVPs, provider refusals, and the callback CUDA boundary.
- [`PINN.md`](../fortml-bench/results/PINN.md), backed by `pinn.csv` for the
  manufactured four-slot PINN adapter, exact nonlinear HVP, bounded L-BFGS-B
  fit, and typed CUDA refusal.
- [`HAMILTONIAN_GENERAL.md`](../fortml-bench/results/HAMILTONIAN_GENERAL.md),
  backed by `hamiltonian_general.csv` for the independent nonseparable
  canonical-field/Jacobian oracle, full-state HNN products, separable
  symplectic checks, typed general-leapfrog refusal, and explicit GPU boundary.
- [`GROUP_KFOLD.md`](../fortml-bench/results/GROUP_KFOLD.md), backed by
  `group_kfold.csv` for deterministic group isolation, balanced fold indices,
  and the CPU-only index-splitter contract.
- [`HYPERPARAMETER_SEARCH.md`](../fortml-bench/results/HYPERPARAMETER_SEARCH.md),
  backed by `hyperparameter_search.csv` for bounded grid, seeded random, and
  FortOpt L-BFGS-B search evidence.
- [`ADAMW_HYPERGRADIENT.md`](../fortml-bench/results/ADAMW_HYPERGRADIENT.md),
  backed by `adamw_training.csv` and `mlp_hypergradient.csv` for independent
  AdamW recurrence and fixed-trajectory hypergradient finite-difference oracles.
- [`ADAM_HYPERGRADIENT.md`](../fortml-bench/results/ADAM_HYPERGRADIENT.md),
  [`ADAGRAD_HYPERGRADIENT.md`](../fortml-bench/results/ADAGRAD_HYPERGRADIENT.md),
  [`RMSPROP_HYPERGRADIENT.md`](../fortml-bench/results/RMSPROP_HYPERGRADIENT.md),
  and [`ADAMW_BETA_HYPERGRADIENT.md`](../fortml-bench/results/ADAMW_BETA_HYPERGRADIENT.md)
  cover exact fixed-trajectory optimizer products. The new
  [`ADAFACTOR_HYPERGRADIENT.md`](../fortml-bench/results/ADAFACTOR_HYPERGRADIENT.md)
  companion lane for unfactored Adafactor is backed by
  `adafactor_hypergradient.csv` and records the active-set and CUDA refusal
  gates.
- [`TRAINING_IMPUTER.md`](../fortml-bench/results/TRAINING_IMPUTER.md), backed
  by `training_imputer.csv` for Adam-independent momentum-SGD/Nesterov MLP
  trajectories and mean/median/constant imputer transform/JVP/VJP products.
- [`TRAINER_CHECKPOINT.md`](../fortml-bench/results/TRAINER_CHECKPOINT.md),
  backed by `trainer_checkpoint.csv` for uninterrupted-versus-resumed
  optimizer trajectories and malformed/truncated/extra-record refusals.
- [`MLP_BINARY_OBJECTIVE.md`](../fortml-bench/results/MLP_BINARY_OBJECTIVE.md),
  backed by `mlp_binary_objective.csv` for weighted BCE objective products,
  mixed HVPs, and bounded L-BFGS-B behavior.
- [`GP_VARIATIONAL_MULTICLASS_CLASSIFICATION.md`](../fortml-bench/results/GP_VARIATIONAL_MULTICLASS_CLASSIFICATION.md),
  backed by `gp_variational_multiclass_classification.csv` for sorted-label
  OVR ELBO/probability/JVP oracles and the typed CUDA refusal.
- [`XGBOOST_WARM_START.md`](../fortml-bench/results/XGBOOST_WARM_START.md),
  backed by `xgboost_warm_start.csv` for suffix continuation, fresh-fit and
  Newton-stump staged-margin agreement, transactional target/control refusals,
  and the explicit CUDA-unavailable row.
- [`XGBOOST_RANKING.md`](../fortml-bench/results/XGBOOST_RANKING.md), backed by
  `xgboost_ranking.csv` for group-isolated pairwise loss derivatives,
  ordering, and singleton refusal.
- [`XGBOOST_DERIVATIVES.md`](../fortml-bench/results/XGBOOST_DERIVATIVES.md),
  backed by `xgboost_derivatives.csv` for independent fixed-tree JVP/VJP
  checks away from split surfaces, boundary refusals, and the explicit CUDA
  refusal; the XGBoost slicing API has a separate staged-prefix oracle.
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
7. Expose FortML probability/model protocols to FortMC and FortBO. Keep MCMC,
   chain diagnostics, acquisition policies, and candidate-search state in
   those companion projects, while retaining priors, likelihoods, posterior
   objects, variational objectives, and differentiable log densities here.
   Benchmark HMC/NUTS and Bayesian optimization against pinned PyMC/Stan/
   BoTorch/GPyTorch references before claiming parity.
8. Add model schemas, C ABI, serving, MPI/sharded execution, and reproducibility
   manifests once in-memory trainer state and ownership rules are stable.
9. Add physics constraints, Hamiltonian/Lagrangian/symplectic models, operator
   GPs, and Ghosttasking/Monge-GP prototypes behind residual and structure
   oracles. Use current FortAD main and FortSym-generated kernels where proven.
10. Add NNGP, PCA, autoencoder, and physics-consistent initializers as explicit
   experiments before making any initializer a default. Expand release
   benchmarks after every slice, and run the `ifx` compiler lane when available.
