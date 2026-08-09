# fortml roadmap

Verified on 2026-08-09. Interfaces are documented in
[`docs/API.md`](docs/API.md), examples in [`docs/EXAMPLES.md`](docs/EXAMPLES.md),
and implementation limits in [`docs/DESIGN.md`](docs/DESIGN.md) and
[`docs/ML_ARCHITECTURE.md`](docs/ML_ARCHITECTURE.md).
The cross-library acceptance table is maintained in
[`docs/PARITY_MATRIX.md`](docs/PARITY_MATRIX.md).

## Verification

The GitHub `v0.1.0` tag currently points to the earlier release-verification
commit `a387cc5`; the trainer, calibration, variational-GP, transform, tree
attribution, binary-GP log-probability, fixed-leaf-product, plateau-trainer,
and CUDA VJP closure slices documented below are post-tag additions.
The broad parity
gate is still open, so this work does not move or recreate that tag.
The checklist currently records 348 completed and 127 open items; open rows are
retained until their implementation, independent oracle, device/refusal
behavior, and benchmark evidence land together.

The current parity wave closes the following bounded contracts: chronological
expanding/rolling validation with scorer and clone/reset metadata,
multiclass focal-softmax value/JVP/VJP/HVP products wired through MLP and
FortOpt objectives, RBF order-three derivative observations, implicit binary
GP-classification kernel hyperparameter HVPs, weighted contrastive loss
products, stable XGBoost classifier log-probability products with categorical
metadata, explicit MLP loss-scale state, interval-routed conditional feature
unions, differentiable cross-validation scoring, and the LightGBM-style
multiclass OVR adapter, exact ordinal-GP evidence gradients/HVPs with bounded
FortOpt optimization, and weighted multilabel MLP FortOpt objectives with
direct or positive log-L2 coordinates, multiclass Laplace-GP hyperparameter
HVPs, the shared trainer learning-rate schedule seam, and fixed-topology
boosted-tree leaf objectives, mixed-observation Matérn 3/2 and 5/2 derivative-GP
hyperparameter HVPs, and Hamiltonian vector-field VJPs. Each has an independent
CPU oracle, a typed CUDA boundary, and a pinned `fortml-bench` record. The wave
also closes finite-feature GP posterior variance/regularization JVP products
and named PINN per-term gradient/HVP diagnostics. The broad parity gate remains
open for the explicit rows below.

The weighted Huber linear-regression slice adds `huber_regression_t` and the
registry-backed `huber_training_objective_t`. It supports vector and
multi-output targets, nonnegative sample weights, optional intercept and L2
regularization, bounded FortOpt L-BFGS-B fitting, and optional packed L2 and
delta coordinates for outer search. Value/gradient/JVP/VJP products are exact;
the objective HVP includes mixed coefficient/L2/delta blocks on a fixed
residual branch and returns `FORTNUM_NOT_IMPLEMENTED` at the Huber kink.
`test_huber_regression`, `app/fortml_bench_huber_regression`, and the dedicated
`fortml-bench` release record provide the independent finite-difference,
adjoint, convergence, and typed CUDA-refusal evidence. Resident CUDA Huber
kernels, quantile fitting, and derivative-through-fit remain separate gaps.

## Next production parity wave

The next waves preserve the shared registry/objective/device/state contract and
prioritize the user-visible parity gaps in this order:

1. **Classification completion:** finish the common estimator adapter for
   binary, multiclass OVR/OVO, multilabel, classifier-chain, ordinal, calibrated,
   Naive-Bayes, discriminant, SVM, neighbor, forest, bagging, AdaBoost,
   histogram, XGBoost, LightGBM, and GP classifiers. Every adapter must expose
   sorted-label metadata, weights, probabilities/decisions, persistence,
   fixed-state products, and an explicit nonsmooth boundary.
2. **GP breadth:** complete derivative observations for every smooth kernel,
   operator layouts, multitask and likelihood state, variational/natural-gradient
   products, deep/physics-aware compositions, and implicit hyperproducts through
   Laplace/variational optima. GPyTorch/GPflow-style inducing and scalable
   workflows remain CPU-first until resident CUDA factorization and transfer
   counters are verified.
3. **Neural training:** finish the module tree (convolution, normalization,
   dropout, attention, transformer, graph, LSTM/GRU, neural operators), loader
   and accumulation state, optimizer groups, mixed precision/master weights,
   checkpoint migration, and exact optimizer/schedule/validation hypergradients
   consumed by FortOpt L-BFGS-B. PINN/HNN/symplectic and physics-consistent
   networks share the same products and manufactured-physics benchmarks.
4. **Boosting and composition:** extend the fixed-topology XGBoost/LightGBM
   objective into histogram growth, GOSS/DART, ranking, categorical/missing
   routing, monotone/interaction constraints, staged predictions, contributions,
   warm starts, and resident GPU histograms. Complete basis/pipeline DAGs with
   polynomial, spline, Fourier, radial, GP, PCA, autoencoder, NNGP, and NTK
   initializers that preserve names, offsets, leakage guards, and derivatives.
5. **Performance closure:** for each completed row, run matched NumPy and
   scikit-learn/PyTorch/JAX/GPyTorch/XGBoost/LightGBM workloads where applicable,
   then add resident CUDA/OpenACC timing, memory, transfer, precision, and
   refusal evidence. No benchmark or release claim closes a row by CPU fallback.

## Production-parity execution contract

The remaining work is executed as one clean-break architecture rather than as
isolated estimator additions. Every new implementation must land at the same
five seams: a named parameter registry, a composable value/objective graph, a
FortOpt derivative callback, a resident device plan, and a versioned state
dictionary. The acceptance order for each family is:

1. **Classification:** binary, multiclass, OVR/OVO, multilabel, chains,
   ordinal, calibrated, Naive Bayes, discriminant, SVM, neighbors, trees,
   forests, and boosted trees share labels, weights, class metadata,
   probabilities, decisions, metrics, persistence, and typed nonsmooth
   boundaries. Missing-value and categorical policies are part of the fitted
   topology, never hidden preprocessing.
2. **Gaussian processes:** exact, derivative-observation, Laplace,
   variational, sparse, SKI/lazy, local, multi-output, deep-kernel, and
   physics-aware models share kernel-expression registries, likelihood state,
   observation layouts, solve plans, and input/parameter products. A kernel or
   likelihood can enter FortOpt only when its gradient and directional HVP are
   declared and independently checked; unsupported derivative orders return a
   typed status.
3. **Neural networks:** MLP, convolutional, recurrent, attention, graph,
   autoencoder, VAE, BNN, PINN, HNN, and symplectic variants use one module
   tree and flat named parameter layout. The trainer owns batches, schedules,
   clipping, accumulation, mixed precision, checkpointing, validation, and
   optimizer state. Training and outer hyperparameter search consume the same
   objective, so FortOpt L-BFGS-B sees exact network, loss, and optimizer
   hypergradients rather than a second approximation.
4. **Boosted/tree models:** XGBoost and LightGBM objectives, histogram/exact
   growth, DART/GOSS, ranking, categorical and missing-value routing,
   monotonic/interaction constraints, warm starts, staged predictions, and
   contributions use one immutable split-topology representation. Leaf/base
   coordinates may be differentiated on a fixed topology; split selection,
   sampling, active sets, and early stopping remain explicit discrete seams.
5. **Basis and pipelines:** polynomial, spline, Fourier, Chebyshev, radial,
   random-feature, GP, PCA, autoencoder, feature-union, conditional, and DAG
   maps compose through named schemas and stable feature/parameter offsets.
   Joint basis coefficients and transform hyperparameters are optimized by
   the same objective and can be initialized from the best linear/PCA/GP
   solution before nonlinear training.
6. **Device and precision:** CPU is the independent behavioral oracle. CUDA
   and OpenACC plans own resident data, workspaces, optimizer state, precision,
   and transfer counters. No host fallback is allowed after device selection;
   a missing resident kernel is a typed refusal. FortSym-generated CUDA is
   preferred for fixed no-autodiff algebra, while FortAD/FortSym products are
   retained for autodiff-bearing paths until resident tests pass.
7. **State and interoperability:** every fitted estimator and trainer saves a
   schema-versioned dictionary containing topology, named parameters and
   transforms, optimizer/schedule/RNG state, validation history, precision,
   device, dependency revisions, and checksums. Loading, cloning, warm-start,
   and partial-fit operations are transactional and preserve prediction and
   derivative oracles at the declared boundary.
8. **Benchmark acceptance:** every completed row has a pinned source and
   dependency revision, an independent NumPy/analytic/scikit-learn/PyTorch/
   JAX/GPyTorch/XGBoost/LightGBM oracle where applicable, CPU and resident
   device timings, memory and transfer counters, refusal rows, and a release
   application. Correctness is promoted before performance comparisons, and
   performance claims are made only for matching workload, precision, and
   residency.

This contract is the implementation filter for the open checklist below. A
feature is not complete when a type exists; it is complete when its registry,
derivatives, state, device behavior, independent oracle, and benchmark all
agree.

The generic trainer now closes the first shared schedule seam. Its
`trainer_options_t%learning_rate_schedule` applies the existing typed
constant, warmup, cosine, exponential, and one-cycle rates to every streaming
optimizer without reinitializing moments, persists the schedule configuration
in checkpoint schema 5, and refuses plateau schedules or L-BFGS-B combinations
that need validation/state not owned by this trainer. The independent
`test_trainer_schedule` recurrence and continuation oracle is the source gate;
the release benchmark is retained in `fortml-bench` alongside the model-family
trajectory lanes. Schedule hyperparameter gradients/HVPs remain model-specific
until their objective adapters expose the same derivative provider.

The multilabel neural-objective slice adds
`mlp_multilabel_training_objective_t` and
`mlp_multilabel_optimize_lbfgsb`. The objective copies indicator data and
effective sample-by-label weights, validates every target and weight before
installing a model pointer, and exposes weighted BCE value/gradient, scalar
JVP/VJP, and exact network/L2 HVP products. The packed tail can hold a direct
nonnegative L2 coordinate or a positive `log(l2)` coordinate. The bounded
FortOpt path routes the same callback through network and L2 bounds and reports
the physical coefficient. `test_mlp_multilabel_objective` independently
checks central finite differences, adjoint duality, mixed HVPs, malformed
weight transactionality, and both coordinate modes. CUDA remains a typed
resident-graph refusal. The companion release record is
`results/MLP_MULTILABEL_OBJECTIVE.md` in `fortml-bench`.

The 2026-08-09 multi-output boosting slice adds transactional
`xgboost_multioutput_t` and `lightgbm_multioutput_t` adapters.  Each adapter
fits one deterministic regression child per target, preserves row-oriented
sample/output conventions, exposes staged margins and output/feature/stage
metadata, and concatenates fixed-structure leaf coordinates for parameter
JVP/VJP products.  Input JVP/VJP products route through all children and retain
their split-boundary refusals.  The independent `test_xgboost_multioutput`
oracle covers closed-form one-tree/two-output values, staged products,
adjoint identities, malformed-fit transactionality, and selected-CUDA
refusals.  `fortml-bench` records the CPU NumPy oracle and typed CUDA rows.
Resident multi-output CUDA tree/histogram kernels, distributed output
parallelism, and differentiable split topology remain open.

The same closure wave adds `basis_residual_pipeline_t`, a named two-branch
residual-sum DAG with transactional branch configuration, stable feature and
parameter offsets, CPU value/JVP/VJP/HVP products, and output-preserving typed
CUDA refusals. `test_basis_residual_pipeline` and the independent NumPy lane in
`fortml-bench/results/BASIS_RESIDUAL_PIPELINE.md` cover the sum, adjoint, mixed
HVP, metadata, and refusal contracts. Conditional/cyclic graphs, sparse/device
graphs, and resident GPU execution remain open.

The conditional feature-union slice adds `conditional_basis_pipeline_t`, a
parallel interval-routed layer over the existing column-basis primitive. Named
branches retain stable feature/parameter offsets and route metadata; exact CPU
value/JVP/VJP/HVP products preserve inactive rows and accumulate selected-column
cotangents. Append, packed-parameter, and dense-schema updates are
transactional, while interval-endpoint derivatives, malformed/nonfinite routes,
and CUDA requests return typed status without touching caller state. The
independent oracle is `test_conditional_pipeline`; release evidence is
`results/CONDITIONAL_PIPELINE.md` in `fortml-bench`. Cyclic/DAG routing,
resident route-mask GPU execution, and graph-wide persistence remain open.

The current closure wave also adds transactional versioned persistence for
preconfigured fitted basis pipelines, affine schedule outer HVPs consumable by
FortOpt L-BFGS-B, and validation-aware multiclass XGBoost OVR training with
weighted log-loss, common-prefix early stopping, best-prefix restoration,
schema-2 metadata validation, and staged probability records. Each slice has a
CPU oracle, a typed CUDA boundary, and a benchmark record in
`fortml-bench`.

The mixed-precision infrastructure slice adds `mlp_loss_scale_state_t`, a
validated deterministic finite-update growth/overflow-backoff recurrence with
skipped-update accounting. The FP64 CPU reference can exercise the policy and
skip a scaled-gradient overflow without changing the disabled default
trajectory. Dynamic state and static policy are validated in memory and in
formatted checkpoint schema 11; FP32/FP16/BF16 and resident CUDA requests
remain typed refusals until master-weight and lower-precision kernels have
independent gates. `test_mlp_loss_scaling` and the companion benchmark provide
the independent recurrence, persistence, and refusal evidence.

The binary Laplace-GP derivative slice now adds an implicit-mode kernel
hyperparameter HVP. `gp_classification_t%hyperparameter_hvp` differentiates the
converged Newton mode through the resident posterior factorization, then uses
the analytic kernel parameter HVP/VJP primitives; `hyperparameter_hvp_device`
keeps CPU dispatch explicit and returns typed CUDA status `3`. Logistic and
probit refit finite differences, transactional parameter refusal, and the
release app are independently checked by `test_gp_classification_hvp` and
`fortml-bench/results/GP_CLASSIFICATION_HYPERPARAMETER_HVP.md`. Full evidence,
likelihood-parameter, coupled multiclass, and resident-GPU HVPs remain open.

The ordinal-GP evidence slice now exposes the exact latent-Gaussian
log-marginal-likelihood gradient and directional HVP over the packed kernel
and log-noise coordinates. `gp_ordinal_optimize_hyperparameters` consumes
those products with bounded FortOpt L-BFGS-B, reports convergence and final
gradient diagnostics, and restores the initial state on failure. Independent
finite-difference and optimizer checks live in
`test_gp_ordinal_classification_hyperparameters`; CPU timing and typed CUDA
refusal rows are pinned in
`fortml-bench/results/GP_ORDINAL_HYPERPARAMETERS.md`. Native cumulative
ordinal likelihoods, optimized cut points, and resident GPU solves remain
open.

The multiclass Laplace-GP closure adds a block-packed directional
`hyperparameter_hvp` over the independent sorted-label OVR models. Each class
delegates to the exact binary implicit-mode product, while the device method
returns the same explicit resident-CUDA refusal until a multiclass factorization
graph is linked. `test_gp_multiclass_classification` checks the HVP against
independent refit-gradient finite differences, and the release record is
`fortml-bench/results/GP_MULTICLASS_HYPER_HVP.md`.

The fixed-topology boosted-tree objective closure adds
`boosted_leaf_objective_t` for XGBoost and LightGBM. It maps the existing
`[base_score, leaf weights]` coordinates into weighted squared or logistic
objectives with exact value/gradient/JVP/VJP/HVP products and bounded FortOpt
L-BFGS-B fitting. Split routing, categorical partitions, sampling, and early
stopping remain immutable discrete state; CPU is the reference and selected
CUDA returns a typed refusal. `test_boosted_leaf_objective`, the release app,
and `fortml-bench/results/BOOSTED_LEAF_OBJECTIVE.md` provide the independent
oracle, optimizer, and provenance gate.
The smooth Matérn derivative-observation slice closes the mixed likelihood HVP
gap for Matérn 3/2 and 5/2. `gp_derivative_regression_t%hyperparameter_hvp`
now propagates exact packed `[log(variance), log(lengthscale),
log(noise_variance)]` products through value/first-derivative covariance blocks,
the dense Cholesky solve, and the likelihood matrix pullback. The radial
`f`, `f'`, and `f''` products use generated Matérn value/HVP kernels and exact
polynomial parameter tangents, including finite coincident-point limits; no
finite-difference implementation is hidden in the production path.
`test_derivative_gp_matern_hvp` independently checks both leaves against a
central-difference covariance oracle and verifies the selected-CUDA typed
refusal. Rational-quadratic, cosine, user-formula, resident-CUDA, and higher
operator-observation products remain explicit open rows.

The rational-quadratic derivative-observation slice now closes the next smooth
kernel HVP row. `gp_derivative_regression_t%hyperparameter_hvp` propagates
analytic mixed value/first-derivative covariance products through all three
logarithmic kernel coordinates (`log(variance)`, `log(lengthscale)`, and
`log(alpha)`) plus log noise. The radial `F_s`/`F_ss` products and their
parameter-direction products are assembled without finite differences, while
the existing exact query-input JVP/VJP path supplies the third-input products.
`test_derivative_gp_products` independently reconstructs the dense
one-dimensional covariance, checks the packed gradient/HVP against central
differences, and verifies that selected CUDA prediction remains a typed
`FORTNUM_NOT_IMPLEMENTED` refusal. The release app and independent benchmark
are `fortml_bench_derivative_gp_rational_quadratic_hvp` and
`fortml-bench/results/DERIVATIVE_GP_RATIONAL_QUADRATIC_HVP.md`; cosine,
user-formula, higher operator orders, and resident CUDA covariance remain open.

The metric-learning loss slice adds a reusable weighted pairwise contrastive
objective to `fortml_losses`: matching/non-matching Euclidean pairs expose
value, JVP, VJP, and HVP products with mean/sum reductions.  Independent
finite-difference and adjoint tests cover the products, while exact
non-matching zero-distance and margin-kink requests return typed domain
refusals.  The value device dispatcher keeps CUDA an explicit
`FORTNUM_NOT_IMPLEMENTED` boundary until a resident pair-distance/reduction
kernel exists.  Triplet, sequence/CTC, and resident CUDA loss contracts remain
open; see [`docs/CONTRASTIVE_LOSS.md`](docs/CONTRASTIVE_LOSS.md).

| Compiler | Command | Result |
| --- | --- | --- |
| GNU Fortran | `fo` | Static build, all 270 behavioral tests, and lint passed at the current integrated FortML/FortAD-main revisions (599 modules, 922 build units). The compiler still emits non-fatal array-temporary warnings; see [`verification/fortml-gfortran.txt`](verification/fortml-gfortran.txt). |
| NVIDIA HPC SDK | `FO_FC=nvfortran fo` | Static and lint checks passed in the recorded older compiler lane. The checked-in NVIDIA log predates the current 270-test GNU run. See [`verification/fortml-nvfortran.txt`](verification/fortml-nvfortran.txt). |
| Intel LLVM Fortran | `ifx` | Compiler unavailable in the verification environment. Not tested. |

The checked-in GNU compiler log is the fresh 2026-08-09 run against FortML code
revision `c652b7d` (including classifier-chain, weighted Huber, and
rational-quadratic GP products plus the earlier chronological validation metadata, multiclass
focal-softmax products, RBF order-three derivative observations, implicit
binary GP-classification HVPs, contrastive loss products, stable XGBoost
classifier log probabilities, multiclass XGBoost validation/early stopping,
pipeline persistence, affine schedule outer HVPs, SAMME.R probability updates,
categorical likelihood temperature HVPs, and finite-feature GP/NTK last-layer
initialization, explicit MLP loss scaling, conditional feature-union routing,
and differentiable cross-validation scoring, exact ordinal-GP evidence
products, normalized LightGBM OVR multiclass probabilities, weighted
multilabel MLP FortOpt objectives, multiclass Laplace-GP HVPs, generic trainer
learning-rate schedules, and fixed-topology boosted-tree objectives, in addition to scheduled AdamW trajectory hypergradients,
calibrated-softmax OOF policies, affine schedule
outer HVPs, and seeded XGBoost DART), FortAD `origin/main` at
`d71cdf724cd8c4f10d849493beaa7c459cd3a96a`, FortFront at
`86eb2ba8b5b842bc1aebf9ee8bb00053f12de2f8` (not a direct fpm dependency), and FortNum at
`7ced2f7aa272920916789fa82a35bfcb2e792d45`, run from the clean checkout
in the clean canonical checkout under `/mnt/storage/code/lazy-fortran/fortml`;
transient checkpoint outputs were moved to the recoverable Trash after
verification.
The run includes the
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
 positive multiclass temperature calibration, multiclass Platt calibration,
 PINN objective fitting, unfactored
 Adafactor recurrence/checkpoint and relative-step/parameter-scale products, the portable trainer checkpoint, pairwise XGBoost
 ranking, physics residual value/JVP/VJP products, and the
 transform-aware hyperparameter registry, sparse variational-GP ELBO products,
 XGBoost warm-start continuation, LightGBM matched-option warm starts,
 RBF second-derivative GP observations, sparse-GP kernel hyperparameter products,
 classifier-chain logistic products, MLP optimizer groups/checkpoint metadata,
 and differentiable basis-pipeline
training objective, fixed-active-set optimizer-group clipping products, affine
constant-schedule outer HVPs with the FortOpt callback seam, seeded XGBoost DART
tree scales with transactional warm-start controls, and calibrated-softmax OOF
temperature, weighted Platt, and isotonic policies, plus coupled categorical
variational-GP likelihood products, bounded SHAP-like XGBoost/LightGBM
attributions, and model-agnostic trainer validation diagnostics.
The build emits non-fatal GNU
array-temporary warnings in FortFront query/generator calls, existing GP
benchmark boundaries, variational-GP batch conversions, and basis-pipeline
shape conversions. They are isolated to array construction; all 270 behavioral
tests pass. Lint has zero unused-import findings and the full `fo` lint stage
passes despite the non-fatal compiler warning corpus. The independent CUDA gate additionally covers the
resident dense-affine value/JVP/VJP path and its single-layer MSE update with
parameter snapshots and transfer counters. NVIDIA
compiler coverage remains an
explicit older-build result.

The checked-in evidence is maintained on the clean FortML-bench revision
`8149aca`; each CSV records the exact clean benchmark revision used to produce
its rows (older CSVs retain their historical provenance),
the trainer-checkpoint, unfactored-Adafactor, binary-objective,
multiclass-calibration, variational-multiclass-GP, PINN/physics-objective,
physics HVP, grouped K-fold, spectral-mixture, XGBoost-ranking,
resident dense-MSE CUDA, binary XGBoost classifier, calibrated neural classifier,
SGD-momentum, sparse preprocessing, derivative-GP covariance and polynomial HVP,
weighted multiclass MLP objective, resident Adagrad, resident RMSprop, random
Fourier basis, Matérn-5/2 derivative-GP, locally-periodic GP, fixed-shape Gamma
XGBoost, leakage-safe calibrated logistic CV, and mini-batch hypergradient
CSV rows record their FortML source revisions and independent NumPy or analytic
behavioral oracles. The basis-pipeline lane now includes the optimized-ridge
coordinate/mixed-HVP case, and the binary Laplace-GP parameter-product test has
an independent fixed-state finite-difference oracle. The current closure slices
also include the pinned exact-GP reference lane in
`fortml-bench/results/GP_REFERENCE.md`, which checks FortML's predictive mean
and variance against scikit-learn on a shared closed-form fixture and records
optional GPyTorch as an explicit unavailable row when its environment is not
installed. They also cover AMSGrad recurrence/MLP training, batched
multi-output GP query
products, typed column-pipeline CPU/CUDA dispatch, packed OVR Laplace-GP
parameter products, weighted LightGBM validation
early stopping, and scheduled-Adagrad and scheduled-AdamW trajectory hypergradients. The latter has
its own CPU-product and CUDA-refusal rows in
`results/mlp_adagrad_schedule_hypergradient.csv`. CUDA rows are explicit `unavailable`/typed-refusal records
rather than host timings. The XGBoost warm-start lane adds an independent
Newton-stump replay, transactional refusal rows, and an explicit CUDA-
unavailable record in `results/xgboost_warm_start.csv`. The classifier-chain
lane adds packed-head NumPy replay, integer-label prediction checks, exact joint
input/parameter probability HVP checks, fit/predict/HVP timing, and explicit
CUDA-unavailable rows in `results/classifier_chain.csv`. The XGBoost
interaction-constraint lane adds
an independent NumPy group-mean/path-mask oracle for unconstrained and
separated-group depth-two trees; its CPU rows pass with zero prediction error
and its CUDA row is an explicit unavailable capability record in
`results/xgboost_interaction.csv`.
The final closure batch additionally records metric-aware plateau transitions,
random-forest OOB probabilities and coverage, the PCA-seeded linear MLP
reconstruction optimum, AMSGrad trajectory products, and spectral-mixture
derivative-GP mixed parameter HVPs with independent NumPy rows and typed CUDA
boundaries.
The current post-tag lanes add binary Laplace-GP log-probability value and
input/parameter JVP/VJP products with transactional fixed-state updates,
fixed-structure XGBoost/LightGBM leaf-coordinate JVP/VJP products, and
metric-aware plateau training with persisted diagnostics and deterministic
split/resume recurrence. Their independent NumPy or hand oracles and typed
CUDA-unavailable rows are pinned in the benchmark data and documentation head
`dadc7a1`.
The ARD derivative-GP lane adds dense mixed-observation input JVP/VJP and
parameter HVP rows in `results/DERIVATIVE_GP.md`; the RAdam trajectory lane is
`results/MLP_RADAM_HYPERGRADIENT.md`, and the Tweedie tree lane is
`results/XGBOOST_TWEEDIE.md`. Each includes explicit CUDA-unavailable rows.
The random-Fourier feature lane records analytic value/JVP/VJP/HVP rows, and
the Matérn-5/2 FortSym HVP lane adds its generated-kernel oracle row to the
same derivative-GP workload.
The Chebyshev basis lane records independent recurrence and NumPy finite-
difference oracles for value/JVP/VJP/HVP products, CPU timings, and a typed
resident-CUDA refusal in `fortml-bench/results/CHEBYSHEV_BASIS.md`.
The current bounded release lanes also include the scalar Matérn-5/2
second-derivative GP observation contract (`results/second_derivative_gp.csv`),
production Lion trainer recurrence and resume/EMA checks
(`results/lion_training.csv`), and named basis fan-out/fan-in JVP/VJP/HVP
products (`results/basis_fanout_pipeline.csv`). Each row pins the current
FortML and benchmark revisions and records CUDA as unavailable where no
resident implementation exists.
The latent-Gaussian ordinal GP and Student-t process lanes add independent
NumPy/analytic contract rows with typed CUDA boundaries; the LightGBM lane now
also records versioned text and binary persistence round trips and malformed-
record refusals. Known-noise heteroskedastic GP has an independent CPU oracle
in `test_heteroskedastic_gp` and a release lane in
`results/HETEROSKEDASTIC_GP.md`.
The robust Poisson/Student-t Laplace GP has matched stationarity, positive-rate,
outlier-resistance, and refusal rows in `results/ROBUST_GP.md`. The release
also records the scalar RBF second-derivative GP covariance/JVP/VJP lane,
transactional LightGBM warm starts and persistence, and the fixed-active-set
optimizer-group clipping trajectory with explicit HVP and CUDA boundaries.

The previous three-slice evidence is pinned to the clean revisions above. The
multiclass `xgboost_multiclass_t` lane persists class metadata and tree state
in one strict text file, rejects malformed or truncated records transactionally,
and matches an independent stable-sigmoid NumPy oracle in
`fortml-bench/results/XGBOOST_MULTICLASS_PERSISTENCE.md`. The layout-aware
matrix-factored Adafactor lane migrates row, column, and vector state through
in-memory and formatted schema-9 checkpoints, with independent continuation
and malformed-layout checks in
`fortml-bench/results/ADAFACTOR_FACTORED.md`. The locally-periodic derivative
GP lane provides analytic value/first-derivative covariance products,
coincident-safe input gradients and mixed Hessians, and parameter JVPs with
finite-difference and adjoint checks in
`fortml-bench/results/DERIVATIVE_GP_LOCAL_PERIODIC.md`.

The latest closure slice adds scheduled-RAdam trajectory hypergradients,
periodic-kernel derivative-observation mixed parameter HVPs, and deterministic
LightGBM GOSS sampling. The RAdam objective differentiates log-rate, L2,
beta, epsilon, and schedule coordinates through CPU trajectories and feeds
FortOpt L-BFGS-B; the GP lane covers all periodic log-kernel/noise coordinates
with coincidence-safe fourth-input products; GOSS records top/other gradient-
Hessian reweighting, persistence, replay, and transactional rate validation.
Their CSV rows pin clean source revisions `af6273b`, `7c2a004`, and `0fe7eff`
to generating benchmark revisions `ef88c30`, `e307326`, and `14afd70`,
respectively. CPU products pass independent oracles (GOSS replay error is zero;
the periodic HVP oracle error is `2.53e-6`), and unsupported CUDA/outer-HVP
paths remain explicit typed refusals.

The latest closure slice adds four independently oracle-backed lanes. Dense
horizontal/sequential/fan-out basis pipelines now validate transactional input
schemas and stable feature names (`results/PIPELINE_SCHEMA.md`); the typed
one-cycle trajectory objective differentiates peak and final-rate coordinates
through warm-up and cosine updates (`results/MLP_ONE_CYCLE_HYPERGRADIENT.md`);
and multiclass calibration adds weighted one-vs-rest isotonic maps with simplex
renormalization (`results/MULTICLASS_ISOTONIC_CALIBRATION.md`). This slice now
also adds smooth weighted one-vs-rest Platt sigmoid maps with exact input and
packed-parameter products (`results/MULTICLASS_PLATT_CALIBRATION.md`). Their
checked-in CSV rows pin FortML `33a5f8a` and clean generating benchmark
revisions `187d2ff` (pipeline), `ced3cee` (one-cycle), and `a86b8d8`
(isotonic). The Platt release row is pinned to FortML `e1359ce` and clean
FortML-bench `a91c050`.
CPU oracle errors are below `5e-12` for the one-cycle products and `2e-16` for
the isotonic lane, with CUDA and unsupported HVP/active-set paths recorded as
typed refusals.

The second 2026-08-08 continuation closes three additional bounded lanes.
LightGBM DART uses deterministic seeded prior-tree dropout with persisted tree
scales, warm-start/slice parity, and an independent depth-one NumPy replay;
its exact generating row pins FortML `628316d` and FortML-bench `d3ad400`.
The generic multiclass calibrator now has weighted one-vs-rest Platt sigmoid
maps with simplex renormalization and smooth packed-parameter products; its
release row pins `e1359ce`/`a91c050` and reports probability error
`2.58e-12`, parameter error `1.73e-10`, and simplex error `2.22e-16`.
The fixed-SGD momentum validation objective accepts positive nonuniform
validation weights with exact value/gradient/JVP products, retains the uniform
HVP path, and returns typed refusals for nonuniform HVP and CUDA; its
independent NumPy row pins `8f0b705`/`232e7b12` and records weighted MSE
`0.014998256378050192`. The calibrated-softmax OOF wrapper now routes
temperature, weighted one-vs-rest Platt sigmoid, and weighted isotonic maps
through the same leakage-safe stratified folds. Temperature and Platt retain
exact packed input/parameter products; isotonic values are complete while
active-set products remain typed refusals. The wrapper state records the
calibration method, packed calibration count, knot count, and derivative
availability.

The affine constant-schedule HVP closure adds an exact outer HVP for the
one-layer affine MLP objective, with central finite-difference, JVP/VJP,
Hessian-symmetry, and FortOpt callback checks. The contract requires a linear
output; the hidden activation is ignored for a one-layer model, while
nonconstant schedules, nonlinear outputs, and CUDA remain typed boundaries.
Its independent NumPy lane is recorded in
`fortml-bench/results/MLP_CONSTANT_SCHEDULE_HVP.md` and
`mlp_constant_schedule_hvp.csv` (maximum oracle error `2.2693e-8`).

The seeded XGBoost DART closure propagates deterministic prior-tree dropout
and per-tree normalization through staged predictions, contributions, slicing,
schema-5 persistence, and transactional warm-start control matching. The
independent NumPy lane is recorded in `fortml-bench/results/xgboost_dart.csv`;
prediction error is `8.88e-16`, replay/persistence/warm-start errors are zero,
and the resident-CUDA path is an explicit typed refusal.

The validation-aware LightGBM continuation closure evaluates weighted
validation loss from the retained prefix before suffix growth, carries
patience and minimum-delta decisions transactionally, and supports both
restore-best and retain-all policies for regression and binary objectives.
`fortml-bench/results/lightgbm_validation_warm_start.csv` reports the
independent NumPy weighted-loss oracle (`40.5` best loss, best round `1`,
patience stop at `3`), zero transactional-prefix error, and typed CUDA status
`3`; resident LightGBM histogram execution remains an explicit refusal.

### 2026-08-08 parity and provenance slice

This verification records the current bounded production contracts without
changing the broader parity claim. `classifier_chain_t` fits one binary logistic head per
output, accepts arbitrary sorted integer label pairs, supports shared sample
weights, per-output class weights and thresholds, and uses observed positive
indicators during fitting followed by smooth positive probabilities at
prediction. Its packed head parameters have exact input and parameter JVP/VJP
products plus a joint input/parameter forward-over-reverse HVP of a probability
cotangent. The HVP differentiates every smooth chain-feature edge and logistic
sigmoid second derivative; hard labels and fit-time optimizer decisions remain
discrete. CPU dispatch is tested and selected CUDA calls, including HVP, return
`FORTNUM_NOT_IMPLEMENTED`.
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
`test_mlp_optimizer_groups`; the schema is now format 7/text schema 5. CUDA
optimizer-group execution, mixed precision, distributed state, and migration
remain open. The source and benchmark pins for this earlier optimizer-group
slice were FortML `05632ce8fa95268417c7a2d979fa1461a202abaa` and
FortML-bench `0fb8ac7`; the current aggregate verification is the newer
`cd3e64c`/`8e5a181` pair recorded above.

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

The binary Laplace-GP and sorted-label OVR wrappers now accept the same finite
nonnegative sample-weight contract. Weights enter the Newton likelihood
curvature, mode log posterior, and envelope kernel hypergradient; zero-weight
rows contribute no likelihood curvature. `test_gp_classification_sample_weights`
checks logistic and probit fits, OVR routing, refit finite differences,
malformed weights, and the typed CUDA boundary. The release oracle is
`fortml-bench/results/GP_CLASSIFICATION_SAMPLE_WEIGHTS.md`.

MLP training now includes a deterministic fixed full-batch Lion trajectory
hypergradient. The packed outer coordinates are log learning rate, log L2, and
two logit betas. Analytic JVP/VJP/HVP products feed FortOpt L-BFGS-B directly;
the sign-margin branch is a named nonsmooth refusal and CUDA remains typed
until resident trajectory state is linked. The independent fixture is
`test_mlp_lion_hypergradient`, with the API contract in
`docs/MLP_LION_HYPERGRADIENT.md`.

The current estimator slice also records deterministic binary and multiclass
SAMME AdaBoost over weighted CART weak learners and dense multi-output
closed-radius regression.
`test_adaboost_classifier` and `test_radius_neighbors_multioutput_regression`
provide independent hand-oracle fixtures, while the companion release rows in
`fortml-bench/results/ADABOOST_CLASSIFIER.md` and
`fortml-bench/results/RADIUS_NEIGHBORS_MULTIOUTPUT.md` retain complete CPU
prediction checks, provenance, and typed CUDA refusals. These slices do not
claim SAMME.R, tree-search backends, or resident radius/boosting kernels.

The same release slice now records CPU RAdam flat-state and MLP training with
format-10/text-schema-11 checkpoint replay, an independent NumPy recurrence, and
a typed CUDA-unavailable row in `fortml-bench/results/RADAM.md`. Ordered-gradient
integer categorical XGBoost partitions are covered by
`fortml-bench/results/XGBOOST_CATEGORICAL.md`; the fixture checks the bounded
cardinality policy, deterministic category masks, serialization metadata, and
the discrete-derivative/CUDA refusal boundaries. These reports pin the exact
FortML source and benchmark revisions.

The RAdam derivative slice now exposes `fortml_mlp_radam_hypergradient` for a
fixed full-batch validation trajectory. It propagates exact value, JVP, and
scalar VJP products through the two moments, bias corrections, `rho_t`
rectification, and epsilon denominator, and its packed objective is consumable
by FortOpt L-BFGS-B. `test_mlp_radam_hypergradient` is an independent
finite-difference/adjoint fixture; the `rho_t = 4` branch, zero square-root,
and CUDA paths are typed refusals rather than hidden subgradients or host
fallbacks. The benchmark report is
`fortml-bench/results/RADAM_HYPERGRADIENT.md`.

The deep-kernel GP slice adds `deep_kernel_gp_t`, which composes an MLP feature
map with an exact dense GP whose base kernel is defined on the learned feature
space. `initialize` validates the input/feature widths and keeps the final
feature layer linear. `transform`, `fit`, `predict`, and
`log_marginal_likelihood` expose the composition, while `feature_gradient` and
`weight_gradient` implement the marginal-likelihood chain rule through the
kernel and the MLP reverse pass. `test_deep_kernel_gp` checks identity-feature
reduction to a plain GP, every-weight central differences, feature movement,
and malformed/unfitted refusals. The implementation is a CPU exact-GP
reference with dense cubic scaling. Joint FortOpt training of feature weights
and kernel hyperparameters, KISS-GP/SKI approximations, and resident CUDA
execution remain open.

The locally-periodic kernel slice adds `make_local_periodic_kernel`, a named
four-parameter product of a squared-exponential envelope and periodic factor.
It is integrated as a first-class exact-GP kernel rather than a user-formula
expansion: dense values, coincident-safe input gradients/mixed Hessians,
parameter JVP/VJP/HVP products, and exact-GP posterior mean/variance all have
an independent oracle in `test_local_periodic_gp`. The static kernel operator
and resident CUDA path return a typed refusal until their program ABI carries
the four-parameter leaf; no host fallback is counted as GPU support. The
companion benchmark is `fortml-bench/results/LOCAL_PERIODIC_GP.md`.

The scalar second-derivative GP reference now accepts Matérn-5/2 observations
of orders zero through two, including exact order-four covariance blocks and
order-five input JVP/VJP products away from coincidence points. Its typed
coincidence, unsupported-order, non-RBF, and CUDA boundaries are covered by an
independent oracle and the companion benchmark. The production MLP trainer
also includes stateful Lion updates with decoupled weight decay, clipping,
schedules, optimizer groups, EMA, validation, checkpointing, and text resume.
The independent recurrence/resume lane records the CPU contract and an
explicit CUDA-unavailable row.

The model-agnostic `fortml_trainer` now also exposes a checkpointable Lion
optimizer (`FORTML_TRAIN_LION`). Its beta1 sign/interpolation update, beta2
momentum, decoupled weight decay, finite-state validation, and formatted-text
resume are covered by an independent quadratic oracle in `test_trainer` and
the generic `trainer_lion` rows in
`fortml-bench/results/trainer_checkpoint.csv`. The generic state remains
CPU-resident; CUDA requests remain typed refusals until model, objective, and
optimizer state can stay resident together.

### 2026-08-07 objective-trainer and tree-contribution slice

The model-agnostic `fortml_trainer` core is now a shared full-batch state
machine for any FortOpt objective. It owns explicit SGD, Adam, AdamW,
Adagrad, RMSprop, Lion, and bounded L-BFGS-B state, gradient clipping, optional
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

### 2026-08-08 generic validation-state slice

The model-agnostic `fortml_trainer` now accepts a process-local validation
callback that evaluates a finite scalar after each full-batch update. The
trainer records validation history, best value and step, consecutive
non-improvement count, and a copy of the best packed parameter vector.
`validation_min_delta`, `validation_patience`, and
`validation_restore_best` provide deterministic early stopping and
transactional best-state restoration. The schema-4 text checkpoint persists
the complete validation state and refuses a missing or unexpected callback
because procedure pointers cannot be serialized. The independent quadratic
oracle covers known-answer patience transitions, restoration, split
continuation, and callback-presence refusal. The generic callback remains
host-owned, so CUDA execution is an explicit typed boundary. Release evidence
is `fortml-bench/results/TRAINER_VALIDATION.md`.

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

### Next production parity wave

The next implementation wave uses the family adapters and state dictionary
defined in [`docs/ML_ARCHITECTURE.md`](docs/ML_ARCHITECTURE.md). The priority
order is:

1. Classification wrappers share labels, weights, calibration, and probability
   policies across linear, neural, Laplace-GP, variational-GP, and tree heads.
   Multilabel, multioutput, ordinal, calibrated, and chain variants retain the
   same packed-state and refusal rules.
2. GP workflows close the likelihood, batch-shape, derivative-observation,
   inducing-state, and hyperparameter-product gaps. Every smooth kernel gets a
   FortSym-derived or hand-proved product and an independent dense oracle.
3. Neural training promotes the current MLP trainer into a reusable module-tree
   trainer. Optimizer, schedule, validation, checkpoint, and callback state
   share one registry. FortOpt L-BFGS-B consumes the same exact hyperparameter
   derivatives used by training, including trajectory and implicit products.
4. Boosting adapters add the remaining validation-aware, categorical, missing,
   distributed, SHAP-compatible, and resident-GPU workflows for XGBoost and
   LightGBM. Split topology and stochastic sampling remain declared fit-time
   boundaries.
5. Basis and pipeline graphs route named features and derivatives into every
   estimator. Polynomial, spline, Fourier, radial, GP, and learned features
   use the same transform registry and device plan.

Each slice ships source, tests, documentation, an independent benchmark oracle,
and a clean provenance row. A CUDA row is either resident and transfer-accounted
or an explicit typed refusal. A CPU timing never closes a GPU requirement.

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

- [x] Record the clean-break object graph and ownership rules for data views,
  transform graphs, parameter registries, derivative providers, FortOpt
  objectives/trainers, resident device plans, state dictionaries, and
  benchmark records in [`docs/ML_ARCHITECTURE.md`](docs/ML_ARCHITECTURE.md).
  The document makes the common value/JVP/VJP/HVP contract, typed device
  boundaries, transactional checkpoints, FortSym/FortAD provenance, and the
  migration order explicit. Estimator and backend implementation rows remain
  open until they satisfy that contract with independent evidence.

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
   slicing state, DART/EFB or typed refusals, and resident histogram
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

### 2026-08-09 neural loss product slice

The shared loss facade now provides stable softmax and log-softmax value,
JVP, VJP, and HVP products, weighted mean/sum softmax cross-entropy, and
analytic focal binary cross-entropy HVPs for relaxed targets. The new
multiclass focal-softmax family adds weighted mean/sum value/JVP/VJP/HVP
products with positive class factors and a typed device boundary; multiclass
MLP fit and FortOpt objectives select it through `focal_gamma`. The independent
`test_neural_loss_products` fixture checks central differences, VJP adjoints,
stable extreme logits, weighted reductions, malformed inputs, and the typed
CUDA refusal. The release lane is
`fortml-bench/results/neural_losses.csv`.

The remaining loss catalog is still open: multilabel focal variants,
count/dispersion extensions, contrastive/triplet/CTC objectives,
probabilistic reconstruction, broader PDE residuals, and resident CUDA loss
kernels require separate contracts.

### 2026-08-09 finite-feature GP/NTK initializer slice

`fortml_mlp_last_layer_gp` now provides a deterministic, production-quality
last-layer kernel-ridge initializer for an existing MLP. It freezes the hidden
feature map, solves the regularized augmented normal equations, exposes
transactional final-layer application, named regularization metadata, and an
analytic fixed-feature regularization JVP. An independent closed-form oracle,
central-difference product check, transaction/refusal checks, and typed CUDA
no-mutation boundary are in `test_mlp_last_layer_gp`; the API contract is
`docs/MLP_LAST_LAYER_GP.md` and the release lane is
`fortml-bench/results/mlp_last_layer_gp.csv`.

The initializer now also retains its positive-definite precision matrix and
exposes the exact finite-feature posterior predictive variance diagonal and
its regularization JVP. An independent dense-solve oracle and release row are
recorded in `fortml-bench/results/mlp_last_layer_gp_posterior.csv`; the
resident CUDA variance path remains a typed refusal. This is a finite-feature
posterior product only: NNGP covariance propagation, posterior weight draws,
and structure-preserving GP initialization remain open below.

This closes only the finite-feature last-layer warm-start contract. It does not
close NNGP covariance propagation, sampled prior draws, full GP-posterior
weight maps, or structure-preserving Hamiltonian/symplectic/PINN
initialization; those remain explicit research and implementation gaps.

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
  isolation, and singleton refusal. Categorical/interaction policies, seeded
  XGBoost DART, EFB, distributed growth, and resident GPU histograms remain
  separate bounded or device gaps.
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
- `basis_fanout_pipeline_t` now composes named sequential basis branches in a
  bounded fan-out/fan-in DAG. Forward blocks concatenate, reverse input
  cotangents sum, and branch-local parameters retain stable packed names and
  offsets through exact CPU JVP/VJP/HVP products. The independent oracle is
  `test_basis_fanout_pipeline`; the benchmark records the same value and
  derivative identities plus the typed CUDA refusal. Residual and conditional
  contracts now have separate named APIs; cyclic and resident-GPU graph
  execution remain open.
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
  `test_gp_variational_multiclass_classification`; coupled categorical
  inference is covered by `gp_variational_categorical_classification_t` and
  its independent oracle. Natural-gradient and resident-GPU inference remain
  open.
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
- `multi_output_gp_t` now accepts independent query batches with explicit
  `(batch, query, feature)` to `(batch, query, output)` shape mapping. Batched
  posterior means, input JVPs, and input VJPs preserve the output-major
  coregionalization layout, reject malformed or nonfinite inputs, and expose
  CPU execution plus a typed CUDA refusal. `test_multi_output_gp_batch` and
  `fortml-bench/results/MULTI_OUTPUT_GP_BATCH.md` provide the dense-RBF,
  finite-difference, adjoint, and device evidence; packed parameter products
  over batches remain open.
- `column_basis_pipeline_t` now dispatches fitted transform, input JVP, input
  VJP, and input HVP products through a typed device selector. CPU delegation
  preserves the existing basis-map behavior, while CUDA returns
  `FORTNUM_NOT_IMPLEMENTED` without a host fallback. The independent
  `test_column_pipeline` gate and `fortml-bench/results/COLUMN_PIPELINE_DEVICE.md`
  record the feature-count, product, and refusal contracts; DAG/parallel
  resident execution remains open.

The FortBO and FortMC companion pins were rechecked against their remote
`main` branches on 2026-08-08: FortBO
`b62a1a0bae1c0766fb35a3127957a39758705160` and FortMC
`e5e42a0ac1d4a4d92fa6b2ee2750b50723342a48`. Their roadmaps remain authoritative
for acquisition and sampling algorithms; FortML owns the posterior/log-density
protocols and does not embed sampler or acquisition state. FortBO additionally
provides analytic EI/PI/UCB/log-EI products and marginal Monte-Carlo EI/PI with
common random numbers, antithetic draws, and pathwise gradients, Sobol TuRBO
candidate generation with deterministic Thompson selection, and a FortML
adapter that chooses value-only versus derivative-observation GPs from the
history contract. FortBO now also runs its gradient-based DTuRBO in-region
acquisition search end to end with Sobol multistarts and FortOpt L-BFGS-B,
beating random search on the Branin fixture at equal budget. The current pin
also implements predictive-entropy constraints C1.2 and C2 by
expectation-propagation latent constraints (with the existing C1.1 and C3
variance safeguard), wires all three TuRBO ordering arms and a pinned
Ackley-200 reference comparison, emits posterior-moment derivative leaves from
FortSym, and enforces host placement for incomplete FortAD acquisition graphs.
It also supplies preference learning and noisy-dominance probabilities with
FortSym-generated Gaussian-comparison derivatives, noisy expected improvement,
joint qEI/qNEI/qUCB batch acquisitions, risk-sensitive and multi-fidelity
criteria, constrained/cost-aware acquisitions, active-learning and level-set
design, max-value entropy search, mixed-integer/categorical candidate search,
per-evaluation benchmark metrics, asynchronous worker bookkeeping with
mean/incumbent/worst fantasies and bounded retries, a Bayesian-linear
posterior provider, high-dimensional gradient fixtures, and the 60D rover
trajectory fixture,
paper-aligned predictive-entropy C1.2/C2 conditioning with its variance
safeguard,
and constrained/noisy/multi-objective benchmark fixtures,
FortSym-derived trust-region length rescaling, exact posterior mean and
standard-deviation Hessians from derivative predictions, and tested TuRBO-1/
TuRBO-m and DTuRBO mode-2 drivers with deterministic region updates,
posterior sampling, posterior-derivative local models, trust-region traces,
and an indefinite-curvature bound-constrained quadratic subproblem, multi-objective
Pareto archives with exact hypervolume and scalarizations, and stopping rules
that report a machine-readable reason. qKG and batch Thompson/fantasy
policies, posterior-gradient device execution, and wider sparse/variational
adapters remain open. Any future adapter must add
a focused oracle, typed GPU/refusal row, and a benchmark record in the companion
harness. The current FortBO pin also adds the 60-dimensional rover trajectory
fixture and its independent oracle. It routes FortML multi-output and
deep-kernel GPs through `fortbo_structured` posterior adapters. Multi-task
adoption requires an explicit `target_output`, and both adapters expose moments
only, with named refusals for moment-gradient acquisition search. The pin also
contains resident OpenACC candidate and TuRBO reductions with typed host/device
provenance, a BoTorch/GPyTorch/JAX/NumPy cross-framework correctness and regret
benchmark, and the 14D robot-pushing fixture. The current FortBO pin also
adds a multi-seed 14D TuRBO-1/TuRBO-m versus quasi-random ordering harness.
That harness records the early 14D ordering and its limits: it does not claim
the paper's rover or Ackley-200 baselines, and the rover wiring still returns
an invalid bit-identical comparison that must be fixed before it is used as
evidence.

The LightGBM multiclass slice adds `lightgbm_multiclass_t`, a transactional
one-vs-rest classifier over the leaf-wise binary estimator. It sorts arbitrary
integer labels, normalizes final and staged probabilities, exposes raw margins,
weighted validation best-prefix metadata, and routes fixed-tree input JVP/VJP
products through the sigmoid/normalization chain rule. Split surfaces and
selected CUDA devices retain typed boundaries. `test_lightgbm_multiclass` and
`fortml-bench/results/lightgbm_multiclass.csv` provide the independent CPU and
NumPy evidence.

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
only a position-valued `value` and a position-gradient `gradient`; its shipped
gradient-free univariate coordinate-sweep sampler is exposed as
`fortmc_slice_sample`/`fortmc_slice_chain`, while packed parameter registries,
transforms, HVPs, HMC/NUTS, and general chain state remain in its roadmap.
FortBO now exposes a capability-gated posterior protocol, durable
gradient-aware history, and normalized continuous/integer/categorical/mixed/
conditional search spaces, analytic EI/PI/UCB/log-EI, exact-envelope knowledge
gradient, noisy expected improvement, and marginal Monte-Carlo EI/PI with CRN,
antithetic, and pathwise-gradient products. It also supplies
Sobol TuRBO candidates, discrete Thompson selection, a tested FortML
value-only/derivative-observation GP adapter, tested multi-task and deep-kernel
FortML posterior adapters, and tested gradient-based DTuRBO mode-2 local models
and in-region optimization, plus trust-region traces. Full batch, entropy and
predictive-entropy search, posterior-gradient device execution, and
sparse/variational adapters remain in its roadmap. Resident candidate and
TuRBO reductions are available when OpenACC is present and otherwise return a
typed refusal. Adapters map FortML posterior
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

The dependency pins used by the current GNU verification are FortAD `7c65a88`,
FortFront `0bce426`, FortSym `26250ce`, FortOpt `883aa7e`
(`release/context-objective`), and FortNum `7ced2f7`, all checked against
their corresponding remote branches on 2026-08-08. Generated derivatives
record the exact FortSym revision and source hash; model-level autodiff uses
the same FortAD `main` pin.

The companion implementation roadmaps are authoritative for their sampler and
acquisition work packages:

- [`fortmc/ROADMAP.md`](https://github.com/lazy-fortran/fortmc/blob/main/ROADMAP.md)
- [`fortbo/ROADMAP.md`](https://github.com/lazy-fortran/fortbo/blob/main/ROADMAP.md)

The companion repositories were checked on 2026-08-08 at FortMC
`e5e42a0ac1d4a4d92fa6b2ee2750b50723342a48` and FortBO
`b62a1a0bae1c0766fb35a3127957a39758705160`, both on their `main` branches. The
FortBO pin now includes a versioned capability-gated posterior contract,
gradient-aware observation history/checkpointing, normalized continuous/integer/
categorical/mixed/conditional search spaces, a differentiable-coordinate mask,
analytic EI/PI/UCB/log-EI, exact-envelope knowledge gradient, noisy expected
improvement, joint qEI/qNEI/qUCB batch acquisitions, risk-sensitive and
multi-fidelity criteria, constrained/cost-aware acquisitions, active-learning
and level-set design, max-value entropy search, mixed-integer/categorical
candidate search, per-evaluation benchmark metrics, marginal Monte-Carlo EI/PI
with CRN, antithetic draws and
pathwise gradients, Sobol TuRBO candidates, Thompson selection,
gradient-based DTuRBO in-region acquisition search, FortSym-derived
trust-region length rescaling, exact posterior mean and
standard-deviation Hessians, tested TuRBO-1/TuRBO-m and DTuRBO mode-2 drivers,
trust-region traces, and an
indefinite-curvature quadratic subproblem, Pareto archives with
exact hypervolume, scalarizations, machine-readable stopping reasons,
preference learning, noisy dominance, the FortML derivative-GP input-HVP
adapter, FortML value/derivative-GP adapters, asynchronous worker bookkeeping
with selectable fantasies and bounded retries, a Bayesian-linear posterior
provider, fixed-choice/constraint-penalty feasibility utilities, constrained
and multi-objective fixtures, paper-aligned PES C1.2/C2 conditioning, and the 60D
rover trajectory fixture, multi-task and deep-kernel posterior adapters,
resident candidate/TuRBO reductions, FortSym-generated posterior-moment
derivatives, host-placement refusals for incomplete autodiff graphs, cross-
framework correctness/regret fixtures, and the 14D robot-pushing fixture;
the three-arm TuRBO ordering harness and pinned Ackley-200 reference are
included, while the standalone OpenACC hardware probe and acquisition-speed
benchmark remain separately gated on a dependency-complete GPU host;
refresh these pins when
their protocol or device contracts change.

The FortBO pin includes a standalone OpenACC hardware probe and an
acquisition-level wall-clock benchmark; these are separate from the
auto-discovered test tree and require a dependency-complete GPU host before
publishing device timings. The 14D ordering harness is a separate slow fixture
and remains limited to the pushing arm. Against the current b62a1a0 tip,
`fo check --json=compact` stops at the build stage because the generated
`fortbo_generated_acquisition_leaf_` symbol is not linked; this is a FortBO
boundary failure, not FortML verification evidence.
FortMC's current checkout builds cleanly and passes its one registered
slice-sampler test (normal and correlated moments, bounded support,
reproducibility, and refusal cases); the remaining samplers, diagnostics, and
checkpoint claims remain roadmap items rather than FortML verification evidence.

This companion check was repeated from clean source trees on 2026-08-08:
`origin/main` resolves exactly to the two pins above, and neither repository
has a runtime dependency on the FortSym executable. FortMC's artifact-cleanup
tip does not expand its sampler catalog: its slice sampler remains the only
registered sampler evidence, while HMC/NUTS/SMC, diagnostics, checkpoints,
and device execution remain roadmap items. These are boundary checks, not a
claim that the remaining FortBO policy catalog or FortMC samplers are shipped.

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

Calibration update (2026-08-08): `CALIBRATION_TEMPERATURE` fits a positive
scalar temperature for binary, pre-oriented logits and exposes exact score and
temperature-parameter JVP/VJP products. Weighted equal-width reliability
diagrams now have a deterministic metric/API and benchmark oracle.
`calibrated_logistic_classifier_t` adds leakage-safe binary calibration from
stratified out-of-fold margins, fold diagnostics, and exact smooth products.
The standalone `multiclass_probability_calibrator_t` now fits weighted
one-vs-rest Platt sigmoid and isotonic maps with simplex normalization.
`calibrated_softmax_classifier_t` routes both policies through deterministic
stratified out-of-fold logits and refits the deployment softmax on all rows;
temperature, one-vs-rest Platt, and weighted one-vs-rest isotonic are all
available. Smooth temperature/Platt products are exact, isotonic active-set
products and all CUDA paths are typed refusals, and failed OOF fits preserve a
previous deployment transactionally. Multiclass generic estimator routing
remains open.

| Family | Required variants | Current FortML baseline | Missing production gates |
| --- | --- | --- | --- |
| Classification | binary, multinomial/softmax, OVR, OVO, multilabel, Naive Bayes, LDA/QDA, tree, neural, dense RBF one-class SVM, Laplace GP, variational GP, calibrated, ordinal | binary/softmax/OVR/OVO and logistic/MLP/independent Laplace-GP multilabel heads with per-label thresholds, weighted ordinal cumulative-logit heads, weighted Gaussian/Bernoulli/Multinomial/Complement/Categorical NB, weighted LDA/QDA, CART, MLP, dense RBF nu-SVM, weighted binary/OVR Laplace GP, bounded Bernoulli variational GP (logistic/probit) including sorted-label OVR multiclass, coupled categorical variational GP with FortOpt fitting and analytic packed/input products, latent-Gaussian ordinal GP, positive temperature, Platt sigmoid, weighted PAVA isotonic calibration, weighted reliability-diagram points, exact and histogram boosted trees including normalized LightGBM OVR multiclass probabilities | sparse/multioutput multilabel, native ordinal GP likelihoods and optimized cut points, natural-gradient and resident-GPU training, shared preprocessing/search, and kernel-SVM parity |
| Regression | OLS, weighted/ridge/lasso/elastic-net, robust, quantile, GLM, multi-output, partial-fit | dense OLS, weighted ridge, weighted elastic-net/lasso, weighted linear SVR, registry-backed weighted Huber regression with optional L2/delta outer coordinates and fixed-branch HVPs, weighted Poisson/Gamma log-link GLM, exact/histogram XGBoost-style squared/squared-log (RMSLE)/Huber/quantile/absolute/Tweedie regression with fixed-state products, multi-output fixed-fit products | positive/Bayesian/ARD, quantile/partial-fit, derivative-through-fit, resident GPU kernels |
| Ensembles | CART, random/extra forests, bagging, AdaBoost, histogram boosting, XGBoost/LightGBM ranking/categorical/DART | weighted CART, deterministic seeded random-forest with stored bootstrap inclusion, transactional OOB decision probabilities/accuracy/coverage, fixed-state deterministic accuracy permutation importance with independent NumPy replay, and randomized-threshold Extra-Trees classification, seeded bootstrap bagging over CART, binary and multiclass SAMME and SAMME.R probability-update AdaBoost over weighted CART, squared/squared-log/Huber/quantile/absolute/Tweedie boosting, exact and bounded histogram second-order XGBoost-style binary/OVR, bounded `rank:pairwise`, exact/histogram per-feature monotonic and interaction-group constraints, bounded ordered-gradient integer categorical partitions with explicit cardinality refusal, bounded weighted LightGBM-style leaf-wise regression/binary, sorted-label normalized OVR LightGBM multiclass, and seeded XGBoost/LightGBM DART paths with validation loss, patience/min-delta early stopping, best-iteration metadata, restore-best slicing, weighted validation objectives, versioned persistence, transactional matched-option warm starts, deterministic GOSS top/other-rate gradient/Hessian reweighting, persisted tree normalisation, and bounded exact-subset SHAP-like raw-margin attributions | Full SHAP interaction/explanation workflows, categorical policies beyond ordered partitions, XGBoost EFB, distributed, validation-aware warm-start, differentiable routing, and resident GPU histograms |
| Gaussian processes | exact, derivative observations, multitask, sparse/variational, SKI/lazy, local experts, classification | exact and derivative GPs with RBF, Matérn, periodic, rational-quadratic, cosine, polynomial, linear, constant, white-noise, user, sum, and product leaves, sparse/local/SKI/structured operators, weighted binary/OVR and independent multilabel and robust Poisson/Student-t Laplace paths, bounded weighted Bernoulli variational classification with logistic/probit ELBOs, coupled categorical variational classification with variance-corrected softmax and fixed-state temperature HVPs, latent-Gaussian ordinal classification, dense Student-t and known-noise heteroskedastic process regression, packed sparse-GP mean/log-Cholesky ELBO gradients/JVPs/VJPs, transformed fixed-state sparse-GP Gaussian-likelihood log-noise ELBO JVP/VJP/HVP products with transactional setters, weighted envelope kernel hypergradients, fixed-state binary/OVR/multilabel Laplace latent/probability kernel-parameter JVP/VJP products, and fixed-state binary/OVR variational kernel-log products | full likelihood/kernel catalog, native ordinal likelihoods, batch/multitask likelihoods, inducing-state and remaining likelihood hyperparameter products, implicit derivatives, natural gradients, resident GPU solves |
| Neural and physics models | MLP, CNN, RNN/GRU/LSTM, attention, autoencoder/VAE, BNN, HNN/LNN/symplectic/PINN | MLP/MLP classifier, named sequential `mlp_chain_t`, BNN, VAE, vanilla RNN, Hamiltonian MLP, composable four-slot physics residual objective, weighted Poisson objective with exact HVP/L-BFGS-B, Lion fixed-trajectory hypergradients, selected optimizer hypergradients, exact fixed-trajectory scheduled-Adagrad, scheduled-RAdam, scheduled-AdamW hyperparameter objectives, and finite-feature GP/NTK last-layer kernel-ridge initialization | broader module tree, recurrent/attention/convolution families, full products, complete loss catalog, NNGP covariance and structure-aware GP initialization, physics samplers/adapters, and long-horizon GPU gates |

### Capability matrix (target versus current state)

| Area | Current state | Production target |
| --- | --- | --- |
| Linear regression and generalized linear models | Linear regression, weighted ridge and weighted elastic-net/lasso coordinate descent, weighted Poisson/Gamma log-link GLMs with bounded FortOpt L-BFGS-B, and logistic/softmax sample and positive sorted-class weights are implemented | Robust, quantile/Tweedie, multinomial, calibrated and regularized classifiers with shared solver and derivative contracts; resident GLM GPU kernels |
| Feature transforms and basis maps | Polynomial, interaction-polynomial, Chebyshev, Fourier, fixed-state random Fourier, radial, B-spline, callback bases, standard/min-max/median-IQR scalers, sparse-safe CSC standard scaling, integer categorical one-hot encoding, transactional dense input schemas with unique names, horizontal/sequential/column pipelines, named residual-sum DAGs, analytic basis/pipeline HVPs, a fitted basis-to-linear estimator, and a joint differentiable basis-pipeline training objective are implemented. The objective can also pack a nonnegative ridge coordinate with exact gradient and mixed HVP blocks | CSR/CSC categorical and indicator views, conditional/cyclic DAGs, estimator-wide metadata routing, leakage-safe cross-validation, callback second derivatives, and resident GPU transforms |
| Nearest-neighbor and margin methods | Dense exact kNN and closed-radius classification plus scalar and multi-output regression, weighted linear SVM/SVR, and dense RBF one-class SVM | KD-tree or ball-tree search, sparse inputs, kernel SVM/SVR, calibrated probabilities, resident GPU kernels, and differentiable soft-neighbor policies |
| Trees and ensembles | Partial | Deterministic finite-only regression stumps, weighted depth-limited CART regression and classification, seeded bootstrap random-forest classification with stored inclusion state, OOB decision probabilities/accuracy/coverage and explicit insufficient-coverage refusal, fixed-state deterministic accuracy permutation importance with transactional output and typed CUDA refusal, seeded randomized-threshold Extra-Trees classification, seeded bootstrap bagging classification, binary and multiclass SAMME and multiclass SAMME.R probability-update AdaBoost over weighted CART, squared-loss stump boosting, exact/histogram depth-limited second-order squared/logistic/Poisson/Tweedie/squared-log/Huber/quantile boosting, and bounded `rank:pairwise` boosting are implemented. XGBoost-style trees support weighted quantile cuts, bounded histograms, explicit NaN rejection, learned default directions, forced-left/right routing, per-feature monotonic and interaction-group constraints with recursive leaf bounds/masks, bounded ordered-gradient integer categorical partitions with explicit max-category refusal, staged predictions, contributions, serialization, transactional fitted-prefix slicing, and seeded DART/dropout scales through staged/contribution/slice/warm-start/schema-5 persistence; bounded seeded LightGBM DART/dropout also persists tree-normalisation scales, and `lightgbm_multiclass_t` adds sorted-label normalized OVR probabilities, common validation prefixes, raw margins, and fixed-tree input products; SHAP workflows, categorical policies beyond ordered partitions, XGBoost EFB, distributed growth, differentiable routing, and resident GPU histograms remain planned |
| Clustering and unsupervised learning | Centered dense `pca_t` is implemented with deterministic SVD signs, rank selection, whitening, reconstruction, variance metadata, and fixed-state input products; `linear_autoencoder_t` reuses fitted PCA as the tied linear optimum with exact encode/reconstruction JVPs; deterministic dense seeded `kmeans_t` provides fit/predict/transform, inertia, and fixed-center input products with explicit empty-cluster and device refusals | Incremental/randomized/sparse/kernel PCA, ICA, NMF, minibatch k-means, Gaussian mixtures/EM, density and graph clustering, manifold methods, outlier detection, matrix factorization, and density metrics |
| Neural networks | MLP/BNN/VAE/RNN primitives, a separable Hamiltonian MLP, a named sequential `mlp_chain_t` parameter tree, dense MLP linear/`tanh`/ReLU/GELU/SiLU/ELU/softplus/leaky-ReLU/sigmoid/Mish products, deterministic MLP Adam/AdamW/Adagrad/RMSprop/SGD/Adafactor training, exact fixed full-batch SGD momentum/Nesterov/AdamW/Adam/RMSprop/Adagrad/Adafactor trajectory hypergradients including scheduled RAdam/AdamW, Adafactor relative-step, and parameter-scaling smooth branches, weighted binary BCE, multiclass cross-entropy, and Poisson log-rate FortOpt/L-BFGS-B objectives with exact mixed HVPs, bounded full-batch MLP and composed-chain L-BFGS-B paths, named group-wise log-L2 hyperparameters with exact mixed HVPs, fixed-SGD optimizer-group multiplier hypergradients, portable trainer checkpoints, callback-driven validation diagnostics with patience and best-state restoration, resident dense-affine CUDA value/JVP/VJP plus single-layer MSE-update primitives, and a resident no-autodiff CUDA Adagrad state plan exist | Alias-aware module/buffer tree, the remaining activation/loss/module catalog, convolution/attention/sequence/graph extensions, mixed precision, distributed training, compile/fusion, serialized/distributed trainers, and resident multi-layer neural training |
| Gaussian processes | Exact, derivative, sparse, structured and local variants are partial-to-implemented. Exact fitted GPs and binary/shared-kernel one-vs-rest Laplace classifiers have bounded FortOpt L-BFGS-B adapters; bounded Bernoulli variational classification has deterministic logistic/probit ELBO, packed gradients, a bounded FortOpt L-BFGS-B adapter, prediction JVPs/VJPs, minibatch scaling, and typed CUDA refusal. Sparse variational GPs now expose packed mean/log-Cholesky ELBO gradients/JVPs/VJPs with central-difference and adjoint oracles plus a separate transformed Gaussian-likelihood log-noise block with analytic fixed-state JVP/VJP/HVP products, transactional updates, and typed CPU/CUDA boundaries. Binary Laplace prediction additionally exposes fixed-state kernel-parameter JVP/VJP products for latent and observed probabilities | GPyTorch/GPflow-style kernels, likelihoods, multitask/batch shapes, exact/variational/lazy inference, derivative operators, kernel/inducing hyperparameter products, constraints, calibration, coupled multiclass GP classification, natural gradients, evidence-corrected and likelihood-parameter training |
| Derivatives | Exact GP, analytic polynomial/Chebyshev/Fourier/radial/spline basis and pipeline HVPs, and selected neural/kernel products exist | Value/JVP/VJP/HVP and implicit/hypergradients for every declared parameter/input path, including preprocessing, likelihood, optimizer/search variables, and device kernels |
| Model selection and metrics | Benchmark-specific checks exist; `fortml_validation` now accepts the shared `estimator_capability_t` contract for pre-flight validation, chronological expanding/rolling time-series splits with gap controls, and explicit scorer/clone/reset metadata | Shared metrics, repeated/grouped/Monte Carlo cross-validation scoring, calibration, grid/random/Bayesian/differentiable search, nested validation, and leakage/refusal checks |
| Persistence and serving | Partial | Fitted horizontal basis-pipeline unions now have a versioned compiler-independent host text dictionary with transactional metadata/parameter restore and typed CUDA refusal; estimator-wide state dictionaries, safe model/trainer serialization, streaming inference, batching, and reproducible deployment manifests remain open |
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
| Classification | Binary, softmax, OVR, OVO, multilabel, ordinal, five Naive Bayes variants, weighted LDA/QDA, CART, deterministic random forest, MLP, linear and dense RBF one-class SVM, temperature/sigmoid/isotonic calibration, weighted binary or OVR Laplace GP classification, bounded Bernoulli variational GP classification including sorted-label OVR multiclass, coupled categorical variational GP classification with FortOpt fitting and analytic packed/input products, and a latent-Gaussian ordinal GP baseline | Sparse and multioutput labels, native ordinal GP likelihoods and optimized cut points, natural-gradient and resident-GPU training, shared preprocessing and search, and kernel SVM/margin parity |
| Regression and bases | OLS, ridge, elastic-net, weighted linear SVR, scalar and multi-output closed-radius neighbors with uniform or distance weighting, Poisson/Gamma GLM, PCA, polynomial/Chebyshev/Fourier/radial/spline maps, analytic basis/pipeline HVPs, sequential or column pipelines, joint differentiable basis-pipeline training, and robust/squared-log/quantile/Tweedie XGBoost-style objectives | KD/ball-tree and approximate neighbors, partial fit, sparse views, graph serialization, callback/pipeline second derivatives, and resident GPU execution |
| Trees and boosting | Weighted CART, deterministic seeded random-forest classification with stored inclusion state and transactional OOB probabilities/score/coverage, fixed-state accuracy permutation importance with independent NumPy replay and transactional CPU/CUDA contracts, seeded bootstrap bagging classification, binary and multiclass SAMME and SAMME.R probability-update AdaBoost over weighted CART, exact and bounded-histogram second-order squared/logistic/Poisson/Tweedie/squared-log/Huber/quantile and `rank:pairwise` boosting, staged binary or OVR predictions, diagnostics, monotonic and interaction-group constraints, bounded ordered-gradient integer categorical partitions, transactional fitted-prefix slicing, versioned XGBoost schema-5 and LightGBM text/binary persistence, XGBoost staged margins/predictions plus additive contributions, LightGBM staged margins/predictions plus additive contributions, bounded exact-subset SHAP-like raw-margin attributions, transactional matched-option warm starts including validation-aware continuation, and bounded seeded XGBoost and LightGBM DART/dropout tree normalisation | Extra-trees extensions, full SHAP interaction/explanation workflows, categorical policies beyond ordered partitions, XGBoost EFB, distributed workers, differentiable routing, and complete GPU histograms |
| Gaussian processes | Exact, derivative-observation, sparse, local, SKI, structured operators, periodic/rational-quadratic/cosine/polynomial leaves, weighted binary or OVR and robust Poisson/Student-t Laplace paths, bounded weighted Bernoulli variational classification, coupled categorical likelihood and latent-Gaussian ordinal classification, fixed-state categorical temperature HVP products, dense Student-t and known-noise heteroskedastic process regression, packed sparse-GP mean/log-Cholesky ELBO products, transformed fixed-state sparse-GP Gaussian-likelihood log-noise ELBO JVP/VJP/HVP products, weighted envelope kernel products, and fixed-state binary/OVR variational kernel-log products | Full likelihood and kernel catalog, native ordinal likelihoods, batch or multitask shapes, inducing-state and remaining likelihood hyperparameter products, Student-t/heteroskedastic/robust derivative products, operator-valued derivatives, implicit products, serialization, and resident GPU solves |
| Trees and boosting | Weighted CART, deterministic seeded random-forest classification with stored inclusion state and transactional OOB probabilities/score/coverage, fixed-state accuracy permutation importance with independent NumPy replay, seeded bootstrap bagging classification, binary and multiclass SAMME and SAMME.R probability-update AdaBoost over weighted CART, exact and bounded-histogram second-order squared/logistic/Poisson/Tweedie/squared-log/Huber/quantile and `rank:pairwise` boosting, staged binary or OVR predictions, diagnostics, monotonic and interaction-group constraints, bounded ordered-gradient integer categorical partitions, transactional fitted-prefix slicing, XGBoost staged margins/predictions plus additive contributions, LightGBM staged margins/predictions plus additive contributions, versioned XGBoost schema-5 and LightGBM text persistence, transactional matched-option warm starts including validation-aware continuation, and bounded seeded XGBoost and LightGBM DART/dropout tree normalisation | Extra-trees extensions, SHAP workflows, categorical policies beyond ordered partitions, XGBoost EFB, distributed workers, differentiable routing, and complete GPU histograms |
| Gaussian processes | Exact, derivative-observation, sparse, local, SKI, structured operators, periodic/rational-quadratic/cosine/polynomial leaves, weighted binary or OVR Laplace classification, bounded weighted Bernoulli variational classification, coupled categorical and latent-Gaussian ordinal classification, dense Student-t and known-noise heteroskedastic process regression, packed sparse-GP mean/log-Cholesky ELBO products, transformed fixed-state sparse-GP Gaussian-likelihood log-noise ELBO JVP/VJP/HVP products, weighted envelope kernel products, and fixed-state binary/OVR variational kernel-log products | Full likelihood and kernel catalog, native ordinal likelihoods, batch or multitask shapes, inducing-state and remaining likelihood hyperparameter products, Student-t/heteroskedastic derivative products, operator-valued derivatives, implicit products, serialization, and resident GPU solves |
| Neural training | Dense MLPs, named sequential chains, BNN/VAE/RNN/Hamiltonian primitives, weighted binary BCE, multiclass cross-entropy, and Poisson log-rate FortOpt/L-BFGS-B objectives with exact mixed HVPs, nine trainer optimizers including unfactored and layout-aware matrix-factored Adafactor, AMSGrad, and RAdam, Lion fixed-trajectory products, schedules, portable checkpoints, fixed-trajectory hypergradients through SGD/Adam/AdamW/Adagrad/RMSprop/Adafactor/RAdam/AMSGrad including scheduled AdamW, relative-step, and parameter-scaling smooth branches, contiguous optimizer groups, and finite-feature GP/NTK last-layer kernel-ridge initialization | Complete loss and module catalog, stochastic and device hypergradients, parameter-group routing beyond the bounded CPU slice, AMP, distributed state, and resident GPU training |
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
| scikit-learn trees/ensembles | Stumps, weighted CART with deterministic NaN routing, binary and multiclass SAMME and SAMME.R AdaBoost over weighted CART, squared boosting, exact and histogram XGBoost-style second-order squared/binary-logistic/Poisson/Tweedie/squared-log/absolute lanes, bounded `rank:pairwise`, per-feature monotonic and interaction-group constraints, bounded ordered-gradient integer categorical partitions with explicit cardinality refusal, transactional XGBoost warm-start continuation, bounded LightGBM-style weighted leaf-wise regression/binary growth with staged margins/predictions, additive contributions, fitted-prefix slicing, versioned persistence, matched-option warm starts including validation-aware continuation, and seeded XGBoost/LightGBM DART/dropout tree normalisation | Random/extra forests, bagging, categorical policies beyond ordered partitions, XGBoost EFB, distributed growth, and resident GPU histograms |
| scikit-learn unsupervised | Basis maps, centered dense PCA, deterministic seeded dense k-means, validation splitters, variational primitives | Incremental/randomized/sparse/kernel PCA, ICA/NMF/TruncatedSVD, minibatch k-means/GMM/density/manifold/outlier methods, sparse/categorical preprocessing, model persistence |
| PyTorch/JAX neural core | Dense MLP, classifier, named sequential MLP chain, BNN, VAE, RNN, Hamiltonian MLP; Adam, AdamW, Adagrad, RMSprop, Adafactor, AMSGrad, RAdam, and SGD momentum/Nesterov; exact fixed full-batch SGD learning-rate/L2/momentum including classical and Nesterov velocity state, Adam/AdamW beta and decay products, RMSprop, Adafactor, AMSGrad max-state products, RAdam moment/bias/rectification products, scheduled-RAdam and scheduled-AdamW schedule-rate products, Adagrad, unfactored Adafactor, and typed schedule trajectory hypergradients | Stochastic/device optimizer hypergradients, complete loss/activation/module tree, convolution/attention/sequence/graph models, AMP, compile/fusion, distributed and device-resident train state |
| GPyTorch/GPflow | Exact, derivative-observation, sparse/local/SKI/structured GP primitives; weighted Laplace binary/OVR and bounded weighted Bernoulli variational GP classification with sorted-label OVR multiclass prediction, fixed-state kernel-log JVPs/VJPs, and weighted envelope hypergradients | Kernel/likelihood/constraint/batch-shape parity, variational categorical/count likelihoods, inducing-state products, multitask, operator-valued derivatives, implicit hypergradients, serialization and resident GPU training |
| XGBoost/LightGBM | Exact and bounded-histogram depth-limited XGBoost-style squared/logistic/Poisson/Tweedie/squared-log (RMSLE)/Huber/quantile/absolute Newton trees, weighted binary/OVR multiclass staged predictions, bounded `rank:pairwise`, margins, gain/weight/cover diagnostics, per-feature monotonic and interaction-group constraints, bounded ordered-gradient integer categorical partitions with explicit max-category refusal, serialization, fitted-prefix slicing, transactional XGBoost warm starts, seeded XGBoost DART scales, and a separately named `lightgbm_t` weighted regression/binary-logistic path with shared weighted-quantile cuts, deterministic globally best-leaf growth up to `num_leaves`, staged margins/predictions, additive contributions, fitted-prefix slicing, versioned text/binary persistence, matched-option warm starts including validation-aware continuation, and bounded seeded DART/dropout tree normalisation | Categorical policies beyond ordered partitions, XGBoost EFB, distributed training, and resident GPU histograms |
| Differentiability and search | Capability-specific JVP/VJP/HVP products, FortOpt L-BFGS-B for selected objectives, exact group-wise log-L2 mixed HVPs, and exact fixed-trajectory MLP hypergradients for SGD momentum/Nesterov, AdamW (including beta logits), RMSprop, AMSGrad (including max-state active sets), and RAdam (including rectification and epsilon) | Complete derivative matrix for every declared parameter/input/hyperparameter, stochastic/device optimizer hypergradients, implicit differentiation, and refusal rather than hidden finite differences |
| Device and performance | OpenACC/native CUDA operator lanes plus explicit device control-plane contract; kNN, dense-affine value/JVP/VJP inference, and direct RMSprop state have correctness-gated native-CUDA oracles, while complete RMSprop training, staged boosting, variational GP classification, and calibration still report CPU-only rows or typed refusals | Resident model/optimizer/batch state for every supported estimator, CPU parity, transfer/memory accounting, mixed precision, and matched PyTorch/JAX/GPyTorch/XGBoost evidence |

The inventory deliberately distinguishes an implemented algorithm from an
implemented *workflow*. For example, an XGBoost-style Newton tree without
histograms, missing-value routing, constraints, staged prediction, and a
release benchmark is a useful exact baseline but not XGBoost parity. The same
rule applies to a GP kernel without likelihood constraints, batch shapes,
train-state serialization, and derivative/hyperparameter products.

## Architecture-locked parity matrix

The next implementation wave uses the same acceptance unit for every family:
API, independent oracle, derivative capability row, typed device behavior,
checkpoint or fitted-state behavior, and a clean benchmark record. These
variants are the target inventory. An entry marked `partial` has a CPU slice
and a declared boundary. An entry marked `open` has no complete public
workflow yet.

| Family | Required variants and shared contract | Current closure state |
| --- | --- | --- |
| Classification | Binary and multinomial linear heads, OVR and OVO coupling, multilabel and classifier chains, ordinal cumulative logits, Gaussian/Bernoulli/Multinomial/Complement/Categorical Naive Bayes, LDA/QDA, linear/kernel/one-class SVM, exact and radius neighbors, calibration, CART, random/extra forests, bagging, AdaBoost/SAMME/SAMME.R, histogram boosting, XGBoost and LightGBM classifiers, Laplace/variational/categorical/ordinal GP classifiers, and neural binary/multiclass/multilabel/ordinal/calibrated heads. Every family shares sorted labels, weights, probabilities, decision values, class metadata, parameter packing, JVP/VJP/HVP boundaries, and fit/predict refusal semantics. | `partial`: most CPU families exist. OOF multiclass calibration, kernel-SVM parity, partial-fit streams, sparse views, broader GP likelihood training, and resident GPU training are open or in parallel slices. |
| Regression and GLM | Linear/ridge/lasso/elastic-net, SVR, Poisson/Gamma/Tweedie/Huber/quantile/absolute GLMs, nearest-neighbor regression, CART, random/extra forests, gradient boosting, XGBoost/LightGBM objectives, multioutput targets, robust losses, and partial-fit workflows. Objectives share weighted reductions, parameter registries, FortOpt callbacks, state dictionaries, and fixed-fit derivative boundaries. | `partial`: dense weighted linear, basis, SVR, Poisson/robust tree, and selected multioutput paths exist. Full objective catalog, sparse/online paths, and resident training remain open. |
| Gaussian processes | RBF, ARD, Matérn, periodic, locally periodic, rational-quadratic, cosine, polynomial, linear, constant, white-noise, spectral-mixture, change-point, sum/product, user, neural, graph, physics-consistent, and operator-valued kernels; value/derivative/operator observations; Gaussian, Bernoulli, categorical, multinomial, Poisson/count, Student-t, heteroskedastic, censored, ordinal, and warped likelihoods; exact, Laplace, variational, sparse, SKI, lazy, local, multitask, matrix-free, and deep-kernel inference. Each selected coordinate has value, input JVP/VJP, parameter/hyperparameter JVP/VJP/HVP, solve-state, serialization, and device rows. | `partial`: broad CPU kernel and inference coverage exists. Batch/multitask likelihoods, higher derivative operators, inducing/likelihood hyperproducts, natural gradients, state serialization, and resident GPU solves remain open. |
| Neural networks | Dense and structured module trees with buffers, aliases, tied/frozen parameters, train/eval mode, residual and normalization blocks, convolution/pooling, embeddings, attention/transformers, RNN/GRU/LSTM, temporal and graph modules, neural operators, BNN/VAE/HNN/LNN/SympNet/PINN/physics-consistent networks, and composable basis encoders. Losses include regression, softmax/log-softmax, BCE/multilabel/focal, Poisson/count, Huber/quantile, contrastive/triplet, sequence/CTC, ELBO, and physics residuals. | `partial`: dense MLP, selected recurrent/variational/physics primitives, and many optimizer trajectories exist. Full module/loss tree, mixed precision, distributed state, and resident multilayer GPU training remain open. |
| Training and search | One trainer owns deterministic loaders, weighted reduction, accumulation, clipping, validation, early stopping, callbacks, checkpoint/resume, RNG state, optimizer groups, schedules, and device plans. Adam/AdamW/SGD/RMSprop/Adagrad/Adafactor/AMSGrad/RAdam/Lion and FortOpt L-BFGS-B share one parameter registry. Hyperparameter optimization uses the same value/gradient/JVP/VJP/HVP callbacks, including optimizer, schedule, validation, basis, kernel, likelihood, and initialization coordinates. | `partial`: CPU full-batch and selected mini-batch trajectories are verified. Stochastic/device hypergradients, implicit differentiation through optima, AMP/loss scaling, distributed reduction, and full resident state are open. |
| Bases and pipelines | Polynomial, interaction-polynomial, spline, Fourier, random Fourier, radial, Chebyshev, PCA/autoencoder, GP/NNP/NTK, preprocessing, imputation, one-hot, feature names, metadata routing, residual/conditional DAGs, sparse views, cross-validation, and estimator composition share named feature and parameter offsets. | `partial`: maps, sequential/column/fan-out pipelines, a bounded residual-sum DAG, joint HVP objectives, and transactional versioned host persistence for fitted horizontal unions exist. Conditional/cyclic graphs, sparse/device graphs, cloning, estimator-wide serialization, metadata routing, and leakage-safe search integration remain open. |
| Trees and boosting | Exact and histogram growth, quantile sketches, missing-value policies, monotone and interaction constraints, categorical partitions, EFB, ranking, DART/GOSS, staged/contribution/SHAP/interaction explanations, validation/warm starts, serialization, distributed workers, and resident GPU histograms. Split topology is an explicit nonsmooth state; leaf and fixed-structure products remain differentiable. | `partial`: broad deterministic CPU XGBoost/LightGBM-style behavior and leaf products exist. Full categorical/EFB/distributed/GPU histogram and interaction-explanation parity remains open. |

This matrix is intentionally wider than the current source tree. It prevents a
single estimator or optimizer slice from being described as package parity.
Each newly checked row must cite its source revision, dependency pins, oracle,
device/refusal result, and benchmark record in the evidence register below.

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
- [x] Weighted Huber vector and multi-output regression with a shared
  parameter-registry coefficient block, optional packed L2/delta coordinates,
  exact value/gradient/JVP/VJP products, fixed-branch mixed HVPs, bounded
  FortOpt L-BFGS-B fitting, positive-mass sample weights, independent kink and
  CUDA refusals, and release benchmark evidence.
- [ ] Quantile, ordinal-GP, multilabel, multioutput partial-fit, and remaining
  robust estimators with the shared parameter registry and sample-weight
  contract.
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
- [x] Add weighted reliability-diagram points with equal-width bins, deterministic
  first-class tie handling, empty-bin zeros, and an independent metric oracle.
  `classification_reliability_diagram` is covered by
  `test_classification_metrics` and the companion reliability-diagram benchmark.
- [x] Add leakage-safe multiclass calibrated softmax cross-validation. Each
  deterministic stratified fold fits an independent softmax head, writes
  held-out logits, fits one positive temperature on those OOF logits, and
  refits the deployment head on all rows. The packed deployment state exposes
  logits, intercepts, sorted classes, and temperature with exact JVP/VJP
  products. `test_calibrated_softmax_classifier` checks weighted OOF replay,
  temperature products, malformed weight shapes, and the typed CUDA refusal;
  `fortml-bench/results/CALIBRATED_SOFTMAX_CV.md` supplies the independent
  NumPy lane. Generic estimator routing and multiclass Platt policies remain
  open; the standalone multiclass isotonic policy is covered below.
- [x] Deterministic seeded random-forest and randomized-threshold Extra-Trees
  classifiers provide aligned probabilities, arbitrary integer labels, Gini or
  entropy criteria, depth/leaf controls, positive sample weights, seeded
  reproducibility, and explicit CPU/CUDA device contracts. The random forest
  now also provides fixed-state deterministic accuracy permutation importance,
  repeat dispersion, transactional invalid-option handling, and a typed CUDA
  refusal with an independent NumPy replay benchmark. Bagging, AdaBoost,
  random patches, histogram gradient boosting, staged/warm-start APIs,
  SHAP, missing/categorical values, monotonic constraints, and differentiable
  routing remain open.
- [x] Add deterministic binary AdaBoost over weighted CART weak learners.
  `adaboost_classifier_t` retains arbitrary sorted integer labels, fits
  weighted depth-limited trees, exposes signed margins and stable probabilities,
  and stops on a perfect learner or rejects a first learner at chance. The
  independent `test_adaboost_classifier` fixture checks the one-stump
  weighted-error/alpha oracle, simplex, labels, split-routing JVP refusal, and
  typed CUDA refusal. The same fixture now covers sorted arbitrary labels,
  multiclass SAMME stage-weight/margin/softmax oracles, deterministic seed
  reproducibility, transactional invalid-fit preservation, and multiclass
  derivative refusals. The multiclass `algorithm=ADABOOST_ALGORITHM_SAMME_R`
  policy now trains from clipped centred log-probability updates, exposes
  geometric-ensemble probabilities and unit stage weights, and preserves
  transactional fits plus typed split-routing/CUDA refusals; the independent
  `test_adaboost_samme_r` and benchmark lane cover the probability oracle.
  Staged explanations and resident GPU boosting remain open.
- [x] Binary and one-vs-rest XGBoost-style staged predictions, cumulative
  margins, gain/weight/cover feature-importance diagnostics, and fixed-tree
  input JVP/VJP products with split-boundary refusals.
- [x] Add weighted validation to `xgboost_multiclass_t`. The adapter validates
  arbitrary integer validation labels and positive weights, scores normalized
  multiclass log-loss after every common OVR stage, exposes requested/best
  iteration, best loss, and early-stop metadata, restores a common best prefix
  transactionally, persists those diagnostics in schema 2, and leaves fitted
  state unchanged on malformed validation input. The independent
  `test_xgboost_multiclass` replay covers weighted loss, best-prefix staged
  probabilities, metadata, and unknown-label refusal; CUDA remains a typed
  resident-tree refusal. See `docs/XGBOOST_MULTICLASS_VALIDATION.md` and the
  release benchmark lane.
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
  partitions, EFB, distributed workers, warm-start LightGBM state,
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
- [x] Add bounded dense probabilistic-process reference paths. The
  `student_t_process_t` contract keeps the GP mean while scaling predictive
  covariance by the observed Mahalanobis distance and is checked against the
  large-`nu` Gaussian limit, data-dependent variance contrast, and typed
  `nu<=2` refusal in `test_student_t_process` and
  `fortml-bench/results/STUDENT_T_PROCESS.md`. The
  `heteroskedastic_gp_t` contract accepts known positive per-row noise,
  interpolates log-noise with a second kernel, reduces exactly to an ordinary
  GP for constant noise, and has the independent
  `fortml-bench/results/HETEROSKEDASTIC_GP.md` lane. Derivative, variational,
  joint-noise-inference, and resident-GPU products remain open.
- [x] Add a bounded Laplace robust/count GP path in `robust_gp_t`. Poisson
  observations use a positive latent log-rate, while Student-t observations
  use curvature flooring in the far tail so outliers cannot contribute negative
  confidence. The independent `test_robust_gp` gate checks mode stationarity,
  positive rates, outlier resistance, convergence, and typed refusal boundaries;
  exact evidence, derivative products, scalable inference, and resident CUDA
  remain open.
- [x] Add a locally-periodic kernel with a four-coordinate logarithmic
  parameter registry, analytic value/input/parameter products, exact-GP
  integration, coincident-point limits, an independent oracle, and typed
  static-operator/CUDA refusal.
- [x] Extend derivative-GP query-input JVP/VJP products to the local-periodic
  leaf. The radial third-input contraction is analytic for value and every
  first-derivative query component, including coincident-point limits; the
  independent `test_derivative_gp_local_periodic` and benchmark lane compare
  posterior mean/variance products against directional finite differences.
- [x] Add an analytic change-point kernel that gates a left child and a right
  child by a smooth logistic transition. Value, input JVP, mixed Hessian,
  packed parameter JVP/VJP/HVP products, and exact-GP integration are covered
  by `test_kernel_change_point` and the independent NumPy lane in
  `fortml-bench/results/CHANGE_POINT_GP.md`; static-operator and resident CUDA
  requests return typed refusals. Neural-network, graph, string, and
  operator-valued kernels remain open.
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
  hyperparameter products, natural gradients, and resident GPU inference stay
  open. The separate coupled categorical variational slice is documented below.
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
- [x] Add the fixed-state Gaussian-likelihood noise block to `sparse_gp_t`.
  `likelihood_parameters()`/`hyperparameters()` expose the transformed packed
  coordinate `[log(noise_variance)]`, while transactional setters reject
  malformed, non-finite, and overflowing values without mutating the model.
  Analytic ELBO JVP/VJP/HVP products and CPU/CUDA device boundaries are covered
  by the independent `test_sparse_gp_likelihood_noise` finite-difference,
  adjoint, state-preservation, and typed-refusal oracle. The variational state,
  kernel, and inducing locations remain fixed by contract.
- [x] Add fixed-state variational-classification kernel-log-parameter JVP/VJP
  products for binary and sorted-label OVR models. Latent margins and
  simplex-normalized probabilities include the inducing solve, cross-kernel,
  predictive variance, KL, and quotient-rule adjoint terms. The independent
  `test_gp_variational_kernel_products` fixture covers central differences,
  duality, simplex preservation, and typed CUDA refusal; inducing-state,
  likelihood, natural-gradient, and resident-GPU products remain open.
- [x] Add the bounded coupled categorical variational-GP slice
  `gp_variational_categorical_classification_t`. It owns one sorted-label
  inducing posterior per class and evaluates a shared categorical ELBO with a
  variance-corrected softmax likelihood and analytic inducing KL terms. The
  FortOpt L-BFGS-B `fit` path, packed ELBO gradients/JVPs, probability
  parameter/input JVPs and VJPs, deterministic tie policy, and explicit CUDA
  refusals are covered by `test_gp_variational_categorical_classification` and
  `docs/GP_VARIATIONAL_CATEGORICAL.md`. HVP, kernel-hyperparameter, natural
  gradient, and resident-GPU products remain open.
- [x] Add the coupled categorical likelihood hyperparameter contract. The
  positive softmax temperature is stored as a separate log coordinate, with
  exact probability and ELBO JVP/VJP products, fixed-state probability and
  ELBO HVP products, weighted likelihood scaling, transactional FortOpt
  L-BFGS-B likelihood-only fitting, and explicit CUDA JVP/VJP/HVP refusals.
  The independent finite-difference/adjoint/directional-Hessian oracle is
  `test_gp_variational_categorical_likelihood`; the release lane is
  `fortml-bench/results/GP_CATEGORICAL_LIKELIHOOD.md`. Inducing-state,
  natural-gradient, and resident-GPU products remain open.
- [x] Add `gp_ordinal_classification_t`, a latent-Gaussian ordered GP baseline.
  Sorted integer labels map to rank targets for a zero-mean Gaussian GP;
  fixed mid-rank cut points convert predictive mean/variance to adjacent
  normal-CDF probabilities. Packed kernel/noise parameter products and
  analytic input JVP/VJP products include the Cholesky solve and uncertainty
  chain. `test_gp_ordinal_classification` independently checks simplex rows,
  class order, parameter/input finite differences, JVP/VJP duality, and typed
  CUDA refusals; see `docs/GP_ORDINAL_CLASSIFICATION.md`. The exact evidence
  surface now also exposes kernel/log-noise hyperparameter gradients, scalar
  JVPs, directional HVPs, and a bounded FortOpt L-BFGS-B adapter with
  transactional failure restoration. The independent
  `test_gp_ordinal_classification_hyperparameters` oracle and release lane
  cover finite differences, HVPs, convergence, and typed CUDA refusals.
  Native cumulative likelihoods, optimized cut points, and resident GPU
  inference remain open.
- [x] Add `gp_multilabel_classification_t`, an independent binary Laplace-GP
  wrapper for dense indicator targets. Each label owns a weighted logistic or
  probit head; probabilities remain independently calibrated rather than being
  simplex-normalized. Packed per-label kernel parameters, latent/probability
  input and fixed-state parameter JVP/VJPs, envelope hyperparameter gradients,
  configurable thresholds, CPU dispatch, and typed CUDA refusals are covered
  by `test_gp_multilabel_classification` and the independent NumPy release
  lane `fortml-bench/results/GP_MULTILABEL.md`.
- [x] Add the implicit-mode binary Laplace-GP kernel hyperparameter HVP.
  `gp_classification_t%hyperparameter_hvp` differentiates the converged mode
  tangent through the posterior factorization and contracts analytic kernel
  `parameter_hvp`/`parameter_vjp` products for the full fitted envelope
  gradient. The selected-CPU device method is explicit and CUDA returns
  `FORTNUM_NOT_IMPLEMENTED`. `test_gp_classification_hvp` independently
  refits logistic and probit probes, checks transactional setter refusal, and
  checks the device boundary; see
  `docs/GP_CLASSIFICATION_HYPERPARAMETER_HVP.md` and the companion benchmark.
- [x] Add `second_derivative_gp_t` as a bounded exact scalar 1-D RBF/Matérn-5/2
  reference for mixed value/first/second-derivative observations and
  predictions. The explicit order vector uses `0:2` for both kernels and `3`
  for RBF; RBF covariance blocks reach total derivative order six and query
  input JVP/VJP products use order seven, while Matérn-5/2 retains order-four
  covariance and order-five query products. Dense latent joint covariance is
  available on CPU. `set_parameters`, likelihood value/JVP/VJP, analytic RBF
  hyperparameter gradients, and RBF likelihood HVPs differentiate the fitted
  Cholesky state transactionally. The independent
  `test_second_derivative_gp` and `test_second_derivative_gp_rbf_order3`
  oracles check both kernels, posterior moments, mixed covariance,
  central-difference input products, likelihood gradient/HVP finite
  differences, adjoint duality, and typed CUDA/coincidence/non-RBF/order
  refusal boundaries; see `docs/SECOND_DERIVATIVE_GP.md` and the release
  benchmarks `fortml-bench/results/SECOND_DERIVATIVE_GP.md` and
  `fortml-bench/results/SECOND_DERIVATIVE_GP_RBF_ORDER3.md`. Higher orders,
  arbitrary kernels/dimensions, operator-valued outputs, Matérn parameter
  jets, and resident derivative covariance/factorization remain open.
- [x] Add the bounded deep-kernel GP composition `deep_kernel_gp_t`. An MLP
  feature map feeds an exact dense GP base kernel on feature space, with
  identity-map reduction, exposed transforms/posterior, exact feature and
  weight likelihood gradients, and malformed/unfitted refusals checked by
  `test_deep_kernel_gp`. Joint FortOpt feature/kernel training, KISS-GP/SKI,
  derivative-observation deep kernels, and resident CUDA execution remain open.
- [ ] Generalize derivative observations to every supported smooth kernel,
  mixed orders beyond two, vector fields, Hessian observations, registered
  linear operators, operator-valued outputs, analytic higher-order query
  products, and covariance products over value/derivative blocks.
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
- [x] Add checked `mlp_t%initialize_linear`/`set_linear_parameters` seams and
  `mlp_t%initialize_from_pca` for finite two-layer affine or PCA-seeded states.
  The PCA path maps the centered (optionally whitened) projection and inverse
  into linear MLP weights and biases, derives the exact autoencoder topology,
  validates all metadata before mutation, and has independent reconstruction,
  transaction, and refusal oracles plus a companion benchmark. These are
  finite linear/PCA optima, not NNGP, NTK, GP-posterior, physics-consistent,
  symplectic, or Hamiltonian equivalences; those structure-aware mappings remain
  open in WP9d.
- [x] Extend dense MLP value/JVP/VJP/HVP products with linear, `tanh`, ReLU,
  tanh-approximate GELU, SiLU, ELU, softplus, fixed-slope leaky ReLU, stable
  sigmoid, and Mish activations. Independent value and central-difference
  first/second-product tests cover every activation; the complete CUDA path
  remains explicitly refused until resident activation and dense-gradient
  kernels are linked.
- [x] Add stable softmax/log-softmax value, JVP, VJP, and HVP products,
  weighted mean/sum softmax cross-entropy, and focal binary BCE value/JVP/VJP/
  HVP products with independent NumPy checks and a typed CUDA boundary. The
  multiclass focal-softmax value/JVP/VJP/HVP family now adds positive
  class-weight factors, stable true-class underflow refusal, and aliases; the
  multiclass MLP fit and FortOpt objectives select the same products through
  `focal_gamma`. See `docs/NEURAL_LOSS_PRODUCTS.md` and the `fortml-bench`
  neural-loss lane.
- [ ] Complete the remaining activation and loss catalog: multilabel focal
  variants, Poisson count/dispersion variants, contrastive,
  triplet, CTC, and physics residual losses, each with explicit derivative and
  refusal contracts.
- [x] Add a weighted one-output Poisson log-rate objective with exact value,
  JVP, VJP, HVP, optional L2 coordinate, and a bounded FortOpt L-BFGS-B
  adapter. `test_mlp_poisson_objective` checks finite differences, adjoint
  scaling, convergence, and typed CUDA refusal; the API contract is in
  `docs/MLP_POISSON.md`.
- [x] Production RMSprop with centered/uncentered running statistics, optional
  momentum, exact checkpoint/resume state, and an independent recurrence oracle.
- [x] Production AMSGrad with bias-corrected first/second moments, an
  elementwise max-second-moment state, exact in-memory and formatted
  checkpoint/resume, and an independent NumPy recurrence/MLP oracle. The
  trainer is CPU-only; resident CUDA execution remains open.
- [x] Production RAdam with bias-corrected first/second moments, the validated
  `rho_t` threshold and rectification factor, exact in-memory and formatted
  checkpoint/resume, and independent recurrence/device-boundary oracles. The
  trainer and flat state are CPU-only; `fortml_mlp_radam_hypergradient` now
  propagates exact fixed full-batch value/JVP/VJP products through moments,
  bias correction, and rectification, with a bounded FortOpt L-BFGS-B adapter
  and independent central-difference/adjoint oracles. The `rho_t = 4` and
  zero-square-root boundaries return typed nonsmooth refusals; resident CUDA
  state remains open.
- [x] Add the exact scheduled RAdam trajectory hypergradient objective over
  `[log(base_rate), log(l2), logit(beta1), logit(beta2), log(epsilon),
  logit(min_rate_fraction), logit(decay_factor)]`. Constant, cosine,
  warmup-cosine, and exponential-decay schedules propagate exact rate and
  RAdam moment/bias/rectification sensitivities through value/gradient, JVP,
  scalar VJP, and the FortOpt L-BFGS-B adapter. Independent central-difference
  fixtures cover cosine minimum-rate and exponential decay-factor branches;
  rho/zero-root, outer-HVP, and CUDA requests return typed refusals. The
  release gate is `fortml-bench/results/MLP_RADAM_SCHEDULE_HYPERGRADIENT.md`.
- [x] Add the stateless metric-aware plateau schedule. The typed schedule now
  takes explicit metric, best-value, bad-observation, and reduction-count
  inputs, implements deterministic patience/min-delta/factor transitions for
  minimizing and maximizing metrics, and returns the next state without a
  hidden cursor. Base-rate and factor derivatives are exact on each active
  branch, while metric, best-value, min-delta, and integer decision products
  are documented zeros. Formatted checkpoint schema 11 validates and round-trips
  the plateau fields. `mlp_train` now owns this state at epoch boundaries,
  selects validation loss when present (training loss otherwise), and reproduces
  the schedule through split checkpoint/resume. Independent transition,
  finite-difference, trainer, persistence, resume, and refusal oracles are in
  `test_mlp_plateau_schedule` and `fortml-bench/results/MLP_PLATEAU_SCHEDULE.md`.
- [ ] Remaining production optimizer gaps: cosine/one-cycle/warmup schedule
  derivatives through optimizer groups, validation policy, and device state.
  The deterministic mini-batch SGD trajectory objective now records a private
  batch cursor (including seeded epoch shuffles), exposes exact learning-rate
  and L2 hypergradients through validation MSE, and is consumable by FortOpt
  L-BFGS-B. The bounded CPU optimizer-group slice validates contiguous ranges,
  names, multipliers, and checkpoint metadata and exposes a public outer HVP
  entry point with a zero-valued `FORTNUM_NOT_IMPLEMENTED` refusal until
  third network derivatives exist; malformed HVP shapes return
  `FORTNUM_DOMAIN_ERROR` and never finite-difference. Fixed global-norm
  clipping now follows the production trainer and propagates derivatives on a
  fixed active set; the clipping boundary is a typed refusal. General
  stochastic-loader, clipping-coordinate, validation-policy, migration, and
  resident-device products remain open.
- [x] Add a weighted validation-measure contract to the fixed full-batch SGD
  momentum/Nesterov trajectory objective. `validation_weight(:)` is copied and
  validated transactionally (finite, non-negative, positive support), and the
  weighted mean is propagated exactly through value/gradient, JVP, VJP, and
  FortOpt L-BFGS-B products. Uniform affine validation keeps the existing exact
  outer HVP; non-uniform HVP requests return an explicit
  `FORTNUM_NOT_IMPLEMENTED` status until residual-weighted second contractions
  are independently certified. `test_mlp_weighted_validation_hypergradient`
  provides central-difference, adjoint, uniform-HVP, malformed-weight, and
  CUDA refusal oracles; see `docs/MLP_WEIGHTED_VALIDATION_HYPERGRADIENT.md`.
- [x] Close the first exact outer-HVP slice for scheduled MLP trajectories.
  A single affine dense layer with a constant typed learning-rate schedule now
  propagates mixed second tangents through the fixed full-batch recurrence and
  exposes an exact FortOpt-consumable `hvp` over log base rate and log L2.
  Inactive schedule coordinates are exact zeros; nonlinear networks and
  nonconstant schedules retain typed third-derivative/rate-second-product
  refusals, and CUDA remains an explicit resident-kernel boundary. Central-FD,
  Hessian-symmetry, FortOpt-callback, and typed-boundary tests plus the
  independent NumPy release gate live in
  `docs/MLP_CONSTANT_SCHEDULE_HVP.md` and
  `fortml-bench/results/MLP_CONSTANT_SCHEDULE_HVP.md`.
- [x] Add production Lion to `mlp_train`. The stateful CPU trainer now uses
  the beta1 interpolation and beta2 momentum recurrence, decoupled weight
  decay, clipping, schedules, EMA, validation, optimizer groups, and exact
  in-memory/text checkpoint resume. `test_mlp_lion_training` independently
  recomputes the linear-model gradient and checks the parameter, momentum,
  EMA, and split-resume trajectories. Resident CUDA Lion state and
  differentiable sign-branch products remain typed follow-up capabilities;
  see `docs/MLP_LION_TRAINING.md`.
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
- [x] Add the layout-aware `fortml_adafactor_factored` recurrence and wire it
  into `mlp_train` through `options%adafactor_factored`. Dense weight blocks
  use row/column state and bias or singleton blocks use the vector fallback;
  `test_mlp_adafactor_factored` and the NumPy benchmark oracle cover both
  recurrences and the explicit CUDA boundary. Ragged factor-state checkpoint
  migration and resident CUDA remain typed follow-up gates.
- [x] Add `mlp_loss_scale_state_t` with deterministic finite-update growth,
  overflow backoff, skipped-update accounting, and static-policy/dynamic-state
  validation. The FP64 CPU trainer checks scaled-gradient overflow and captures
  the state in schema-11 checkpoints; an independent recurrence/persistence
  oracle and typed lower-precision boundary are released. Master-weight FP32,
  FP16, BF16, and resident CUDA training remain open.
- [ ] Exact fixed-trajectory and implicit hypergradients through all supported
  optimizers, schedules, batch cursors, clipping, weight decay, validation,
  early stopping, and optimizer state. Fixed full-batch SGD/AdamW/Adagrad/
  RMSprop/RAdam (including typed scheduled RAdam and scheduled AdamW) and deterministic mini-batch
  SGD now have analytic trajectory products. Every unsupported stochastic or
  device path must return a typed refusal.
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
  difference checks where a trusted analytic oracle is unavailable. The
  canonical row schema is documented in
  `docs/DERIVATIVE_CAPABILITY_MATRIX.md`; the implementation ledger remains
  open until every declared public surface has a row.
- [ ] Implicit differentiation through linear solves, fixed points, Laplace
  modes, variational optima, constrained tree policies, and optimizer fixed
  points, with FortOpt L-BFGS-B consuming the same parameter registry.
- [x] Add interval-routed `conditional_basis_pipeline_t` over selected-column
  basis stages, with transactional configuration/schema updates, stable branch
  offsets and route metadata, exact CPU value/JVP/VJP/HVP products, endpoint
  derivative refusals, independent finite-difference/adjoint tests, and an
  output-preserving typed CUDA boundary.
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
- [x] Add the orthogonal Chebyshev first-kind basis family with a stable
  three-term recurrence, optional intercept, exact CPU JVP/VJP/HVP products,
  and a typed CUDA capability boundary. The independent oracle and benchmark
  are recorded in `test_basis_chebyshev` and
  `../fortml-bench/results/CHEBYSHEV_BASIS.md`.
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
while mixed-observation HVPs use exact generated/closed-form products for
RBF, Matérn 3/2/5/2, periodic, linear, constant, polynomial, and supported
compositions. Other leaves retain documented typed refusals until their
input-parameter second products are complete. The full
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
| Classification | Partial | `fortml_logistic_regression` and `fortml_softmax_regression` provide binary and multinomial integer-label fitting with sample-weighted reductions, `fortml_ovr_logistic_classifier` adds deterministic one-vs-rest binary logistic fits with normalized probabilities and parameter products, `fortml_ovo_logistic_classifier` adds deterministic one-vs-one pairwise-vote probabilities and input/parameter products, `fortml_multilabel_logistic_classifier` adds independent dense indicator heads, `fortml_ordinal_logistic_classifier` adds weighted cumulative-logit fitting with ordered cut points and input/packed-parameter JVP/VJPs, `fortml_gaussian_naive_bayes` adds weighted Gaussian class moments and input/parameter probability products, `fortml_bernoulli_naive_bayes` adds weighted relaxed-[0,1] Bernoulli features, positive smoothing, packed state, and input/parameter probability products, `fortml_multinomial_naive_bayes` adds weighted relaxed nonnegative counts, token-mass smoothing, packed state, and input/parameter probability products, `fortml_complement_naive_bayes` adds weighted complement distributions, optional weight normalization, packed state, and input/parameter probability products, `fortml_lda_classifier`/`fortml_qda_classifier` add weighted Cholesky Gaussian discriminants with input/parameter products, `fortml_probability_calibration` adds Platt sigmoid and weighted PAVA isotonic calibration with score/parameter products and active-set refusals, `fortml_calibrated_logistic_classifier` adds stratified out-of-fold binary calibration and fold log-loss diagnostics, `fortml_knn_classifier` adds deterministic dense uniform or inverse-distance neighbors with explicit discrete derivative refusals, `fortml_logistic_training` adds an exact weighted logistic objective with parameter/L2 gradients and HVPs plus bounded FortOpt L-BFGS-B, `fortml_linear_svm_classifier` and `fortml_rbf_svm_classifier` add weighted hinge/squared-hinge binary margins with fixed-state probability and product contracts, `fortml_cart_classifier` adds deterministic weighted Gini/entropy trees and leaf probabilities, `fortml_random_forest_classifier` adds seeded bootstrap CART probabilities, `fortml_extra_trees_classifier` adds seeded randomized-threshold/feature ensembles, `fortml_mlp_classifier` adds deterministic multiclass logits training with Adam and sample/class weights, `fortml_mlp_ordinal_classifier` adds a scalar-score neural cumulative-logit head with ordered cut points and deterministic FortOpt L-BFGS-B training, `fortml_gp_classification` adds binary Laplace logistic/probit inference plus an exact mode-envelope kernel gradient and query-feature JVP/VJP products, `fortml_gp_multiclass_classification` adds packed one-vs-rest envelope gradients, latent margins, and query-feature JVP/VJP products, and shared metrics cover accuracy, top-k, balanced accuracy, confusion, precision/recall/F1, Brier, binary Matthews, weighted accuracy, log loss, and expected/maximum calibration error. | Binary and multiclass linear, multilabel, ordinal, tree, neural, GP, and boosted-tree classifiers share label, probability, weighting, metric, and calibration conventions. |

| Estimator contracts, pipelines, and bases | Partial | `basis_map_t`, horizontal and sequential basis pipelines, transactional `basis_input_schema_t` names/validation, fitted standard/min-max scalers with input JVPs, `basis_linear_regression_t`, joint `basis_pipeline_training_objective_t`, weighted multi-output `ridge_regression_t` and `elastic_net_regression_t`, weighted `linear_svr_regression_t`, row-oriented sample conventions, status objects, and the parameter registry are public. | Fitted transformers and estimators compose without data leakage, expose routed parameters, and run through cross-validation. |
| Tree boosting | Partial | `decision_stump_t`, weighted depth-limited `cart_regressor_t` and `cart_classifier_t`, squared-loss `gradient_boosting_regressor_t`, `xgboost_t`, `xgboost_multiclass_t`, and separately named `lightgbm_t` provide deterministic exhaustive, bounded-histogram, and best-first leaf-wise split products. The CART lanes have weighted squared-error or Gini/entropy criteria, depth and leaf constraints, fixed feature/threshold tie ordering, piecewise input JVP/refusal for regression, and finite-only probability/prediction paths for classification. The XGBoost-style lane has exact and weighted-histogram squared/logistic/Poisson/Huber/quantile/absolute and bounded pairwise-ranking gradients, Hessians, regularized gains, recursive Newton leaves, per-feature monotonic bounds, tree-depth/node diagnostics, binary and one-vs-rest multiclass probabilities, staged predictions/margins, and gain/weight/cover feature importance. The LightGBM-style lane supports weighted regression/binary logistic objectives, shared weighted-quantile cuts, deterministic global best-leaf growth up to `num_leaves`, validation monitoring with patience/min-delta, best-round metadata, restore-best or retain-all ensembles, staged/contribution/slice products, versioned text persistence, matched-option warm starts, bounded seeded DART/dropout tree normalisation, and typed split-boundary/CUDA refusals. | Regression and classification trees support validation-based stopping, categorical objectives, interaction constraints, deeper growth, and model persistence. |
| Current bounded additions | Partial | The current source and evidence also include seeded CART bagging with multiclass probability alignment, random-forest stored bootstrap inclusion plus transactional OOB probabilities/score/coverage and explicit insufficient-coverage/CUDA refusals, fixed-state random-forest accuracy permutation importance with deterministic repeat dispersion and an independent NumPy replay, weighted binary/OVR Laplace-GP fits and FortOpt envelope hyperparameter adapters, binary Laplace-GP log-probability value and input/parameter JVP/VJP products with transactional fixed-state parameter updates, fixed-state sparse-GP Gaussian-likelihood log-noise ELBO JVP/VJP/HVP products with transactional setters and independent finite-difference/adjoint/device oracles, fixed-SGD optimizer-group multiplier trajectory JVP/VJP products with fixed global-norm clipping plus an explicit outer-HVP and clipping-boundary refusal, batched multi-output GP posterior means with query JVP/VJP products, exact multi-output prior-covariance parameter JVP/VJP products over kernel/log-noise/coregionalization coordinates, typed CPU/CUDA dispatch for fitted column basis pipelines, CPU RAdam training with exact format-9/text-schema-9 checkpoint resume, factored Adafactor row/column/vector checkpoint migration, fixed full-batch RAdam and AMSGrad trajectory value/JVP/VJP products with bounded FortOpt adapters and typed rho/max-active-set/CUDA refusals, multiclass XGBoost text persistence, fixed-structure XGBoost/LightGBM leaf-coordinate JVP/VJP products, registry-backed weighted Huber regression with optional L2/delta outer coordinates and fixed-branch mixed HVPs, locally-periodic derivative-GP parameter plus query-input JVP/VJP products with coincidence-safe third-input rules, metric-aware plateau schedule transitions and persisted plateau trainer diagnostics with deterministic split/resume recurrence, spectral-mixture mixed-observation parameter HVPs with exact four-jet CPU products, and periodic mixed-observation parameter HVPs with coincidence-safe fourth-input products. | SHAP workflows, coupled GP likelihoods, stochastic/device optimizer groups, parameter products over batched GP queries, broader stochastic/device optimizer hypergradients, and resident GPU execution remain open. |
| Training infrastructure | Partial | Model-specific gradients, exact MSE+L2 neural HVPs including the L2 mixed hyperparameter block, weighted multiclass MLP cross-entropy value/JVP/VJP/HVP products with bounded FortOpt L-BFGS-B, `mlp_training_objective_t` scalar JVP/VJP products (including the optional optimized L2 coordinate), named group-wise log-L2 objective/HVP products, differentiable Huber/quantile loss products, a joint basis-pipeline value/gradient/JVP/VJP/HVP objective, `fortopt_adam` and FortOpt SGD momentum/Nesterov integration, AdamW with decoupled decay, coupled-L2 Adam with exact fixed full-batch trajectory hypergradients, scheduled AdamW/RAdam and Adagrad trajectory products, Adagrad with an explicit accumulated-square state, RMSprop with centered/uncentered statistics and optional momentum, CPU AMSGrad with bias-corrected max-second-moment state and exact checkpoint/resume, CPU RAdam with validated rho-threshold rectification, exact checkpoint/resume, and exact fixed-trajectory value/JVP/VJP products, deterministic seeded batch cursors, per-update callback and typed learning-rate schedules, norm clipping, sample-weighted gradient accumulation, validation/early stopping, resumable optimizer state, generic versioned portable trainer checkpoints, versioned portable MLP checkpoint files, fixed-trajectory full-batch SGD/Adam/AdamW/RMSprop/Adagrad/RAdam/AMSGrad and deterministic mini-batch SGD/coupled-L2 Adam hypergradients, unfactored Adafactor rate/accumulator/validation hypergradients including smooth relative-step and parameter-scaling branches, scheduled-Adagrad rate/accumulator/validation hypergradients with FortOpt L-BFGS-B, fixed-SGD optimizer-group derivatives through a fixed global clipping active set, explicit optimizer-group outer-HVP and clipping-boundary refusal contracts, typed precision refusal contracts, natural-gradient seams, and seeded variational draws exist. | One trainer owns batches, optimizer state, schedules, clipping, validation, early stopping, callbacks, and resumable state for every model with a completed trainer adapter; stochastic, mixed-precision, and device-resident optimizer hypergradients remain open. |
| GP derivatives and hyperparameters | Partial | Exact GP likelihood and prediction products include parameter gradients and HVPs. Mixed value and first-derivative observations can be fitted and predicted; value-only HVPs use the analytic kernel-HVP/differentiated-solve path, while RBF/ARD-RBF/Matérn 3/2/Matérn 5/2/linear/constant/polynomial/periodic/spectral-mixture and their supported sum/product mixed-observation HVPs now use analytic covariance-block second products. Polynomial HVPs cover all four log kernel coordinates, including the degree-one limit; Matérn 3/2/5/2 HVPs cover both logarithmic kernel coordinates with finite coincident limits; periodic HVPs cover all three logarithmic kernel coordinates plus log noise with coincidence-safe fourth-input products; spectral-mixture HVPs carry exact four-jets through separable factors. These slices have independent likelihood finite-difference oracles. Correlated multi-output GPs now expose packed coregionalization/noise/kernel parameters plus posterior-mean query and parameter JVP/VJP products through a differentiated Cholesky solve, with independent finite-difference and adjoint oracles; independent query batches also have shape-checked means and input JVP/VJPs with explicit CUDA refusal. Binary and OVR Laplace classifiers now expose fixed-state kernel-parameter JVP/VJP products for latent and normalized observed predictions. Other mixed leaves return an explicit `FORTNUM_NOT_IMPLEMENTED` refusal rather than finite-differencing. Scalar likelihood VJPs include the packed noise block. Dense joint posterior covariance now also exposes exact parameter JVP/VJP products through the same solve, with independent finite-difference and adjoint oracles. | Exact, derivative, multi-output, sparse, and matrix-free GP families expose documented trainable parameters, scalar objectives, parameter gradients, and train-state adapters. |
| Serialization and distributed execution | Partial | `fortml_mlp_checkpoint` provides a versioned compiler-independent formatted-text representation with schema magic/version, exact optimizer/iterator/history state, validated temporary loading, and malformed/truncated/extra-record refusals. Other model/pipeline files and distributed execution remain open. | Versioned model and trainer files round-trip across supported compilers, and MPI training or inference agrees with a one-rank oracle. |
| Benchmark coverage | Partial | Correctness-gated model and GP applications feed release harnesses in `../fortml-bench`; current release lanes include Bernoulli/Multinomial/ComplementNB, integer one-hot encoding, weighted ridge and elastic-net derivative products, weighted Huber value/gradient and FortOpt convergence (`HUBER_REGRESSION.md`), scalar and multi-output radius neighbors, seeded CART bagging, random-forest OOB classification and fixed-state permutation importance, binary and multiclass SAMME AdaBoost, CPU RAdam flat/MLP training with checkpoint replay, fixed full-batch RAdam and AMSGrad trajectory value/JVP/VJP/FortOpt products (`MLP_RADAM_HYPERGRADIENT.md`, `AMSGRAD_HYPERGRADIENT.md`), metric-aware plateau scheduling and trainer diagnostics, binary Laplace-GP log-probability products, fixed-structure XGBoost/LightGBM leaf products, PCA-seeded linear MLP initialization, ordered-gradient categorical XGBoost partitions, OVR/OVO/multilabel/ordinal/RBF-SVM classification, multilabel precision/recall/F1/Jaccard/Hamming and ROC/PR-AUC metrics, temperature/sigmoid/isotonic probability calibration, weighted Laplace and variational GP classification, batched multi-output GP query products, typed column-pipeline device dispatch, AMSGrad recurrence/MLP training, the shared objective trainer and portable checkpoint, weighted binary and multiclass MLP objectives with bounded L-BFGS-B, optimizer-group trajectory hypergradients with fixed clipping-active-set products and explicit outer-HVP/boundary refusal rows, typed MLP precision capability rows, Lion hypergradients, transformed softmax log-L2 products, additive XGBoost contributions, pairwise ranking, and interaction constraints, bounded LightGBM leaf-wise boosting with validation early stopping, XGBoost warm-start continuation, MLP activation products, MLP SGD/Nesterov/Adam/AdamW, affine one-layer SGD momentum outer HVP products (`MLP_SGD_MOMENTUM_HYPERGRADIENT_HVP.md`), coupled-Adam and other fixed-trajectory hypergradients, typed schedules including one-cycle, fixed-trajectory and resident-state Adagrad, scheduled-Adagrad hypergradients, differentiable imputation, Chebyshev basis recurrence/JVP/VJP/HVP products, transactional pipeline input-schema validation and stable input-name routing, basis/pipeline, exact/approximate and correlated multi-output GP products including weighted OVR variational and packed Laplace multiclass prediction products, analytic GP likelihood products, derivative-GP spectral-mixture query and mixed-observation HVP products, polynomial mixed-observation GP HVPs, exact and weighted-histogram squared/logistic/Poisson boosting, monotonic-constraint query grids, general Hamiltonian MLP products, canonical symplectic-form residual products, generic grid/L-BFGS-B search, resident CUDA dense-affine value/JVP/VJP and MSE-update device-contract gates, and resident forest prediction. | Every completed parity package has a pinned external oracle, release timings, memory measurements, provenance, raw data, and a maintained report. |

The ledger now also records contiguous CPU MLP optimizer groups with checkpoint
metadata and fixed-SGD multiplier hypergradients, weighted Laplace-GP
kernel-log envelope products, fixed-state sparse-GP and
variational-classification kernel-log-parameter products, the seeded CART
bagging classifier, the classifier-chain logistic adapter, XGBoost interaction
groups, the weighted Poisson MLP objective, the change-point GP kernel, the
multiclass calibrated-softmax OOF estimator, and layout-aware factored
Adafactor. These slices are production evidence for
their stated contracts. They do not close the broader GPU, stochastic,
distributed, multitask, or architecture-family gates listed above.

The physics ledger now also records `fortml_symplectic` canonical-form
residuals and exact first-order products over map Jacobians. The independent
Verlet oracle, release application, and typed CUDA boundary are pinned in the
symplectic residual benchmark lane. Lagrangian, Poisson, implicit general-
Hamiltonian, and resident GPU structure paths remain open.

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

The calibration-aware binary cross-validation slice adds
`calibrated_logistic_classifier_t`. Stratified folds produce one held-out
logistic margin per row before the binary sigmoid, temperature, or isotonic
map is fitted. The deployment model is refit on all rows, and the state keeps
both out-of-fold log-loss values. `test_calibrated_logistic_classifier` checks
the simplex, arbitrary sorted labels, smooth JVP/VJP products, isotonic
active-set refusal, and explicit CUDA refusal. The companion benchmark is
`results/calibrated_logistic_cv.csv`; multiclass calibration CV and generic
estimator callbacks remain open.

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
- [x] Add fixed-input multiclass MLP probability parameter products. The
  `predict_proba_parameter_jvp`/`predict_proba_parameter_vjp` pair differentiates
  the complete packed network state with a zero input tangent, while the
  corresponding device wrappers execute CPU explicitly and return typed CUDA
  refusals. The independent `test_mlp_classifier_parameter_products` gate
  checks central differences, VJP/JVP duality, and both device boundaries; see
  `docs/MLP_CLASSIFIER_PARAMETER_PRODUCTS.md` and the companion benchmark.
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
  `test_gp_variational_multiclass_classification`; quadrature, calibrated
  uncertainty, and resident GPU inference remain open. Coupled categorical
  inference is covered by `gp_variational_categorical_classification_t` below.
- [x] Add a shared-head multilabel neural classifier with indicator validation,
  configurable per-label thresholds, deterministic full-batch Adam, exact
  logits/probability input and parameter JVP/VJP products, BCE parameter HVPs,
  and typed CUDA refusal. Binary and ordinal heads have separate contracts;
  calibrated neural heads are covered by the next checked slice.
- [x] Add `mlp_multilabel_training_objective_t` with copied sample-by-label
  weights, direct or positive log-L2 coordinates, exact packed
  value/gradient/JVP/VJP/HVP products, and a bounded FortOpt L-BFGS-B adapter.
  The independent `test_mlp_multilabel_objective` oracle covers finite
  differences, adjoint duality, mixed hyperparameter HVPs, malformed-weight
  transactionality, and both coordinate modes; CUDA remains a typed refusal.
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
  benchmark lane.
- [x] Extend `multiclass_probability_calibrator_t` with weighted one-vs-rest
  isotonic probability calibration. Each raw softmax column is fitted with a
  weighted pool-adjacent-violators map, linearly interpolated at prediction,
  and renormalized to the simplex while preserving sorted arbitrary labels.
  `test_multiclass_isotonic_calibration` checks an independent PAVA oracle,
  weighted simplex values, deterministic ties, typed active-set JVP/VJP and
  parameter-product refusals, and the typed CUDA boundary. The companion
  `fortml-bench/results/MULTICLASS_ISOTONIC_CALIBRATION.md` report records the
  NumPy/Fortran correctness-gated lane. The calibrated-softmax OOF wrapper
  now routes this value-only policy with the same typed active-set boundary;
  generic estimator routing remains open.
- [x] Add weighted one-vs-rest Platt sigmoid calibration to
  `multiclass_probability_calibrator_t`. Stable raw softmax columns receive
  deterministic weighted sigmoid fits and are renormalized to a simplex while
  preserving sorted arbitrary integer labels. Interleaved slope/intercept
  parameters expose exact smooth input and parameter JVP/VJP products;
  transactional failed fits leave the previously fitted model unchanged.
  `test_multiclass_platt_calibration` supplies central-difference and adjoint
  oracles, deterministic weighted replay, tie handling, and typed CUDA refusal;
  `fortml-bench/results/MULTICLASS_PLATT_CALIBRATION.md` records the independent
  NumPy correctness-gated lane. The calibrated-softmax OOF wrapper now routes
  this smooth policy and records its packed calibration metadata and products.
- [x] Add leakage-safe multiclass calibrated-softmax OOF routing. Each
  deterministic stratified fold fits an independent deployment-base softmax,
  writes one held-out logit row per sample, and fits temperature, weighted
  one-vs-rest Platt, or weighted isotonic calibration only on those rows before
  refitting the deployment model on all samples. The wrapper state records the
  method, calibration packed count, isotonic knot count, and derivative
  availability. Temperature and Platt expose exact packed/input JVP/VJPs;
  isotonic values are complete while active-set products and every CUDA path
  return typed refusals. `test_calibrated_softmax_classifier` and
  `fortml-bench/results/CALIBRATED_SOFTMAX_CV.md` provide independent oracles
  and clean multi-policy provenance.
- [x] Make multiclass calibrated-softmax OOF fitting transactional. The public
  `fit` method now trains an isolated candidate and commits the softmax,
  calibration buffers, method metadata, and OOF diagnostics only after every
  fold, calibration map, and deployment refit succeeds. Malformed dimensions,
  weights, options, and class support therefore leave an existing deployment
  fitted and numerically unchanged; `test_calibrated_softmax_classifier`
  checks this preservation contract independently.
- [x] Add leakage-safe binary calibration-aware cross-validation through
  `calibrated_logistic_classifier_t`. Each deterministic stratified fold fits
  an independent logistic model, produces held-out margins for every sample,
  fits sigmoid, positive-temperature, or weighted isotonic calibration on the
  out-of-fold margins, and then fits the deployment model on all rows. The
  state records fold count and pre/post-calibration out-of-fold log loss.
  Smooth temperature and sigmoid maps expose exact joint input/parameter
  JVP/VJP products; isotonic active sets and CUDA prediction return typed
  refusals. `test_calibrated_logistic_classifier` supplies central
  finite-difference, adjoint, sorted-label, simplex, and device-boundary
  oracles. The parallel multiclass calibrated-softmax OOF route covers
  temperature, Platt, and isotonic policies with typed isotonic active-set and
  CUDA boundaries; generic estimator CV routing remains open.

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
- [x] Add the explicit `make_cubic_spline_basis` convenience constructor and
  matching `initialize_cubic_spline` initializer. It fixes FortNum's clamped
  spline order to four (degree three), preserving the generic feature layout
  and fixed-span derivative contract; `test_basis_cubic_spline` supplies an
  independent Cox--de Boor oracle and finite-difference product checks.
- [x] Provide a parameter-free Chebyshev first-kind basis through
  `make_chebyshev_basis`. The per-input `T_1` through `T_degree` recurrence,
  optional shared intercept, exact input JVP/VJP/HVP products, independent
  recurrence/finite-difference/adjoint tests, and explicit resident-CUDA
  refusal are covered by `test_basis_chebyshev` and the release lane
  `results/CHEBYSHEV_BASIS.md`.
- [x] Add a deterministic fixed-state random Fourier feature map,
  `make_random_fourier_basis`, with explicit frequency/phase inputs, optional
  intercept, analytic value/input JVP/VJP/HVP products, and an independent
  oracle plus release row in `../fortml-bench`. Its zero-parameter contract
  keeps sampled features reproducible across folds; resident CUDA execution
  remains an explicit capability boundary.
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
- [x] Add a deterministic `time_series_splitter_t` with chronological
  expanding or rolling training windows, fixed-size contiguous test windows,
  an explicit gap that is never included in training, replayable reset state,
  and typed invalid-window refusals. `test_validation` checks exact one-based
  windows against a hand-computed fixture and verifies reset replay. The
  release app and independent `fortml-bench` NumPy lane record all train/test
  indices before timing; this is an index-only CPU contract, so CUDA requests
  remain a declared capability boundary.
- [x] Add `estimator_score_metadata_t` and
  `estimator_validation_metadata_t`. Scorer metadata records input
  representation, metric kind, maximize/minimize orientation, sample-weight
  support, and differentiability. Validation metadata bundles an estimator
  capability with explicit clone/reset declarations and parameter count, and
  refuses malformed or name-mismatched records without mutating a candidate.
  Independent tests cover scorer orientation and clone/reset guards. The
  records describe a model-specific clone protocol; they do not silently
  reuse fitted estimators.
- [x] Add `fortml_cross_validation` scoring over K-fold, stratified, grouped,
  and chronological splitters. `cross_validation_fold_proc` receives one
  disjoint train/test partition and returns a natural-orientation score,
  parameter gradient, and positive fold weight. `cross_validation_result_t`
  retains fold diagnostics and computes weighted mean or weighted sum values,
  oriented scores, and minimization gradients for FortOpt. The evaluator
  refuses undeclared clone/reset state before invoking a callback, rejects
  nonfinite fold products, and returns a typed CUDA refusal. A borrowed
  `cross_validation_objective_t` exposes the same differentiable validation
  objective through `fortopt_objective::objective_t` for grid/random/
  L-BFGS-B search. `test_cross_validation` is an independent hand-computed
  fold/gradient oracle; `docs/CROSS_VALIDATION.md` records the callback and
  lifetime contract.
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
- [ ] Add quantile and power transforms, normalization, ordinal encoding,
  target encoding with leakage guards, hashing, and sparse CSR/CSC feature
  views. The total-degree `make_polynomial_interaction_basis` now provides
  deterministic polynomial interactions with analytic value/JVP/VJP/HVP
  products; fitted interaction transforms and sparse views remain open.
  Every fitted transform records statistics, feature names, dtypes, and the
  treatment of unseen or missing categories.
- [x] Add a total-degree `make_polynomial_interaction_basis` with deterministic
  exponent enumeration and analytic value/JVP/VJP/HVP products. The independent
  `test_basis_polynomial_interactions` oracle and companion benchmark gate the
  feature; fitted interaction transforms and sparse views remain open.
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
- [x] Add `basis_fanout_pipeline_t` as a bounded named DAG layer. Each branch
  is a validated `sequential_basis_pipeline_t`; forward features concatenate
  in branch order, reverse input cotangents sum across branches, and packed
  branch parameters expose stable names and one-based offsets. CPU JVP/VJP/HVP
  products are exact and independent finite-difference/adjoint checks cover a
  nontrivial two-branch composition. Device-dispatch methods delegate only to
  selected CPU execution and return `FORTNUM_NOT_IMPLEMENTED` for CUDA. The
  graph is deliberately acyclic with no residual or conditional edges;
  `test_basis_fanout_pipeline` and the companion correctness-gated benchmark
  record this bounded contract.
- [x] Add `basis_residual_pipeline_t` as a bounded named residual-sum DAG. A
  main and residual `sequential_basis_pipeline_t` must agree on input and
  feature shapes; value products sum the branches, reverse products sum input
  cotangents, and packed branch parameters retain stable names and offsets.
  CPU value/JVP/VJP/HVP products, metadata, and transactional typed CUDA
  refusals are covered by `test_basis_residual_pipeline` and the independent
  `results/BASIS_RESIDUAL_PIPELINE.md` lane.
- [x] Add a versioned, compiler-independent state dictionary for fitted
  horizontal `basis_pipeline_t` unions. The host text format preserves input
  schema, stage/feature/parameter names, one-based feature/parameter offsets,
  fit state, dimensions, and packed parameters. Capture and restore validate a
  complete metadata contract on a deep candidate before committing, while
  malformed/version-mismatched/nonfinite files are transactional refusals.
  `save_basis_pipeline_device` and `load_basis_pipeline_device` dispatch CPU
  text I/O and return typed CUDA refusals. `test_pipeline_persistence`,
  `docs/PIPELINE_PERSISTENCE.md`, and the release benchmark provide independent
  round-trip, value/JVP/VJP/HVP, metadata, invalid-input, and device evidence;
  sequential/residual/linear-estimator, callback-registration, sparse, and
  resident-device schemas remain separate follow-up contracts.
- [ ] Add parallel feature-union execution, device-resident transforms, and
  general column-wise transformer graphs beyond fixed basis maps.
- [ ] Extend the bounded named fan-out/fan-in layer with conditional stages,
  explicit cycle detection, and parallel/device-resident execution. Pipeline
  nodes expose local parameter blocks so hyperparameter derivatives and
  optimizer routing remain composable through the full graph.
- [x] Add bounded dense input-schema propagation to horizontal, sequential,
  and fan-out basis pipelines. `basis_input_schema_t` gives every input column
  a deterministic default or caller-supplied unique name, validates candidate
  names transactionally before a transform, and exposes stable name accessors;
  `test_pipeline_metadata` and `results/PIPELINE_SCHEMA.md` provide the
  independent refusal and release evidence.
- [ ] Extend schema metadata to dtypes, sparse layouts, feature names through
  every transformer, estimator-wide metadata routing, train-only fitting,
  `fit_transform`, `transform`, `inverse_transform` where mathematically
  defined, and partial-fit propagation for streaming data.
- [ ] Add deterministic train/test, repeated K-fold, blocked, and Monte Carlo
  split iterators plus routed estimator parameters. Ordinary, stratified,
  grouped, and chronological splitters now feed weighted cross-validation
  scoring; repeated/Monte Carlo policies, estimator-wide metadata routing, and
  train-only transformer fitting remain open.
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
- [x] Complete binary XGBoost classifier probability products. Add stable
  `predict_log_proba` values computed directly from the raw margin, with
  fixed-tree input JVP/VJP products and finite-difference/adjoint oracles.
  Expose ordered categorical policy, category cardinality, categorical-feature,
  and interaction-group metadata through the classifier facade. The release
  benchmark records log-probability round-trip error; categorical products
  retain the typed discrete derivative refusal and selected CUDA remains an
  explicit `FORTNUM_NOT_IMPLEMENTED` boundary.
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
- [x] Add fixed-structure tree leaf-weight products to `xgboost_t` and
  `lightgbm_t`. The packed `[base_score, leaf weights]` coordinates expose
  raw-margin `predict_leaf_jvp`/`predict_leaf_vjp` products with learning-rate
  and DART scale factors, deterministic tree/node ordering, finite/query and
  shape guards, and a transactional malformed-tangent refusal. Split routing
  remains held fixed: input products retain their split-boundary refusal, while
  leaf products remain defined on the boundary. `test_tree_leaf_products` is
  an independent hand stump oracle and `fortml-bench/results/TREE_LEAF_PRODUCTS.md`
  records CPU correctness/timing plus the explicit resident-CUDA gap.
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
- [x] Add gradient-boosted regression for Tweedie losses. `xgboost_t` now
  exposes `fit_tweedie`, stable compound-Poisson log-link value/gradient/
  Hessian products for `1 < variance_power < 2`, weighted exact and histogram
  growth, validation/early stopping, and typed CUDA refusal. The independent
  `test_xgboost_tweedie` oracle and `fortml-bench/results/XGBOOST_TWEEDIE.md`
  report record zero CPU oracle error and the unavailable CUDA row.
- [x] Add a fixed-shape Gamma log-link objective to `xgboost_t`. `fit_gamma`
  and the generic `gamma`/`reg:gamma` aliases validate strictly positive
  targets and `gamma_shape`, expose exact weighted value/gradient/Hessian
  products, use positive `exp(margin)` predictions, preserve objective
  metadata through warm starts and text snapshots, and support exact and
  weighted-histogram CPU growth. `test_xgboost_gamma` and
  `fortml-bench/results/XGBOOST_GAMMA.md` provide independent Newton oracles;
  tree CUDA remains a typed refusal until a resident kernel is linked.
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
  OOB decision probabilities, accuracy, and coverage are transactional; the
  fixed-state accuracy permutation-importance diagnostic now has a dedicated
  repeat/std API, typed CUDA refusal, independent behavioral test, and NumPy
  replay benchmark. Extra-trees, bagging, random-subspace, isolation forests,
  SHAP, and differentiable routing remain open.
- [x] Add XGBoost-style second-order boosting: per-leaf gradient/Hessian
  aggregation, regularized split gain, weighted quantile cuts, exact and
  histogram algorithms, sparse-aware default directions, monotonic constraints,
  recursive depth-limited growth, and deterministic feature ordering.
- [x] Add interaction constraints and bounded ordered-gradient categorical
  partitions. `xgboost_options_t%categorical_features` accepts sorted one-based
  integer-coded feature indices with `categorical_policy="ordered"` and an
  explicit 2--64 `categorical_max_categories` bound. Per-node category prefixes
  are ordered by gradient/Hessian score with code tie-breaks; metadata survives
  warm starts, slices, and version-4 text snapshots. The independent
  `test_xgboost_categorical` fixture covers the hand partition, cardinality
  refusal, save/load metadata, discrete derivative refusal, and CPU/CUDA
  behavior; `xgboost_categorical.csv` is the release oracle lane.
- [x] Add the bounded XGBoost `reg:tweedie` objective with explicit
  `1 < tweedie_variance_power < 2` validation. `xgb_tweedie_loss` and
  `xgb_tweedie_derivatives` expose the finite weighted value, exact log-mean
  gradient, and positive Hessian; the public `xgboost_t%fit_tweedie` binding
  uses the positive `exp(margin)` prediction
  link, preserves the power in text snapshots, and rejects invalid powers,
  negative targets, and CUDA prediction. `test_xgboost_tweedie` checks the
  formulas, finite-difference value/gradient, fit/prediction semantics, and
  device boundary; `xgboost_tweedie.csv` is the independent release lane.
- [ ] Add column blocks and
  distributed histogram reduction with independent oracles.
- [x] Add a bounded, separately named LightGBM-style numeric histogram and
  leaf-wise policy. `lightgbm_t` reuses the XGBoost weighted-quantile cut
  primitive, supports weighted squared regression and binary logistic loss,
  and splits the globally best current leaf until `num_leaves` (or
  `max_depth`) is reached. `test_lightgbm` is an independent hand oracle for
  weighted leaves, logistic probabilities, deterministic tree shape, fixed-tree
  JVP/VJP products, split-boundary refusals, and CPU/CUDA capability. Staged
  margins/predictions, additive base-plus-tree contributions, and transactional
  fitted-prefix slicing are covered by the independent tree-walk oracle
  `test_lightgbm_staged_slice` and the release lane `lightgbm_leafwise.csv`.
  Versioned `FORTML_LIGHTGBM_TEXT` save/load now round-trips metadata and live
  node arrays with transactional schema, structural, finite-value, and EOF
  validation; `test_lightgbm_persistence` covers staged/contribution parity
  and malformed/trailing-record refusals. GOSS is available through
  `boosting_type="goss"`: deterministic top-gradient retention and hash-ranked
  other-row sampling apply the exact `(1-top_rate)/other_rate` correction to
  selected small-gradient gradient/Hessian pairs for regression and binary
  logistic fits, warm starts, slicing, and schema-3 persistence. The
  `test_lightgbm_goss` hand oracle checks the weighted leaves, seed replay,
  malformed rates, and CUDA refusal. EFB, categorical statistics,
  external-data iterators, distributed workers, and resident GPU histograms
  remain separate gaps. `fit_warm_start` transactionally extends a fitted
  prefix to a larger tree target under matched controls; `test_lightgbm_warm_start`
  covers full-fit equivalence, prefix preservation, malformed controls, and
  explicit validation-state refusal.
- [x] Add bounded seeded DART/dropout boosting to `lightgbm_t`. `dart_drop_rate`,
  `dart_skip_drop`, and `dart_max_drop` select prior trees from a stable hash
  stream; dropped and new trees use explicit `1/(k+1)` tree normalisation, and
  per-tree scales are preserved through staged predictions, contributions,
  slices, warm starts, and schema-3 text persistence. `test_lightgbm_dart` and
  `results/lightgbm_dart.csv` replay the independent depth-one NumPy tree-walk
  oracle, deterministic scales, persistence, warm start, invalid-rate refusal,
  fixed-tree derivatives, and CUDA refusal. Fit-time dropout derivatives remain
  explicitly unsupported because tree selection is discrete. Class- and
  query-weighted ranking objectives
  (pairwise logistic and Lambda-style NDCG), survival/count objectives,
  custom objective callbacks, and custom
  evaluation metrics. Unsupported objectives must return a structured refusal.
- [x] Add the LightGBM-style multiclass OVR classifier adapter. `lightgbm_multiclass_t`
  fits one binary leaf-wise child per sorted integer class, normalizes final and
  staged probabilities, exposes raw margins and validation best-prefix metadata,
  and applies the fixed-tree sigmoid/normalization chain rule to input JVP/VJP
  products. `test_lightgbm_multiclass` is an independent probability, staged
  margin, validation-transaction, derivative, and label oracle; the release
  lane is `lightgbm_multiclass.csv`. Split-boundary products and resident CUDA
  histogram execution remain typed refusals.
- [x] Add deterministic LightGBM GOSS sampling to `lightgbm_t`. `boosting_type="goss"`
  keeps the largest absolute-gradient rows and a seed-ranked subset of the
  remainder, rescales selected small-gradient gradient/Hessian pairs by
  `(1-top_rate)/other_rate`, and preserves the policy through warm starts,
  prefix slices, and schema-3 text persistence. `test_lightgbm_goss` provides
  a hand leaf oracle, deterministic replay, invalid-rate refusal, and explicit
  CUDA refusal; the release lane is `lightgbm_goss.csv`.
- [x] Add bounded seeded DART/dropout boosting to `xgboost_t`. `booster="dart"`
  selects prior trees through a compiler-independent integer hash stream;
  selected and newly fitted trees use explicit `1/(k+1)` normalisation, and
  per-tree scales are preserved by staged margins, additive contributions,
  fitted-prefix slices, transactional warm starts, and schema-5 text
  persistence. `test_xgboost_dart` and `fortml-bench/results/xgboost_dart.csv`
  replay the independent depth-one NumPy Newton oracle, deterministic scales,
  replay/persistence/warm-start parity, malformed-control refusal, and typed
  CUDA refusal. Fit-time dropout derivatives remain explicitly unsupported
  because tree selection is discrete; EFB, distributed growth, and resident
  GPU histograms remain open.
- [x] Add bounded SHAP-like additive contribution products to both boosted-tree
  families. `xgboost_t%predict_shap` and `lightgbm_t%predict_shap` use exact
  feature-subset Shapley enumeration with path-dependent expected baselines;
  XGBoost integrates omitted splits with stored cover and LightGBM with child
  row-count proportions. Rows sum to the raw margin, fitted DART scales and
  prefixes are retained, models wider than 12 features return a typed refusal,
  and CPU/CUDA dispatch never hides a host fallback. `test_tree_shap` and
  `fortml-bench/results/TREE_SHAP.md` provide independent one-stump NumPy
  oracles. Monotone prediction checks, partial dependence, model-size/tree
  export diagnostics, and resident GPU explanation kernels remain open.
- [ ] Extend categorical support beyond the ordered-gradient policy (exhaustive
  partitions for very small cardinalities, categorical statistics/target
  encoding, categorical interaction constraints) and add distributed/resident
  GPU histogram implementations.

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
  nonsmooth/CUDA boundaries. Kernel SVM/SVR and Nyström approximation remain
  open, with explicit solver/feature-memory limits. Fixed random Fourier
  features are available through `make_random_fourier_basis` and retain an
  explicit CPU derivative contract and CUDA refusal.
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
  prediction, packed-head input/parameter JVP/VJPs, an exact joint
  parameter/input probability-cotangent HVP, thresholds, weights, and typed
  CUDA refusals. The bounded release evidence is `test_classifier_chain` and
  `fortml-bench/results/classifier_chain.csv` (including independent HVP
  finite-difference rows).
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
- [ ] Add train/test, repeated K-fold, and Monte Carlo splitters plus shared
  cross-validation scoring. Ordinary, stratified, grouped, and chronological
  time-series/blocked splitters now exist as independent index-only contracts;
  index generation remains independent of estimator state and safe for empty
  or uneven folds.
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
- [x] Add exact outer hyper-HVP products to the fixed full-batch SGD
  momentum/Nesterov objective on the one-layer all-linear MLP branch. The
  constant network Hessian permits analytic mixed second tangents through
  velocity, look-ahead, learning-rate, L2, and momentum state; an independent
  central-difference oracle covers both classical and Nesterov recurrences.
  Nonlinear/multilayer networks, mini-batches, schedules, clipping, and CUDA
  remain typed third-derivative or resident-state refusals.
- [x] Add the exact fixed full-batch AdamW trajectory hypergradient objective
  over `[log(learning_rate), log(l2), log(weight_decay)]`, including analytic
  moment/decoupled-decay sensitivities, JVP/VJP products, independent central
  differences, and a FortOpt L-BFGS-B adapter.
- [x] Extend the AdamW trajectory contract with exact beta1/beta2 sensitivities
  over unconstrained logits, bias-correction derivatives, independent central
  differences, and the same FortOpt L-BFGS-B adapter. Mini-batch, schedule, and
  CUDA AdamW hypergradients remain explicit follow-up contracts.
- [x] Add exact scheduled AdamW trajectory hypergradients over base rate, L2,
  decoupled weight decay, beta logits, epsilon, and active typed schedule
  coordinates. Constant, cosine, warmup-cosine, and exponential schedules use
  analytic CPU value/gradient/JVP/VJP products through moment, bias-correction,
  and shrinkage state and feed FortOpt L-BFGS-B directly. The independent
  central-difference/NumPy release lane is
  `fortml-bench/results/MLP_ADAMW_SCHEDULE_HYPERGRADIENT.md`; CUDA, lower
  precision, zero-root, and outer-HVP requests are typed refusals.
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
- [x] Add the exact fixed full-batch AMSGrad trajectory hypergradient objective
  over `[log(learning_rate), log(l2), logit(beta1), logit(beta2),
  log(epsilon)]`. The analytic product propagates the first and second moments,
  the elementwise max-second-moment active set, bias correction, and the
  stabilized denominator through validation MSE. `value_gradient`, JVP, scalar
  VJP, and the bounded FortOpt L-BFGS-B adapter are covered by independent
  central-difference and adjoint tests. Max ties, zero square-root or update
  denominators, and CUDA return typed refusals. The release lane is
  `fortml_bench_amsgrad_hypergradient` with the independent NumPy record in
  `fortml-bench/results/amsgrad_hypergradient.csv`.
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
- [x] Add the metric-aware plateau schedule through the same typed object. Its
  `rate_with_metric` surface carries metric/best/bad-count/reduction-count
  state explicitly, and its discrete trainer boundary returns a typed refusal
  rather than silently treating a validation policy as an update-indexed rate.
- [x] Add `mlp_schedule_hypergradient_objective_t` for exact differentiable
  schedule/optimizer trajectories. The packed log/logit layout has independent
  finite-difference and scalar-adjoint tests, a complete-array release app and
  NumPy benchmark, and explicit second-order hyper-HVP scope (third network
  derivatives are not approximated). L-BFGS-B consumes the same reverse
  products; unsupported device trajectories return typed refusal.
- [x] Close the affine scheduled outer-HVP slice for the stateless schedule
  families. `mlp_learning_rate_schedule_t%rate_with_second_derivatives`
  returns the exact raw rate Hessian; the scheduled trajectory propagates
  mixed state tangents through constant, linear-warm-up, cosine,
  warm-up-plus-cosine, exponential, and one-cycle rates, including the
  log/logit chain rule for schedule fields. Independent cosine and one-cycle
  central-difference/Hessian-symmetry checks, `fortml_bench_mlp_schedule_hvp`,
  and `results/mlp_schedule_hvp.csv` gate the CPU affine path. Nonlinear
  networks, plateau branch changes, and CUDA remain typed refusals rather than
  hidden finite-difference or host fallbacks; optimizer-group and mini-batch
  outer HVPs remain open.
- [x] Extend the scheduled trajectory objective with exact one-cycle
  peak/final-rate derivatives. The shared four-vector uses logarithmic
  peak/final coordinates for `MLP_SCHEDULE_ONE_CYCLE`, with exact products
  through its linear warm-up and cosine tail, metadata identifying the layout,
  typed domain checks, a FortOpt adapter, and independent central-difference,
  JVP, VJP, and CUDA-refusal coverage. Optimizer-group, validation-policy, and
  resident-device schedule derivatives remain open in the bounded row above.
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
- [x] Add a weighted pairwise contrastive metric-learning loss to the shared
  neural-loss facade. Matching and non-matching Euclidean pairs expose value,
  JVP, VJP, and HVP products under the common mean/sum reduction contract;
  independent formula, finite-difference, and adjoint oracles cover both
  embedding inputs. Non-matching zero distances and exact margin boundaries
  refuse derivative products transactionally, and the value dispatcher returns
  a typed CUDA refusal. Triplet/sequence losses and resident CUDA kernels remain
  open.
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
- [x] Extend the model-agnostic trainer with a finite validation callback,
  minimum-improvement threshold, patience stop, best-parameter restoration,
  validation diagnostics, and schema-4 checkpoint persistence. The independent
  quadratic oracle covers the callback sequence, split continuation, and
  transactional callback-presence refusal. Validation callbacks remain
  process-local and host-owned.
- [x] Integrate the typed metric-aware plateau schedule into `mlp_train`.
  Epoch validation loss (or training loss without a held-out stream) drives a
  deterministic best-metric/bad-observation/reduction state machine.  The
  base-rate and plateau-factor products are analytic on the active branch;
  version-10 in-memory and formatted checkpoints preserve the counters and
  best metric, and an independent fixture checks recurrence, malformed state,
  and interrupted-versus-uninterrupted resume.  Resident CUDA metric
  reduction and optimizer state remain an explicit typed boundary.
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
- [x] Add the typed MLP precision capability boundary. `precision_kind` and
  `mlp_precision_name` distinguish FP64/FP32/FP16/BF16 in options, state, and
  in-memory checkpoints; FP64 follows the independent deterministic reference
  recurrence, while recognized lower-precision requests return non-mutating
  `FORTNUM_NOT_IMPLEMENTED` until master weights, loss scaling, overflow
  recovery, and deterministic reductions exist. Unknown modes are domain
  errors, and CPU/CUDA lower-precision execution remains explicitly unavailable.
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
  observation lists use exact generated/closed-form products for RBF, Matérn
  3/2, Matérn 5/2, periodic, linear, constant, polynomial, and supported
  compositions. Rational-quadratic, cosine, user-formula, and other leaves
  return typed refusals until their second input-parameter products are
  generated and independently checked. ARD-RBF mixed observations now also expose analytic
  parameter HVPs and query-input JVP/VJPs, with a dense finite-difference
  oracle in `test_derivative_gp_ard` and a 94-row CPU/CUDA benchmark lane.
  `test_derivative_gp_products` checks both branches.
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
  oracle in `test_fortsym_matern32`. The Matérn 5/2 HVP leaf is now emitted by
  FortSym `873d33f` (80 IR nodes, 65 compound operations), with an independent
  oracle in `test_fortsym_matern52`; complete proof and source-hash sidecars
  remain a release task, and the
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
  posterior covariance, packed parameter gradients, query adjoints, and the
  mixed-observation likelihood HVP. The HVP carries an exact four-jet through
  every separable factor and is checked against an independent central-
  difference likelihood oracle. The CPU reference is complete; CUDA remains a
  typed `FORTNUM_NOT_IMPLEMENTED` refusal until resident covariance and
  factorization kernels are linked. See
  `fortml-bench/results/DERIVATIVE_GP_SPECTRAL_MIXTURE_HVP.md`.
- [x] Extend mixed-observation derivative-GP HVPs to the periodic kernel. The
  radial `sin(pi*sqrt(s)/period)**2` derivatives include the coincidence-safe
  fourth-input term needed by period/period products; all three logarithmic
  kernel coordinates and log observation noise are differentiated analytically
  through covariance blocks and the Cholesky solve. The independent
  `test_derivative_gp_periodic_hvp` central-difference likelihood oracle and
  `fortml-bench/results/DERIVATIVE_GP_PERIODIC_HVP.md` gate the CPU contract;
  resident CUDA covariance/factorization remains a typed refusal.
- [x] Extend mixed-observation derivative-GP HVPs to the rational-quadratic
  kernel. The exact radial `F_s`/`F_ss` parameter-direction products cover
  variance, lengthscale, alpha, and log-noise coordinates through the dense
  Cholesky solve; independent covariance/HVP and query-product checks cover
  the CPU path, while selected CUDA prediction remains a typed refusal. See
  `docs/GP_RATIONAL_QUADRATIC_MIXED_HVP.md` and the corresponding
  `fortml-bench/results/DERIVATIVE_GP_RATIONAL_QUADRATIC_HVP.md` release gate.
- [x] Extend the bounded scalar 1-D `second_derivative_gp_t` reference from
  RBF to Matérn-5/2. Mixed orders `0:2`, exact order-four covariance blocks,
  order-five query JVP/VJP products away from Matérn coincidences, dense latent
  covariance, and explicit CUDA/coincident-fifth-derivative/non-RBF/order
  refusals are independently checked. The later RBF order-three lane adds
  order-six/seven products and analytic likelihood HVPs; see
  `docs/SECOND_DERIVATIVE_GP.md` and the release benchmark lanes.
- [ ] Add scalar objectives and parameter gradients for multi-output, sparse
  variational, local, SKI, Lanczos, and matrix-free GP paths. Inducing-point and
  local-gate training remain separate parameter blocks.
- [x] Add binary Laplace GP classification for logistic and probit likelihoods,
  with damped Newton state, latent/probability prediction, input JVP/VJP
  products over the kernel derivative contract, and an exact envelope gradient
  for the converged mode log posterior (without evidence correction).
- [x] Add sklearn-style binary Laplace-GP `predict_log_proba` with finite
  probit-tail clipping, input and fixed-state kernel-parameter JVP/VJP
  products, and a transactional `set_parameters` seam that rebuilds
  covariance factorizations while retaining the fitted Newton state. The
  independent `test_gp_classification_log_proba` oracle checks value/log
  round trips, central differences, adjoint identities, and the typed CUDA
  refusal; see `docs/GP_CLASSIFICATION_LOG_PROBA.md` and the release benchmark.
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
- [x] Refuse `end_residency` when no residency is active with a typed domain
  status, preserving ownership and transfer counters. The lifecycle assertion
  is covered by `test_device_contract`; no kernel timing is claimed.
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

- [x] Deliver the first production persistence slice for fitted horizontal
  basis pipelines: versioned host text, complete schema/name/offset dictionary,
  deep-candidate transactional restore, and explicit CUDA refusal are covered
  by `fortml_pipeline_persistence`. This does not claim estimator-wide or
  resident-device serialization.
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
- [x] Add the fixed-state random Fourier feature release lane with direct
  trigonometric value and central-difference input-product oracles, explicit
  zero-parameter metadata, and a typed CUDA capability row in
  [`results/RANDOM_FOURIER.md`](../fortml-bench/results/RANDOM_FOURIER.md).
- [ ] Add a release-app benchmark for joint basis-pipeline training, including
  linear and Fourier initializations, FortOpt convergence, and a typed CUDA
  refusal row.
- [x] Add the deterministic seeded random-forest classifier benchmark with a
  direct NumPy threshold oracle, aligned probability-simplex checks, CPU fit
  and prediction timings, and an explicit CUDA refusal in
  [`results/RANDOM_FOREST.md`](../fortml-bench/results/RANDOM_FOREST.md).
- [x] Add deterministic random-forest permutation importance with a fixed
  fitted-state accuracy contract, population repeat dispersion, transactional
  invalid/CUDA refusals, and an independent NumPy Park--Miller replay in
  [`results/random_forest_permutation.csv`](../fortml-bench/results/random_forest_permutation.csv).
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
- [x] Add the deterministic mini-batch coupled-L2 Adam trajectory
  hypergradient objective over `[log(learning_rate), log(l2)]`. A private
  seeded batch cursor is replayed for every FortOpt evaluation, and analytic
  parameter, first/second-moment, bias-correction, and stabilized-denominator
  tangents provide value/gradient, JVP, and scalar VJP products for validation
  MSE. `mlp_optimize_minibatch_adam_hyperparameters` consumes the same callback
  through bounded FortOpt L-BFGS-B. `test_mlp_minibatch_adam_hypergradient`
  independently checks central differences, adjoints, convergence, and typed
  CUDA refusal; the release workload is
  `fortml_bench_mlp_minibatch_adam_hypergradient`. The outer HVP and resident
  CUDA trajectory remain explicit refusals until third network derivatives and
  a resident Adam state are available.
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
- [x] Keep named PINN products independently addressable. `term_gradients` and
  `term_hvps` expose `(n_parameters,4)` matrices in the stable
  `[data,residual,boundary,conservation]` order; their column sums equal the
  aggregate products and inactive terms are zero. The release app and
  independent nonlinear oracle are recorded in the PINN term-products lane.
- [x] Add a reusable canonical symplectic-form term over map Jacobians.
  `fortml_symplectic` forms `D = A^T Omega A - Omega` for canonical `[q,p]`
  coordinates, exposes packed residual value/JVP/VJP and normalized weighted
  value/JVP/VJP products, and checks a caller-supplied form-defect tolerance.
  `symplectic_constraint_t` adapts exact map Jacobian callbacks into the
  existing `physics_constraint_t` seam while preserving the configured weight.
  The independent harmonic-oscillator velocity-Verlet oracle in
  `test_symplectic` checks the form identity, residual adjoint, value products,
  bridge, and typed CUDA refusal. Nondimensionalizing transforms,
  coordinate/time metadata, and richer diagnostics remain future work.
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
`931dac5f39eb6ea5ab3854d5af49b346bea950af` and `fortsym` `main` at
`26250ce24e3890711343a2f18d5feae2be832508`. The checked-in RBF HVP/product
module was generated by FortAD `5e1bfe0`; the RBF primal and first-order leaf
was generated by FortSym `f71a1aa` (the earlier primal-only leaf remains
generator revision `16fc3a8`). The RBF log-length tangent used
by the derivative-GP products was independently checked with a temporary
FortSym proof against the dense numerical oracle. The Matérn 1/2 HVP leaf is
also generated by FortSym `9482261`; its independent oracle is the
`test_fortsym_matern12` target. The Matérn 3/2 HVP leaf is generated by the
  FortSym `b72a23a` and checked by `test_fortsym_matern32`; the Matérn 5/2
  HVP leaf is generated by FortSym `873d33f` and checked by
  `test_fortsym_matern52`. Future derivative work must refresh both
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
  integrators for separable or otherwise splittable Hamiltonians. The current
  `hamiltonian_mlp_t%leapfrog` remains the CPU map provider used by the
  diagnostic, while general Hamiltonians require an applicable implicit
  symplectic method or an explicit refusal. Training can differentiate through
  the map, while inference reports the integrator and step size used.
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
- [x] Add the finite-feature last-layer kernel-ridge initialization contract
  for an existing affine-output MLP. The initializer records dimensions,
  regularization, fixed-feature derivative scope, and its explicit
  `exact_infinite_width=.false.` boundary; it has independent normal-equation,
  central-difference, transaction, and typed CUDA refusal oracles plus a
  benchmark lane. This is a deterministic posterior-mean warm start, not an
  NNGP covariance or exact infinite-width claim.
- [ ] Add sampled prior draws, NNGP covariance propagation, and deterministic
  full GP-posterior/NTK weight maps. Each must record the kernel, architecture,
  width, seed, design set, solve tolerance, and whether it promises a mean fit
  or a covariance approximation.
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
- [`CHANGE_POINT_GP.md`](../fortml-bench/results/CHANGE_POINT_GP.md), backed by
  `change_point_gp.csv` for gated-child covariance, exact-GP prediction,
  input/mixed and parameter products, and the typed static-operator/CUDA
  refusal.
- [`CALIBRATED_SOFTMAX_CV.md`](../fortml-bench/results/CALIBRATED_SOFTMAX_CV.md),
  backed by `calibrated_softmax_cv.csv` for stratified OOF logits, positive
  temperature, weighted Platt, and isotonic replay, fold log-loss diagnostics,
  packed-product metadata, and the typed CUDA/refusal boundaries.
- [`MLP_CONSTANT_SCHEDULE_HVP.md`](../fortml-bench/results/MLP_CONSTANT_SCHEDULE_HVP.md),
  backed by `mlp_constant_schedule_hvp.csv` for affine value/gradient/JVP/HVP
  products, Hessian symmetry, FortOpt callback parity, and typed schedule or
  CUDA boundaries.
- [`XGBOOST_DART.md`](../fortml-bench/results/XGBOOST_DART.md), backed by
  `xgboost_dart.csv` for seeded dropout replay, persisted tree scales,
  staged/contribution/slice parity, transactional warm starts, and the typed
  CUDA boundary.
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
- [`XGBOOST_TWEEDIE.md`](../fortml-bench/results/XGBOOST_TWEEDIE.md), backed by
  `xgboost_tweedie.csv` for the independent compound-Poisson value/gradient/
  Hessian oracle, exact/histogram CPU timings, and typed CUDA refusal.
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
- [`TRAINER_VALIDATION.md`](../fortml-bench/results/TRAINER_VALIDATION.md),
  backed by `trainer_validation.csv` for callback-driven patience,
  best-state restoration, schema-4 continuation, and callback-presence
  refusal.
- [`ADAFACTOR_FACTORED.md`](../fortml-bench/results/ADAFACTOR_FACTORED.md),
  backed by `adafactor_factored.csv` for the independent row/column state,
  vector fallback, split/resume recurrence, and typed CUDA boundary.
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
