# fortml public API

FortML has no umbrella module. Import each public type or procedure from the
module named below. Generated derivative kernels, CUDA stubs, and
`fortml_basis_impl` are implementation modules. Ordinary applications should
use the facade in `fortml_basis`.

## Common conventions

The library uses double precision from `fortnum_kinds`. Public examples may
instead rename `iso_fortran_env::real64` to `dp` when declaring application
arrays.

| Object | Shape or ordering |
| --- | --- |
| Tabular input | `x(n_samples,n_features)` |
| Tabular output | `y(n_samples,n_outputs)` |
| Scalar GP target | `y(n_samples)` |
| Recurrent input | `x(n_steps,n_batch,n_inputs)` |
| Recurrent output | `y(n_steps,n_batch,n_outputs)` |
| Recurrent hidden history | `hidden(n_steps,n_batch,n_hidden)` |
| Matrix parameter storage | Fortran column-major order |

Most checked procedures return `type(fortnum_status_t)`. Test it with
`status_ok(status)` from `fortnum_status`. `fortml_sparse_operator` uses
`fortsparse_status_t` and `FORTSPARSE_OK`. Methods inherited from
`linear_operator_t` return `fortnum_krylov` information codes for CG instead of
a status object.

`fortml_device` provides the explicit backend and residency control-plane
contract. Select `FORTML_DEVICE_CPU` or `FORTML_DEVICE_CUDA` with
`fortml_device_t%select`; query availability with `fortml_query_device` or
`fortml_device_available`. CUDA is refused with `FORTNUM_NOT_IMPLEMENTED` when
the current build exposes only the CUDA stubs. The context records backend
identity, declared resident bytes, ownership metadata, and host/device
transfer counters through `begin_residency`, `record_host_to_device`,
`record_device_to_host`, and `end_residency`. These methods do not allocate or
copy arrays, and they never turn a CPU execution into a GPU claim. See
[`docs/DEVICE.md`](DEVICE.md) for the ownership and refusal contract.

## Estimator capability contracts

`fortml_estimator_capabilities` provides one value-object contract for generic
model-selection, pipeline, and validation code. `estimator_capability_t`
records the estimator role (`FORTML_ROLE_TRANSFORMER`, `PREDICTOR`,
`REGRESSOR`, or `CLASSIFIER`), fitted state and feature/target/class counts,
accepted dense/sparse/missing/sample-weight inputs, partial-fit support,
transform/predict/probability methods, input/parameter/hyperparameter
JVP/VJP/HVP products, and CPU/OpenACC/CUDA/resident-device support. A false
tag is an explicit refusal; generic callers must not infer support from a
method name.

Use `make_transformer_capabilities`, `make_predictor_capabilities`,
`make_regressor_capabilities`, or `make_classifier_capabilities` for common
defaults, then set additional products supplied by a concrete model. Query
with `has_role`, `supports_input`, `supports_derivative`, and
`supports_device`; call `validate` before publishing a record. A requested
record can be checked with `satisfies` or `require_estimator_capability`.
The horizontal, sequential, column-selecting, fan-out, and conditional basis
pipelines expose `%capabilities(report,status)` and report their fitted state, feature
shape, analytic input/parameter products, dense CPU support, and explicit
sparse, missing, sample-weight, and CUDA refusals. `fortml_validation` re-exports
`validate_estimator_capability` and its requirement check so split/search code
can reject an incompatible estimator before consuming a fold.

## FortBO and FortMC interoperability boundary

The companion repositories are optional consumers of FortML model and
probability objects; FortML does not import either package. At the pinned
2026-08-08 revisions (FortBO
`b62a1a0bae1c0766fb35a3127957a39758705160`, FortMC
`e5e42a0ac1d4a4d92fa6b2ee2750b50723342a48`), their public modules contain
versioned contracts, tested acquisition/candidate-search foundations, and a
FortML GP adapter. FortMC additionally ships a gradient-free univariate slice
sampler, while FortBO's
candidate policies remain a partial catalog rather than a complete BO suite:

| Companion | Current public protocol | Not yet supplied by the companion boundary |
| --- | --- | --- |
| FortMC `fortmc` | `fortmc_log_density_t%value(position,status)` and `%gradient(position,gradient,status)`, plus version constants, a default divergence threshold, and `fortmc_slice_sample`/`fortmc_slice_chain` for coordinate-sweep slice sampling | Chain state, transforms, packed parameter registries, HVPs, HMC/NUTS/SMC and other samplers, checkpoints, diagnostics, and device execution |
| FortBO `fortbo` | Versioned `fortbo_posterior_t` with capability-gated moments, covariance, joint/reparameterized samples, predictive log density, moment gradients/Hessians; `fortbo_history_t` gradient-observation/checkpoint state; `fortbo_space_t` normalized continuous/integer/categorical/mixed/conditional spaces with differentiable masks; analytic EI/PI/UCB/log-EI; exact-envelope knowledge gradient; noisy expected improvement; joint qEI/qNEI/qUCB batch acquisitions; risk-sensitive and multi-fidelity criteria; constrained/cost-aware acquisitions; active-learning and level-set design; max-value entropy search; predictive-entropy C1.1/C1.2/C2/C3 expectation-propagation constraints; mixed-integer/categorical candidate search; per-evaluation benchmark metrics; marginal Monte-Carlo EI/PI with CRN, antithetic draws, and pathwise gradients; Sobol TuRBO candidates and Thompson selection; all three TuRBO ordering arms with a pinned Ackley-200 reference; gradient-based DTuRBO in-region acquisition search; FortSym-derived trust-region and posterior-moment derivative leaves; exact posterior mean Hessians from derivative predictions; DTuRBO mode-2 posterior-derivative local models; an indefinite-curvature bound-constrained quadratic subproblem; Pareto archives with exact hypervolume and scalarizations; machine-readable stopping rules; trust-region traces; `fortbo_fit_from_history` value-only/derivative-GP adapters; `fortbo_structured` multi-task/deep-kernel moments adapters; asynchronous worker bookkeeping with fantasies/retries; Bayesian-linear posterior provider; fixed-choice and quadratic/exact constraint penalties; constrained/noisy/multi-objective fixtures; 60D rover and 14D robot-pushing fixtures; resident OpenACC candidate/TuRBO reductions and hardware probes; BoTorch/GPyTorch/JAX/NumPy cross-framework correctness and regret fixtures | qKG and batch Thompson/fantasy policies, wider sparse/variational adapters, posterior-gradient device execution, and full surrogate moment-gradient coverage. Rover plumbing and some paper-baseline comparisons remain open |

FortML does not yet ship FortMC samplers or a direct FortML-side BO policy; FortBO
now ships the tested value/derivative-GP, multi-task, and deep-kernel posterior
adapters described above. Its resident candidate and TuRBO reductions require
OpenACC and otherwise return typed refusals. Do not claim HMC/NUTS,
Bayesian-optimization, or GPU parity for a FortML model until the corresponding companion adapter has
an independent behavioral oracle, explicit refusal behavior, and a benchmark
record.

Several modules export a procedure both as a type-bound method and as a free
procedure. This reference uses the type-bound spelling. Free `mlp_jvp`,
`mlp_vjp`, `mlp_hvp`, `gp_fit`, `gp_predict`, `gp_predict_jvp`,
`gp_predict_vjp`, `gp_predict_hvp`, `gp_log_marginal_likelihood`,
`gp_hyperparameter_gradient`, `gp_hyperparameter_hvp`, `gp_derivative_fit`,
and `gp_derivative_predict` take the model first.

Returned GP variances are latent-function variances unless a procedure says
otherwise. Observation noise is used in fitting and is not added to prediction
variance.

## Product coverage

The derivative surface is model-specific. A method absent from this table is
not supplied through a hidden generic interface.

### `fortml_cuda_dense_api`

`cuda_dense_plan_t` is the typed Fortran wrapper for the resident, no-autodiff
CUDA dense-affine inference primitive. `create(weights,bias,activation,
device_index,status)` validates finite `weights(n_inputs,n_outputs)` and
`bias(n_outputs)`, uploads one resident layer, and records the selected device.
`predict(query_x,outputs,status)` accepts `query_x(n_query,n_inputs)` and
`outputs(n_query,n_outputs)`; only the query batch and result cross the host /
device boundary. `jvp(query_x,query_x_dot,weights_dot,bias_dot,outputs,
outputs_dot,status)` evaluates the forward output and tangent for the resident
weights, a feature tangent, and output-major parameter tangents. `vjp(query_x,
output_bar,query_x_bar,weights_bar,bias_bar,status)` returns query, weight,
and bias cotangents for an output cotangent. The supported
activations are `MLP_LINEAR`, `MLP_TANH`,
`MLP_RELU`, `MLP_GELU`, `MLP_SILU`, `MLP_ELU`, `MLP_SOFTPLUS`, and
`MLP_LEAKY_RELU`. `fitted`, `input_count`, `output_count`, `activation_kind`,
and `device` expose the plan metadata, while `destroy(status)` releases the
resident allocations. `fortml_cuda_dense_available()` is the native capability
probe.

`train_mse(query_x,target,learning_rate,loss,status)` performs one resident
full-batch mean-squared-error update for the affine layer and returns the
pre-update mean loss. `parameters(weights,bias,status)` snapshots the updated
resident parameters, and `transfer_stats(host_to_device_bytes,
device_to_host_bytes,resident_bytes,status)` reports successful copy bytes and
the permanent model allocation. This is deliberately a single-layer
no-autodiff kernel; it does not expose optimizer moments or a general
multi-layer trainer.

The ordinary GNU build links an unavailable stub and returns
`FORTNUM_NOT_IMPLEMENTED`; it never executes a CPU fallback. The plan exposes
no HVP or optimizer state. The JVP and VJP graphs are resident for the eight
CUDA activation codes, with independent CPU recurrence/oracle checks; the CPU
MLP path also covers stable sigmoid and Mish. The MSE update has an
independent tanh loss/gradient/parameter oracle. HVP and full MLP training
remain on the FortAD/FortSym reference path until their complete resident
graphs are linked. See
[`docs/CUDA_DENSE_PLAN.md`](CUDA_DENSE_PLAN.md) and the independent
`test/run_cuda_dense_plan.sh` gate for the ABI layout, finite-input refusal,
eight-activation value/JVP/VJP oracle, MSE update, transfer counters, and
repeated resident-batch evidence.

| Type | Value or prediction | JVP | VJP or gradient | HVP |
| --- | --- | --- | --- | --- |
| `linear_regression_t` | `predict` | Free `linear_predict_jvp` | Free `linear_predict_vjp` | No |
| `ridge_regression_t` | Weighted `predict` | Packed-parameter and continuous-input JVP | Packed-parameter and continuous-input VJP | No |
| `elastic_net_regression_t` | Weighted `predict` | Packed-parameter and continuous-input JVP | Packed-parameter and continuous-input VJP | No |
| `linear_svr_regression_t` | Weighted epsilon-insensitive `predict` | Packed-parameter and continuous-input JVP | Packed-parameter and continuous-input VJP; objective value/gradient | No |
| `radius_neighbors_regressor_t` | Closed-radius scalar weighted average | Refused across discrete neighbor-selection boundaries | Refused across discrete neighbor-selection boundaries | No |
| `radius_neighbors_multioutput_regressor_t` | Closed-radius multi-output weighted average | Zero away from radius boundaries; boundary refusal | Zero away from radius boundaries; boundary refusal | No |
| `glm_regression_t` | Weighted positive-response Poisson/Gamma log-link `predict` | Packed-parameter and continuous-input JVP | Packed-parameter and continuous-input VJP; value/gradient objective | No |
| `pca_t` | Centered projection and reconstruction | Input JVP for a fixed fitted state | Input VJP for a fixed fitted state | Fit-time SVD derivatives are not exposed |
| `linear_autoencoder_t` | Tied centered linear encode/decode/reconstruct, initialized from PCA | Input JVP for encode and reconstruction | No parameter VJP (weights are fixed PCA state) | No |
| `kmeans_t` | Deterministic dense seeded k-means labels, centers, Euclidean distances, and inertia | Fixed-center input JVP away from zero distances | Fixed-center input VJP away from zero distances | Fit/assignment is discrete; zero-distance derivatives and CUDA are typed refusals |
| `logistic_regression_t` | Decision score and probabilities | Parameter/input JVP, probability JVP | Parameter/input VJP, probability VJP | No |
| `linear_svm_classifier_t` | Signed decision score and labels | Parameter/input JVP away from fit-time boundaries | Parameter/input VJP; hinge objective value/gradient | No |
| `one_class_svm_t` | RBF nu-SVM signed anomaly score and ±1 labels | Fixed-state continuous-input JVP | Fixed-state continuous-input VJP | No |
| `softmax_regression_t` | Multiclass decision scores and probabilities | Parameter/input JVP, probability JVP | Parameter/input VJP, probability VJP | No |
| `softmax_training_objective_t` | Weighted multiclass cross-entropy + feature L2 (or positive log-L2) | Packed parameter/hyperparameter JVP | Packed parameter/hyperparameter VJP and gradient | Exact joint parameter/hyperparameter HVP |
| `gaussian_naive_bayes_t` | Log probabilities and probabilities | Input and packed-parameter JVP | Input and packed-parameter VJP | No |
| `bernoulli_naive_bayes_t` | Log probabilities and probabilities | Input and packed-parameter JVP | Input and packed-parameter VJP | No |
| `multinomial_naive_bayes_t` | Log probabilities and probabilities | Input and packed-parameter JVP | Input and packed-parameter VJP | No |
| `complement_naive_bayes_t` | Log probabilities and probabilities | Input and packed-parameter JVP | Input and packed-parameter VJP | No |
| `lda_classifier_t` | Gaussian log probabilities, probabilities, and labels | Input and packed-parameter JVP | Input and packed-parameter VJP | No |
| `qda_classifier_t` | Class-specific Gaussian log probabilities, probabilities, and labels | Input and packed-parameter JVP | Input and packed-parameter VJP | No |
| `multilabel_logistic_classifier_t` | Independent positive probabilities for an indicator matrix | Input and packed-parameter JVP | Input and packed-parameter VJP | No |
| `classifier_chain_t` | Sequential binary logistic heads with soft chain probabilities and arbitrary per-output label pairs | Input and packed-parameter JVP | Input and packed-parameter VJP | No |
| `ordinal_logistic_classifier_t` | Ordered cumulative-logit probabilities and labels | Input and packed-parameter JVP | Input and packed-parameter VJP | No |
| `basis_map_t` | `evaluate` | Parameters and inputs | Parameters and inputs | Analytic for polynomial/Fourier/radial/spline; callback maps refuse |
| `one_hot_encoder_t` | Dense one-hot `transform` | Refused: integer categories have no canonical tangent space | Refused: integer categories have no canonical cotangent space | No |
| `mlp_t` | `predict` | Parameters and inputs | Parameters and inputs | Weighted-output HVP |
| `mlp_last_layer_gp_initializer_t` | Finite-feature GP/NTK last-layer `fit`, `fit_apply`, and `predict` | Fixed-feature regularization JVP; named hyperparameter metadata | No VJP (the exposed product is a scalar regularization JVP) | Resident CUDA fit/predict/apply/JVP are typed `FORTNUM_NOT_IMPLEMENTED` refusals |
| `mlp_classifier_t` | Logits, probabilities, and labels | Parameter/input JVP, probability JVP, fixed-input probability-parameter JVP | Parameter/input VJP, probability VJP, fixed-input probability-parameter VJP | No |
| `mlp_classifier_training_objective_t` | Weighted multiclass cross-entropy or focal-softmax (`focal_gamma`) + optional L2 | Packed network/L2 JVP | Packed network/L2 gradient and scalar VJP | Exact joint network/L2 HVP |
| `mlp_calibrated_classifier_t` | MLP logits with binary sigmoid/temperature/isotonic or multiclass temperature probabilities and labels | Exact joint network/input plus smooth calibration JVP; isotonic active-set refusal | Exact joint network/input plus smooth calibration VJP; isotonic active-set refusal | No |
| `mlp_ordinal_classifier_t` | Ordered cumulative-logit neural score, probabilities, and labels | Packed network/threshold and input JVP | Packed network/threshold and input VJP | No |
| `mlp_binary_classifier_t` | One-logit sigmoid probabilities and binary labels | Parameter/input JVP, probability JVP | Parameter/input VJP, probability VJP; weighted BCE gradient | Exact weighted BCE parameter HVP |
| `mlp_multilabel_classifier_t` | Independent sigmoid probabilities and indicator labels | Packed parameter/input JVP, probability JVP | Packed parameter/input VJP, probability VJP; weighted mean BCE gradient | Exact weighted mean BCE parameter HVP |
| `mlp_multilabel_training_objective_t` | Weighted multilabel BCE plus direct L2 or positive log-L2 coordinate | Packed network/L2 JVP | Packed network/L2 gradient and scalar VJP | Exact joint network/L2 HVP |
| `mlp_chain_t` | Sequential composition of named MLP stages | Packed all-stage parameters and inputs | Packed all-stage parameters and inputs | Differentiated reverse chain rule for parameters and inputs |
| `mlp_training_objective_t` | MSE+L2 scalar objective | Packed network/L2 JVP | Packed network/L2 gradient and scalar VJP | Joint network/L2 HVP |
| `mlp_poisson_training_objective_t` | Weighted one-output Poisson NLL in log-rate coordinates with optional L2 | Packed network/L2 JVP | Packed network/L2 gradient and scalar VJP | Exact joint network/L2 HVP |
| `mlp_grouped_training_objective_t` | MSE with one positive log-L2 coefficient per named parameter range | Packed network/log-L2 JVP | Packed network/log-L2 gradient and scalar VJP | Exact mixed network/log-L2 HVP |
| `mlp_chain_objective_t` | MSE+L2 scalar objective over a sequential MLP tree | Packed all-stage/L2 JVP | Packed all-stage/L2 gradient and scalar VJP | Exact all-stage/L2 HVP |
| `mlp_hypergradient_objective_t` | Validation MSE after fixed full-batch GD trajectory | Outer `[log(learning_rate),log(l2)]` JVP | Exact trajectory value gradient and scalar VJP | Reverse trajectory products; inner MLP HVP |
| `mlp_adamw_full_hypergradient_objective_t` | Validation MSE after fixed full-batch AdamW trajectory | Packed `[log(learning_rate),log(l2),log(weight_decay),logit(beta1),logit(beta2)]` JVP | Exact trajectory value gradient and scalar VJP | Forward state sensitivities through moments, bias correction, and decoupled decay |
| `mlp_adam_hypergradient_objective_t` | Validation MSE after fixed full-batch coupled-L2 Adam trajectory | Packed `[log(learning_rate),log(l2),logit(beta1),logit(beta2)]` JVP | Exact trajectory value gradient and scalar VJP | Forward state sensitivities through coupled loss, moments, and bias correction |
| `mlp_rmsprop_hypergradient_objective_t` | Validation MSE after fixed full-batch RMSprop trajectory | Packed `[log(learning_rate),log(l2),decay,log(epsilon),momentum]` JVP | Exact trajectory value gradient and scalar VJP | Forward state sensitivities; inner MLP HVP |
| `mlp_adagrad_hypergradient_objective_t` | Validation MSE after fixed full-batch Adagrad trajectory | Packed `[log(learning_rate),log(l2),log(epsilon)]` JVP | Exact trajectory value gradient and scalar VJP | Forward accumulated-square sensitivities; inner MLP HVP |
| `mlp_adafactor_hypergradient_objective_t` | Validation MSE after fixed full-batch unfactored Adafactor trajectory | Packed `[log(learning_rate),log(l2),decay,log(epsilon),log(clip_threshold)]` JVP | Exact trajectory value gradient and scalar VJP | Forward second-moment, update-RMS clipping, and denominator sensitivities; active-set and discrete branches refuse |
| `adafactor_factored_t` | Layout-aware matrix-factorized Adafactor with vector fallback | Parameter update | Dense second-moment inspection | Explicit row/column state for matrix blocks; CPU recurrence; formatted and in-memory schema-11 checkpoint migration; CUDA remains a typed refusal |
| `mlp_schedule_hypergradient_objective_t` | Validation MSE after a typed scheduled full-batch trajectory | Packed `[log(base_rate),log(l2),logit(min_fraction),logit(decay_factor)]` JVP, or `[log(base_rate),log(l2),log(peak_fraction),log(final_fraction)]` for one-cycle | Exact schedule/trajectory value gradient and scalar VJP | Exact affine outer HVP for stateless schedules; nonlinear/plateau/CUDA boundaries are typed |
| `mlp_radam_schedule_hypergradient_objective_t` | Validation MSE after a typed scheduled full-batch RAdam trajectory | Packed base-rate/L2/beta/epsilon/schedule JVP | Exact trajectory value gradient and scalar VJP | Moment, bias-correction, rectification, and schedule sensitivities; outer HVP/CUDA refusal |
| `mlp_minibatch_hypergradient_objective_t` | Validation MSE after a fixed seeded mini-batch SGD trajectory | Packed `[log(learning_rate),log(l2)]` JVP | Exact batch-cursor trajectory value gradient and scalar VJP | Per-batch MLP HVP; outer hyper-HVP is a typed refusal |
| `mlp_minibatch_adam_hypergradient_objective_t` | Validation MSE after a fixed seeded mini-batch coupled-L2 Adam trajectory | Packed `[log(learning_rate),log(l2)]` JVP | Exact batch-cursor trajectory value gradient and scalar VJP | Forward parameter/moment/bias-correction sensitivities; outer hyper-HVP is a typed refusal |
| `trainer_t` | Any `fortopt_objective::objective_t` with explicit full-batch training state | Optimizer updates are stateful; the objective supplies exact products | The same objective value/gradient callback is used for every update | L-BFGS-B consumes the objective gradient; no hidden HVP or finite-difference fallback |
| `bnn_t` | `elbo` | ELBO | ELBO | ELBO |
| `vae_t` | `elbo`, `reconstruct` | No | ELBO gradient | No |
| `rnn_t` | `forward`, squared-error `loss` | No | Loss gradient by BPTT | No |
| `kernel_t` | Scalar value and matrix | Parameter JVP | Parameter VJP | Parameter HVP |
| `xgboost_t` | Squared/squared-log (RMSLE)/logistic/Poisson/fixed-shape Gamma/Tweedie/Huber/quantile/absolute/rank:pairwise margins, predictions, additive tree contributions, fitted-prefix slicing, bounded ordered-gradient integer categorical partitions, and packed fixed-structure base/leaf coordinates | Fixed-tree input JVP away from split boundaries; categorical models refuse discrete tangents; raw-margin leaf-coordinate JVP | Fixed-tree input VJP away from split boundaries; categorical models refuse discrete cotangents; raw-margin leaf-coordinate VJP | No |
| `xgboost_classifier_t` | Binary integer labels, logistic `(n,2)` probabilities and stable log probabilities, staged margins, feature diagnostics, and categorical/interaction metadata | Fixed-tree probability/log-probability/input JVP away from split boundaries | Fixed-tree probability/log-probability/input VJP away from split boundaries | No |
| `lightgbm_t` | Weighted numeric regression/binary logistic histogram boosting with deterministic globally best-leaf growth, GOSS top/other-rate gradient/Hessian sampling, seeded DART/dropout with persisted tree-normalisation scales, and packed fixed-structure base/leaf coordinates | Fixed-tree input JVP away from split boundaries; raw-margin leaf-coordinate JVP | Fixed-tree input VJP away from split boundaries; raw-margin leaf-coordinate VJP | No |
| `lightgbm_multiclass_t` | Sorted-integer-label one-vs-rest LightGBM-style binary logistic children with normalized final/staged probabilities, raw margins, weighted validation best-prefix metadata, and transactional fit | Fixed-tree normalized probability/input JVP away from split boundaries | Fixed-tree normalized probability/input VJP away from split boundaries | No |
| `random_forest_classifier_t` | Bootstrap-ensemble probabilities/labels plus transactional OOB decision probabilities, OOB accuracy, coverage, bootstrap-inclusion audit state, and deterministic fixed-state accuracy permutation importance | Refused: split routing and permutation membership are discrete | Refused: split routing and permutation membership are discrete | CPU OOB/permutation diagnostics; CUDA returns typed `FORTNUM_NOT_IMPLEMENTED` |
| `extra_trees_classifier_t` | Randomized-threshold ensemble probabilities and labels | Refused: split routing is discrete | Refused: split routing is discrete | No |
| `bagging_classifier_t` | Seeded bootstrap or without-replacement CART probabilities and labels | Refused: split routing is discrete | Refused: split routing is discrete | No |
| `gp_regression_t` | Mean, variance, LML | Prediction and LML parameters | Prediction and LML parameters | Mean and LML parameters |
| `gp_derivative_regression_t` | Mean, variance, and LML | Prediction and LML parameter JVP | Prediction parameter VJP and analytic LML hyperparameter gradient | Directional HVP (finite difference of the analytic gradient) |
| `second_derivative_gp_t` | Exact scalar 1-D RBF/Matérn-5/2 GP over mixed value/first/second-derivative rows, plus RBF third-derivative rows; latent joint covariance and packed likelihood state | Query-coordinate JVP; RBF likelihood JVP; selected-CPU device dispatch | Query-coordinate VJP; likelihood VJP; selected-CPU device dispatch | RBF order >3, Matérn-5/2 order >2, Matérn-5/2 fifth derivative at coincidence, Matérn parameter products, and CUDA prediction/covariance/product requests are typed refusals |
| `gp_classification_t` | Latent, observed, and log-observed probabilities; fixed-state kernel parameter setter; implicit-mode kernel hyperparameter HVP | Input and fixed-state kernel-parameter JVP for probabilities and log probabilities | Input and fixed-state kernel-parameter VJP for probabilities and log probabilities; Laplace-mode kernel hyperparameter gradient | Implicit-mode hyperparameter HVP on CPU; typed CUDA refusal |
| `gp_multiclass_classification_t` | Latent one-vs-rest margins and normalized observed probabilities | Input and packed fixed-state kernel-parameter JVPs for margins and probabilities | Input and packed fixed-state kernel-parameter VJPs for margins and probabilities; packed one-vs-rest Laplace-mode kernel hyperparameter gradient | No |
| `gp_multilabel_classification_t` | Independent binary Laplace-GP probabilities and indicator labels | Input and packed per-label fixed-state kernel-parameter JVPs for latent/probability outputs | Input and packed per-label fixed-state kernel-parameter VJPs; concatenated Laplace-mode kernel hyperparameter gradient | No |
| `multi_output_gp_t` | Correlated mean and LML; prior covariance; batched `(batch,query,output)` prediction | Packed kernel/log-noise/output-major W/independent posterior-mean and prior-covariance JVP; query-input and batch-query JVP | Fitted posterior-mean and prior-covariance parameter VJP; query-input and batch-query VJP | No |
| Approximate GP types | Mean, variance, or ELBO as listed below | No | No | No |

`xgboost_t%save_text(path,status)` and `load_text(path,status)` provide the
portable persistence boundary for fitted tree ensembles. The versioned
`FORTML_XGBOOST_TEXT` schema records objective and fit options, base margin,
monotone constraints, learned missing-value routing, every live node array,
and validation/early-stopping diagnostics. Values are named records with
17-significant-digit real output; loaders validate schema order, finite
values, child topology, and complete EOF before replacing the destination.
Truncated, unknown, duplicate, or structurally unsafe records return
`FORTNUM_DOMAIN_ERROR` and leave an existing destination model unchanged.

`parameter_products_t` gives `mlp_t` and fitted `gp_regression_t` one common
packed value/JVP/VJP/HVP interface. Inputs remain fixed in that interface.

## `fortml_trainer`

`trainer_t` is the model-agnostic training state machine for objectives that
already implement `fortopt_objective::objective_t`. This is the shared seam
for basis/pipeline, linear, GP, classifier, neural-tree, and future
physics-informed adapters; it does not reinterpret a model-specific callback
or hide a data transfer. `initialize(objective,initial,status,options)`
validates a finite packed parameter vector and evaluates the initial scalar
value/gradient. `step(status)` performs one deterministic full-batch update;
`fit(status)` runs to the declared limit or convergence. The available
optimizers are `FORTML_TRAIN_SGD`, `FORTML_TRAIN_ADAM`, `FORTML_TRAIN_ADAMW`,
`FORTML_TRAIN_ADAGRAD`, `FORTML_TRAIN_RMSPROP`, and
`FORTML_TRAIN_ADAFACTOR`, and `FORTML_TRAIN_LION`; `FORTML_TRAIN_LBFGSB` is a
fit-level bounded solve.

`FORTML_TRAIN_ADAFACTOR` uses a deterministic unfactored vector state: the
exponentially averaged squared gradient is clipped by its update RMS and the
epsilon-stabilized update is applied to the packed parameter vector. Set
`adafactor_decay`, `adafactor_clip_threshold`, `adafactor_relative_step`, and
`adafactor_scale_parameter` in `trainer_options_t` to control this recurrence.
Because this API has no matrix-layout metadata, true row/column factored state
is intentionally not claimed; a layout-aware adapter remains an explicit
extension. The second moment and step counter are checkpointed.

`trainer_options_t` owns optimizer coefficients, gradient clipping, optional
parameter bounds, EMA decay, convergence tolerances, an optional typed step
callback, and an optional stateless `mlp_learning_rate_schedule_t`. The
schedule is evaluated at each streaming update for SGD, Adam, AdamW, Adagrad,
RMSprop, Adafactor, and Lion without resetting optimizer moments; invalid
plateau or L-BFGS-B combinations return a typed refusal. `trainer_state_t`
reports counters, objective and gradient histories, clipping, convergence,
final parameters, and EMA parameters.
`state_copy()` is an in-memory checkpoint and `clone(copy,status)` copies the
complete optimizer and objective state, including moments and L-BFGS-B
parameters, so an interrupted run can resume without process-global state.
`parameters()` and `value_gradient()` return copies/products for deployment and
outer hyperparameter search. The core is CPU objective execution; a device
adapter must supply a resident objective or return `FORTNUM_NOT_IMPLEMENTED`.
Mini-batching, validation streams, and stochastic data-loader state belong to
the owning objective/trainer adapter and are never silently emulated here.

`trainer_t%save_checkpoint(path,status)` writes a versioned,
compiler-independent formatted-text snapshot containing optimizer options,
parameters, EMA values, objective/history state, bounds, and the complete
SGD/Adam/AdamW/Adagrad/RMSprop/Adafactor/Lion recurrence. `load_checkpoint(path,status)` is
transactional: it requires an initialized destination with the same packed
dimension, validates schema/order/counts/finite values, and refuses truncated,
unknown, extra, or incompatible records without changing the destination.
Schema version 5 is a deliberate clean break that records the typed schedule
configuration and rejects older or newer trainer snapshots rather than
silently changing a trajectory.
Procedure callbacks and objective closures remain process-local and must be
attached by the caller; L-BFGS-B has no resumable streaming state in this
trainer format.

`mlp_t%parameter_layout()` exposes the same packed vector as a deterministic
named parameter tree. Each dense layer contributes `layer_n.weight` followed by
`layer_n.bias`; every descriptor reports its one-based `first:last` range,
matrix shape, `kind`, and trainable/buffer role. `parameter_block_count()`
reports the number of descriptors, and `parameter_range(name,first,last,found)`
resolves a stable path without exposing private layer storage. MLPs currently
have no mutable buffers, so `is_buffer` is false for every descriptor; the
explicit role field keeps checkpoint, optimizer, and pipeline selectors
forward-compatible with future non-trainable state.

## Regression and basis maps

### `fortml_linear_autoencoder`

`linear_autoencoder_t` is the explicit linear-optimum initialization seam for
future nonlinear autoencoders. `fit(x,status[,n_components])` fits centered
PCA and stores the loading matrix as an encoder and its transpose as a tied
decoder. `initialize_from_pca(pca,status)` performs the same copy from an
already fitted `pca_t`; it does not claim that a finite nonlinear network or
an NNGP posterior is identical to this optimum.

`encode`, `decode`, and `reconstruct` use row-oriented sample semantics and
preserve the PCA center. `encode_jvp` and `reconstruct_jvp` are exact for a
fixed fitted state. `encoder_weights`, `decoder_weights`, and `mean` return
copies for inspection. `device_supported(FORTML_DEVICE_CPU)` is true after
initialization; CUDA is explicitly false until a resident matrix-product
lowering is linked, so callers cannot count a host fallback as GPU execution.

### `fortml_linear_regression`

`linear_regression_t%fit(x,y,status[,ridge,fit_intercept])` accepts a vector or
matrix target. It solves the dense least-squares problem by SVD. `ridge` must
be nonnegative and does not penalize the intercept. The default is no ridge
penalty with an intercept.

`model%predict(x,y,status)` has vector and matrix forms. The fitted coefficient
array has shape `(n_features+1,n_outputs)`. Row 1 is the intercept slot and is
zero when `fit_intercept=.false.`.

The free procedures
`linear_predict_jvp(coef,x,dcoef,dx,y,dy[,fit_intercept])` and
`linear_predict_vjp(coef,x,u,coef_bar,x_bar[,fit_intercept])` operate on an
explicit coefficient array. They do not return a status object, so all arrays
must have the model shapes described above.

### `fortml_ridge_regression`

`ridge_regression_t%fit(x,y,status[,alpha,fit_intercept,sample_weight])`
fits a weighted dense ridge model for a vector or matrix target. Inputs use
`(n_samples,n_features)` row semantics and targets use
`(n_samples,n_outputs)`; the vector overload accepts a one-dimensional target.
The objective is the weighted squared residual plus
`alpha*sum(feature_coefficients**2)`. `alpha` must be finite and
nonnegative. An intercept is fitted by default and is never regularized.
`sample_weight` must be finite, nonnegative, have one entry per sample, and
have positive total mass. The solve is an SVD least-squares problem with
square-root ridge rows, so rank-deficient designs have a deterministic
minimum-norm solution. The estimator stores no training rows after `fit`.

`predict(x,y,status)` has vector and matrix forms. `coefficients()` returns a
copy whose first row is the intercept when `fit_intercept()` is true;
`parameters()` flattens this matrix in Fortran column-major order, and
`set_parameters(values,status)` updates the fitted prediction state after
checking the packed size and finiteness. `parameter_count()`,
`feature_count()`, `output_count()`, `regularization()`, `fit_intercept()`,
and `fitted()` expose the corresponding metadata.

`predict_jvp(x,theta_dot,x_dot,y,y_dot,status)` (also `jvp`) and
`predict_vjp(x,y_bar,theta_bar,x_bar,status)` (also `vjp`) provide exact
products with respect to the packed coefficients and continuous input rows.
They hold the fitted SVD state, `alpha`, and sample weights fixed. Tangents,
cotangents, inputs, and outputs must have finite compatible shapes. There is
no derivative through the SVD fit, rank decisions, sample-weight support, or
`alpha` hyperparameter; those requests are an explicit boundary of this
estimator rather than a silent host fallback. Unfitted calls, malformed
packs, nonfinite arrays, nonpositive weight mass, and shape mismatches return
`FORTNUM_DOMAIN_ERROR`.

### `fortml_elastic_net_regression`

`elastic_net_regression_t%fit(x,y,status[,alpha,l1_ratio,fit_intercept,
sample_weight,max_iterations,tolerance])` fits weighted multi-output elastic
net regression by deterministic coordinate descent. The normalized objective is
`0.5*sum(w*(y-pred)**2)/sum(w) + alpha*(l1_ratio*L1 +
0.5*(1-l1_ratio)*L2)`, with the intercept excluded from both penalties.
`alpha` is nonnegative and `l1_ratio` lies in `[0,1]`; `l1_ratio=1` is lasso
and `l1_ratio=0` is ridge-like coordinate descent. Finite nonnegative sample
weights must have positive mass. Nonconvergence, malformed options, and
nonfinite data return typed status errors.

`predict`, `coefficients`, `parameters`, `set_parameters`, and the feature,
output, and solver metadata methods expose the fitted state. `jvp`/`vjp`
provide exact fixed-state products over packed coefficients and continuous
inputs. The nonsmooth coordinate fit, active-set decisions, solver tolerance,
and regularization hyperparameters are not differentiated; those fit-time
boundaries are explicit rather than hidden finite-difference fallbacks.
`device_supported(kind)` reports CPU support for a fitted model and refuses
CUDA until a resident elastic-net prediction kernel exists. `predict_device`
dispatches only to a selected CPU context; a selected CUDA context returns
`FORTNUM_NOT_IMPLEMENTED` rather than silently executing on the host.

### `fortml_glm_regression`

`glm_regression_t%fit(x,y,status[,family,alpha,fit_intercept,sample_weight,
max_iterations,tolerance,dispersion,lower_bound,upper_bound])` fits a weighted
positive-response generalized linear model. `family` is
`GLM_FAMILY_POISSON` (nonnegative targets) or `GLM_FAMILY_GAMMA` (strictly
positive targets); both use the stable canonical log link
`GLM_LINK_LOG`. The normalized negative log likelihood is

```text
Poisson:  sum_i w_i [ exp(eta_i) - y_i eta_i ] / sum_i w_i
Gamma:    sum_i w_i [ y_i exp(-eta_i) + eta_i ] / (dispersion sum_i w_i)
```

where `eta` is the intercept-plus-feature linear predictor. `alpha` adds an
L2 penalty only to feature coefficients. Fits use bounded FortOpt L-BFGS-B;
the default finite coefficient bounds are `[-30,30]` to keep the exponential
link in a numerically safe domain and can be replaced by finite ordered
`lower_bound`/`upper_bound` values. `dispersion` is positive and affects only
the Gamma likelihood. Sample weights must be finite, nonnegative, and have
positive mass. The fit-time optimizer and stopping decisions are discrete;
the returned products hold the fitted coefficient state fixed.

`predict(x,y,status)` has vector and matrix forms and returns strictly
positive means. `coefficients()` stores an intercept in row 1 followed by
feature rows; `parameters()` flattens this matrix in Fortran column-major
order, and `set_parameters` validates the configured bounds. Metadata methods
include `family`, `link`, `regularization`, `dispersion`, `feature_count`,
`output_count`, `fit_intercept`, `lower_bound`, `upper_bound`, and `fitted`.

`predict_jvp`/`jvp` and `predict_vjp`/`vjp` are exact analytic log-link products
over packed coefficients and continuous inputs. `objective_value_gradient`
exposes the weighted likelihood and its analytic coefficient gradient for
hyperparameter/search adapters; optional `alpha_gradient` and
`dispersion_gradient` outputs provide exact L2 and Gamma-dispersion
derivatives. It does not finite-difference the objective.
`device_supported(kind)` reports CPU support for a fitted model. A selected
CUDA context is refused with `FORTNUM_NOT_IMPLEMENTED` until a resident GLM
kernel exists, so the device API never hides a host fallback.

### `fortml_pca`

`pca_t%fit(x,status[,n_components,whiten])` fits centered dense PCA by a thin
LAPACK SVD. Inputs use `(n_samples,n_features)` row semantics and must be
finite with at least two rows. `n_components` defaults to
`min(n_samples,n_features)` and must be a positive integer no larger than that
bound. Component rows are sign-flipped deterministically so their largest
magnitude loading is positive. `components()`, `mean()`, `singular_values()`,
`explained_variance()`, and `explained_variance_ratio()` return copies of the
fitted state; `n_components()`, `feature_count()`, `sample_count()`,
`whiten()`, and `fitted()` expose metadata.

`transform(x,z,status)` centers and projects rows. With `whiten=.true.`, each
coordinate is divided by the square root of its sample explained variance.
`inverse_transform(z,x,status)` applies the matching unwhitening and adds the
fitted mean. `fit_transform` combines fitting and projection. Fixed-state
`transform_jvp(x_dot,z_dot,status)` and
`transform_vjp(z_bar,x_bar,status)` are exact linear input products. The
fit-time SVD, sign choices, and rank changes are discrete boundaries and are
not differentiated; invalid shapes, nonfinite arrays, unfitted calls, and
one-row fits return a domain status.

### `fortml_kmeans`

`kmeans_t%fit(x,status[,n_clusters,max_iter,tolerance,initialization_seed,
device_kind])` fits dense row-oriented samples with deterministic Lloyd
iterations. The defaults are eight clusters, 300 iterations, tolerance
`1e-4`, and seed `1`. Initialization takes a cyclic seeded sample of distinct
rows (seed `s` starts at row `modulo(s-1,n_samples)+1`); assignment ties use
the lowest cluster index. This is an explicit reproducible baseline, not a
claim of scikit-learn's k-means++ initialization. Empty clusters are reported
as `FORTNUM_CONVERGENCE_ERROR` and are never silently reseeded. Hitting the
iteration limit before the requested tolerance also returns that status while
retaining the last finite fit.

`predict` returns integer labels, `transform` returns Euclidean distances to
each fitted center, and `cluster_centers`, `labels`, `inertia`, `n_iter`, and
the remaining metadata accessors return copies or scalar fit state.
`fit_transform` combines fitting and distance transformation. Fixed-center
`transform_jvp` and `transform_vjp` provide exact input products when every
distance is nonzero; a zero distance is a typed domain refusal because the
Euclidean norm is nonsmooth there. Labels and fit-time assignment are discrete
and have no derivative contract. Inputs must be finite nonempty dense arrays,
and CUDA/device-resident fit, prediction, and transformation return
`FORTNUM_NOT_IMPLEMENTED` until a resident kernel is linked; no host fallback
is hidden behind the device argument.

### `fortml_logistic_regression`

`logistic_regression_t%fit(x,labels,status[,l2,fit_intercept,max_iterations,
tolerance,sample_weight,class_weight])` fits a binary logistic model with a stable weighted
cross-entropy objective and L2 penalty on the feature coefficients. The fit is
delegated to `fortopt_lbfgsb`. Labels are arbitrary integers, but exactly two
distinct values must occur. They are stored in ascending order and define the
two probability columns. A nonnegative `sample_weight` vector changes the
cross-entropy reduction to a positive-weight-mass average while leaving the
feature penalty unchanged. A positive `class_weight` vector of length two is
ordered by the stored classes and multiplies the sample weights before that
reduction.

`decision_function(x,scores,status)` returns one logit per row.
`predict_proba(x,probabilities,status)` returns `(n_samples,2)` with columns
`classes()(1)` and `classes()(2)`. `predict(x,labels,status)` uses a zero-logit
tie rule that selects the second class. `coefficients()`, `intercept_value()`,
`regularization()`, `set_regularization(value,status)`, `classes()`,
`feature_count()`, and `fitted()` expose or update the fitted state.
Three-class data, one-class data, nonfinite inputs, invalid penalties, and
shape mismatches return a domain or convergence status.

`parameter_count()`, `parameters()`, and `set_parameters(values,status)` expose
the packed coefficient-then-intercept vector (the intercept is omitted when
`fit_intercept=.false.`). `jvp(x,theta_dot,x_dot,scores,scores_dot,status)`
and `vjp(x,scores_bar,theta_bar,x_bar,status)` (also available as the explicit
`decision_function_jvp` and `decision_function_vjp` names) differentiate
scores with respect to both packed parameters and row inputs. `predict_proba_jvp`
and `predict_proba_vjp` provide the corresponding
stable probability products. Tangent and cotangent arrays must be finite and
shape-compatible. Unfitted models, malformed packs, and nonfinite inputs are
refused. There is intentionally no HVP until a second-order classifier
contract is added.

### `fortml_linear_svm_classifier`

`linear_svm_classifier_t%fit(x,labels,status[,l2,fit_intercept,loss,
max_iterations,tolerance,sample_weight])` fits a weighted dense primal linear
support-vector classifier through FortOpt L-BFGS-B. Labels may be any two
distinct integers; `classes()` stores them in ascending order and encodes them
as `-1` and `+1` for the margin objective. The default
`SVM_LOSS_SQUARED_HINGE` objective is the weighted mean of
`max(0,1-y*score)**2` plus feature-only L2 regularization. The ordinary
`SVM_LOSS_HINGE` loss is also available. Intercepts are not regularized.
For ordinary-hinge fitting, the FortOpt callback uses a deterministic tiny
epsilon-Huber continuation solely to make the nonsmooth Armijo line search
well-defined; `objective_value_gradient` remains the exact hinge objective.

`decision_function(x,scores,status)` returns the signed affine margin and
`predict(x,labels,status)` maps nonnegative scores to the second stored class
(the deterministic zero-margin tie rule). `coefficients()`, `intercept()`,
`regularization()`, `loss()`, `fit_intercept()`, `classes()`,
`feature_count()`, `parameter_count()`, `parameters()`, `set_parameters()`,
and `fitted()` expose the packed model state. The packed order is feature
coefficients followed by the intercept when enabled.

`decision_function_jvp(x,theta_dot,x_dot,scores,scores_dot,status)` and
`decision_function_vjp(x,scores_bar,theta_bar,x_bar,status)` (also available
as `jvp` and `vjp`) are exact products of the fixed fitted affine map. They
differentiate continuously with respect to both packed parameters and
continuous input rows; the discrete fit, label encoding, and hard prediction
are not differentiated. `objective_value_gradient(x,labels,theta,value,
gradient,status[,l2,fit_intercept,loss,sample_weight,l2_gradient])` exposes the
weighted hinge objective and feature-L2 gradient for FortOpt search. It returns
the exact `l2_gradient = 0.5*||theta_features||**2` when requested. An exact
ordinary-hinge margin (`y*score == 1`) is a split derivative and returns
`FORTNUM_NOT_IMPLEMENTED`; squared hinge has a continuous first derivative but
its second derivative changes at that boundary. Nonfinite data, malformed
weights, unsupported losses, and non-binary labels return a domain status.

`device_supported(FORTML_DEVICE_CPU)` is true for a fitted model.
`decision_function_device(device,x,scores,status)` and
`predict_device(device,x,labels,status)` dispatch the score and hard-label
surfaces separately. CUDA requests are intentionally typed
`FORTNUM_NOT_IMPLEMENTED` until a resident linear-SVM kernel is linked; no
host fallback is performed.

### `fortml_one_class_svm`

`one_class_svm_t%fit(x,status[,nu,gamma,max_iterations,tolerance])` fits a
dense RBF one-class SVM. The implementation solves the standard capped-simplex
dual with `0 <= alpha_i <= 1/(nu*n)` and `sum(alpha)=1` using a deterministic
projected-gradient iteration. `nu` must lie in `(0,1]`; `gamma` is a positive
RBF coefficient and defaults to `1/n_features`. The fitted offset is selected
from the free-support-vector KKT interval (or its deterministic midpoint when
all support weights are at a bound).

`decision_function(x,scores,status)` returns `sum_i alpha_i*K(x,x_i)-offset`.
`predict(x,labels,status)` maps nonnegative scores to `+1` and negative scores
to `-1`. `support_weights()`, `offset()`, `gamma()`, `nu()`,
`support_vector_count()`, `feature_count()`, `sample_count()`,
`iterations()`, and `fitted()` expose the state. Fixed-state RBF
`decision_function_jvp`/`decision_function_vjp` (also `jvp`/`vjp`) differentiate
continuous query inputs exactly. Fit-time active-set and hyperparameter
derivatives are not claimed. CPU dispatch is complete; CUDA score and label
requests return `FORTNUM_NOT_IMPLEMENTED` until a resident RBF reduction is
linked, with no hidden host fallback.

### `fortml_rbf_svm_classifier`

`rbf_svm_classifier_t%fit(x,labels,status[,c,gamma,max_iterations,tolerance,
sample_weight])` fits a dense binary RBF support-vector classifier. The finite
training rows are the explicit RBF feature basis. FortOpt L-BFGS-B minimizes
the deterministic weighted convex squared-hinge RKHS objective
`0.5*cvec^T*K*cvec + C/sum(w)*sum_i w_i*max(0,1-y_i*(K*cvec+b))**2`.
This is a primal-coordinate kernel expansion with exact finite-basis
semantics; it is deliberately named separately from `linear_svm_classifier_t`
and does not claim a general sparse/exact-dual SMO implementation. `C` and
`gamma` must be finite and positive. `sample_weight` is finite and
nonnegative with positive total mass. Labels may be any two distinct integers;
`classes()` stores the ascending pair and the zero-score tie goes to the
second class.

`decision_function(x,scores,status)` returns the signed RBF score and
`predict(x,labels,status)` applies that deterministic threshold. `predict_proba`
returns two columns in `classes()` order using the explicitly uncalibrated
sigmoid of the score (`p(positive)=sigmoid(score)`). `coefficients()` are the
training-basis expansion weights, `intercept()`, `gamma()`, `c_parameter()`,
`support_vector_count()`, `classes()`, `feature_count()`, `sample_count()`,
`iterations()`, and `fitted()` expose state. The packed `parameters()` order is
`[coefficients,intercept,log(gamma)]`; `set_parameters` updates only this
fixed-state prediction map and validates finite positive `gamma`.

`decision_function_jvp(x,theta_dot,x_dot,scores,scores_dot,status)` and
`decision_function_vjp(x,scores_bar,theta_bar,x_bar,status)` (also `jvp` and
`vjp`) are analytic fixed-state products with respect to packed coefficients,
intercept, log-gamma, and query inputs. `predict_proba_jvp` and
`predict_proba_vjp` apply the same map through the sigmoid. Fit-state,
active-set, hyperparameter-search, and hard-label derivatives are not exposed:
the squared-hinge margin changes curvature at one and hard prediction is
discrete. No finite-difference fallback is used. The four corresponding
`*_device` product methods dispatch exactly on CPU and return
`FORTNUM_NOT_IMPLEMENTED` for CUDA until resident RBF-SVM derivative kernels
are linked. `device_supported(CPU)` is true for a fitted model; all CUDA score,
label, probability, and derivative requests refuse explicitly with no host
fallback.

### `fortml_linear_svr`

`linear_svr_regression_t%fit(x,targets,status[,l2,epsilon,fit_intercept,loss,
max_iterations,tolerance,sample_weight])` fits a weighted dense primal linear
support-vector regressor through FortOpt L-BFGS-B. Targets are arbitrary finite
real values. The default `SVR_LOSS_SQUARED_EPSILON` objective is the weighted
mean of `max(0,abs(prediction-target)-epsilon)**2` plus feature-only L2
regularization; `SVR_LOSS_EPSILON` selects the ordinary epsilon-insensitive
loss. Epsilon and L2 are finite and nonnegative, and the effective sample
weight mass must be positive. Intercepts are not regularized.

`predict(x,targets,status)` and `decision_function(x,targets,status)` return
the same fitted affine prediction. `coefficients()`, `intercept()`,
`regularization()`, `epsilon()`, `loss()`, `fit_intercept()`,
`feature_count()`, `parameter_count()`, `parameters()`, `set_parameters()`,
and `fitted()` expose the packed coefficient-then-intercept state.
`predict_jvp(x,theta_dot,x_dot,targets,targets_dot,status)` and
`predict_vjp(x,targets_bar,theta_bar,x_bar,status)` (also available as `jvp`
and `vjp`) are exact fixed-state affine products with respect to packed
parameters and continuous input rows. Hard fit-time loss boundaries are not
differentiated.

`objective_value_gradient(x,targets,theta,value,gradient,status[,l2,
epsilon,fit_intercept,loss,sample_weight,l2_gradient,epsilon_gradient])`
exposes the weighted objective for FortOpt search. It returns the exact
feature-L2 derivative and, when requested, the epsilon derivative. For the
ordinary loss, an exact residual kink (`abs(prediction-target)==epsilon`) is a
split derivative and returns `FORTNUM_NOT_IMPLEMENTED`; the squared loss has a
continuous first derivative. Ordinary-loss fitting uses only a small C1
continuation inside its optimizer callback so the Armijo line search remains
well-defined; the public objective remains exact.

`device_supported(FORTML_DEVICE_CPU)` is true for a fitted model. CPU device
prediction dispatches to the same affine product. CUDA prediction is an
explicit `FORTNUM_NOT_IMPLEMENTED` refusal until a resident linear-SVR kernel
is linked; no host fallback is performed.

### `fortml_logistic_training`

`logistic_training_objective_t` packages a fitted
`logistic_regression_t` and a weighted binary data set as a FortOpt-ready
objective. `initialize(model,x,labels,l2,status[,optimize_l2,sample_weight,
class_weight])` validates the fitted class order and positive effective weight
mass, then exposes `parameters`, `parameter_count`, `value_gradient`, and
`hvp`. The objective uses the same positive-weight-mass cross-entropy and
feature-only L2 convention as `logistic_regression_t%fit`. With
`optimize_l2=.true.`, the packed vector appends a non-negative L2 coordinate.
Its gradient is the exact half squared feature norm and its HVP includes the
mixed parameter/L2 block. No finite differences are used by the adapter.

`fortopt(objective,status)` installs a context callback for
`fortopt_objective`. `logistic_optimize_lbfgsb(model,x,labels,options,result,
status[,sample_weight,class_weight])` owns the bounded FortOpt L-BFGS-B
lifecycle. `logistic_lbfgsb_options_t` supplies explicit bounds for model
parameters and, when requested, the L2 coordinate. The result reports
convergence, iterations, line-search evaluations, objective, gradient norm,
and final L2. The adapter requires a fitted binary model. Use `fit` first when
an initial parameter state is needed.

### `fortml_gaussian_naive_bayes`

`gaussian_naive_bayes_t%fit(x,labels,status[,var_smoothing,priors,
sample_weight,class_weight])` fits a finite-only Gaussian Naive Bayes model.
Rows are samples and columns are features. Integer labels are sorted
deterministically and define the probability-column order. Means and
population variances are weighted per class. `var_smoothing` adds a global
maximum-variance floor (and a machine-tiny floor for constant features), so
every fitted density is strictly positive. Optional positive `priors` are
normalized in sorted class order. Nonnegative sample weights and positive
sorted-class weights multiply before moments and empirical priors are formed.
every class must retain positive effective mass.

`predict_log_proba`, `predict_proba`, and `predict` use a shifted log-density
normalization and deterministic first-class ties. `classes`, `means`,
`variances`, `class_prior`, `weighted_class_counts`, `var_smoothing_value`,
`epsilon_value`, `feature_count`, `class_count`, and `fitted` expose the
fitted state. `parameter_count`, `parameters`, and `set_parameters` use the
packed order `[means, variances, priors]`, with each matrix in Fortran
column-major order. Priors are normalized by `set_parameters` and variances
must remain strictly positive.

The log-probability and probability methods expose input and packed-parameter
JVP/VJP products. Products differentiate through the log-softmax normalization
and the normalized prior block, and reject unfitted models, nonfinite values,
invalid shapes, nonpositive variances, and nonsensical prior mass.

### `fortml_bernoulli_naive_bayes`

`bernoulli_naive_bayes_t%fit(x,labels,status[,alpha,priors,sample_weight,
class_weight])` fits a finite Bernoulli Naive Bayes model. Features may be
binary or relaxed values in `[0,1]`; the relaxed extension makes the
prediction map smooth for input products. Classes are sorted integer labels.
`alpha` is a strictly positive Laplace/Beta smoothing mass, and optional
nonnegative sample weights and positive sorted-class weights form the effective
class moments and empirical priors. Explicit priors are normalized in sorted
class order.

`predict_log_proba`, `predict_proba`, and `predict` use shifted
log-probability normalization and deterministic first-class ties.
`feature_probabilities`, `class_prior`, `weighted_class_counts`, `classes`,
`alpha_value`, `parameter_count`, `parameters`, and `set_parameters` expose the
fitted state. The packed parameter order is feature probabilities in
Fortran column-major order followed by normalized class priors.

Input and packed-parameter JVP/VJP products cover both log probabilities and
probabilities, including the log-softmax and normalized-prior projections.
Nonfinite values, out-of-range features, nonpositive alpha, malformed packs,
unfitted models, and nonpositive effective class mass are refused explicitly.

### `fortml_multinomial_naive_bayes`

`multinomial_naive_bayes_t%fit(x,labels,status[,alpha,priors,sample_weight,
class_weight])` fits a finite Multinomial Naive Bayes model. Rows contain
nonnegative real-valued feature counts; real counts are accepted so the
prediction map remains smooth for input products. Classes are sorted integer
labels. `alpha` is strictly positive additive smoothing, and optional
nonnegative sample weights and positive sorted-class weights scale class and
feature masses before probabilities and empirical priors are formed. Explicit
priors are normalized in sorted class order.

`predict_log_proba`, `predict_proba`, and `predict` use stable shifted
log-likelihood normalization and deterministic first-class ties.
`feature_probabilities`, `feature_counts`, `weighted_class_counts`, `classes`,
`class_prior`, `alpha_value`, `parameter_count`, `parameters`, and
`set_parameters` expose the fitted state. Packed parameters contain the
feature-probability matrix in Fortran column-major order followed by the
normalized class-prior block.

Input and packed-parameter JVP/VJP products cover log probabilities and
probabilities, including the simplex and log-softmax projections. Negative or
nonfinite counts, nonpositive smoothing, malformed packs, nonpositive class
mass, unfitted models, and nonfinite prediction inputs are refused.

### `fortml_complement_naive_bayes`

`complement_naive_bayes_t%fit(x,labels,status[,alpha,priors,sample_weight,
class_weight,norm])` fits Complement Naive Bayes on finite nonnegative real
counts. Classes are sorted integer labels. For class `c`, the feature mass is
formed from all weighted rows outside `c`, smoothed by positive `alpha`, and
converted to the positive complement weight `-log(q)`. With `norm=.true.`,
each class's weights receive the second normalization used by ComplementNB.
Empirical or explicit sorted-class priors supply the differentiable intercept.

`predict_log_proba`, `predict_proba`, and `predict` use stable normalization and
deterministic first-class ties. `feature_probabilities`, `feature_weights`,
`feature_counts`, `weighted_class_counts`, `class_prior`, `alpha_value`,
`norm_enabled`, `parameter_count`, `parameters`, and `set_parameters` expose
the fitted state. Packed parameters contain the complement probability matrix
in Fortran column-major order followed by the normalized class-prior block.
Input and packed-parameter JVP/VJP products cover log probabilities and
probabilities. Negative/nonfinite counts, nonpositive smoothing, malformed
packs, nonpositive effective class mass, unfitted models, and nonfinite query
inputs are refused.

### `fortml_categorical_naive_bayes`

`categorical_naive_bayes_t%fit(x,labels,status[,alpha,priors,sample_weight,
class_weight,handle_unknown])` fits CategoricalNB on integer category codes.
Categories are sorted independently per feature and exposed through packed
`category_values` and one-based `category_offsets` metadata. Positive additive
smoothing, nonnegative sample weights, positive sorted-class weights, and
explicit normalized priors follow the same conventions as the other Naive
Bayes estimators. Unknown query categories raise a domain error by default;
`handle_unknown=.true.` skips their likelihood contribution.

`predict_log_proba`, `predict_proba`, and `predict` use stable normalization
and deterministic first-class ties. `classes`, `category_count`,
`category_values`, `category_offsets`, `class_prior`,
`weighted_class_counts`, `alpha_value`, `parameter_count`, and `fitted` expose
the fitted state. Category lookup is discrete, so `predict_proba_jvp` returns
an explicit `FORTNUM_NOT_IMPLEMENTED` status rather than a hidden finite
difference.

### `fortml_classification_metrics`

The shared metric procedures keep arbitrary integer labels and an explicit
class order: `classification_accuracy`,
`classification_balanced_accuracy`, `classification_confusion_matrix`, and
`classification_precision_recall_f1`. `classification_accuracy` accepts an
optional nonnegative sample-weight vector. `classification_log_loss` accepts a
row-wise probability matrix, matching integer labels and class order, and
optional sample weights. `classification_top_k_accuracy` uses deterministic
class-order tie breaking. `classification_brier_score` normalizes each
positive probability row before computing the multiclass squared error, and
`classification_binary_matthews` provides the weighted binary Matthews
correlation coefficient. `classification_calibration_error` and
`classification_maximum_calibration_error` use normalized row confidence,
deterministic first-maximum predictions, equal-width bins, and optional
sample weights. Empty bins do not contribute, and confidence one belongs to
the final bin. Shape, duplicate-class, unknown-label, nonfinite, negative-
weight, invalid-bin, and zero-weight-mass cases return a domain status.
`classification_reliability_diagram(probabilities, labels, classes, bins,
mean_confidence, mean_accuracy, bin_weight, status[, sample_weight])` exposes
the corresponding per-bin curve points for plotting or calibration-aware
model selection. Empty bins return zero confidence and accuracy with zero
weight; populated bins return weighted means using the same normalized
confidence and deterministic tie policy.

`classification_multilabel_precision_recall_f1` evaluates binary indicator
matrices with `CLASSIFICATION_AVERAGE_MICRO`,
`CLASSIFICATION_AVERAGE_MACRO`, or `CLASSIFICATION_AVERAGE_SAMPLES`. Micro
aggregates weighted TP/FP/FN across all labels; macro averages each label's
score equally, including labels with no positive support; samples computes
each row's score before the optional sample-weighted average. The optional
`zero_division` is `CLASSIFICATION_ZERO_DIVISION_ZERO` (the default) or
`CLASSIFICATION_ZERO_DIVISION_ONE` and is applied independently to precision,
recall, and F1 whenever its denominator is zero. Nonbinary indicators,
malformed weights, empty matrices, and invalid averaging or zero-division
codes return a domain status.

`classification_multilabel_precision_recall_fbeta` provides the same weighted
micro, macro, and samples reductions for any finite positive `beta`. The
returned F-beta score uses `(1+beta**2)*TP / ((1+beta**2)*TP + beta**2*FN +
FP)`, so `beta>1` emphasizes recall and `beta<1` emphasizes precision. The
per-label and per-row score is reduced before macro or samples averaging,
matching scikit-learn's F-beta semantics. The
`classification_multilabel_probability_fbeta` wrapper applies the explicit
`>= threshold` rule before evaluating the same contract. Nonpositive or
nonfinite beta, malformed probabilities, invalid thresholds, and unsupported
zero-division policies return a domain status. These are CPU reference
metrics; no implicit CUDA transfer or fallback is performed.

`classification_multilabel_probability_metrics` applies an explicit global
threshold to a probability matrix (`>= threshold` is positive; the default is
`0.5`) and delegates to the same averaging and zero-division contract. The
threshold must be finite and in `[0,1]`; this hard prediction path is not
differentiated. These metrics are CPU reference routines with no implicit
CUDA transfer or fallback.

`classification_roc_auc` computes binary ROC AUC from pairwise positive/negative
score comparisons, awarding half credit to exact score ties. Labels may be
arbitrary integers; the caller identifies the positive label explicitly.
`classification_roc_auc_ovr` applies the same contract one-vs-rest to a score
matrix and returns the macro average plus optional per-class values. Both
accept optional nonnegative sample weights and refuse nonfinite scores,
single-class support, malformed shapes, or zero pair mass. Their
`*_device` entry points preserve outputs and return `FORTNUM_NOT_IMPLEMENTED`
for CUDA until a resident ranking kernel is linked.

`classification_pr_auc` computes binary average precision (the step area under
the precision-recall curve) from descending score thresholds. Equal-score rows
are consumed as one threshold group, and optional sample weights contribute to
both precision and recall mass. `classification_pr_auc_ovr` applies the same
contract to each score column and returns the unweighted macro average plus
optional per-class values. Both refuse malformed/nonfinite inputs and require
positive weighted support for both the selected positive class and its negative
class. The corresponding `*_device` entry points preserve outputs and return
`FORTNUM_NOT_IMPLEMENTED` for CUDA until a resident ranking reduction is
linked; no host fallback is implicit.

`classification_multilabel_jaccard` evaluates intersection over union for
binary indicator matrices with micro, macro, or samples averaging. Its optional
sample weights are row weights, and `zero_division` explicitly selects the
value for empty unions. `classification_multilabel_hamming_loss` reports the
weighted fraction of mismatched indicators with the same averaging choices;
`classification_multilabel_hamming` is a short alias. Both reject empty,
malformed, nonbinary matrices, invalid averaging policies, and nonfinite or
zero-mass weights with a domain status. These reductions are CPU reference
routines until resident CUDA kernels are linked.

### `fortml_probability_calibration`

`probability_calibrator_t%fit(scores,labels,status[,options,sample_weight,state])`
fits a binary probability map from scalar decision scores and arbitrary integer
labels.  `probability_calibration_options_t%method` selects
`CALIBRATION_TEMPERATURE` (positive scalar temperature scaling),
`CALIBRATION_SIGMOID` (Platt scaling), or `CALIBRATION_ISOTONIC` (weighted
pool-adjacent-violators).  Temperature scaling maps a pre-oriented logit
`s` to `sigmoid(s/T)` with fitted `T > 0`; unlike Platt scaling it does not
fit an intercept.  Labels are retained in ascending order and
`predict_proba` returns columns `[1-p,p]` in that class order.  Each fitted method
validate finite scores, nonnegative weights, positive total mass, and positive
mass for each class.  The temperature fit uses a positive-domain damped
Newton solve in inverse temperature, while the sigmoid fit uses a stable
damped Newton solve for the two parameters `[slope,intercept]` with optional
L2 regularization;
`state` reports the objective and convergence.  Isotonic fits store weighted
score knots and linearly interpolate between them, with constant extrapolation
outside the fitted range.

`predict_proba_jvp` and `predict_proba_vjp` provide exact score products for
temperature and sigmoid calibration and for isotonic interpolation away from
knots.  `predict_proba_parameter_jvp` and `_vjp` expose the temperature
product with respect to `[T]` and the two sigmoid products with respect to
`[slope,intercept]`.  Isotonic products through fitted PAVA parameters, and
score products at a knot where the active interpolation segment is ambiguous,
return
`FORTNUM_NOT_IMPLEMENTED` rather than differentiating through an active-set
change.  `parameters`, `parameter_count`, `classes`, `method`, and `fitted`
expose deterministic state.  `predict` uses the second class only when its
probability is strictly greater, preserving first-class ties.  The explicit
device contract supports selected CPU contexts; CUDA returns
`FORTNUM_NOT_IMPLEMENTED` until a resident calibration kernel is linked.

`multiclass_probability_calibrator_t` provides multiclass temperature,
weighted one-vs-rest Platt sigmoid, and weighted one-vs-rest isotonic
calibration.  `fit(logits,labels,status[,options,sample_weight,state])`
requires one logit column for every observed class and stores arbitrary integer
classes in ascending order.  Temperature scaling fits one positive scalar by
weighted softmax negative log likelihood.  Platt scaling first computes a
stable raw softmax, fits a weighted smooth sigmoid map to every one-vs-rest
column, and normalizes the sigmoid outputs back to a simplex.  Isotonic
scaling uses weighted PAVA maps and the same normalization.  `predict_proba`
returns columns in stored class order; `predict` chooses the first class on
equal probabilities.

Temperature `predict_proba_jvp/vjp` products cover logit tangents and
cotangents, and its parameter products cover the single packed `[temperature]`
coordinate.  Platt score and parameter products are exact and smooth; its
packed parameter vector is interleaved
`[slope_1,intercept_1,slope_2,intercept_2,...]`.  Isotonic values are complete
on selected CPU, while score and parameter products return
`FORTNUM_NOT_IMPLEMENTED` because PAVA active-set derivatives are not yet
defined; isotonic `parameter_count()` is zero.
`classes`, `parameters`, `parameter_count`, `method`, `fitted`, and
`device_supported` expose deterministic state.  Selected CPU prediction is
supported for both methods; CUDA prediction returns a typed
`FORTNUM_NOT_IMPLEMENTED` refusal until a resident calibration kernel is linked.

### `fortml_calibrated_logistic_classifier`

`calibrated_logistic_classifier_t%fit(x,labels,status[,options,state,
sample_weight,class_weight])` implements a leakage-safe binary calibration
workflow. It uses `stratified_kfold_splitter_t` to fit a fresh logistic model
on each training fold, writes one held-out margin for every sample, fits the
selected binary calibration map on those out-of-fold margins, and then fits
the deployment logistic model on all rows. `cv_folds`, `cv_shuffle`, and
`cv_seed` are explicit options. The state records the fold count, convergence
flags, and uncalibrated and calibrated out-of-fold log losses, so a caller can
check calibration without evaluating a map on the margins used to fit it.

`predict_proba`, `predict`, `classes`, `decision_function`, `parameters`,
`parameter_count`, and `set_parameters` expose the final deployment model.
The packed vector is `[logistic parameters, calibration parameters]`; the
calibration suffix is `[T]` for temperature, `[slope,intercept]` for sigmoid,
and empty for isotonic. Smooth temperature and sigmoid maps provide exact
joint input/parameter JVP and VJP products. Isotonic derivative calls return
`FORTNUM_NOT_IMPLEMENTED` because the PAVA active set is discrete. Selected
CPU prediction is supported. CUDA prediction and decision requests return a
typed `FORTNUM_NOT_IMPLEMENTED` refusal until a resident logistic-plus-
calibration kernel is linked. `test_calibrated_logistic_classifier` checks
simplex and sorted-label behavior, OOF diagnostics, central finite differences,
the JVP/VJP adjoint identity, isotonic refusal, and the CUDA boundary.

### `fortml_calibrated_softmax_classifier`

`calibrated_softmax_classifier_t%fit(x,labels,status[,options,state,
sample_weight,class_weight])` implements the multiclass analogue. It performs
deterministic stratified out-of-fold softmax fits, fits one positive
temperature, one-vs-rest Platt sigmoid, or weighted one-vs-rest isotonic maps on
held-out logits, and refits the deployment softmax model on all rows. Sorted
integer classes, nonnegative sample weights, positive class weights, and a
minimum positive-weight count per fold are validated before any state is
replaced. The packed deployment vector contains the softmax coefficients,
intercepts, and either `[temperature]`, interleaved `[slope,intercept]` Platt
coordinates, or no calibration coordinates for isotonic. Prediction and
packed input or parameter JVP/VJP products are exact on the smooth temperature
and Platt paths. Isotonic values and labels are complete, while active-set
products return `FORTNUM_NOT_IMPLEMENTED`. A failed fit is transactional: a
previously fitted deployment remains usable when malformed inputs or options
are refused. Selected CPU prediction is supported; every CUDA request returns
a typed `FORTNUM_NOT_IMPLEMENTED` refusal until a resident softmax-plus-
calibration kernel is linked. `test_calibrated_softmax_classifier` checks OOF
replay for all policies, sorted labels, smooth products, malformed weights,
transactional refits, isotonic refusals, and the CUDA boundary.

### `fortml_regression_metrics`

The regression metric procedures accept row-oriented target and prediction
matrices with matching nonempty shapes. `regression_mean_squared_error`,
`regression_root_mean_squared_error`, `regression_mean_absolute_error`,
`regression_median_absolute_error`, `regression_max_error`,
`regression_r2_score`, and `regression_explained_variance` cover the core
continuous-target measures. `regression_mean_squared_log_error` requires
nonnegative targets and predictions. `regression_mean_absolute_percentage_error`
requires nonzero targets, and `regression_mean_pinball_loss` takes a quantile
in `[0,1]`. Optional row weights are finite and nonnegative with positive
total mass. All mean metrics use a uniform average over output columns. The
median is a deterministic weighted median of flattened absolute errors. R2 and
explained variance refuse constant target columns, making degenerate behavior
visible to callers instead of silently applying a force-finite policy.

### `fortml_cuda_metrics`

`cuda_mean_squared_error(device,target,prediction,value,status[,sample_weight])`
is the explicit native-CUDA counterpart for the weighted MSE reduction. It
requires a selected, available CUDA context and contiguous finite host arrays
with matching nonempty shapes. The native path copies the arrays to a
temporary device allocation, performs the elementwise squared-error and block
reduction on CUDA, then copies the block partials back for the final scalar
accumulation; the transfer-inclusive boundary is intentional and is never
relabeled as resident model execution. The device context records the exact
temporary residency and transfer counters.
Weights are finite, nonnegative, and have positive mass. The default Fortran
build links an unavailable stub, so the routine returns
`FORTNUM_NOT_IMPLEMENTED` without changing the output when no native CUDA
object is linked. `fortml_cuda_mse_available()` exposes the runtime probe.

The companion C ABI in `src/validation/fortml_cuda_mse_plan.h` provides a
resident no-autodiff plan: `fortml_cuda_mse_plan_create` uploads the immutable
target, prediction, and optional row-weight arrays once;
`fortml_cuda_mse_plan_execute` repeats the CUDA reduction without re-uploading
those inputs; and `fortml_cuda_mse_plan_destroy` releases the device buffers.
The independent `test/run_cuda_mse_plan.sh` gate executes five reductions and
compares each scalar with a host oracle. It is a concrete resident primitive,
not an end-to-end estimator GPU claim.

### `fortml_losses`

The loss facade provides stable matrix-valued `sigmoid_value`, `softmax_value`,
and `log_softmax_value` procedures with matching JVP and VJP products.
`binary_cross_entropy_with_logits_value` uses a mean reduction over all matrix
entries and accepts targets in `[0,1]`. `softmax_cross_entropy_value` uses one
one-based integer class label per row and a mean reduction over rows. Both loss
families expose JVP and VJP procedures, reject nonfinite inputs, and evaluate
the value with a shifted log-sum-exp or softplus expression. These routines are
the shared objective layer for neural, multiclass, GP, and boosting adapters.
`huber_loss_value`/`huber_loss_jvp`/`huber_loss_vjp` add a mean robust
regression loss with a positive finite transition `delta`; its first derivative
is continuous at the transition. `quantile_loss_value` and its products
implement mean pinball loss for `0 < quantile < 1`; JVP/VJP calls refuse a zero
residual because that loss is nondifferentiable there. Both families treat
targets as constants and differentiate predictions only.

The facade also exposes Hessian-vector products for the smooth portions of
these objectives: `binary_cross_entropy_with_logits_hvp`,
`softmax_cross_entropy_hvp`, and `weighted_mse_loss_hvp` return the exact
prediction-space curvature product. `huber_loss_hvp` returns the piecewise
quadratic curvature (zero in the linear branch) and refuses an exact
`abs(prediction-targets)==delta` transition instead of inventing a second
derivative. `weighted_mse_loss_value` and its JVP/VJP/HVP products accept a
finite nonnegative row-weight vector and `LOSS_REDUCTION_MEAN` or
`LOSS_REDUCTION_SUM`; mean reduction divides by positive weight mass, while sum
reduction leaves the weighted sum unnormalised. The weighted-MSE products are
the implementation used by `mlp_loss_value_gradient` and `mlp_loss_hvp`, so
standalone checks and MLP objectives share one reduction and derivative oracle.

`mae_loss_value` implements weighted mean/sum mean-absolute error with the
same optional row-weight and reduction contract. `mae_loss_jvp` and
`mae_loss_vjp` differentiate predictions with the signed residual derivative;
both return a domain refusal when any residual is exactly zero, where MAE has
no unique derivative. `mean_absolute_error_loss_*` are generic aliases for
the same procedures.

`focal_binary_cross_entropy_with_logits_value` and its JVP/VJP products use a
stable logits formulation with positive-class weight `alpha` in `[0,1]` and
focusing exponent `gamma >= 0`. Binary targets recover the standard
`alpha_t (1-p_t)**gamma BCE` objective; relaxed targets in `[0,1]` are also
accepted. Optional row weights and `LOSS_REDUCTION_MEAN`/
`LOSS_REDUCTION_SUM` follow the weighted-MSE semantics. The equivalent
`binary_focal_cross_entropy_with_logits_*` names are generic aliases. Invalid
parameters, nonfinite data, malformed weights, and zero-support reductions
return typed domain statuses. No CUDA loss kernel is linked; device requests
remain an explicit unavailable capability rather than a host fallback.

`gaussian_nll_*` adds a heteroscedastic Gaussian negative log likelihood in
`(prediction, log_variance)` coordinates, including the normalizing
`log(2*pi)` constant. Value, JVP, VJP, and HVP products cover both mean and log
variance; `gaussian_nll_hvp` returns both output blocks. `poisson_nll_*` (also
available through the `poisson_count_nll_*` aliases) uses log-rate coordinates
and includes `log_gamma(target+1)`, accepting finite nonnegative real counts.
Both families accept optional nonnegative row weights and
`LOSS_REDUCTION_MEAN`/`LOSS_REDUCTION_SUM`; mean divides by positive weight
mass and sum leaves the weighted sum unnormalised. Log variances below
`log(tiny)` and log rates above `log(huge)`, nonfinite products, malformed
weights, and negative counts return typed domain errors. No CUDA NLL kernels are
resident, so CUDA requests remain an explicit unavailable capability with no
host fallback.

`multilabel_binary_cross_entropy_with_logits_*` treats every output column as
an independent relaxed indicator head. The value, JVP, VJP, and HVP procedures
accept optional finite nonnegative row weights and
`LOSS_REDUCTION_MEAN`/`LOSS_REDUCTION_SUM`; mean reduction divides by positive
row-weight mass and sum reduction is unnormalised. The HVP is the exact
diagonal sigmoid curvature for each logit. Targets outside `[0,1]`, malformed
weights, nonfinite products, and zero-support batches return typed domain
errors.

`ordinal_cumulative_logit_loss_*` implements the ordered cumulative-logit
negative log likelihood. `logits(i,k)` means
`P(Y<=k)=sigmoid(logits(i,k))`, labels are one-based in
`1:size(logits,2)+1`, and every row must have strictly increasing cumulative
logits. Weighted mean/sum value, JVP, VJP, and exact HVP products are
available; the HVP differentiates the difference of the two sigmoid terms
analytically. A nonpositive class probability or malformed ordering is a
typed domain refusal. These new loss kernels are CPU reference paths only;
CUDA requests remain unavailable until resident loss kernels are linked and
never fall back through a host copy.

`contrastive_loss_value`, `contrastive_loss_jvp`, `contrastive_loss_vjp`, and
`contrastive_loss_hvp` provide a pairwise metric-learning objective for two
equal-shaped row-wise embedding matrices. A one label denotes a matching pair
and zero denotes a non-matching pair; the branches are `0.5*d**2` and
`0.5*max(0,margin-d)**2` for Euclidean distance `d`. Optional nonnegative row
weights and `LOSS_REDUCTION_MEAN`/`LOSS_REDUCTION_SUM` follow the shared
positive-weight-mass contract. Product calls return a typed domain error for a
non-matching zero distance or an exact margin boundary, while matching pairs
remain smooth at zero. `contrastive_loss_value_device` routes CPU exactly and
returns `FORTNUM_NOT_IMPLEMENTED` for CUDA until a resident pair-distance and
reduction kernel is available; it never hides a host fallback.

### `fortml_softmax_regression`

`softmax_regression_t%fit(x,labels,status[,l2,fit_intercept,max_iterations,
tolerance,sample_weight,class_weight])` fits a multinomial softmax model with one column
per sorted integer
class label. The objective is mean softmax cross-entropy with L2 regularization
on feature coefficients and is optimized by `fortopt_lbfgsb`. A nonnegative
`sample_weight` vector selects the corresponding positive-weight-mass
cross-entropy reduction. A positive class-weight vector with one entry per
sorted class multiplies the sample weights before reduction.
`decision_function` returns one logit column per class, `predict_proba` applies
the stable row-wise softmax, and `predict` maps the largest probability back to
the stored class label with a first-column tie rule. `coefficients`,
`intercept_values`, `classes`, `feature_count`, `class_count`, and `fitted`
expose the model state. At least two distinct classes are required. Sparse
targets and multilabel weighting remain roadmap work.

`parameter_count()`, `parameters()`, and `set_parameters(values,status)` use
column-major coefficient blocks followed by the intercept block when enabled.
`regularization()` and `set_regularization(value,status)` expose the finite
nonnegative feature-L2 coefficient for explicit training-state management.
`jvp`/`vjp` (also available as `decision_function_jvp`/`decision_function_vjp`)
differentiate logits with
respect to packed parameters and inputs, while `predict_proba_jvp`/
`predict_proba_vjp` compose the stable softmax products with those affine
products. All derivative paths validate finite tangents/cotangents and exact
shapes, and return a domain status for unfitted or malformed calls. HVPs and
parameter products for GP classifiers remain separate roadmap contracts.

### `fortml_softmax_training`

`softmax_training_objective_t` packages a fitted `softmax_regression_t` and a
weighted multiclass data set as an exact FortOpt objective. Call
`initialize(model,x,labels,l2,status[,optimize_l2,sample_weight,class_weight,
optimize_log_l2])`, then use `parameters`, `parameter_count`,
`value_gradient`, `jvp`, `vjp`, and `hvp`.
The packed vector contains column-major coefficient blocks followed by the
intercept block and, with `optimize_l2=.true.` or `optimize_log_l2=.true.`, one
final L2 coordinate. The direct coordinate is non-negative; the transformed coordinate
stores `log(L2)` and is exponentiated inside the objective, which keeps the
regularization strictly positive. The objective is the weighted mean
cross-entropy plus feature-only L2; class weights are in sorted-class order and
multiply sample weights before the positive-mass reduction. The gradient, JVP,
VJP, and HVP include the exact mixed parameter/hyperparameter blocks, so
hyperparameter derivatives do not use finite differences. The two optimization
modes are mutually exclusive.

`fortopt(objective,status)` installs a context callback for
`fortopt_objective`. `softmax_optimize_lbfgsb(model,x,labels,options,result,
status[,sample_weight,class_weight])` owns a bounded FortOpt L-BFGS-B lifecycle.
`softmax_lbfgsb_options_t` supplies explicit model bounds, direct
`l2_lower_bound`/`l2_upper_bound`, or transformed
`log_l2_lower_bound`/`log_l2_upper_bound`. The result reports convergence,
iterations, line-search evaluations, objective, gradient norm, and final L2. The
adapter requires a fitted model; use `softmax_regression_t%fit` first to obtain
an initial state. The model also exposes `regularization()` and
`set_regularization(value,status)` for explicit L2 state management.

### `fortml_ovr_logistic_classifier`

`ovr_logistic_classifier_t%fit(x,labels,status[,l2,fit_intercept,
max_iterations,tolerance,sample_weight,class_weight])` fits one independent
binary logistic estimator per sorted integer class. `sample_weight` is shared
across the binary fits. `class_weight` has one positive finite entry per
sorted multiclass label and scales the corresponding rows. `decision_function`
returns one binary score column per class. `predict_proba` takes the positive
probability from each binary model and normalizes rows to a deterministic
multiclass simplex. `predict` uses the first class on probability ties.

`classes`, `class_count`, `feature_count`, `parameter_count`, `parameters`,
`set_parameters`, and `fitted` expose model metadata and the concatenated
coefficient/intercept blocks. `predict_proba_jvp` and `predict_proba_vjp`
differentiate with respect to inputs. The separate
`predict_proba_parameter_jvp` and `predict_proba_parameter_vjp` methods expose
the packed fitted-parameter products. All normalized products include the
quotient rule. Tangents and cotangents must be finite and shape-compatible.
Hyperparameter derivatives through the discrete optimizer fit remain a
separate trainer contract.

### `fortml_multilabel_logistic_classifier`

`multilabel_logistic_classifier_t%fit(x,indicators,status[,l2,fit_intercept,
max_iterations,tolerance,sample_weight,class_weight,thresholds])` fits one
independent binary logistic head for every column of an integer indicator
matrix. `x` has shape `(n_samples,n_features)` and `indicators` has shape
`(n_samples,n_labels)`; each entry must be exactly zero or one, and every
column must contain both values. `sample_weight` is shared by all heads.
`class_weight`, when present, has shape `(2,n_labels)` in negative/positive
order. The optional `thresholds` vector controls hard prediction and defaults
to `0.5` for every head.

`decision_function` returns one score column per label. `predict_proba` returns
the positive probability matrix with shape `(n_samples,n_labels)`, rather than
a list of two-column matrices. `predict` applies the per-label thresholds and
returns an integer indicator matrix. `label_count`, `feature_count`,
`parameter_count`, `parameters`, `set_parameters`, `thresholds`,
`set_thresholds`, and `fitted` expose the packed state. Parameter blocks are
concatenated by label, with each block using the underlying logistic
coefficient/intercept order.

`predict_proba_jvp`/`predict_proba_vjp` differentiate the fixed fitted model
with respect to the continuous input batch. The corresponding
`predict_proba_parameter_jvp`/`predict_proba_parameter_vjp` methods use the
packed parameter vector. Integer target and hard-threshold prediction paths
are intentionally not differentiated. All products validate finite values and
exact shapes.

`device_supported(kind)` reports CPU support for fitted models and no CUDA
support in the current build. `decision_function_device`,
`predict_proba_device`, and `predict_device` dispatch selected CPU contexts;
CUDA requests return `FORTNUM_NOT_IMPLEMENTED` until a resident multi-head
kernel is linked, with no hidden host fallback.

### `fortml_classifier_chain`

`classifier_chain_t%fit(x,labels,status[,l2,fit_intercept,max_iterations,
tolerance,sample_weight,class_weight,thresholds])` fits one binary logistic
head per output column. `labels(n_samples,n_outputs)` may use any two distinct
integer labels in each column; `classes()` returns each pair in sorted order.
Head `j` is trained on the original features followed by the observed positive
indicators for outputs `1:j-1`. `sample_weight` is shared, while
`class_weight(2,n_outputs)` and `thresholds(n_outputs)` are in each output's
sorted negative/positive order.

`decision_function` returns the sequential logits and `predict_proba` returns
the positive probability matrix `(n_samples,n_outputs)`. At prediction time
the chain uses prior positive probabilities as continuous features; `predict`
applies the per-output thresholds and maps back to the stored integer labels.
`parameters()` concatenates head coefficient/intercept blocks in output order,
so later heads have one additional packed feature parameter each. The input and
packed-parameter `predict_proba_jvp`/`predict_proba_vjp` products propagate the
same forward and reverse chain rules exactly. Hard labels and fit-time optimizer
paths are discrete and are not differentiated.

`device_supported(FORTML_DEVICE_CPU)` is true for fitted models. The selected
CPU device methods dispatch to the host implementation; selected CUDA requests
return `FORTNUM_NOT_IMPLEMENTED` until a resident classifier-chain kernel is
linked, with no hidden host fallback. `fortml_classifier_chain_logistic_classifier`
re-exports the type under the longer compatibility name
`classifier_chain_logistic_classifier_t`.

### `fortml_ordinal_logistic_classifier`

`ordinal_logistic_classifier_t%fit(x,labels,status[,l2,fit_intercept,
max_iterations,tolerance,sample_weight,class_weight])` fits a weighted
cumulative-logit model for sorted integer classes. The latent score is
`x·coefficient + intercept`, and the ordered thresholds satisfy
`P(Y <= k) = sigmoid(threshold(k) - score)`. Thresholds are fitted through
positive increments internally, while `parameters()` exposes the packed
coefficient, optional intercept, and actual strictly increasing thresholds;
`set_parameters` validates that ordering. `predict_proba` returns the ordered
class probabilities and `predict` returns the sorted integer labels.

`predict_proba_jvp`/`predict_proba_vjp` differentiate the fixed fitted model
with respect to continuous inputs. The corresponding
`predict_proba_parameter_jvp`/`predict_proba_parameter_vjp` methods use the
packed coefficient/intercept/threshold vector. `thresholds`, `classes`,
`class_count`, `feature_count`, and `parameter_count` expose metadata. CPU
device dispatch is exact; CUDA prediction methods return
`FORTNUM_NOT_IMPLEMENTED` until a resident ordinal kernel is linked, with no
hidden host fallback.

### `fortml_ovo_logistic_classifier`

`ovo_logistic_classifier_t%fit(x,labels,status[,l2,fit_intercept,
max_iterations,tolerance,sample_weight,class_weight])` fits one binary
logistic estimator for every pair of sorted integer classes. `pair_classes()`
returns the deterministic `(negative,positive)` label columns in lexicographic
pair order, and `decision_function` returns one binary score per pair.
`predict_proba` aggregates the positive and negative pair probabilities as
votes and divides by the number of fitted pair models, so each row is a probability
simplex. This explicit vote policy is differentiable and does not claim to
implement scikit-learn's separate pairwise coupling solver. `predict` uses the
first class on probability ties.

`classes`, `pair_count`, `class_count`, `feature_count`, `parameter_count`,
`parameters`, `set_parameters`, and `fitted` expose packed pair-model state.
Input and packed-parameter products are available through
`predict_proba_jvp`/`predict_proba_vjp` and
`predict_proba_parameter_jvp`/`predict_proba_parameter_vjp`. The pairwise vote
aggregation is linear, so every product propagates the binary logistic
products with a fixed opponent-count factor. Finite, shape, zero-support, and
unfitted violations return status errors.
`device_supported(kind)` and `predict_device`/`predict_proba_device` expose the
same explicit backend boundary: CPU dispatch is supported for fitted models,
while CUDA returns `FORTNUM_NOT_IMPLEMENTED` because no resident pairwise
logistic kernel is linked.

### `fortml_knn_classifier`

`knn_classifier_t%fit(x,labels,status[,n_neighbors,weights,sample_weight])`
stores a dense training matrix and sorted integer class labels. `predict_proba`
uses the `k` nearest rows under squared Euclidean distance. Uniform voting and
inverse-distance voting (`KNN_WEIGHTS_UNIFORM` and `KNN_WEIGHTS_DISTANCE`) are
available; exact zero-distance matches receive the normalized weight of the
matching rows. Optional sample weights are finite, nonnegative, and must have
positive mass. Ties are stable in original training-row order, and `predict`
maps the first maximum probability back to `classes`.

`predict_proba_jvp` and `predict_proba_vjp` are explicit discrete-boundary
contracts: they return `FORTNUM_NOT_IMPLEMENTED` because a perturbation can
change the selected neighbor set. KD/ball-tree, radius, sparse, and soft
neighbor variants are separate roadmap items. Unfitted models, nonfinite data,
invalid `k`, and malformed weights are refused with a domain status.

`predict_device` uses the selected CPU context directly. A CUDA build that
links `fortml_cuda_knn.cu` creates a resident training-set plan and runs the
distance and stable-tie reduction in native CUDA, with explicit query and label
transfers. The default stub build returns `FORTNUM_NOT_IMPLEMENTED` for CUDA.
The kernel and Fortran API are checked by `test/run_cuda_knn_plan.sh` and
`test/run_knn_classifier_cuda.sh`.

### `fortml_radius_neighbors_classifier`

`radius_neighbors_classifier_t%fit(x,labels,status[,radius,weights,
sample_weight,outlier_label])` stores a dense training matrix and includes
every row within the closed squared-Euclidean radius. Uniform and
inverse-distance votes use the same deterministic sorted-class and sample
weight conventions as kNN. An optional `outlier_label` must name a fitted
class and receives a one-hot probability when a query has no neighbors;
without it, empty neighborhoods return a domain status.

`predict_proba`, `predict`, `classes`, `radius`, and `device_supported` expose
the fitted state. Neighbor selection is discontinuous, so input JVP/VJP calls
return `FORTNUM_NOT_IMPLEMENTED`. CPU dispatch is complete; CUDA is an
explicit `FORTNUM_NOT_IMPLEMENTED` refusal until a resident radius-search
kernel is linked, with no host fallback.

### `fortml_radius_neighbors_regression`

`radius_neighbors_regressor_t%fit(x,targets,status[,radius,weights,
sample_weight,outlier_value])` stores a dense scalar-target training set and
averages every row within the closed squared-Euclidean radius. Uniform and
inverse-distance averaging use `RADIUS_REGRESSION_WEIGHTS_UNIFORM` and
`RADIUS_REGRESSION_WEIGHTS_DISTANCE`; sample weights must be finite,
nonnegative, and have positive total mass. An optional finite `outlier_value`
is returned for an empty neighborhood; without it, prediction returns
`FORTNUM_DOMAIN_ERROR`.

`predict`, `radius`, `weighting`, `feature_count`, `sample_count`, and
`device_supported` expose the fitted state. Radius membership is discrete, so
`predict_jvp` and `predict_vjp` return `FORTNUM_NOT_IMPLEMENTED` rather than
claiming a zero derivative across a selection boundary. CPU dispatch is
complete. CUDA is an explicit `FORTNUM_NOT_IMPLEMENTED` refusal until a
resident radius-search reduction is linked; no hidden host fallback is used.

### `fortml_radius_neighbors_multioutput_regression`

`radius_neighbors_multioutput_regressor_t%fit(x,targets,status[,radius,
weights,sample_weight,outlier_value])` is the multi-output counterpart of the
scalar radius regressor. `targets` has shape `(n_samples,n_outputs)` and every
output column is averaged over the same closed squared-Euclidean neighborhood.
Uniform and inverse-distance weighting use
`RADIUS_MULTI_REGRESSION_WEIGHTS_UNIFORM` and
`RADIUS_MULTI_REGRESSION_WEIGHTS_DISTANCE`; finite nonnegative sample weights
must have positive total mass. An optional finite `outlier_value(n_outputs)`
is returned for an empty neighborhood. Without it, an empty query returns
`FORTNUM_DOMAIN_ERROR`.

`predict`, `radius`, `weighting`, `feature_count`, `sample_count`,
`output_count`, `fitted`, and `device_supported` expose the fitted state.
Away from a radius boundary, `predict_jvp` and `predict_vjp` return exact zero
products for the piecewise-constant map. If a query lies exactly on a retained
radius boundary they return `FORTNUM_DOMAIN_ERROR`, making the undefined
selection derivative explicit. Selected CPU device calls delegate to the host
implementation; selected CUDA calls return `FORTNUM_NOT_IMPLEMENTED` without
silently copying the model through the host. See
`test_radius_neighbors_multioutput_regression` for independent weighted,
outlier, boundary, and device oracles.

### `fortml_preprocessing`

`standard_scaler_t%fit` stores column means and population standard deviations.
Zero-variance columns use unit scale. `transform`, `inverse_transform`, and
`transform_jvp` operate on row-oriented batches. `minmax_scaler_t%fit` stores
column extrema and maps to an increasing caller-selected range (default
`[0,1]`), with the same transform, inverse, and input-JVP operations.
`robust_scaler_t%fit` stores the per-feature median and an interpolated finite
quantile range (default 25th--75th percentile IQR), uses unit scale for
constant features, and exposes the same transform, inverse, and input-JVP
operations. Fitted statistics are state rather than differentiable parameters.
The JVPs are with respect to the input batch. Unfitted models, nonfinite values,
invalid quantile ranges, and shape mismatches are refused.

### `fortml_simple_imputer`

`simple_imputer_t%fit(x,status[,strategy,fill_value])` fits one statistic per
feature using row-oriented data. `strategy` is `"mean"` (the default),
`"median"`, or `"constant"`; the latter uses the finite `fill_value` (default
zero) and permits a feature whose entire training column is missing. Missing
values are IEEE NaNs. Infinities and empty matrices are refused, and mean or
median fitting refuses an entirely missing feature. `transform` replaces NaNs
while preserving observed values. `statistics`, `feature_count`, `strategy`,
and `fitted` expose the fitted state.

`transform_jvp(x,x_dot,transformed_dot,status)` and
`transform_vjp(x,output_bar,input_bar,status)` implement the exact piecewise
input derivative: observed entries are identity and missing entries are locally
constant, hence have zero tangent and cotangent. Fitted statistics are state,
not silently promoted to trainable parameters; callers that need statistic
hypergradients must differentiate the fitting objective explicitly.

### `fortml_missing_indicator`

`missing_indicator_t%fit(x,status[,features])` fits a dense binary missingness
mask over row-oriented IEEE-NaN data. `features="all"` emits one column per
input feature; the default `"missing-only"` records only columns that contain
at least one missing value in the fit data. `transform` emits `1` for a NaN and
`0` for an observed entry, while `feature_indices`, `input_count`,
`output_count`, `mode`, and `fitted` expose the fitted schema. Infinities,
empty matrices, unknown policies, and shape mismatches return a domain status.

The mask is locally constant, so `transform_jvp` and `transform_vjp` return
exact zero products after validating finite tangents/cotangents. This is an
explicit smoothness contract, not a hidden finite-difference approximation;
the sparse-view and device-resident indicator kernels remain separate
follow-up work.

### `fortml_sparse_preprocessing`

`sparse_standard_scaler_t` is the sparse-safe `StandardScaler` branch for
`fortsparse` real CSC matrices. `fit(input,status[,with_mean,with_std])`
counts implicit zero entries when computing each feature mean and population
variance. Sparse centering is rejected with `FORTSPARSE_INVALID_MATRIX`
because subtracting a nonzero mean would change the storage class; callers
must select `with_mean=.false.` (the default). Zero-variance features use unit
scale, matching the dense scaler convention.

`transform`, `inverse_transform`, `means`, `scales`, `feature_count`,
`sample_count`, `with_std`, and `fitted` expose the fitted state. Transform
and inverse transform copy the CSC structure and scale only stored values,
so no implicit zeros are materialized. `transform_jvp` and
`transform_vjp` accept CSC cotangent/tangent matrices with the same structure
contract and apply the exact diagonal value map. Invalid CSC structure,
unfitted state, and feature-count mismatches return a typed
`FORTSPARSE_INVALID_MATRIX` status. The independent dense expansion oracle is
`test_sparse_preprocessing`.

### `fortml_one_hot_encoder`

`one_hot_encoder_t%fit(x,status[,handle_unknown,missing_value,handle_missing,drop_first])`
fits an integer categorical matrix with samples in rows.  Each feature's
categories are sorted in ascending integer order and packed into
`categories()` with one-based `category_offsets()`; `output_offsets()` gives
the corresponding dense output blocks.  `handle_unknown` is `"error"` (the
default) or `"ignore"` (an all-zero block at transform time).  Passing an
integer `missing_value` enables missing-value handling: `handle_missing` may be
`"error"`, `"ignore"`, or `"category"` (the sentinel is retained as an
explicit sorted category).  `drop_first=.true.` keeps the first sorted category
in metadata but omits its output column, matching a reference-category design.
Empty matrices, all-missing features under the ignore policy, malformed policy
names, unknown/error values, and output shape mismatches return a domain
status.  `feature_count`, `output_count`, per-feature category/output counts,
policy accessors, and `fitted` expose the packed state.

Categorical integer inputs intentionally do not pretend to be differentiable:
`transform_jvp` and `transform_vjp` validate all shapes and return
`FORTNUM_NOT_IMPLEMENTED` with an explicit tangent/cotangent-space refusal.
Use a differentiable numerical basis or ordinal representation when a smooth
input derivative is required.  This boundary is covered by
`test_one_hot_encoder` independently of the implementation.

### `fortml_basis`

`basis_map_t` has these constructors and equivalent type-bound initializers:

| Constructor | Features | Active parameter vector |
| --- | --- | --- |
| `make_polynomial_basis(n_inputs,degree,status[,include_intercept])` | Separate powers 1 through `degree` for each input | Empty |
| `make_polynomial_interaction_basis(n_inputs,degree,status[,include_intercept])` | All nonconstant monomials through total degree `degree`, in deterministic graded order | Empty |
| `make_chebyshev_basis(n_inputs,degree,status[,include_intercept])` | Per-input Chebyshev first-kind features `T_1(x),...,T_degree(x)` in input-major order | Empty |
| `make_fourier_basis(n_inputs,frequencies,status[,include_intercept])` | Sine/cosine pair for each positive frequency and input | Log frequencies, column-major |
| `make_random_fourier_basis(n_inputs,frequencies,phases,status[,include_intercept])` | Fixed `sqrt(2/m) cos(w_k dot x + b_k)` random-feature map | Empty (frequencies and phases are fixed transform state) |
| `make_radial_basis(n_inputs,centers,scales,status[,include_intercept])` | One anisotropic Gaussian feature per center | Centers followed by log scales |
| `make_spline_basis(n_inputs,order,breakpoints,status[,include_intercept])` | B-spline basis functions for each input | Empty |
| `map%initialize_callback(...)` | Caller-defined | Caller-defined flat vector |

The public operations are `feature_count`, `parameter_count`, `parameters`,
`set_parameters`, `evaluate`, `jvp`, `vjp`, `hvp`, and
`static_lowering_eligible`. For `jvp`, supply both `theta_dot(:)` and
`x_dot(:,:)`. For `vjp`, the output cotangent has the evaluated feature shape.
`hvp(x,u,theta_dot,x_dot,theta_hvp,x_hvp,status)` differentiates the VJP of
the scalar contraction `sum(u*evaluate(x))` in the joint parameter/input
direction. Polynomial, Chebyshev, Fourier, radial, and spline maps use analytic
second-order products (with the spline span held fixed); callback maps return
`FORTNUM_NOT_IMPLEMENTED` because their first-order callback ABI does not
declare second derivatives. The intercept column has no active parameter.

`BASIS_POLYNOMIAL`, `BASIS_CHEBYSHEV`, `BASIS_FOURIER`, `BASIS_RADIAL`,
`BASIS_SPLINE`, and `BASIS_CALLBACK` are the public family codes used by
extension and test code. Chebyshev maps use the three-term recurrence
`T_0=1`, `T_1=x`, `T_k=2*x*T_(k-1)-T_(k-2)`; `include_intercept` adds the shared
`T_0` column. They are parameter-free and exact on CPU; resident-CUDA calls
return a typed refusal until a static device lowering is available.

`make_polynomial_interaction_basis` is the interaction-aware polynomial
variant. For two inputs and degree two its optional-intercept feature order is
`1, x1, x2, x1**2, x1*x2, x2**2`; its value, JVP, VJP, and HVP products are
analytic and use no finite-difference fallback.

Callback initialization takes explicit value, JVP, and VJP procedures matching
`basis_value_callback`, `basis_jvp_callback`, and `basis_vjp_callback`. A
callback map returns false from `static_lowering_eligible` and stays on the
host. The `create_polynomial_impl`, `create_chebyshev_impl`, `create_fourier_impl`,
`create_radial_impl`, `create_spline_impl`, and `create_callback_impl`
constructors in `fortml_basis_impl` serve compiled basis families.
An extension of `basis_impl_t` supplies input and feature counts, parameter
packing, value/JVP/VJP operations, validation, and optional static eligibility.

### `fortml_pipeline`

`basis_pipeline_t` is a horizontal feature union. Construct it with
`make_basis_pipeline(n_inputs,status)`, append initialized `basis_map_t` stages
with `append`, and call `fit(x,status)` before `transform(x,features,status)`.
Every stage receives the original sample matrix. The output columns are the
stage feature blocks in append order, so polynomial, Fourier, radial, spline,
and callback maps can be composed without an implicit parameter remapping.

`parameters` and `set_parameters` flatten the stage parameter blocks in the
same order. `jvp`, `vjp`, and `hvp` return exact products over both stage parameters
and inputs. `stage_count`, `feature_count`, `parameter_count`, `valid`,
`is_fitted`, and `static_lowering_eligible` expose shape and backend
capabilities. Pass an optional unique `name` to `append`; unnamed stages use
the deterministic `stage_N` convention. `stage_name`, `feature_name`, and
`parameter_name` expose stable names such as `seasonal.feature_2` and
`seasonal.parameter_1`, while `stage_feature_offset` and
`stage_parameter_offset` return one-based starts in the packed output and
parameter vectors. These accessors let a caller route coefficients or
diagnostics without duplicating the pipeline's packing rules. Sequential
transforms and DAG branches require separate input and output contracts and
are not inferred from this horizontal union.

### Versioned pipeline state

`fortml_pipeline_persistence` supplies `basis_pipeline_state_t` plus
`capture_basis_pipeline_state`, `restore_basis_pipeline_state`,
`save_basis_pipeline_text`, and `load_basis_pipeline_text`. A state dictionary
stores dimensions, fit state, input/stage/feature/parameter names, one-based
stage offsets, and the packed parameter vector under
`FORTML_PIPELINE_STATE_VERSION`. Load onto a preconfigured fitted topology:
structural basis descriptors and callback procedures are never invented from
untrusted text. Every name, offset, shape, and finite-value check runs before a
candidate copy is committed, so malformed or mismatched state leaves the target
unchanged. `save_basis_pipeline_device` and `load_basis_pipeline_device` route
CPU text I/O and return `FORTNUM_NOT_IMPLEMENTED` for resident CUDA rather than
hiding a host transfer. See
[`docs/PIPELINE_PERSISTENCE.md`](PIPELINE_PERSISTENCE.md) and
`test_pipeline_persistence` for the independent round-trip and derivative
oracle.

Each pipeline also carries a dense `basis_input_schema_t`. Its default names
are `feature_1`, ..., `feature_n`; call `set_input_schema(names,status)` to
install nonempty unique names such as `time` and `velocity`, then use
`input_schema_name(index)` or `validate_input_schema(names,status)` at a data
boundary. Name updates and validation are transactional: duplicate, empty,
overlong, wrong-count, or mismatched names return `FORTNUM_DOMAIN_ERROR` and
leave the previous schema unchanged. The same methods are available on
`sequential_basis_pipeline_t` and `basis_fanout_pipeline_t`, so metadata can
be routed before a transform consumes a batch. This is a dense name/schema
contract; dtype, sparse layout, and estimator-wide metadata routing remain
explicit follow-up boundaries.

`sequential_basis_pipeline_t` provides that explicit sequential contract.
Construct it with `make_sequential_basis_pipeline`, append a stage whose input
count equals the previous stage's feature count, and call `fit` before
`transform`. Its flattened parameters follow stage order. Forward JVPs,
reverse VJPs, and HVPs propagate through every stage, including the input
cotangent. The
same optional unique stage names, feature/parameter names, and one-based stage
offsets are available; feature names refer to the final output block. Shape
mismatches, duplicate names, empty chains, and unfitted transforms return
status errors.

`column_basis_pipeline_t` provides the column-wise variant. Construct it with
`make_column_basis_pipeline(n_inputs,status)`, then append a basis map and a
one-based integer column list. A stage's input count must equal the list length.
Indices must be in range and unique within that stage, while different stages
may reuse columns. The transform gathers only the selected columns and
concatenates stage feature blocks. Parameter packing follows stage order, JVPs
and HVPs gather input tangents, and VJPs scatter-add stage cotangents into the original
input columns. Optional unique stage names and the same stable feature and
parameter metadata are available; `stage_columns` returns a defensive copy of
the selected input indices. This is a deterministic feature union, not a DAG
scheduler. `transform_device`, `jvp_device`, `vjp_device`, and `hvp_device`
take a selected `fortml_device_t`: CPU dispatch delegates to the exact host
products, while CUDA returns `FORTNUM_NOT_IMPLEMENTED` without touching output
arrays until a resident basis executor is linked. `device_supported` reports
the same contract (fitted CPU unions only). This explicit boundary prevents a
feature-union call from hiding host transfers behind an accelerator request.

### `conditional_basis_pipeline_t`

`conditional_basis_pipeline_t` is the bounded piecewise-routing feature-union
layer. Construct it with `make_conditional_basis_pipeline(n_inputs,status)`;
append a basis map, its selected original columns, a one-based route column,
and finite ordered bounds with `append(stage,columns,route_column,lower,upper,
status[,name])`. A branch is active on the half-open interval
`[lower,upper)`. Branch outputs are concatenated in append order and inactive
rows are zero, so overlapping and fallback intervals are representable without
adding a second DAG scheduler. Each branch delegates selected-column value and
derivative work to `column_basis_pipeline_t`.

`branch_feature_offset` and `branch_parameter_offset` expose stable one-based
packed offsets. `branch_name`, `feature_name`, `parameter_name`,
`branch_route_column`, `branch_lower_bound`, and `branch_upper_bound` expose
the corresponding routing metadata. `set_input_schema`,
`input_schema_name`, and `validate_input_schema` provide the transactional
dense schema contract. Duplicate names, malformed intervals, nonfinite data,
shape errors, and invalid parameter packs are refused without committing a
partial update.

The CPU `transform`, `jvp`, `vjp`, and `hvp` methods are exact products. Value
evaluation remains defined at an interval endpoint, but derivative requests
there return `FORTNUM_DOMAIN_ERROR` because the route mask is discontinuous.
The device methods delegate only to a selected available CPU; CUDA returns
`FORTNUM_NOT_IMPLEMENTED` without modifying output arrays until a resident
route-mask executor exists. See
[`docs/CONDITIONAL_PIPELINE.md`](CONDITIONAL_PIPELINE.md) and
`test_conditional_pipeline` for the finite-difference, adjoint, transaction,
and endpoint-oracle contracts.

`basis_fanout_pipeline_t` is a one-layer named DAG that composes arbitrary
`sequential_basis_pipeline_t` branches. Construct it with
`make_basis_fanout_pipeline(n_inputs,status)`, append nonempty sequential
branches with `append(branch,status[,name])`, and call `fit` before
`transform`. The input is fanned out to every branch and their outputs are
concatenated in append order (the fan-in). Branch parameters are packed in
branch order; `branch_name`, `branch_feature_offset`, and
`branch_parameter_offset` expose stable routing metadata, while
`feature_name` and `parameter_name` prefix each branch's local names. Reverse
VJPs sum the input cotangents from all branches, and JVP/HVP products use the
same packed direction, giving the exact chain rule for this acyclic graph.
Empty or invalid branches, input-count mismatches, duplicate names, malformed
packs, and shape errors return `FORTNUM_DOMAIN_ERROR`. The device-dispatch
methods (`transform_device`, `jvp_device`, `vjp_device`, `hvp_device`) delegate
to the CPU implementation only for a selected available CPU; CUDA returns
`FORTNUM_NOT_IMPLEMENTED` without a host fallback. This deliberately bounded
graph has no cross-branch edges, conditional routing, residual addition, or
cycle representation; those remain separate graph contracts.

### `fortml_basis_linear_regression`

`basis_linear_regression_t` composes a fitted `basis_pipeline_t` with a
multi-output `linear_regression_t`. `fit(pipeline,x,y,status[,ridge,
fit_intercept])` fits the linear coefficients on the transformed features and
retains the pipeline parameter layout. `transform` and `predict` expose the
two stages without hiding their feature shape.

`parameters` packs all basis parameters first and the column-major regression
coefficients second. `predict_jvp` and `predict_vjp` chain exact basis and
linear products with respect to both packed parameters and input rows.
`set_parameters` updates the same packed state. Empty pipelines, unfitted
models, nonfinite data, malformed packs, and shape mismatches return status
errors.

`static_lowering_eligible()` reports whether the fitted feature pipeline is
made only of statically lowerable basis maps. This is a transform capability
indicator, not an end-to-end GPU guarantee: the current basis-linear estimator
has no resident CUDA solve or prediction kernel, and callers must keep the
host path (or receive a typed device refusal) until one is linked. Callback
maps therefore make the result false instead of introducing a hidden host
callback in an accelerator path.

### `fortml_basis_pipeline_training`

`basis_pipeline_training_objective_t` keeps a composable basis pipeline and a
multi-output linear coefficient block in one differentiable least-squares
objective. `initialize(pipeline,x,target,status[,ridge,fit_intercept,
device_kind,optimize_ridge])` fits the pipeline's declared state, stores the
training rows and targets, and initializes coefficients. The packed vector is
`[pipeline parameters, column-major coefficients]`, including an intercept row
when requested. With `optimize_ridge=.true.`, one final nonnegative scalar
coordinate is appended for the ridge coefficient.
`set_parameters` updates both blocks without refitting the targets and refuses
a negative optimized ridge value.

`value_gradient` evaluates mean half-squared error plus optional ridge penalty.
When the ridge coordinate is optimized, its exact derivative is half the
non-intercept coefficient squared norm; the HVP includes both mixed
ridge/coefficient blocks and the zero ridge-ridge curvature. `jvp`, `vjp`, and
`hvp` use the pipeline's chained analytic products and the linear map's exact
contractions. `fortopt` supplies the same value/gradient callback to FortOpt
L-BFGS-B. CPU is the current execution path. A CUDA
request returns `FORTNUM_NOT_IMPLEMENTED` before fitting, so no host callback
is hidden behind a device selection. The independent
`test_basis_pipeline_training` fixture checks value/JVP, coordinate and
directional HVP finite differences, and the typed CUDA refusal.

### `fortml_validation`

`kfold_splitter_t`, `stratified_kfold_splitter_t`, `group_kfold_splitter_t`, and
`time_series_splitter_t` are index-only cross-validation iterators. Initialize
the first three with `n_samples` (or integer labels/groups) and `n_splits`, or
initialize a time-series splitter with `n_samples`, `n_splits`, and optional
`test_size`, `gap`, and `max_train_size`. Call
`next_split(train_indices,test_indices,has_split,status)` until `has_split` is
false. `reset()` replays the same sequence. Shuffled iterators use a local
positive seed and never touch process-global RNG state. Test folds are
balanced; stratified folds distribute each class round-robin; grouped folds
keep each group entirely in one fold and greedily balance uneven group sizes.
Time-series folds are chronological: each test window follows its training
window, `gap` rows immediately before the test window are excluded from
training, and a positive `max_train_size` selects a rolling rather than
expanding training window. Invalid fold counts, seeds, windows, and
pre-initialization calls return status errors. The splitters do not fit or
store transformers, so callers must fit preprocessing on each training index
set explicitly. All splitters are CPU index operations with no derivative or
resident-device capability.

`estimator_score_metadata_t` records a scorer name, input representation
(labels, decision values, or probabilities), metric kind, maximize/minimize
orientation, sample-weight support, and differentiability. Its
`oriented_value` and `prefer` methods let a search retain the best candidate
without embedding metric-specific sign conventions. `estimator_validation_metadata_t`
bundles this scorer with an `estimator_capability_t`, parameter count, and
explicit `cloneable`/`resettable` declarations. These declarations are
validation guards only: model-specific strongly typed clone/reset methods
remain responsible for copying fitted state, and a false flag must never be
silently treated as permission to reuse a fitted estimator.

### `fortml_hyperparameter_search`

`hyperparameter_grid_search(objective,lower,upper,points,result,status[,options,device])`
enumerates a deterministic Cartesian grid. `points(i)==1` evaluates the
interval midpoint; larger counts include both bounds. Each candidate is sent
through the complete `fortopt_objective_t%value_gradient` product, so a grid
run is still protected by the objective's finite-value and derivative checks.
`hyperparameter_search_options_t%max_evaluations` bounds the Cartesian product
before any allocation or evaluation.

`hyperparameter_lbfgsb_search(objective,initial,lower,upper,result,status[,
options,device])` routes the same objective directly to FortOpt L-BFGS-B.
`hyperparameter_lbfgsb_multistart_search(objective,lower,upper,starts,seed,
result,status[,options,device])` draws deterministic bounded initial points
from the supplied seed, runs the same bounded FortOpt solve from each point,
retains the best converged state, sums objective evaluations, and reports
`start_count` and `successful_starts`. A nonpositive start count or a run with
no converged start is a typed domain error; no finite-difference objective is
introduced.
`hyperparameter_random_search(objective,lower,upper,samples,seed,result,status[,
options,device])` draws a deterministic bounded candidate stream from the
caller-provided seed. It evaluates the complete value/gradient product for
each candidate, retains the lowest finite value, and reports the method code
`HYPERPARAMETER_SEARCH_RANDOM` together with the evaluation count. The random
stream is a search coordinate, not a differentiable model variable.
`hyperparameter_search_result_t` reports the best packed coordinates, value,
evaluation count, method, and convergence flag. Bounds are projected by the
optimizer and malformed or nonfinite products return status errors. This layer
does not finite-difference objectives and does not reinterpret a missing
hypergradient as zero.

The generic orchestrator supports selected CPU contexts. A selected CUDA
context returns `FORTNUM_NOT_IMPLEMENTED` until objective state, candidate
buffers, and optimizer history are resident on the device; no host search is
timed or hidden behind a CUDA request. Model-specific adapters can still use
their own resident products when they expose a stronger contract.

## Neural models and variational inference

### `fortml_mlp`

`mlp_t%initialize(layer_sizes,status[,hidden_activation,output_activation,
initialization_seed])` constructs dense layers. `MLP_LINEAR`, `MLP_TANH`,
`MLP_RELU`, `MLP_GELU`, `MLP_SILU`, `MLP_ELU`, `MLP_SOFTPLUS`,
`MLP_LEAKY_RELU`, `MLP_SIGMOID`, and `MLP_MISH` are accepted activation
constants. GELU uses the standard tanh approximation; leaky ReLU uses a fixed
slope of `0.01`; sigmoid uses a branch-stable extreme-logit evaluation; and
Mish is `x*tanh(softplus(x))`. The default hidden
activation is `tanh`, and the default output activation is linear. Weights use
deterministic Xavier scaling (or He scaling for ReLU and leaky-ReLU hidden
layers), with a reproducible
phase sequence controlled by `initialization_seed` (default `17`).

The parameter vector stores each layer's weight array
`(input_width,output_width)` in column-major order followed by its bias, from
the input layer to the output layer. Use `parameter_count`, `parameters`, and
`set_parameters` to manage it.

For a deterministic affine/PCA warm start, `initialize_linear(weight1,bias1,
weight2,bias2,status)` constructs exactly the two-layer topology
`[size(weight1,1),size(weight1,2),size(weight2,2)]`, sets both activations to
`MLP_LINEAR`, and copies finite arrays after checking
`size(bias1)==size(weight1,2)` and `size(weight2,1)==size(weight1,2)`.
`set_linear_parameters(...)` applies the same checked replacement to an
already initialized two-layer linear MLP. Both routines reject nonfinite
values, zero-width arrays, nonlinear topologies, and shape mismatches before
mutating the destination. This is a linear-state loading contract only; it
does not assert an NNGP, NTK, or GP-posterior equivalence.

`initialize_from_pca(pca,status)` builds a two-layer linear autoencoder from a
fitted `pca_t`. The first layer applies the centered PCA projection and the
second applies its inverse; a whitened PCA preserves its sample-variance
scales in the two weight blocks and the center is represented in the biases.
The generated topology is `[pca%feature_count(), pca%n_components(),
pca%feature_count()]`. The PCA must be fitted and all exposed components,
mean, and variance metadata must be finite and shape-consistent. Validation and
weight construction happen before the destination is changed, so an unfitted
or malformed PCA leaves an existing MLP untouched and returns
`FORTNUM_DOMAIN_ERROR`. This is the finite linear/PCA reconstruction optimum,
not an NNGP, NTK, or GP-posterior equivalence.

`predict(x,y,status)` evaluates a batch. The product signatures are:

```text
jvp(x, dtheta, dx, y, dy, status)
vjp(x, output_bar, theta_bar, x_bar, status)
hvp(x, output_bar, dtheta, dx, theta_hvp, x_hvp, status)
```

`backprop` is an alias for `vjp`. Every smooth activation exposes analytic
first and second derivatives through these products, so `jvp`, `vjp`, and
`hvp` remain exact up to floating-point arithmetic. ReLU uses derivative zero
at the kink; leaky ReLU uses its fixed negative-side slope and zero second
derivative away from the kink. The resident CUDA dense plan currently accepts
only the original eight activation codes and returns a typed refusal for
sigmoid or Mish.

### `fortml_mlp_last_layer_gp`

`mlp_last_layer_gp_initializer_t` is a finite-feature kernel-ridge posterior
mean and last-layer NTK warm-start adapter. `fit(model,x,target,status,
[regularization])` evaluates `model%feature_map`, appends an intercept, and
solves `(Z^T Z + lambda I) C = Z^T target` for a positive `lambda`. Hidden
layers remain fixed. `apply(model,status)` validates an affine output topology
and replaces only its final weight and bias; `fit_apply` combines both steps.
`predict(model,x,y,status)` evaluates the fitted posterior mean without
mutating the model.

The initializer exposes `metadata`, `parameters`, `parameter_metadata`, and a
single `regularization` coordinate for search/optimizer adapters. `jvp(model,
x,regularization_direction,y,dy,status)` differentiates the posterior mean
with respect to that coordinate while holding the finite feature map and
targets fixed. The metadata explicitly reports
`exact_infinite_width=.false.` and `derivative_scope="fixed-feature-map"`:
this is not an exact NNGP, infinite-width NTK, or full GP-posterior weight map.
`predict_cuda`, `apply_cuda`, and `jvp_cuda` return
`FORTNUM_NOT_IMPLEMENTED` without mutating model or outputs until a resident
feature-map and solve path exists. See
[`MLP_LAST_LAYER_GP.md`](MLP_LAST_LAYER_GP.md).

### `fortml_mlp_chain`

`mlp_chain_t` is the composable neural-module seam. Initialize it with the
input width and append initialized `mlp_t` stages with unique names:

```text
chain%initialize(n_features,status)
chain%append(encoder,status,name="encoder")
chain%append(head,status,name="head")
```

Each stage's input width must equal the preceding stage's output width. The
chain owns copies of the children and exposes `stage_count`, `input_count`,
`output_count`, `parameter_count`, `parameters`, `set_parameters`, and
`parameter_range`. The packed tree is insertion ordered, with each named stage
occupying one contiguous range; this is the stable routing contract for
optimizers and checkpoints.

`predict`, `jvp`, `vjp`, and `hvp` apply the exact sequential chain rule. The
HVP differentiates the reverse cotangent recurrence, including the tangent of
downstream cotangents, so it covers both stage parameters and input
directions. `mlp_chain_objective_t` adds mean MSE plus an optional L2 block,
scalar JVP/VJP, a joint parameter/L2 HVP, and a `fortopt` callback. The
convenience `mlp_chain_optimize_lbfgsb` consumes this same analytic
value/gradient callback with explicit parameter and L2 bounds. The chain
currently reports CPU-only; CUDA objective and optimizer requests return
`FORTNUM_NOT_IMPLEMENTED` rather than executing a hidden host fallback. A
resident fused chain kernel is a separate GPU work package.

### `fortml_hamiltonian_mlp`

`hamiltonian_mlp_t%initialize(n_coordinates,potential_layers,kinetic_layers,
status[,hidden_activation,initialization_seed])` builds a separable Hamiltonian
`H(q,p)=V(q)+T(p)` from two scalar MLPs. `energy` and `energy_gradient` expose
the scalar energy and canonical gradient, while `vector_field` returns
`(dH/dp,-dH/dq)` in the row-oriented state layout `[q,p]`. Packed parameters
place the potential network before the kinetic network. `energy_jvp` and
`energy_vjp` differentiate with respect to both state and packed parameters.
`vector_field_jvp` supplies the corresponding mixed product. `leapfrog` is an
explicit velocity-Verlet split map and is symplectic for the separable model.
`hamiltonian_mlp_t%initialize_general(n_coordinates,layers,status[,hidden_activation,
initialization_seed])` instead builds one scalar MLP with input `[q,p]`, so it
represents a general nonseparable `H(q,p)`. The same energy, canonical
vector-field, JVP, VJP, and HVP-backed `vector_field_jvp` products then act on
the full state and packed parameter vector; `is_general()` reports the mode.
The explicit leapfrog method refuses general mode with
`FORTNUM_NOT_IMPLEMENTED`, because a nonseparable Hamiltonian needs an
implicit symplectic integrator. Both modes refuse nonfinite states/directions
and malformed layer or output shapes; no finite-difference fallback is used.

### `fortml_symplectic`

`symplectic_form_diagnostic_t` evaluates the canonical form defect for a map
Jacobian `A` in `[q,p]` coordinates, `D = transpose(A) * Omega * A - Omega`,
with `Omega = [0,I;-I,0]`. `residual` packs `D` in column-major order and
`value` returns the positive weighted reduction
`weight*dot_product(D,D)/(2*n_state**2)`. `residual_jvp` and `residual_vjp`
are exact products of the form map. `value_jvp` and `value_vjp` contract the
same products without forming a Jacobian of the residual. `is_symplectic`
checks the maximum packed defect against a caller-supplied tolerance.

`symplectic_constraint_t` accepts caller-owned map Jacobian, Jacobian-JVP, and
Jacobian-VJP callbacks and exposes the same value and first-order products. Its
`as_constraint` method adapts the raw form residual to
`physics_constraint_t`, preserving the configured reduction weight for
`physics_objective_t` composition. The callbacks own model parameters,
coordinates, and any integrator state. A CPU diagnostic is the current
implementation. CUDA initialization or selection returns
`FORTNUM_NOT_IMPLEMENTED` and leaves the CPU object and output buffers
unchanged. No finite-difference or host fallback is hidden behind a product.
The independent harmonic-oscillator Verlet gate is `test_symplectic`.

### `fortml_physics_objective`

`physics_constraint_t` is the explicit residual seam for PINN, physics-informed
GP, and conservation adapters. `initialize` binds a positive weight, a
parameter/residual shape, and caller-owned residual, JVP, and VJP callbacks;
the optional `hvp_proc` callback is a reverse-over-forward product accepting
normalized residual and residual-JVP cotangents. When supplied, `hvp` returns
the exact weighted least-squares Hessian-vector product without forming a
Jacobian or Hessian. Without it, `hvp` retains the typed
`FORTNUM_NOT_IMPLEMENTED` refusal. `physics_objective_t` composes four named slots—`data`,
`residual`, `boundary`, and `conservation`—and sums their value/gradient and
first-order products. `term_values(theta, values, status)` returns the four
normalized weighted slot contributions in that order; inactive slots are zero,
and their sum equals `value(theta)`. This is the stable residual/conservation
balancing diagnostic. `as_objective` adapts the value/gradient path to FortOpt.
Callbacks own coordinate/time layouts, units, derivative implementation, and
device residency;
there is no hidden finite-difference or host/GPU fallback. See
[`docs/PHYSICS_MODELS.md`](PHYSICS_MODELS.md) and the independent
`test_physics_objective` affine-residual oracle.

### `fortml_pinn`

`pinn_training_adapter_t` is the bounded training facade over an initialized
`physics_objective_t`. `initialize(objective,status[,device_kind])` retains the
callback-owned objective and exposes `value`, `gradient`, `value_gradient`,
`jvp`, `vjp`, `hvp`, and the four-slot `term_values` diagnostic. `hvp` is exact
when every active constraint supplies the optional
`physics_residual_hvp_proc`; otherwise it preserves the typed refusal. The
`fit_lbfgsb(parameters,lower,upper,options,result,status)` method runs the
same value/gradient graph through FortOpt's bounded L-BFGS-B optimizer. The
adapter is CPU-only in this slice: CUDA initialization and selection return
`FORTNUM_NOT_IMPLEMENTED` and never relabel a host execution. See
[`docs/PHYSICS_MODELS.md`](PHYSICS_MODELS.md) and the independent
`test_pinn` manufactured-solution oracle.

### `fortml_mlp_training`

`mlp_train(model,x,target,status,options,state[,validation_x,validation_target,checkpoint])`
trains an existing `mlp_t` with deterministic Adam, AMSGrad, RAdam, Lion, AdamW, Adagrad, RMSprop,
unfactored Adafactor, or FortOpt-backed SGD. A zero `batch_size`
selects full-batch updates.
Mini-batch shuffling uses an explicit Park-Miller stream controlled by
`shuffle_seed`, and does not mutate process-global random state. The options
also provide optimizer selection (`MLP_OPTIMIZER_ADAM`, `MLP_OPTIMIZER_SGD`,
`MLP_OPTIMIZER_ADAMW`, `MLP_OPTIMIZER_ADAGRAD`, `MLP_OPTIMIZER_RMSPROP`,
`MLP_OPTIMIZER_ADAFACTOR`, `MLP_OPTIMIZER_AMSGRAD`, `MLP_OPTIMIZER_RADAM`,
or `MLP_OPTIMIZER_LION`),
learning-rate and Adam
coefficients, optional SGD
momentum/Nesterov acceleration, L2 regularization, gradient tolerance,
patience, best-state restoration, and an epoch callback.
To route different parameter blocks through one shared optimizer state,
allocate `options%optimizer_groups(:)` and initialize each
`mlp_optimizer_group_t` with a non-overlapping one-based `[first,last]`
range and a positive `learning_rate_multiplier`. After the optimizer updates
its moments, the selected block receives that multiplier on the complete
parameter delta; uncovered parameters use one. This post-update definition
is identical for Adam, AdamW, Adagrad, RMSprop, Adafactor, RAdam, and SGD (including
AdamW's decoupled decay), keeps moment state shared, and is deterministic.
Duplicate names, overlapping ranges, non-finite/non-positive multipliers, and
ranges beyond the model parameter count are rejected before the first update.
`MLP_OPTIMIZER_RMSPROP` uses the canonical FortOpt running squared-gradient
recurrence. Set `rmsprop_centered` to use the centered variance estimate and
`rmsprop_momentum` for optional classical momentum. The running square,
optional running gradient mean, momentum buffer, optimizer kind, and step
counter are checkpointed and restored exactly; optimizer-trajectory RMSprop
derivatives remain explicitly refused. A separate exact fixed full-batch
RMSprop trajectory adapter is documented below.
`MLP_OPTIMIZER_ADAMW` uses the same bias-corrected moments as Adam and applies
decoupled multiplicative `weight_decay` after each update. Weight decay is
validated as finite and non-negative, is checkpointed with the optimizer
configuration, and is compared on resume. The update trajectory is analytic
for fixed options; optimizer-trajectory hypergradients through learning rate,
decay, moments, and stopping remain a separate capability.
`MLP_OPTIMIZER_ADAGRAD` uses FortOpt's canonical accumulated-square
recurrence, `G <- G + gradient**2`, followed by the epsilon-stabilized diagonal
step. Its accumulator and step counter are checkpointed and restored exactly.
`MLP_OPTIMIZER_ADAFACTOR` uses the same explicit unfactored vector recurrence
as `FORTML_TRAIN_ADAFACTOR`, with `adafactor_decay`,
`adafactor_clip_threshold`, optional relative-step and parameter scaling. Its
second moment and step counter are checkpointed and compared on resume; true
matrix-factorized state is available by setting
`options%adafactor_factored=.true.`. The layout-aware path factors dense weight
blocks and keeps bias/singleton blocks unfactored; see
[`ADAFACTOR_FACTORED.md`](ADAFACTOR_FACTORED.md). Its ragged row/column state is
included in the in-memory and formatted schema-11 checkpoints, with block shape
metadata and transactional layout validation; CUDA-resident Adafactor remains
a typed refusal.
`MLP_OPTIMIZER_AMSGRAD` keeps the Adam first and second moments plus an
elementwise maximum second moment. Bias correction is applied to both moments,
and the maximum is checkpointed in `max_second_moment` (in-memory and text
format 8). The independent `test_mlp_amsgrad` fixture checks the
recurrence and formatted checkpoint continuation. AMSGrad remains CPU-only,
and there is no hidden CUDA fallback. `fortml_mlp_amsgrad_hypergradient` adds
exact fixed full-batch value/gradient/JVP/VJP products through the max state
with a bounded FortOpt L-BFGS-B adapter. Max ties, zero square-root or update
denominators, and CUDA trajectory requests are typed refusals. See
[`MLP_AMSGRAD_HYPERGRADIENT.md`](MLP_AMSGRAD_HYPERGRADIENT.md).
`MLP_OPTIMIZER_RADAM` keeps the same bias-corrected first and second moments,
then applies the RAdam variance-rectification factor once `rho_t > 4`; before
that threshold it uses the bias-corrected first moment. The two moment arrays,
step count, and common beta/epsilon configuration are captured in the
in-memory and text-schema-11 checkpoints. The independent
`test_mlp_radam` fixture checks both sides of the threshold, uninterrupted versus
formatted checkpoint resume, invalid hyperparameters, and the CPU/CUDA device
boundary. RAdam is CPU-only in this slice: `radam_t%step_device` returns
`FORTNUM_NOT_IMPLEMENTED` for CUDA without modifying parameters. Optimizer-
trajectory hypergradients, matrix/device state, and FortOpt RAdam adapters are
deliberately not claimed yet.
`MLP_OPTIMIZER_LION` uses the beta1 interpolation of the current gradient and
the stored beta2 momentum for its sign update, then advances the momentum with
the beta2 recurrence. `weight_decay` is decoupled from the loss gradient,
gradient clipping is applied before the sign branch, and the momentum vector is
stored in the checkpoint `first_moment` slot. The independent
`test_mlp_lion_training` oracle checks the recurrence, EMA, and uninterrupted
versus resumed trajectories. The production trainer is FP64 CPU; resident CUDA
Lion state and differentiable sign-branch products remain typed follow-up
capabilities. See [`MLP_LION_TRAINING.md`](MLP_LION_TRAINING.md).
The separate `mlp_adagrad_hypergradient_objective_t` provides that fixed full-batch product
over learning rate, L2, and epsilon; it is CPU-only and routes exact
value/JVP/VJP products to FortOpt L-BFGS-B with explicit log bounds. Mini-batch,
schedule, clipping, and CUDA Adagrad trajectories remain refused until their
state derivatives and residency contracts are complete.

`mlp_training_state_t` records epoch and update counts, the best epoch and
loss, the final loss and gradient norm, convergence flags, a compact loss
history, the effective learning rate per epoch, and the number of norm-clipped
updates. Passing both optional validation arrays evaluates a held-out MSE at
each `validation_interval`. Patience and best-state restoration monitor that
validation objective, and the state records initial/best/final validation
losses, the best validation epoch, and its history. The pair must be supplied
together and is finite, shape-checked, and never used for parameter updates.
`mlp_loss_value_gradient` returns the mean-squared-error value, the
packed network gradient, and the analytic derivative with respect to the
scalar L2 hyperparameter. This scalar product is the first outer
hyperparameter-search seam for neural training. `mlp_loss_hvp` adds the exact joint Hessian-vector
product for a parameter direction and an L2 direction, including the mixed
parameter/L2 block used by outer hyperparameter optimization. The HVP is
checked against independent central differences for linear and nonlinear MLP
fixtures. The fixed full-batch learning-rate/L2 trajectory contract is provided
by `fortml_mlp_hypergradient`; stochastic optimizer trajectories, schedules,
and Adam-beta hypergradients remain separate contracts.

The data term is evaluated through the shared `fortml_losses` weighted-MSE
value/VJP/HVP kernels. Optional row weights and mean/sum reduction therefore
have the same semantics in standalone loss calls and in the MLP objective.

The same loss entry point accepts optional `sample_weight`, `reduction`, and
`diagnostics` arguments. `MLP_REDUCTION_MEAN` divides the weighted data loss
and gradient by positive weight mass, while `MLP_REDUCTION_SUM` leaves the
weighted data sum unnormalized. Weights must be finite, non-negative, and have
positive mass. L2 remains a single parameter regularizer in either reduction.
`mlp_loss_diagnostics_t` reports `data_loss`, `regularization_loss`,
`weight_mass`, and `sample_count`, so callers can log named scalar components
without reconstructing the reduction.

The shared `fortml_losses` facade also exposes stable softmax and log-softmax
value/JVP/VJP products plus explicit-cotangent HVPs
(`softmax_hvp`/`log_softmax_hvp`). `softmax_cross_entropy_value`,
`softmax_cross_entropy_jvp`, `softmax_cross_entropy_vjp`, and
`softmax_cross_entropy_hvp` accept optional row `sample_weight` and
`LOSS_REDUCTION_MEAN`/`LOSS_REDUCTION_SUM` arguments. Mean reduction divides
by positive row-weight mass; sum reduction is unnormalized. The focal
binary-cross-entropy-with-logits family additionally provides an exact
`focal_binary_cross_entropy_with_logits_hvp` for relaxed `[0,1]` targets and
nonnegative focusing exponent. Shape, finite-input, alpha/gamma, label, and
zero-support weight violations return a domain status. Device-dispatch value
wrappers route CPU to this reference and return `FORTNUM_NOT_IMPLEMENTED` for
CUDA until resident loss kernels exist; they never relabel a host fallback as
GPU execution. See [`NEURAL_LOSS_PRODUCTS.md`](NEURAL_LOSS_PRODUCTS.md).

The multiclass focal-softmax family (`focal_softmax_cross_entropy_*`, with
`multiclass_focal_cross_entropy_*` aliases) uses
`a(class)*(1-p(class))**gamma*(-log(p(class)))`. It accepts optional positive
class factors and row weights, supports both reductions, and returns analytic
value/JVP/VJP/HVP products. `gamma=0` is the ordinary weighted softmax
cross-entropy limit; true-class probability underflow is a typed domain
refusal. `focal_softmax_cross_entropy_value_device` keeps the scalar unchanged
for an unavailable CUDA request. Multiclass MLP fit and FortOpt options expose
the same contract through `focal_gamma`.

`mlp_training_checkpoint_t` is the in-memory resumable trainer state. Pass an
uninitialized checkpoint to `mlp_train` to capture it after each completed
epoch (and at every microbatch boundary). Pass the initialized checkpoint back
to a later call to resume. `options%max_epochs` is the total target epoch, not
an additional count. The snapshot includes packed model parameters,
Adam/AdamW first and second moments (or Adagrad accumulated squares, RMSprop
running statistics, or SGD velocity) plus optimizer step and configuration,
permutation/order and Park--Miller state, active epoch/microbatch cursor and
accumulated gradient, learning-rate schedule position/history, validation and
early-stopping counters, and the best-parameter state. A typed built-in
schedule selected by `options%use_typed_schedule` is serialized with its kind,
update counts, and fractions; the resumed options must provide the same
schedule.
When optimizer groups are configured, their ranges and multipliers are also
captured and compared on resume; the versioned text schema stores the same
metadata. Group names are caller-owned labels and do not affect the update
trajectory.
Procedure pointers are not serializable: install the same deterministic
callback on the resumed options. A checkpoint
is rejected when dimensions, batch/accumulation policy, shuffle seed, optimizer
configuration, Adam coefficients, L2, validation/early-stopping policy, or clipping/tolerance
policy differ. A resumed call intentionally clears terminal convergence or
early-stop flags so increasing the total epoch target continues training.
Best-state restoration changes model parameters
after the last optimizer state and therefore marks that snapshot
`resume_safe=.false.`. Use `restore_best=.false.` when a run must be resumed.
`checkpoint%valid()` validates allocation, dimensions, finite values, and the
format version, while `checkpoint%clear()` releases its arrays.

### `fortml_mlp_optimizer_group_hypergradient`

`mlp_optimizer_group_hypergradient_options_t` exposes a fixed full-batch SGD
trajectory whose packed FortOpt coordinates are
`[log_learning_rate,log_l2,log(multiplier_i)]`. Group ranges and names are
validated as contiguous, non-overlapping metadata. Each multiplier is applied
to the post-optimizer parameter delta used by `mlp_train`; uncovered
parameters use multiplier one. Set `gradient_clip_norm` to apply the same
global norm clipping as `mlp_train` after L2 and before group scaling. Its
derivatives propagate on a fixed active set, while a trajectory on the clipping
boundary returns `FORTNUM_NOT_IMPLEMENTED`; the clipping norm is not a packed
coordinate. `value_gradient`, `jvp`, `vjp`, and the bounded FortOpt adapter
share the same analytic MLP HVP products. Momentum,
Adam-family state, schedules, minibatch cursors, and resident CUDA groups are
explicitly outside this adapter and return typed refusals where requested.
See [`docs/MLP_OPTIMIZER_GROUP_HYPERGRADIENT.md`](MLP_OPTIMIZER_GROUP_HYPERGRADIENT.md)
and `test_mlp_optimizer_group_hypergradient` for the independent oracle.

Set `mlp_training_options_t%ema_decay` to a finite value in `[0,1)` to track
an exponential moving average of post-update network parameters. Zero (the
default) disables the extra state. The initialized average starts at the
initial parameter vector and is updated after every optimizer step as
`ema <- decay*ema + (1-decay)*parameters`. The resulting
`mlp_training_state_t%ema_parameters` is a diagnostic/export surface; training
does not silently replace the live model with the averaged parameters. EMA
state, decay, and resume validation are included in checkpoints and the
versioned text schema, so interrupted and uninterrupted deterministic runs
produce identical averages. If best-state restoration is enabled, the model
may be restored after the final EMA update and the checkpoint is marked
non-resumable as for any other parameter/state mismatch.

`fortml_mlp_checkpoint` adds the file boundary without exposing compiler
runtime state. `mlp_checkpoint_save(checkpoint,path,status)` writes a valid
snapshot as versioned formatted text with the magic
`FORTML_MLP_CHECKPOINT_TEXT`; `mlp_checkpoint_load(checkpoint,path,status)`
reads it into a temporary value, validates every scalar and array, and only
then replaces the destination. The schema records all optimizer variants,
including Adam/AdamW moments, Adagrad squares, RMSprop square/mean/momentum
state, or SGD velocity, together with the exact iterator permutation and
shuffle stream. Schema version 4 also records typed schedule and optimizer-group fields. Real
values use 17 significant decimal digits, and array
lengths and record names are explicit, so files are independent of compiler
endianness and unformatted-record conventions. Unknown schema versions,
truncation, extra records, invalid values, and optimizer-state shape mismatches
are refused with a non-OK status. Procedure pointers remain intentionally
outside the file contract and must be installed by the resumed caller.

`mlp_training_objective_t` packages the same objective for FortOpt. Call
`initialize(model,x,target,l2,status[,optimize_l2])`, then use `parameters`,
`parameter_count`, `value_gradient`, `jvp`, `vjp`, and `hvp`. The scalar JVP
contracts the packed direction with the analytic value gradient; the scalar VJP
scales that gradient by its output cotangent. With `optimize_l2=.true.`, the
packed vector appends the non-negative L2 coefficient and both the gradient and
HVP include its mixed block. `fortopt(objective,status)` installs a context
callback directly into `fortopt_objective`. An L-BFGS-B caller can therefore
optimize network parameters and L2 with analytic products and explicit bounds.

For a complete bounded fit, `mlp_optimize_lbfgsb(model,x,target,options,result,status)`
owns that adapter and the FortOpt `lbfgsb_t` lifecycle. The
`mlp_lbfgsb_options_t` bounds the packed network block and, when
`optimize_l2=.true.`, appends a bounded L2 hyperparameter to the same vector.
The result reports convergence, iterations, line-search evaluations, objective,
gradient norm, and the final L2 value. This is a deterministic full-batch
optimizer path. Mini-batch Adam, Adagrad, and SGD (with optional
momentum/Nesterov acceleration) are available stochastic trainers.
The optimizer consumes the analytic value/gradient path above, so no
finite-difference hyperparameter approximation is introduced.

### `fortml_mlp_poisson`

`mlp_poisson_training_objective_t` composes an initialized one-output MLP with
the Poisson negative log likelihood in log-rate coordinates. The target matrix
has shape `(n_samples,1)` and contains finite non-negative counts or relaxed
count weights. Optional sample weights use the positive-mass mean reduction
implemented by `fortml_losses`.

The packed vector contains the network parameters. With `optimize_l2=.true.`
it appends the non-negative L2 coefficient. `value_gradient`, `jvp`, `vjp`, and
`hvp` are analytic. The HVP combines the Poisson output curvature with the
MLP reverse-over-forward product and includes the exact mixed L2 block. The
same objective is available through `fortopt(objective,status)`, and
`mlp_poisson_optimize_lbfgsb` supplies bounded FortOpt L-BFGS-B controls and
result diagnostics. The independent `test_mlp_poisson_objective` fixture
checks central differences, JVP contraction, VJP scaling, an L-BFGS-B solve,
and malformed CUDA requests. The objective currently accepts CPU state only,
so a CUDA request returns `FORTNUM_NOT_IMPLEMENTED` before fitting.

### `fortml_mlp_grouped_training`

`mlp_parameter_group_t` names a contiguous network-parameter range and stores
its initial log regularization coefficient.  Build one or more groups with
`initialize(name,first,last,log_l2,status)`, then pass the array to
`mlp_grouped_training_objective_t%initialize(model,x,target,groups,status)`.
The objective's packed vector is

```text
[ network parameters, log(lambda_1), ..., log(lambda_group_count) ]
```

where `lambda_i = exp(log(lambda_i))` and the safe log range is `[-50,50]`.
Groups must have unique names and non-overlapping one-based ranges; parameters
not included in a group remain unregularized.  `parameters`, `group_name`,
`group_range`, `parameter_count`, and `group_count` expose the stable packing
metadata.  `value_gradient` adds the group penalties and their exact
log-coefficient derivatives.  `jvp`, `vjp`, and `hvp` are analytic; the HVP
contains both the `lambda_i dtheta_i + lambda_i theta_i dlog(lambda_i)` block
and the corresponding scalar mixed block.  Thus the same product can be passed
to `fortopt_objective` through `fortopt` and optimized with FortOpt L-BFGS-B
using explicit bounds on network and log-coefficient coordinates.

For a complete grouped fit, `mlp_grouped_optimize_lbfgsb(model,x,target,groups,
options,result,status)` owns the FortOpt `lbfgsb_t` lifecycle. The
`mlp_grouped_lbfgsb_options_t` bounds the network block and all log-L2
coordinates independently (equal bounds intentionally freeze a coordinate),
while `mlp_grouped_lbfgsb_result_t` reports convergence, iteration and line
search counts, objective, gradient norm, and final group log coefficients.
The adapter uses the exact grouped value/gradient callback; it does not
finite-difference hyperparameters. `device_kind=FORTML_DEVICE_CUDA` returns
`FORTNUM_NOT_IMPLEMENTED` until a resident dense-MLP derivative graph exists,
so a CUDA request cannot silently execute the CPU optimizer.

The independent `test_mlp_grouped_training` fixture checks the value gradient,
mixed HVP, JVP central-difference oracle, scalar VJP, and FortOpt callback
against a hand-derived linear ridge objective. It also checks the bounded
grouped L-BFGS-B fit against the closed-form solution and verifies the typed
CUDA refusal. The objective currently has a CPU-only dense graph:
`device_kind=FORTML_DEVICE_CUDA` returns `FORTNUM_NOT_IMPLEMENTED`; no host
fallback is hidden behind a CUDA request.

`mlp_batch_iterator_t` is the reusable deterministic row-index cursor used by
`mlp_train`. Initialize it with `n_samples`, an optional `batch_size`,
`shuffle`, and positive `seed`. Call `reset` once per epoch and
`next_batch(indices,has_batch,status)` until `has_batch` is false. The final
batch is returned without padding, and the cursor never advances implicitly to
the next epoch. Its explicit position, epoch, and copied RNG state make an
in-memory batch boundary resumable. `mlp_learning_rate_schedule_proc` can be
installed in `mlp_training_options_t%learning_rate_schedule`. It receives the
epoch, one-based update number, and base rate and must return a finite positive
rate. For a portable built-in schedule, pass
`options%use_typed_schedule=.true.` and assign
`options%typed_schedule=mlp_learning_rate_schedule_t(...)`. The typed schedule
is validated once, evaluated analytically at every update, and cannot
be combined with a callback. Its kind and continuous fields are retained in
the in-memory and formatted-text checkpoint, so resumed runs reject a
different schedule instead of silently changing the trajectory. The
checkpoint schema is versioned for this state. `gradient_clip_norm` applies
global norm clipping before each selected optimizer step.
Zero disables clipping. `accumulation_steps` combines that many consecutive
microbatches into one sample-weighted mean gradient before an Adam or SGD step. The
last uneven group is flushed at the epoch boundary. L2 is added once per
optimizer update, clipping is applied after accumulation, and the state records
`microbatches`, `updates`, and the configured accumulation count. This gives an
exact full-batch equivalence oracle when the model and options are otherwise
identical, while reducing optimizer-state updates for memory-constrained
training.

`fortml_mlp_schedules` supplies stateless built-in schedule values for that
callback seam: constant, linear warm-up, cosine decay, warm-up plus cosine,
exponential decay, one-cycle warm-up/cosine, and metric-aware plateau
schedules.
`mlp_learning_rate_schedule_t%rate` validates the
update and returns a finite positive rate; `rate_with_derivatives` additionally
returns exact products with respect to the base rate, minimum-rate fraction,
and decay factor. `rate_with_full_derivatives` also returns exact products with
respect to one-cycle peak and final rate fractions (zero for other families).
The schedule receives an explicit update index rather than
owning hidden mutable state, so a replayed training run uses the same rates.
The plateau constructor takes `patience_updates`, `min_delta`, `factor`, and an
optional minimizing or maximizing metric mode. Its
`rate_with_metric_derivatives` method takes the current metric, best metric,
bad-observation count, and reduction count and returns their explicit next
state. The rate is `base_rate*factor**next_reductions`. Base-rate and factor
products are exact on the selected branch. Metric, best-metric, and
`min_delta` products are documented zeros because the comparison is a discrete
active-set decision. Integer update and patience fields have no products.
Calling ordinary `rate` with a plateau schedule returns
`FORTNUM_NOT_IMPLEMENTED` because a metric state is required. `mlp_train` now
owns that state at epoch boundaries: it monitors validation loss when a held-out
stream is supplied (training loss otherwise), and preserves the best metric,
bad-observation count, and reduction count in version-10 in-memory and
formatted checkpoints and in the returned training state. Split checkpoint/resume therefore reproduces the
uninterrupted plateau trajectory. The metric-aware method remains available for
custom trainers and its active-set products retain the same typed boundaries.
`device_supported(kind)` reports CPU-only support in this release: schedules
have no resident CUDA optimizer lowering yet, so they must not be timed as GPU
workloads or used to imply device-resident trajectory hypergradients. A CUDA
typed schedule request therefore remains an explicit capability boundary in
the trajectory APIs.
See [`docs/MLP_SCHEDULES.md`](MLP_SCHEDULES.md) for constructors and a
callback adapter.

`mlp_loss_scale_state_t` is the explicit automatic-mixed-precision policy and
dynamic state embedded in `mlp_training_options_t`, the returned
`mlp_training_state_t`, and `mlp_training_checkpoint_t`. Call
`options%loss_scale%initialize(status, enabled=.true., initial_scale=...,`
`growth_factor=..., backoff_factor=..., growth_interval=...,`
`minimum_scale=..., maximum_scale=...)` to configure deterministic growth and
overflow backoff. `observe(finite_gradient,update_applied,status)` is public
for custom trainers and increments the same good-step, overflow, and skipped
update counters used by `mlp_train`. The FP64 reference path checks scaled
gradients and skips an overflowing update; disabled scaling leaves the existing
trajectory unchanged. FP32/FP16/BF16 and CUDA resident training return a typed
`FORTNUM_NOT_IMPLEMENTED` until master-weight and resident lower-precision
kernels are independently gated. Loss-scale policy and dynamic counters are
validated and persisted in formatted checkpoint schema 11.

`mlp_training_options_t%event_callback` installs a typed
`mlp_training_event_proc` callback for `train_begin`, `update`, `validation`,
`epoch_end`, `checkpoint`, and `train_end` events. Each event carries the
epoch/update counters and current loss, validation loss, gradient norm, and
learning rate. The callback can request an early stop or return a non-success
status; callback failures propagate from `mlp_train` without being swallowed.
The callback receives caller-owned scalar data and is not serialized in an
in-memory checkpoint.

### `fortml_mlp_hypergradient`

`mlp_hypergradient_objective_t` provides an exact, deliberately bounded outer
objective for a fixed full-batch gradient-descent trajectory. The packed outer
vector is `[log(learning_rate), log(l2)]`; the model parameters at
`initialize` are the fixed inner starting state. Every evaluation applies the
configured number of updates and returns unregularized validation MSE.
`value_gradient` and `vjp` reverse the stored trajectory, while `jvp` pushes a
tangent through each update using the analytic MLP HVP. The
`mlp_optimize_hyperparameters` adapter sends these products directly to
FortOpt L-BFGS-B in log space. Adam, momentum/Nesterov, mini-batch or schedule
trajectories, and CUDA-resident state return `FORTNUM_NOT_IMPLEMENTED` until
their state and reproducibility contracts have matching derivative products;
they are never approximated by hidden finite differences. See
[`docs/MLP_HYPERGRADIENT.md`](MLP_HYPERGRADIENT.md) for the complete layout and
example.

### `fortml_mlp_sgd_momentum_hypergradient`

`mlp_sgd_momentum_hypergradient_objective_t` provides the exact fixed
full-batch SGD momentum/Nesterov trajectory contract. Its packed vector is
`[log(learning_rate),log(l2),momentum]`; the velocity state and, for Nesterov,
the look-ahead direction are differentiated with the analytic MLP HVP. Both
classical and Nesterov modes expose `value_gradient`, `jvp`, scalar `vjp`, and
a bounded FortOpt L-BFGS-B adapter through
`mlp_optimize_sgd_momentum_hyperparameters`. The Nesterov branch is a fixed
discrete choice and requires a positive momentum bound. Mini-batch, schedules,
clipping, stochastic/device-resident state, and CUDA products remain typed
refusals until their full state derivatives are available. The `hvp` entry
point is exact for a one-layer MLP with linear output (constant network Hessian) and
returns `FORTNUM_NOT_IMPLEMENTED` for nonlinear or multilayer models. An
optional `validation_weight(:)` argument defines a finite positive-support
weighted validation mean and is differentiated by value/JVP/VJP and FortOpt.
The affine HVP is certified for uniform weights; non-uniform weights return a
typed HVP refusal until the residual-weighted second contraction is generalized.
Invalid weight vectors are rejected without mutating the objective. See
[`docs/MLP_SGD_MOMENTUM_HYPERGRADIENT.md`](MLP_SGD_MOMENTUM_HYPERGRADIENT.md).

`mlp_adamw_hypergradient_objective_t` provides the corresponding exact
full-batch AdamW trajectory contract. Its packed vector is
`[log(learning_rate),log(l2),log(weight_decay)]`; bias-corrected first and
second moments, decoupled decay, and the analytic MLP HVP are differentiated
without finite differences. `value_gradient`, `jvp`, and scalar `vjp` are
available, and `mlp_optimize_adamw_hyperparameters` routes the products to
FortOpt L-BFGS-B with explicit log bounds. Mini-batch, schedules, beta
hypergradients, and CUDA state remain refused until their complete state
derivatives are specified.

`mlp_adamw_schedule_hypergradient_objective_t` extends this contract with a
fixed typed schedule. Its packed vector is
`[log(base_rate),log(l2),log(weight_decay),logit(beta1),logit(beta2),
log(epsilon),logit(min_rate_fraction),logit(decay_factor)]`. Constant, cosine,
warmup-cosine, and exponential-decay schedules expose exact CPU
`value_gradient`, `jvp`, scalar `vjp`, and FortOpt L-BFGS-B products through
AdamW moments, bias correction, and decoupled decay. Schedule shape and integer
update counts are fixed. The outer `hvp`, CUDA, and lower-precision requests
are typed refusals until resident state and third network derivatives exist;
see [`docs/MLP_ADAMW_SCHEDULE_HYPERGRADIENT.md`](MLP_ADAMW_SCHEDULE_HYPERGRADIENT.md).

`mlp_rmsprop_hypergradient_objective_t` provides the exact fixed full-batch
RMSprop trajectory contract. Its packed vector is
`[log(learning_rate),log(l2),decay,log(epsilon),momentum]`. Square-average,
centered gradient-average, and momentum-buffer recurrences are differentiated
with the analytic MLP HVP; both centered and uncentered modes use the same
value/JVP/VJP products. `mlp_optimize_rmsprop_hyperparameters` routes these
products to FortOpt L-BFGS-B with explicit bounds for every packed component.
The `centered` flag is a fixed discrete branch rather than a differentiable
variable, and changing it requires a new objective adapter. Mini-batch,
schedules, clipping, and CUDA-resident RMSprop state remain typed follow-up
contracts until their state and reproducibility derivatives are implemented.

`mlp_adagrad_hypergradient_objective_t` provides the corresponding exact
fixed full-batch Adagrad trajectory contract. Its packed vector is
`[log(learning_rate),log(l2),log(epsilon)]`; the accumulated-square state and
epsilon-stabilized diagonal step are differentiated with the analytic MLP HVP.
`value_gradient`, `jvp`, and scalar `vjp` are available, and
`mlp_optimize_adagrad_hyperparameters` routes the same products to FortOpt
L-BFGS-B under explicit log bounds. The independent
`test_mlp_adagrad_hypergradient` fixture checks central differences, a
directional JVP, the scalar adjoint, optimizer convergence, and typed
non-Adagrad/CUDA refusals. Mini-batch, schedules, clipping, and CUDA-resident
Adagrad state remain explicit follow-up contracts.

### `fortml_mlp_adafactor_hypergradient`

`mlp_adafactor_hypergradient_objective_t` differentiates the fixed full-batch
unfactored Adafactor trajectory used by `fortml_trainer`. Its packed vector is
`[log(learning_rate),log(l2),decay,log(epsilon),log(clip_threshold)]`.
The exact CPU products propagate the second-moment, update-RMS clipping, and
epsilon-stabilized denominator states; `value_gradient`, `jvp`, scalar `vjp`,
and `mlp_optimize_adafactor_hyperparameters` are available for FortOpt
L-BFGS-B. Relative-step and parameter-scaling modes are fixed discrete
branches, exact clip-boundary trajectories refuse, and CUDA is a typed refusal
until a resident trajectory graph exists. The independent
`test_mlp_adafactor_hypergradient` fixture covers all five coordinates,
finite-difference and adjoint products, active-set behavior, and optimizer
integration. See [`docs/MLP_ADAFACTOR_HYPERGRADIENT.md`](MLP_ADAFACTOR_HYPERGRADIENT.md).

### `fortml_mlp_lion_hypergradient`

`mlp_lion_hypergradient_objective_t` differentiates a fixed full-batch Lion
trajectory. Its packed outer vector is
`[log(learning_rate),log(l2),logit(beta1),logit(beta2)]`. The analytic CPU
products propagate the first-moment state, sign-margin update, and validation
loss through the MLP HVP. `value_gradient`, `jvp`, scalar `vjp`, and
`mlp_optimize_lion_hyperparameters` feed the same products to FortOpt
L-BFGS-B. A configured sign-margin neighborhood is a named nonsmooth refusal;
CUDA trajectory requests return `FORTNUM_NOT_IMPLEMENTED` until the complete
state is resident. The independent fixture is
`test_mlp_lion_hypergradient`; see
[`docs/MLP_LION_HYPERGRADIENT.md`](MLP_LION_HYPERGRADIENT.md).

### `fortml_mlp_adam_hypergradient`

`mlp_adam_hypergradient_objective_t` differentiates a fixed full-batch Adam
trajectory with *coupled* L2 regularization. Its packed vector is
`[log(learning_rate),log(l2),logit(beta1),logit(beta2)]`; the logits preserve
the open `(0,1)` moment domain while the outer optimizer explores bounded
coordinates. The regularized loss gradient enters both moment recurrences,
which distinguishes this API from the decoupled shrinkage in AdamW. The
parameter, first-moment, second-moment, and bias-correction state tangents use
the analytic MLP HVP. `value_gradient`, `jvp`, scalar `vjp`, and
`mlp_optimize_adam_hyperparameters` are exact CPU products and a direct FortOpt
L-BFGS-B adapter. The independent `test_mlp_adam_hypergradient` fixture checks
central differences for all four coordinates, a directional JVP, the scalar
adjoint, optimizer use, and the typed CUDA refusal. Mini-batch, schedules,
stochastic state, and resident GPU Adam products remain explicit follow-up
contracts.

### `fortml_mlp_schedule_hypergradient`

`mlp_schedule_hypergradient_objective_t` differentiates a fixed full-batch MLP
trajectory using one of the stateless typed schedules from
`fortml_mlp_schedules`. `mlp_schedule_hypergradient_options_t%schedule` fixes
the schedule kind and integer warm-up/total-update shape. For ordinary
schedules the packed outer vector is `[log(base_rate), log(l2),
logit(min_rate_fraction), logit(decay_factor)]`; for one-cycle it is
`[log(base_rate), log(l2), log(peak_rate_fraction),
log(final_rate_fraction)]`, identified by `metadata%one_cycle_coordinates`.
The one-cycle peak/final products are exact through both the linear warm-up
and cosine tail, while unused fields retain exact zero derivatives.
`value_gradient`, `jvp`, and scalar `vjp` reverse or push through every update
with the analytic MLP HVP. `mlp_optimize_schedule_hyperparameters` feeds the
same reverse product to FortOpt L-BFGS-B under explicit log/logit bounds
(the two existing fraction-bound fields are logarithmic peak/final bounds for
one-cycle).
The `hvp` entry point is exact for one-layer affine networks with constant,
linear warm-up, cosine, warm-up-plus-cosine, exponential, and one-cycle
schedules. `rate_with_second_derivatives` supplies the raw schedule rate
Hessian, and the objective propagates mixed state tangents through the affine
MSE Hessian; transformed log/logit coordinates are included exactly. Nonlinear
networks require third network derivatives and return a typed
`FORTNUM_NOT_IMPLEMENTED` boundary, as do metric plateau branches.

The independent `test_mlp_schedule_hypergradient` fixture checks all packed
components with central differences, a directional JVP, the scalar adjoint,
the one-cycle coordinate metadata and domain, the FortOpt context callback,
and the L-BFGS-B smoke solve, including cosine and one-cycle affine HVP finite
differences. CUDA trajectory requests return `FORTNUM_NOT_IMPLEMENTED` until a
resident MLP trajectory kernel exists; the benchmark therefore records an
explicit capability refusal. See
[`docs/MLP_SCHEDULE_HYPERGRADIENT.md`](MLP_SCHEDULE_HYPERGRADIENT.md).

### `fortml_mlp_adagrad_schedule_hypergradient`

`mlp_adagrad_schedule_hypergradient_objective_t` combines the typed schedule
trajectory with the fixed full-batch Adagrad accumulator. Its packed outer
vector is `[log(base_rate), log(l2), log(epsilon), logit(min_rate_fraction),
logit(decay_factor)]`; schedule kind and integer update counts remain fixed.
The value/gradient, JVP, scalar VJP, and FortOpt adapter differentiate the
schedule rate, accumulated-square state, denominator, and parameter update
analytically. CUDA requests return `FORTNUM_NOT_IMPLEMENTED` until resident
MLP, schedule, and derivative state exists. The independent
`test_mlp_adagrad_schedule_hypergradient` fixture checks central differences,
directional products, adjointness, optimizer integration, malformed options,
and the CUDA refusal. See
[`docs/MLP_ADAGRAD_SCHEDULE_HYPERGRADIENT.md`](MLP_ADAGRAD_SCHEDULE_HYPERGRADIENT.md).

### `fortml_mlp_radam_hypergradient`

`mlp_radam_hypergradient_objective_t` differentiates a fixed full-batch RAdam
trajectory through its parameter, first-moment, second-moment, bias-correction,
and rectification state. The packed vector is `[log(learning_rate), log(l2),
logit(beta1), logit(beta2), log(epsilon)]`. `value_gradient`, `jvp`, and the
scalar `vjp` are exact CPU products based on the MLP loss HVP, and
`fortopt`/`mlp_optimize_radam_hyperparameters` expose the same registry to
bounded FortOpt L-BFGS-B. The `rho_t = 4` branch and zero second-moment square
roots return typed `FORTNUM_NOT_IMPLEMENTED` nonsmooth refusals; CUDA
trajectory requests use the same explicit refusal until resident state exists.
The independent `test_mlp_radam_hypergradient` fixture checks central
differences, a directional product, the scalar adjoint, FortOpt integration,
and both refusal contracts. See
[`docs/MLP_RADAM_HYPERGRADIENT.md`](MLP_RADAM_HYPERGRADIENT.md).

### `fortml_mlp_radam_schedule_hypergradient`

`mlp_radam_schedule_hypergradient_objective_t` extends fixed full-batch RAdam
with a typed stateless learning-rate schedule. Its packed vector is
`[log(base_rate), log(l2), logit(beta1), logit(beta2), log(epsilon),
logit(min_rate_fraction), logit(decay_factor)]`. Constant, cosine,
warmup-cosine, and exponential-decay schedule families have exact rate
sensitivities; inactive family fields return zero products. Value/gradient,
JVP, scalar VJP, and the FortOpt L-BFGS-B adapter propagate through moments,
bias correction, rectification, and schedule state. The outer hyper-HVP
requires third network derivatives and is a typed refusal, as are CUDA
trajectory requests, the `rho_t = 4` branch, and zero square-root derivatives.
See [`docs/MLP_RADAM_SCHEDULE_HYPERGRADIENT.md`](MLP_RADAM_SCHEDULE_HYPERGRADIENT.md)
and `test_mlp_radam_schedule_hypergradient`.

### `fortml_mlp_amsgrad_hypergradient`

`mlp_amsgrad_hypergradient_objective_t` differentiates a fixed full-batch
AMSGrad trajectory through the parameter, first-moment, second-moment, and
elementwise maximum-second-moment state. The packed vector is
`[log(learning_rate), log(l2), logit(beta1), logit(beta2), log(epsilon)]`.
`value_gradient`, `jvp`, and scalar `vjp` are exact CPU products based on the
MLP loss HVP. `fortopt` and `mlp_optimize_amsgrad_hyperparameters` expose the
same callback to bounded FortOpt L-BFGS-B. A max active-set tie, zero
second-moment square root, zero bias-correction or update denominator, and
CUDA trajectory request return typed `FORTNUM_NOT_IMPLEMENTED` refusals. The
independent `test_mlp_amsgrad_hypergradient` fixture checks central
differences, a directional product, the scalar adjoint, FortOpt integration,
and the nonsmooth/device boundaries. See
[`docs/MLP_AMSGRAD_HYPERGRADIENT.md`](MLP_AMSGRAD_HYPERGRADIENT.md).

### `fortml_mlp_minibatch_hypergradient`

`mlp_minibatch_hypergradient_objective_t` differentiates a fixed mini-batch
SGD trajectory. `mlp_minibatch_hypergradient_options_t` fixes the epoch count,
batch size, and optional private seeded shuffle stream. The packed outer vector
is `[log(learning_rate), log(l2)]`. Every objective evaluation replays the
recorded batch cursor, computes validation MSE after the final update, and
uses the MLP's analytic per-batch HVP for exact trajectory products.
`value_gradient`, `jvp`, and scalar `vjp` are available, and
`mlp_optimize_minibatch_hyperparameters` routes the same products to FortOpt
L-BFGS-B under explicit bounds. The outer `hvp` and CUDA trajectory paths
return typed refusals until third network derivatives and resident state are
available. The independent `test_mlp_minibatch_hypergradient` fixture checks
central differences, a directional JVP, the scalar adjoint, the FortOpt
callback, optimizer convergence, and the CUDA boundary. See
[`docs/MLP_MINIBATCH_HYPERGRADIENT.md`](MLP_MINIBATCH_HYPERGRADIENT.md).

### `fortml_mlp_minibatch_adam_hypergradient`

`mlp_minibatch_adam_hypergradient_objective_t` differentiates a fixed seeded
mini-batch Adam trajectory with the coupled-L2 convention used by the regular
MLP Adam optimizer. `mlp_minibatch_adam_hypergradient_options_t` fixes epochs,
batch size, shuffle seed, beta1, beta2, and epsilon. The packed outer vector is
`[log(learning_rate), log(l2)]`; each evaluation replays the private cursor and
computes validation MSE after the final update. Forward products propagate
parameter, first/second moment, bias-correction, and stabilized-denominator
sensitivities analytically through every batch. `value_gradient`, `jvp`, and
scalar `vjp` are available, and `mlp_optimize_minibatch_adam_hyperparameters`
routes the same callback to bounded FortOpt L-BFGS-B. The outer `hvp` requires
third network derivatives and returns `FORTNUM_NOT_IMPLEMENTED`; CUDA
trajectory requests return the same typed refusal until Adam state is resident.
The independent `test_mlp_minibatch_adam_hypergradient` fixture checks central
differences, directional and scalar adjoints, FortOpt convergence, and the
CUDA boundary. See
[`docs/MLP_MINIBATCH_ADAM_HYPERGRADIENT.md`](MLP_MINIBATCH_ADAM_HYPERGRADIENT.md).

### `fortml_mlp_classifier`

`mlp_classifier_t%fit(x,labels,status[,hidden_layer_sizes,options,state,class_weight])`
builds a deterministic MLP logits model and minimizes stable multiclass
softmax cross-entropy with Adam. Integer labels are sorted and retained as
class metadata. The final layer has one logit per class, and the options
control hidden activation, seeded initialization and shuffling, mini-batches,
L2 regularization, early stopping, and best-state restoration. An optional
positive `class_weight` vector follows the sorted class order and is applied in
full-batch and minibatch reductions.

`decision_function` returns logits, `predict_proba` applies the shared stable
softmax, and `predict` maps the largest probability back to the stored labels.
`classes`, `feature_count`, `class_count`, `parameter_count`, `parameters`,
`set_parameters`, and `fitted` expose the state. `loss_gradient` returns the
cross-entropy value and packed network gradient for a fitted model. Its
optional sample-weight vector uses the same positive-mass reduction.
`mlp_binary_classifier_t` is the one-logit sigmoid counterpart;
`ordinal_logistic_classifier_t` is the separate cumulative-logit contract.

`decision_function_jvp`/`decision_function_vjp` provide exact fixed-state
products with respect to packed network parameters and continuous inputs.
`predict_proba_jvp`/`predict_proba_vjp` compose those products with the stable
softmax. The fixed-input `predict_proba_parameter_jvp` and
`predict_proba_parameter_vjp` specializations expose the same graph with a
zero input tangent, which is useful for hyperparameter and posterior-sensitivity
calculations that only perturb the classifier state. Their
`predict_proba_parameter_jvp_device` and
`predict_proba_parameter_vjp_device` wrappers dispatch CPU explicitly and
return `FORTNUM_NOT_IMPLEMENTED` for CUDA until a resident MLP classifier
graph is linked. The device methods `decision_function_device`,
`predict_proba_device`, and `predict_device` execute on a selected CPU context;
selected CUDA contexts return `FORTNUM_NOT_IMPLEMENTED` until a resident MLP
classifier kernel is linked. The derivative tests cover central differences,
the VJP/JVP duality identity, and the explicit device refusal, including the
fixed-input parameter products.

`mlp_classifier_training_objective_t` is the FortOpt-facing multiclass
objective adapter. `initialize(model,x,labels,l2,status[,optimize_l2,
sample_weight,class_weight,focal_gamma])` validates the fitted classifier's
sorted class metadata and stores one weighted cross-entropy reduction, or the
focal-softmax reduction selected by nonnegative `focal_gamma`. Its packed state is
the logits-network parameters, optionally followed by a non-negative L2
coordinate. `value_gradient`, `jvp`, `vjp`, and `hvp` are analytic; the HVP
includes the MLP's second-order parameter product and the softmax Hessian, not
finite differences. `fortopt` exposes the same callback to FortOpt. The
`mlp_classifier_optimize_lbfgsb` helper applies explicit parameter/L2 bounds
and reports convergence diagnostics through
`mlp_classifier_lbfgsb_result_t`. Invalid labels, weights, bounds, and
unfitted models return typed domain errors. The independent
`test_mlp_classifier_objective` fixture checks value, directional, adjoint,
and HVP products against central-difference oracles and exercises the bounded
L-BFGS-B path.

### `fortml_mlp_calibrated_classifier`

`mlp_calibrated_classifier_t` composes the preceding MLP classifier with a
deterministic calibration head.  Its `fit` signature mirrors
`mlp_classifier_t%fit`, but `mlp_calibrated_classifier_options_t` contains a
`classifier` options object and a `calibration` options object.  Binary
models calibrate the oriented margin `logit(2)-logit(1)` with the shared
`probability_calibrator_t`: sigmoid, positive temperature, and weighted
isotonic methods are available.  Multiclass models currently support one
positive softmax temperature; sigmoid and isotonic multiclass fits return a
typed `FORTNUM_NOT_IMPLEMENTED` refusal rather than changing the probability
policy to independent one-vs-rest maps.  Labels are sorted and probabilities
retain that column order.

`decision_function`, `predict_proba`, `predict`, `classes`,
`feature_count`, `class_count`, `parameters`, `parameter_count`,
`set_parameters`, `calibration_method`, and `temperature` expose deployment
state.  The packed vector is `[network parameters, smooth calibration
parameters]`; isotonic has no calibration parameter slice.  Temperature and
sigmoid `predict_proba_jvp`/`predict_proba_vjp` products are exact through
network parameters, continuous inputs, and calibration parameters.
The explicit `predict_proba_parameter_jvp` and
`predict_proba_parameter_vjp` spellings are available when the caller wants
to hold the input fixed (each also accepts an optional input tangent/cotangent).
`decision_function_jvp/vjp` include the network slice and zero calibration
coordinates because calibration does not alter logits.  Every product for a
binary isotonic head returns `FORTNUM_NOT_IMPLEMENTED` to make its PAVA
active-set boundary explicit.  Device methods dispatch selected CPU contexts
and return the same typed refusal for CUDA until resident MLP/calibration
kernels are linked.  See [`docs/MLP_CALIBRATED_CLASSIFIER.md`](MLP_CALIBRATED_CLASSIFIER.md)
and `test_mlp_calibrated_classifier` for independent fit, finite-difference,
adjoint, active-set, and device-oracle coverage.

### `fortml_mlp_ordinal_classifier`

`mlp_ordinal_classifier_t%fit(x,labels,status[,hidden_layer_sizes,options,
state,sample_weight])` fits a deterministic scalar-score MLP with an ordered
cumulative-logit head. Integer labels are sorted and preserved in `classes()`;
the fitted cut points are strictly increasing and `thresholds()` returns them
in that order. The default topology is `[n_features,8,1]`; passing
`hidden_layer_sizes` replaces the hidden stack. Full-batch CPU training uses
FortOpt L-BFGS-B on the network parameters and log-positive threshold
increments, with optional L2 regularisation and a deterministic initialization
seed. Nonnegative sample weights are normalized by their positive total mass.

`decision_function` returns the latent score, `predict_proba` evaluates the
stable cumulative-logit probabilities, and `predict` maps the largest category
probability back to the original integer label. `parameters` packs the MLP
parameters followed by the actual thresholds; `set_parameters` validates the
strict ordering before replacing the fitted state.

`predict_proba_jvp` and `predict_proba_vjp` are exact fixed-state products for
continuous inputs and the packed network/threshold vector. The parameter JVP
and VJP methods are also available separately, and the independent test checks
their finite-difference and adjoint identities. CPU device dispatch executes
the host path; selected CUDA contexts return `FORTNUM_NOT_IMPLEMENTED` until a
resident ordinal neural kernel is linked.

### `fortml_mlp_binary_classifier`

`mlp_binary_classifier_t%fit(x,labels,status[,hidden_layer_sizes,options,state,
sample_weight,class_weight])` builds a deterministic MLP with one linear logit
and minimizes weighted binary cross-entropy-with-logits with Adam. Labels may
be arbitrary integers, but exactly two distinct values must occur; `classes()`
stores them in ascending order and defines probability columns one (negative)
and two (positive). `sample_weight` is nonnegative and uses positive-weight-mass
normalisation. A positive `class_weight(2)` vector follows the sorted class
order. The options provide hidden activation, seeded initialization and
shuffling, minibatches, L2 regularisation, early stopping, and best-state
restoration.

`decision_function` returns one score per row, `predict_proba` returns the two
stable sigmoid probabilities, and `predict` uses the nonnegative-logit tie rule
for the second class. `feature_count`, `parameter_count`, `parameters`,
`set_parameters`, and `fitted` expose the packed MLP state. `loss_gradient`
returns the weighted BCE plus L2 value and packed gradient for a fitted model;
`loss_hvp` returns its exact packed-parameter Hessian-vector product, including
the nonlinear MLP term and the L2 contribution.

`mlp_binary_training_objective_t` packages the same weighted BCE for the
model-agnostic FortOpt seam. Its packed parameter block can optionally append
the non-negative L2 coefficient; `value_gradient`, scalar `jvp`/`vjp`, and the
joint parameter/L2 `hvp` are analytic. `mlp_binary_optimize_lbfgsb` provides a
bounded full-batch L-BFGS-B path with explicit network and optional L2 bounds,
sample/class weights, and optimizer diagnostics. Every evaluation updates the
fitted model, so a generic `trainer_t` and the direct optimizer use the same
objective products.

`decision_function_jvp`/`decision_function_vjp` and
`predict_proba_jvp`/`predict_proba_vjp` are exact products with respect to both
packed parameters and continuous input rows. The device methods dispatch to a
selected CPU context; selected CUDA contexts return
`FORTNUM_NOT_IMPLEMENTED` until a resident MLP classifier kernel is linked.
The independent `test_mlp_binary_classifier` oracle checks fit behavior,
finite-difference JVP/gradient/HVP products, VJP duality, deterministic Adam,
and the typed CUDA refusal.

### `fortml_mlp_multilabel_classifier`

`mlp_multilabel_classifier_t%fit(x,targets,status[,hidden_layer_sizes,options,
state])` fits one shared MLP whose final layer emits one logit per indicator
column. Targets are finite zero/one indicators with shape
`(n_samples,n_labels)`. The options define deterministic full-batch Adam,
activation, initialization seed, early stopping, and L2 controls; sample and
class weights are intentionally not hidden in this shared-head contract.

`decision_function` returns one logit column per label, `predict_proba` returns
independent sigmoid probabilities, and `predict` returns an integer indicator
matrix using configurable per-label thresholds. `label_count`,
`feature_count`, `parameter_count`, `parameters`, `set_parameters`,
`thresholds`, `set_thresholds`, and `fitted` expose the packed state.
`loss_gradient` and `loss_hvp` return the mean multilabel BCE+L2 value and exact
shared-parameter products.

`mlp_multilabel_training_objective_t` packages the same weighted loss for
FortOpt. Its packed vector contains the network parameters and, optionally,
either a nonnegative L2 coordinate or `log(l2)`. The objective exposes exact
value/gradient, scalar JVP/VJP, and mixed HVP products. Sample weights are
nonnegative, class weights have shape `(2,n_labels)` with positive entries, and
each label must retain positive effective weight. `fortopt` installs the
generic objective callback, while `mlp_multilabel_optimize_lbfgsb` provides
bounded network and L2/log-L2 coordinates. Invalid initialization is
transactional and selected CUDA contexts retain the typed resident-kernel
refusal. See [MLP_MULTILABEL_OBJECTIVE.md](MLP_MULTILABEL_OBJECTIVE.md).

`decision_function_jvp`/`decision_function_vjp` and
`predict_proba_jvp`/`predict_proba_vjp` are exact products with respect to the
packed head parameters and continuous input rows. CPU device dispatch is
exact; selected CUDA contexts return `FORTNUM_NOT_IMPLEMENTED` until a
resident multi-head MLP graph is linked. The independent
`test_mlp_multilabel_classifier` oracle checks finite-difference JVP/HVP
products, VJP duality, indicator validation, and the typed CUDA refusal.

### `fortml_tree`

`decision_stump_t%fit(x,y,status[,min_samples_leaf])` exhaustively selects the
lowest squared-error one-feature split using deterministic sorted thresholds.
`predict` has vector and one-column matrix forms. `jvp` returns zero away from
the split and refuses a query exactly on the discontinuity. The exact tree
lane is dense and finite-only: NaN and infinite values in fit data, prediction
inputs, or JVP inputs are rejected with a domain status. Missing-value routing
is not inferred as a default branch.

`gradient_boosting_regressor_t%fit(x,y,status[,n_estimators,learning_rate,
min_samples_leaf])` fits a squared-loss residual sequence of stumps. Its
prediction and input-JVP products are deterministic. Split selection is a
discrete fit operation, and it shares the finite-only/refusal policy. Thus
differentiable split surrogates, histogram growth, missing-value routing,
classification objectives, and XGBoost/LightGBM policy variants remain in the
tree roadmap.

`cart_regressor_t%fit(x,y,status[,max_depth,min_samples_leaf,sample_weight])`
builds a deterministic numeric CART regression tree. Each node exhaustively
searches sorted midpoints and minimizes weighted squared error. Feature order,
threshold order, and strict-improvement ties are fixed. `max_depth` defaults to
3 and `min_samples_leaf` to 1. The finite-only policy applies to fit data,
weights, prediction inputs, and JVP tangents. `predict` traverses the fitted
tree, while `predict_jvp` returns zero within leaves and refuses a query on a
split boundary. `node_count`, `depth`, `input_count`, and `is_initialized`
expose structural diagnostics. Missing-value routing, histogram growth, and
differentiable split selection remain unsupported for this regression tree.

`cart_classifier_t%fit(x,labels,status[,max_depth,min_samples_leaf,
sample_weight,criterion,missing_policy])` builds a deterministic numeric classification tree.
`criterion` is `CART_CRITERION_GINI` (the default) or
`CART_CRITERION_ENTROPY`. Each node searches weighted impurity over ascending
feature and threshold order, accepts only strict improvements, and stores
weighted class frequencies. `predict_proba` returns the leaf probabilities and
`predict` maps their first maximum back to sorted integer `classes`. The dense
fit and query paths reject infinities. Positive finite sample weights, depth up
to 12, and count-based `min_samples_leaf` are supported. The default
`missing_policy="error"` preserves the finite-only contract; `"learn"`
evaluates both NaN default branches at every candidate split and stores the
strictly best branch (left wins exact ties), while `"left"` and `"right"`
force a branch. `missing_policy()` and `accepts_missing()` expose the fitted
policy. Missing routing is piecewise and has no input derivative product;
histogram growth and differentiable split selection remain unsupported.

### `fortml_random_forest_classifier`

`random_forest_classifier_t%fit(x,labels,status[,n_trees,max_depth,
min_samples_leaf,criterion,seed])` builds a deterministic bootstrap ensemble
of depth-limited CART classification trees. The seeded bootstrap stream keeps
every global class represented, so each tree's probability columns are aligned
before averaging. `criterion` accepts `CART_CRITERION_GINI` or
`CART_CRITERION_ENTROPY`; `RANDOM_FOREST_MAX_TREES` bounds the ensemble size.
`predict_proba` averages the aligned leaf probabilities and `predict` maps the
first maximum back to the sorted integer `classes`. Accessors expose the class,
feature, tree, depth, criterion, seed, and fitted metadata.

The fitted bootstrap-inclusion matrix is retained for audit through
`bootstrap_inclusion()` and `oob_coverage()`. `oob_decision_function(x,p,status)`
(also named `oob_predict_proba` and `predict_proba_oob`) requires the original
training row set and
averages only trees that did not include each row; `oob_score(x,labels,score,
status)` computes accuracy from that same transactional product. Every OOB row
must have at least one excluded tree. Otherwise both methods leave caller
outputs untouched and return `RANDOM_FOREST_OOB_INSUFFICIENT` (the generic
`FORTNUM_CONVERGENCE_ERROR` code), never falling back to in-bag predictions.
`oob_decision_function_device` and `oob_score_device` dispatch on CPU and
return `FORTNUM_NOT_IMPLEMENTED` for CUDA while preserving their output
buffers. Class columns are mapped by the sorted global `classes()` labels even
when an individual bootstrap CART omits a class. See
`docs/RANDOM_FOREST_OOB.md` and the independent benchmark report for the
coverage, score, and CUDA evidence.

`permutation_importance(x,labels,importance,status[,n_repeats,seed,
importance_std,baseline_score])` computes fixed-state accuracy decrease for
each feature.  It validates the fitted classes and finite query rows, then
uses a deterministic Park--Miller Fisher--Yates stream to permute one column
at a time.  `importance` is the baseline accuracy minus the mean permuted
accuracy; the optional `importance_std` is the population standard deviation
over repeats and `baseline_score` receives the unpermuted accuracy.  Defaults
are five repeats, a fixed positive seed, and a maximum of
`RANDOM_FOREST_MAX_PERMUTATION_REPEATS` (1024).  The operation reads fitted
trees only: no tree is refit and split routing is not differentiated.  All
outputs are transactional on invalid state, dimensions, labels, options, or
prediction failure.  `permutation_importance_device` runs the same CPU
operation for `FORTML_DEVICE_CPU`; selected CUDA contexts return
`FORTNUM_NOT_IMPLEMENTED` and preserve every supplied output.  See
`docs/RANDOM_FOREST_PERMUTATION.md` and the independent NumPy benchmark for
the exact stream and oracle.

Tree routing is piecewise constant, so this estimator intentionally exposes no
derivative products. `device_supported`, `predict_proba_device`, and
`predict_device` execute on a selected CPU context and return
`FORTNUM_NOT_IMPLEMENTED` for CUDA until the private CART storage is bound to a
resident ensemble kernel; there is no hidden host fallback. Independent tests
cover separated clusters, probability-simplex alignment, seeded determinism,
invalid options, and the CUDA refusal. The explicit flattened
`cuda_forest_plan_t` below is the supported native-CUDA boundary; it does not
pretend that a fitted `random_forest_classifier_t` has been copied to the
device.

For callers that already have a flattened tree representation, the
`fortml_cuda_forest_api` module exposes `cuda_forest_plan_t`. `create` accepts
zero-based `tree_offset`, child, and feature arrays, node-major leaf
probabilities, sorted class labels, and a CUDA device index; `predict_proba`
and `predict` transfer only each query batch and return the native result.
Model arrays remain resident across calls. The ordinary Fortran build links a
typed unavailable stub, while the native C ABI in
`src/classification/fortml_cuda_forest.{h,cu}` supplies the device kernel.
This wrapper intentionally does not expose or copy private CART storage, and
it provides no autodiff path or hidden CPU fallback.

### `fortml_extra_trees_classifier`

`extra_trees_classifier_t%fit(x,labels,status[,n_trees,max_depth,
min_samples_leaf,max_features,random_splits,criterion,seed,sample_weight])`
builds a deterministic Extra-Trees style classifier. Each tree uses the full
training set and, at every node, evaluates seeded random thresholds on a
randomized feature subset before retaining the best strict Gini or entropy
improvement. `EXTRA_TREES_MAX_TREES` bounds the ensemble; defaults are 100
trees, depth six, `sqrt(n_features)` candidate features, and 16 random
thresholds per feature. Positive finite sample weights are accepted, class
labels are sorted and retained verbatim, and `predict_proba` averages aligned
leaf frequencies across trees. Accessors expose class, feature, tree, depth,
candidate, criterion, seed, and fitted metadata.

Routing is piecewise constant, so no derivative products are declared. The
CPU path is available through `predict_proba_device` and `predict_device`;
CUDA requests return `FORTNUM_NOT_IMPLEMENTED` without modifying caller
buffers until a resident no-autodiff tree kernel is linked. Independent
behavior tests cover empirical leaf probabilities, separated-cluster labels,
seeded determinism, invalid options, and the no-fallback CUDA contract. The
release benchmark is `../fortml-bench/scripts/bench_extra_trees.py` with report
[`EXTRA_TREES.md`](../fortml-bench/results/EXTRA_TREES.md).

### `fortml_bagging_classifier`

`bagging_classifier_t%fit(x,labels,status[,n_trees,max_depth,min_samples_leaf,
max_samples,bootstrap,criterion,seed,sample_weight,missing_policy])` builds a
deterministic bagging ensemble of numeric CART classifiers. The seeded
without-replacement or bootstrap sampler forces one observation from every
sorted class into each subset, so every tree's probability columns remain
aligned even for small `max_samples`. `BAGGING_MAX_ESTIMATORS` bounds the
ensemble; defaults are 10 trees, depth three, and `max_samples=n_samples`.
`criterion` accepts `CART_CRITERION_GINI` or `CART_CRITERION_ENTROPY`, and the
`missing_policy` is passed through to each CART tree. Positive finite sample
weights are copied along with sampled rows.

`predict_proba` averages the aligned tree probabilities and `predict` maps the
first maximum back to the original sorted integer `classes`. Accessors expose
tree count, subset size, depth, leaf size, criterion, seed, bootstrap policy,
and fitted state. Because routing is discrete, probability input JVP/VJP
products return `FORTNUM_NOT_IMPLEMENTED` rather than claiming a smooth zero.
Selected CPU device calls execute normally; selected CUDA calls return the same
typed refusal without a hidden host fallback. The independent
`test_bagging_classifier` fixture covers depth-zero empirical probabilities,
weighted leaves, seeded bootstrap determinism, class-coverage validation,
discrete derivative refusals, and output-preserving CUDA refusals. The release
workload is `fortml_bench_bagging_classifier`.

### `fortml_adaboost_classifier`

`adaboost_classifier_t%fit(x,labels,status[,n_estimators,max_depth,
min_samples_leaf,sample_weight,seed,algorithm])` fits a deterministic binary or
multiclass SAMME ensemble of weighted CART classifiers. Set
`algorithm=ADABOOST_ALGORITHM_SAMME_R` for the multiclass probability-update
policy; the default `ADABOOST_ALGORITHM_SAMME` retains discrete SAMME. Labels may be any two
or more finite integer values and are retained in sorted order. The default
weak-learner depth is one. Binary learner weights are
`0.5*log((1-error)/error)`; multiclass SAMME weights are
`log((1-error)/error)+log(K-1)`. Fitting stops for a perfect learner or when a
learner reaches the random-guessing bound `1-1/K`; a first learner at that
bound is a typed domain error. `seed` is validated and retained for
reproducible future seeded weak-learner policies; the current CART traversal
is deterministic and resolves all ties by sorted feature/threshold order.

The rank-one `decision_function` returns the accumulated signed binary margin;
the rank-two overload returns multiclass weighted-vote margins, and
`stage_weights`, `class_count`, `classes`, and `algorithm` expose the fitted
state. SAMME.R stores unit stage weights and updates training weights from
centred clipped log probabilities. Its multiclass decision margins are the
centred log-probability sum and `predict_proba` applies the corresponding
`1/(K-1)` softmax scale (the geometric probability ensemble). `predict_proba`
maps binary margins through a stable logistic link and SAMME margins through a
stabilized softmax; `predict` uses the
nonnegative binary margin or deterministic lowest-class argmax. CART split
routing is discrete, so `predict_proba_jvp` returns `FORTNUM_NOT_IMPLEMENTED`;
the CPU prediction path is available through `predict_proba_device`, while
CUDA requests return the same typed refusal until a resident tree ensemble
kernel is linked. Invalid fits are transactional for an already fitted model.
The independent oracle is `test_adaboost_classifier`.

### `fortml_discriminant_analysis`

`lda_classifier_t%fit(x,labels,status[,reg_param,priors,sample_weight])` and
`qda_classifier_t%fit(...)` implement weighted linear and quadratic
discriminant analysis for arbitrary integer labels. Labels are sorted and
retained verbatim. LDA fits one pooled covariance; QDA fits one covariance per
class. `reg_param` is a finite diagonal shrinkage in `[0,1]`, and optional
positive priors are normalized in sorted class order. Every class must retain
positive effective sample mass. Covariances are Cholesky-factorized and
probabilities use a stabilized log-sum-exp normalization.

`predict_log_proba`, `predict_proba`, and `predict` return log probabilities,
probabilities, or original integer labels. `classes`, `means`, `covariance`
(LDA), `covariances` (QDA), `class_prior`, `weighted_class_counts`,
`regularization`, `feature_count`, `class_count`, `parameter_count`,
`parameters`, `set_parameters`, and `fitted` expose the state. The packed
parameter order is column-major means, lower-triangle covariance entries
(pooled for LDA or class-major for QDA), and class-prior coordinates.

Input log-probability/probability JVPs and VJPs, plus packed-parameter JVPs
and VJPs, differentiate the continuous Gaussian normalization and hold the
fitted class labels fixed. Argmax labels are discrete and have no derivative
product. `predict_device` and `predict_proba_device` dispatch CPU normally and
return `FORTNUM_NOT_IMPLEMENTED` for CUDA until resident discriminant kernels
are linked; there is no hidden host fallback.

### `fortml_xgboost`

`xgboost_t` is a deterministic second-order boosting estimator. Use
`fit_regression` for a squared objective, `fit_binary` for a logistic
objective, `fit_poisson` for nonnegative count targets with a log link,
`fit_tweedie` for a compound-Poisson Tweedie objective with a log link,
`fit_gamma` for the fixed-shape Gamma log-link objective,
`fit_huber` for robust Huber regression, or `fit_quantile` for pinball
regression. Use `fit_squared_log` for XGBoost's `reg:squaredlogerror`
(RMSLE) objective, or `fit_ranking` for the `rank:pairwise` objective.
The generic `fit` accepts `objective="squared"`,
`"squaredlog"`/`"reg:squaredlogerror"`/`"rmsle"`, `"logistic"`,
`"poisson"`, `"gamma"`/`"reg:gamma"`, `"tweedie"`/`"reg:tweedie"`, `"huber"`/`"pseudohuber"`,
`"quantile"`/`"pinball"`, or
`"rank:pairwise"`.
All fit methods accept an optional positive `sample_weight(:)`;
weights affect the base score and every gradient/Hessian reduction. The
`xgboost_options_t` fields `tree_method="exact"` (the default) and
`tree_method="hist"` select exhaustive split enumeration or deterministic
weighted-quantile histogram growth. In histogram mode, `max_bin` bounds the
number of finite bins per node; every NaN remains an explicit missing bin and
is routed by `missing_policy` (`error`, `learn`, `left`, or `right`). The
histogram policy remains weighted-quantile even when `max_bin` is large; use
`tree_method="exact"` when exhaustive split equivalence is required. The
`huber_delta`, `quantile_alpha`, `gamma_shape`, and
`tweedie_variance_power` options
control the robust and Tweedie objectives; `tweedie_variance_power` must
satisfy `1 < tweedie_variance_power < 2`. The fitted value is available
through `objective_parameter_value()`.
Huber uses the exact piecewise gradient with a positive Hessian floor on its
linear tails. Quantile uses the declared subgradient (`alpha` at a zero or
positive residual, `alpha-1` below zero) and the same explicit Hessian floor.
The remaining options control estimator count, learning rate,
minimum leaf size, L1/L2 leaf regularization, split gamma, and minimum child
Hessian. Candidate splits aggregate exact gradients and Hessians and use the
regularized gain. `predict_margin`, `predict`, `predict_proba`,
`decision_function`, `split_gain`, `leaf_weights`, `tree_node_count`, and
`tree_depth` expose diagnostics.
`tree_method()` and `max_bin_count()` expose the fitted execution policy.
Set the optional allocatable `xgboost_options_t%monotone_constraints(:)` to
one entry per feature, using `-1`, `0`, or `+1` for decreasing,
unconstrained, or increasing predictions. Fit validates the vector and
propagates per-node leaf bounds through exact and histogram recursion;
`monotone_constraint(feature)` reports the fitted value. The fit remains a
piecewise/discrete operation, so the existing split-boundary derivative
refusal still applies.
`predict_jvp` is zero away from learned split boundaries and returns a
structured refusal at a discontinuity. `max_depth` grows each exact tree
recursively, with deterministic feature/threshold tie ordering and
regularized Newton leaves at every node. Histogram quantile approximation and
categorical features remain separate policies. Allocate
`xgboost_options_t%interaction_groups(:)` with one nonnegative integer per
feature to constrain feature co-occurrence along a tree path: a positive
label restricts descendant splits to that same group, while zero leaves the
feature unconstrained. `interaction_group(feature)` reports the fitted label;
the vector is validated, copied by `slice` and warm starts, and persisted by
the versioned text snapshot. A changed or malformed policy is a typed domain
error, and tree fit remains discrete with the existing split-boundary
derivative refusal.

Integer-coded categorical columns use the bounded ordered-gradient policy.
Set `categorical_policy="ordered"`, provide sorted one-based feature indices
in `categorical_features(:)`, and choose `categorical_max_categories` between
2 and 64.  At each node, category codes are ordered by accumulated
gradient/Hessian score (code order breaks ties), and every prefix partition is
evaluated with the ordinary second-order gain.  Non-integer codes and a
cardinality above the explicit bound return `FORTNUM_NOT_IMPLEMENTED`.
The policy and feature metadata survive warm starts, prefix slices, and the
version-4 text snapshot.  CPU value prediction is supported; CUDA dispatch is
a typed refusal until a resident categorical-tree kernel exists.  Categorical
models refuse `predict_jvp` and `predict_vjp` because integer categories have
no canonical continuous tangent space.  See
[`docs/XGBOOST_CATEGORICAL.md`](XGBOOST_CATEGORICAL.md) and the independent
`test_xgboost_categorical` fixture.

The Tweedie lane uses the compound-Poisson variance-power interval
`1 < p < 2`, where `p` is `tweedie_variance_power`. Margins are log means and
`predict`/`predict_staged` apply `exp(margin)`. Up to a target-only constant,
the weighted mean objective is
`y*exp((1-p)*margin)/(p-1) + exp((2-p)*margin)/(2-p)`;
`xgb_tweedie_loss` exposes this value and
`xgb_tweedie_derivatives` exposes the exact per-row gradient and Hessian.
Nonfinite inputs, negative targets, powers at either endpoint, and nonfinite
objective products are rejected. The tree fit and CPU prediction are
deterministic; CUDA prediction returns `FORTNUM_NOT_IMPLEMENTED` rather than
silently falling back to the host.

The fixed-shape Gamma lane uses a positive `gamma_shape` and strictly positive
targets. In log-mean margin `eta`, its weighted objective (up to a
target-only constant) is `shape*(eta + y*exp(-eta))`, with exact products
`gradient=shape*(1-y*exp(-eta))` and `hessian=shape*y*exp(-eta)`. Predictions
use `exp(eta)`, objective metadata survives text persistence and warm starts,
and invalid shape/target values are typed domain errors. Exact and weighted
histogram CPU growth are supported; CUDA prediction remains an explicit
`FORTNUM_NOT_IMPLEMENTED` refusal until a resident tree kernel is linked.

The
`missing_policy` option is `error` by default and rejects IEEE NaNs. `learn`
evaluates both default directions for every finite threshold and stores the
strictly best direction (left wins exact ties); `left` and `right` force a
deterministic default branch. In these three modes, fit and prediction accept
IEEE NaNs while infinities remain domain errors. The learned branch is used
consistently by prediction and JVP; a NaN query has zero local input JVP, while
a finite query exactly on a split still returns the structured boundary
refusal. `missing_policy()` and `accepts_missing()` report the fitted policy.
`predict_vjp(x,output_bar,x_bar,status)` is the matching reverse product: it
returns a zero feature cotangent away from split boundaries and refuses the
same discontinuities, nonfinite cotangents, and malformed shapes.

`leaf_parameter_count()` and `leaf_parameters(status)` expose the continuous
fixed-structure coordinates `[base_score, leaf weights in estimator/node
order]`. `predict_leaf_jvp(x,parameter_dot,margin,margin_dot,status)` and
`predict_leaf_vjp(x,output_bar,parameter_bar,status)` differentiate raw margins
with respect to those coordinates while holding split routing, categorical
partitions, DART choices, and tree scales fixed. The products are therefore
defined on split surfaces; they do not make fit or routing differentiable.
For logistic models apply the sigmoid product separately to obtain probability
derivatives. CUDA leaf products remain an explicit refusal until resident tree
state and transfer accounting are available.

`predict_leaf_jvp_device` and `predict_leaf_vjp_device` make this boundary
explicit: CPU dispatches to the methods above, while selected CUDA contexts
return `FORTNUM_NOT_IMPLEMENTED` without mutating caller buffers.

Every fit method also accepts an optional validation set through
`validation_x`, `validation_y`, and `validation_weight`. Set
`early_stopping_rounds` to a positive patience count and optionally set
`early_stopping_min_delta` to require a minimum weighted-loss improvement.
The validation objective matches the selected squared, logistic, Poisson,
Tweedie, squared-log, Huber, or quantile objective. `restore_best` trims the fitted
ensemble to the best round. `best_iteration()`, `best_validation_loss()`, and
`early_stopped()` expose the resulting lifecycle state. Validation arguments
must be supplied together and are shape-, weight-, target-, and NaN-checked.
This is deterministic validation-based stopping. Serialized tree state remains
a separate contract. `slice(n_trees,destination,
status)` copies the first fitted boosting rounds into a valid standalone model,
preserving objective/link, base margin, routing, constraints, regularization,
and diagnostics; the best-iteration diagnostic is clamped to the retained
prefix. Invalid prefixes or malformed sources are refused transactionally.

`fit_warm_start(x,y,status,options[,sample_weight,...])` continues a fitted
`xgboost_t` to the larger total `options%n_estimators` without changing the
existing tree prefix. Objective, tree method, regularisation, sampling seed,
row/feature fractions, missing policy, and monotone constraints must match the
original fit; a non-increasing target or changed control returns
`FORTNUM_DOMAIN_ERROR` transactionally. The same training data, weights, and
ranking groups must be supplied by the caller. Optional validation data may be
used for stopping the appended suffix, with the same `restore_best` policy as
ordinary fitting. `requested_estimator_count()` reports the requested total,
while `estimator_count()` reports the retained fitted prefix.

`fit_ranking(x,relevance,group,status[,options,sample_weight,...])` selects
the `rank:pairwise` objective. Rows with the same positive integer query ID
form a group; only unequal-relevance pairs contribute the stable logistic loss,
gradient, and positive Hessian, and pair weights use the smaller endpoint
weight. Groups are isolated from one another, validation ranking data carries
its own group IDs, and a query set without an unequal pair is refused. The
public `xgb_pairwise_loss` and `xgb_pairwise_derivatives` procedures provide
independent objective products for FortOpt or custom trainers. Ranking margins
have no probability link, use a zero initial base margin, and retain the
existing piecewise tree split/refusal and typed CUDA boundaries.

Set `subsample` to a positive fraction no greater than one to draw a
without-replacement training-row subset independently for each boosting
round. Set `colsample_bytree` similarly to select the feature subset used by
each tree. Both streams use the positive `integer(int64)` `seed` in
`xgboost_options_t`; repeated fits with the same options are bitwise
deterministic, and selected rows/features retain ascending original ordering
for stable tie handling. The defaults (`subsample=1`,
`colsample_bytree=1`) take the full-data path and preserve the historical
model exactly. Fractions outside `(0,1]` and nonpositive seeds are refused
with `FORTNUM_DOMAIN_ERROR`. This sampling is a fit-time discrete policy;
prediction and the existing split-boundary derivative refusal are unchanged.

For `fit_squared_log`, margins are the transformed coordinate
`log(1 + prediction)` and the public inverse link is `exp(margin) - 1`. The
constant base margin is the weighted mean of `log(1 + target)`, which is the
geometric optimum for the RMSLE coordinate rather than the transform of the
arithmetic target mean. The objective products are
`gradient = (margin - log(1 + target))/exp(margin)` and
`hessian = (1 - margin + log(1 + target))/exp(margin)`, with the same positive
Hessian floor used by the Newton tree gain when the exact curvature is
nonpositive. Targets must be finite and nonnegative. The guarded inverse link
is finite and has the mathematical lower bound `-1`; it does not silently
claim a nonnegative output when a Newton update crosses below zero. Exact and
weighted-histogram paths have an independent hand-formula oracle. No resident
CUDA tree kernel is linked, so CUDA prediction returns
`FORTNUM_NOT_IMPLEMENTED`.

For `fit_poisson`, margins are log expected counts and `predict`/staged
predictions apply a finite guarded exponential. The Newton products are
`gradient = exp(margin) - target` and `hessian = exp(margin)` (with a positive
floor for degenerate zero-rate nodes); negative targets are rejected. Poisson
input JVP/VJP products follow the same piecewise-constant tree contract as
the squared and logistic lanes. The CPU exact and weighted-histogram paths
are independently oracle-tested. No resident CUDA tree kernel is linked in
this release, so `predict_device` on CUDA returns
`FORTNUM_NOT_IMPLEMENTED` instead of silently falling back to the host.

`predict_device(device,x,y,status)` is the explicit device-control-plane
entry point for vector predictions. CPU dispatch uses the validated host path.
The current release has no resident CUDA tree or histogram kernel, so a
selected CUDA device returns `FORTNUM_NOT_IMPLEMENTED` and never falls back
to host execution. `device_supported(FORTML_DEVICE_CPU/FORTML_DEVICE_CUDA)`
reports this capability boundary.

`predict_staged` returns cumulative predictions after every fitted tree. For
regression, `predict_staged_margin` returns the same cumulative margins; for
binary classification, `predict_proba_staged` returns an
`(sample,class,stage)` probability tensor and `decision_function_staged`
returns cumulative margins. `feature_importance(kind,normalize)` reports
deterministic gain, split-count (`weight`), or cover totals. These diagnostics
are fitted-state queries and preserve the selected NaN routing policy.

`predict_contributions(x,contributions,status)` returns an additive raw-link
decomposition with shape `(n_samples,n_estimators+1)`. Column one is the base
margin and column `i+1` is the learning-rate-scaled output of tree `i`; summing
the columns reproduces `predict_margin` exactly up to floating-point rounding.
For logistic, Poisson, and squared-log objectives the decomposition remains in
the raw link, so apply the objective link after summing. The corresponding
`predict_contributions_device` entry point dispatches CPU and returns a typed
`FORTNUM_NOT_IMPLEMENTED` refusal for CUDA until a resident tree kernel is
linked. Invalid shapes and unsupported NaN policies return
`FORTNUM_DOMAIN_ERROR`.

`predict_shap(x,shap,status)` returns a bounded per-feature SHAP-like
raw-margin decomposition with shape `(n_samples,n_features+1)`. Column one is
the path-dependent expected margin and the remaining columns are exact subset
Shapley attributions using fitted node-cover branch probabilities; the row
sum reproduces `predict_margin`. The subset path is limited to 12 features and
returns `FORTNUM_NOT_IMPLEMENTED` for wider models rather than silently
approximating. `predict_shap_device` dispatches CPU and returns a typed CUDA
refusal until a resident explanation kernel is linked. See
[`docs/TREE_SHAP.md`](TREE_SHAP.md).

`xgboost_multiclass_t` wraps the binary logistic estimator in a deterministic
one-vs-rest classifier. `fit(x,labels,status[,options,sample_weight])` sorts
arbitrary integer labels, applies an optional positive sample-weight vector to
each one-vs-rest booster, and normalizes the
positive OVR probabilities. `classes`, `class_count`, `feature_count`,
`estimator_count`, `fitted`, `decision_function`, `predict`, and
`predict_proba` expose the fitted state. `predict_proba_jvp` applies the exact
quotient-rule JVP away from learned split boundaries and propagates the binary
boundary refusal. The classifier inherits the binary estimator's dense,
finite-input scope unless its options select `missing_policy="learn"`,
`"left"`, or `"right"`; the selected default branch is retained independently
in every one-vs-rest tree and is used for NaN prediction. Infinities and an
invalid policy are always refused.
`predict_proba_vjp` applies the corresponding reverse product and propagates
the same split-boundary refusal; the fitted tree structure and OVR
normalization are held fixed.

The multiclass wrapper forwards `monotone_constraint(feature)` from its
one-vs-rest estimators and exposes `predict_proba_device` and
`predict_device`. CPU dispatch is equivalent to the ordinary methods. CUDA
returns `FORTNUM_NOT_IMPLEMENTED` until resident binary-tree kernels and OVR
normalization are linked; `device_supported` reports this explicitly.

`predict_proba_staged` and `decision_function_staged` expose the normalized
one-vs-rest probabilities and summed margins after each boosting stage.
`feature_importance(kind,normalize)` aggregates the per-class binary gain,
weight, or cover diagnostics. A final staged slice is identical to the
corresponding ordinary prediction within floating-point rounding.

### `fortml_xgboost_classifier`

`xgboost_classifier_t` is the classifier-shaped binary facade over the
logistic `xgboost_t` lane. `fit(x,labels,status[,options,sample_weight,
validation_x,validation_labels,validation_weight])` accepts exactly two
arbitrary integer labels, stores them in sorted order, and forwards positive
sample/validation weights and the complete XGBoost options policy (exact or
weighted histogram growth, NaN routing, monotone constraints, regularization,
subsampling, and validation early stopping). `predict_proba` returns an
`(n,2)` simplex in that class order; `predict` returns integer labels with the
first class winning an exact tie. `decision_function` returns the positive
class raw logit, while `predict_proba_staged` and
`decision_function_staged` expose cumulative rounds. `classes`,
`feature_count`, `estimator_count`, `feature_importance`, `missing_policy`,
`accepts_missing`, `tree_method`, monotone metadata, and validation lifecycle
queries expose fitted state without exposing tree storage.

`predict_log_proba` evaluates stable class log probabilities directly from the
margin. `predict_log_proba_jvp` and `predict_log_proba_vjp` provide the matching
fixed-tree input products. `categorical_policy`,
`categorical_max_categories`, `categorical_feature`, and `interaction_group`
expose ordered-partition metadata and preserve the typed discrete-product
boundary for integer categorical features.

`predict_proba_jvp`/`predict_proba_vjp` are fixed-tree input products. They
return zero away from finite split thresholds and preserve the underlying
structured boundary refusal; argmax labels remain discrete. CPU device
dispatch is equivalent to ordinary prediction. Selected CUDA devices return
`FORTNUM_NOT_IMPLEMENTED` from `predict_device` and
`predict_proba_device` until resident tree kernels are linked, with no hidden
host fallback. See [XGBOOST_CLASSIFIER.md](XGBOOST_CLASSIFIER.md) and the
independent `test_xgboost_classifier` oracle.

### `fortml_bnn`

`bnn_t%initialize(layer_sizes,n_mc_samples,seed,status[,prior_variance,
noise_variance,hidden_activation])` constructs an MLP with a factorized
Gaussian variational posterior. The seed fixes the Monte Carlo table, so
repeated ELBO calls are deterministic. A zero seed and nonpositive variances
are rejected.

If the MLP has `p` parameters, the BNN parameter vector is
`[mu(1:p),log_sigma(1:p)]`. Public methods are `parameter_count`, `parameters`,
`set_parameters`, `kl`, `kl_gradient`, `elbo`, `elbo_jvp`, `elbo_vjp`, and
`elbo_hvp`. ELBO methods take `x(n,d)` and `y(n,p_out)` and differentiate the
scalar Gaussian-likelihood ELBO.

### `fortml_vae`

`vae_t%initialize(input_dim,hidden_dim,latent_dim,batch_size,seed,status
[,likelihood_variance])` creates one MLP encoder and one MLP decoder. The
Gaussian likelihood variance is a scalar. The fixed seeded noise table has the
declared batch size, and every later input must use that size.

Parameters are the encoder vector followed by the decoder vector. The public
model methods are `parameter_count`, `parameters`, `set_parameters`, `elbo`,
`elbo_gradient`, and `reconstruct`. `reconstruct` uses the fixed latent draw,
not the posterior mean. The sampled encoder code and latent array are internal.

### `fortml_rnn`

`rnn_t%initialize(input_dim,hidden_dim,output_dim,status)` constructs a single
vanilla `tanh` recurrent layer and a linear output layer. Every call starts
from a zero hidden state. `forward(inputs,outputs,hidden,status)` returns the
hidden state after every step. Initial-state input, stacked recurrent layers,
GRU cells, and LSTM cells are not implemented.

`loss` is one half of the unnormalized sum of squared errors over every time,
batch, and output entry. `loss_gradient` returns its packed parameter gradient.
Use `parameter_count`, `parameters`, and `set_parameters` for the input weight,
hidden weight, hidden bias, output weight, then output bias packing.

### `fortml_variational`

`gaussian_family_t%initialize(dimension,n_mc_samples,seed,status
[,prior_variance,full_covariance])` creates a full or diagonal Gaussian family.
Whitening requires `n_mc_samples > dimension`. The seeded sample table stays
fixed after initialization.

The parameter vector begins with the mean. Each covariance-factor column then
stores its log diagonal followed by the entries below that diagonal. A diagonal
family stores only the log diagonal. Public methods are `parameter_count`,
`parameters`, `set_parameters`, `covariance`, `draw`, `draw_tangent`, `kl`, and
`kl_gradient`.

`vi_elbo` and `vi_elbo_gradient` accept an `extra(:)` parameter block, a
callback matching `vi_log_likelihood_i`, and an explicit minibatch `scale`.
The callback returns the log likelihood and gradients for one latent draw.
The ELBO scales the likelihood term and leaves the KL term unscaled.

## Kernels and dense Gaussian processes

### `fortml_kernels`

The factory functions validate positive variance and lengthscale arguments:

```text
make_rbf_kernel(input_dim, variance, lengthscale, status)
make_rbf_ard_kernel(input_dim, variance, lengthscales, status)
make_ard_rbf_kernel(input_dim, variance, lengthscales, status)  ! alias
make_matern12_kernel(input_dim, variance, lengthscale, status)
make_matern32_kernel(input_dim, variance, lengthscale, status)
make_matern52_kernel(input_dim, variance, lengthscale, status)
make_linear_kernel(input_dim, variance, status)
make_constant_kernel(input_dim, variance, status)
make_white_noise_kernel(input_dim, variance, status)
make_periodic_kernel(input_dim, variance, lengthscale, period, status)
make_local_periodic_kernel(input_dim, variance, envelope_lengthscale,
    periodic_lengthscale, period, status)
make_rational_quadratic_kernel(input_dim, variance, lengthscale, alpha, status)
make_cosine_kernel(input_dim, variance, lengthscale, status)
make_polynomial_kernel(input_dim, variance, scale, offset, degree, status)
make_spectral_mixture_kernel(input_dim, num_mixtures, weights, means, scales, status)
make_user_kernel(input_dim, variance, formula, status)
```

The corresponding kind constants are `KERNEL_RBF`, `KERNEL_RBF_ARD`, `KERNEL_MATERN12`,
`KERNEL_MATERN32`, `KERNEL_MATERN52`, `KERNEL_LINEAR`, `KERNEL_CONSTANT`,
`KERNEL_WHITE_NOISE`, `KERNEL_PERIODIC`, `KERNEL_RATIONAL_QUADRATIC`,
`KERNEL_COSINE`, `KERNEL_POLYNOMIAL`, `KERNEL_SPECTRAL_MIXTURE`,
`KERNEL_LOCAL_PERIODIC`, `KERNEL_CHANGE_POINT`, `KERNEL_SUM`, `KERNEL_PRODUCT`,
and `KERNEL_USER`.
Combine initialized kernels with `kernel_add(left,right,status)` or
`kernel_multiply(left,right,status)`.
`clone_kernel(kernel)` makes an independent copy of the complete expression
tree, including composite children. Use it for temporary optimizer or
derivative probes instead of intrinsic assignment, which aliases pointer
children.

Leaf parameters are stored as logarithms. Isotropic RBF and Matérn leaves have
`[log_variance,log_lengthscale]`. `make_rbf_ard_kernel` stores
`[log_variance,log_lengthscale_1,...,log_lengthscale_d]`, with one positive
length scale per input feature. Its value is
`variance*exp(-0.5*sum_j((x1_j-x2_j)/lengthscale_j)**2)`. ARD RBF leaves
provide the same dense value, input derivative, parameter JVP/VJP, and
parameter HVP contract as isotropic RBF leaves; the packed parameter order is
stable under composite-kernel concatenation. Exact `gp_regression_t` fitting,
prediction, log-marginal-likelihood, and their parameter products consume an
ARD kernel without a special GP code path. The resident CUDA covariance path
does not yet support ARD kernels and therefore reports the typed device
refusal rather than falling back to host execution.

Periodic leaves use
`[log_variance,log_lengthscale,log_period]`; rational-quadratic leaves use
`[log_variance,log_lengthscale,log_alpha]`. Linear, constant, white-noise, and
user leaves have `[log_variance]`. Cosine leaves use
`[log_variance,log_lengthscale]`. Polynomial leaves use
`[log_variance,log_scale,log_offset,log_degree]` and require
`offset + scale*dot(x1,x2) > 0` for input derivatives. Composite vectors
concatenate the complete left vector and then the complete right vector.

Locally-periodic leaves use
`make_local_periodic_kernel(input_dim,variance,envelope_lengthscale,
periodic_lengthscale,period,status)` and pack
`[log_variance,log_envelope_lengthscale,log_periodic_lengthscale,log_period]`.
Their covariance is the product of a squared-exponential envelope and a
periodic factor:
`variance*exp(-r2/(2*envelope_lengthscale**2) -
2*sin(pi*sqrt(r2)/period)**2/periodic_lengthscale**2)`. Dense values,
input gradients/mixed Hessians, parameter JVP/VJP/HVP products, and exact-GP
likelihood integration are analytic. Coincident-point limits are evaluated
without a radial division. `kernel_operator_t` reports a typed refusal because
the static CUDA program does not yet carry this four-parameter leaf; host
matrix and derivative products remain available.

Change-point leaves use `make_change_point_kernel(left,right,feature,center,
width,status)`. The covariance is
`s(x)s(z)k_left(x,z)+(1-s(x))(1-s(z))k_right(x,z)` with a positive transition
width and a smooth logistic gate on the selected feature. Child parameters
are followed by `[log(width),center]` in the packed vector. Value, input
gradients, mixed Hessians, parameter JVP/VJP/HVP products, and exact-GP
integration are analytic and covered by `test_kernel_change_point`. The
static operator and resident CUDA paths return typed refusals until the full
gated expression is available on the device.

Spectral-mixture leaves use `make_spectral_mixture_kernel` with positive
`weights` and positive frequency-standard-deviation `scales` plus signed
`means`, each shaped by mixture and feature. The covariance factor is
`exp(-2*pi**2*tau**2*scale**2)*cos(2*pi*tau*mean)`, matching GPyTorch's
spectral-mixture convention.
Their packed block is `[log_weight,log_scale(:),mean(:)]` per mixture. Dense
values, input derivatives, parameter JVP/VJP/HVP products, and exact-GP
likelihood integration are analytic; resident CUDA execution is an explicit
typed refusal until a resident spectral-mixture kernel is linked. See
`docs/GP_SPECTRAL_MIXTURE.md` for the formula and independent oracle.

`kernel_t` exposes `parameter_count`, `parameters`, `set_parameters`, `value`,
`matrix`, `matrix_jvp`, `parameter_vjp`, `parameter_hvp`, and
`input_derivatives`. Input derivatives return the value, gradients with respect
to both arguments, and the mixed Hessian. Periodic and rational-quadratic
leaves provide smooth coincident-point limits for these products. Cosine and
polynomial leaves also expose analytic parameter JVP/VJP/HVP products in their
logarithmic parameters. The periodic scale is the Euclidean period and the
rational-quadratic tail is
controlled by positive `alpha`. Matérn 1/2 input derivatives remain undefined
at coincident points, and white-noise derivative observations are rejected.
Validated user formulas use the same forward derivative stack for their value,
both gradients, and mixed Hessian. A `push_distance` formula refuses coincident
points where its derivative is singular. The free `kernel_input_derivatives`
procedure has the same arguments as the type-bound method with the kernel
supplied first.

`kernel_operator_t` currently refuses periodic, rational-quadratic, cosine, and
polynomial leaves at initialization: the static host/operator program and
resident CUDA ABI do not yet carry the complete leaf parameter payloads. This
is a typed refusal, not a host fallback. Dense `kernel_t%matrix` and all
declared derivative products remain available, and a resident device
implementation will be added only when the operator ABI can evaluate the
complete expression without hidden transfers.

### `fortml_kernel_formula`

`kernel_formula_t` builds a bounded postfix expression from squared distance,
distance, inner product, constants, addition, subtraction, multiplication,
negation, exponential, and division by a nonzero constant. Call `reset`, append
operations with `push_*`, `add`, `subtract`, `multiply`, `negate`,
`exponential`, and `divide_by_constant`, then call `validate(status)`.

`static_lowering_eligible()` becomes true only after successful validation.
`evaluate(squared_distance,inner_product)` is the host reference function.
`make_user_kernel` refuses any other formula. `MAX_FORMULA_LENGTH` is 64 and
`MAX_FORMULA_STACK` is 16. The `OPCODE_*` constants and
`formula_opcode_is_known` support serialization and backend tests.

```text
OPCODE_PUSH_R2=-1  OPCODE_PUSH_R=-2  OPCODE_PUSH_DOT=-3  OPCODE_PUSH_CONST=-4
OPCODE_ADD=-5  OPCODE_SUBTRACT=-6  OPCODE_MULTIPLY=-7  OPCODE_NEGATE=-8
OPCODE_EXP=-9  OPCODE_DIVIDE_CONST=-10
```

### `fortml_gaussian_process`

`gp_regression_t%fit(x,y,kernel,noise_variance,status[,jitter,mean])` fits one or
more output columns with a shared kernel and independent output values. The
packed model parameter vector is the recursive kernel vector followed by
`log_noise_variance`. An optional `gp_mean_t` template from
`fortml_gp_mean` adds a trainable constant or linear trend for every output;
its coefficients are packed in output-column order after the noise parameter.
The default is the zero mean. `set_parameters` refactorizes the fitted
covariance and updates the mean residual state.

`make_zero_mean`, `make_constant_mean`, and `make_linear_mean` validate the
feature dimension and finite template coefficients. A constant mean has one
intercept; a linear mean has an intercept followed by one slope per feature.
`model%mean_parameter_count()` and `model%mean_parameters()` expose the
expanded per-output block, while the ordinary `parameters()` method retains
the existing kernel/noise prefix. Mean coefficients participate in
`hyperparameter_gradient`, `hyperparameter_hvp`, prediction JVP/VJP/HVP, and
the FortOpt GP hyperparameter objective, so finite-difference tuning does not
silently omit the trend. The adapter is a dense CPU exact-GP path; resident
CUDA covariance and factorization remain an explicit refusal boundary.

`predict(x,mean,variance,status)` returns one mean column per training output
and one shared latent variance vector. Parameter product methods keep query
inputs fixed:

```text
predict_jvp(x, direction, mean, mean_dot, variance, variance_dot, status)
predict_vjp(x, mean_bar, variance_bar, parameter_bar, status)
predict_hvp(x, mean_bar, direction, parameter_hvp, status)
```

The HVP covers a weighted predictive mean. The LML methods are
`log_marginal_likelihood`, `log_marginal_likelihood_jvp`,
`hyperparameter_gradient`, and `hyperparameter_hvp`.

### `fortml_deep_kernel_gp`

`deep_kernel_gp_t` composes a dense MLP feature map `g(x,w)` with an exact GP
base kernel on the feature space, following the construction in Wilson et al.,
*Deep Kernel Learning* (arXiv:1511.02222). Call
`initialize(layer_sizes,base_kernel,status[,hidden_activation,
initialization_seed])`; the first layer width is the input dimension and the
last is the base-kernel dimension. The output feature layer is linear, and a
kernel whose input dimension does not equal the feature width is rejected.

`fit(x,y,noise_variance,status[,jitter])` maps the training inputs and fits the
ordinary dense GP. `transform`, `predict`, and `log_marginal_likelihood` expose
the mapped features and posterior. `feature_gradient(x,y,gradient,status)`
returns the exact log-marginal-likelihood gradient with respect to mapped
features. `weight_gradient(x,y,gradient,status)` backpropagates that seed
through the MLP weights. The independent `test_deep_kernel_gp` fixture checks
the identity-map reduction, every-weight central differences, and refusal
boundaries. The implementation is CPU-only with dense cubic GP scaling. It
does not yet provide a joint feature/kernel FortOpt training loop, KISS-GP/SKI
approximation, derivative-observation deep kernels, or resident CUDA execution.

### `fortml_gp_training`

`gp_optimize_hyperparameters(model,options,result,status[,device])` minimizes the
negative exact-GP log marginal likelihood with FortOpt L-BFGS-B. The model
must already be fitted, and each objective evaluation refactorizes the model
through `set_parameters`, so the optimizer uses the same analytic
hyperparameter gradient as the public likelihood product. Parameters are the
kernel log parameters followed by log observation-noise variance. Bounds are
applied uniformly through `gp_hyperparameter_options_t`. The default interval
is `[-20,20]`. `gp_optimize_hyperparameters_multistart` is the explicit
multi-start entry point; the main entry point delegates to it. Set
`options%starts` and `options%seed` for deterministic uniform starts in the
closed box. With `include_current=.true.` (the default), the fitted parameter
vector is the first start. Only finite converged runs compete for retention,
and the model is restored to the lowest negative log marginal likelihood.
The result reports `start_count`, `successful_starts`, `best_start`, and
`objective_evaluations` in addition to the single-start diagnostics. An
optional selected CUDA device returns `FORTNUM_NOT_IMPLEMENTED`; exact GP
factorization and this optimizer do not silently fall back to a host path.

`gp_hyperparameter_result_t` reports convergence, iteration count, final
negative log marginal likelihood, and the final gradient norm. A nonconverged
iteration limit or nonfinite objective returns a convergence status. The
adapter currently targets exact fitted GPs. Derivative-observation training is
provided separately by `fortml_derivative_gp_training`.

### `fortml_derivative_gp_training`

`gp_optimize_derivative_hyperparameters(model,options,result,status)` minimizes
the negative mixed value/first-derivative GP likelihood with FortOpt
L-BFGS-B. It uses the derivative GP's analytic likelihood gradient and the
same bounded log-parameter options/result type as `fortml_gp_training`. The
optimizer does not consume the derivative GP's finite-difference HVP. That
product is available for diagnostics and second-order callers only.

### `fortml_gp_classification`

`gp_classification_t%fit(x,labels,kernel,status[,options,state,sample_weight])`
fits a binary Laplace GP classifier. Labels are arbitrary integers and are
retained in ascending order. `GP_LIKELIHOOD_LOGISTIC` uses a MacKay logistic
predictive integral. `GP_LIKELIHOOD_PROBIT` uses the analytic probit predictive
map. `gp_classification_options_t` controls Newton iterations, damping,
tolerance, and jitter. Optional finite, nonnegative `sample_weight` values
weight each training likelihood contribution and require positive total mass;
zero-weight rows contribute no likelihood curvature. The fitted state,
envelope `hyperparameter_gradient`, and `state%log_posterior` all use those
weights. `predict_latent` returns posterior latent mean and variance.
`predict_proba` returns two observed-probability columns. `predict_log_proba`
returns their natural logarithms in the same stored class order, with a finite
floor for floating-point probit tails. Both probability APIs have input-JVP
and input-VJP variants; their fixed-state kernel-parameter JVP/VJP products
are also available under matching `predict_log_proba_*` names. `set_parameters`
updates kernel log parameters while keeping the converged Newton mode,
`alpha`, and likelihood curvature fixed, rebuilding only the covariance
factorizations needed by those products. This makes finite-difference checks
and outer hyperparameter search agree on the fixed-state derivative contract.
`predict_log_proba_device` dispatches CPU explicitly and returns a typed CUDA
refusal. `predict` returns the stored integer labels. The
`predict_latent_parameter_jvp`/`_vjp` and `predict_proba_parameter_jvp`/`_vjp`
variants differentiate the fixed fitted Laplace prediction with respect to
the packed kernel log parameters. They propagate `matrix_jvp` and
`parameter_vjp` through the posterior covariance solve; the Newton mode,
`alpha`, and likelihood curvature are intentionally held fixed and are not
silently differentiated. Every kernel that
supplies the existing matrix and input-derivative contracts is supported.
`parameter_count()` and `parameters()` expose read-only kernel-log-parameter
metadata in the fitted model. `hyperparameter_gradient` returns the exact
envelope gradient of the fitted Laplace-mode log posterior (without the
optional evidence correction). At a converged mode this is the kernel-VJP
contraction `0.5 * alpha * alpha^T : dK/dtheta`, so sums, products, and other
kernels implementing the parameter-VJP contract are supported.
`hyperparameter_hvp(direction,product,status)` differentiates this envelope
through the converged Newton mode using the posterior factorization and the
kernel parameter-HVP/VJP primitives. It is an implicit-optimum HVP, not the
fixed-mode transaction used by `set_parameters`; `hyperparameter_hvp_device`
dispatches CPU and returns `FORTNUM_NOT_IMPLEMENTED` for selected CUDA.
Differentiating the full Laplace evidence, including mode-curvature terms, is
a separate contract. Variational likelihoods, shared multiclass coupling, and
derivative-observation classifier paths remain explicit roadmap work. See
[`GP_CLASSIFICATION_HYPERPARAMETER_HVP.md`](GP_CLASSIFICATION_HYPERPARAMETER_HVP.md).

`gp_multiclass_classification_t` provides deterministic one-vs-rest multiclass
GP classification over the same binary Laplace models. It fits one model per
sorted integer class, normalizes their positive probabilities onto a simplex,
and accepts the same optional `sample_weight` vector for every class head;
the row weights are validated before any class state is allocated.
and exposes `classes`, `class_count`, `feature_count`, `predict_proba`,
`predict`, and `fitted`. `parameter_count()` and `parameters()` concatenate
the read-only kernel metadata for each one-vs-rest model in sorted class order.
`decision_function` returns the one-vs-rest latent posterior means in sorted
class order, before probability-simplex normalization; its
`decision_function_jvp` and `decision_function_vjp` propagate query-feature
tangents and cotangents through every binary GP. `predict_proba_vjp` applies
the simplex normalization adjoint before accumulating binary GP input bars.
`decision_function_parameter_jvp`/`_vjp` and
`predict_proba_parameter_jvp`/`_vjp` provide packed fixed-state kernel-log
parameter products. Each block follows the corresponding sorted class in
`parameters()`, and the probability products include the simplex quotient
rule before dispatching the binary fixed-state products. The explicit
`predict_proba_parameter_jvp_device`/`_vjp_device` methods dispatch selected
CPU contexts and return `FORTNUM_NOT_IMPLEMENTED` for CUDA until a resident
OVR Laplace/reduction graph is linked; no host fallback is implied.
`hyperparameter_gradient` concatenates the exact binary envelope gradients in
sorted-class order. It is the gradient of the sum of the independent
one-vs-rest Laplace-mode log posteriors; a shared coupled categorical objective
remains a separate contract. The wrapper inherits the selected logistic or probit likelihood and
kernel/refusal behavior. It is a coupling policy rather than a multinomial
likelihood, so variational categorical likelihoods and shared multiclass
hyperparameter training remain separate work.

`predict_proba_device` and `predict_device` dispatch exactly to the reference
path for a selected CPU context.  A selected CUDA context returns
`FORTNUM_NOT_IMPLEMENTED`: the independent per-class covariance and Laplace
states are not resident on CUDA yet, and no hidden host fallback or unaccounted
transfer is allowed.  `device_supported(FORTML_DEVICE_CPU)` is true only for
a fitted wrapper; CUDA is always reported unsupported until a resident
multiclass Laplace kernel is linked.

The same module exposes a backend-independent likelihood primitive,
`gp_classification_log_likelihood_value(eta,likelihood,value,status)`, for a
vector of signed latent margins. `eta(i)` is the encoded label times the
latent function, and the result is the summed Bernoulli log likelihood.
`gp_classification_log_likelihood_jvp` and `_vjp` provide exact products with
respect to those margins for both `GP_LIKELIHOOD_LOGISTIC` and
`GP_LIKELIHOOD_PROBIT`, including stable evaluation in the negative probit
tail. Finite inputs and tangent/cotangent shapes are checked explicitly.
This primitive is separate from fitted Laplace state so it can be composed
into variational or minibatch objectives; its presence does not imply a
complete resident-GPU GP training path. `device_supported(kind)` reports
CPU-only support for fitted models, and `predict_latent_device`/
`predict_proba_device` return `FORTNUM_NOT_IMPLEMENTED` for CUDA until a
resident covariance/Laplace kernel exists. The likelihood helper's
`gp_classification_likelihood_device_supported` function likewise refuses
CUDA value/JVP/VJP products, preserving the no-hidden-host-fallback contract.

### `fortml_gp_multilabel_classification`

`gp_multilabel_classification_t%fit(x,indicators,kernel,status[,options,state,
sample_weight,thresholds])` fits one independent binary Laplace GP per
indicator column.  The indicator matrix must contain only zero and one, each
column must contain both values, and finite nonnegative sample weights must
have positive mass.  The selected logistic or probit likelihood and Newton
settings are copied to every head.  `predict_proba` returns positive
probabilities without cross-label normalization; `predict` applies the stored
per-label thresholds and returns an indicator matrix.

`predict_latent` and `predict_proba` expose query-input JVP/VJP products, while
their `*_parameter_jvp`/`*_parameter_vjp` variants use the concatenated fixed-
state kernel-log parameter vector in label order.  `parameters()`,
`parameter_count()`, and `hyperparameter_gradient()` expose the same packed
layout; the gradient is the concatenation of the binary Laplace envelope
gradients for the independent mode posteriors.  `set_thresholds` updates only
the prediction policy and validates every threshold in `(0,1)`.

Selected CPU device calls dispatch to the reference path.  CUDA latent,
probability, and label requests return `FORTNUM_NOT_IMPLEMENTED` until resident binary
Laplace states, solves, and the multilabel reduction are linked; no host
fallback is implied.  The independent behavioral oracle is
`test_gp_multilabel_classification`, and the cross-engine correctness record
is `fortml-bench/results/GP_MULTILABEL.md`.

### `fortml_gp_classification_training`

`gp_classification_optimize_hyperparameters(model,x,labels,kernel,options,
result,status[,sample_weight])` runs bounded FortOpt L-BFGS-B over the binary classifier's
recursive kernel-log parameter vector. Every trial refits the damped Laplace
mode and consumes `hyperparameter_gradient`; it therefore differentiates the
converged mode log posterior without finite-difference gradients. The caller's
`kernel` is updated in place and the final fitted state is left in `model`.
When supplied, finite nonnegative row weights are validated once and passed
through every refit, so the optimizer objective and envelope gradient match
the weighted fit contract.
`gp_classification_hyperparameter_options_t%fit` carries the logistic/probit
Newton settings, while the remaining fields carry memory, convergence, and
uniform log-parameter bounds. A failed mode solve, invalid bound, nonfinite
value, or iteration limit is returned through `fortnum_status_t`.

`gp_multiclass_optimize_hyperparameters(model,x,labels,kernel,options,result,
status[,sample_weight])` provides the corresponding shared-kernel one-vs-rest adapter. It
optimizes one constructor-kernel vector shared by all sorted classes and sums
the independent binary envelope gradients, routing one validated row-weight
vector to every class. The packed per-class metadata
returned by `gp_multiclass_classification_t%parameters()` is intentionally not
optimized as independent blocks by this adapter. Neither adapter differentiates
the full Laplace evidence, likelihood parameters, or an implicit mode HVP;
those boundaries remain explicit refusals/roadmap work.

### `fortml_derivative_gaussian_process`

`gp_derivative_regression_t%fit(x,components,y,kernel,noise_variance,status
[,jitter])` accepts an observation component for each row. Component 0 is a
function value. Component `j` in `1:n_features` is the first derivative with
respect to input `j`. Rows may be interleaved in any order and targets may have
multiple columns.

`predict(x,components,mean,variance,status)` uses the same component convention.
`observation_count()` returns the number of fitted rows. The packed parameter
order is the kernel log parameters followed by log observation-noise variance.
`parameter_count`, `parameters`, and `set_parameters` expose that state.
`predict_jvp(x,components,direction,mean,mean_dot,variance,variance_dot,status)`
returns the prediction and its parameter JVP. `predict_vjp(x,components,
mean_bar,variance_bar,parameter_bar,status)` is the corresponding reverse
product over packed kernel/noise parameters. `predict_input_jvp(x,components,
direction,mean,mean_dot,variance,variance_dot,status)` and
`predict_input_vjp(x,components,mean_bar,variance_bar,x_bar,status)` provide
query-input products while holding model parameters and training inputs fixed.
`predict_input_hvp(x,components,direction,mean,mean_dot,variance,variance_dot,
status)` provides the value-query Hessian-vector product with respect to the
query coordinates (the fitted parameters and training inputs remain fixed).
The HVP is symmetric and linear in `direction`; derivative-observation query
components are rejected with a typed `FORTNUM_NOT_IMPLEMENTED` boundary until
the required fourth input derivatives are generated.
The query products use exact third-input products for RBF, Matérn 3/2, Matérn
5/2, periodic, local-periodic, rational-quadratic, cosine, linear, constant, polynomial,
spectral-mixture, and sum/product kernels when the polynomial base is positive.
The smooth leaf and composition rules are propagated directly through the
value, first-gradient, and mixed-Hessian covariance blocks; no finite-
difference fallback is used. Matérn 1/2 remains restricted to noncoincident
queries because its derivative covariance is singular at coincidence. User
formula leaves and other nonsmooth/unsupported leaves return
`FORTNUM_NOT_IMPLEMENTED` for query-input products. Joint posterior covariance
is available through `joint_covariance(x,components,covariance,status)`. It
returns the dense latent posterior covariance in the requested mixed-query
order (observation noise is excluded) and applies the same smoothness and
white-noise refusal rules as `predict`. `joint_covariance_device(device,x,
components,covariance,status)` dispatches selected CPU contexts exactly and
returns `FORTNUM_NOT_IMPLEMENTED` for CUDA until the resident derivative-GP
covariance graph is linked. No hidden host fallback is used.
`joint_covariance_jvp(x,components,direction,covariance,covariance_dot,status)`
and `joint_covariance_vjp(x,components,covariance_bar,parameter_bar,status)`
differentiate that dense latent posterior with respect to the packed
kernel-log/noise-log parameters. The JVP differentiates the prior, cross
covariance, and Cholesky solve exactly; the VJP uses the symmetric covariance
cotangent and is adjoint to the JVP. These parameter products are CPU-only
until the resident covariance graph is linked and never silently finite-
difference.
`joint_covariance_jvp_device(device,x,components,direction,covariance,
covariance_dot,status)` and `joint_covariance_vjp_device(device,x,components,
covariance_bar,parameter_bar,status)` make that backend boundary explicit:
selected CPU contexts dispatch exactly, while selected CUDA contexts return
`FORTNUM_NOT_IMPLEMENTED` before touching outputs.
`log_marginal_likelihood`, `log_marginal_likelihood_jvp`,
`log_marginal_likelihood_vjp`, `hyperparameter_gradient`,
`hyperparameter_vjp`, and `hyperparameter_hvp` provide likelihood products.
The scalar VJP accepts an objective cotangent and returns the packed pullback.
The gradient uses analytic parameter tangents of the supported RBF,
Matérn 1/2, 3/2, 5/2, periodic, rational-quadratic, cosine, linear, constant,
polynomial, spectral-mixture, validated user-formula, and sum/product kernels.
Matérn 1/2 still refuses
coincident derivative
observations, as do user formulas containing `push_distance` at coincidence.

`predict_device(device,x,components,mean,variance,status)` is the explicit
backend boundary for mixed observations. A selected CPU context dispatches to
`predict` exactly. CUDA currently returns `FORTNUM_NOT_IMPLEMENTED` because a
resident covariance/factorization/derivative-query kernel is not linked;
`device_supported(FORTML_DEVICE_CPU)` is true only for a fitted model and
CUDA is reported false. Parameter and query-input JVP/VJP products remain
host-only until the same resident derivative graph exists, so no derivative
product silently copies arrays to the host.
Value-only covariances and their variance-parameter products remain defined at
coincidence. The refusal applies only when an input derivative is requested.
For mixed value/first-derivative observations, `hyperparameter_hvp` is analytic
for RBF, periodic, linear, constant, polynomial, spectral-mixture, and sums/products
built solely from those leaves. The polynomial path differentiates all four
logarithmic kernel coordinates in closed form, including the degree-one limit,
and returns a typed domain error when its positive base is invalid.
Periodic and spectral-mixture value/first-derivative parameter gradients, query products,
and mixed parameter HVPs are analytic on the CPU reference path. Its
four-jet factor rule differentiates each packed log-weight/log-scale/signed-
mean coordinate along an arbitrary parameter direction. The periodic path
uses coincidence-safe radial fourth-input products for all three logarithmic
kernel coordinates. Other leaves return
`FORTNUM_NOT_IMPLEMENTED` until their second input/parameter products are
generated; the implementation never silently finite-differences the likelihood
gradient. CUDA mixed covariance/factorization remains an explicit typed refusal.
The RBF parameter JVP/VJP path uses the checked FortSym-generated
natural-leaf value and first derivatives (FortSym `f71a1aa`, 15 IR nodes, 7
compound operations). The Matérn 1/2 HVP now uses a FortSym-generated leaf
(`9482261`, 37 IR nodes, 28 compound operations), and the Matérn 3/2 HVP now
uses a FortSym-generated leaf (`b72a23a`, 60 IR nodes, 48 compound
operations), each after an independent analytic/directional finite-difference
test. Matérn 5/2 now uses the FortSym-generated leaf `873d33f` (80 IR nodes,
65 compound operations), checked by `test_fortsym_matern52`. User formulas use the validated forward
derivative stack and do not call a procedure pointer.

`fortml_derivative_gp_training` provides
`gp_optimize_derivative_hyperparameters` with the same bounded FortOpt
L-BFGS-B options and result type as exact value GPs. Optimization consumes the
analytic gradient. It does not silently substitute the finite-difference HVP.

### `fortml_multi_output_gp`

`multi_output_gp_t` implements intrinsic coregionalization with
`B = W W^T + diag(independent)` and one shared input kernel. Call
`initialize(kernel,weights,independent,noise_variance,status)`, then
`fit(inputs,targets,status)` and `predict(query,mean,status)`. Target and mean
arrays use `(sample,output)` order.

`joint_covariance(inputs,matrix,status)` returns `B` Kronecker `K` in
output-major vector order. `log_marginal_likelihood(targets,value,status)`
expects the same targets used for the fitted factorization. This type exposes
posterior mean plus `parameter_count()`/`parameters()` in the packed order
`[kernel parameters, log(noise variance), output-major W, independent]`.
`predict_input_jvp`/`predict_input_vjp` provide fixed-fit query products;
`predict_parameter_jvp`/`predict_parameter_vjp` differentiate both the
cross-covariance and the Cholesky solve. The independent
`test_multi_output_gp_products` and `fortml-bench` lane check finite differences,
adjoints, and the typed CUDA refusals. See
[docs/MULTI_OUTPUT_GP_PRODUCTS.md](MULTI_OUTPUT_GP_PRODUCTS.md). Posterior
variance, parameter HVPs, and resident CUDA covariance/derivative kernels remain
open. `joint_covariance_parameter_jvp`/`joint_covariance_parameter_vjp` expose
exact prior-covariance products over the same packed coordinates; the
log-noise coordinate is zero because the prior excludes observation noise.
Their CPU device wrappers are exact and CUDA dispatch is a typed refusal. The
`predict_batch` family accepts `(batch,query,feature)` inputs and
returns `(batch,query,output)` means; `predict_batch_input_jvp` and
`predict_batch_input_vjp` apply the same fixed-fit query products independently
to each batch member. CPU device wrappers are exact, while the batch CUDA
wrappers return typed `FORTNUM_NOT_IMPLEMENTED` until resident
coregionalized covariance, factorization, and derivative kernels exist.

## Approximate Gaussian processes

### `fortml_sparse_gp`

`sparse_gp_t` is a scalar-output inducing-point variational GP with Gaussian
likelihood. Initialize it with inducing points, a kernel, and a positive noise
variance. `set_variational(mean,factor,status)` takes the mean and the lower
Cholesky factor of the inducing covariance. Its diagonal must be positive.

`elbo(x,y,value,status[,expected_log_likelihood,kl_value])` evaluates the
closed-form bound. `predict(x_star,mean,variance,status)` returns variational
latent marginals. `parameter_count()`/`parameters()` and `set_parameters()` pack
the variational mean followed by lower-Cholesky columns (log diagonals), while
`elbo_gradient`, `elbo_jvp`, and the scalar-cotangent `elbo_vjp` provide exact
CPU products for that vector. Their independent finite-difference and
dot-product oracles live in `test_sparse_gp`; `elbo_device` and the product
dispatchers execute on CPU and return `FORTNUM_NOT_IMPLEMENTED` for CUDA until
the inducing solve and reductions are resident. The complementary
`elbo_kernel_parameter_jvp` and `elbo_kernel_parameter_vjp` differentiate the
packed `kernel%parameters()` vector at fixed variational state, inducing
locations, and noise variance, including the explicit inducing solve
sensitivity. Their device dispatchers execute on CPU and return the same typed
CUDA refusal. The Gaussian likelihood has a separate one-coordinate packed
block: `likelihood_parameter_count()` is one after initialization,
`likelihood_parameters()`/`hyperparameters()` return
`[log(noise_variance)]`, and `set_likelihood_parameters()`/
`set_hyperparameters()` validate and commit it transactionally. The fixed-state
`elbo_likelihood_parameter_jvp` (also named
`elbo_hyperparameter_jvp`), scalar-cotangent VJP, and HVP provide analytic
products through this coordinate while holding the variational state, kernel,
and inducing locations fixed. CPU device wrappers are exact; CUDA wrappers
return typed `FORTNUM_NOT_IMPLEMENTED` rather than copying data to the host.
The independent `test_sparse_gp_likelihood_noise` fixture covers central
differences, adjoint duality, HVP finite differences, malformed/overflowing
state preservation, and device refusals.

### `fortml_gp_variational_classification`

`gp_variational_classification_t` is the Bernoulli counterpart of
`sparse_gp_t`. `initialize(inducing_points,kernel,n_mc_samples,seed,status
[,likelihood,jitter])` constructs an inducing posterior and fixes a seeded
reparameterization table. The supported likelihood constants are
`GP_VARIATIONAL_LOGISTIC` and `GP_VARIATIONAL_PROBIT`; labels are integer
zero/one values. The packed vector returned by `parameters()` contains the
inducing mean followed by lower-Cholesky covariance columns, with each
diagonal represented in log coordinates. `set_parameters` validates positive
diagonals and finite values.

`elbo(x,labels,value,status[,expected_log_likelihood,kl_value,scale,sample_weight])`
uses the deterministic Monte Carlo expected Bernoulli log
likelihood and the analytic `KL(q(u)||N(0,K_uu))`. `elbo_gradient` gives the
analytic gradient of that same packed objective, including the variance
reparameterization and KL terms. `elbo_jvp` is the matching forward
directional product; finite differences of `elbo` therefore provide a direct
independent oracle. Optional finite, nonnegative `sample_weight` values weight
each row of the expected likelihood and require positive total mass; `scale`
scales only that weighted likelihood for minibatch callers and never the KL.
The same weights are accepted by the bounded
`gp_variational_classification_optimize` FortOpt adapter. `predict_latent` returns
the posterior latent mean and variance, while `predict_proba` applies the
logistic variance correction or analytic probit Gaussian integral and returns
columns `[negative,positive]`. Their parameter-JVP and parameter-VJP variants
differentiate the packed variational mean/log-Cholesky vector. The VJP accepts
cotangents for both latent outputs or both probability columns and satisfies
the JVP/VJP dot-product identity. `predict_latent_input_jvp`/
`predict_latent_input_vjp` and their probability counterparts differentiate
query coordinates with the inducing state fixed; an independent oracle checks
central differences and the adjoint identity. The fixed-state kernel products
`predict_latent_kernel_parameter_jvp`,
`predict_proba_kernel_parameter_jvp`,
`predict_latent_kernel_parameter_vjp`, and
`predict_proba_kernel_parameter_vjp` use the kernel log-hyperparameter vector
(`kernel_parameter_count()`) and differentiate the `K_uu` solve, `K_ux`
cross-covariance, and diagonal `K_xx` terms. Their independent
finite-difference and JVP/VJP oracle is
`test_gp_variational_kernel_products`.
CPU dispatch executes these products exactly; CUDA prediction and
reverse-product paths return
`FORTNUM_NOT_IMPLEMENTED`
until the inducing solve, likelihood evaluation, and reduction are resident.
No hidden host fallback is used. `gp_variational_classification_optimize`
provides a bounded FortOpt L-BFGS-B adapter over the packed ELBO state, with
`gp_variational_classification_lbfgsb_options_t` bounds/tolerances and a
`gp_variational_classification_lbfgsb_result_t` convergence/ELBO/gradient
diagnostic. The adapter maximizes the deterministic ELBO through its negative
objective, commits the packed state only on convergence, and restores the
initial packed state on optimizer refusal, nonfinite results, or an iteration
limit. It returns a typed CUDA refusal. Kernel and inducing-point hyperparameter products,
natural-gradient updates, and resident GPU inference remain separate roadmap
work.

### `fortml_gp_variational_multiclass_classification`

`gp_variational_multiclass_classification_t` is a bounded one-vs-rest wrapper
around independent Bernoulli variational GPs. `initialize(inducing_points,
classes,kernel,n_mc_samples,seed,status[,likelihood,jitter])` sorts and
validates unique integer classes, creates one seeded model per class, and
packs their mean/log-Cholesky vectors in sorted-class order. `elbo`,
`elbo_gradient`, and `elbo_jvp` sum the independent binary objectives; their
optional `sample_weight` argument is validated once and shared by every class
head, while `scale` multiplies each weighted likelihood term;
`predict_latent` returns per-class margins/variances and `predict_proba`
normalizes positive margins to a simplex. `predict_proba_parameter_jvp` and
`predict_proba_parameter_vjp` provide the exact packed-parameter tangent and
reverse product through the OVR simplex normalization, and `predict` uses
first-max ties in sorted class order. CPU dispatch is exact; CUDA ELBO and
prediction/reverse-product requests return a typed refusal until a resident OVR
graph is linked. Coupled categorical likelihoods, natural gradients, and
kernel/inducing hyperparameter products remain open. `kernel_parameter_count()`
and `predict_latent_kernel_parameter_jvp`,
`predict_proba_kernel_parameter_jvp`,
`predict_latent_kernel_parameter_vjp`, and
`predict_proba_kernel_parameter_vjp` expose shared fixed-state kernel products;
the direction is applied to every copied binary kernel and reverse class terms
are accumulated. Resident GPU kernel products remain open.

### `fortml_gp_variational_categorical_classification`

`gp_variational_categorical_classification_t` is the coupled categorical
counterpart to the OVR variational wrapper. `initialize` sorts unique integer
classes and creates one inducing posterior per class. `fit` initializes that
state and maximizes a deterministic categorical ELBO with FortOpt L-BFGS-B.
The likelihood is a stable softmax of variance-corrected latent means, with
`pi/8` for logistic and `1` for probit. The ELBO subtracts the analytic KL of
every inducing posterior. `elbo_gradient` and `elbo_jvp` include both terms.

`predict_latent` returns per-class means and variances. `predict_proba` returns
coupled simplex probabilities and `predict` uses sorted-class first-max ties.
`predict_proba_parameter_jvp`/`predict_proba_parameter_vjp` and
`predict_proba_input_jvp`/`predict_proba_input_vjp` differentiate the complete
softmax, variance correction, and inducing projection. `parameters()` packs
the per-class mean/log-Cholesky vectors in sorted-class order. The fixed-state
temperature coordinate additionally exposes
`predict_proba_likelihood_parameter_hvp`, the forward-over-reverse product of
the probability VJP, and `elbo_likelihood_parameter_hvp`, the exact weighted
ELBO curvature product. Kernel hyperparameter products remain outside this
bounded slice. CPU dispatch is exact;
CUDA methods return `FORTNUM_NOT_IMPLEMENTED` until resident inducing solves,
softmax reductions, reverse kernels, and HVP reductions are linked. See
`docs/GP_VARIATIONAL_CATEGORICAL.md` and the independent oracle
`test_gp_variational_categorical_classification`.

### `fortml_gp_ordinal_classification`

`gp_ordinal_classification_t` is an ordered latent-Gaussian GP baseline. `fit`
sorts integer labels, maps them to ranks, and fits one zero-mean
`gp_regression_t`; fixed mid-rank cut points map the predictive Gaussian to
adjacent normal-CDF class probabilities. `classes()`, `thresholds()`,
`predict_latent`, `predict_proba`, and `predict` expose the ordered state while
preserving the caller's integer labels. `gp_ordinal_classification_options_t`
controls positive latent noise variance and jitter.

Packed kernel/noise products are available through
`predict_latent_parameter_jvp/vjp` and
`predict_proba_parameter_jvp/vjp`. Input JVP/VJP products differentiate the
kernel cross-covariance and posterior variance analytically. The exact latent
evidence additionally exposes `hyperparameter_gradient`,
`hyperparameter_hvp(direction, product)`, and
`log_marginal_likelihood_jvp`; these operate over the packed
`[kernel..., log(noise variance)]` block. The companion
`gp_ordinal_optimize_hyperparameters` adapter consumes the analytic gradient
with bounded FortOpt L-BFGS-B and restores the initial state on failure.
CPU is the exact reference; prediction, reverse-product, evidence-product,
and optimizer CUDA methods return `FORTNUM_NOT_IMPLEMENTED` until resident
ordinal kernels and factorization are linked. This is a latent-Gaussian
ordered surrogate, not a native cumulative-likelihood fit; optimized cut
points and ordinal likelihood hyperparameters remain open. The independent
prediction oracle is `test_gp_ordinal_classification`, the evidence and
optimizer oracle is `test_gp_ordinal_classification_hyperparameters`, and the
design note is
[`docs/GP_ORDINAL_CLASSIFICATION.md`](GP_ORDINAL_CLASSIFICATION.md).

### `fortml_student_t_process`

`student_t_process_t` is a dense scalar Student-t process regression baseline.
`fit(x,y,kernel,nu,noise_variance,status[,jitter])` requires `nu > 2`, factors
the noisy kernel matrix, and records the observed Mahalanobis distance. `predict`
returns the posterior mean and marginal variance; unlike a Gaussian process,
the variance is multiplied by `(nu + beta - 2)/(nu + n - 2)`, so it responds to
how surprising the observed values were. `posterior_dof`, `covariance_scale`,
and `log_marginal_likelihood` expose the fitted state. The large-`nu` GP limit,
data-dependent variance contrast, and invalid-input refusals are checked by
`test_student_t_process`; see [`docs/GP_STUDENT_T_PROCESS.md`](GP_STUDENT_T_PROCESS.md).
This is a CPU dense reference path: derivative products, sparse/variational
inference, and resident CUDA execution remain open.

### `fortml_heteroskedastic_gp`

`heteroskedastic_gp_t` accepts a positive observation-variance vector in
`fit(x,y,noise_variance,signal_kernel,noise_kernel,status[,jitter])`. The
signal posterior uses the row-specific diagonal and `predict` returns latent
mean/variance. `noise_at` interpolates the supplied log variances with the
second kernel and exponentiates the result, retaining positivity and reverting
to the geometric-mean level away from observations. Constant noise reduces
exactly to `gp_regression_t`; `test_heteroskedastic_gp` covers that oracle,
quiet/noisy posterior behavior, interpolation, and typed refusals. Joint noise
inference, derivative products, approximate inference, and resident CUDA remain
open; see [`docs/GP_HETEROSKEDASTIC.md`](GP_HETEROSKEDASTIC.md).

### `fortml_robust_gp`

`robust_gp_t` is a dense Laplace GP for non-Gaussian observations. Select
`FORTML_LIKELIHOOD_POISSON` for nonnegative counts and a positive latent
log-rate, or `FORTML_LIKELIHOOD_STUDENT_T` for an outlier-resistant location
model. `fit` exposes convergence, iteration, mode, curvature, and stationary
`alpha` state; `predict_latent` returns the Laplace latent marginal and
`predict_response` returns a positive Poisson rate or Student-t response
summary. `test_robust_gp` independently checks Poisson stationarity/rates,
Student-t outlier resistance, and typed input/convergence refusals. Exact
non-Gaussian evidence, derivative products, approximate scalable inference,
and resident CUDA remain open; see [`docs/GP_ROBUST.md`](GP_ROBUST.md).

### `fortml_sparse_prior_gp`

`sparse_prior_gp_t` implements `SPARSE_SOR`, `SPARSE_DTC`, `SPARSE_FITC`, and
`SPARSE_PITC`. `sparse_prior_method_name` returns the display name.

Call `initialize(inducing_points,kernel,noise_variance,method,status
[,block_size])`. PITC requires a positive `block_size`. Then call
`fit(x,y,status)`, `predict(query,mean,variance,status)`, or
`log_marginal_likelihood(value,status)`. Targets are scalar. Inducing locations
and kernel parameters remain fixed during a fit.

### `fortml_local_experts`

`local_expert_gp_t` implements `AGGREGATE_NLE`, `AGGREGATE_POE`,
`AGGREGATE_GPOE`, `AGGREGATE_BCM`, `AGGREGATE_RBCM`, `AGGREGATE_GRBCM`, and
`AGGREGATE_MOE`. `aggregation_name` returns the display name.

After `initialize(kernel,noise_variance,method,status)`,
`fit(x,y,n_experts,status[,communication_seed])` uses consecutive groups.
`fit_clustered(x,y,n_experts,status[,max_iterations][,communication_seed])`
uses Lloyd k-means with deterministic farthest-point initialization. Both fit
scalar targets.

For GRBCM, `n_experts = M` must be at least two and includes the communication
expert. A seeded draw without replacement selects disjoint `D_c` of size
`ceiling(n/M)`. The other observations form `M-1` groups. The communication
model fits `D_c`, and each enhanced expert fits `D_c` union `D_i`. The default
seed is 104729. Prediction sets the first enhanced weight to one, derives later
weights from the communication-to-enhanced entropy drop, and applies the
communication correction. `expert_count()` returns `M`.

`predict(query,mean,variance,status)` aggregates scalar marginals.

### `fortml_ski_gp`

`ski_operator_t` represents `W K_uu W^T + noise I` on a regular inducing grid.
Call `initialize(inputs,kernel,n_grid,noise_variance,status)`. Inputs must be
finite and every dimension must span a nonzero range. For one input,
`n_grid` is the grid size. The caller must supply a stationary kernel for its
cached Toeplitz covariance. For `d > 1`,
`n_grid` is a total budget. The implementation chooses the largest common
extent `q` with `q**d <= n_grid`, uses multilinear interpolation over `2**d`
corners, and accepts only one isotropic RBF leaf. The type implements the
`linear_operator_t` methods. `interpolation_weights(row,indices,values)`
reports the two weights only for the one-dimensional path and returns zeros
otherwise.

`cross_matvec(query,input,output,status)` applies
`W_query K_uu W_train^T` to training-space coefficients. Query coordinates
outside the training grid clamp to its boundary. Observation noise is excluded.

`subset_of_data_indices(n_samples,subset_size,indices,status)` returns a
deterministic evenly spaced SoD subset. It includes the first and last input
when the subset has more than one element.

### `fortml_review_toy`

This module is a reproducible benchmark fixture. `review_toy_truth(x)` is the
normalized sinc target. `review_toy_data` produces seeded noisy training data,
and `review_toy_grid` produces a one-dimensional regular grid. The constants
`REVIEW_TOY_SAMPLES`, `REVIEW_TOY_NOISE_VARIANCE`, `REVIEW_TOY_LOWER`, and
`REVIEW_TOY_UPPER` give the 120-sample default, noise variance 0.04, and range
`[-7,7]`.

## Linear operators and large GP inference

### `fortml_linear_operator`

An extension of `linear_operator_t` must provide `matvec`, `matmat`,
`diagonal`, and `sample_count`. The base type supplies:

```text
solve_cg(rhs, solution, tolerance, max_iterations,
         info, iterations, residual_norm [,use_diagonal_preconditioner])
solve_cg_multi(rhs, solution, tolerance, max_iterations,
               info, iterations, residual_norm
               [,use_diagonal_preconditioner,
                 preconditioner_block_size,
                 preconditioner_nystrom_rank])
```

`solution` is an `intent(inout)` initial guess. The diagonal preconditioner is
the default. The base type rejects nonzero block or Nystrom options. The kernel
operators below expose named variants for those choices.

### `fortml_kernel_operator`

`rbf_operator_t%initialize(points,variance,lengthscale,diagonal_shift,status
[,tile_size])` is the specialized RBF path. `kernel_operator_t%initialize(
points,kernel,diagonal_shift,status[,tile_size])` accepts built-in composite
kernels and validated user formulas. Both expose `matvec`, `matmat`,
`diagonal`, `sample_count`, scalar and multi-RHS CG, and block or Nystrom
multi-RHS variants. The free `rbf_matvec_tiled` and `rbf_matmat_tiled`
procedures expose the reference tiled product.

`kernel_operator_t%device_supported()` reports whether its static program fits
the device contract. Call `enter_data(status[,n_rhs])` before
`matvec_device`, `matmat_device`, `solve_cg_device`, or
`solve_cg_multi_device`, and call `exit_data(status)` after the last resident
operation. The optional `n_rhs` allocates a persistent Krylov workspace of
that width. Exit the current data before changing the width.

The specialized RBF operator also accepts `enter_data(status[,n_rhs])`. Its
ordinary solve can manage a temporary workspace when no resident lifetime was
opened. Native CUDA is optional. The Fortran/OpenACC implementation is the
fallback.

### Other operator modules

| Module and type | Initialization and added operations |
| --- | --- |
| `fortml_sparse_operator::sparse_gp_operator_t` | `initialize(n_samples,rows,columns,values,status)` from one-based triplets. Resident `matvec_device`/`matmat_device`. `nonzero_count`. |
| `fortml_structured_operator::structured_gp_operator_t` | `initialize(tensor_factor_t(:),status)`. Resident products. `set_derivative_factors` and derivative products by tensor dimension. |
| `fortml_toeplitz_operator::toeplitz_gp_operator_t` | `initialize(column,status[,row])`. Host cached Toeplitz products and inherited CG. |
| `fortml_banded_precision::banded_precision_operator_t` | `initialize(band(0:,:),status)`. Factorization, precision solve, precision log determinant, and covariance log determinant. |

The sparse triplet constructor follows `fortsparse` indexing and status rules.
For the banded type, `band(k,i)` stores the entry `k` rows below the diagonal
in column `i`. `make_ornstein_uhlenbeck_precision` constructs the tridiagonal
band for a regular one-dimensional grid.

### `fortml_multilevel_grid`

`multilevel_grid_t%initialize(dimensions,n_levels,status)` creates a hierarchy
with level 1 finest. `level_count`, `level_size`, and `level_dimensions` query
it. `prolong(level,coarse,fine,status)` maps level `level+1` to `level`.
`restrict` is its exact transpose.

### `fortml_lanczos`

`lanczos_log_determinant(operator,n_probes,n_steps,seed,value,status)` applies
seeded stochastic Lanczos quadrature to any positive-definite
`linear_operator_t`. `lanczos_predictive_variance(operator,cross_covariance,
prior_variance,n_steps,variance,status)` evaluates one LOVE-style marginal.
The latter runs one Lanczos sweep per cross-covariance vector. It is not a
batched constant-time predictive cache.

### `fortml_inference_policy`

Fill `inference_problem_t` with sample, feature, and output counts and at most
one declared structure: tensor grid, compact support, or Markov precision.
An optional positive `inducing_budget` requests the inducing-point path.
`select_inference_policy(problem,choice,status)` returns one of:

```text
INFERENCE_EXACT_CHOLESKY   INFERENCE_STRUCTURED_GRID
INFERENCE_SPARSE_COMPACT  INFERENCE_BANDED_PRECISION
INFERENCE_MATRIX_FREE_KRYLOV  INFERENCE_INDUCING_POINT
```

`inference_policy_name` returns a stable lowercase name. The dense default
limit is `DENSE_SAMPLE_LIMIT = 4096`, applied to
`n_samples*n_outputs` when no structure or inducing budget is declared.
`inference_choice_t%policy` stores the selected constant and `%reason` stores
the allocation or structure that selected it.

## Parameter packing

### `fortml_parameter_registry`

`parameter_block_t%initialize(name,n_parameters,context,getter,setter,status)`
binds a named range to a live `target` object. The context must outlive the
block and every registry containing it. `parameter_block_from_mlp`,
`parameter_block_from_kernel`, `parameter_block_from_gp`,
`parameter_block_from_ridge`, `parameter_block_from_elastic_net`,
`parameter_block_from_logistic`, `parameter_block_from_softmax`,
`parameter_block_from_basis_pipeline`, and
`parameter_block_from_sequential_pipeline` provide typed adapters. A block
also exposes `name`, `size`, `get`, `set`, and `initialized`.

`parameter_registry_t` exposes `clear`, `add`, `block_count`,
`parameter_count`, `pack`, `unpack`, and `range`. Names must be nonempty and <!-- slop-ok -->
unique. Blocks retain insertion order. The public callback contracts are
`parameter_get_proc` and `parameter_set_proc`. `parameter_block_from_mlp_chain`
binds one named range to a live `mlp_chain_t`, so a composed network can share
the same flat registry as MLPs, kernels, and estimators.

### `fortml_hyperparameter_registry`

`hyperparameter_block_t` adds optimizer metadata to a named block. Use
`initialize` for a live callback target or `initialize_values` for an owned
vector. `HYPERPARAMETER_IDENTITY`, `HYPERPARAMETER_LOG`, and
`HYPERPARAMETER_LOGIT` define the physical/unconstrained coordinate maps;
logit blocks require explicit finite distinct bounds. Blocks expose
`get/set_physical`, `get/set_unconstrained`, `unconstrained_bounds`,
`project_unconstrained`, `trainable`, `hvp_available`, `provenance`, and
`device`. Non-finite values, invalid bounds, and transform-domain violations
return `FORTNUM_DOMAIN_ERROR`.

`hyperparameter_registry_t` preserves insertion order and exposes `pack`,
`unpack`, `pack_unconstrained`, `unpack_unconstrained`, and trainable-only
`pack_trainable`/`unpack_trainable` vectors. `optimizer_bounds` and `project`
return the trainable unconstrained vector and its projected L-BFGS-B bounds;
frozen blocks are omitted without changing their live model state. The
registry is a coordinate/metadata layer only: an objective must provide its
own analytic gradient/HVP and explicit device capability.

`physical_derivatives(unconstrained,physical,first,second,status)` returns the
exact physical value, first derivative, and second derivative for each
separable coordinate. `unconstrained_gradient` pulls a physical gradient back
to the trainable optimizer vector. `unconstrained_hvp` applies the exact
coordinate chain rule to a physical HVP evaluated along the transformed
direction, including transform curvature. These are pure coordinate products;
the model objective still owns cross-block derivatives.

### `fortml_parameter_products`

`parameter_products_from_mlp(products,name,model,status)` and
`parameter_products_from_gp(products,name,model,status)` bind a live target
model. `parameter_products_from_mlp_chain(products,name,model,status)` provides
the same packed value/JVP/VJP/HVP seam for a named sequential MLP tree. A GP
must already be fitted. The resulting `parameter_products_t`
exposes `initialized`, `parameter_count`, `pack`, `unpack`, `range`, `value`, <!-- slop-ok -->
`jvp`, `vjp`, `hvp`, and `has_hvp`.

The MLP adapter evaluates `predict`. Its JVP/VJP/HVP vary packed model
parameters while holding inputs fixed. The GP adapter evaluates predictive
mean and packs kernel parameters followed by log noise variance. Both current
adapters return true from `has_hvp`. The callback contracts
`parameter_value_proc`, `parameter_jvp_proc`, `parameter_vjp_proc`, and
`parameter_hvp_proc` define the extension seam.
### `fortml_lightgbm`

`lightgbm_t` is a separately named, bounded LightGBM-style numeric boosting
policy. It does not change `xgboost_t`: each tree uses the shared deterministic
weighted-quantile histogram cut primitive, then repeatedly splits the current
leaf with the largest regularized second-order gain until `num_leaves` (or the
optional `max_depth`) is reached. `fit_regression` and `fit_binary` accept
positive sample weights and support squared regression and binary logistic
losses. `predict_margin`, `predict`, and binary `predict_proba` expose the
identity or sigmoid link, while `num_leaves`, `tree_node_count`, and
`tree_depth` expose the growth policy. `predict_staged_margin` and
`predict_staged` return cumulative margins or linked predictions after every
retained tree. `predict_contributions` returns the base margin followed by
learning-rate and tree-scale-scaled per-tree terms; summing its columns reproduces
`predict_margin`. `slice(n_trees,destination,status)` copies a fitted prefix
transactionally, including all allocatable node/row state, so a prefix can be
served without refitting.
`save_text(path,status)` and `load_text(path,status)` round-trip a versioned
`FORTML_LIGHTGBM_TEXT` snapshot containing model metadata and all live node
arrays. Loading validates record order, schema, finite scalar bounds, child
indices, and EOF before replacing the destination; truncated, unknown,
trailing, or malformed records return `FORTNUM_DOMAIN_ERROR` and leave it
unchanged.

`predict_shap(x,shap,status)` provides the same bounded per-feature raw-margin
contract as XGBoost. Column one is the path-dependent expected margin and
columns two through `n_features+1` are exact subset Shapley values, with
LightGBM child row counts supplying omitted-feature branch probabilities.
Rows sum to `predict_margin`; models wider than 12 features return
`FORTNUM_NOT_IMPLEMENTED`. `predict_shap_device` has explicit CPU dispatch and
a typed CUDA refusal with no hidden host fallback.

`fit_warm_start(x,y,status,options[,sample_weight])` continues a fitted model
to a strictly larger `options%n_estimators` target. The fitted prefix and all
new trees are built in temporary storage and committed only after success;
objective, leaf, histogram, depth, weight, shrinkage, and gain options must
match the prefix. This bounded continuation does not retain validation rows or
early-stopping state, so nonzero early-stopping controls return
`FORTNUM_NOT_IMPLEMENTED`; malformed targets, shapes, weights, or options
return `FORTNUM_DOMAIN_ERROR` without changing the source.

Set `options%boosting_type="goss"` to enable LightGBM's gradient-based
one-side sampling. `top_rate` retains the largest absolute-gradient rows and
`other_rate` selects a deterministic seed-ranked subset of the remainder;
selected small-gradient rows receive the exact `(1-top_rate)/other_rate`
gradient/Hessian correction before leaf gains and weights are evaluated.
`boosting_type()`, `top_rate()`, and `other_rate()` expose fitted policy
metadata. GOSS is available for both regression and binary logistic paths,
warm-start continuation, prefix slicing, and schema-3 text persistence. Rates
must be positive with `top_rate+other_rate<1`; invalid combinations refuse
transactionally. The CPU path is the only supported device and selected CUDA
prediction returns `FORTNUM_NOT_IMPLEMENTED`.

Set `options%boosting_type="dart"` to enable the bounded DART policy. Each
round uses a compiler-independent hash stream seeded by `options%seed` to
drop prior trees with probability `dart_drop_rate`, skips a round with
`dart_skip_drop`, and caps selected trees with `dart_max_drop` (`0` means no
cap). The selected prior trees and the new tree are multiplied by the explicit
tree-normalisation `1/(k+1)`; their scales are part of staged predictions,
contributions, prefix slices, warm starts, and schema-3 snapshots. The public
`dart_drop_rate()`, `dart_skip_drop()`, `dart_max_drop()`, and `tree_scale(i)`
accessors expose this state. DART fit/dropout selection is discrete and has no
fit-time hyperparameter derivative; fixed-tree input JVP/VJP products retain
the zero-away-from-splits contract. Invalid rates refuse transactionally.

`leaf_parameter_count()` and `leaf_parameters(status)` expose the continuous
fixed-structure coordinates `[base_score, leaf weights in estimator/node
order]`. `predict_leaf_jvp(x,parameter_dot,margin,margin_dot,status)` and
`predict_leaf_vjp(x,output_bar,parameter_bar,status)` differentiate raw margins
with respect to those coordinates while holding split routing and persisted
tree scales fixed. These products are defined on split surfaces; input
JVP/VJP products retain their boundary refusal. CUDA leaf products remain a
typed refusal until resident leaf-wise state and transfer accounting exist.

`predict_leaf_jvp_device` and `predict_leaf_vjp_device` dispatch the same
products on CPU and return `FORTNUM_NOT_IMPLEMENTED` for selected CUDA
contexts without a host fallback.

The finite numeric contract is explicit: NaN and infinity inputs are refused,
and categorical, missing-value-default, EFB, and distributed policies are not
silently approximated. Fixed-tree input JVP/VJP products are zero away from
learned split surfaces and return `FORTNUM_DOMAIN_ERROR` exactly on a split
boundary. CPU dispatch is supported; `predict_device` on a selected CUDA
device returns `FORTNUM_NOT_IMPLEMENTED` until resident leaf-wise histogram
state is available. Independent hand, tree-walk, DART, and persistence oracles
are `test_lightgbm`, `test_lightgbm_staged_slice`, `test_lightgbm_persistence`,
`test_lightgbm_goss`, and `test_lightgbm_dart`; the release benchmarks are
`lightgbm_leafwise.csv`, `lightgbm_goss.csv`, and `lightgbm_dart.csv` in
`../fortml-bench`.

### `fortml_lightgbm_multiclass`

`lightgbm_multiclass_t` is the classifier adapter for the LightGBM-style
leaf-wise binary path. `fit(x,labels,status,options[,sample_weight,
validation_x,validation_labels,validation_weight])` sorts arbitrary integer
labels, fits one binary child per class, and commits the complete ensemble only
after all children and validation checks succeed. Child early stopping is
disabled during the fit. The normalized multiclass weighted log loss selects a
common best prefix, preserving equal stage counts across classes. `classes`,
`class_count`, `feature_count`, `estimator_count`,
`requested_estimator_count`, `best_iteration`, `best_validation_loss`,
`early_stopped`, `boosting_type`, and `num_leaves` expose fitted metadata.

`predict_proba` returns a row-normalized `(n_samples,n_classes)` matrix and
`predict` returns the original sorted integer labels. `predict_proba_staged`
returns `(n_samples,n_classes,n_estimators)` normalized probabilities after
each common tree prefix. `decision_function` and
`decision_function_staged` return the corresponding unnormalized binary raw
margins.

`predict_proba_jvp` and `predict_proba_vjp` apply the sigmoid and normalization
chain rule while holding split routing fixed. The wrapped LightGBM child returns
a typed boundary error on a split surface. Away from a boundary the tree map is
locally constant, so the input products are zero. CPU device dispatch is
explicit. Selected CUDA probability and label requests return
`FORTNUM_NOT_IMPLEMENTED` until a resident LightGBM histogram kernel exists.
The independent oracle is `test_lightgbm_multiclass`; the release workload is
`lightgbm_multiclass.csv` in `../fortml-bench`.

### `fortml_xgboost_multioutput`

`xgboost_multioutput_t` is a transactional row-oriented adapter around one
`xgboost_t` regression child per target column.  `fit(x,targets,status,options,
sample_weight,validation_x,validation_targets,validation_weight)` routes
matching columns to the deterministic exact or histogram child policy and
commits the ensemble only after every child succeeds.  `predict` and
`predict_margin` return `(n_samples,n_outputs)` matrices.  `predict_staged_margin`
returns `(n_samples,n_estimators,n_outputs)` and requires matching retained
stage counts.  `feature_count`, `output_count`, `estimator_count`,
`parameter_count`, `leaf_parameter_count`, `leaf_parameters`, and `fitted`
expose shape and fixed-state metadata.

`predict_jvp`/`predict_vjp` route input products per output and sum reverse
products into the feature cotangent.  `predict_leaf_jvp`/`predict_leaf_vjp`
use the concatenated child vectors `[base_score, leaf weights]` while holding
tree topology fixed.  Malformed fit, prediction, stage, and product arguments
are transactional and return `FORTNUM_DOMAIN_ERROR`.  CPU device dispatch is
supported; selected CUDA calls return `FORTNUM_NOT_IMPLEMENTED` without a
host fallback.  See [`MULTIOUTPUT_BOOSTING.md`](MULTIOUTPUT_BOOSTING.md).

### `fortml_xgboost_multioutput` LightGBM companion

`lightgbm_multioutput_t` exposes the same public shape, transactional,
staged-margin, derivative, leaf-parameter, metadata, and device contracts,
but routes each target through `lightgbm_t`'s weighted-quantile best-first
leaf-wise growth.  Its children preserve LightGBM validation, GOSS, DART, and
serialization semantics; resident CUDA histogram execution remains an
explicit typed refusal.
