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

| Type | Value or prediction | JVP | VJP or gradient | HVP |
| --- | --- | --- | --- | --- |
| `linear_regression_t` | `predict` | Free `linear_predict_jvp` | Free `linear_predict_vjp` | No |
| `logistic_regression_t` | Decision score and probabilities | Parameter/input JVP, probability JVP | Parameter/input VJP, probability VJP | No |
| `softmax_regression_t` | Multiclass decision scores and probabilities | Parameter/input JVP, probability JVP | Parameter/input VJP, probability VJP | No |
| `basis_map_t` | `evaluate` | Parameters and inputs | Parameters and inputs | No |
| `mlp_t` | `predict` | Parameters and inputs | Parameters and inputs | Weighted-output HVP |
| `bnn_t` | `elbo` | ELBO | ELBO | ELBO |
| `vae_t` | `elbo`, `reconstruct` | No | ELBO gradient | No |
| `rnn_t` | `forward`, squared-error `loss` | No | Loss gradient by BPTT | No |
| `kernel_t` | Scalar value and matrix | Parameter JVP | Parameter VJP | Parameter HVP |
| `gp_regression_t` | Mean, variance, LML | Prediction and LML parameters | Prediction and LML parameters | Mean and LML parameters |
| `gp_derivative_regression_t` | Mean, variance, and LML | Prediction and LML parameter JVP | Prediction parameter VJP and analytic LML hyperparameter gradient | Directional HVP (finite difference of the analytic gradient) |
| `gp_classification_t` | Latent and observed probabilities | Input JVP | No fitted-parameter product | No |
| `gp_multiclass_classification_t` | Normalized observed probabilities | Input JVP | No fitted-parameter product | No |
| `multi_output_gp_t` | Correlated mean and LML | No | No | No |
| Approximate GP types | Mean, variance, or ELBO as listed below | No | No | No |

`parameter_products_t` gives `mlp_t` and fitted `gp_regression_t` one common
packed value/JVP/VJP/HVP interface. Inputs remain fixed in that interface.

## Regression and basis maps

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
`classes()`, `feature_count()`, and `fitted()` expose the fitted state.
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

### `fortml_ovr_logistic_classifier`

`ovr_logistic_classifier_t` fits one binary `logistic_regression_t` per sorted
integer class. `fit(x,labels,status[,l2,fit_intercept,max_iterations,tolerance,
sample_weight,class_weight])` applies optional sample and sorted-class weights
to every one-vs-rest objective. `predict_proba` normalizes the positive binary
probabilities row-wise, `predict` uses a deterministic first-column tie rule,
and `decision_function` returns the unnormalized binary logits. `classes`,
`class_count`, `feature_count`, `parameter_count`, `parameters`,
`set_parameters`, and `fitted` expose the packed state.

`predict_proba_jvp` and `predict_proba_vjp` differentiate through the input
rows. `predict_proba_parameter_jvp` and `predict_proba_parameter_vjp` provide
the corresponding products for the packed binary-model parameters. The
quotient-rule normalization is included in every product, and finite,
unfitted, shape, and zero-support cases return status errors.

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

### `fortml_losses`

The loss facade provides stable matrix-valued `sigmoid_value`, `softmax_value`,
and `log_softmax_value` procedures with matching JVP and VJP products.
`binary_cross_entropy_with_logits_value` uses a mean reduction over all matrix
entries and accepts targets in `[0,1]`. `softmax_cross_entropy_value` uses one
one-based integer class label per row and a mean reduction over rows. Both loss
families expose JVP and VJP procedures, reject nonfinite inputs, and evaluate
the value with a shifted log-sum-exp or softplus expression. These routines are
the shared objective layer for neural, multiclass, GP, and boosting adapters.

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
`jvp`/`vjp` (also available as `decision_function_jvp`/`decision_function_vjp`)
differentiate logits with
respect to packed parameters and inputs, while `predict_proba_jvp`/
`predict_proba_vjp` compose the stable softmax products with those affine
products. All derivative paths validate finite tangents/cotangents and exact
shapes, and return a domain status for unfitted or malformed calls. HVPs and
parameter products for GP classifiers remain separate roadmap contracts.

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

### `fortml_preprocessing`

`standard_scaler_t%fit` stores column means and population standard deviations.
Zero-variance columns use unit scale. `transform`, `inverse_transform`, and
`transform_jvp` operate on row-oriented batches. `minmax_scaler_t%fit` stores
column extrema and maps to an increasing caller-selected range (default
`[0,1]`), with the same transform, inverse, and input-JVP operations. Fitted
statistics are state rather than differentiable parameters. The JVPs are with
respect to the input batch. Unfitted models, nonfinite values, and shape
mismatches are refused.

### `fortml_basis`

`basis_map_t` has these constructors and equivalent type-bound initializers:

| Constructor | Features | Active parameter vector |
| --- | --- | --- |
| `make_polynomial_basis(n_inputs,degree,status[,include_intercept])` | Separate powers 1 through `degree` for each input | Empty |
| `make_fourier_basis(n_inputs,frequencies,status[,include_intercept])` | Sine/cosine pair for each positive frequency and input | Log frequencies, column-major |
| `make_radial_basis(n_inputs,centers,scales,status[,include_intercept])` | One anisotropic Gaussian feature per center | Centers followed by log scales |
| `make_spline_basis(n_inputs,order,breakpoints,status[,include_intercept])` | B-spline basis functions for each input | Empty |
| `map%initialize_callback(...)` | Caller-defined | Caller-defined flat vector |

The public operations are `feature_count`, `parameter_count`, `parameters`,
`set_parameters`, `evaluate`, `jvp`, `vjp`, and
`static_lowering_eligible`. For `jvp`, supply both `theta_dot(:)` and
`x_dot(:,:)`. For `vjp`, the output cotangent has the evaluated feature shape.
The intercept column has no active parameter.

`BASIS_POLYNOMIAL`, `BASIS_FOURIER`, `BASIS_RADIAL`, `BASIS_SPLINE`, and
`BASIS_CALLBACK` are the public family codes used by extension and test code.

Callback initialization takes explicit value, JVP, and VJP procedures matching
`basis_value_callback`, `basis_jvp_callback`, and `basis_vjp_callback`. A
callback map returns false from `static_lowering_eligible` and stays on the
host. The `create_polynomial_impl`, `create_fourier_impl`,
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
same order. `jvp` and `vjp` return exact products over both stage parameters
and inputs. `stage_count`, `feature_count`, `parameter_count`, `valid`,
`is_fitted`, and `static_lowering_eligible` expose shape and backend
capabilities. Sequential transforms and DAG branches require separate input
and output contracts and are not inferred from this horizontal union.

`sequential_basis_pipeline_t` provides that explicit sequential contract.
Construct it with `make_sequential_basis_pipeline`, append a stage whose input
count equals the previous stage's feature count, and call `fit` before
`transform`. Its flattened parameters follow stage order. Forward JVPs and
reverse VJPs propagate through every stage, including the input cotangent.
Shape mismatches, empty chains, and unfitted transforms return status errors.

`column_basis_pipeline_t` provides the column-wise variant. Construct it with
`make_column_basis_pipeline(n_inputs,status)`, then append a basis map and a
one-based integer column list. A stage's input count must equal the list length.
Indices must be in range and unique within that stage, while different stages
may reuse columns. The transform gathers only the selected columns and
concatenates stage feature blocks. Parameter packing follows stage order, JVPs
gather input tangents, and VJPs scatter-add stage cotangents into the original
input columns. This is a deterministic feature union, not a DAG scheduler or a
parallel device executor. Those capabilities remain separate roadmap items.

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

### `fortml_validation`

`kfold_splitter_t` and `stratified_kfold_splitter_t` are index-only, seeded
cross-validation iterators. Initialize with `n_samples` (or integer labels)
and `n_splits`, then call `next_split(train_indices,test_indices,has_split,
status)` until `has_split` is false. `reset()` replays the same sequence.
Shuffled iterators use a local positive seed and never touch process-global RNG
state. Test folds are balanced. Stratified folds distribute each class
round-robin. Invalid fold counts, seeds, and pre-initialization calls return
status errors. The splitters do not fit or store transformers, so callers must
fit preprocessing on each training index set explicitly.

## Neural models and variational inference

### `fortml_mlp`

`mlp_t%initialize(layer_sizes,status[,hidden_activation,output_activation,
initialization_seed])` constructs dense layers. `MLP_LINEAR`, `MLP_TANH`, and
`MLP_RELU` are accepted activation constants. The default hidden activation is
`tanh`, and the default output activation is linear. Weights use deterministic
Xavier scaling (or He scaling for ReLU hidden layers), with a reproducible
phase sequence controlled by `initialization_seed` (default `17`).

The parameter vector stores each layer's weight array
`(input_width,output_width)` in column-major order followed by its bias, from
the input layer to the output layer. Use `parameter_count`, `parameters`, and
`set_parameters` to manage it.

`predict(x,y,status)` evaluates a batch. The product signatures are:

```text
jvp(x, dtheta, dx, y, dy, status)
vjp(x, output_bar, theta_bar, x_bar, status)
hvp(x, output_bar, dtheta, dx, theta_hvp, x_hvp, status)
```

`backprop` is an alias for `vjp`. ReLU uses derivative zero at the kink.

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
The implementation refuses nonfinite states, malformed layer shapes, and
nonfinite directions. General nonseparable Hamiltonians, learned Poisson
structures, and implicit integrators remain separate research contracts.

### `fortml_mlp_training`

`mlp_train(model,x,target,status,options,state[,validation_x,validation_target])`
trains an existing `mlp_t` with deterministic Adam. A zero `batch_size`
selects full-batch updates.
Mini-batch shuffling uses an explicit Park-Miller stream controlled by
`shuffle_seed`, and does not mutate process-global random state. The options
also provide learning-rate and Adam coefficients, L2 regularization,
gradient tolerance, patience, best-state restoration, and an epoch callback.

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
fixtures. Optimizer-trajectory, learning-rate, and Adam-beta hypergradients
remain separate contracts.

The same loss entry point accepts optional `sample_weight`, `reduction`, and
`diagnostics` arguments. `MLP_REDUCTION_MEAN` divides the weighted data loss
and gradient by positive weight mass, while `MLP_REDUCTION_SUM` leaves the
weighted data sum unnormalized. Weights must be finite, non-negative, and have
positive mass. L2 remains a single parameter regularizer in either reduction.
`mlp_loss_diagnostics_t` reports `data_loss`, `regularization_loss`,
`weight_mass`, and `sample_count`, so callers can log named scalar components
without reconstructing the reduction.

`mlp_training_objective_t` packages the same objective for FortOpt. Call
`initialize(model,x,target,l2,status[,optimize_l2])`, then use `parameters`,
`parameter_count`, `value_gradient`, and `hvp`. With `optimize_l2=.true.`, the
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
optimizer path. Mini-batch Adam remains the appropriate stochastic trainer.
The optimizer consumes the analytic value/gradient path above, so no
finite-difference hyperparameter approximation is introduced.

`mlp_batch_iterator_t` is the reusable deterministic row-index cursor used by
`mlp_train`. Initialize it with `n_samples`, an optional `batch_size`,
`shuffle`, and positive `seed`. Call `reset` once per epoch and
`next_batch(indices,has_batch,status)` until `has_batch` is false. The final
batch is returned without padding, and the cursor never advances implicitly to
the next epoch. Its explicit position, epoch, and copied RNG state make an
in-memory batch boundary resumable. `mlp_learning_rate_schedule_proc` can be
installed in `mlp_training_options_t%learning_rate_schedule`. It receives the
epoch, one-based update number, and base rate and must return a finite positive
rate. `gradient_clip_norm` applies global norm clipping before each Adam step.
Zero disables clipping. `accumulation_steps` combines that many consecutive
microbatches into one sample-weighted mean gradient before an Adam step. The
last uneven group is flushed at the epoch boundary. L2 is added once per
optimizer update, clipping is applied after accumulation, and the state records
`microbatches`, `updates`, and the configured accumulation count. This gives an
exact full-batch equivalence oracle when the model and options are otherwise
identical, while reducing optimizer-state updates for memory-constrained
training.

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
optional sample-weight vector uses the same positive-mass reduction. Binary,
multilabel, ordinal, and GP likelihood classifier adapters remain roadmap work.

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
sample_weight,criterion])` builds a deterministic numeric classification tree.
`criterion` is `CART_CRITERION_GINI` (the default) or
`CART_CRITERION_ENTROPY`. Each node searches weighted impurity over ascending
feature and threshold order, accepts only strict improvements, and stores
weighted class frequencies. `predict_proba` returns the leaf probabilities and
`predict` maps their first maximum back to sorted integer `classes`. The dense
fit and query paths reject nonfinite values. Positive finite sample weights,
depth up to 12, and count-based `min_samples_leaf` are supported. Missing-value
routing, input derivatives, histogram growth, and differentiable split
selection remain unsupported.

### `fortml_xgboost`

`xgboost_t` is a deterministic exact-split second-order boosting estimator. Use
`fit_regression` for a squared objective or `fit_binary` for a logistic
objective. `xgboost_options_t` controls estimator count, learning rate,
minimum leaf size, L1/L2 leaf regularization, split gamma, and minimum child
Hessian. Candidate splits aggregate exact gradients and Hessians and use the
regularized gain. `predict_margin`, `predict`, `predict_proba`,
`decision_function`, `split_gain`, `leaf_weights`, `tree_node_count`, and
`tree_depth` expose diagnostics.
`predict_jvp` is zero away from learned split boundaries and returns a
structured refusal at a discontinuity. `max_depth` grows each exact tree
recursively, with deterministic feature/threshold tie ordering and
regularized Newton leaves at every node. Histogram quantile approximation,
missing-value routing, categorical features, ranking, and constraints are
deliberately refused until their independent contracts land.

`xgboost_multiclass_t` wraps the binary logistic estimator in a deterministic
one-vs-rest classifier. `fit(x,labels,status[,options])` sorts arbitrary
integer labels, fits one depth-limited booster per class, and normalizes the
positive OVR probabilities. `classes`, `class_count`, `feature_count`,
`estimator_count`, `fitted`, `decision_function`, `predict`, and
`predict_proba` expose the fitted state. `predict_proba_jvp` applies the exact
quotient-rule JVP away from learned split boundaries and propagates the binary
boundary refusal. The classifier inherits the binary estimator's dense,
finite-input scope.

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
make_matern12_kernel(input_dim, variance, lengthscale, status)
make_matern32_kernel(input_dim, variance, lengthscale, status)
make_matern52_kernel(input_dim, variance, lengthscale, status)
make_linear_kernel(input_dim, variance, status)
make_constant_kernel(input_dim, variance, status)
make_white_noise_kernel(input_dim, variance, status)
make_user_kernel(input_dim, variance, formula, status)
```

The corresponding kind constants are `KERNEL_RBF`, `KERNEL_MATERN12`,
`KERNEL_MATERN32`, `KERNEL_MATERN52`, `KERNEL_LINEAR`, `KERNEL_CONSTANT`,
`KERNEL_WHITE_NOISE`, `KERNEL_SUM`, `KERNEL_PRODUCT`, and `KERNEL_USER`.
Combine initialized kernels with `kernel_add(left,right,status)` or
`kernel_multiply(left,right,status)`.
`clone_kernel(kernel)` makes an independent copy of the complete expression
tree, including composite children. Use it for temporary optimizer or
derivative probes instead of intrinsic assignment, which aliases pointer
children.

Leaf parameters are stored as logarithms. Radial leaves have
`[log_variance,log_lengthscale]`. Linear, constant, white-noise, and user leaves
have `[log_variance]`. Composite vectors concatenate the complete left vector
and then the complete right vector.

`kernel_t` exposes `parameter_count`, `parameters`, `set_parameters`, `value`,
`matrix`, `matrix_jvp`, `parameter_vjp`, `parameter_hvp`, and
`input_derivatives`. Input derivatives return the value, gradients with respect
to both arguments, and the mixed Hessian. Matérn 1/2 input derivatives are
undefined at coincident points. White-noise derivative observations are
rejected, and user formulas have no input-derivative rule. The free
`kernel_input_derivatives` procedure has the same arguments as the type-bound
method with the kernel supplied first.

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

`gp_regression_t%fit(x,y,kernel,noise_variance,status[,jitter])` fits one or
more output columns with a shared kernel and independent output values. The
packed model parameter vector is the recursive kernel vector followed by
`log_noise_variance`. `set_parameters` refactorizes the fitted covariance.

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

### `fortml_gp_training`

`gp_optimize_hyperparameters(model,options,result,status)` minimizes the
negative exact-GP log marginal likelihood with FortOpt L-BFGS-B. The model
must already be fitted, and each objective evaluation refactorizes the model
through `set_parameters`, so the optimizer uses the same analytic
hyperparameter gradient as the public likelihood product. Parameters are the
kernel log parameters followed by log observation-noise variance. Bounds are
applied uniformly through `gp_hyperparameter_options_t`. The default interval
is `[-20,20]`.

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

`gp_classification_t%fit(x,labels,kernel,status[,options,state])` fits a binary
Laplace GP classifier. Labels are arbitrary integers and are retained in
ascending order. `GP_LIKELIHOOD_LOGISTIC` uses a MacKay logistic predictive
integral. `GP_LIKELIHOOD_PROBIT` uses the analytic probit predictive map.
`gp_classification_options_t` controls Newton iterations, damping, tolerance,
and jitter. `predict_latent` returns posterior latent mean and variance.
`predict_proba` returns two observed-probability columns. Both have input-JVP
variants, and `predict` returns the stored integer labels. Every kernel that
supplies the existing matrix and input-derivative contracts is supported.
`parameter_count()` and `parameters()` expose read-only kernel-log-parameter
metadata in the fitted model. `hyperparameter_gradient` validates its output
shape and then returns a domain-status refusal: the Laplace mode, likelihood
curvature, and posterior factorization are fit-time state and are not yet
recomputed by a classifier parameter product. Variational likelihoods,
multiclass coupling, kernel hyperparameter products, and derivative observations
remain explicit roadmap work.

`gp_multiclass_classification_t` provides deterministic one-vs-rest multiclass
GP classification over the same binary Laplace models. It fits one model per
sorted integer class, normalizes their positive probabilities onto a simplex,
and exposes `classes`, `class_count`, `feature_count`, `predict_proba`,
`predict`, and `fitted`. `parameter_count()` and `parameters()` concatenate
the read-only kernel metadata for each one-vs-rest model in sorted class order.
`hyperparameter_gradient` returns an explicit domain-status refusal until the
coupled Laplace objective and all posterior refactorizations are differentiable.
The wrapper inherits the selected logistic or probit likelihood and
kernel/refusal behavior. It is a coupling policy rather than a multinomial
likelihood, so variational categorical likelihoods and shared multiclass
hyperparameter training remain separate work.

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
The query products use a deterministic central finite difference of the
existing covariance contract because derivative-observation queries can need
third-order mixed input derivatives. An analytic third-order kernel contract
remains open. Joint posterior covariance remains open.
`log_marginal_likelihood`, `log_marginal_likelihood_jvp`,
`hyperparameter_gradient`, and `hyperparameter_hvp` provide likelihood
products. The gradient uses analytic parameter tangents of the supported RBF,
Matérn 1/2, 3/2, 5/2, linear, constant, and sum/product kernels. Matérn 1/2
still refuses coincident derivative observations. The HVP is a deterministic
directional finite difference of that analytic gradient. It is intentionally
listed as such until a generated second-order derivative product is added.
Validated user-formula leaves currently refuse derivative hyperparameter
products rather than silently approximating their input partials.

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
posterior mean but no posterior variance or parameter products.

## Approximate Gaussian processes

### `fortml_sparse_gp`

`sparse_gp_t` is a scalar-output inducing-point variational GP with Gaussian
likelihood. Initialize it with inducing points, a kernel, and a positive noise
variance. `set_variational(mean,factor,status)` takes the mean and the lower
Cholesky factor of the inducing covariance. Its diagonal must be positive.

`elbo(x,y,value,status[,expected_log_likelihood,kl_value])` evaluates the
closed-form bound. `predict(x_star,mean,variance,status)` returns variational
latent marginals. The type does not optimize or pack its inducing and
variational parameters. The caller owns that update loop.

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
`parameter_block_from_kernel`, and `parameter_block_from_gp` provide typed
adapters. A block also exposes `name`, `size`, `get`, `set`, and `initialized`.

`parameter_registry_t` exposes `clear`, `add`, `block_count`,
`parameter_count`, `pack`, `unpack`, and `range`. Names must be nonempty and <!-- slop-ok -->
unique. Blocks retain insertion order. The public callback contracts are
`parameter_get_proc` and `parameter_set_proc`.

### `fortml_parameter_products`

`parameter_products_from_mlp(products,name,model,status)` and
`parameter_products_from_gp(products,name,model,status)` bind a live target
model. A GP must already be fitted. The resulting `parameter_products_t`
exposes `initialized`, `parameter_count`, `pack`, `unpack`, `range`, `value`, <!-- slop-ok -->
`jvp`, `vjp`, `hvp`, and `has_hvp`.

The MLP adapter evaluates `predict`. Its JVP/VJP/HVP vary packed model
parameters while holding inputs fixed. The GP adapter evaluates predictive
mean and packs kernel parameters followed by log noise variance. Both current
adapters return true from `has_hvp`. The callback contracts
`parameter_value_proc`, `parameter_jvp_proc`, `parameter_vjp_proc`, and
`parameter_hvp_proc` define the extension seam.
