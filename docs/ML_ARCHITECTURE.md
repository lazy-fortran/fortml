# fortml machine-learning architecture

This document is the cross-repository design contract for the regression,
Gaussian-process, and neural-network surface. Current implementation details
are identified below. The final section lists the remaining design targets.

## Ownership

| Concern | Owner | Public boundary |
|---|---|---|
| kinds, status, BLAS/LAPACK, factorizations, Krylov, FFT, tensor products, interpolation, RNG | `fortnum` | numerical procedures and status values |
| source transformation, JVP/VJP/HVP, sparse products, checkpointing | `fortad` | generated Fortran source and derivative products |
| symbolic identities, kernel/derivative derivations, code generation | `fortsym` | derivation records and generated source inputs |
| model topology, parameter packing, likelihoods, kernels, GP inference, MLP/VAE/RNN | `fortml` | model objects and value/product methods |
| objective callbacks, line searches, trust regions, optimizer state | `fortopt` | objective/gradient/JVP/VJP/HVP procedure interfaces |
| sparse storage and direct sparse backends | `fortsparse` | CSC/CSR construction, sparse products, direct solves |
| matched external baselines, oracles, scaling, profiler traces | `fortml-bench` | recorded CSV, metadata, plots, and reports |

`fortml` may depend on the public interfaces of `fortnum`, `fortad`,
`fortsym`-generated artifacts, `fortopt`, and `fortsparse`. No model module
owns an optimizer loop, no optimizer reaches into a model's private state, and
no backend-specific GPU handle appears in a public statistical API.

## Common differentiable model contract

Models that implement the optimizer-facing product contract use a stable flat
parameter vector and separate three kinds of state:

1. immutable topology/configuration (layer sizes, kernel expression, observation
   layout).
2. trainable parameters `theta(:)` (weights, biases, kernel parameters,
   variational parameters).
3. mutable execution/training state (optimizer moments, recurrent hidden
   state, running statistics, random-number state, and accelerator residency).

The target product surface is:

```text
value(theta, input, state) -> output, state/status
jvp(theta, input, theta_dot, input_dot, state) -> output, output_dot
vjp(theta, input, output_bar, state) -> theta_bar, input_bar
hvp(theta, input, direction, state) -> objective_gradient, gradient_dot
```

Fortran implementations may expose these as type-bound procedures rather than
literal callbacks. Current model families expose only the products named in
their public API. `fortad` generated products are used when their generated
source passes the same oracle as the explicit reference implementation.
`fortsym` supplies derivations and stable rewrites for kernels and small
algebraic leaves. It is not a runtime dependency.

The independent test hierarchy starts with a hand-derived scalar or matrix
formula. JVPs use convergence-tested central or complex-step finite
differences. VJPs use the dot-product identity. HVPs use Hessian symmetry or a
differentiated independent gradient. Probabilistic inference uses a trusted
dense or high-precision solve. Repository-state checks never substitute for
these.

## Composition, training, and backend boundaries

The production path has one ownership chain:

```text
data view -> fitted transform/basis graph -> estimator parameter registry
          -> objective and derivative products -> FortOpt trainer/search
          -> checkpoint and device plan -> benchmark oracle
```

Each arrow carries an explicit shape, parameter offset, reduction, status, and
residency contract. A basis or preprocessing graph owns child registries and
maps their offsets into the estimator registry. A trainer owns batches,
optimizer state, schedules, validation, callbacks, and checkpoints. The model
owns topology and trainable values. No layer reaches into another layer's
private arrays.

The derivative capability record for every public product states the mode
(`analytic`, `fortsym`, `fortad`, `finite_difference`, or `refused`), the
parameter/input blocks it covers, the smoothness boundary, and the independent
oracle. FortOpt consumes the same registry and derivative callback used by
hyperparameter search, so an L-BFGS-B run cannot optimize a parameter that a
model silently omitted from its gradient.

Device execution has three separately measured layers:

1. The Fortran CPU reference, which supplies the behavioral oracle.
2. A resident OpenACC or native-CUDA plan, which owns model, optimizer, batch,
   and workspace allocations.
3. A transfer-inclusive workload measurement with compiler, device, precision,
   transfer counters, and peak memory.

An operation with no resident plan reports a typed device refusal. The current
elastic-net, OVO, binary/OVR-multiclass Laplace-GP, GP-likelihood,
probability-calibration, XGBoost, and typed-schedule APIs follow
this rule. Fixed no-autodiff reductions may use CUDA kernels when FortSym or a
hand-derived oracle proves the same semantics. Autodiff-bearing paths retain
FortAD/FortSym reference products until their device graph is complete.

## Regression and basis maps

`linear_regression_t` is the stable dense baseline: sample rows, feature
columns, output columns, SVD fitting, optional intercept, and ridge penalty.
`ridge_regression_t` adds weighted SVD fitting with an explicit packed state;
`elastic_net_regression_t` adds weighted multi-output lasso/elastic-net fitting
through deterministic coordinate descent. Both estimators expose the same
fixed-state coefficient/input JVP/VJP contract, so a basis pipeline can feed
either estimator without changing parameter routing. Their nonsmooth fit
solvers, active-set decisions, and regularization hyperparameters remain
declared derivative boundaries.
`basis_map_t` has a value/JVP/VJP contract and an explicit parameter layout.
The implemented basis families are:

- separate polynomial powers for each input with degree/ordering recorded;
  `make_polynomial_interaction_basis` additionally enumerates all nonconstant
  total-degree monomials in deterministic graded order.
- Fourier sine/cosine features with explicit angular frequencies.
- fixed-state random Fourier features with explicit frequency vectors and
  phases (`sqrt(2/m) cos(w dot x + b)`).
- radial basis features with centers and positive log-scales.
- spline features backed by `fortnum` B-splines.
- a user-supplied callback path with explicit value, JVP, and VJP
  procedure interfaces.

The map returns a design matrix or its products without forcing a downstream
linear model to know which family produced it. A map's differentiable
parameters are explicit. Fourier frequencies are positive log parameters.
Random Fourier frequencies and phases are fixed transform state with an
explicit zero-parameter derivative block.
Radial centers and log scales are parameters. Polynomial and spline maps have
no active parameter vector. Spline breakpoints are configuration.

Callback maps own a flat parameter vector and receive it in every product
procedure. The callback map exposes `static_lowering_eligible()`, which is
false for user procedures. This is the refusal boundary for OpenACC and native
CUDA execution. A callback remains a host reference until a future
`fortad`/`fortsym` lowering supplies an equivalent static kernel and oracle.

`basis_pipeline_t` concatenates maps that all consume the complete input matrix,
while `sequential_basis_pipeline_t` feeds one map's feature block to the next.
`column_basis_pipeline_t` is the explicit feature-selection variant: each stage
stores a validated one-based column list, gathers that submatrix, and scatters
its reverse cotangent into the original columns. Stage parameters remain in a
single deterministic flat vector. Optional unique stage names expose stable
feature and parameter names plus one-based stage offsets, so downstream
estimators can route packed coefficients without private knowledge of stage
order; column pipelines also expose defensive copies of their input
selections. This fixed union is differentiable and auditable, but it is not yet
a general DAG or device-parallel execution graph.

## MLP, VAE, and recurrent models

The existing `mlp_t` flat column-major layout is the reference neural-network
layer contract. Dense layers, activations, batched products, and explicit
backpropagation remain the reference beside the `fortad`-generated scalar
fixture. The VAE uses the same MLP type for its encoder and decoder.

The MLP parameter vector has a named structural view. `parameter_layout()`
returns stable `layer_1.weight`, `layer_1.bias`, and subsequent layer paths
with one-based ranges, matrix shapes, and trainable or buffer roles.
`parameter_range(path,...)` resolves one block without exposing private layer
storage. Grouped objectives and external trainers can therefore select or
freeze a block by name while retaining the single packed registry used by
FortOpt. The current dense MLP has trainable weight and bias blocks only.
Buffers, aliases, tied parameters, and stateful module modes remain explicit
extensions of this tree contract.

`mlp_training` separates deterministic data traversal from optimizer updates.
`mlp_batch_iterator_t` owns the one-based permutation, epoch boundary, and
Park--Miller state. It returns an unpadded final batch and never silently starts
another epoch. `mlp_train` consumes that cursor, applies an optional per-update
learning-rate callback and global norm clipping, and records the effective rate
and clipping count in `mlp_training_state_t`. `accumulation_steps` makes the
optimizer boundary explicit: microbatch data gradients are weighted by sample
count, averaged, regularized once, then clipped and stepped. The final uneven
group is flushed rather than dropped, and the state exposes both microbatch and
optimizer-update counts. Optional held-out arrays add a validation stream:
validation is evaluated at a configurable epoch interval, patience and
best-state restoration monitor it, and the state records the validation
history and best/final values. The `fortml_mlp_checkpoint` module provides the
file boundary with a compiler-independent, versioned formatted-text schema.
`mlp_checkpoint_save` and `mlp_checkpoint_load` validate all scalar metadata
and array lengths, refuse unknown versions, truncation, extra records, and
invalid optimizer state, and load into a temporary value before replacing the
destination. Real values are emitted with 17 significant decimal digits,
while procedure pointers remain caller-owned. Serialized
`mlp_training_checkpoint_t` now makes the in-memory boundary explicit: it
snapshots packed parameters, Adam moments and step, iterator permutation/RNG,
active microbatch accumulation, schedule position/history, validation
counters, and best-state metadata. Adam/AdamW moments, AMSGrad moments and its
max-second-moment envelope, Adagrad accumulated squares, RMSprop
running-square/centered-mean/momentum state, or SGD velocity are stored in the
same explicit checkpoint blocks. A
resumed call validates the training
contract before restoring the snapshot. Procedure pointers remain caller-owned
and checkpoints whose best-state restoration changed parameters are marked
non-resumable. Distributed checkpoint coordination remains a separate
contract. The trainer also exposes a typed event stream through
`mlp_training_event_proc`. `TRAIN_BEGIN`, `UPDATE`, `VALIDATION`,
`EPOCH_END`, `CHECKPOINT`, and `TRAIN_END` events carry epoch/update
counters, objective metrics, gradient norm, effective learning rate, and a
stop flag. A callback may request deterministic early stopping or return a
`fortnum_status_t` failure; failures abort the call without being converted
into a successful checkpoint. Event procedure pointers are caller-owned and
are intentionally not serialized. The MSE objective has an explicit
reduction boundary: optional finite non-negative sample weights use positive
weight mass for the mean reduction, while the sum reduction remains
unnormalized. Named diagnostics expose data and regularization components and
the effective weight mass. This keeps reduction semantics visible to an outer
optimizer instead of hiding them in a trainer callback.

## Tree boosting lifecycle

`xgboost_t` keeps split topology, leaf values, objective metadata, and
validation state separate. Exact and weighted-histogram CPU growth share the
same objective derivative contract. A validation set is optional. When
`early_stopping_rounds` is positive, the fit loop evaluates the selected
objective after each round, retains the lowest finite weighted loss, and either
trims the ensemble to that round or preserves all completed rounds according
to `restore_best`. The fitted object reports `best_iteration`,
`best_validation_loss`, and `early_stopped`. Validation data, weights, NaN
routing, and objective domains are validated before growth. This lifecycle is
deterministic and has an independent stage-by-stage oracle. Warm-start
continuation, serialized tree state, categorical and interaction constraints,
and resident GPU histograms remain separate contracts.

The VAE is a composition of two MLPs and an explicit diagonal Gaussian
reparameterization:

```text
q(z|x) = Normal(mu_encoder(x), exp(log_std_encoder(x)))
z = mu + exp(log_std) * epsilon,  epsilon ~ Normal(0, I)
ELBO = E[log p(x|z)] - KL(q(z|x) || p(z))
```

The sample noise is supplied by `fortnum_rng`, making deterministic oracle
tests possible. The implemented reconstruction likelihood is Gaussian with one
fixed variance. The VAE exposes the ELBO, its packed gradient, and a seeded
reconstruction. JVP and HVP methods are not exposed.

The implemented recurrent model is a sequence-batched vanilla `tanh` RNN.
Time is the leading axis, followed by batch and feature axes. Every scan starts
from zero and returns the hidden history. The squared-error loss has an explicit
reverse BPTT gradient. GRU and LSTM cells, caller-supplied initial state,
stacked recurrent layers, checkpointing, and long-lag tests remain design
targets.

Binary probability calibration is a post-estimator composition boundary.  The
`probability_calibrator_t` stores a smooth positive-temperature map, a smooth
two-parameter Platt map, or a weighted pool-adjacent-violators map over scalar
decision scores.  The smooth maps have analytic score and parameter products
and can therefore be included in a FortOpt objective; the latter linearly interpolates between fitted knots and
returns a typed refusal at active-set boundaries rather than differentiating
through a changing PAVA partition.  All maps preserve arbitrary integer
class order and `[1-p,p]` probability columns.  CPU prediction is complete;
CUDA remains an explicit capability refusal until calibration state and
interpolation are resident in a native kernel.

Weighted LDA and QDA follow the same explicit-state boundary. Their fitted
means, Cholesky factors, precisions, priors, and sorted integer labels are
resident host state; Gaussian log-probability JVP/VJP products differentiate
the continuous normalization while argmax labels remain discrete. The packed
parameter seam is stable for future optimizer adapters. CUDA prediction is a
typed refusal until a resident discriminant kernel owns the factors and class
metadata, so no OpenACC data movement is mistaken for GPU residency.

## GP and derivative observations

Binary GP classification shares one signed-margin likelihood layer across
Laplace, future variational, and minibatch objectives. The public
`gp_classification_log_likelihood_value/jvp/vjp` products implement the
logistic and probit Bernoulli likelihoods analytically, with stable negative
probit-tail evaluation and explicit finite-input/refusal checks. The
primitive is backend independent and therefore suitable for later device
lowering, but the current release only claims host execution for this scalar
likelihood layer; a CUDA GP classifier must still provide resident covariance,
mode-solve, and derivative buffers before it can be reported as end-to-end
GPU.

The current kernel expression tree composes covariance leaves with sum and
product nodes. Its built-in leaves are RBF, Matérn 1/2, 3/2, and 5/2, periodic,
rational quadratic, cosine, polynomial, linear, constant, and white noise.
Validated postfix formulas provide a bounded user leaf. Spectral-mixture and
direct Wendland constructors remain design targets.

The derivative GP assigns an integer component to each observation row:
component 0 is a function value and component `j` is the first derivative with
respect to input `j`. Covariance entries are generated from the kernel
partials:

```text
K_obs[i,j] = L_i^x L_j^{x'} k(x_i, x_j)
```

Derivative prediction uses the same operator on the cross-covariance and never
duplicates a solver. At coincident points, Matérn and white-noise behavior is
an explicit status contract. Undefined higher derivatives are refused.

The derivative GP packs kernel log parameters followed by log observation-noise
variance. Its likelihood gradient contracts analytic parameter tangents of the
supported radial and composed kernels with the usual Cholesky trace product.
The FortOpt adapter consumes this gradient directly. The public directional
HVP currently uses a deterministic finite difference of that analytic gradient,
so it is not presented as a generated second-order kernel product. `fortsym`
generated products remain the preferred route when a smaller proven expression
is available, with FortAD and an independent dense oracle retained as checks.

The ordinary RBF kernel now uses a FortSym-generated natural-parameter leaf for
its value and first derivatives with respect to variance, squared distance, and
lengthscale. The leaf is lowered once and shared by the kernel parameter JVP
and VJP paths; its generated header records FortSym `f71a1aa`, 15 IR nodes, and
7 compound operations. The independent scalar/finite-difference test is
`test_fortsym_rbf_leaf`. Matérn 1/2 HVPs now use a second generated FortSym
leaf (`9482261`, 37 IR nodes, 28 compound operations), checked by
`test_fortsym_matern12`. Matérn 3/2 now also uses a FortSym-generated HVP
leaf (`b72a23a`, 60 IR nodes, 48 compound operations), checked by
`test_fortsym_matern32`; Matérn 5/2 retains the FortAD product until a
symbolic operation-count comparison accepts a smaller replacement.

Exact inference uses dense Cholesky for small problems. Large problems use the
same lazy operator boundary for tiled products, tensor/Kronecker,
Toeplitz/FFT, compact sparse, and Krylov solves. Implemented approximations
include inducing-point variational and prior approximations, local experts,
stochastic Lanczos log determinants, single-point LOVE-style predictive
variance, and tensor-grid structured interpolation. Multidimensional SKI uses
an equal extent per axis and one isotropic RBF leaf. Correlated outputs use an
intrinsic coregionalization type with an explicit output axis.

## Optimizer boundary

`fortopt` defines procedure interfaces for objective value, gradient, JVP, VJP,
and HVP plus a convergence/status record. It implements Adam, AdamW, Adagrad,
RMSprop,
L-BFGS-B,
Levenberg-Marquardt/Gauss-Newton, trust-region, and Newton-Krylov in that
order. Bound constraints, line searches, damping, stopping rules, and failure
messages are optimizer state. `fortml` only supplies callbacks and parameter
packing. The independent references are SciPy/Ceres behavior and hand-derived
updates/convergence cases, not copied source.

## Accelerator rule

Every GPU path has three layers: a backend-independent reference product, a
resident accelerator product, and a transfer-inclusive complete-workload
measurement. `nvfortran`/OpenACC is the first supported production backend.
Native CUDA/C bindings are allowed for fixed hot kernels when they preserve the
Fortran API and pass the same oracle. Dynamic user callbacks stay on the host
until `fortad` can emit a safe static kernel. No performance claim is accepted
without compiler/device metadata, precision, memory, correctness error, and a
scaling record in `fortml-bench`.

The portable accelerator boundary is an opaque C ABI owned by the Fortran
operator: flat coordinates, right-hand sides, postfix kernel opcodes,
parameters, residency, and status codes. The CPU reference remains Fortran.
Fixed hot reductions use native CUDA C++ first, with HIP and SYCL adapters kept
possible behind the same ABI. These backends share the same independent oracle
and operation-level benchmark. They do not duplicate model semantics. The
forward reduction is not runtime-autodiffed: derivative products are separate
backend kernels generated or specialized from the same static plan.

## Remaining design targets

1. Add VAE JVP/HVP products and recurrent JVP/HVP products only with independent
   whole-vector oracles.
2. Add caller-supplied recurrent state, stacked scans, GRU cells, and LSTM
   cells before adding checkpointed long-sequence execution.
3. Extend the built-in kernel catalog where a direct constructor and its input
   smoothness contract are preferable to a user formula.
4. Batch or cache LOVE-style predictive variance and extend multidimensional
   SKI beyond one isotropic RBF leaf.
5. Add parameter products for derivative and approximate GP inference after
   their differentiated solve paths have independent checks.
