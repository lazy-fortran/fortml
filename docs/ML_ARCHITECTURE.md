# fortml machine-learning architecture

This document is the cross-repository design contract for the requested
regression, Gaussian-process, and neural-network surface. It turns the survey
in `.provenance/SURVEY.md` into interfaces that can be implemented and tested
incrementally.

## Ownership

| Concern | Owner | Public boundary |
|---|---|---|
| kinds, status, BLAS/LAPACK, factorizations, Krylov, FFT, tensor products, interpolation, RNG | `fortnum` | numerical procedures and status values |
| source transformation, JVP/VJP/HVP, sparse products, checkpointing | `fortad` | generated Fortran source and derivative products |
| symbolic identities, kernel/derivative derivations, code generation | `fortsym` | derivation records and generated source inputs |
| model topology, parameter packing, likelihoods, kernels, GP inference, MLP/VAE/RNN | `fortml` | model objects and value/product methods |
| objective callbacks, line searches, trust regions, optimizer state | `fortopt` | objective/gradient/JVP/VJP/HVP procedure interfaces |
| sparse storage and direct sparse backends | `fortsparse` | CSC/CSR construction, sparse products, direct solves |
| matched external baselines, oracles, scaling, profiler traces | `fortml-bench` | committed CSV, metadata, plots, and reports |

`fortml` may depend on the public interfaces of `fortnum`, `fortad`,
`fortsym`-generated artifacts, `fortopt`, and `fortsparse`. No model module
owns an optimizer loop, no optimizer reaches into a model's private state, and
no backend-specific GPU handle appears in a public statistical API.

## Common differentiable model contract

Every differentiable model uses a stable flat parameter vector and separates
three kinds of state:

1. immutable topology/configuration (layer sizes, kernel expression, observation
   layout);
2. trainable parameters `theta(:)` (weights, biases, kernel parameters,
   variational parameters); and
3. mutable execution/training state (optimizer moments, recurrent hidden
   state, running statistics, random-number state, and accelerator residency).

The minimum product surface is:

```text
value(theta, input, state) -> output, state/status
jvp(theta, input, theta_dot, input_dot, state) -> output, output_dot
vjp(theta, input, output_bar, state) -> theta_bar, input_bar
hvp(theta, input, direction, state) -> objective_gradient, gradient_dot
```

Fortran implementations may expose these as type-bound procedures rather than
literal callbacks, but the mathematical contract stays the same. `fortad`
generated products are used when their generated source passes the same oracle
as the explicit reference implementation. `fortsym` supplies derivations and
stable rewrites for kernels and small algebraic leaves; it is not a runtime
dependency.

The independent test hierarchy is mandatory: hand-derived scalar or matrix
formula first; convergence-tested central/complex-step finite differences for
JVPs; the dot-product identity for VJPs; Hessian symmetry or a differentiated
independent gradient for HVPs; and a trusted dense/high-precision solve for
probabilistic inference. Repository-state checks never substitute for these.

## Regression and basis maps

`linear_regression_t` remains the stable dense baseline: sample rows, feature
columns, output columns, SVD fitting, optional intercept, and ridge penalty.
The next map layer is a `basis_map_t` with the same value/JVP/VJP contract and
an explicit parameter layout. The initial basis families are:

- polynomial powers and tensor products with degree/ordering recorded;
- Fourier sine/cosine features with explicit angular frequencies;
- radial basis features with centers and positive log-scales;
- spline features backed by `fortnum` B-splines; and
- a bounded user-supplied callback path with explicit value, JVP, and VJP
  procedure interfaces.

The map returns a design matrix or its products without forcing a downstream
linear model to know which family produced it. A map's differentiable
parameters are explicit; knots and frequencies are configuration unless the
caller opts into their derivative contract.

Callback maps own a flat parameter vector and receive it in every product
procedure. The callback map exposes `static_lowering_eligible()`, which is
false for user procedures. This is the refusal boundary for OpenACC and native
CUDA execution. A callback remains a host reference until a future
`fortad`/`fortsym` lowering supplies an equivalent static kernel and oracle.

## MLP, VAE, and recurrent models

The existing `mlp_t` flat column-major layout is the reference neural-network
layer contract. Dense layers, activations, batched products, and explicit
backpropagation remain valid even when a `fortad`-generated product is added.
The MLP must be usable as a feature extractor by a GP and as encoder/decoder
components by a VAE.

The VAE is a composition of two MLPs and an explicit diagonal Gaussian
reparameterization:

```text
q(z|x) = Normal(mu_encoder(x), exp(log_std_encoder(x)))
z = mu + exp(log_std) * epsilon,  epsilon ~ Normal(0, I)
ELBO = E[log p(x|z)] - KL(q(z|x) || p(z))
```

The sample noise is supplied by `fortnum_rng`, making deterministic oracle
tests possible. The reconstruction likelihood is an explicit object (Gaussian
first; Bernoulli later), and the ELBO exposes value/JVP/VJP products without
burying sampling or optimizer state in the model.

The first recurrent model is a sequence-batched vanilla RNN followed by a
gated GRU and LSTM. Time is an explicit leading/trailing axis chosen once in
the API; hidden and cell states are explicit inputs/outputs. A scan has a
forward product and a reverse BPTT product, with checkpointing delegated to
`fortad`/`revolve` for long sequences. Tests include a hand-derived one-step
cell, a short-sequence finite-difference gradient, and a long-lag memory
convergence case.

## GP and derivative observations

The kernel expression tree remains composable: means, covariance kernels,
likelihoods, and sum/product composition are separate. The kernel catalog will
cover RBF, Matérn 1/2/3/2/5/2, rational quadratic, periodic/exp-sine-squared,
linear/dot-product, polynomial, constant, white-noise, spectral-mixture, and
compact-support Wendland families where the requested smoothness exists.

Each kernel declares its input smoothness. An observation operator maps a
function value, input derivative, or later mixed derivative to a row in the GP
observation vector. Covariance blocks are generated from the kernel partials:

```text
K_obs[i,j] = L_i^x L_j^{x'} k(x_i, x_j)
```

Derivative prediction uses the same operator on the cross-covariance and never
duplicates a solver. At coincident points, Matérn and white-noise behavior is
an explicit status/contract; undefined higher derivatives are refused.

Exact inference uses dense Cholesky for small problems. Large problems use the
same lazy operator boundary for tiled/KeOps-style products, tensor/Kronecker,
Toeplitz/FFT, compact sparse, and Krylov solves. Scalable extensions are
inducing-point variational GPs, stochastic Lanczos log determinants, LOVE
predictive roots, and structured interpolation. Multi-output uses a linear
model of coregionalization or a separable coregionalization matrix, with a
separate output-axis contract rather than duplicated scalar GP types.

## Optimizer boundary

`fortopt` defines procedure interfaces for objective value, gradient, JVP, VJP,
and HVP plus a convergence/status record. It implements Adam, L-BFGS-B,
Levenberg–Marquardt/Gauss–Newton, trust-region, and Newton–Krylov in that
order. Bound constraints, line searches, damping, stopping rules, and failure
messages are optimizer state; `fortml` only supplies callbacks and parameter
packing. The independent references are SciPy/Ceres behavior and hand-derived
updates/convergence cases, not copied source.

## Accelerator rule

Every GPU path has three layers: a backend-independent reference product, a
resident accelerator product, and a transfer-inclusive complete-workload
measurement. `nvfortran`/OpenACC is the first supported production backend;
native CUDA/C bindings are allowed for fixed hot kernels when they preserve the
Fortran API and pass the same oracle. Dynamic user callbacks stay on the host
until `fortad` can emit a safe static kernel. No performance claim is accepted
without compiler/device metadata, precision, memory, correctness error, and a
scaling record in `fortml-bench`.

The portable accelerator boundary is an opaque C ABI owned by the Fortran
operator: flat coordinates, right-hand sides, postfix kernel opcodes,
parameters, residency, and status codes. The CPU reference remains Fortran.
Fixed hot reductions use native CUDA C++ first, with HIP and SYCL adapters kept
possible behind the same ABI. These backends share the same independent oracle
and operation-level benchmark; they do not duplicate model semantics. The
forward reduction is not runtime-autodiffed: derivative products are separate
backend kernels generated or specialized from the same static plan.

## Implementation order

1. Basis maps and objective/gradient callback contracts.
2. `fortad`-generated MLP/kernel products beside explicit references.
3. Matérn derivative observation/prediction and expanded kernel catalog.
4. Persistent generic GP workspaces, block/Nystrom preconditioners, stochastic
   log determinant, LOVE, and multi-output/variational GP layers.
5. VAE reparameterization and RNN/GRU/LSTM scan/BPTT.
6. Optimizer completion and matched end-to-end benchmarks.

This order keeps the numerical core reusable and ensures every later model is
built from already tested value/JVP/VJP/HVP and optimizer contracts.
