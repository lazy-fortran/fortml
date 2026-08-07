# fortml design

The first interface is a matrix-oriented regression API. Samples occupy rows,
features occupy columns, and outputs occupy columns. The linear model stores its
parameters as a `(features + intercept, outputs)` array. This layout matches
Fortran `matmul`, supports multiple outputs without a separate model type, and
maps directly to a flat active-vector layout for `fortnum` and `fortad`.

## Repository boundaries

`fortnum` supplies dense solves, factorizations, interpolation, random numbers,
and numerical status types. `fortad` supplies source-transformed derivative
kernels and derivative rules for numerical primitives. `fortopt` supplies
optimizer state and iteration algorithms. `fortml` supplies model semantics,
likelihoods, kernels, and inference. A model may call a `fortnum` operation only
through its public interface, with the derivative boundary named in the model
documentation.

The least-squares fit uses LAPACK SVD least squares. Ridge regularization adds
`sqrt(lambda)` rows for the slope parameters, so the fit does not square the
condition number through normal equations. Rank-deficient designs return the
minimum-norm SVD solution and are checked by an independent prediction oracle.
The SVD path is still a dense baseline. A future structured or randomized path
must retain its rank and residual checks.

## Differentiable model contract

Smooth maps expose the products that their public contracts declare. Each
declared product uses the same parameter layout:

- `value`: evaluate the model or objective.
- `jvp`: evaluate `J v` for a tangent direction.
- `vjp`: evaluate `J^T u` for an output cotangent.
- `hvp`: evaluate `H v` where the model or objective needs second-order data.

Every product has an independent check. JVPs use a convergence-tested central
finite difference or a high-precision reference. VJPs satisfy
`u^T (J v) = v^T (J^T u)`. HVPs are checked by differentiating an independently
computed gradient or by Hessian symmetry. `fortad` generated code and `fortsym`
derivations are production candidates, never the sole correctness oracle.

`basis_map_t%initialize_callback` accepts one explicit value procedure, one
JVP procedure, and one VJP procedure over a flat parameter vector. The map
stores that vector and routes `evaluate`, `jvp`, `vjp`, `parameters`, and
`set_parameters` through the same contract. `static_lowering_eligible()` is
false for callback maps. This keeps arbitrary host procedure pointers outside
accelerator regions until a generated static lowering exists.

`basis_map_t` is a façade over an abstract `basis_impl_t`. Polynomial, Fourier,
radial, spline, and callback maps implement the same deferred feature,
parameter, value, JVP, and VJP operations. The façade owns intercept columns,
shape checks, and the public parameter boundary. A new map implementation adds
one type and its product methods, not a case to every façade operation. The
kernel expression tree follows the same separation. Explicit opcodes appear
only at its static lowering boundary.

## MLP baseline

The MLP baseline stores each dense weight as `(input_width, output_width)` and
each bias as `(output_width)`. Samples are rows, so a layer evaluates `A W + b`.
The flat parameter vector stores every weight in Fortran column-major order,
followed by that layer's bias, from input to output. This layout is stable for
`fortopt` updates and can be mapped to a `fortad` active vector without a
structure-specific optimizer adapter.

The current implementation provides batched prediction, parameter/input JVPs,
parameter/input VJPs, HVPs, and a `backprop` alias for the reverse product.
Hidden and output layers support linear, `tanh`, ReLU, GELU (the standard tanh
approximation), SiLU, ELU, softplus, and fixed-slope leaky ReLU activations.
Smooth activations provide analytic first and second derivatives, which feed
the MLP JVP, VJP, and HVP products without finite differences. ReLU uses
derivative zero at its kink; leaky ReLU uses a fixed negative-side slope and
zero second derivative away from the kink. The explicit products are the
reference for the `fortad`-generated scalar fixture comparison.

## Exact GP baseline

`fortml_kernels` represents stationary and algebraic kernels as a recursive
expression tree. Positive variance and lengthscale parameters use logarithmic
storage, while sum and product nodes apply the corresponding product rules to
values, JVPs, VJPs, and HVPs. The exact GP uses this kernel API with a dense
Cholesky factorization from `fortnum`, supports shared covariance across
multiple outputs, and exposes predictive mean/variance, log marginal
likelihood, and hyperparameter products. Its tests use direct covariance
formulas and an independent finite-difference/LU oracle.

The derivative GP extends the same kernel tree with input gradients and mixed
input Hessians. RBF, Matérn, linear, constant, and supported compositions store
function-value and first-derivative observations in one covariance system,
so the same path supports derivative predictions and multiple output columns.
Component 0 denotes a function value and components 1 through the input
dimension denote first derivatives. White-noise derivatives and undefined
Matérn derivatives at coincident points are refused. The implementation is
verified against an independent finite-difference kernel oracle and a
hand-derived dense mixed-covariance solve. Parameter JVP/VJP/HVP products for
this extended covariance are not exposed.

`fortml_gp_training` is the first optimizer adapter at the model boundary. It
binds a fitted exact GP to FortOpt's context-aware objective, packs kernel and
noise hyperparameters in the same order as `gp_regression_t`, and evaluates
the negative log marginal likelihood and its analytic gradient after every
parameter update. The adapter owns no duplicate covariance state. Bounds,
convergence tolerances, and result diagnostics are explicit options, which
leaves derivative-observation and approximate-GP adapters on the same seam.

`fortml_gp_classification` reuses the kernel matrix and Cholesky boundaries for
a binary Laplace classifier. Newton updates solve the symmetric
`I + sqrt(W) K sqrt(W)` system, then prediction uses the posterior latent mean
and variance. Logistic observations use the MacKay probability integral and
probit observations use the closed-form Gaussian-CDF map. Latent and observed
probability input JVPs are assembled from the kernel input derivatives, so a
kernel that refuses an input derivative (for example a coincident Matérn
1/2 query) propagates that refusal rather than silently differentiating a
finite-difference approximation. `fortml_gp_multiclass_classification` wraps
that binary contract as deterministic one-vs-rest Laplace inference, normalizes
the positive probabilities, and preserves sorted integer labels. Variational
likelihoods and joint multiclass objectives need their own objective and state
contracts.

The exact depth-limited recursive `fortml_xgboost` lane is deliberately separate from the
smooth derivative graph. Each boosting iteration computes objective gradients
and Hessians at the current margins, exhaustively evaluates every feature
threshold, and stores the regularized Newton leaf weights. Predictions are
piecewise constant. Their input JVP is zero in an open leaf and refuses on a
learned split. This makes the second-order objective contract testable without
pretending that discrete split selection is differentiable. The exact backend
has an explicit IEEE-NaN policy: `error` rejects NaNs, `learn` scores both
default directions for each finite threshold, and `left`/`right` force a
direction. The selected route is stored per node and shared by fit,
prediction, multiclass normalization, and JVP validation; infinities remain
domain errors. Histograms, categorical, or monotonic constraints will be added
as distinct policies with independent oracles.

`fortml_preprocessing` keeps fitted statistics in model state and exposes only
the smooth input derivative of a transform. Standard scaling uses unit scale
for a constant column. Min-max scaling maps constant columns to the lower
range endpoint. These choices are explicit so a basis pipeline can compose a
scaler without inventing a hidden parameter derivative or changing the sample
axis.

## Lazy operators and iterative solves

`fortml_linear_operator` defines the matrix-free boundary used by scalable<!-- slop-ok -->
GP inference. An operator exposes vector products, batched products, a
diagonal, and its sample count. The caller never needs to know whether the
product uses a dense array, a tiled CPU loop, OpenACC, an FFT, a Kronecker
contraction, or a sparse backend. The first concrete implementation is
`rbf_operator_t`, which evaluates the RBF reduction without allocating a
covariance matrix.

`kernel_operator_t` wraps any supported composable kernel in a blocked
matrix-free host implementation. It is the reference path for kernel sums,
products, Matérn variants, and future derivative blocks. Leaf RBF kernels share
the specialized operator's fused matrix-free reduction and explicit device-data
lifetime, including a batched product path. Built-in composite expression trees
are flattened into a static postfix program before device execution. A
validated `kernel_formula_t` is copied into the same postfix program, including
when it appears inside a sum or product.

Its batched product evaluates each kernel block once and multiplies that block
by all right-hand sides together. This keeps the generic KeOps-style operation
boundary intact for composite kernels: changing from one to several right-hand
sides increases the block contraction, not the pairwise kernel evaluation.

`sparse_gp_operator_t` is the compact-support branch of the same contract. It
accepts coordinate triplets through `fortsparse`, retains a CSR view for
row-owned sums, and therefore has no output-write atomics in its OpenACC
product. Its resident device lifetime is explicit through `enter_data` and
`exit_data`. The caller owns the input/output data region. This is an
iterative product/CG backend, not a direct-solver wrapper. Matched dense
PyTorch and KeOps measurements are kept as a separate compact-support
workload in `fortml-bench`.

`ski_operator_t` is the interpolation branch. One-dimensional input uses a
cached Toeplitz grid and exposes its two interpolation weights. Higher
dimensions use an equal-axis tensor grid whose `q**d` points fit the declared
budget and accept one isotropic RBF leaf. Local experts offer contiguous groups
or deterministic Lloyd clustering. GRBCM first draws a reproducible disjoint
communication set, then trains each enhanced expert on that set plus one of
the remaining groups. Its weights and communication correction follow the
published aggregation rather than the ordinary prior-referenced expert rules.

`rbf_operator_t%enter_data` and `%exit_data` own resident points and the
optional five-matrix Krylov workspace selected by `n_rhs`. Backend choice stays
internal and recurrence scalars stay on the host. A solve without an explicit
resident lifetime uses a local data region.

The operator also delegates SPD solves to `fortnum`'s generic
preconditioned CG routine. The default preconditioner is the operator
diagonal. This keeps the KeOps-style split explicit: FortML owns the kernel
formula and data, while FortNum owns Krylov iteration and convergence status.
The RBF MVM, batched MVM, diagonal, and CG result are checked against direct
pairwise formulas and an independent dense solve. Persistent device data,
block and Nystrom preconditioners, stochastic Lanczos log determinants, and
LOVE-style single-point predictive variance are implemented. Autodiff rules
for the iterative solve remain outside this operator contract.

The experimental `solve_cg_multi_block` factors contiguous kernel blocks once
and applies their triangular solves to every right-hand side. It passes the
dense-solve oracle, but its first matched workload increased iterations. A
production block preconditioner still needs measured spatial reordering.

`solve_cg_multi_nystrom` constructs a landmark RBF factor and applies its
Woodbury inverse in one fused projection. The factors stay device-resident,
but the benchmark gate remains open until rank and landmark selection improve
the complete workload.

`rbf_operator_t%solve_cg_multi` applies the same PCG recurrence independently
to each right-hand side while evaluating all active search directions with one
batched operator product per iteration. This preserves the scalar CG
convergence contract and exposes the matrix-matrix fusion used by the native
CUDA path. Its preconditioned and unpreconditioned results are checked against
independent dense multi-RHS solves.

This batched recurrence is the default `linear_operator_t` multi-RHS contract.
`rbf_operator_t` keeps its accelerator override.

`kernel_operator_t%solve_cg_device` and `%solve_cg_multi_device` are the
generic accelerator entry points. They require `enter_data`, fuse one
matrix-matrix product per iteration, retain a scalar PCG recurrence per column,
and verify true residuals. The independent device test builds right-hand sides
from known solutions with a separate pairwise formula. Resident workspaces and
low-rank/block preconditioners share the path. An `nvfortran` trace confirms
the fused product and RHS-block updates. Convergence control remains host-side.

`structured_gp_operator_t` wraps the reusable `fortnum_tensor_product`
contraction and exposes the same persistent OpenACC lifetime shape through
`enter_data(status, n_rhs)`, `exit_data(status)`, `matvec_device`, and
`matmat_device`. Its factors and contraction workspaces remain resident while
the caller owns the input/output data region. The nvfortran path is checked by
an independent dense Kronecker oracle. Its device data region currently
excludes the generic FortML CG recurrence.

The RBF OpenACC CG override keeps Krylov vectors and repeated products in one
data region. Its matched Python lanes share the recurrence, tolerance, cap,
and true-residual check. Validated user formulas use the built-in resident
program path. Procedure callbacks remain host-only.

The eight-feature path has an optional native CUDA bridge enabled by
`FORTML_NATIVE_CUDA=1`. Its matrix-vector kernel uses warp-owned output rows
and 128-neighbor tiles. Its matrix-matrix kernel reuses each distance for up to
eight right-hand sides. The default build links
`fortml_cuda_rbf_stub.f90`. Direct MVM and matmat benchmarks check both native
paths against host pairwise oracles before timing them.

Generic static kernel programs have a separate opaque CUDA plan ABI in
`src/gp/fortml_cuda_kernel.cu`. The plan owns sample points and postfix data.
device products accept caller-owned RHS and output pointers. Its ten opcodes
pass the independent pairwise oracle in `test/run_cuda_kernel_plan.sh`. HIP and
SYCL can use the same ABI.

Without the native bridge, the tiled OpenACC/CPU reduction reuses each RBF
evaluation for up to eight right-hand sides. Larger batches use the per-column
fallback. The eight-feature MVM maps two output rows per gang and has a
five-row tail oracle. Other feature counts keep the general mapping.

## Implemented model sequence

The implementation order follows numerical dependencies:

1. Linear and basis-function regression.
2. MLPs with a flat parameter layout and explicit activation smoothness.
3. Exact GPs with composable kernels, scalar Gaussian observation noise, and
   Cholesky inference.
4. Derivative observations and derivative prediction by differentiating the
   kernel in both arguments. Each row records its function-value or first input
   derivative component.
5. A correlated multi-output GP, scalar inducing-point approximations, local
   experts, and structured linear operators.
6. A variational autoencoder and a single vanilla recurrent layer with
   backpropagation through time.

Bayesian global optimization is outside the current scope.

## State-of-the-art constraints

PyTorch `torch.func`, JAX transformations, and `fortad` point to a composable
product API. GPyTorch and `linear_operator` point to a lazy linear-operator
layer for scalable<!-- slop-ok --> GP inference. TensorFlow Probability and AbstractGPs show
that variational GP parameters, likelihoods, inducing points, and predictive
distributions should be separate objects. Stan Math and Julia's Enzyme/Zygote
ecosystem reinforce the requirement that custom primitives carry tested
partials and preserve higher-order differentiation where promised.

The Fortran implementation keeps arrays explicit and compiler-visible. A GPU
implementation must have an `nvfortran` build and an oracle on the cluster.
Benchmark claims compare matched CPU and GPU workloads against pinned GPyTorch
references and require the 30% runtime target stated in `ROADMAP.md`.
