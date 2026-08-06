# fortml design

The first interface is a matrix-oriented regression API. Samples occupy rows,
features occupy columns, and outputs occupy columns. A model stores its
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

Each smooth map will provide the following products over the same parameter
layout:

- `value`: evaluate the model or objective.
- `jvp`: evaluate `J v` for a tangent direction.
- `vjp`: evaluate `J^T u` for an output cotangent.
- `hvp`: evaluate `H v` where the model or objective needs second-order data.

Every product has an independent check. JVPs use a convergence-tested central
finite difference or a high-precision reference. VJPs satisfy
`u^T (J v) = v^T (J^T u)`. HVPs are checked by differentiating an independently
computed gradient or by Hessian symmetry. `fortad` generated code and `fortsym`
derivations are production candidates, never the sole correctness oracle.

## MLP baseline

The MLP baseline stores each dense weight as `(input_width, output_width)` and
each bias as `(output_width)`. Samples are rows, so a layer evaluates `A W + b`.
The flat parameter vector stores every weight in Fortran column-major order,
followed by that layer's bias, from input to output. This layout is stable for
`fortopt` updates and can be mapped to a `fortad` active vector without a
structure-specific optimizer adapter.

The current implementation provides batched prediction, parameter/input JVPs,
parameter/input VJPs, and a `backprop` alias for the reverse product. Hidden
layers support `tanh` and ReLU. The output layer also supports linear and ReLU.
The ReLU derivative is defined as zero at its kink, and finite-difference tests
stay away from that point. The explicit products are the reference
implementation for the open `fortad`-generated-kernel comparison.

## Exact GP baseline

`fortml_kernels` represents stationary and algebraic kernels as a recursive
expression tree. Positive variance and lengthscale parameters use logarithmic
storage, while sum and product nodes apply the corresponding product rules to
values, JVPs, and VJPs. The initial exact GP uses this kernel API with a dense
Cholesky factorization from `fortnum`, supports shared covariance across
multiple outputs, and exposes predictive mean/variance, log marginal
likelihood, and hyperparameter products. Its tests use direct covariance
formulas and an independent finite-difference/LU oracle.

The derivative-GP pilot extends the same kernel tree with input gradients and
mixed input Hessians. RBF function-value and first-derivative observations are
stored in one interleaved covariance system, so the same path supports
derivative predictions and multiple output columns. The implementation is
verified against an independent finite-difference kernel oracle and a
hand-derived dense mixed-covariance solve. The RBF formulas are also recorded
in `.provenance/derivations/rbf_input_derivatives.txt` from `fortsym`. Parameter
JVP/VJP/HVP products for this extended covariance remain part of the open
`fortad` integration gate.

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
products, Matérn variants, and future derivative blocks. Leaf RBF kernels now
share the specialized operator's fused matrix-free reduction and explicit
device-data lifetime, including a batched product path. Built-in composite
expression trees are flattened into a static postfix program before device
execution; user-supplied formulas remain on the blocked host reference until
they have an explicit lowering contract.

Its batched product evaluates each kernel block once and multiplies that block
by all right-hand sides together. This keeps the generic KeOps-style operation
boundary intact for composite kernels: changing from one to several right-hand
sides increases the block contraction, not the pairwise kernel evaluation.

`sparse_gp_operator_t` is the compact-support branch of the same contract. It
accepts coordinate triplets through `fortsparse`, retains a CSR view for
row-owned sums, and therefore has no output-write atomics in its OpenACC
product. Its resident device lifetime is explicit through `enter_data` and
`exit_data`; the caller owns the input/output data region. This is an
iterative product/CG backend, not a direct-solver wrapper. Matched dense
PyTorch and KeOps measurements are kept as a separate compact-support
workload in `fortml-bench`.

`rbf_operator_t%enter_data` and `%exit_data` own the device lifetime of the
reusable sample-point array. Calls through the same operator keep the
OpenACC/native-CUDA backend choice internal to the operator. Right-hand-side
and Krylov-workspace residency are now part of the same lifetime contract:
`enter_data(status, n_rhs)` allocates and makes the five large Krylov matrices
resident, while `exit_data(status)` releases them. The small recurrence scalars
remain host-side control state. A solve without an explicit `enter_data` still
manages the workspace through its local accelerator data region.

The operator also delegates SPD solves to `fortnum`'s generic
preconditioned CG routine. The default preconditioner is the operator
diagonal. This keeps the KeOps-style split explicit: FortML owns the kernel
formula and data, while FortNum owns Krylov iteration and convergence status.
The RBF MVM, batched MVM, diagonal, and CG result are checked against direct
pairwise formulas and an independent dense solve. Persistent device-data
ownership, block and Nystrom preconditioners, stochastic log determinants, and
autodiff rules for the iterative solve remain separate milestones.

The specialized RBF path also exposes `solve_cg_multi_block` as an experimental
contiguous block-Jacobi variant. It factors each kernel block once and applies
the two triangular solves to every right-hand side inside the accelerator data
region. The path is correctness-gated against the independent dense solve, but
it is not the default benchmark lane: the first matched workload showed that
this ordering and block size can increase iterations, so a production
preconditioner still needs a measured Nystrom or spatially reordered design.

The same specialized path exposes `solve_cg_multi_nystrom`. It constructs a
landmark RBF factor and applies the resulting Woodbury inverse to all right-hand
sides in one fused projection. Its low-rank factors remain device-resident for
the solve, but the matched benchmark gate is still open until rank and landmark
selection are shown to improve the complete workload.

`rbf_operator_t%solve_cg_multi` applies the same PCG recurrence independently
to each right-hand side while evaluating all active search directions with one
batched operator product per iteration. This preserves the scalar CG
convergence contract and exposes the matrix-matrix fusion used by the native
CUDA path. Its preconditioned and unpreconditioned results are checked against
independent dense multi-RHS solves.

The same batched recurrence is now the default `linear_operator_t` multi-RHS
contract. Generic composable-kernel operators therefore retain fusion all the
way through CG, while `rbf_operator_t` keeps its specialized accelerator
override.

`kernel_operator_t%solve_cg_device` is the first generic accelerator solve
entry point. It requires `enter_data` for the sample points and static kernel
program, maps only the right-hand side and per-solve Krylov vectors at the
call boundary, and recomputes the true residual before accepting convergence.
The independent device test constructs the right-hand side from a known
solution using a separate pairwise formula. A persistent generic multi-RHS
workspace and low-rank/block preconditioners remain separate milestones.

`structured_gp_operator_t` wraps the reusable `fortnum_tensor_product`
contraction and exposes the same persistent OpenACC lifetime shape through
`enter_data(status, n_rhs)`, `exit_data(status)`, `matvec_device`, and
`matmat_device`. Its factors and contraction workspaces remain resident while
the caller owns the input/output data region. The nvfortran path is checked by
an independent dense Kronecker oracle; the generic FortML CG recurrence has not
yet been moved into that device data region.

The specialized RBF type also has an OpenACC CG override. Its Krylov vectors,
reductions, and repeated RBF products execute inside one accelerator data
region, while an enclosing benchmark region keeps the sample points and
right-hand side resident across repeated solves. The benchmark uses the same
unpreconditioned recurrence for the Python comparison lanes so its tolerance,
iteration cap, and true-residual check are directly comparable. The generic
`kernel_operator_t` remains a blocked host-reference path for user-supplied
formulas until persistent backend-owned data and a static accelerator lowering
contract are added.

The eight-feature path also has an optional native CUDA bridge. The benchmark
drivers enable it with `FORTML_NATIVE_CUDA=1`, compile
`src/gp/fortml_cuda_rbf.cu` with `nvcc`, and link the object into the
`nvfortran` executable. The matrix-vector kernel assigns one warp to each of
four output rows, loads a 128-neighbor tile into shared memory, and reduces the
tile in warp lanes. The matrix-matrix kernel keeps up to eight right-hand sides
in the same block, evaluates each pairwise distance once, and reduces the
result for every right-hand side before writing the outputs. The default
Fortran build links `fortml_cuda_rbf_stub.f90`, so gfortran and ordinary
nvfortran/OpenACC builds remain self-contained. The direct MVM and matmat
benchmarks check the native paths against host pairwise oracles before timing
them.

When the native bridge is disabled, the RBF matrix-matrix fallback still fuses
up to eight right-hand sides in one OpenACC/CPU tiled reduction. Pairwise
distances and the exponential are evaluated once per sample pair, then applied
to scalar RHS accumulators. Larger RHS batches retain the conservative
per-column fallback until a larger device workspace contract is added.

The eight-feature OpenACC RBF MVM maps two output rows to worker lanes inside
each gang. This keeps the sample-major neighbor loop contiguous while reducing
gang count. The tail condition is covered by a five-row, eight-feature direct
pairwise oracle test. The mapping is specialized to the fixed eight-feature
path, while other feature counts retain the general tiled implementation.

## Model sequence

The implementation order follows numerical dependencies:

1. Linear and basis-function regression.
2. MLPs with a flat parameter layout and explicit activation smoothness.
3. Exact GPs with composable means, kernels, likelihoods, and Cholesky
   inference.
4. Derivative observations and derivative prediction by differentiating the
   kernel in both arguments. The covariance block ordering is function values,
   input derivatives, and optional mixed derivatives.
5. Multi-output GPs using an explicit linear model of coregionalization and
   inducing-point variational inference. Structured linear operators stay lazy
   until a dense result is required.
6. Variational autoencoders and deep recurrent networks after reparameterized
   gradients and recurrent scan/backpropagation have their own oracle suites.

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
