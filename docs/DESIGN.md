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
