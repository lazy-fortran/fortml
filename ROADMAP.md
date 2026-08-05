# fortml roadmap

`fortml` provides differentiable regression and probabilistic models in modern
Fortran. `fortnum` owns general numerical kernels and `fortad` owns source-level
automatic differentiation and generated JVP/VJP/HVP code. `fortopt` owns
optimizers. This boundary keeps downstream numerical code stable while each
repository can test its own contracts.

## Completion rules

Each item is completed on `main` with implementation, an independent
behavioral oracle, documentation, and focused tests. A repository-state check
is never accepted as the test oracle. Correctness evidence must use a
hand-derived result, a convergence-tested finite difference, a complex-step
check, a dot-product adjoint identity, or a trusted high-precision reference.

Every performance item records compiler, flags, CPU/GPU model, problem size,
precision, wall time, peak memory, build time, and generated-code size. CPU and
GPU runs use matched workloads and report a comparison against a pinned
GPyTorch baseline. The target is within 30% of that baseline on each workload
where the algorithms and precision are comparable. A target is not claimed
without a plot and raw data.

The accelerator lane must build and run with NVIDIA `nvfortran` on the cluster,
including CUDA-aware numerical kernels where supported. A local compiler result
does not close the cluster gate. The benchmark harness will accept `FC=nvfortran`
and record `nvfortran --version`, CUDA version, GPU model, and compiler flags.

When a benchmark plot is completed, report the public plot URL, workload,
correctness result, and comparison table to Chris in the agreed Zulip DM. A
missing report leaves the item open.

## Work order

- [x] Establish the package, MIT license, local `fortnum` dependency, and the
  oracle/performance/`nvfortran` rules.
- [x] Implement multi-output linear regression with intercept and ridge
  regularization using an SVD least-squares solve.
- [x] Implement linear prediction JVP and VJP products and check them against
  finite differences and the adjoint identity.
- [x] Replace the normal-equation fitting path with SVD for stable dense
  least-squares fits.
- [ ] Benchmark conditioning against a high-precision reference.
- [ ] Add basis-function maps with value, JVP, and VJP products. The initial
  set is polynomial, Fourier, radial, spline, and user-supplied differentiable
  maps.
- [x] Add the explicit MLP baseline: flat column-major parameters, batched
  forward products, `tanh`/linear/ReLU activations, JVPs, VJPs, and
  backpropagation, with independent finite-difference and adjoint checks.
- [x] Exercise the MLP parameter/VJP seam with the separate `fortopt_adam`
  implementation on a known squared-loss objective.
- [ ] Connect the MLP product contract to `fortad`-generated JVP/VJP/HVP code
  and compare the generated kernels with the explicit baseline.
- [x] Add RBF, Matern, linear, constant, white-noise, sum, and product kernels
  with log-parameter layouts and independent value/product checks.
- [x] Add multi-output exact Gaussian-process regression with Cholesky
  inference, predictive variance, log marginal likelihood, hyperparameter
  gradients, prediction JVPs, and prediction VJPs.
- [x] Add a correctness-gated Fortran-native tiled RBF matrix-vector product
  with OpenACC GPU execution and resident-data support.
- [ ] Connect the kernel and GP product contracts to `fortad`-generated
  JVP/VJP/HVP code and compare generated kernels with the explicit baseline.
- [ ] Add function-value and derivative observations/predictions using kernel
  partial derivatives verified against symbolic expressions and independent
  finite differences.
- [ ] Add a public lazy-operator contract with kernel MVM, MVM batching,
  diagonal, and persistent host/device data; hide tiling and backend choice
  from GP callers in the style of the KeOps/GPyTorch split.
- [ ] Use `fortnum` CG with diagonal/block/Nystrom preconditioners for large
  exact-GP solves, and add stochastic Lanczos log-determinant and LOVE-style
  predictive-variance products.
- [ ] Add compact-support sparse covariance/precision dispatch through
  `fortsparse` or iterative sparse MVM, with fill-in and memory diagnostics.
- [ ] Add regular-grid operators: 1-D and multilevel Toeplitz FFT products,
  Kronecker/tensor-product contractions, and banded Markov-precision paths.
- [ ] Add multi-output GPs, inducing-point variational GPs, and the structured
  inference policies that sit above these operator contracts.
- [ ] Add variational autoencoders and deep recurrent networks after the
  regression and GP contracts are stable. Their likelihoods, reparameterized
  gradients, scan/backpropagation, and higher-order derivative behavior each
  require separate oracle cases.
- [x] Add correctness-gated CPU benchmark targets for linear regression, MLP,
  and exact GP workloads.
- [ ] Add matched CPU benchmark harnesses and GPyTorch reference runs.
- [ ] Add `nvfortran` cluster builds, GPU correctness checks, and CPU/GPU plots.
- [ ] Publish the repository and verified benchmark artifacts under the MIT
  license.

## Research record

The ignored `.provenance/` tree contains shallow upstream clones, downloaded
open-access papers, checksums, exact revisions, and the survey manifest. Books
and papers that are not legally redistributable are represented by citation and
publisher or DOI metadata rather than copied files.

## Current evidence

The first local CPU/compiler plot is available at
https://box.sloppy.at/d4f68.png. It compares only host/LAPACK execution for the
SVD-based linear-regression smoke workload. It does not close the GPyTorch, GPU
offload, peak-memory, or generated-code-size gates.

The explicit MLP and its `fortopt_adam` training seam pass the focused test
suite with gfortran and `nvfortran` 26.5. No MLP performance claim is made yet.
The matched PyTorch/GPyTorch reference and accelerator plot remain open.

The exact GP baseline now passes independent kernel-value, kernel-product,
JVP/VJP, prediction, likelihood, and multi-output checks with gfortran and
`nvfortran` 26.5. Its correctness-gated benchmark covers a 128-sample,
4-feature, 2-output workload and checks predictions against an independent LU
solve. The host results are 0.722 ms with gfortran and 0.633 ms with
`nvfortran`. GPU offload, matched GPyTorch comparisons, and the 30% runtime
target remain open.

The first Fortran-native tiled RBF MVM also passes its direct pairwise oracle
with gfortran and an OpenACC `nvfortran` build. For 2,048 samples, 8 features,
and 12 MVM repetitions, the host result was 24.74 ms per MVM; the RTX 5060 Ti
result was 1.225 ms transfer-inclusive and 1.168 ms with resident data in one
run. The GPU compile report shows one gang per output tile and 128 vector
threads reducing over each neighbor tile. This is kernel-only evidence, not a
matched GPyTorch comparison, so the 30% target remains open.
