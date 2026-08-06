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

- [x] Add automated operation-level comparison traces for dense PyTorch,
  KeOps, GPyTorch-KeOps, gfortran, and nvfortran using torch.profiler,
  perf, NV_ACC_TIME, and Nsight Systems.

- [x] Establish the package, MIT license, local `fortnum` dependency, and the
  oracle/performance/`nvfortran` rules.
- [x] Implement multi-output linear regression with intercept and ridge
  regularization using an SVD least-squares solve.
- [x] Implement linear prediction JVP and VJP products and check them against
  finite differences and the adjoint identity.
- [x] Replace the normal-equation fitting path with SVD for stable dense
  least-squares fits.
- [x] Benchmark conditioning against a high-precision reference.
- [x] Add basis-function maps with value, JVP, and VJP products. The initial
  set is polynomial, Fourier, radial, spline, and user-supplied differentiable
  maps.
  - [x] Add polynomial and log-frequency Fourier maps with flat parameters,
    value/JVP/VJP products, finite-difference checks, and adjoint identities.
  - [x] Add radial maps with differentiable centers and positive log-scales.
  - [x] Add B-spline maps backed by `fortnum_bspline` with fixed-span
    smoothness/status rules.
  - [x] Add a user-supplied value/JVP/VJP callback contract and static-lowering
    refusal boundary.
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
- [x] Add RBF function-value and derivative observations/predictions using
  kernel partial derivatives verified against `fortsym`, independent finite
  differences, and a hand-derived dense mixed-covariance solve.
- [ ] Extend derivative observation rules to Matérn and white-noise kernels
  with an explicit coincident-point smoothness contract.
- [x] Add the first public lazy-operator contract with kernel MVM, MVM
  batching, diagonal, sample count, and a backend-independent CG entry point.
  The RBF implementation keeps the covariance matrix implicit and delegates
  Krylov iteration to `fortnum`.
- [x] Add operator-owned `enter_data`/`exit_data` hooks for reusable RBF
  sample points, keeping the OpenACC/native CUDA choice inside the operator.
- [x] Add a resident generic leaf-RBF backend with fused matrix-free vector
  and multi-right-hand-side products, matching the specialized KeOps-style
  formula under `nvfortran`/OpenACC.
- [x] Extend operator-owned residency and static postfix-program lowering to
  built-in composite kernel expression trees, including fused products for up
  to eight right-hand sides.
- [ ] Add a safe static lowering contract for user-supplied kernel formulas.
- [x] Connect the RBF operator to `fortnum` CG with a diagonal
  preconditioner and an independent dense-solve oracle.
- [x] Add an OpenACC `nvfortran` RBF CG path that keeps the sample points and
  right-hand side resident across repeated solves, with device reductions and
  an automated residual benchmark.
- [x] Add optional native CUDA shared-neighbor tiles for the fixed eight-feature
  matrix-vector and up-to-eight-right-hand-side matrix-matrix paths, linked
  into the `nvfortran` benchmark through a Fortran C binding and retained
  behind the OpenACC fallback.
- [x] Fuse the OpenACC RBF matrix-matrix fallback for up to eight right-hand
  sides so the default GPU lane reuses each pairwise distance even without the
  native CUDA bridge.
- [x] Add operator-owned reusable multi-RHS Krylov workspaces with explicit
  `enter_data(status, n_rhs)` and `exit_data(status)` lifetime hooks, checked
  by repeated resident solves against the dense multi-RHS oracle.
- [x] Add fused multi-right-hand-side RBF CG with an independent PCG recurrence
  per output column and one batched operator product per iteration, checked
  against independent dense multi-RHS solves.
- [x] Fuse generic composable-kernel operator products across all right-hand
  sides so blocked kernel evaluation is not repeated per output column.
- [x] Expose the same independent-recurrence, batched matrix-product CG
  contract through the generic linear-operator base type.
- [x] Add an explicit generic `solve_cg_device` path that keeps the lowered
  kernel program and sample points resident while Krylov products and vector
  updates execute through OpenACC.
- [x] Extend the generic resident CG override to fused multi-RHS products with
  one independent PCG recurrence per column and an independent true-residual
  check.
- [x] Define a backend-neutral opaque C ABI for flat matrix-free plans and
  residency. Keep the Fortran CPU reference, implement generic postfix
  matvec/matmat reductions in native CUDA C++ first, and leave HIP and SYCL
  adapters open behind the same oracle and operation-level benchmark contract.
- [ ] Add persistent generic multi-RHS workspaces and block/Nystrom
  preconditioners.
- [ ] Add block/Nystrom preconditioners, stochastic Lanczos log determinants,
  and LOVE-style predictive-variance products for large exact-GP solves.
- [x] Add compact-support sparse covariance/precision dispatch through
  `fortsparse` CSC construction and iterative sparse MVM, with independent
  dense-oracle tests and nonzero-count diagnostics.
- [x] Add a resident OpenACC CSR view for compact-support sparse products.
  matched sparse CPU/GPU scaling against dense PyTorch and KeOps is recorded
  in `fortml-bench`.
- [x] Consume `fortnum_tensor_product` through the structured GP operator for
  separable tensor-grid covariance products, with vector, multi-RHS, diagonal,
  and CG correctness checks against dense oracles.
- [x] Expose the structured GP operator's persistent OpenACC factor and
  contraction workspaces through device vector and multi-RHS products, with a
  direct nvfortran CUDA oracle test.
- [x] Record the reusable `fortnum_toeplitz` 1-D cached circulant-embedding
  dependency and its independent dense-oracle/scaling evidence.
- [x] Add the FortML Toeplitz-backed GP wrapper with dense-oracle products and
  a CG solve check.
- [ ] Add multilevel tensor-grid embeddings, derivative products, and banded
  Markov-precision paths. The current Toeplitz FFT wrapper is host-resident.
- [ ] Add multi-output GPs, inducing-point variational GPs, and the structured
  inference policies that sit above these operator contracts.
- [ ] Add variational autoencoders and deep recurrent networks after the
  regression and GP contracts are stable. Their likelihoods, reparameterized
  gradients, scan/backpropagation, and higher-order derivative behavior each
  require separate oracle cases.
- [x] Add correctness-gated CPU benchmark targets for linear regression, MLP,
  and exact GP workloads.
- [x] Add matched CPU benchmark harnesses and GPyTorch reference runs through
  the separate `fortml-bench` repository.
- [x] Add `nvfortran` cluster builds, GPU correctness checks, and CPU/GPU plots
  for the first RBF matrix-vector workload.
- [x] Publish the repository and verified benchmark artifacts under the MIT
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
The MLP reference and accelerator plot remain open.

The first basis-map slice is now implemented in `fortml_basis`. It provides polynomial
powers, Fourier sine/cosine features, differentiable ARD radial features, and
fixed-knot B-spline features use stable layouts and expose value, JVP, and VJP
products. Independent central finite differences and VJP adjoint identities
pass in `test_basis`. The user callback contract stores a flat parameter vector
and dispatches value, JVP, and VJP products through explicit procedure
interfaces. `static_lowering_eligible()` returns false for callback maps, so
dynamic user code cannot enter an accelerator region by accident. The callback
case is checked by finite differences and the VJP adjoint identity.

The first matched RBF matrix-vector benchmark is now recorded in
`lazy-fortran/fortml-bench`. It uses 2048 samples, 8 features, float64, and
12 repetitions. With 16 physical CPU cores, the Fortran operator is 9 percent
slower than GPyTorch-KeOps. On the RTX 5060 Ti it is 68 percent faster on the
resident GPU lane. Every row passes the independent blocked NumPy oracle.
The comparison plot is https://box.sloppy.at/8ba9a.png and the raw CSV is
committed beside it in the benchmark repository. Matched log-determinant and
full-GP training evidence remain open. The matched CG harness is now recorded
in the benchmark repository as a separate workload.

The exact GP baseline now passes independent kernel-value, kernel-product,
JVP/VJP, prediction, likelihood, and multi-output checks with gfortran and
`nvfortran` 26.5. Its correctness-gated benchmark covers a 128-sample,
4-feature, 2-output workload and checks predictions against an independent LU
solve. The host results are 0.722 ms with gfortran and 0.633 ms with
`nvfortran`. GPU offload, matched GPyTorch comparisons, and the 30% runtime
target remain open.

The first Fortran-native tiled RBF MVM also passes its direct pairwise oracle
with gfortran and an OpenACC `nvfortran` build. For 2,048 samples, 8 features,
and 12 MVM repetitions, the host result was 24.74 ms per MVM. The RTX 5060 Ti
result was 1.225 ms transfer-inclusive and 1.168 ms with resident data in one
run. The GPU compile report shows one gang per output tile and 128 vector
threads reducing over each neighbor tile. This is kernel-only evidence, not a
matched GPyTorch comparison, so the 30% target remains open.

## Optimization update

The RBF operator now stores samples contiguously for the neighbor reduction,
uses an eight-feature unrolled distance path, and replaces explicit square
power operations with multiplies. The changes passed the full gfortran test
suite and the independent direct pairwise oracle, and are published on main
in fortml commit a205898.

The refreshed matched sweep uses 256, 512, 1024, 2048, and 4096 samples,
float64, twelve repetitions, nvfortran 26.5 on 16 physical CPU cores and an
RTX 5060 Ti. The CPU lane uses nvfortran -O3 -mp and the GPU lane uses
nvfortran -O3 -acc. The resident GPU curve is below dense PyTorch, KeOps, and
GPyTorch-KeOps at every tested size. At 4096 it takes 4.32 ms versus 6.31 ms
for GPyTorch-KeOps, while dense PyTorch is out of memory. The CPU endpoint is
4.66 ms versus 6.22 ms for GPyTorch-KeOps and 7.56 ms for KeOps. The Fortran
curve is lowest at every tested size on both devices.

The current CPU and GPU plots are
https://box.sloppy.at/0f460.png and https://box.sloppy.at/e1f7f.png.
Operation-level findings and raw traces are recorded in the fortml-bench
operation profile. Nsight Compute is installed but blocked by
ERR_NVGPUCTRPERM. Occupancy and memory-counter work remains open until the
cluster grants performance-counter access.

A high-N follow-up now extends the same float64 RBF workload through 16,384
samples. The resident nvfortran/OpenACC GPU timings scale with local slopes
1.992 and 1.997 from 4,096 to 8,192 and 16,384 samples, respectively. Three
additional unpinned nvfortran CPU runs per size give median timings of 5.707,
16.202, and 59.851 ms, with local slopes 1.505 and 1.885. The CPU result is
approaching the expected quadratic regime but remains sensitive to host
affinity. Dense PyTorch is OOM on the GPU at these high-N points, while KeOps
and GPyTorch-KeOps pass the independent oracle.

The extended CPU and GPU plots are
https://box.sloppy.at/c7d09.png and https://box.sloppy.at/465d6.png.

The accelerator follow-up adds a two-row worker tile to the eight-feature
OpenACC reduction and checks its tail with an independent five-row oracle. The
change is published in fortml commit `e3068a3`. The refreshed 2,048-sample
matrix-free CG run takes 0.162 s on the 16-thread nvfortran CPU lane and
0.187 s on the RTX 5060 Ti. At 4,096 samples it takes 0.829 s on CPU and
0.872 s on CUDA. The corresponding KeOps and GPyTorch-KeOps CUDA times at
4,096 are 1.876 s and 1.446 s. Dense PyTorch is OOM at that CUDA size.

The matched CG workload uses the same float64 RBF parameters, diagonal shift,
unpreconditioned recurrence, tolerance `1e-8`, and 500-iteration cap for all
four implementations. Every non-OOM row passes the blocked NumPy residual
check. The 2,048-sample rows also use an independent dense solve in the
correctness suite. These measurements include a true-residual check and are
operator-level evidence for the KeOps-style matrix-free path. They do not
close preconditioned solves, stochastic log determinants, or full GP training.
The raw record and scaling plots are in `fortml-bench/results/rbf_cg.csv` and
its CG plot files.

From 2,048 to 4,096 samples, the FortML CUDA solve has a local doubling slope
of 2.22. KeOps and GPyTorch-KeOps have slopes of 1.31 and 1.28 on the same
run. The FortML kernel is below both KeOps lanes at every tested CG size, but
its high-N slope still reflects a quadratic dense pair interaction. Persistent
backend-owned workspaces, multi-right-hand-side fusion, and block or Nystrom
preconditioners remain the next accelerator gates. The refreshed extended
plots are recorded in `fortml-bench/results/rbf_cg_scaling_extended_cpu.png`
and `fortml-bench/results/rbf_cg_scaling_extended_cuda.png`. Public copies are
https://box.sloppy.at/9cef6.png for CPU and
https://box.sloppy.at/4d9a5.png for CUDA.

The fused matmat benchmark records one, two, four, and eight right-hand sides
at 2,048 samples in `fortml-bench/results/rbf_matmat.csv`. The native CUDA
resident path takes 1.363 ms for four RHS and 1.405 ms for eight RHS, compared
with 3.666 ms and 7.327 ms for the OpenACC loop. The CPU and GPU plots are
published at https://box.sloppy.at/98dcc.png and
https://box.sloppy.at/aabb5.png. Every row passes the direct pairwise oracle
for every RHS.

The derivative-GP pilot now provides `gp_derivative_regression_t` for RBF
function-value and first-input-derivative observations and predictions. Its
multi-output mixed covariance system is checked against a hand-derived dense
solve, while the kernel input derivatives are checked by independent central
finite differences and the symbolic `fortsym` derivation recorded under
`.provenance/derivations/`. Parameter products for this extended covariance
remain open under the `fortad` integration gate.

The optional native CUDA bridge is now correctness-gated by the direct MVM and
matmat benchmarks and the benchmark profiler. Its four-warp block loads each
128-neighbor tile once into shared memory and uses one warp per output row. At
2,048 samples, Nsight Systems measured 915.6 us for the native MVM kernel and
922.1 us for OpenACC. The application measured 940.7 us and 946.4 us per
resident MVM, respectively. For four right-hand sides, the native matmat path
took 1.363 ms per resident call versus 3.666 ms for the OpenACC loop. Every
native result passed the direct pairwise oracle. The native MVM path is within
the current OpenACC timing envelope, while the fused matmat path shows the
expected block reuse. OpenACC remains the comparison backend for CG.

The fused multi-RHS CG sweep now covers 256, 512, 1024, and 2048 samples with
four float64 right-hand sides. Every row passes the blocked NumPy matmat
residual oracle, and the FortML rows pass the independent dense multi-RHS
solve. At 2048 samples, the default OpenACC lane takes 0.769 s on CUDA versus
0.848 s for GPyTorch-KeOps and 0.965 s for KeOps. The native CUDA lane takes
0.328 s, below dense PyTorch at 0.358 s and below both matrix-free comparison
lanes. The CPU lane is 0.699 s versus 0.735 s for GPyTorch-KeOps. Scaling
plots are published at https://box.sloppy.at/8801e.png for native CUDA and
https://box.sloppy.at/2344d.png for OpenACC. The raw records and exact
workload are in `fortml-bench/results/rbf_cg_multi_scaling.csv` and
`fortml-bench/results/rbf_cg_multi_scaling.md`. The native GPU slope remains
the next optimization target under block or Nystrom preconditioning.

The reusable higher-dimensional tensor-product contraction primitive is now
implemented and independently tested in `fortnum` as
`fortnum_tensor_product`, and FortML now exposes it through
`structured_gp_operator_t`, including persistent OpenACC vector and multi-RHS
products. The one-dimensional `fortnum_toeplitz` dependency also has cached
FFT products and independent scaling evidence, and FortML now wraps it in
`toeplitz_gp_operator_t` with a dense-oracle CG check. The generic kernel
operator now has resident single- and multi-RHS CG recurrences. The multi-RHS
path fuses each kernel matrix product while keeping independent scalar PCG
state and recomputing true residuals before accepting convergence. Persistent
workspace ownership, derivative products, multilevel embeddings, and matched
CPU/GPU scaling evidence for the Toeplitz GP path remain open.
The direct `nvfortran` launch trace confirms that the pairwise composite
evaluation remains one fused matrix-matrix kernel per product. Per-column
step/beta updates are now fused across the RHS block, leaving host-controlled
convergence and candidate-column cleanup as the next operation-level target.

The compact-support sparse branch now consumes `fortsparse` triplets and
retains a CSR view for row-owned host and OpenACC products. Its float64,
radius-8, four-RHS workload passes the independent row-wise oracle at every
tested size. On the RTX 5060 Ti, resident CUDA takes 0.0687 ms at N=512,
0.1196 ms at N=4096, and 0.4557 ms at N=16384. The same workload is recorded
against dense PyTorch and KeOps in `fortml-bench`. This sparse comparison is
not a claim about the Gaussian pairwise benchmark: KeOps still evaluates all
pairs here, while FortML exploits the explicit compact-support sparsity.

The generic `kernel_operator_t` now lowers leaf RBF kernels to the same fused
matrix-free reduction as `rbf_operator_t`. Its sample points have explicit
`enter_data`/`exit_data` lifetime hooks, and vector/multi-RHS products are
available through `matvec_device`/`matmat_device`. A direct pairwise vector and
matrix oracle checks the path with gfortran and direct `nvfortran`/OpenACC.
Built-in composite kernel trees are flattened into a static postfix program
before execution, so sum/product nodes never invoke recursive callbacks in an
accelerator region. User-supplied formulas still need an explicit lowering
contract before they can enter this path.

The linear-regression conditioning gate is now recorded in
`benchmark/reference/linear_conditioning.csv`. A 16-sample, three-feature
near-collinear design was solved by FortML's SVD fit and independently solved
with `mpmath.qr_solve` at 80 decimal digits. The 2-norm condition numbers range
from 42.8 to 4.05e15. All ten gfortran and nvfortran rows pass the independent
prediction oracle with relative errors below 6e-16. Coefficient error grows as
the design becomes unidentifiable, reaching 11.6 percent for gfortran and 3.03
percent for nvfortran at condition 4.05e15, while the fitted predictions remain
at machine precision. Fresh FPM builds record complete-workload times of 8.20 s
for gfortran and 6.49 s for nvfortran, peak RSS of 218844 and 54676 kB, and
generated executable sizes of 72416 and 47984 bytes. The reproducible driver is
`benchmark/linear_conditioning.py`.
