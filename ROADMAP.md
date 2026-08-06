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

The differentiable parameter vector covers every quantity exposed to an
optimizer: model weights, basis and kernel hyperparameters, likelihood
parameters, inducing or structured-GP parameters, and variational parameters.
Each such block must participate in value, JVP, VJP, and HVP products where the
model declares the product, with an independent check for the complete packed
vector.

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

The prerequisite build driver is `fo`'s four-way compiler dialect layer:
GNU Fortran, NVIDIA `nvfortran`, Intel LLVM `ifx`, and LLVM Flang. FortML's
nvfortran gate is now exercised through `FO_FC` using the cluster's nvfortran
26.5 and the path-only dependency graph. Legacy `ifort` is not a supported
Intel lane. Keep the ifx executable gate open until an ifx installation is
available.

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
  - [x] Put built-in and callback maps behind one abstract implementation
    contract. Keep intercept handling and shape validation in the public
    wrapper so adding a map does not add dispatch cases to every product.
- [x] Add the explicit MLP baseline: flat column-major parameters, batched
  forward products, `tanh`/linear/ReLU activations, JVPs, VJPs, and
  forward-over-reverse HVPs, with independent finite-difference and adjoint
  checks.
- [x] Exercise the MLP parameter/VJP seam with the separate `fortopt_adam`
  implementation on a known squared-loss objective.
- [x] Define one optimizer-facing parameter registry and packing contract for
  model weights, basis/kernel hyperparameters, likelihood parameters, inducing
  variables, and variational parameters. The typed callback seam has live MLP
  and kernel adapters and rejects duplicate block names.
- [x] Add the registry-facing parameter-product contract over one packed live
  vector. MLP value/JVP/VJP/HVP products are routed through the registry and
  checked by finite-difference, adjoint, and Hessian-symmetry oracles; fitted
  GP mean products route kernel and log-noise hyperparameters through the same
  contract. RBF fitted-GP mean HVPs now include the differentiated solve path.
- [x] Connect the static scalar MLP product fixture to `fortad`-generated
  JVP/VJP/HVP code and compare the generated kernels with the explicit
  baseline. The dynamic registry-wide product routing remains open.
- [x] Add Bayesian neural networks with explicit priors over weights,
  reparameterized variational posteriors, deterministic seeded Monte Carlo, and
  optimizer-visible ELBO value/JVP/VJP/HVP products. Check the Gaussian KL term
  against its analytic formula and the complete ELBO gradient against an
  independent finite-difference oracle.
- [x] Add a reusable variational-inference contract for GPs and neural models,
  including variational means, covariance or factor parameters, inducing-point
  parameters, likelihood parameters, minibatch scaling, and natural-gradient
  or `fortopt` updates. Check ELBO decomposition, KL positivity, seeded
  reparameterization, and convergence on a small conjugate model against an
  analytic posterior.
- [x] Add RBF, Matern, linear, constant, white-noise, sum, and product kernels
  with log-parameter layouts and independent value/product checks.
- [x] Add multi-output exact Gaussian-process regression with Cholesky
  inference, predictive variance, log marginal likelihood, hyperparameter
  gradients and HVPs, prediction JVPs, prediction VJPs, and prediction HVPs.
- [x] Add a correctness-gated Fortran-native tiled RBF matrix-vector product
  with OpenACC GPU execution and resident-data support.
- [x] Connect the RBF kernel and fitted-GP value/JVP/VJP contracts to
  `fortad`-generated JVP/VJP/HVP code and compare the generated RBF products
  with explicit finite-difference and adjoint baselines. Generated FortAD
  products now also cover Matérn 1/2, 3/2, and 5/2 scalar HVPs; analytic
  elementary and recursive sum/product rules cover the remaining built-in
  kernel parameter products. The fitted-GP mean HVP differentiates both solve
  adjoints and changing kernel VJP cotangents; the packed product contract
  exposes it to optimizers. The complete built-in kernel HVP rule is checked
  against central finite differences of the independent kernel VJP.
- [x] Add a differentiated GP linear-solve/Cholesky path so the fitted-GP
  mean optimizer contract exposes an HVP instead of refusing it. The complete
  packed vector is checked against a central finite difference of the GP VJP.
- [x] Add RBF function-value and derivative observations/predictions using
  kernel partial derivatives verified against `fortsym`, independent finite
  differences, and a hand-derived dense mixed-covariance solve.
- [x] Extend derivative observation rules to Matérn and white-noise kernels
  with an explicit coincident-point smoothness contract. Generated radial
  derivatives cover Matérn 3/2 and 5/2, Matérn 1/2 derivatives refuse
  coincident points, and white noise accepts function-value observations while
  refusing derivative observations.
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
- [x] Add a safe static lowering contract for user-supplied kernel formulas.
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
  FortSym's validated shared-IR emitter is the source-generation boundary for
  future static user formulas; this runtime ABI remains independent of
  FortSym and keeps launch policy, residency, and autodiff outside the leaf.
- [x] Consume a FortSym-generated RBF CUDA leaf in the generic postfix plan,
  with the existing independent dense matvec/matmat oracle covering the
  generated leaf through the full resident-plan ABI.
- [x] Add persistent generic multi-RHS workspaces and block/Nystrom
  preconditioners.
- [x] Add block/Nystrom preconditioners, stochastic Lanczos log determinants,
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
- [x] Add multilevel tensor-grid embeddings, derivative products, and banded
  Markov-precision paths. The current Toeplitz FFT wrapper is host-resident.
- [x] Add multi-output GPs, inducing-point variational GPs, and the structured
  inference policies that sit above these operator contracts.
- [x] Add variational autoencoders and deep recurrent networks after the
  regression, GP, and variational-inference contracts are stable. Their
  likelihoods, reparameterized gradients, scan/backpropagation, and
  higher-order derivative behavior each require separate oracle cases.
- [x] Add correctness-gated CPU benchmark targets for linear regression, MLP,
  and exact GP workloads.
- [x] Add matched CPU benchmark harnesses and GPyTorch reference runs through
  the separate `fortml-bench` repository.
- [x] Add `nvfortran` cluster builds, GPU correctness checks, and CPU/GPU plots
  for the first RBF matrix-vector workload.
- [x] Publish the repository and verified benchmark artifacts under the MIT
  license.

## Scalable-GP review benchmark

The reference is Liu, Ong, Shen and Cai, "When Gaussian Process Meets Big
Data: A Review of Scalable GPs", IEEE TNNLS 31(11):4405-4423, 2020
(DOI 10.1109/TNNLS.2019.2957109). The review has no numeric result tables: its
reproducible evidence is the one-dimensional toy of Figs. 4 and 5
(`y = sinc(x) + eps`, `eps ~ N(0, 0.04)`, 120 training points), the qualitative
behaviours it reports there, the complexity claims of Fig. 2, the library list
of Table I, and the data set list of Table II. Matching the paper therefore
means reproducing those behaviours and complexity orders, not fitting numbers
that do not exist.

- [x] Implement the prior sparse approximations SoR, DTC, FITC and PITC with a
  shared inducing framework. Each collapses to the exact GP when the inducing
  set is the training set, and the SoR overconfidence and DTC return-to-prior
  of Fig. 4 are checked directly.
- [x] Implement the local approximations NLE, PoE, GPoE, BCM, RBCM and GRBCM.
  PoE, the normalized GPoE and BCM reproduce the exact GP with one expert; the
  PoE overconfidence and the GPoE return-to-prior of Fig. 5 are checked.
- [x] Implement SKI grid interpolation and subset-of-data selection. The SKI
  product is checked against a dense `W K_uu W^T` assembly and its convergence
  under grid refinement.
- [ ] Implement the mixture-of-experts gating aggregation of Fig. 5.
- [ ] Add the paper's 1-D toy as a shared fixture and check every method's
  reported Fig. 4/Fig. 5 behaviour automatically.
- [ ] Record wall time and peak resident memory for every method in
  `fortml-bench`, with scaling sweeps in the sample count, the inducing size,
  the expert count and the input dimension, and document the measured order
  against the complexity claimed in Fig. 2.
- [ ] Add the KeOps-style matrix-free exact lane to the same comparison and
  answer whether it is good enough on its own, or where the approximations
  still win.
- [ ] Record provenance for the paper and for every third-party implementation
  compared against, with download scripts and checksums.
- [ ] Publish the plots and report the comparison to Chris on Zulip.

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

The optimizer-facing parameter registry now packs live MLP weights and GP
kernel hyperparameters into one named vector and unpacks updates back into the
original objects. Its test checks exact block ranges, a hand-known packed
vector, round-trip mutation, and duplicate-name refusal. Full-vector
JVP/VJP/HVP routing is now exposed by `fortml_parameter_products`; the live
MLP path has all three products and the fitted-GP mean path has value/JVP/VJP.
The product contract treats inputs as fixed data for optimizer updates, while
model-specific input derivatives remain on the MLP/kernel/GP APIs.

The registry now has a first-class fitted-GP adapter for kernel and
log-noise-variance parameters. Repeated parameter updates refactor the
Cholesky state safely before prediction products; the independent product
test caught and fixed the previously latent re-fit allocation error in
`gp_refactor`.

The explicit MLP now exposes a scalar-output-cotangent HVP over joint parameter
and input directions. Its finite-difference VJP oracle passes with gfortran and
nvfortran 26.5. The FortAD-generated scalar MLP fixture now passes independent
value, finite-difference VJP, finite-difference HVP, and Hessian-symmetry
checks with gfortran. The downstream `nvfortran` gate is open because
`nvfortran` 26.5 currently ICEs in FortAD's source transformer before the
FortML test can link; this does not invalidate the generated-code oracle.

A scalar RBF formula fixture now generates JVP, VJP, and HVP code through
FortAD and checks it against a hand-derived gradient, converged central
differences, the directional adjoint identity, and Hessian symmetry. It is
also the source for `gen_rbf_products`, which emits the checked product module
consumed by `kernel_t`.

The shared FortSym kernel IR now also has a checked Fortran consumer generator
for the RBF primal leaf. `fortml_kernels` uses the generated leaf for scalar
values, covariance matrices, matrix JVP values, input derivatives, and the
log-variance VJP contribution. `test_kernels` compares those paths with an
independent pairwise RBF formula and finite-difference/adjoint product checks.
Generated FortAD parameter JVP/VJP/HVP dispatch through `kernel_t` is now
covered for RBF. The fitted GP value/JVP/VJP path reaches those generated
kernel products, and the new `gp_predict_hvp` path differentiates the
Cholesky solve and reverse cotangents for the full packed RBF hyperparameter
  vector. Its independent central-difference oracle passes in
  `test_parameter_products`; the LML gradient HVP is also checked in
  `test_gaussian_process`. The generated scalar Matérn HVPs and analytic
  white-noise/elementary/composite rules are now covered by the kernel-product
  oracle. Matérn derivative-observation covariance is checked against a
  hand-derived dense Matern-3/2 solve; Matérn-1/2 coincident derivatives and
  all white-noise derivative observations refuse explicitly.

`fortml_bnn` provides the first Bayesian neural network. It puts a factorized
zero-mean Gaussian prior on every network weight, a factorized Gaussian
variational posterior packed as `[mu, log_sigma]`, and a fixed Monte Carlo
table drawn once from the counter-based `fortnum_rng` stream, so the ELBO and
all its products are deterministic functions of the variational vector. The
Gaussian KL term is checked against the textbook formula, against a
hand-evaluated constant case (`mu = 1/2`, `sigma = 2`, prior variance 4 gives
`1/32` per parameter), and against central differences of the KL value. The
ELBO gradient, directional derivative, and Hessian product over the complete
packed vector are checked against central finite differences of the ELBO value
and of the ELBO gradient, the JVP is checked against the adjoint identity, and
the Hessian is checked for symmetry. Determinism and seed sensitivity are
checked directly. `test_bnn` passes with gfortran and `nvfortran` 26.5. The
likelihood variance is fixed data here; learned likelihood parameters and
minibatch scaling belong to the variational-inference contract below.

`fortml_variational` holds the reusable variational-inference contract.
`gaussian_family_t` is a full or diagonal Gaussian posterior over one latent
block - network weights, inducing values, or any other block a model integrates
out - packed as a mean plus a lower-triangular factor with a log diagonal, so
the vector is unconstrained and goes straight to `fortopt`. Deterministic
parameters such as inducing-point locations, likelihood parameters, and kernel
hyperparameters travel beside it as `extra`, and the model supplies its log
likelihood and gradients through one procedure interface. Minibatch scaling is
an explicit factor on the likelihood term only.

The Monte Carlo table is drawn once from a seed, then centred and whitened, so
a log likelihood that is quadratic in the latent block is integrated exactly.
That makes the conjugate oracle exact rather than approximate: `test_variational`
optimizes the ELBO of a Bayesian linear model with `fortopt_adam` and recovers
the analytic posterior mean and covariance, formed independently by inverting
the 2x2 posterior precision, to better than 1e-5. The same test checks the ELBO
decomposition against separately computed terms, that minibatch scaling never
touches the KL, that KL is zero at the prior and never negative over a sweep of
parameters, that the draw table has exactly zero empirical mean and identity
empirical covariance, and the full gradient against central finite differences
of the ELBO. It passes with gfortran and `nvfortran` 26.5. Natural-gradient
updates and non-Gaussian likelihoods are not covered here.

Four upstream defects surfaced while wiring these items and are fixed on `main`
of their own repositories. `fo` dropped any source its front end could not
parse, which silently removed `fortad_opt` and two of `fo`'s own modules from
the build graph and produced missing-module errors in unrelated files; the scan
now recovers such a unit with the line scanner (`fo` commit `f1a8e56`). `fortad`
read only the first physical line of a procedure signature, so a wrapped
dummy-argument list lowered to a procedure with no parameters at all, which
made every forward-over-reverse pass over a wide adjoint emit an argument-less
HVP; signatures are now joined across continuation lines and covered by
`test_wide_signature_oracle` (`fortad` commit `8666fb0`). That fix restores
`test_fortad_mlp_products`. The real cause of the `fo` scan failures was a
stale dependency: `fo` bootstraps a git dependency once and never refreshes it,
so it had been running a months-old FortFront. `fo update` now drops the cached
clone, the fpm cache record, and the compiled dependency objects, and a build
re-fetches any dependency whose sources are missing (`fo` commits `7f62878`
and `f36c429`). FortFront itself rejected several legal constructs because
Fortran keywords are not reserved words: a call on an object named `operator`,
a declaration whose first entity is named `error` or `data`, and an assignment
to a variable or array element named `file`, `pure` or `precision`. All three
are fixed and covered by `test_call_on_keyword_named_object` (`fortfront`
commits `5618a1e8`, `147aaceb`, `346813d3`). Two slow FortFront suites and one
FortSym suite were renamed `*_slow`, and `fo` now gives a slow-marked test its
own timeout budget instead of holding it to the fast one (`fo` commit
`f36c429`).

`fortml_vae` composes two explicit MLPs into a variational autoencoder with a
diagonal Gaussian posterior, a seeded reparameterized draw, and a
fixed-variance Gaussian likelihood. Its gradient is one decoder VJP and one
encoder VJP per batch: the decoder's input gradient is the cotangent the
reparameterization carries back, where the analytic KL gradient is added.
`fortml_rnn` adds the sequence-batched vanilla recurrent network with an exact
backpropagation-through-time reverse scan over a leading time axis.

`test_vae_rnn` checks the VAE KL against the analytic diagonal-Gaussian formula
evaluated independently from the encoder output, the complete ELBO gradient
against central finite differences, and seed determinism; it checks the RNN
against a hand-rolled two-step forward reference and its BPTT gradient against
central finite differences. Both pass with gfortran and `nvfortran` 26.5.
Gated recurrent cells, checkpointed long sequences, Bernoulli likelihoods, and
higher-order products for these two models are not part of this item.

`fortml_sparse_gp` adds the inducing-point variational GP. With a Gaussian
likelihood the expected log likelihood is closed form, so the ELBO needs no
sampling. Its oracle is an identity rather than a tolerance: place the inducing
inputs on the data, set `q(u)` to the exact posterior over `f`, and the ELBO
equals the exact log marginal likelihood computed from an independent dense
Cholesky in the test. Away from that setting the bound must stay below the
exact evidence, its decomposition must be consistent, and the KL must be
non-negative; the collapsed predictive marginals must match the exact posterior
mean and variance. Learned inducing locations and non-Gaussian likelihoods are
not part of this item.

`fortml_multi_output_gp` adds the correlated multi-output path through the
intrinsic coregionalization model, `B (x) K` with `B = W W^T + diag(kappa)`. Two
independent identities check it: with `W = 0` the outputs decouple and each
posterior mean must equal a separate single-output fit, and with a coupled `B`
the whole joint system is solved densely in the test and compared entry by
entry, which also catches a Kronecker index-order mistake.

`fortml_inference_policy` is the policy layer above these operators. A caller
declares the structure that holds - tensor grid, compact support, banded Markov
precision - plus size and any inducing budget, and the policy returns the
solver together with the reason. It refuses rather than guesses: two declared
structures, grid extents that do not multiply out to the sample count, a band
as wide as the matrix, or an inducing budget at or above the sample count are
all errors.

The structured lane now has its three missing pieces. `fortml_multilevel_grid`
builds the tensor-grid hierarchy by halving each dimension, and moves values
between levels by separable linear interpolation; `restrict` is the exact
transpose of `prolong`, which is what keeps a two-level correction symmetric
enough to precondition a Krylov solve. `structured_gp_operator_t` gained
separable derivative products: swapping one factor for its derivative turns the
same contraction into the derivative of the covariance along that dimension, so
a derivative product costs one ordinary tensor pass. `fortml_banded_precision`
adds the Gaussian Markov path - a banded precision with banded Cholesky,
solves, and log determinants in both directions.

`test_structured_multilevel` checks that prolongation reproduces a linear
function exactly on the fine grid, that the adjoint identity
`<P c, f> = <c, R f>` holds, and that both derivative products match the dense
Kronecker product assembled independently in the test.
`test_banded_precision` uses the exact Markov identity as its oracle: the
tridiagonal Ornstein-Uhlenbeck precision is the exact inverse of the dense
exponential covariance, so the band times that covariance must be the identity,
a banded solve must reproduce the dense covariance product, and the log
determinant must match a dense Cholesky. Both pass with gfortran and
`nvfortran` 26.5. A device-resident Toeplitz path and matched scaling evidence
remain open.

`fortml_lanczos` adds the two matrix-free spectral estimators over any
`linear_operator_t`. The log determinant is stochastic Lanczos quadrature:
seeded Rademacher probes, Lanczos with full reorthogonalization, and Gauss
quadrature through the eigendecomposition of the tridiagonal, so an estimate is
a reproducible function of its seed. The predictive variance is the LOVE form,
running Lanczos from the cross-covariance itself so that `k_*^T A^{-1} k_*`
becomes one tridiagonal solve.

`test_lanczos` checks the log determinant against a hand-written Cholesky of
the dense covariance, within 5 percent for 64 probes, for two independent
seeds, and checks that the same seed reproduces exactly while a different seed
does not. The full-rank predictive variance is checked against a
partial-pivoting LU solve of the same system to 1e-9, which is an equality
rather than a sampling tolerance; a truncated run is checked to stay within
`[0, prior]`. Matched scaling evidence against GPyTorch for these two
estimators remains open.

The generic `kernel_operator_t` now has the same preconditioner contract as
the specialized RBF operator, built from the lowered kernel program rather than
from RBF parameters, so it covers every built-in kernel tree and any validated
user formula. `solve_cg_multi_block` and `solve_cg_multi_nystrom` run one
independent PCG recurrence per right-hand side over the persistent workspace
that `enter_data(status, n_rhs)` already owns, with one batched kernel product
per iteration and a recomputed true residual before any column is accepted as
converged. `test_kernel_operator` checks both solves against independent dense
multi-RHS solves, and checks the preconditioners themselves rather than only
their effect: the block factors must reproduce the shifted kernel block
exactly, and the Nystrom factor must be the Cholesky of its own normal matrix.
A CG answer alone cannot catch a wrong preconditioner, only a slower one.
Device-resident preconditioned solves and matched scaling evidence remain open.

`fortml_kernel_formula` closes the user-formula lowering boundary. A user
kernel is supplied as data, not as code: a short postfix program over a fixed
opcode set, which `kernel_operator_t` copies into the same static program it
already uses for built-in kernel trees, so no device loop ever calls back into
user code. Every opcode is a total function of its operands - the only value
sources are the squared distance, the distance, the inner product and a
constant; the only operations are add, subtract, multiply, negate, exponential
and division by a validated non-zero constant - and `push_distance` is a
primitive rather than a general square root, so no operand can reach `sqrt`
negative. Validation additionally proves the stack never underflows, never
exceeds its fixed depth, and ends holding exactly one value.

`static_lowering_eligible()` is the refusal boundary: `make_user_kernel` and
the operator both refuse a formula that has not validated, including one edited
after validating. `test_kernel_formula` checks an RBF-spelling formula against
the built-in RBF leaf, checks a non-built-in formula through lowered `matvec`
and `matmat` against a direct pairwise loop, checks a user leaf composed with a
built-in constant kernel, and checks each refusal case by name. It passes with
gfortran and `nvfortran` 26.5. Parameter products for user formulas are not
part of this item: only the log-variance factor is a parameter, and formula
constants are fixed data.

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

The generic CUDA postfix plan now consumes a FortSym-generated RBF scalar leaf
from the shared IR emitter. Its full resident matvec/matmat plan still passes
the independent dense oracle; a 2026-08-06 nvfortran 26.5 smoke run at 1,024
samples and 8 features measured 0.317 ms per resident operation on an RTX
5060 Ti (three repetitions). This is integration evidence, not a new
GPyTorch/KeOps comparison or scaling claim.

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

The standalone operation profiler and GPU benchmark now compile the
FortAD-generated RBF product module and FortSym-generated primal leaf before
`fortml_kernels`. This keeps direct nvfortran CPU/GPU profiling valid after
generated-code integration; the full fpm nvfortran graph remains separately
blocked by the known FortAD 26.5 ICE.

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
