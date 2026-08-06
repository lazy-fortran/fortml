# fortml benchmark protocol

Benchmark programs are correctness-gated before timing. The
`fortml_bench_linear` program fits a deterministic 512-sample, 16-feature,
4-output workload. The `fortml_bench_mlp` program evaluates and reverse-
propagates through a deterministic 16-32-4 MLP on the same batch. Both report
CSV rows with model, samples, features, outputs, measured repetitions, and wall
time. The `fortml_bench_gp` program fits and predicts with a deterministic
128-sample, 4-feature, 2-output exact GP and checks the result against an
independent dense LU reference. Select a workload with
`TARGET=fortml_bench_mlp` or `TARGET=fortml_bench_gp`.

The runner writes a sidecar metadata file with target, compiler version, flags,
compiler/build duration, CUDA version, and GPU identity when available. Peak
memory and generated-code size remain separate benchmark fields to add before
the performance gate closes.

The CPU reference is a pinned GPyTorch/PyTorch program using the same data
layout and mathematical operation. The GPU reference uses the same dtype,
batching, and device. Results are compared only when the algorithms perform the
same work. The acceptance target is within 30% of the reference runtime.

The cluster lane is run with:

```text
FC=nvfortran ./benchmark/run.sh
```

The runner records `nvfortran --version`, CUDA version, GPU model, and compiler
flags. A missing CUDA or GPU report leaves the GPU gate open. Plots consume the
raw CSV and are published at a stable public URL before the result is reported
in the Zulip DM.

The focused `run_rbf_cg_multi.sh` driver accepts an optional `BLOCK_SIZE` for
the experimental RBF block-Jacobi path. Leave it unset for the matched
unpreconditioned comparison lane. Block timings are exploratory until their
iteration and runtime behavior are competitive.

The compact-support sparse product can be exercised with
`FC=nvfortran ./benchmark/run_sparse.sh`. It builds the CSC input through
`fortsparse`, checks the result against an independent row-wise Wendland C2
sum, and reports host, transfer-inclusive, and resident OpenACC modes for
float64 four-RHS products. The cross-engine comparison and scaling plot live
in `fortml-bench/results/sparse_compact_support.csv` and its companion report.

It also accepts `NYSTROM_RANK` for the experimental fused Woodbury path.
`BLOCK_SIZE` and `NYSTROM_RANK` are mutually exclusive.

## Linear-regression conditioning

`benchmark/linear_conditioning.py` runs a deterministic 16-sample,
three-feature family whose second feature approaches the first. The design
also includes an intercept and a quadratic feature. Each case is fit with the
production SVD path and checked independently with `mpmath.qr_solve` at 80
decimal digits. The driver rejects a run when the high-precision prediction or
the well-conditioned coefficient checks fail.

Run both compiler lanes with fresh build directories:

```text
python benchmark/linear_conditioning.py --compiler gfortran \
  --flags '-O3 -march=native' --output /tmp/linear-conditioning-gfortran.csv
python benchmark/linear_conditioning.py --compiler nvfortran \
  --flags '-O3 -mp=multicore' --output /tmp/linear-conditioning-nvfortran.csv
```

The committed record is
`benchmark/reference/linear_conditioning.csv`. It includes the 2-norm
condition estimate, coefficient and prediction errors, fit time, complete
build-and-run wall time, peak RSS, compiler version, flags, and generated
executable size. The benchmark is a host LAPACK conditioning gate. It makes no
GPU performance claim.
