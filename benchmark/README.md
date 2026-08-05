# fortml benchmark protocol

Benchmark programs are correctness-gated before timing. The initial
`fortml_bench_linear` program fits a deterministic 512-sample, 16-feature,
4-output workload and reports CSV rows with model, samples, features, outputs,
measured repetitions, and wall time. The runner writes a sidecar metadata file
with compiler version, flags, compiler/build duration, CUDA version, and GPU
identity when available. Peak memory and generated-code size remain separate
benchmark fields to add before the performance gate closes.

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
