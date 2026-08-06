#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
fortnum_dir=$(cd "$repo_dir/../fortnum" && pwd)
fc=${FC:-gfortran}
flags=${FFLAGS:--O3 -march=native -fopenmp -fno-math-errno}
out=${OUT:-$repo_dir/.provenance/benchmarks/fortml_composite_operator_cpu.csv}
meta=${META:-${out%.csv}.meta}
build_dir=$(mktemp -d)
trap 'rm -rf "$build_dir"' EXIT

command -v "$fc" >/dev/null
if [[ "$(basename "$fc")" == "gfortran" ]]; then
    module_flag=(-J "$build_dir")
else
    module_flag=(-module "$build_dir")
fi
sources=(
    "$fortnum_dir/src/fortnum_kinds.f90"
    "$fortnum_dir/src/fortnum_status.f90"
    "$fortnum_dir/src/linalg/fortnum_krylov.f90"
    "$repo_dir/src/gp/fortml_kernels.f90"
    "$repo_dir/src/gp/fortml_linear_operator.f90"
    "$repo_dir/src/gp/fortml_kernel_operator.f90"
    "$repo_dir/app/fortml_bench_composite_operator.f90"
    "$repo_dir/src/gp/fortml_cuda_rbf_stub.f90"
)
mkdir -p "$(dirname "$out")"
"$fc" $flags "${module_flag[@]}" \
    -o "$build_dir/fortml_bench_composite_operator" "${sources[@]}"
result=$("$build_dir/fortml_bench_composite_operator" cpu \
    "${N_SAMPLES:-2048}" "${N_FEATURES:-8}" "${REPETITIONS:-12}")
compiler_version=$($fc --version 2>&1 | awk 'NF {print; exit}')
printf 'model,samples,features,outputs,repetitions,seconds_per_operation,compiler,flags\n' >"$out"
printf '%s,%s,%s\n' "$result" "$fc" "$flags" >>"$out"
{
    printf 'target=fortml_bench_composite_operator\n'
    printf 'compiler=%s\n' "$fc"
    printf 'compiler_version=%s\n' "$compiler_version"
    printf 'flags=%s\n' "$flags"
    printf 'correctness_oracle=direct_RBF_plus_constant_pairwise_sum\n'
} >"$meta"
cat "$out"
cat "$meta"
