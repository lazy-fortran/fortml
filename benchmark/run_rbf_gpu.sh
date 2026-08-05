#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
fortnum_dir=$(cd "$repo_dir/../fortnum" && pwd)
fc=${FC:-nvfortran}
flags=${FFLAGS:--O3 -acc}
out=${OUT:-$repo_dir/.provenance/benchmarks/fortml_rbf_operator_nvfortran.csv}
meta=${META:-${out%.csv}.meta}
build_dir=$(mktemp -d)
trap 'rm -rf "$build_dir"' EXIT

command -v "$fc" >/dev/null
mkdir -p "$(dirname "$out")"
if ! "$fc" $flags -module "$build_dir" -o "$build_dir/fortml_bench_rbf_operator" \
        "$fortnum_dir/src/fortnum_kinds.f90" \
        "$fortnum_dir/src/fortnum_status.f90" \
        "$fortnum_dir/src/linalg/fortnum_krylov.f90" \
        "$repo_dir/src/gp/fortml_kernels.f90" \
        "$repo_dir/src/gp/fortml_linear_operator.f90" \
        "$repo_dir/src/gp/fortml_kernel_operator.f90" \
        "$repo_dir/app/fortml_bench_rbf_operator.f90"; then
    exit 1
fi

transfer=$("$build_dir/fortml_bench_rbf_operator" transfer \
    "${N_SAMPLES:-2048}" "${N_FEATURES:-8}" "${REPETITIONS:-12}")
resident=$("$build_dir/fortml_bench_rbf_operator" resident \
    "${N_SAMPLES:-2048}" "${N_FEATURES:-8}" "${REPETITIONS:-12}")
compiler_version=$($fc --version 2>&1 | awk 'NF {print; exit}')
printf 'model,samples,features,outputs,repetitions,seconds_per_operation,compiler,flags\n' >"$out"
printf '%s,%s,%s\n' "$transfer" "$fc" "$flags" >>"$out"
printf '%s,%s,%s\n' "$resident" "$fc" "$flags" >>"$out"
{
    printf 'target=fortml_bench_rbf_operator\n'
    printf 'compiler=%s\n' "$fc"
    printf 'compiler_version=%s\n' "$compiler_version"
    printf 'flags=%s\n' "$flags"
    printf 'correctness_oracle=direct_RBF_pairwise_sum\n'
    if command -v nvidia-smi >/dev/null 2>&1; then
        printf 'gpu=%s\n' "$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | paste -sd ';' -)"
    else
        printf 'gpu=unavailable\n'
    fi
} >"$meta"
cat "$out"
cat "$meta"
