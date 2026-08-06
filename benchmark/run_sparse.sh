#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
fortnum_dir=$(cd "$repo_dir/../fortnum" && pwd)
fortsparse_dir=$(cd "$repo_dir/../fortsparse" && pwd)
fc=${FC:-gfortran}
flags=${FFLAGS:--O3}
build_dir=$(mktemp -d)
trap 'rm -rf "$build_dir"' EXIT
if [[ "$(basename "$fc")" == "nvfortran" ]]; then
    flags=${FFLAGS:--O3 -acc}
    module_flag=(-module "$build_dir")
else
    module_flag=(-J"$build_dir")
fi
out=${OUT:-$repo_dir/benchmark/sparse_operator.csv}
meta=${META:-${out%.csv}.meta}

command -v "$fc" >/dev/null
sources=(
    "$fortnum_dir/src/fortnum_kinds.f90"
    "$fortnum_dir/src/fortnum_status.f90"
    "$fortnum_dir/src/linalg/fortnum_krylov.f90"
    "$fortsparse_dir/src/fortsparse_kinds.f90"
    "$fortsparse_dir/src/fortsparse_status.f90"
    "$fortsparse_dir/src/fortsparse_csc.f90"
    "$repo_dir/src/gp/fortml_linear_operator.f90"
    "$repo_dir/src/gp/fortml_sparse_operator.f90"
    "$repo_dir/app/fortml_bench_sparse_operator.f90"
)
mkdir -p "$(dirname "$out")"
"$fc" $flags "${module_flag[@]}" -o "$build_dir/fortml_bench_sparse_operator" "${sources[@]}"

n=${N_SAMPLES:-4096}
radius=${RADIUS:-8}
rhs=${N_RHS:-4}
repetitions=${REPETITIONS:-40}
host=$("$build_dir/fortml_bench_sparse_operator" host "$n" "$radius" "$rhs" "$repetitions")
transfer=$("$build_dir/fortml_bench_sparse_operator" transfer "$n" "$radius" "$rhs" "$repetitions")
resident=$("$build_dir/fortml_bench_sparse_operator" resident "$n" "$radius" "$rhs" "$repetitions")
compiler_version=$($fc --version 2>&1 | awk 'NF {print; exit}')
printf 'model,samples,radius,rhs,mode,repetitions,nonzeros,seconds_per_operation,relative_error,compiler,flags\n' >"$out"
for row in "$host" "$transfer" "$resident"; do
    printf '%s,%s,%s\n' "$row" "$fc" "$flags" >>"$out"
done
{
    printf 'target=fortml_bench_sparse_operator\n'
    printf 'compiler=%s\n' "$fc"
    printf 'compiler_version=%s\n' "$compiler_version"
    printf 'flags=%s\n' "$flags"
    printf 'precision=float64\n'
    printf 'kernel=wendland_c2_compact_support\n'
    printf 'correctness_oracle=independent_rowwise_compact_support_sum\n'
    if command -v nvcc >/dev/null 2>&1; then
        printf 'cuda_version=%s\n' "$(nvcc --version | tail -n 1)"
    else
        printf 'cuda_version=unavailable\n'
    fi
    if command -v nvidia-smi >/dev/null 2>&1; then
        printf 'gpu=%s\n' "$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | paste -sd ';' -)"
    else
        printf 'gpu=unavailable\n'
    fi
} >"$meta"
cat "$out"
cat "$meta"
