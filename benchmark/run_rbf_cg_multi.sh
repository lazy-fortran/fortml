#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
fortnum_dir=$(cd "$repo_dir/../fortnum" && pwd)
fc=${FC:-nvfortran}
flags=${FFLAGS:--O3 -acc}
native_cuda=${FORTML_NATIVE_CUDA:-0}
out=${OUT:-$repo_dir/.provenance/benchmarks/fortml_rbf_cg_multi_nvfortran.csv}
meta=${META:-${out%.csv}.meta}
build_dir=$(mktemp -d)
trap 'rm -rf "$build_dir"' EXIT

command -v "$fc" >/dev/null
mkdir -p "$(dirname "$out")"
if [[ "$(basename "$fc")" == "nvfortran" ]]; then
    module_flag=(-module "$build_dir")
else
    module_flag=(-J "$build_dir")
fi
sources=(
    "$fortnum_dir/src/fortnum_kinds.f90"
    "$fortnum_dir/src/fortnum_status.f90"
    "$fortnum_dir/src/linalg/fortnum_krylov.f90"
    "$repo_dir/src/gp/fortml_kernels.f90"
    "$repo_dir/src/gp/fortml_linear_operator.f90"
    "$repo_dir/src/gp/fortml_kernel_operator.f90"
    "$repo_dir/app/fortml_bench_rbf_cg_multi.f90"
)
link_inputs=()
if [[ "$native_cuda" == "1" ]]; then
    if [[ "$(basename "$fc")" != "nvfortran" ]]; then
        echo "FORTML_NATIVE_CUDA requires nvfortran" >&2
        exit 1
    fi
    if [[ "$flags" != *-acc* ]]; then
        echo "FORTML_NATIVE_CUDA requires an OpenACC build" >&2
        exit 1
    fi
    command -v nvcc >/dev/null
    cuda_root=${CUDA_HOME:-/opt/cuda}
    nvcc_flags=${NVCCFLAGS:--O3 -arch=native}
    nvcc $nvcc_flags -c "$repo_dir/src/gp/fortml_cuda_rbf.cu" \
        -o "$build_dir/fortml_cuda_rbf.o"
    link_inputs=(
        "$build_dir/fortml_cuda_rbf.o"
        "-L$cuda_root/lib64"
        -lcudart
        -c++libs
    )
else
    sources+=("$repo_dir/src/gp/fortml_cuda_rbf_stub.f90")
fi
"$fc" $flags "${module_flag[@]}" -o "$build_dir/fortml_bench_rbf_cg_multi" \
    "${sources[@]}" "${link_inputs[@]}"
arguments=("${N_SAMPLES:-2048}" "${N_FEATURES:-8}" "${N_RHS:-4}" \
    "${REPETITIONS:-8}")
if [[ -n "${BLOCK_SIZE:-}" ]]; then
    arguments+=("$BLOCK_SIZE")
elif [[ -n "${NYSTROM_RANK:-}" ]]; then
    arguments+=("0")
fi
if [[ -n "${NYSTROM_RANK:-}" ]]; then
    arguments+=("$NYSTROM_RANK")
fi
row=$("$build_dir/fortml_bench_rbf_cg_multi" "${arguments[@]}")
compiler_version=$($fc --version 2>&1 | awk 'NF {print; exit}')
printf 'model,samples,features,rhs,repetitions,setup_seconds,seconds_per_solve,iterations,residual_norm,compiler,flags\n' >"$out"
printf '%s,%s,%s\n' "$row" "$fc" "$flags" >>"$out"
{
    printf 'target=fortml_bench_rbf_cg_multi\n'
    printf 'compiler=%s\n' "$fc"
    printf 'compiler_version=%s\n' "$compiler_version"
    printf 'flags=%s\n' "$flags"
    printf 'native_cuda_kernel=%s\n' "$native_cuda"
    printf 'block_size=%s\n' "${BLOCK_SIZE:-0}"
    printf 'nystrom_rank=%s\n' "${NYSTROM_RANK:-0}"
    printf 'workspace_residency=operator_owned_multi_rhs_krylov\n'
    printf 'correctness_check=converged_true_residual\n'
    if command -v nvidia-smi >/dev/null 2>&1; then
        printf 'gpu=%s\n' "$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | paste -sd ';' -)"
    else
        printf 'gpu=unavailable\n'
    fi
} >"$meta"
cat "$out"
cat "$meta"
