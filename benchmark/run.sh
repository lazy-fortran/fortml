#!/usr/bin/env bash
set -euo pipefail

fc=${FC:-gfortran}
flags=${FFLAGS:--O3 -march=native}
out=${OUT:-benchmark/results.csv}
meta=${META:-${out%.csv}.meta}

mkdir -p "$(dirname "$out")"
command -v "$fc" >/dev/null
compiler_version=$("$fc" --version 2>&1 | awk 'NF {print; exit}')
log=$(mktemp)
trap 'rm -f "$log"' EXIT
start=$SECONDS
if ! FC="$fc" fpm run --profile release --flag "$flags" >"$log" 2>&1; then
    cat "$log" >&2
    exit 1
fi
build_seconds=$((SECONDS - start))
row=$(grep '^linear_regression,' "$log")
printf 'model,samples,features,outputs,repetitions,seconds_per_fit,compiler,flags\n' >"$out"
printf '%s,%s,%s\n' "$row" "$fc" "$flags" >>"$out"
{
    printf 'compiler=%s\n' "$fc"
    printf 'compiler_version=%s\n' "$compiler_version"
    printf 'flags=%s\n' "$flags"
    printf 'build_and_run_seconds=%s\n' "$build_seconds"
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
