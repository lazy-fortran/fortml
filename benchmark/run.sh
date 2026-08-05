#!/usr/bin/env bash
set -euo pipefail

fc=${FC:-gfortran}
flags=${FFLAGS:--O3 -march=native}
target=${TARGET:-fortml_bench_linear}
out=${OUT:-benchmark/results.csv}
meta=${META:-${out%.csv}.meta}
mode=${MODE:-cpu}

mkdir -p "$(dirname "$out")"
command -v "$fc" >/dev/null
compiler_version=$("$fc" --version 2>&1 | awk 'NF {print; exit}')
log=$(mktemp)
trap 'rm -f "$log"' EXIT
start=$SECONDS
run_args=()
if [[ "$target" == "fortml_bench_rbf_operator" ]]; then
    run_args=(-- "$mode" "${N_SAMPLES:-2048}" "${N_FEATURES:-8}" "${REPETITIONS:-12}")
fi
if ! FPM_FC="$fc" fpm run --target "$target" --profile release --flag "$flags" \
        "${run_args[@]}" >"$log" 2>&1; then
    cat "$log" >&2
    exit 1
fi
build_seconds=$((SECONDS - start))
case "$target" in
    fortml_bench_linear) row=$(grep '^linear_regression,' "$log") ;;
    fortml_bench_mlp) row=$(grep '^mlp,' "$log") ;;
    fortml_bench_gp) row=$(grep '^gp,' "$log") ;;
    fortml_bench_rbf_operator) row=$(grep '^rbf_operator,' "$log") ;;
    *) printf 'unknown benchmark target: %s\n' "$target" >&2; exit 1 ;;
esac
printf 'model,samples,features,outputs,repetitions,seconds_per_operation,compiler,flags\n' >"$out"
printf '%s,%s,%s\n' "$row" "$fc" "$flags" >>"$out"
{
    printf 'target=%s\n' "$target"
    printf 'compiler=%s\n' "$fc"
    printf 'compiler_version=%s\n' "$compiler_version"
    printf 'flags=%s\n' "$flags"
    printf 'mode=%s\n' "$mode"
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
