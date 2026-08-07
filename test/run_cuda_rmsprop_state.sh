#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if ! command -v nvcc >/dev/null 2>&1; then
    echo "test_cuda_rmsprop_state: CUDA compiler unavailable; skipped"
    exit 0
fi
if ! nvidia-smi >/dev/null 2>&1; then
    echo "test_cuda_rmsprop_state: CUDA device unavailable; skipped"
    exit 0
fi
build_dir=$(mktemp -d)
trap 'rm -rf "$build_dir"' EXIT
nvcc ${NVCCFLAGS:--O3 -arch=native} \
    "$repo_dir/src/mlp/fortml_cuda_rmsprop.cu" \
    "$repo_dir/test/test_cuda_rmsprop_state.cu" \
    -o "$build_dir/test_cuda_rmsprop_state"
"$build_dir/test_cuda_rmsprop_state"
