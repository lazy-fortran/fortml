#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if ! command -v nvcc >/dev/null 2>&1; then
    echo "test_cuda_dense_plan: CUDA compiler unavailable; skipped"
    exit 0
fi
if ! nvidia-smi >/dev/null 2>&1; then
    echo "test_cuda_dense_plan: CUDA device unavailable; skipped"
    exit 0
fi
build_dir=$(mktemp -d /mnt/storage/fortml-cuda-dense-plan.XXXXXX)
trap 'rm -rf "$build_dir"' EXIT
nvcc ${NVCCFLAGS:--O3 -arch=native} -std=c++17 \
    "$repo_dir/src/mlp/fortml_cuda_dense.cu" \
    "$repo_dir/test/test_cuda_dense_plan.cu" \
    -o "$build_dir/test_cuda_dense_plan"
"$build_dir/test_cuda_dense_plan"
