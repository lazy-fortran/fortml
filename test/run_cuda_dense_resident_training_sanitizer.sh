#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if ! command -v compute-sanitizer >/dev/null 2>&1; then
    echo "test_cuda_dense_resident_training_sanitizer: compute-sanitizer unavailable; skipped"
    exit 0
fi
if ! command -v nvcc >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
    echo "test_cuda_dense_resident_training_sanitizer: CUDA device/compiler unavailable; skipped"
    exit 0
fi
build_dir=$(mktemp -d /mnt/storage/fortml-cuda-dense-training-sanitizer.XXXXXX)
trap 'rm -rf "$build_dir"' EXIT
nvcc ${NVCCFLAGS:--O1 -lineinfo -arch=native} -std=c++17 \
    "$repo_dir/src/mlp/fortml_cuda_dense.cu" \
    "$repo_dir/test/test_cuda_dense_resident_training.cu" \
    -o "$build_dir/test_cuda_dense_resident_training"
compute-sanitizer --tool memcheck --error-exitcode 1 \
    "$build_dir/test_cuda_dense_resident_training"
