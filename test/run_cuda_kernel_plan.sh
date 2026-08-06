#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if ! command -v nvcc >/dev/null 2>&1; then
    echo "CUDA compiler unavailable; CUDA plan oracle skipped"
    exit 0
fi

build_dir=$(mktemp -d)
trap 'rm -rf "$build_dir"' EXIT
nvcc_flags=${NVCCFLAGS:--O2 -arch=native}
nvcc $nvcc_flags -std=c++17 \
    "$repo_dir/src/gp/fortml_cuda_kernel.cu" \
    "$repo_dir/test/test_cuda_kernel_plan.cu" \
    -o "$build_dir/test_cuda_kernel_plan"
"$build_dir/test_cuda_kernel_plan"
