#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
fortnum_dir=$(cd "$repo_dir/../fortnum" && pwd)
if ! command -v nvfortran >/dev/null 2>&1 || ! command -v nvcc >/dev/null 2>&1; then
    echo "test_cuda_metric: NVIDIA toolchain unavailable; skipped"
    exit 0
fi
if ! nvidia-smi >/dev/null 2>&1; then
    echo "test_cuda_metric: CUDA device unavailable; skipped"
    exit 0
fi

build_dir=$(mktemp -d /mnt/storage/fortml-cuda-metric.XXXXXX)
trap 'rm -rf "$build_dir"' EXIT
cuda_root=${CUDA_HOME:-/opt/cuda}
nvcc ${NVCCFLAGS:--O3 -arch=native} -std=c++17 -c \
    "$repo_dir/src/validation/fortml_cuda_metrics.cu" \
    -o "$build_dir/fortml_cuda_metrics.o"
nvfortran ${FFLAGS:--O3} -module "$build_dir" \
    "$fortnum_dir/src/fortnum_kinds.f90" \
    "$fortnum_dir/src/fortnum_status.f90" \
    "$repo_dir/src/gp/fortml_cuda_kernel_stub.f90" \
    "$repo_dir/src/gp/fortml_cuda_rbf_stub.f90" \
    "$repo_dir/src/classification/fortml_cuda_knn_stub.f90" \
    "$repo_dir/src/fortml_device.f90" \
    "$repo_dir/src/validation/fortml_cuda_metrics.f90" \
    "$repo_dir/test/test_cuda_metric_contract.f90" \
    "$build_dir/fortml_cuda_metrics.o" \
    -L"$cuda_root/lib64" -lcudart -c++libs \
    -o "$build_dir/test_cuda_metric_contract"
"$build_dir/test_cuda_metric_contract"
