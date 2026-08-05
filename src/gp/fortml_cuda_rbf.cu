#include <cuda_runtime.h>

#include <cmath>

namespace {

constexpr int kTileSize = 128;
constexpr int kFeatures = 8;
constexpr int kRowsPerBlock = 4;

__global__ void rbf_matvec_kernel(
    const double* __restrict__ points,
    const double* __restrict__ input,
    double* __restrict__ output,
    int n_samples,
    double variance,
    double inverse_scale,
    double diagonal_shift) {
  __shared__ double tile_points[kFeatures * kTileSize];
  __shared__ double tile_input[kTileSize];

  const int lane = threadIdx.x & 31;
  const int row_slot = threadIdx.x >> 5;
  const int row = blockIdx.x * kRowsPerBlock + row_slot;
  __shared__ double row_points[kRowsPerBlock * kFeatures];
  double accumulated = 0.0;

  if (lane < kFeatures) {
    if (row < n_samples) {
      row_points[row_slot * kFeatures + lane] =
          points[lane * n_samples + row];
    } else {
      row_points[row_slot * kFeatures + lane] = 0.0;
    }
  }
  __syncthreads();

  for (int first_neighbor = 0; first_neighbor < n_samples;
       first_neighbor += kTileSize) {
    if (threadIdx.x < kTileSize) {
      const int neighbor = first_neighbor + threadIdx.x;
      if (neighbor < n_samples) {
        tile_input[threadIdx.x] = input[neighbor];
        for (int feature = 0; feature < kFeatures; ++feature) {
          tile_points[feature * kTileSize + threadIdx.x] =
              points[feature * n_samples + neighbor];
        }
      } else {
        tile_input[threadIdx.x] = 0.0;
        for (int feature = 0; feature < kFeatures; ++feature) {
          tile_points[feature * kTileSize + threadIdx.x] = 0.0;
        }
      }
    }
    __syncthreads();

    double partial = 0.0;
    if (row < n_samples) {
      const int tile_count =
          (n_samples - first_neighbor < kTileSize)
              ? n_samples - first_neighbor
              : kTileSize;
      const double point_1 = row_points[row_slot * kFeatures];
      const double point_2 = row_points[row_slot * kFeatures + 1];
      const double point_3 = row_points[row_slot * kFeatures + 2];
      const double point_4 = row_points[row_slot * kFeatures + 3];
      const double point_5 = row_points[row_slot * kFeatures + 4];
      const double point_6 = row_points[row_slot * kFeatures + 5];
      const double point_7 = row_points[row_slot * kFeatures + 6];
      const double point_8 = row_points[row_slot * kFeatures + 7];
      for (int local_neighbor = lane; local_neighbor < tile_count;
           local_neighbor += 32) {
        const double difference_1 =
            point_1 - tile_points[local_neighbor];
        const double difference_2 =
            point_2 - tile_points[kTileSize + local_neighbor];
        const double difference_3 =
            point_3 - tile_points[2 * kTileSize + local_neighbor];
        const double difference_4 =
            point_4 - tile_points[3 * kTileSize + local_neighbor];
        const double difference_5 =
            point_5 - tile_points[4 * kTileSize + local_neighbor];
        const double difference_6 =
            point_6 - tile_points[5 * kTileSize + local_neighbor];
        const double difference_7 =
            point_7 - tile_points[6 * kTileSize + local_neighbor];
        const double difference_8 =
            point_8 - tile_points[7 * kTileSize + local_neighbor];
        const double distance =
            difference_1 * difference_1 + difference_2 * difference_2 +
            difference_3 * difference_3 + difference_4 * difference_4 +
            difference_5 * difference_5 + difference_6 * difference_6 +
            difference_7 * difference_7 + difference_8 * difference_8;
        partial += variance * exp(-inverse_scale * distance) *
                   tile_input[local_neighbor];
      }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
      partial += __shfl_down_sync(0xffffffff, partial, offset);
    }
    if (lane == 0) {
      accumulated += partial;
    }
    __syncthreads();
  }
  if (lane == 0 && row < n_samples) {
    output[row] = diagonal_shift * input[row] + accumulated;
  }
}

}  // namespace

extern "C" int fortml_cuda_rbf_available() { return 1; }

extern "C" int fortml_cuda_rbf_matvec(
    const double* points,
    const double* input,
    double* output,
    int n_samples,
    double variance,
    double inverse_scale,
    double diagonal_shift) {
  constexpr int block_size = kRowsPerBlock * 32;
  const int grid_size = (n_samples + kRowsPerBlock - 1) / kRowsPerBlock;
  rbf_matvec_kernel<<<grid_size, block_size>>>(
      points, input, output, n_samples, variance, inverse_scale,
      diagonal_shift);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  status = cudaStreamSynchronize(0);
  return status == cudaSuccess ? 0 : static_cast<int>(status);
}
