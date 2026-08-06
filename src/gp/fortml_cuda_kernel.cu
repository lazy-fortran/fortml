#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>

#include "fortml_generated_rbf_leaf.cu"

namespace {

constexpr int kTileSize = 128;
constexpr int kRowsPerBlock = 4;
constexpr int kMaxRhs = 8;
constexpr int kMaxProgram = 64;
constexpr int kMaxFeatures = 64;

// These values are the public fortml_kernels postfix ABI. Keeping the mapping
// in this one backend file means the Fortran kernel semantics do not acquire a
// CUDA dialect switch.
constexpr int kRbf = 1;
constexpr int kMatern12 = 2;
constexpr int kMatern32 = 3;
constexpr int kMatern52 = 4;
constexpr int kLinear = 5;
constexpr int kConstant = 6;
constexpr int kWhiteNoise = 7;
constexpr int kSum = 8;
constexpr int kProduct = 9;

struct KernelPlan {
  double* points = nullptr;
  int* program_kind = nullptr;
  double* program_variance = nullptr;
  double* program_lengthscale = nullptr;
  int n_samples = 0;
  int n_features = 0;
  int program_size = 0;
};

__device__ inline double evaluate_program(
    double distance, double linear_value, const int* program_kind,
    const double* program_variance, const double* program_lengthscale,
    int program_size) {
  double stack[kMaxProgram];
  int stack_top = 0;
  for (int instruction = 0; instruction < program_size; ++instruction) {
    const int kind = program_kind[instruction];
    const double variance = program_variance[instruction];
    const double lengthscale = program_lengthscale[instruction];
    switch (kind) {
      case kRbf:
        {
          double value = 0.0;
          fortml_generated_rbf_leaf(variance, distance, lengthscale, &value);
          stack[stack_top++] = value;
        }
        break;
      case kMatern12: {
        const double r = sqrt(distance) / lengthscale;
        stack[stack_top++] = variance * exp(-r);
        break;
      }
      case kMatern32: {
        const double a = sqrt(3.0);
        const double r = sqrt(distance) / lengthscale;
        const double exponential = exp(-a * r);
        stack[stack_top++] = variance * (1.0 + a * r) * exponential;
        break;
      }
      case kMatern52: {
        const double a = sqrt(5.0);
        const double r = sqrt(distance) / lengthscale;
        const double exponential = exp(-a * r);
        stack[stack_top++] = variance *
            (1.0 + a * r + 5.0 * r * r / 3.0) * exponential;
        break;
      }
      case kLinear:
        stack[stack_top++] = variance * linear_value;
        break;
      case kConstant:
        stack[stack_top++] = variance;
        break;
      case kWhiteNoise:
        stack[stack_top++] = variance * (distance == 0.0 ? 1.0 : 0.0);
        break;
      case kSum:
        stack[stack_top - 2] += stack[stack_top - 1];
        --stack_top;
        break;
      case kProduct:
        stack[stack_top - 2] *= stack[stack_top - 1];
        --stack_top;
        break;
    }
  }
  return stack[0];
}

__global__ void kernel_matvec(
    const double* __restrict__ points, const double* __restrict__ input,
    double* __restrict__ output, int n_samples, int n_features,
    const int* __restrict__ program_kind,
    const double* __restrict__ program_variance,
    const double* __restrict__ program_lengthscale, int program_size,
    double diagonal_shift) {
  extern __shared__ double shared[];
  double* tile_points = shared;
  double* tile_input = tile_points + n_features * kTileSize;
  double* row_points = tile_input + kTileSize;

  const int lane = threadIdx.x & 31;
  const int row_slot = threadIdx.x >> 5;
  const int row = blockIdx.x * kRowsPerBlock + row_slot;

  for (int linear = threadIdx.x; linear < kRowsPerBlock * n_features;
       linear += blockDim.x) {
    const int feature = linear % n_features;
    const int local_row = linear / n_features;
    const int source_row = blockIdx.x * kRowsPerBlock + local_row;
    row_points[linear] = source_row < n_samples
        ? points[feature * n_samples + source_row]
        : 0.0;
  }
  __syncthreads();

  double accumulated = row < n_samples
      ? diagonal_shift * input[row]
      : 0.0;
  for (int first_neighbor = 0; first_neighbor < n_samples;
       first_neighbor += kTileSize) {
    if (threadIdx.x < kTileSize) {
      const int neighbor = first_neighbor + threadIdx.x;
      tile_input[threadIdx.x] = neighbor < n_samples ? input[neighbor] : 0.0;
      for (int feature = 0; feature < n_features; ++feature) {
        tile_points[feature * kTileSize + threadIdx.x] = neighbor < n_samples
            ? points[feature * n_samples + neighbor]
            : 0.0;
      }
    }
    __syncthreads();

    double partial = 0.0;
    if (row < n_samples) {
      const int tile_count = min(kTileSize, n_samples - first_neighbor);
      for (int local_neighbor = lane; local_neighbor < tile_count;
           local_neighbor += 32) {
        double distance = 0.0;
        double linear_value = 0.0;
        for (int feature = 0; feature < n_features; ++feature) {
          const double difference =
              row_points[row_slot * n_features + feature] -
              tile_points[feature * kTileSize + local_neighbor];
          distance += difference * difference;
          linear_value +=
              row_points[row_slot * n_features + feature] *
              tile_points[feature * kTileSize + local_neighbor];
        }
        const double kernel_value = evaluate_program(
            distance, linear_value, program_kind, program_variance,
            program_lengthscale, program_size);
        partial += kernel_value * tile_input[local_neighbor];
      }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
      partial += __shfl_down_sync(0xffffffff, partial, offset);
    }
    if (lane == 0) accumulated += partial;
    __syncthreads();
  }
  if (lane == 0 && row < n_samples) output[row] = accumulated;
}

__global__ void kernel_matmat(
    const double* __restrict__ points, const double* __restrict__ input,
    double* __restrict__ output, int n_samples, int n_features, int n_rhs,
    const int* __restrict__ program_kind,
    const double* __restrict__ program_variance,
    const double* __restrict__ program_lengthscale, int program_size,
    double diagonal_shift) {
  extern __shared__ double shared[];
  double* tile_points = shared;
  double* tile_input = tile_points + n_features * kTileSize;
  double* row_points = tile_input + kMaxRhs * kTileSize;

  const int lane = threadIdx.x & 31;
  const int row_slot = threadIdx.x >> 5;
  const int row = blockIdx.x * kRowsPerBlock + row_slot;
  double accumulated[kMaxRhs] = {};

  for (int linear = threadIdx.x; linear < kRowsPerBlock * n_features;
       linear += blockDim.x) {
    const int feature = linear % n_features;
    const int local_row = linear / n_features;
    const int source_row = blockIdx.x * kRowsPerBlock + local_row;
    row_points[linear] = source_row < n_samples
        ? points[feature * n_samples + source_row]
        : 0.0;
  }
  __syncthreads();

  if (row < n_samples) {
    for (int rhs = 0; rhs < n_rhs; ++rhs) {
      accumulated[rhs] = diagonal_shift * input[rhs * n_samples + row];
    }
  }
  for (int first_neighbor = 0; first_neighbor < n_samples;
       first_neighbor += kTileSize) {
    if (threadIdx.x < kTileSize) {
      const int neighbor = first_neighbor + threadIdx.x;
      for (int rhs = 0; rhs < kMaxRhs; ++rhs) {
        tile_input[rhs * kTileSize + threadIdx.x] =
            neighbor < n_samples && rhs < n_rhs
                ? input[rhs * n_samples + neighbor]
                : 0.0;
      }
      for (int feature = 0; feature < n_features; ++feature) {
        tile_points[feature * kTileSize + threadIdx.x] = neighbor < n_samples
            ? points[feature * n_samples + neighbor]
            : 0.0;
      }
    }
    __syncthreads();

    double partial[kMaxRhs] = {};
    if (row < n_samples) {
      const int tile_count = min(kTileSize, n_samples - first_neighbor);
      for (int local_neighbor = lane; local_neighbor < tile_count;
           local_neighbor += 32) {
        double distance = 0.0;
        double linear_value = 0.0;
        for (int feature = 0; feature < n_features; ++feature) {
          const double difference =
              row_points[row_slot * n_features + feature] -
              tile_points[feature * kTileSize + local_neighbor];
          distance += difference * difference;
          linear_value +=
              row_points[row_slot * n_features + feature] *
              tile_points[feature * kTileSize + local_neighbor];
        }
        const double kernel_value = evaluate_program(
            distance, linear_value, program_kind, program_variance,
            program_lengthscale, program_size);
        for (int rhs = 0; rhs < kMaxRhs; ++rhs) {
          partial[rhs] += kernel_value *
              tile_input[rhs * kTileSize + local_neighbor];
        }
      }
    }
    for (int rhs = 0; rhs < kMaxRhs; ++rhs) {
      for (int offset = 16; offset > 0; offset >>= 1) {
        partial[rhs] += __shfl_down_sync(0xffffffff, partial[rhs], offset);
      }
      if (lane == 0) accumulated[rhs] += partial[rhs];
    }
    __syncthreads();
  }
  if (lane == 0 && row < n_samples) {
    for (int rhs = 0; rhs < n_rhs; ++rhs) {
      output[rhs * n_samples + row] = accumulated[rhs];
    }
  }
}

int validate_plan_inputs(const int* program_kind, const double* variance,
                         const double* lengthscale, int program_size) {
  if (program_size < 1 || program_size > kMaxProgram) return 0;
  int stack_top = 0;
  for (int i = 0; i < program_size; ++i) {
    switch (program_kind[i]) {
      case kRbf:
      case kMatern12:
      case kMatern32:
      case kMatern52:
      case kLinear:
      case kConstant:
      case kWhiteNoise:
        if (variance[i] <= 0.0 || lengthscale[i] <= 0.0) return 0;
        ++stack_top;
        break;
      case kSum:
      case kProduct:
        if (stack_top < 2) return 0;
        --stack_top;
        break;
      default:
        return 0;
    }
    if (stack_top > kMaxProgram) return 0;
  }
  return stack_top == 1;
}

int shared_bytes(const KernelPlan* plan, bool matmat) {
  const int rhs_slots = matmat ? kMaxRhs : 1;
  const std::size_t doubles =
      static_cast<std::size_t>(plan->n_features) * kTileSize +
      rhs_slots * kTileSize +
      static_cast<std::size_t>(plan->n_features) * kRowsPerBlock;
  if (doubles > 48 * 1024 / sizeof(double)) return -1;
  return static_cast<int>(doubles * sizeof(double));
}

void destroy_plan(KernelPlan* plan) {
  if (plan == nullptr) return;
  cudaFree(plan->points);
  cudaFree(plan->program_kind);
  cudaFree(plan->program_variance);
  cudaFree(plan->program_lengthscale);
  delete plan;
}

}  // namespace

extern "C" int fortml_cuda_kernel_available() { return 1; }

extern "C" void* fortml_cuda_kernel_plan_create(
    const double* points, const int* program_kind,
    const double* program_variance, const double* program_lengthscale,
    int n_samples, int n_features, int program_size, int* status) {
  if (status == nullptr || points == nullptr || program_kind == nullptr ||
      program_variance == nullptr || program_lengthscale == nullptr ||
      n_samples < 1 || n_features < 1 || n_features > kMaxFeatures ||
      !validate_plan_inputs(program_kind, program_variance,
                            program_lengthscale, program_size)) {
    if (status != nullptr) *status = static_cast<int>(cudaErrorInvalidValue);
    return nullptr;
  }

  KernelPlan* plan = new KernelPlan;
  plan->n_samples = n_samples;
  plan->n_features = n_features;
  plan->program_size = program_size;
  cudaError_t error = cudaMalloc(&plan->points,
                                 sizeof(double) * n_samples * n_features);
  if (error == cudaSuccess) {
    error = cudaMalloc(&plan->program_kind, sizeof(int) * program_size);
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&plan->program_variance, sizeof(double) * program_size);
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&plan->program_lengthscale,
                       sizeof(double) * program_size);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(plan->points, points,
                       sizeof(double) * n_samples * n_features,
                       cudaMemcpyHostToDevice);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(plan->program_kind, program_kind,
                       sizeof(int) * program_size, cudaMemcpyHostToDevice);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(plan->program_variance, program_variance,
                       sizeof(double) * program_size, cudaMemcpyHostToDevice);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(plan->program_lengthscale, program_lengthscale,
                       sizeof(double) * program_size, cudaMemcpyHostToDevice);
  }
  if (error != cudaSuccess) {
    destroy_plan(plan);
    *status = static_cast<int>(error);
    return nullptr;
  }
  *status = 0;
  return plan;
}

extern "C" int fortml_cuda_kernel_plan_destroy(void* opaque_plan) {
  destroy_plan(static_cast<KernelPlan*>(opaque_plan));
  return 0;
}

extern "C" int fortml_cuda_kernel_plan_matvec(
    void* opaque_plan, const double* input, double* output,
    double diagonal_shift) {
  KernelPlan* plan = static_cast<KernelPlan*>(opaque_plan);
  if (plan == nullptr || input == nullptr || output == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const int bytes = shared_bytes(plan, false);
  if (bytes < 0) return static_cast<int>(cudaErrorInvalidConfiguration);
  constexpr int block_size = kRowsPerBlock * 32;
  const int grid_size =
      (plan->n_samples + kRowsPerBlock - 1) / kRowsPerBlock;
  kernel_matvec<<<grid_size, block_size, bytes>>>(
      plan->points, input, output, plan->n_samples, plan->n_features,
      plan->program_kind, plan->program_variance, plan->program_lengthscale,
      plan->program_size, diagonal_shift);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) return static_cast<int>(error);
  error = cudaStreamSynchronize(0);
  return error == cudaSuccess ? 0 : static_cast<int>(error);
}

extern "C" int fortml_cuda_kernel_plan_matmat(
    void* opaque_plan, const double* input, double* output, int n_rhs,
    double diagonal_shift) {
  KernelPlan* plan = static_cast<KernelPlan*>(opaque_plan);
  if (plan == nullptr || input == nullptr || output == nullptr ||
      n_rhs < 1 || n_rhs > kMaxRhs) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const int bytes = shared_bytes(plan, true);
  if (bytes < 0) return static_cast<int>(cudaErrorInvalidConfiguration);
  constexpr int block_size = kRowsPerBlock * 32;
  const int grid_size =
      (plan->n_samples + kRowsPerBlock - 1) / kRowsPerBlock;
  kernel_matmat<<<grid_size, block_size, bytes>>>(
      plan->points, input, output, plan->n_samples, plan->n_features, n_rhs,
      plan->program_kind, plan->program_variance, plan->program_lengthscale,
      plan->program_size, diagonal_shift);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) return static_cast<int>(error);
  error = cudaStreamSynchronize(0);
  return error == cudaSuccess ? 0 : static_cast<int>(error);
}
