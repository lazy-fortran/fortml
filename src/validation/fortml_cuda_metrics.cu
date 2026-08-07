#include <cuda_runtime.h>

#include "fortml_cuda_mse_plan.h"

#include <cmath>
#include <cstddef>
#include <new>
#include <vector>

namespace {

constexpr int kThreads = 256;

struct MsePlan {
  double* target = nullptr;
  double* prediction = nullptr;
  double* sample_weight = nullptr;
  double* block_sum = nullptr;
  int n_samples = 0;
  int n_outputs = 0;
  int blocks = 0;
  double denominator = 0.0;
  int device_index = 0;
};

__global__ void weighted_mse_kernel(const double* target,
                                    const double* prediction,
                                    const double* sample_weight,
                                    double* block_sum, std::size_t count,
                                    int n_samples) {
  __shared__ double partial[kThreads];
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
      threadIdx.x;
  double value = 0.0;
  if (index < count) {
    const int row = static_cast<int>(index % static_cast<std::size_t>(n_samples));
    const double difference = target[index] - prediction[index];
    const double weight = sample_weight == nullptr ? 1.0 : sample_weight[row];
    value = weight * difference * difference;
  }
  partial[threadIdx.x] = value;
  __syncthreads();
  for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) block_sum[blockIdx.x] = partial[0];
}

bool finite_array(const double* values, std::size_t count) {
  for (std::size_t i = 0; i < count; ++i) {
    if (!std::isfinite(values[i])) return false;
  }
  return true;
}

}  // namespace

extern "C" int fortml_cuda_mse_available() {
  int count = 0;
  const cudaError_t error = cudaGetDeviceCount(&count);
  if (error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver) return 0;
  return error == cudaSuccess && count > 0 ? 1 : 0;
}

extern "C" int fortml_cuda_mean_squared_error(
    const double* target, const double* prediction, const double* sample_weight,
    double* value, int n_samples, int n_outputs, int device_index) {
  if (target == nullptr || prediction == nullptr || value == nullptr ||
      n_samples <= 0 || n_outputs <= 0 || device_index < 0) return 1;
  const std::size_t count = static_cast<std::size_t>(n_samples) *
      static_cast<std::size_t>(n_outputs);
  if (!finite_array(target, count) || !finite_array(prediction, count)) return 1;
  double denominator = static_cast<double>(n_samples);
  if (sample_weight != nullptr) {
    if (!finite_array(sample_weight, static_cast<std::size_t>(n_samples))) return 1;
    denominator = 0.0;
    for (int row = 0; row < n_samples; ++row) {
      if (sample_weight[row] < 0.0) return 1;
      denominator += sample_weight[row];
    }
    if (!(denominator > 0.0) || !std::isfinite(denominator)) return 1;
  }

  cudaError_t error = cudaSetDevice(device_index);
  if (error != cudaSuccess) return 2;
  double* device_target = nullptr;
  double* device_prediction = nullptr;
  double* device_weight = nullptr;
  double* device_block_sum = nullptr;
  const int blocks = static_cast<int>((count + kThreads - 1) / kThreads);
  std::vector<double> block_sum(static_cast<std::size_t>(blocks));
  error = cudaMalloc(&device_target, count * sizeof(double));
  if (error == cudaSuccess) error = cudaMalloc(&device_prediction, count * sizeof(double));
  if (error == cudaSuccess && sample_weight != nullptr)
    error = cudaMalloc(&device_weight, static_cast<std::size_t>(n_samples) * sizeof(double));
  if (error == cudaSuccess) error = cudaMalloc(&device_block_sum, block_sum.size() * sizeof(double));
  if (error == cudaSuccess) error = cudaMemcpy(device_target, target, count * sizeof(double), cudaMemcpyHostToDevice);
  if (error == cudaSuccess) error = cudaMemcpy(device_prediction, prediction, count * sizeof(double), cudaMemcpyHostToDevice);
  if (error == cudaSuccess && sample_weight != nullptr)
    error = cudaMemcpy(device_weight, sample_weight, static_cast<std::size_t>(n_samples) * sizeof(double), cudaMemcpyHostToDevice);
  if (error == cudaSuccess) {
    weighted_mse_kernel<<<blocks, kThreads>>>(device_target, device_prediction,
        device_weight, device_block_sum, count, n_samples);
    error = cudaGetLastError();
  }
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  if (error == cudaSuccess)
    error = cudaMemcpy(block_sum.data(), device_block_sum, block_sum.size() * sizeof(double), cudaMemcpyDeviceToHost);
  if (error == cudaSuccess) {
    double total = 0.0;
    for (double part : block_sum) total += part;
    *value = total / (denominator * static_cast<double>(n_outputs));
  }
  cudaFree(device_target);
  cudaFree(device_prediction);
  cudaFree(device_weight);
  cudaFree(device_block_sum);
  return error == cudaSuccess ? 0 : 2;
}

extern "C" int fortml_cuda_mse_plan_create(
    const double* target, const double* prediction, const double* sample_weight,
    int n_samples, int n_outputs, int device_index, void** opaque_plan) {
  if (opaque_plan == nullptr || target == nullptr || prediction == nullptr ||
      n_samples <= 0 || n_outputs <= 0 || device_index < 0) return 1;
  const std::size_t count = static_cast<std::size_t>(n_samples) *
      static_cast<std::size_t>(n_outputs);
  if (!finite_array(target, count) || !finite_array(prediction, count)) return 1;
  double denominator = static_cast<double>(n_samples);
  if (sample_weight != nullptr) {
    if (!finite_array(sample_weight, static_cast<std::size_t>(n_samples))) return 1;
    denominator = 0.0;
    for (int row = 0; row < n_samples; ++row) {
      if (sample_weight[row] < 0.0) return 1;
      denominator += sample_weight[row];
    }
    if (!(denominator > 0.0) || !std::isfinite(denominator)) return 1;
  }
  cudaError_t error = cudaSetDevice(device_index);
  if (error != cudaSuccess) return static_cast<int>(error);
  MsePlan* plan = new (std::nothrow) MsePlan();
  if (plan == nullptr) return 1;
  plan->n_samples = n_samples;
  plan->n_outputs = n_outputs;
  plan->blocks = static_cast<int>((count + kThreads - 1) / kThreads);
  plan->denominator = denominator;
  plan->device_index = device_index;
  error = cudaMalloc(&plan->target, count * sizeof(double));
  if (error == cudaSuccess) error = cudaMalloc(&plan->prediction, count * sizeof(double));
  if (error == cudaSuccess && sample_weight != nullptr)
    error = cudaMalloc(&plan->sample_weight, static_cast<std::size_t>(n_samples) * sizeof(double));
  if (error == cudaSuccess)
    error = cudaMalloc(&plan->block_sum, static_cast<std::size_t>(plan->blocks) * sizeof(double));
  if (error == cudaSuccess)
    error = cudaMemcpy(plan->target, target, count * sizeof(double), cudaMemcpyHostToDevice);
  if (error == cudaSuccess)
    error = cudaMemcpy(plan->prediction, prediction, count * sizeof(double), cudaMemcpyHostToDevice);
  if (error == cudaSuccess && sample_weight != nullptr)
    error = cudaMemcpy(plan->sample_weight, sample_weight,
                       static_cast<std::size_t>(n_samples) * sizeof(double),
                       cudaMemcpyHostToDevice);
  if (error != cudaSuccess) {
    cudaFree(plan->target);
    cudaFree(plan->prediction);
    cudaFree(plan->sample_weight);
    cudaFree(plan->block_sum);
    delete plan;
    return static_cast<int>(error);
  }
  *opaque_plan = plan;
  return 0;
}

extern "C" int fortml_cuda_mse_plan_execute(void* opaque_plan, double* value) {
  if (opaque_plan == nullptr || value == nullptr) return 1;
  MsePlan* plan = static_cast<MsePlan*>(opaque_plan);
  cudaError_t error = cudaSetDevice(plan->device_index);
  if (error != cudaSuccess) return static_cast<int>(error);
  const std::size_t count = static_cast<std::size_t>(plan->n_samples) *
      static_cast<std::size_t>(plan->n_outputs);
  weighted_mse_kernel<<<plan->blocks, kThreads>>>(
      plan->target, plan->prediction, plan->sample_weight, plan->block_sum,
      count, plan->n_samples);
  error = cudaGetLastError();
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  if (error != cudaSuccess) return static_cast<int>(error);
  std::vector<double> block_sum(static_cast<std::size_t>(plan->blocks));
  error = cudaMemcpy(block_sum.data(), plan->block_sum,
                     block_sum.size() * sizeof(double), cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) return static_cast<int>(error);
  double total = 0.0;
  for (double partial : block_sum) total += partial;
  *value = total / (plan->denominator * static_cast<double>(plan->n_outputs));
  return std::isfinite(*value) ? 0 : 1;
}

extern "C" int fortml_cuda_mse_plan_destroy(void* opaque_plan) {
  if (opaque_plan == nullptr) return 0;
  MsePlan* plan = static_cast<MsePlan*>(opaque_plan);
  cudaError_t error = cudaSetDevice(plan->device_index);
  if (error != cudaSuccess) return static_cast<int>(error);
  error = cudaFree(plan->target);
  if (error == cudaSuccess) error = cudaFree(plan->prediction);
  if (error == cudaSuccess) error = cudaFree(plan->sample_weight);
  if (error == cudaSuccess) error = cudaFree(plan->block_sum);
  delete plan;
  return static_cast<int>(error);
}
