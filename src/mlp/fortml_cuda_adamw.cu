#include "fortml_cuda_adamw.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <climits>

namespace {

// The plan owns every optimizer and model array on one explicitly selected
// CUDA device.  A step accepts a device-resident gradient and has no hidden
// host fallback or state round-trip.
struct AdamwPlan {
  double* parameters = nullptr;
  double* first_moment = nullptr;
  double* second_moment = nullptr;
  int n_parameters = 0;
  int device_index = 0;
  int step_count = 0;
  double learning_rate = 0.0;
  double beta1 = 0.0;
  double beta2 = 0.0;
  double epsilon = 0.0;
  double weight_decay = 0.0;
  int* invalid = nullptr;
};

__device__ inline bool finite_value(double value) {
  return isfinite(value);
}

__global__ void adamw_step_kernel(
    double* __restrict__ parameters, double* __restrict__ first_moment,
    double* __restrict__ second_moment, const double* __restrict__ gradient,
    int n_parameters, int next_step, double learning_rate, double beta1,
    double beta2, double epsilon, double weight_decay,
    int* __restrict__ invalid) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n_parameters) return;

  const double g = gradient[index];
  const double old_parameter = parameters[index];
  const double old_first = first_moment[index];
  const double old_second = second_moment[index];
  if (!finite_value(g) || !finite_value(old_parameter) ||
      !finite_value(old_first) || !finite_value(old_second) ||
      old_second < 0.0) {
    atomicExch(invalid, 1);
    return;
  }

  const double next_first = beta1 * old_first + (1.0 - beta1) * g;
  const double next_second = beta2 * old_second +
      (1.0 - beta2) * g * g;
  const double bias1 = 1.0 - pow(beta1, static_cast<double>(next_step));
  const double bias2 = 1.0 - pow(beta2, static_cast<double>(next_step));
  if (!finite_value(next_first) || !finite_value(next_second) ||
      next_second < 0.0 || !finite_value(bias1) || bias1 <= 0.0 ||
      !finite_value(bias2) || bias2 <= 0.0) {
    atomicExch(invalid, 1);
    return;
  }

  const double first_hat = next_first / bias1;
  const double second_hat = next_second / bias2;
  const double denominator = sqrt(second_hat) + epsilon;
  const double decay_factor = 1.0 - learning_rate * weight_decay;
  const double next_parameter = decay_factor * old_parameter -
      learning_rate * first_hat / denominator;
  if (!finite_value(first_hat) || !finite_value(second_hat) ||
      !finite_value(denominator) || denominator <= 0.0 ||
      !finite_value(decay_factor) || !finite_value(next_parameter)) {
    atomicExch(invalid, 1);
    return;
  }

  first_moment[index] = next_first;
  second_moment[index] = next_second;
  parameters[index] = next_parameter;
}

bool valid_hyperparameters(int n_parameters, double learning_rate,
                           double beta1, double beta2, double epsilon,
                           double weight_decay, int device_index) {
  return n_parameters > 0 && device_index >= 0 &&
      std::isfinite(learning_rate) && learning_rate > 0.0 &&
      std::isfinite(beta1) && beta1 >= 0.0 && beta1 < 1.0 &&
      std::isfinite(beta2) && beta2 >= 0.0 && beta2 < 1.0 &&
      std::isfinite(epsilon) && epsilon > 0.0 &&
      std::isfinite(weight_decay) && weight_decay >= 0.0;
}

template <typename T>
cudaError_t copy_or_zero(T* destination, const T* source, std::size_t count) {
  if (source == nullptr)
    return cudaMemset(destination, 0, sizeof(T) * count);
  return cudaMemcpy(destination, source, sizeof(T) * count,
                    cudaMemcpyHostToDevice);
}

bool valid_initial_state(const double* parameters, const double* first_moment,
                         const double* second_moment, int n_parameters) {
  for (int i = 0; i < n_parameters; ++i) {
    if (!std::isfinite(parameters[i])) return false;
    if (first_moment != nullptr && !std::isfinite(first_moment[i]))
      return false;
    if (second_moment != nullptr &&
        (!std::isfinite(second_moment[i]) || second_moment[i] < 0.0))
      return false;
  }
  return true;
}

void destroy_plan(AdamwPlan* plan) {
  if (plan == nullptr) return;
  // The selected device is part of the plan contract.  cudaFree is therefore
  // always issued in that context, even when the caller changed its current
  // device between the last step and destruction.
  cudaSetDevice(plan->device_index);
  cudaFree(plan->parameters);
  cudaFree(plan->first_moment);
  cudaFree(plan->second_moment);
  cudaFree(plan->invalid);
  delete plan;
}

}  // namespace

extern "C" int fortml_cuda_adamw_available() {
  int count = 0;
  const cudaError_t error = cudaGetDeviceCount(&count);
  if (error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver)
    return 0;
  return error == cudaSuccess && count > 0 ? 1 : 0;
}

extern "C" int fortml_cuda_adamw_plan_create(
    const double* parameters, const double* first_moment,
    const double* second_moment, int n_parameters, double learning_rate,
    double beta1, double beta2, double epsilon, double weight_decay,
    int device_index, void** opaque_plan) {
  if (opaque_plan == nullptr) return static_cast<int>(cudaErrorInvalidValue);
  *opaque_plan = nullptr;
  if (parameters == nullptr || !valid_hyperparameters(
          n_parameters, learning_rate, beta1, beta2, epsilon, weight_decay,
          device_index) ||
      !valid_initial_state(parameters, first_moment, second_moment,
                           n_parameters))
    return static_cast<int>(cudaErrorInvalidValue);

  // Device selection is explicit and deterministic; no current-device
  // inference is performed.
  cudaError_t error = cudaSetDevice(device_index);
  if (error != cudaSuccess) return static_cast<int>(error);
  AdamwPlan* plan = new AdamwPlan;
  plan->n_parameters = n_parameters;
  plan->device_index = device_index;
  plan->learning_rate = learning_rate;
  plan->beta1 = beta1;
  plan->beta2 = beta2;
  plan->epsilon = epsilon;
  plan->weight_decay = weight_decay;

  const std::size_t count = static_cast<std::size_t>(n_parameters);
  error = cudaMalloc(&plan->parameters, sizeof(double) * count);
  if (error == cudaSuccess)
    error = cudaMalloc(&plan->first_moment, sizeof(double) * count);
  if (error == cudaSuccess)
    error = cudaMalloc(&plan->second_moment, sizeof(double) * count);
  if (error == cudaSuccess) error = cudaMalloc(&plan->invalid, sizeof(int));
  if (error == cudaSuccess)
    error = cudaMemcpy(plan->parameters, parameters, sizeof(double) * count,
                       cudaMemcpyHostToDevice);
  if (error == cudaSuccess)
    error = copy_or_zero(plan->first_moment, first_moment, count);
  if (error == cudaSuccess)
    error = copy_or_zero(plan->second_moment, second_moment, count);
  if (error == cudaSuccess) error = cudaMemset(plan->invalid, 0, sizeof(int));
  if (error != cudaSuccess) {
    destroy_plan(plan);
    return static_cast<int>(error);
  }
  *opaque_plan = plan;
  return 0;
}

extern "C" int fortml_cuda_adamw_plan_step(void* opaque_plan,
                                             const double* gradient) {
  AdamwPlan* plan = static_cast<AdamwPlan*>(opaque_plan);
  if (plan == nullptr || gradient == nullptr || plan->step_count == INT_MAX)
    return static_cast<int>(cudaErrorInvalidValue);
  cudaError_t error = cudaSetDevice(plan->device_index);
  if (error != cudaSuccess) return static_cast<int>(error);
  error = cudaMemset(plan->invalid, 0, sizeof(int));
  if (error != cudaSuccess) return static_cast<int>(error);
  constexpr int kThreads = 256;
  const int grid = (plan->n_parameters - 1) / kThreads + 1;
  adamw_step_kernel<<<grid, kThreads>>>(
      plan->parameters, plan->first_moment, plan->second_moment, gradient,
      plan->n_parameters, plan->step_count + 1, plan->learning_rate,
      plan->beta1, plan->beta2, plan->epsilon, plan->weight_decay,
      plan->invalid);
  error = cudaGetLastError();
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  if (error != cudaSuccess) return static_cast<int>(error);
  int invalid = 0;
  error = cudaMemcpy(&invalid, plan->invalid, sizeof(int),
                     cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) return static_cast<int>(error);
  if (invalid != 0) return static_cast<int>(cudaErrorInvalidValue);
  ++plan->step_count;
  return 0;
}

extern "C" int fortml_cuda_adamw_plan_download(
    void* opaque_plan, double* parameters, double* first_moment,
    double* second_moment, int* step_count) {
  AdamwPlan* plan = static_cast<AdamwPlan*>(opaque_plan);
  if (plan == nullptr || parameters == nullptr || first_moment == nullptr ||
      second_moment == nullptr || step_count == nullptr)
    return static_cast<int>(cudaErrorInvalidValue);
  cudaError_t error = cudaSetDevice(plan->device_index);
  const std::size_t count = static_cast<std::size_t>(plan->n_parameters);
  if (error == cudaSuccess)
    error = cudaMemcpy(parameters, plan->parameters, sizeof(double) * count,
                       cudaMemcpyDeviceToHost);
  if (error == cudaSuccess)
    error = cudaMemcpy(first_moment, plan->first_moment,
                       sizeof(double) * count, cudaMemcpyDeviceToHost);
  if (error == cudaSuccess)
    error = cudaMemcpy(second_moment, plan->second_moment,
                       sizeof(double) * count, cudaMemcpyDeviceToHost);
  if (error == cudaSuccess) *step_count = plan->step_count;
  return static_cast<int>(error);
}

extern "C" int fortml_cuda_adamw_plan_destroy(void* opaque_plan) {
  destroy_plan(static_cast<AdamwPlan*>(opaque_plan));
  return 0;
}
