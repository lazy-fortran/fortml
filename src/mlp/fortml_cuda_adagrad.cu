#include "fortml_cuda_adagrad.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <climits>
#include <new>

namespace {

struct AdagradPlan {
  double* parameters = nullptr;
  double* accumulated_square = nullptr;
  int n_parameters = 0;
  int device_index = 0;
  int step_count = 0;
  double learning_rate = 0.0;
  double epsilon = 0.0;
  int* invalid = nullptr;
};

__device__ inline bool finite_value(double value) { return isfinite(value); }

__global__ void adagrad_step_kernel(
    double* __restrict__ parameters,
    double* __restrict__ accumulated_square,
    const double* __restrict__ gradient, int n_parameters,
    double learning_rate, double epsilon, int* __restrict__ invalid) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n_parameters) return;

  const double old_parameter = parameters[index];
  const double old_accumulated = accumulated_square[index];
  const double g = gradient[index];
  if (!finite_value(old_parameter) || !finite_value(old_accumulated) ||
      old_accumulated < 0.0 || !finite_value(g)) {
    atomicExch(invalid, 1);
    return;
  }

  const double next_accumulated = old_accumulated + g * g;
  const double denominator = sqrt(next_accumulated) + epsilon;
  const double next_parameter =
      old_parameter - learning_rate * g / denominator;
  if (!finite_value(next_accumulated) || next_accumulated < 0.0 ||
      !finite_value(denominator) || denominator <= 0.0 ||
      !finite_value(next_parameter)) {
    atomicExch(invalid, 1);
    return;
  }

  accumulated_square[index] = next_accumulated;
  parameters[index] = next_parameter;
}

bool valid_hyperparameters(int n_parameters, double learning_rate,
                           double epsilon, int device_index) {
  return n_parameters > 0 && device_index >= 0 &&
      std::isfinite(learning_rate) && learning_rate > 0.0 &&
      std::isfinite(epsilon) && epsilon > 0.0;
}

bool valid_initial_state(const double* parameters,
                         const double* accumulated_square,
                         int n_parameters) {
  for (int i = 0; i < n_parameters; ++i) {
    if (!std::isfinite(parameters[i])) return false;
    if (accumulated_square != nullptr &&
        (!std::isfinite(accumulated_square[i]) ||
         accumulated_square[i] < 0.0))
      return false;
  }
  return true;
}

void destroy_plan(AdagradPlan* plan) {
  if (plan == nullptr) return;
  cudaSetDevice(plan->device_index);
  cudaFree(plan->parameters);
  cudaFree(plan->accumulated_square);
  cudaFree(plan->invalid);
  delete plan;
}

}  // namespace

extern "C" int fortml_cuda_adagrad_available() {
  int count = 0;
  const cudaError_t error = cudaGetDeviceCount(&count);
  if (error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver)
    return 0;
  return error == cudaSuccess && count > 0 ? 1 : 0;
}

extern "C" int fortml_cuda_adagrad_plan_create(
    const double* parameters, const double* accumulated_square,
    int n_parameters, double learning_rate, double epsilon, int device_index,
    void** opaque_plan) {
  if (opaque_plan == nullptr) return static_cast<int>(cudaErrorInvalidValue);
  *opaque_plan = nullptr;
  if (parameters == nullptr || !valid_hyperparameters(
          n_parameters, learning_rate, epsilon, device_index) ||
      !valid_initial_state(parameters, accumulated_square, n_parameters))
    return static_cast<int>(cudaErrorInvalidValue);

  cudaError_t error = cudaSetDevice(device_index);
  if (error != cudaSuccess) return static_cast<int>(error);
  AdagradPlan* plan = new (std::nothrow) AdagradPlan;
  if (plan == nullptr) return static_cast<int>(cudaErrorMemoryAllocation);
  plan->n_parameters = n_parameters;
  plan->device_index = device_index;
  plan->learning_rate = learning_rate;
  plan->epsilon = epsilon;

  const std::size_t count = static_cast<std::size_t>(n_parameters);
  error = cudaMalloc(&plan->parameters, sizeof(double) * count);
  if (error == cudaSuccess)
    error = cudaMalloc(&plan->accumulated_square, sizeof(double) * count);
  if (error == cudaSuccess) error = cudaMalloc(&plan->invalid, sizeof(int));
  if (error == cudaSuccess)
    error = cudaMemcpy(plan->parameters, parameters, sizeof(double) * count,
                       cudaMemcpyHostToDevice);
  if (error == cudaSuccess) {
    if (accumulated_square == nullptr)
      error = cudaMemset(plan->accumulated_square, 0, sizeof(double) * count);
    else
      error = cudaMemcpy(plan->accumulated_square, accumulated_square,
                         sizeof(double) * count, cudaMemcpyHostToDevice);
  }
  if (error == cudaSuccess) error = cudaMemset(plan->invalid, 0, sizeof(int));
  if (error != cudaSuccess) {
    destroy_plan(plan);
    return static_cast<int>(error);
  }
  *opaque_plan = plan;
  return 0;
}

extern "C" int fortml_cuda_adagrad_plan_step(void* opaque_plan,
                                               const double* gradient) {
  AdagradPlan* plan = static_cast<AdagradPlan*>(opaque_plan);
  if (plan == nullptr || gradient == nullptr || plan->step_count == INT_MAX)
    return static_cast<int>(cudaErrorInvalidValue);
  cudaError_t error = cudaSetDevice(plan->device_index);
  if (error != cudaSuccess) return static_cast<int>(error);
  error = cudaMemset(plan->invalid, 0, sizeof(int));
  if (error != cudaSuccess) return static_cast<int>(error);
  constexpr int kThreads = 256;
  const int grid = (plan->n_parameters + kThreads - 1) / kThreads;
  adagrad_step_kernel<<<grid, kThreads>>>(
      plan->parameters, plan->accumulated_square, gradient,
      plan->n_parameters, plan->learning_rate, plan->epsilon, plan->invalid);
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

extern "C" int fortml_cuda_adagrad_plan_download(
    void* opaque_plan, double* parameters, double* accumulated_square,
    int* step_count) {
  AdagradPlan* plan = static_cast<AdagradPlan*>(opaque_plan);
  if (plan == nullptr || parameters == nullptr || accumulated_square == nullptr ||
      step_count == nullptr)
    return static_cast<int>(cudaErrorInvalidValue);
  cudaError_t error = cudaSetDevice(plan->device_index);
  const std::size_t count = static_cast<std::size_t>(plan->n_parameters);
  if (error == cudaSuccess)
    error = cudaMemcpy(parameters, plan->parameters, sizeof(double) * count,
                       cudaMemcpyDeviceToHost);
  if (error == cudaSuccess)
    error = cudaMemcpy(accumulated_square, plan->accumulated_square,
                       sizeof(double) * count, cudaMemcpyDeviceToHost);
  if (error == cudaSuccess) *step_count = plan->step_count;
  return static_cast<int>(error);
}

extern "C" int fortml_cuda_adagrad_plan_destroy(void* opaque_plan) {
  destroy_plan(static_cast<AdagradPlan*>(opaque_plan));
  return 0;
}
