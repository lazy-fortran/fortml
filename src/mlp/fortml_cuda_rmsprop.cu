#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <climits>

namespace {

// The plan owns all optimizer and model arrays on the selected CUDA device.
// `fortml_cuda_rmsprop_plan_step` accepts a device-resident gradient and never
// copies the model or optimizer state through the host between updates.
struct RmspropPlan {
  double* parameters = nullptr;
  double* square_average = nullptr;
  double* gradient_average = nullptr;
  double* momentum_buffer = nullptr;
  int n_parameters = 0;
  int device_index = 0;
  int step_count = 0;
  double learning_rate = 0.0;
  double decay = 0.0;
  double epsilon = 0.0;
  double momentum = 0.0;
  int centered = 0;
  int* invalid = nullptr;
};

__device__ inline bool finite_value(double value) {
  return isfinite(value);
}

__global__ void rmsprop_step_kernel(
    double* __restrict__ parameters, double* __restrict__ square_average,
    double* __restrict__ gradient_average,
    double* __restrict__ momentum_buffer, const double* __restrict__ gradient,
    int n_parameters, double learning_rate, double decay, double epsilon,
    double momentum, int centered, int* __restrict__ invalid) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n_parameters) return;

  const double g = gradient[index];
  const double old_square = square_average[index];
  const double old_mean = gradient_average[index];
  const double old_momentum = momentum_buffer[index];
  if (!finite_value(g) || !finite_value(parameters[index]) ||
      !finite_value(old_square) || old_square < 0.0 ||
      !finite_value(old_mean) || !finite_value(old_momentum)) {
    atomicExch(invalid, 1);
    return;
  }

  const double next_square = decay * old_square +
      (1.0 - decay) * g * g;
  double next_mean = old_mean;
  if (centered != 0) {
    next_mean = decay * old_mean + (1.0 - decay) * g;
  }
  if (!finite_value(next_square) || !finite_value(next_mean)) {
    atomicExch(invalid, 1);
    return;
  }

  double variance = next_square;
  if (centered != 0) {
    // Roundoff may make this mathematically non-negative quantity slightly
    // negative.  Match fortopt_rmsprop and clamp only that representation.
    variance = next_square - next_mean * next_mean;
    if (!finite_value(variance)) {
      atomicExch(invalid, 1);
      return;
    }
    variance = fmax(variance, 0.0);
  }
  const double denominator = sqrt(variance) + epsilon;
  double direction = g / denominator;
  double next_momentum = old_momentum;
  if (momentum > 0.0) {
    next_momentum = momentum * old_momentum + direction;
    direction = next_momentum;
  }
  const double next_parameter = parameters[index] - learning_rate * direction;
  if (!finite_value(next_momentum) || !finite_value(next_parameter)) {
    atomicExch(invalid, 1);
    return;
  }

  square_average[index] = next_square;
  gradient_average[index] = next_mean;
  if (momentum > 0.0) momentum_buffer[index] = next_momentum;
  parameters[index] = next_parameter;
}

bool valid_hyperparameters(int n_parameters, double learning_rate, double decay,
                           double epsilon, double momentum, int centered,
                           int device_index) {
  return n_parameters > 0 && device_index >= 0 &&
      std::isfinite(learning_rate) && learning_rate > 0.0 &&
      std::isfinite(decay) && decay >= 0.0 && decay < 1.0 &&
      std::isfinite(epsilon) && epsilon > 0.0 && std::isfinite(momentum) &&
      momentum >= 0.0 && momentum < 1.0 && (centered == 0 || centered == 1);
}

template <typename T>
cudaError_t copy_or_zero(T* destination, const T* source, std::size_t count) {
  if (source == nullptr)
    return cudaMemset(destination, 0, sizeof(T) * count);
  return cudaMemcpy(destination, source, sizeof(T) * count,
                    cudaMemcpyHostToDevice);
}

void destroy_plan(RmspropPlan* plan) {
  if (plan == nullptr) return;
  cudaFree(plan->parameters);
  cudaFree(plan->square_average);
  cudaFree(plan->gradient_average);
  cudaFree(plan->momentum_buffer);
  cudaFree(plan->invalid);
  delete plan;
}

}  // namespace

extern "C" int fortml_cuda_rmsprop_available() { return 1; }

extern "C" int fortml_cuda_rmsprop_plan_create(
    const double* parameters, const double* square_average,
    const double* gradient_average, const double* momentum_buffer,
    int n_parameters, double learning_rate, double decay, double epsilon,
    double momentum, int centered, int device_index, void** opaque_plan) {
  if (opaque_plan == nullptr) return static_cast<int>(cudaErrorInvalidValue);
  *opaque_plan = nullptr;
  if (parameters == nullptr || !valid_hyperparameters(
          n_parameters, learning_rate, decay, epsilon, momentum, centered,
          device_index)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  cudaError_t error = cudaSetDevice(device_index);
  if (error != cudaSuccess) return static_cast<int>(error);
  RmspropPlan* plan = new RmspropPlan;
  plan->n_parameters = n_parameters;
  plan->device_index = device_index;
  plan->learning_rate = learning_rate;
  plan->decay = decay;
  plan->epsilon = epsilon;
  plan->momentum = momentum;
  plan->centered = centered;

  const std::size_t count = static_cast<std::size_t>(n_parameters);
  error = cudaMalloc(&plan->parameters, sizeof(double) * count);
  if (error == cudaSuccess)
    error = cudaMalloc(&plan->square_average, sizeof(double) * count);
  if (error == cudaSuccess)
    error = cudaMalloc(&plan->gradient_average, sizeof(double) * count);
  if (error == cudaSuccess)
    error = cudaMalloc(&plan->momentum_buffer, sizeof(double) * count);
  if (error == cudaSuccess) error = cudaMalloc(&plan->invalid, sizeof(int));
  if (error == cudaSuccess)
    error = cudaMemcpy(plan->parameters, parameters, sizeof(double) * count,
                       cudaMemcpyHostToDevice);
  if (error == cudaSuccess)
    error = copy_or_zero(plan->square_average, square_average, count);
  if (error == cudaSuccess)
    error = copy_or_zero(plan->gradient_average, gradient_average, count);
  if (error == cudaSuccess)
    error = copy_or_zero(plan->momentum_buffer, momentum_buffer, count);
  if (error == cudaSuccess) error = cudaMemset(plan->invalid, 0, sizeof(int));
  if (error != cudaSuccess) {
    destroy_plan(plan);
    return static_cast<int>(error);
  }
  *opaque_plan = plan;
  return 0;
}

extern "C" int fortml_cuda_rmsprop_plan_step(void* opaque_plan,
                                               const double* gradient) {
  RmspropPlan* plan = static_cast<RmspropPlan*>(opaque_plan);
  if (plan == nullptr || gradient == nullptr || plan->step_count == INT_MAX)
    return static_cast<int>(cudaErrorInvalidValue);
  cudaError_t error = cudaSetDevice(plan->device_index);
  if (error != cudaSuccess) return static_cast<int>(error);
  error = cudaMemset(plan->invalid, 0, sizeof(int));
  if (error != cudaSuccess) return static_cast<int>(error);
  constexpr int kThreads = 256;
  const int grid = (plan->n_parameters + kThreads - 1) / kThreads;
  rmsprop_step_kernel<<<grid, kThreads>>>(
      plan->parameters, plan->square_average, plan->gradient_average,
      plan->momentum_buffer, gradient, plan->n_parameters, plan->learning_rate,
      plan->decay, plan->epsilon, plan->momentum, plan->centered,
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

extern "C" int fortml_cuda_rmsprop_plan_download(
    void* opaque_plan, double* parameters, double* square_average,
    double* gradient_average, double* momentum_buffer, int* step_count) {
  RmspropPlan* plan = static_cast<RmspropPlan*>(opaque_plan);
  if (plan == nullptr || parameters == nullptr || square_average == nullptr ||
      gradient_average == nullptr || momentum_buffer == nullptr ||
      step_count == nullptr)
    return static_cast<int>(cudaErrorInvalidValue);
  cudaError_t error = cudaSetDevice(plan->device_index);
  const std::size_t count = static_cast<std::size_t>(plan->n_parameters);
  if (error == cudaSuccess)
    error = cudaMemcpy(parameters, plan->parameters, sizeof(double) * count,
                       cudaMemcpyDeviceToHost);
  if (error == cudaSuccess)
    error = cudaMemcpy(square_average, plan->square_average,
                       sizeof(double) * count, cudaMemcpyDeviceToHost);
  if (error == cudaSuccess)
    error = cudaMemcpy(gradient_average, plan->gradient_average,
                       sizeof(double) * count, cudaMemcpyDeviceToHost);
  if (error == cudaSuccess)
    error = cudaMemcpy(momentum_buffer, plan->momentum_buffer,
                       sizeof(double) * count, cudaMemcpyDeviceToHost);
  if (error == cudaSuccess) *step_count = plan->step_count;
  return static_cast<int>(error);
}

extern "C" int fortml_cuda_rmsprop_plan_destroy(void* opaque_plan) {
  destroy_plan(static_cast<RmspropPlan*>(opaque_plan));
  return 0;
}
