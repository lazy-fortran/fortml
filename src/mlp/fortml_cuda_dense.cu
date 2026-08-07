#include "fortml_cuda_dense.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <new>

namespace {

constexpr int kThreads = 128;
constexpr int kActivationLinear = 1;
constexpr int kActivationTanh = 2;
constexpr int kActivationRelu = 3;
constexpr int kActivationGelu = 4;
constexpr int kActivationSilu = 5;
constexpr int kActivationElu = 6;
constexpr int kActivationSoftplus = 7;
constexpr int kActivationLeakyRelu = 8;

struct DensePlan {
  double *weights = nullptr;
  double *bias = nullptr;
  int n_inputs = 0;
  int n_outputs = 0;
  int activation = kActivationLinear;
  int device_index = 0;
};

__device__ inline double activate(double value, int activation) {
  switch (activation) {
    case kActivationTanh:
      return tanh(value);
    case kActivationRelu:
      return fmax(value, 0.0);
    case kActivationGelu:
      return 0.5 * value *
          (1.0 + tanh(0.79788456080286535588 *
                      (value + 0.044715 * value * value * value)));
    case kActivationSilu:
      return value / (1.0 + exp(-value));
    case kActivationElu:
      return value >= 0.0 ? value : exp(value) - 1.0;
    case kActivationSoftplus:
      // Stable enough for the double-precision inference contract while
      // retaining the exact CPU reference branch at both tails.
      if (value > 20.0) return value;
      if (value < -20.0) return exp(value);
      return log1p(exp(value));
    case kActivationLeakyRelu:
      return value >= 0.0 ? value : 0.01 * value;
    case kActivationLinear:
    default:
      return value;
  }
}

__global__ void dense_forward_kernel(
    const DensePlan plan, const double *query_x, int n_query,
    double *outputs) {
  const int output = blockIdx.x * blockDim.x + threadIdx.x;
  if (output >= plan.n_outputs) return;
  for (int query = 0; query < n_query; ++query) {
    double value = plan.bias[output];
    for (int input = 0; input < plan.n_inputs; ++input) {
      value += plan.weights[output * plan.n_inputs + input] *
          query_x[input * n_query + query];
    }
    outputs[output * n_query + query] = activate(value, plan.activation);
  }
}

bool finite_array(const double *values, std::size_t count) {
  for (std::size_t i = 0; i < count; ++i)
    if (!std::isfinite(values[i])) return false;
  return true;
}

bool valid_activation(int activation) {
  return activation >= kActivationLinear &&
      activation <= kActivationLeakyRelu;
}

void destroy_plan(DensePlan *plan) {
  if (plan == nullptr) return;
  cudaSetDevice(plan->device_index);
  cudaFree(plan->weights);
  cudaFree(plan->bias);
  delete plan;
}

}  // namespace

extern "C" int fortml_cuda_dense_available() {
  int count = 0;
  const cudaError_t error = cudaGetDeviceCount(&count);
  if (error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver)
    return 0;
  return error == cudaSuccess && count > 0 ? 1 : 0;
}

extern "C" int fortml_cuda_dense_plan_create(
    const double *weights, const double *bias, int n_inputs, int n_outputs,
    int activation, int device_index, void **opaque_plan) {
  if (opaque_plan == nullptr) return static_cast<int>(cudaErrorInvalidValue);
  *opaque_plan = nullptr;
  if (weights == nullptr || bias == nullptr || n_inputs < 1 || n_outputs < 1 ||
      !valid_activation(activation) || device_index < 0 ||
      !finite_array(weights, static_cast<std::size_t>(n_inputs) * n_outputs) ||
      !finite_array(bias, static_cast<std::size_t>(n_outputs)))
    return static_cast<int>(cudaErrorInvalidValue);

  cudaError_t error = cudaSetDevice(device_index);
  if (error != cudaSuccess) return static_cast<int>(error);
  DensePlan *plan = new (std::nothrow) DensePlan();
  if (plan == nullptr) return static_cast<int>(cudaErrorMemoryAllocation);
  plan->n_inputs = n_inputs;
  plan->n_outputs = n_outputs;
  plan->activation = activation;
  plan->device_index = device_index;
  const std::size_t weight_count =
      static_cast<std::size_t>(n_inputs) * n_outputs;
  error = cudaMalloc(&plan->weights, sizeof(double) * weight_count);
  if (error == cudaSuccess)
    error = cudaMalloc(&plan->bias, sizeof(double) * n_outputs);
  if (error == cudaSuccess)
    error = cudaMemcpy(plan->weights, weights, sizeof(double) * weight_count,
                       cudaMemcpyHostToDevice);
  if (error == cudaSuccess)
    error = cudaMemcpy(plan->bias, bias, sizeof(double) * n_outputs,
                       cudaMemcpyHostToDevice);
  if (error != cudaSuccess) {
    destroy_plan(plan);
    return static_cast<int>(error);
  }
  *opaque_plan = plan;
  return 0;
}

extern "C" int fortml_cuda_dense_plan_predict(
    void *opaque_plan, const double *query_x, int n_query, double *outputs) {
  DensePlan *plan = static_cast<DensePlan *>(opaque_plan);
  if (plan == nullptr || query_x == nullptr || outputs == nullptr ||
      n_query < 1 ||
      !finite_array(query_x, static_cast<std::size_t>(plan->n_inputs) * n_query))
    return static_cast<int>(cudaErrorInvalidValue);

  cudaError_t error = cudaSetDevice(plan->device_index);
  if (error != cudaSuccess) return static_cast<int>(error);
  double *device_query = nullptr;
  double *device_output = nullptr;
  const std::size_t query_count =
      static_cast<std::size_t>(plan->n_inputs) * n_query;
  const std::size_t output_count =
      static_cast<std::size_t>(plan->n_outputs) * n_query;
  error = cudaMalloc(&device_query, sizeof(double) * query_count);
  if (error == cudaSuccess)
    error = cudaMalloc(&device_output, sizeof(double) * output_count);
  if (error == cudaSuccess)
    error = cudaMemcpy(device_query, query_x, sizeof(double) * query_count,
                       cudaMemcpyHostToDevice);
  if (error == cudaSuccess) {
    dense_forward_kernel<<<(plan->n_outputs + kThreads - 1) / kThreads,
                           kThreads>>>(*plan, device_query, n_query,
                                       device_output);
    error = cudaGetLastError();
  }
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  if (error == cudaSuccess)
    error = cudaMemcpy(outputs, device_output, sizeof(double) * output_count,
                       cudaMemcpyDeviceToHost);
  cudaFree(device_query);
  cudaFree(device_output);
  return static_cast<int>(error);
}

extern "C" int fortml_cuda_dense_plan_destroy(void *opaque_plan) {
  destroy_plan(static_cast<DensePlan *>(opaque_plan));
  return 0;
}
