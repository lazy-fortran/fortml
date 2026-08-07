#include "fortml_cuda_dense.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
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
  std::uint64_t host_to_device_bytes = 0;
  std::uint64_t device_to_host_bytes = 0;
};

inline void add_host_to_device(DensePlan *plan, std::size_t bytes) {
  plan->host_to_device_bytes += static_cast<std::uint64_t>(bytes);
}

inline void add_device_to_host(DensePlan *plan, std::size_t bytes) {
  plan->device_to_host_bytes += static_cast<std::uint64_t>(bytes);
}

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

__device__ inline double activation_derivative(double value, int activation) {
  switch (activation) {
    case kActivationTanh: {
      const double t = tanh(value);
      return 1.0 - t * t;
    }
    case kActivationRelu:
      return value >= 0.0 ? 1.0 : 0.0;
    case kActivationGelu: {
      constexpr double c = 0.79788456080286535588;
      constexpr double a = 0.044715;
      const double value2 = value * value;
      const double inner = c * (value + a * value * value2);
      const double t = tanh(inner);
      return 0.5 * (1.0 + t) +
          0.5 * value * (1.0 - t * t) * c * (1.0 + 3.0 * a * value2);
    }
    case kActivationSilu: {
      const double sigmoid = 1.0 / (1.0 + exp(-value));
      return sigmoid + value * sigmoid * (1.0 - sigmoid);
    }
    case kActivationElu:
      return value >= 0.0 ? 1.0 : exp(value);
    case kActivationSoftplus:
      return 1.0 / (1.0 + exp(-value));
    case kActivationLeakyRelu:
      return value >= 0.0 ? 1.0 : 0.01;
    case kActivationLinear:
    default:
      return 1.0;
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

__global__ void dense_jvp_kernel(
    const DensePlan plan, const double *query_x, const double *query_x_dot,
    const double *weights_dot, const double *bias_dot, int n_query,
    double *outputs, double *outputs_dot) {
  const int output = blockIdx.x * blockDim.x + threadIdx.x;
  if (output >= plan.n_outputs) return;
  for (int query = 0; query < n_query; ++query) {
    double value = plan.bias[output];
    double tangent = bias_dot[output];
    for (int input = 0; input < plan.n_inputs; ++input) {
      const double x = query_x[input * n_query + query];
      const double x_dot = query_x_dot[input * n_query + query];
      value += plan.weights[output * plan.n_inputs + input] * x;
      tangent += plan.weights[output * plan.n_inputs + input] * x_dot +
          weights_dot[output * plan.n_inputs + input] * x;
    }
    outputs[output * n_query + query] = activate(value, plan.activation);
    outputs_dot[output * n_query + query] =
        activation_derivative(value, plan.activation) * tangent;
  }
}

__global__ void dense_vjp_zbar_kernel(
    const DensePlan plan, const double *query_x, const double *output_bar,
    int n_query, double *zbar) {
  const int flat = blockIdx.x * blockDim.x + threadIdx.x;
  const int count = plan.n_outputs * n_query;
  if (flat >= count) return;
  const int output = flat / n_query;
  const int query = flat - output * n_query;
  double value = plan.bias[output];
  for (int input = 0; input < plan.n_inputs; ++input)
    value += plan.weights[output * plan.n_inputs + input] *
        query_x[input * n_query + query];
  zbar[flat] = output_bar[flat] * activation_derivative(value, plan.activation);
}

__global__ void dense_vjp_input_kernel(
    const DensePlan plan, const double *zbar, int n_query,
    double *query_x_bar) {
  const int flat = blockIdx.x * blockDim.x + threadIdx.x;
  const int count = plan.n_inputs * n_query;
  if (flat >= count) return;
  const int input = flat / n_query;
  const int query = flat - input * n_query;
  double value = 0.0;
  for (int output = 0; output < plan.n_outputs; ++output)
    value += plan.weights[output * plan.n_inputs + input] *
        zbar[output * n_query + query];
  query_x_bar[flat] = value;
}

__global__ void dense_vjp_parameter_kernel(
    const DensePlan plan, const double *query_x, const double *zbar,
    int n_query, double *weights_bar, double *bias_bar) {
  const int output = blockIdx.x * blockDim.x + threadIdx.x;
  if (output >= plan.n_outputs) return;
  double bias_value = 0.0;
  for (int query = 0; query < n_query; ++query) {
    const double cotangent = zbar[output * n_query + query];
    bias_value += cotangent;
    for (int input = 0; input < plan.n_inputs; ++input)
      weights_bar[output * plan.n_inputs + input] +=
          cotangent * query_x[input * n_query + query];
  }
  bias_bar[output] = bias_value;
}

__global__ void dense_train_mse_weight_gradient_kernel(
    const DensePlan plan, const double *query_x, const double *target,
    int n_query, double inv_count, double *weights_gradient) {
  const int flat = blockIdx.x * blockDim.x + threadIdx.x;
  const int count = plan.n_outputs * plan.n_inputs;
  if (flat >= count) return;
  const int output = flat / plan.n_inputs;
  const int input = flat - output * plan.n_inputs;
  double gradient = 0.0;
  for (int query = 0; query < n_query; ++query) {
    double value = plan.bias[output];
    for (int feature = 0; feature < plan.n_inputs; ++feature)
      value += plan.weights[output * plan.n_inputs + feature] *
          query_x[feature * n_query + query];
    const double residual = activate(value, plan.activation) -
        target[output * n_query + query];
    gradient += residual * activation_derivative(value, plan.activation) *
        query_x[input * n_query + query];
  }
  weights_gradient[flat] = gradient * inv_count;
}

__global__ void dense_train_mse_bias_loss_kernel(
    const DensePlan plan, const double *query_x, const double *target,
    int n_query, double inv_count, double *bias_gradient, double *loss) {
  const int output = blockIdx.x * blockDim.x + threadIdx.x;
  if (output >= plan.n_outputs) return;
  double gradient = 0.0;
  double loss_value = 0.0;
  for (int query = 0; query < n_query; ++query) {
    double value = plan.bias[output];
    for (int input = 0; input < plan.n_inputs; ++input)
      value += plan.weights[output * plan.n_inputs + input] *
          query_x[input * n_query + query];
    const double residual = activate(value, plan.activation) -
        target[output * n_query + query];
    gradient += residual * activation_derivative(value, plan.activation);
    loss_value += 0.5 * residual * residual;
  }
  bias_gradient[output] = gradient * inv_count;
  loss[output] = loss_value * inv_count;
}

__global__ void dense_train_mse_update_kernel(
    DensePlan plan, int count, double learning_rate,
    const double *weights_gradient, const double *bias_gradient) {
  const int flat = blockIdx.x * blockDim.x + threadIdx.x;
  if (flat >= count) return;
  plan.weights[flat] -= learning_rate * weights_gradient[flat];
  if (flat < plan.n_outputs)
    plan.bias[flat] -= learning_rate * bias_gradient[flat];
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
  if (error == cudaSuccess) {
    add_host_to_device(plan, sizeof(double) * weight_count);
    error = cudaMemcpy(plan->bias, bias, sizeof(double) * n_outputs,
                       cudaMemcpyHostToDevice);
  }
  if (error == cudaSuccess)
    add_host_to_device(plan, sizeof(double) * n_outputs);
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
  if (error == cudaSuccess)
    add_host_to_device(plan, sizeof(double) * query_count);
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
  if (error == cudaSuccess)
    add_device_to_host(plan, sizeof(double) * output_count);
  cudaFree(device_query);
  cudaFree(device_output);
  return static_cast<int>(error);
}

extern "C" int fortml_cuda_dense_plan_jvp(
    void *opaque_plan, const double *query_x, const double *query_x_dot,
    const double *weights_dot, const double *bias_dot, int n_query,
    double *outputs, double *outputs_dot) {
  DensePlan *plan = static_cast<DensePlan *>(opaque_plan);
  if (plan == nullptr || query_x == nullptr || query_x_dot == nullptr ||
      weights_dot == nullptr || bias_dot == nullptr || outputs == nullptr ||
      outputs_dot == nullptr || n_query < 1 ||
      !finite_array(query_x, static_cast<std::size_t>(plan->n_inputs) * n_query) ||
      !finite_array(query_x_dot,
                    static_cast<std::size_t>(plan->n_inputs) * n_query) ||
      !finite_array(weights_dot,
                    static_cast<std::size_t>(plan->n_inputs) * plan->n_outputs) ||
      !finite_array(bias_dot, static_cast<std::size_t>(plan->n_outputs)))
    return static_cast<int>(cudaErrorInvalidValue);

  cudaError_t error = cudaSetDevice(plan->device_index);
  if (error != cudaSuccess) return static_cast<int>(error);
  double *device_query = nullptr;
  double *device_query_dot = nullptr;
  double *device_weights_dot = nullptr;
  double *device_bias_dot = nullptr;
  double *device_output = nullptr;
  double *device_output_dot = nullptr;
  const std::size_t query_count =
      static_cast<std::size_t>(plan->n_inputs) * n_query;
  const std::size_t weight_count =
      static_cast<std::size_t>(plan->n_inputs) * plan->n_outputs;
  const std::size_t output_count =
      static_cast<std::size_t>(plan->n_outputs) * n_query;
  error = cudaMalloc(&device_query, sizeof(double) * query_count);
  if (error == cudaSuccess)
    error = cudaMalloc(&device_query_dot, sizeof(double) * query_count);
  if (error == cudaSuccess)
    error = cudaMalloc(&device_weights_dot, sizeof(double) * weight_count);
  if (error == cudaSuccess)
    error = cudaMalloc(&device_bias_dot, sizeof(double) * plan->n_outputs);
  if (error == cudaSuccess)
    error = cudaMalloc(&device_output, sizeof(double) * output_count);
  if (error == cudaSuccess)
    error = cudaMalloc(&device_output_dot, sizeof(double) * output_count);
  if (error == cudaSuccess)
    error = cudaMemcpy(device_query, query_x, sizeof(double) * query_count,
                       cudaMemcpyHostToDevice);
  if (error == cudaSuccess)
    add_host_to_device(plan, sizeof(double) * query_count);
  if (error == cudaSuccess)
    error = cudaMemcpy(device_query_dot, query_x_dot,
                       sizeof(double) * query_count, cudaMemcpyHostToDevice);
  if (error == cudaSuccess)
    add_host_to_device(plan, sizeof(double) * query_count);
  if (error == cudaSuccess)
    error = cudaMemcpy(device_weights_dot, weights_dot,
                       sizeof(double) * weight_count, cudaMemcpyHostToDevice);
  if (error == cudaSuccess)
    add_host_to_device(plan, sizeof(double) * weight_count);
  if (error == cudaSuccess)
    error = cudaMemcpy(device_bias_dot, bias_dot,
                       sizeof(double) * plan->n_outputs,
                       cudaMemcpyHostToDevice);
  if (error == cudaSuccess)
    add_host_to_device(plan, sizeof(double) * plan->n_outputs);
  if (error == cudaSuccess) {
    dense_jvp_kernel<<<(plan->n_outputs + kThreads - 1) / kThreads,
                       kThreads>>>(*plan, device_query, device_query_dot,
                                   device_weights_dot, device_bias_dot, n_query,
                                   device_output, device_output_dot);
    error = cudaGetLastError();
  }
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  if (error == cudaSuccess)
    error = cudaMemcpy(outputs, device_output, sizeof(double) * output_count,
                       cudaMemcpyDeviceToHost);
  if (error == cudaSuccess)
    add_device_to_host(plan, sizeof(double) * output_count);
  if (error == cudaSuccess)
    error = cudaMemcpy(outputs_dot, device_output_dot,
                       sizeof(double) * output_count, cudaMemcpyDeviceToHost);
  if (error == cudaSuccess)
    add_device_to_host(plan, sizeof(double) * output_count);
  cudaFree(device_query);
  cudaFree(device_query_dot);
  cudaFree(device_weights_dot);
  cudaFree(device_bias_dot);
  cudaFree(device_output);
  cudaFree(device_output_dot);
  return static_cast<int>(error);
}

extern "C" int fortml_cuda_dense_plan_vjp(
    void *opaque_plan, const double *query_x, const double *output_bar,
    int n_query, double *query_x_bar, double *weights_bar, double *bias_bar) {
  DensePlan *plan = static_cast<DensePlan *>(opaque_plan);
  if (plan == nullptr || query_x == nullptr || output_bar == nullptr ||
      query_x_bar == nullptr || weights_bar == nullptr || bias_bar == nullptr ||
      n_query < 1 ||
      !finite_array(query_x, static_cast<std::size_t>(plan->n_inputs) * n_query) ||
      !finite_array(output_bar,
                    static_cast<std::size_t>(plan->n_outputs) * n_query))
    return static_cast<int>(cudaErrorInvalidValue);

  cudaError_t error = cudaSetDevice(plan->device_index);
  if (error != cudaSuccess) return static_cast<int>(error);
  double *device_query = nullptr;
  double *device_output_bar = nullptr;
  double *device_zbar = nullptr;
  double *device_query_bar = nullptr;
  double *device_weights_bar = nullptr;
  double *device_bias_bar = nullptr;
  const std::size_t query_count =
      static_cast<std::size_t>(plan->n_inputs) * n_query;
  const std::size_t output_count =
      static_cast<std::size_t>(plan->n_outputs) * n_query;
  const std::size_t weight_count =
      static_cast<std::size_t>(plan->n_inputs) * plan->n_outputs;
  error = cudaMalloc(&device_query, sizeof(double) * query_count);
  if (error == cudaSuccess)
    error = cudaMalloc(&device_output_bar, sizeof(double) * output_count);
  if (error == cudaSuccess)
    error = cudaMalloc(&device_zbar, sizeof(double) * output_count);
  if (error == cudaSuccess)
    error = cudaMalloc(&device_query_bar, sizeof(double) * query_count);
  if (error == cudaSuccess)
    error = cudaMalloc(&device_weights_bar, sizeof(double) * weight_count);
  if (error == cudaSuccess)
    error = cudaMalloc(&device_bias_bar, sizeof(double) * plan->n_outputs);
  if (error == cudaSuccess)
    error = cudaMemset(device_weights_bar, 0, sizeof(double) * weight_count);
  if (error == cudaSuccess)
    error = cudaMemcpy(device_query, query_x, sizeof(double) * query_count,
                       cudaMemcpyHostToDevice);
  if (error == cudaSuccess)
    add_host_to_device(plan, sizeof(double) * query_count);
  if (error == cudaSuccess)
    error = cudaMemcpy(device_output_bar, output_bar,
                       sizeof(double) * output_count, cudaMemcpyHostToDevice);
  if (error == cudaSuccess)
    add_host_to_device(plan, sizeof(double) * output_count);
  if (error == cudaSuccess) {
    dense_vjp_zbar_kernel<<<(output_count + kThreads - 1) / kThreads,
                            kThreads>>>(*plan, device_query, device_output_bar,
                                        n_query, device_zbar);
    error = cudaGetLastError();
  }
  if (error == cudaSuccess) {
    dense_vjp_input_kernel<<<(query_count + kThreads - 1) / kThreads,
                             kThreads>>>(*plan, device_zbar, n_query,
                                         device_query_bar);
    error = cudaGetLastError();
  }
  if (error == cudaSuccess) {
    dense_vjp_parameter_kernel<<<(plan->n_outputs + kThreads - 1) / kThreads,
                                 kThreads>>>(*plan, device_query, device_zbar,
                                             n_query, device_weights_bar,
                                             device_bias_bar);
    error = cudaGetLastError();
  }
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  if (error == cudaSuccess)
    error = cudaMemcpy(query_x_bar, device_query_bar,
                       sizeof(double) * query_count, cudaMemcpyDeviceToHost);
  if (error == cudaSuccess)
    add_device_to_host(plan, sizeof(double) * query_count);
  if (error == cudaSuccess)
    error = cudaMemcpy(weights_bar, device_weights_bar,
                       sizeof(double) * weight_count, cudaMemcpyDeviceToHost);
  if (error == cudaSuccess)
    add_device_to_host(plan, sizeof(double) * weight_count);
  if (error == cudaSuccess)
    error = cudaMemcpy(bias_bar, device_bias_bar,
                       sizeof(double) * plan->n_outputs, cudaMemcpyDeviceToHost);
  if (error == cudaSuccess)
    add_device_to_host(plan, sizeof(double) * plan->n_outputs);
  cudaFree(device_query);
  cudaFree(device_output_bar);
  cudaFree(device_zbar);
  cudaFree(device_query_bar);
  cudaFree(device_weights_bar);
  cudaFree(device_bias_bar);
  return static_cast<int>(error);
}

extern "C" int fortml_cuda_dense_plan_train_mse(
    void *opaque_plan, const double *query_x, const double *target, int n_query,
    double learning_rate, double *loss) {
  DensePlan *plan = static_cast<DensePlan *>(opaque_plan);
  if (plan == nullptr || query_x == nullptr || target == nullptr ||
      loss == nullptr || n_query < 1 || !std::isfinite(learning_rate) ||
      learning_rate <= 0.0 ||
      !finite_array(query_x, static_cast<std::size_t>(plan->n_inputs) * n_query) ||
      !finite_array(target,
                    static_cast<std::size_t>(plan->n_outputs) * n_query))
    return static_cast<int>(cudaErrorInvalidValue);

  cudaError_t error = cudaSetDevice(plan->device_index);
  if (error != cudaSuccess) return static_cast<int>(error);
  double *device_query = nullptr;
  double *device_target = nullptr;
  double *device_weights_gradient = nullptr;
  double *device_bias_gradient = nullptr;
  double *device_loss = nullptr;
  const std::size_t query_count =
      static_cast<std::size_t>(plan->n_inputs) * n_query;
  const std::size_t output_count =
      static_cast<std::size_t>(plan->n_outputs) * n_query;
  const std::size_t weight_count =
      static_cast<std::size_t>(plan->n_inputs) * plan->n_outputs;
  error = cudaMalloc(&device_query, sizeof(double) * query_count);
  if (error == cudaSuccess)
    error = cudaMalloc(&device_target, sizeof(double) * output_count);
  if (error == cudaSuccess)
    error = cudaMalloc(&device_weights_gradient, sizeof(double) * weight_count);
  if (error == cudaSuccess)
    error = cudaMalloc(&device_bias_gradient, sizeof(double) * plan->n_outputs);
  if (error == cudaSuccess)
    error = cudaMalloc(&device_loss, sizeof(double) * plan->n_outputs);
  if (error == cudaSuccess)
    error = cudaMemcpy(device_query, query_x, sizeof(double) * query_count,
                       cudaMemcpyHostToDevice);
  if (error == cudaSuccess)
    add_host_to_device(plan, sizeof(double) * query_count);
  if (error == cudaSuccess)
    error = cudaMemcpy(device_target, target, sizeof(double) * output_count,
                       cudaMemcpyHostToDevice);
  if (error == cudaSuccess)
    add_host_to_device(plan, sizeof(double) * output_count);
  const double inv_count = 1.0 /
      static_cast<double>(n_query * plan->n_outputs);
  if (error == cudaSuccess) {
    dense_train_mse_weight_gradient_kernel<<<
        (weight_count + kThreads - 1) / kThreads, kThreads>>>(
        *plan, device_query, device_target, n_query, inv_count,
        device_weights_gradient);
    error = cudaGetLastError();
  }
  if (error == cudaSuccess) {
    dense_train_mse_bias_loss_kernel<<<
        (plan->n_outputs + kThreads - 1) / kThreads, kThreads>>>(
        *plan, device_query, device_target, n_query, inv_count,
        device_bias_gradient, device_loss);
    error = cudaGetLastError();
  }
  if (error == cudaSuccess) {
    dense_train_mse_update_kernel<<<
        (weight_count + kThreads - 1) / kThreads, kThreads>>>(
        *plan, static_cast<int>(weight_count), learning_rate,
        device_weights_gradient, device_bias_gradient);
    error = cudaGetLastError();
  }
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  double *host_loss = new (std::nothrow) double[plan->n_outputs];
  if (error == cudaSuccess && host_loss == nullptr)
    error = cudaErrorMemoryAllocation;
  if (error == cudaSuccess)
    error = cudaMemcpy(host_loss, device_loss,
                       sizeof(double) * plan->n_outputs,
                       cudaMemcpyDeviceToHost);
  if (error == cudaSuccess) {
    add_device_to_host(plan, sizeof(double) * plan->n_outputs);
    *loss = 0.0;
    for (int output = 0; output < plan->n_outputs; ++output)
      *loss += host_loss[output];
  }
  delete[] host_loss;
  cudaFree(device_query);
  cudaFree(device_target);
  cudaFree(device_weights_gradient);
  cudaFree(device_bias_gradient);
  cudaFree(device_loss);
  return static_cast<int>(error);
}

extern "C" int fortml_cuda_dense_plan_get_parameters(
    void *opaque_plan, double *weights, double *bias) {
  DensePlan *plan = static_cast<DensePlan *>(opaque_plan);
  if (plan == nullptr || weights == nullptr || bias == nullptr)
    return static_cast<int>(cudaErrorInvalidValue);
  cudaError_t error = cudaSetDevice(plan->device_index);
  if (error != cudaSuccess) return static_cast<int>(error);
  const std::size_t weight_count =
      static_cast<std::size_t>(plan->n_inputs) * plan->n_outputs;
  error = cudaMemcpy(weights, plan->weights, sizeof(double) * weight_count,
                     cudaMemcpyDeviceToHost);
  if (error == cudaSuccess)
    add_device_to_host(plan, sizeof(double) * weight_count);
  if (error == cudaSuccess)
    error = cudaMemcpy(bias, plan->bias, sizeof(double) * plan->n_outputs,
                       cudaMemcpyDeviceToHost);
  if (error == cudaSuccess)
    add_device_to_host(plan, sizeof(double) * plan->n_outputs);
  return static_cast<int>(error);
}

extern "C" int fortml_cuda_dense_plan_transfer_stats(
    void *opaque_plan, std::uint64_t *host_to_device_bytes,
    std::uint64_t *device_to_host_bytes, std::uint64_t *resident_bytes) {
  DensePlan *plan = static_cast<DensePlan *>(opaque_plan);
  if (plan == nullptr || host_to_device_bytes == nullptr ||
      device_to_host_bytes == nullptr || resident_bytes == nullptr)
    return static_cast<int>(cudaErrorInvalidValue);
  *host_to_device_bytes = plan->host_to_device_bytes;
  *device_to_host_bytes = plan->device_to_host_bytes;
  *resident_bytes = sizeof(double) *
      (static_cast<std::size_t>(plan->n_inputs) * plan->n_outputs +
       static_cast<std::size_t>(plan->n_outputs));
  return 0;
}

extern "C" int fortml_cuda_dense_plan_destroy(void *opaque_plan) {
  destroy_plan(static_cast<DensePlan *>(opaque_plan));
  return 0;
}
