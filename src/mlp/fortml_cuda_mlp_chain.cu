#include "fortml_cuda_mlp_chain.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <vector>

namespace {

constexpr int kThreads = 128;
constexpr int kLinear = 1;
constexpr int kTanh = 2;
constexpr int kRelu = 3;
constexpr int kGelu = 4;
constexpr int kSilu = 5;
constexpr int kElu = 6;
constexpr int kSoftplus = 7;
constexpr int kLeakyRelu = 8;

struct ChainPlan {
  double *weights = nullptr;
  double *biases = nullptr;
  int *layer_sizes = nullptr;
  int *activations = nullptr;
  int n_layers = 0;
  int input_width = 0;
  int output_width = 0;
  int max_width = 0;
  int total_weights = 0;
  int total_biases = 0;
  int device_index = 0;
  std::vector<int> weight_offsets;
  std::vector<int> bias_offsets;
  std::vector<int> activation_offsets;
  std::vector<int> layer_sizes_host;
  std::vector<int> activations_host;
  std::size_t activation_slots = 0;
  int capacity_query = 0;
  double *input = nullptr;
  double *input_dot = nullptr;
  double *layer_outputs = nullptr;
  double *layer_preacts = nullptr;
  double *layer_tangents = nullptr;
  double *weights_dot = nullptr;
  double *biases_dot = nullptr;
  double *weights_bar = nullptr;
  double *biases_bar = nullptr;
  double *bar_a = nullptr;
  double *bar_b = nullptr;
  std::uint64_t host_to_device_bytes = 0;
  std::uint64_t device_to_host_bytes = 0;
  std::uint64_t resident_bytes = 0;
};

inline void add_h2d(ChainPlan *plan, std::size_t bytes) {
  plan->host_to_device_bytes += static_cast<std::uint64_t>(bytes);
}

inline void add_d2h(ChainPlan *plan, std::size_t bytes) {
  plan->device_to_host_bytes += static_cast<std::uint64_t>(bytes);
}

__device__ inline double activate(double value, int activation) {
  switch (activation) {
    case kTanh:
      return tanh(value);
    case kRelu:
      return fmax(value, 0.0);
    case kGelu:
      return 0.5 * value *
          (1.0 + tanh(0.79788456080286535588 *
                      (value + 0.044715 * value * value * value)));
    case kSilu:
      return value / (1.0 + exp(-value));
    case kElu:
      return value >= 0.0 ? value : exp(value) - 1.0;
    case kSoftplus:
      if (value > 20.0) return value;
      if (value < -20.0) return exp(value);
      return log1p(exp(value));
    case kLeakyRelu:
      return value >= 0.0 ? value : 0.01 * value;
    case kLinear:
    default:
      return value;
  }
}

__device__ inline double activation_derivative(double value, int activation) {
  switch (activation) {
    case kTanh: {
      const double t = tanh(value);
      return 1.0 - t * t;
    }
    case kRelu:
      return value >= 0.0 ? 1.0 : 0.0;
    case kGelu: {
      constexpr double c = 0.79788456080286535588;
      constexpr double a = 0.044715;
      const double value2 = value * value;
      const double t = tanh(c * (value + a * value * value2));
      return 0.5 * (1.0 + t) +
          0.5 * value * (1.0 - t * t) * c * (1.0 + 3.0 * a * value2);
    }
    case kSilu: {
      const double sigmoid = 1.0 / (1.0 + exp(-value));
      return sigmoid + value * sigmoid * (1.0 - sigmoid);
    }
    case kElu:
      return value >= 0.0 ? 1.0 : exp(value);
    case kSoftplus:
      return 1.0 / (1.0 + exp(-value));
    case kLeakyRelu:
      return value >= 0.0 ? 1.0 : 0.01;
    case kLinear:
    default:
      return 1.0;
  }
}

__global__ void chain_forward_kernel(
    const double *weights, const double *biases, int weight_offset,
    int bias_offset, int input_width, int output_width, int activation,
    const double *source, int n_query, double *destination, double *preact) {
  const int flat = blockIdx.x * blockDim.x + threadIdx.x;
  const int count = output_width * n_query;
  if (flat >= count) return;
  const int output = flat / n_query;
  const int query = flat - output * n_query;
  double value = biases[bias_offset + output];
  for (int input = 0; input < input_width; ++input)
    value += weights[weight_offset + output * input_width + input] *
        source[input * n_query + query];
  preact[flat] = value;
  destination[flat] = activate(value, activation);
}

__global__ void chain_jvp_kernel(
    const double *weights, const double *biases, int weight_offset,
    int bias_offset, const double *weights_dot, const double *biases_dot,
    int input_width, int output_width, int activation, const double *source,
    const double *source_dot, int n_query, double *destination,
    double *destination_dot, double *preact) {
  const int flat = blockIdx.x * blockDim.x + threadIdx.x;
  const int count = output_width * n_query;
  if (flat >= count) return;
  const int output = flat / n_query;
  const int query = flat - output * n_query;
  double value = biases[bias_offset + output];
  double tangent = biases_dot[bias_offset + output];
  for (int input = 0; input < input_width; ++input) {
    const int weight = weight_offset + output * input_width + input;
    const double x = source[input * n_query + query];
    value += weights[weight] * x;
    tangent += weights[weight] * source_dot[input * n_query + query] +
        weights_dot[weight] * x;
  }
  preact[flat] = value;
  destination[flat] = activate(value, activation);
  destination_dot[flat] = activation_derivative(value, activation) * tangent;
}

__global__ void chain_vjp_zbar_kernel(
    const double *preact, int activation, int count, double *bar) {
  const int flat = blockIdx.x * blockDim.x + threadIdx.x;
  if (flat >= count) return;
  bar[flat] *= activation_derivative(preact[flat], activation);
}

__global__ void chain_vjp_input_kernel(
    const double *weights, int weight_offset, int input_width,
    int output_width, const double *zbar, int n_query, double *source_bar) {
  const int flat = blockIdx.x * blockDim.x + threadIdx.x;
  const int count = input_width * n_query;
  if (flat >= count) return;
  const int input = flat / n_query;
  const int query = flat - input * n_query;
  double value = 0.0;
  for (int output = 0; output < output_width; ++output)
    value += weights[weight_offset + output * input_width + input] *
        zbar[output * n_query + query];
  source_bar[flat] = value;
}

__global__ void chain_vjp_parameter_kernel(
    const double *source, const double *zbar, int input_width,
    int output_width, int n_query, int weight_offset, int bias_offset,
    double *weights_bar, double *biases_bar) {
  const int flat = blockIdx.x * blockDim.x + threadIdx.x;
  const int count = output_width * input_width;
  if (flat >= count) return;
  const int output = flat / input_width;
  const int input = flat - output * input_width;
  double value = 0.0;
  for (int query = 0; query < n_query; ++query)
    value += zbar[output * n_query + query] * source[input * n_query + query];
  weights_bar[weight_offset + output * input_width + input] = value;
  if (input == 0) {
    double bias = 0.0;
    for (int query = 0; query < n_query; ++query)
      bias += zbar[output * n_query + query];
    biases_bar[bias_offset + output] = bias;
  }
}

bool finite_array(const double *values, std::size_t count) {
  for (std::size_t i = 0; i < count; ++i)
    if (!std::isfinite(values[i])) return false;
  return true;
}

bool valid_activation(int activation) {
  return activation >= kLinear && activation <= kLeakyRelu;
}

void free_pointer(double *&pointer) {
  if (pointer != nullptr) cudaFree(pointer);
  pointer = nullptr;
}

void free_pointer(int *&pointer) {
  if (pointer != nullptr) cudaFree(pointer);
  pointer = nullptr;
}

void release_workspace(ChainPlan *plan) {
  free_pointer(plan->input);
  free_pointer(plan->input_dot);
  free_pointer(plan->layer_outputs);
  free_pointer(plan->layer_preacts);
  free_pointer(plan->layer_tangents);
  free_pointer(plan->weights_dot);
  free_pointer(plan->biases_dot);
  free_pointer(plan->weights_bar);
  free_pointer(plan->biases_bar);
  free_pointer(plan->bar_a);
  free_pointer(plan->bar_b);
  plan->capacity_query = 0;
  plan->resident_bytes = 0;
  if (plan->weights != nullptr)
    plan->resident_bytes += sizeof(double) *
        static_cast<std::size_t>(plan->total_weights);
  if (plan->biases != nullptr)
    plan->resident_bytes += sizeof(double) *
        static_cast<std::size_t>(plan->total_biases);
  plan->resident_bytes += sizeof(int) *
      static_cast<std::size_t>(plan->n_layers + 1 + plan->n_layers);
}

cudaError_t ensure_workspace(ChainPlan *plan, int n_query) {
  if (n_query <= plan->capacity_query) return cudaSuccess;
  release_workspace(plan);
  plan->capacity_query = n_query;
  const std::size_t query_input =
      static_cast<std::size_t>(plan->input_width) * n_query;
  const std::size_t slots = plan->activation_slots * n_query;
  const std::size_t bars = static_cast<std::size_t>(plan->max_width) * n_query;
  cudaError_t error = cudaMalloc(reinterpret_cast<void **>(&plan->input),
                                 sizeof(double) * query_input);
  if (error == cudaSuccess)
    error = cudaMalloc(reinterpret_cast<void **>(&plan->input_dot),
                       sizeof(double) * query_input);
  if (error == cudaSuccess)
    error = cudaMalloc(reinterpret_cast<void **>(&plan->layer_outputs),
                       sizeof(double) * slots);
  if (error == cudaSuccess)
    error = cudaMalloc(reinterpret_cast<void **>(&plan->layer_preacts),
                       sizeof(double) * slots);
  if (error == cudaSuccess)
    error = cudaMalloc(reinterpret_cast<void **>(&plan->layer_tangents),
                       sizeof(double) * slots);
  if (error == cudaSuccess)
    error = cudaMalloc(reinterpret_cast<void **>(&plan->weights_dot),
                       sizeof(double) * plan->total_weights);
  if (error == cudaSuccess)
    error = cudaMalloc(reinterpret_cast<void **>(&plan->biases_dot),
                       sizeof(double) * plan->total_biases);
  if (error == cudaSuccess)
    error = cudaMalloc(reinterpret_cast<void **>(&plan->weights_bar),
                       sizeof(double) * plan->total_weights);
  if (error == cudaSuccess)
    error = cudaMalloc(reinterpret_cast<void **>(&plan->biases_bar),
                       sizeof(double) * plan->total_biases);
  if (error == cudaSuccess)
    error = cudaMalloc(reinterpret_cast<void **>(&plan->bar_a),
                       sizeof(double) * bars);
  if (error == cudaSuccess)
    error = cudaMalloc(reinterpret_cast<void **>(&plan->bar_b),
                       sizeof(double) * bars);
  if (error != cudaSuccess) {
    release_workspace(plan);
    return error;
  }
  plan->resident_bytes += sizeof(double) *
      (2 * query_input + 3 * slots + 2 *
       static_cast<std::size_t>(plan->total_weights) + 2 *
       static_cast<std::size_t>(plan->total_biases) + 2 * bars);
  return cudaSuccess;
}

cudaError_t run_forward(ChainPlan *plan, int n_query) {
  const double *source = plan->input;
  for (int layer = 0; layer < plan->n_layers; ++layer) {
    const int input_width = plan->layer_sizes_host[layer];
    const int output_width = plan->layer_sizes_host[layer + 1];
    double *destination = plan->layer_outputs +
        static_cast<std::size_t>(plan->activation_offsets[layer]) *
            plan->capacity_query;
    double *preact = plan->layer_preacts +
        static_cast<std::size_t>(plan->activation_offsets[layer]) *
            plan->capacity_query;
    chain_forward_kernel<<<(output_width * n_query + kThreads - 1) /
                               kThreads,
                           kThreads>>>(
        plan->weights, plan->biases, plan->weight_offsets[layer],
        plan->bias_offsets[layer], input_width, output_width,
        plan->activations_host[layer], source, n_query, destination,
        preact);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) return error;
    source = destination;
  }
  return cudaDeviceSynchronize();
}

void destroy_plan(ChainPlan *plan) {
  if (plan == nullptr) return;
  cudaSetDevice(plan->device_index);
  release_workspace(plan);
  free_pointer(plan->weights);
  free_pointer(plan->biases);
  free_pointer(plan->layer_sizes);
  free_pointer(plan->activations);
  delete plan;
}

}  // namespace

extern "C" int fortml_cuda_mlp_chain_available() {
  int count = 0;
  const cudaError_t error = cudaGetDeviceCount(&count);
  if (error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver)
    return 0;
  return error == cudaSuccess && count > 0 ? 1 : 0;
}

extern "C" int fortml_cuda_mlp_chain_create(
    const int *layer_sizes, const int *activations, const double *weights,
    const double *biases, int n_layers, int device_index, void **opaque_plan) {
  if (opaque_plan == nullptr) return static_cast<int>(cudaErrorInvalidValue);
  *opaque_plan = nullptr;
  if (layer_sizes == nullptr || activations == nullptr || weights == nullptr ||
      biases == nullptr || n_layers < 1 || device_index < 0)
    return static_cast<int>(cudaErrorInvalidValue);
  std::size_t total_weights = 0;
  std::size_t total_biases = 0;
  std::size_t activation_slots = 0;
  int max_width = 0;
  for (int layer = 0; layer < n_layers; ++layer) {
    if (layer_sizes[layer] < 1 || layer_sizes[layer + 1] < 1 ||
        !valid_activation(activations[layer]))
      return static_cast<int>(cudaErrorInvalidValue);
    total_weights += static_cast<std::size_t>(layer_sizes[layer]) *
        static_cast<std::size_t>(layer_sizes[layer + 1]);
    total_biases += static_cast<std::size_t>(layer_sizes[layer + 1]);
    activation_slots += static_cast<std::size_t>(layer_sizes[layer + 1]);
    max_width = std::max(max_width, layer_sizes[layer + 1]);
  }
  if (total_weights > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      total_biases > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      !finite_array(weights, total_weights) || !finite_array(biases, total_biases))
    return static_cast<int>(cudaErrorInvalidValue);
  cudaError_t error = cudaSetDevice(device_index);
  if (error != cudaSuccess) return static_cast<int>(error);
  ChainPlan *plan = new (std::nothrow) ChainPlan();
  if (plan == nullptr) return static_cast<int>(cudaErrorMemoryAllocation);
  plan->n_layers = n_layers;
  plan->input_width = layer_sizes[0];
  plan->output_width = layer_sizes[n_layers];
  plan->max_width = std::max(max_width, plan->input_width);
  plan->total_weights = static_cast<int>(total_weights);
  plan->total_biases = static_cast<int>(total_biases);
  plan->device_index = device_index;
  plan->activation_slots = activation_slots;
  plan->layer_sizes_host.assign(layer_sizes, layer_sizes + n_layers + 1);
  plan->activations_host.assign(activations, activations + n_layers);
  plan->weight_offsets.resize(n_layers);
  plan->bias_offsets.resize(n_layers);
  plan->activation_offsets.resize(n_layers);
  int weight_offset = 0;
  int bias_offset = 0;
  int activation_offset = 0;
  for (int layer = 0; layer < n_layers; ++layer) {
    plan->weight_offsets[layer] = weight_offset;
    plan->bias_offsets[layer] = bias_offset;
    plan->activation_offsets[layer] = activation_offset;
    weight_offset += layer_sizes[layer] * layer_sizes[layer + 1];
    bias_offset += layer_sizes[layer + 1];
    activation_offset += layer_sizes[layer + 1];
  }
  error = cudaMalloc(reinterpret_cast<void **>(&plan->weights),
                     sizeof(double) * total_weights);
  if (error == cudaSuccess)
    error = cudaMalloc(reinterpret_cast<void **>(&plan->biases),
                       sizeof(double) * total_biases);
  if (error == cudaSuccess)
    error = cudaMalloc(reinterpret_cast<void **>(&plan->layer_sizes),
                       sizeof(int) * (n_layers + 1));
  if (error == cudaSuccess)
    error = cudaMalloc(reinterpret_cast<void **>(&plan->activations),
                       sizeof(int) * n_layers);
  if (error == cudaSuccess)
    error = cudaMemcpy(plan->weights, weights, sizeof(double) * total_weights,
                       cudaMemcpyHostToDevice);
  if (error == cudaSuccess) add_h2d(plan, sizeof(double) * total_weights);
  if (error == cudaSuccess)
    error = cudaMemcpy(plan->biases, biases, sizeof(double) * total_biases,
                       cudaMemcpyHostToDevice);
  if (error == cudaSuccess) add_h2d(plan, sizeof(double) * total_biases);
  if (error == cudaSuccess)
    error = cudaMemcpy(plan->layer_sizes, layer_sizes,
                       sizeof(int) * (n_layers + 1), cudaMemcpyHostToDevice);
  if (error == cudaSuccess)
    add_h2d(plan, sizeof(int) * (n_layers + 1));
  if (error == cudaSuccess)
    error = cudaMemcpy(plan->activations, activations, sizeof(int) * n_layers,
                       cudaMemcpyHostToDevice);
  if (error == cudaSuccess) add_h2d(plan, sizeof(int) * n_layers);
  if (error != cudaSuccess) {
    destroy_plan(plan);
    return static_cast<int>(error);
  }
  plan->resident_bytes = sizeof(double) * (total_weights + total_biases) +
      sizeof(int) * static_cast<std::size_t>(2 * n_layers + 1);
  *opaque_plan = plan;
  return 0;
}

extern "C" int fortml_cuda_mlp_chain_predict(
    void *opaque_plan, const double *query_x, int n_query, double *outputs) {
  ChainPlan *plan = static_cast<ChainPlan *>(opaque_plan);
  if (plan == nullptr || query_x == nullptr || outputs == nullptr ||
      n_query < 1 || !finite_array(query_x, static_cast<std::size_t>(
          plan->input_width) * n_query))
    return static_cast<int>(cudaErrorInvalidValue);
  cudaError_t error = cudaSetDevice(plan->device_index);
  if (error == cudaSuccess) error = ensure_workspace(plan, n_query);
  if (error == cudaSuccess)
    error = cudaMemcpy(plan->input, query_x, sizeof(double) *
        static_cast<std::size_t>(plan->input_width) * n_query,
        cudaMemcpyHostToDevice);
  if (error == cudaSuccess)
    add_h2d(plan, sizeof(double) * static_cast<std::size_t>(plan->input_width) *
        n_query);
  if (error == cudaSuccess) error = run_forward(plan, n_query);
  const double *result = nullptr;
  if (error == cudaSuccess)
    result = plan->layer_outputs + static_cast<std::size_t>(
        plan->activation_offsets.back()) * plan->capacity_query;
  if (error == cudaSuccess)
    error = cudaMemcpy(outputs, result, sizeof(double) *
        static_cast<std::size_t>(plan->output_width) * n_query,
        cudaMemcpyDeviceToHost);
  if (error == cudaSuccess)
    add_d2h(plan, sizeof(double) * static_cast<std::size_t>(plan->output_width) *
        n_query);
  return static_cast<int>(error);
}

extern "C" int fortml_cuda_mlp_chain_jvp(
    void *opaque_plan, const double *query_x, const double *query_x_dot,
    const double *weights_dot, const double *biases_dot, int n_query,
    double *outputs, double *outputs_dot) {
  ChainPlan *plan = static_cast<ChainPlan *>(opaque_plan);
  const std::size_t input_count = plan == nullptr ? 0 :
      static_cast<std::size_t>(plan->input_width) * n_query;
  if (plan == nullptr || query_x == nullptr || query_x_dot == nullptr ||
      weights_dot == nullptr || biases_dot == nullptr || outputs == nullptr ||
      outputs_dot == nullptr || n_query < 1 || !finite_array(query_x, input_count) ||
      !finite_array(query_x_dot, input_count) ||
      !finite_array(weights_dot, static_cast<std::size_t>(plan->total_weights)) ||
      !finite_array(biases_dot, static_cast<std::size_t>(plan->total_biases)))
    return static_cast<int>(cudaErrorInvalidValue);
  cudaError_t error = cudaSetDevice(plan->device_index);
  if (error == cudaSuccess) error = ensure_workspace(plan, n_query);
  if (error == cudaSuccess)
    error = cudaMemcpy(plan->input, query_x, sizeof(double) * input_count,
                       cudaMemcpyHostToDevice);
  if (error == cudaSuccess) add_h2d(plan, sizeof(double) * input_count);
  if (error == cudaSuccess)
    error = cudaMemcpy(plan->input_dot, query_x_dot, sizeof(double) * input_count,
                       cudaMemcpyHostToDevice);
  if (error == cudaSuccess) add_h2d(plan, sizeof(double) * input_count);
  if (error == cudaSuccess)
    error = cudaMemcpy(plan->weights_dot, weights_dot,
                       sizeof(double) * plan->total_weights,
                       cudaMemcpyHostToDevice);
  if (error == cudaSuccess) add_h2d(plan, sizeof(double) * plan->total_weights);
  if (error == cudaSuccess)
    error = cudaMemcpy(plan->biases_dot, biases_dot,
                       sizeof(double) * plan->total_biases,
                       cudaMemcpyHostToDevice);
  if (error == cudaSuccess) add_h2d(plan, sizeof(double) * plan->total_biases);
  const double *source = plan->input;
  const double *source_dot = plan->input_dot;
  for (int layer = 0; error == cudaSuccess && layer < plan->n_layers; ++layer) {
    const int input_width = plan->layer_sizes_host[layer];
    const int output_width = plan->layer_sizes_host[layer + 1];
    double *destination = plan->layer_outputs + static_cast<std::size_t>(
        plan->activation_offsets[layer]) * plan->capacity_query;
    double *destination_dot = plan->layer_tangents + static_cast<std::size_t>(
        plan->activation_offsets[layer]) * plan->capacity_query;
    double *preact = plan->layer_preacts + static_cast<std::size_t>(
        plan->activation_offsets[layer]) * plan->capacity_query;
    chain_jvp_kernel<<<(output_width * n_query + kThreads - 1) / kThreads,
                       kThreads>>>(
        plan->weights, plan->biases, plan->weight_offsets[layer],
        plan->bias_offsets[layer], plan->weights_dot, plan->biases_dot,
        input_width,
        output_width, plan->activations_host[layer], source, source_dot, n_query,
        destination, destination_dot, preact);
    error = cudaGetLastError();
    source = destination;
    source_dot = destination_dot;
  }
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  const double *result = nullptr;
  const double *result_dot = nullptr;
  if (error == cudaSuccess) {
    result = plan->layer_outputs + static_cast<std::size_t>(
        plan->activation_offsets.back()) * plan->capacity_query;
    result_dot = plan->layer_tangents + static_cast<std::size_t>(
        plan->activation_offsets.back()) * plan->capacity_query;
    error = cudaMemcpy(outputs, result, sizeof(double) *
        static_cast<std::size_t>(plan->output_width) * n_query,
        cudaMemcpyDeviceToHost);
  }
  if (error == cudaSuccess) add_d2h(plan, sizeof(double) *
      static_cast<std::size_t>(plan->output_width) * n_query);
  if (error == cudaSuccess)
    error = cudaMemcpy(outputs_dot, result_dot, sizeof(double) *
        static_cast<std::size_t>(plan->output_width) * n_query,
        cudaMemcpyDeviceToHost);
  if (error == cudaSuccess) add_d2h(plan, sizeof(double) *
      static_cast<std::size_t>(plan->output_width) * n_query);
  return static_cast<int>(error);
}

extern "C" int fortml_cuda_mlp_chain_vjp(
    void *opaque_plan, const double *query_x, const double *output_bar,
    int n_query, double *query_x_bar, double *weights_bar, double *biases_bar) {
  ChainPlan *plan = static_cast<ChainPlan *>(opaque_plan);
  const std::size_t input_count = plan == nullptr ? 0 :
      static_cast<std::size_t>(plan->input_width) * n_query;
  const std::size_t output_count = plan == nullptr ? 0 :
      static_cast<std::size_t>(plan->output_width) * n_query;
  if (plan == nullptr || query_x == nullptr || output_bar == nullptr ||
      query_x_bar == nullptr || weights_bar == nullptr || biases_bar == nullptr ||
      n_query < 1 || !finite_array(query_x, input_count) ||
      !finite_array(output_bar, output_count))
    return static_cast<int>(cudaErrorInvalidValue);
  cudaError_t error = cudaSetDevice(plan->device_index);
  if (error == cudaSuccess) error = ensure_workspace(plan, n_query);
  if (error == cudaSuccess)
    error = cudaMemcpy(plan->input, query_x, sizeof(double) * input_count,
                       cudaMemcpyHostToDevice);
  if (error == cudaSuccess) add_h2d(plan, sizeof(double) * input_count);
  if (error == cudaSuccess) error = run_forward(plan, n_query);
  if (error == cudaSuccess)
    error = cudaMemcpy(plan->bar_a, output_bar,
        sizeof(double) * output_count, cudaMemcpyHostToDevice);
  if (error == cudaSuccess) add_h2d(plan, sizeof(double) * output_count);
  if (error == cudaSuccess)
    error = cudaMemset(plan->weights_bar, 0,
                       sizeof(double) * plan->total_weights);
  if (error == cudaSuccess)
    error = cudaMemset(plan->biases_bar, 0,
                       sizeof(double) * plan->total_biases);
  double *current_bar = plan->bar_a;
  double *previous_bar = plan->bar_b;
  for (int layer = plan->n_layers - 1;
       error == cudaSuccess && layer >= 0; --layer) {
    const int input_width = plan->layer_sizes_host[layer];
    const int output_width = plan->layer_sizes_host[layer + 1];
    const int count = output_width * n_query;
    const double *preact = plan->layer_preacts + static_cast<std::size_t>(
        plan->activation_offsets[layer]) * plan->capacity_query;
    chain_vjp_zbar_kernel<<<(count + kThreads - 1) / kThreads, kThreads>>>(
        preact, plan->activations_host[layer], count, current_bar);
    error = cudaGetLastError();
    const double *source = layer == 0 ? plan->input :
        plan->layer_outputs + static_cast<std::size_t>(
            plan->activation_offsets[layer - 1]) * plan->capacity_query;
    if (error == cudaSuccess)
      chain_vjp_parameter_kernel<<<(output_width * input_width + kThreads - 1) /
                                       kThreads,
                                   kThreads>>>(
          source, current_bar, input_width, output_width, n_query,
          plan->weight_offsets[layer], plan->bias_offsets[layer],
          plan->weights_bar, plan->biases_bar);
    if (error == cudaSuccess) error = cudaGetLastError();
    if (error == cudaSuccess)
      chain_vjp_input_kernel<<<(input_width * n_query + kThreads - 1) /
                                  kThreads,
                              kThreads>>>(
          plan->weights, plan->weight_offsets[layer], input_width,
          output_width, current_bar, n_query, previous_bar);
    if (error == cudaSuccess) error = cudaGetLastError();
    std::swap(current_bar, previous_bar);
  }
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  if (error == cudaSuccess)
    error = cudaMemcpy(query_x_bar, current_bar,
                       sizeof(double) * input_count, cudaMemcpyDeviceToHost);
  if (error == cudaSuccess) add_d2h(plan, sizeof(double) * input_count);
  if (error == cudaSuccess)
    error = cudaMemcpy(weights_bar, plan->weights_bar,
                       sizeof(double) * plan->total_weights,
                       cudaMemcpyDeviceToHost);
  if (error == cudaSuccess) add_d2h(plan, sizeof(double) * plan->total_weights);
  if (error == cudaSuccess)
    error = cudaMemcpy(biases_bar, plan->biases_bar,
                       sizeof(double) * plan->total_biases,
                       cudaMemcpyDeviceToHost);
  if (error == cudaSuccess) add_d2h(plan, sizeof(double) * plan->total_biases);
  return static_cast<int>(error);
}

extern "C" int fortml_cuda_mlp_chain_transfer_stats(
    void *opaque_plan, std::uint64_t *host_to_device_bytes,
    std::uint64_t *device_to_host_bytes, std::uint64_t *resident_bytes) {
  ChainPlan *plan = static_cast<ChainPlan *>(opaque_plan);
  if (plan == nullptr || host_to_device_bytes == nullptr ||
      device_to_host_bytes == nullptr || resident_bytes == nullptr)
    return static_cast<int>(cudaErrorInvalidValue);
  *host_to_device_bytes = plan->host_to_device_bytes;
  *device_to_host_bytes = plan->device_to_host_bytes;
  *resident_bytes = plan->resident_bytes;
  return 0;
}

extern "C" int fortml_cuda_mlp_chain_destroy(void *opaque_plan) {
  ChainPlan *plan = static_cast<ChainPlan *>(opaque_plan);
  if (plan == nullptr) return static_cast<int>(cudaErrorInvalidValue);
  destroy_plan(plan);
  return 0;
}
