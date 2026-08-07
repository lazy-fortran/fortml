#include "../src/mlp/fortml_cuda_dense.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>

namespace {

double cpu_activate(double value, int activation) {
  switch (activation) {
    case 2:
      return std::tanh(value);
    case 3:
      return value > 0.0 ? value : 0.0;
    case 4:
      return 0.5 * value *
          (1.0 + std::tanh(0.79788456080286535588 *
                           (value + 0.044715 * value * value * value)));
    case 5:
      return value / (1.0 + std::exp(-value));
    case 6:
      return value >= 0.0 ? value : std::exp(value) - 1.0;
    case 7:
      if (value > 20.0) return value;
      if (value < -20.0) return std::exp(value);
      return std::log1p(std::exp(value));
    case 8:
      return value >= 0.0 ? value : 0.01 * value;
    case 1:
    default:
      return value;
  }
}

double cpu_activation_derivative(double value, int activation) {
  switch (activation) {
    case 2: {
      const double t = std::tanh(value);
      return 1.0 - t * t;
    }
    case 3:
      return value >= 0.0 ? 1.0 : 0.0;
    case 4: {
      constexpr double c = 0.79788456080286535588;
      constexpr double a = 0.044715;
      const double value2 = value * value;
      const double t = std::tanh(c * (value + a * value * value2));
      return 0.5 * (1.0 + t) +
          0.5 * value * (1.0 - t * t) * c * (1.0 + 3.0 * a * value2);
    }
    case 5: {
      const double sigmoid = 1.0 / (1.0 + std::exp(-value));
      return sigmoid + value * sigmoid * (1.0 - sigmoid);
    }
    case 6:
      return value >= 0.0 ? 1.0 : std::exp(value);
    case 7:
      return 1.0 / (1.0 + std::exp(-value));
    case 8:
      return value >= 0.0 ? 1.0 : 0.01;
    case 1:
    default:
      return 1.0;
  }
}

void cpu_oracle(const double *weights, const double *bias, int n_inputs,
                int n_outputs, int activation, const double *query_x,
                int n_query, double *outputs) {
  for (int output = 0; output < n_outputs; ++output) {
    for (int query = 0; query < n_query; ++query) {
      double value = bias[output];
      for (int input = 0; input < n_inputs; ++input)
        value += weights[output * n_inputs + input] *
            query_x[input * n_query + query];
      outputs[output * n_query + query] = cpu_activate(value, activation);
    }
  }
}

void cpu_jvp_oracle(const double *weights, const double *bias,
                    const double *weights_dot, const double *bias_dot,
                    int n_inputs, int n_outputs, int activation,
                    const double *query_x, const double *query_x_dot,
                    int n_query, double *outputs, double *outputs_dot) {
  for (int output = 0; output < n_outputs; ++output) {
    for (int query = 0; query < n_query; ++query) {
      double value = bias[output];
      double tangent = bias_dot[output];
      for (int input = 0; input < n_inputs; ++input) {
        const double x = query_x[input * n_query + query];
        value += weights[output * n_inputs + input] * x;
        tangent += weights[output * n_inputs + input] *
                query_x_dot[input * n_query + query] +
            weights_dot[output * n_inputs + input] * x;
      }
      outputs[output * n_query + query] = cpu_activate(value, activation);
      outputs_dot[output * n_query + query] =
          cpu_activation_derivative(value, activation) * tangent;
    }
  }
}

void cpu_vjp_oracle(const double *weights, const double *bias,
                    int n_inputs, int n_outputs, int activation,
                    const double *query_x, const double *output_bar,
                    int n_query, double *query_x_bar, double *weights_bar,
                    double *bias_bar) {
  for (int input = 0; input < n_inputs; ++input) {
    for (int query = 0; query < n_query; ++query) {
      double value = 0.0;
      for (int output = 0; output < n_outputs; ++output) {
        double affine = bias[output];
        for (int feature = 0; feature < n_inputs; ++feature)
          affine += weights[output * n_inputs + feature] *
              query_x[feature * n_query + query];
        value += weights[output * n_inputs + input] *
            output_bar[output * n_query + query] *
            cpu_activation_derivative(affine, activation);
      }
      query_x_bar[input * n_query + query] = value;
    }
  }
  for (int output = 0; output < n_outputs; ++output) {
    double bias_value = 0.0;
    for (int input = 0; input < n_inputs; ++input) {
      double value = 0.0;
      for (int query = 0; query < n_query; ++query) {
        double affine = bias[output];
        for (int feature = 0; feature < n_inputs; ++feature)
          affine += weights[output * n_inputs + feature] *
              query_x[feature * n_query + query];
        const double cotangent = output_bar[output * n_query + query] *
            cpu_activation_derivative(affine, activation);
        value += cotangent * query_x[input * n_query + query];
      }
      weights_bar[output * n_inputs + input] = value;
    }
    for (int query = 0; query < n_query; ++query) {
      double affine = bias[output];
      for (int input = 0; input < n_inputs; ++input)
        affine += weights[output * n_inputs + input] *
            query_x[input * n_query + query];
      bias_value += output_bar[output * n_query + query] *
          cpu_activation_derivative(affine, activation);
    }
    bias_bar[output] = bias_value;
  }
}

double cpu_train_mse_oracle(double *weights, double *bias, int n_inputs,
                            int n_outputs, int activation,
                            const double *query_x, const double *target,
                            int n_query, double learning_rate) {
  const double inv_count =
      1.0 / static_cast<double>(n_query * n_outputs);
  double loss = 0.0;
  double weight_gradient[6] = {};
  double bias_gradient[2] = {};
  for (int output = 0; output < n_outputs; ++output) {
    for (int query = 0; query < n_query; ++query) {
      double value = bias[output];
      for (int input = 0; input < n_inputs; ++input)
        value += weights[output * n_inputs + input] *
            query_x[input * n_query + query];
      const double residual = cpu_activate(value, activation) -
          target[output * n_query + query];
      const double scaled = residual *
          cpu_activation_derivative(value, activation);
      loss += 0.5 * residual * residual * inv_count;
      bias_gradient[output] += scaled * inv_count;
      for (int input = 0; input < n_inputs; ++input)
        weight_gradient[output * n_inputs + input] += scaled *
            query_x[input * n_query + query] * inv_count;
    }
  }
  for (int output = 0; output < n_outputs; ++output) {
    bias[output] -= learning_rate * bias_gradient[output];
    for (int input = 0; input < n_inputs; ++input)
      weights[output * n_inputs + input] -= learning_rate *
          weight_gradient[output * n_inputs + input];
  }
  return loss;
}

bool close(const double *actual, const double *expected, int count,
           double *max_error) {
  *max_error = 0.0;
  for (int i = 0; i < count; ++i)
    *max_error = fmax(*max_error, fabs(actual[i] - expected[i]));
  return *max_error < 3.0e-13;
}

}  // namespace

int main() {
  if (fortml_cuda_dense_available() == 0) {
    std::printf("CUDA dense plan unavailable; test skipped\n");
    return 0;
  }
  if (cudaSetDevice(0) != cudaSuccess) return 2;

  constexpr int n_inputs = 3;
  constexpr int n_outputs = 2;
  constexpr int n_query = 5;
  const double weights[n_inputs * n_outputs] = {
      0.5, -1.0, 0.25, -0.75, 0.4, 1.2};
  const double bias[n_outputs] = {-0.1, 0.2};
  const double query_x[n_inputs * n_query] = {
      -1.0, 0.0, 0.5, 2.0, -0.25,
      1.0, -0.5, 1.5, -2.0, 0.75,
      0.25, -1.0, 2.0, 0.5, -1.5};
  const double query_x_dot[n_inputs * n_query] = {
      0.25, -0.5, 1.0, -0.75, 0.2,
      -0.4, 0.8, -0.6, 0.3, -0.1,
      0.7, -0.2, 0.5, 0.9, -0.8};
  const double weights_dot[n_inputs * n_outputs] = {
      -0.2, 0.3, 0.1, 0.4, -0.5, 0.6};
  const double bias_dot[n_outputs] = {0.15, -0.25};
  const double output_bar[n_outputs * n_query] = {
      0.2, -0.4, 0.7, 0.1, -0.3,
      -0.6, 0.5, -0.2, 0.8, 0.15};
  double expected[n_outputs * n_query];
  double actual[n_outputs * n_query];
  double expected_dot[n_outputs * n_query];
  double actual_dot[n_outputs * n_query];
  double expected_query_bar[n_inputs * n_query];
  double actual_query_bar[n_inputs * n_query];
  double expected_weights_bar[n_inputs * n_outputs];
  double actual_weights_bar[n_inputs * n_outputs];
  double expected_bias_bar[n_outputs];
  double actual_bias_bar[n_outputs];
  double max_error = 0.0;

  // Exercise every activation exposed by fortml_mlp.  The CPU recurrence is
  // independent of the CUDA implementation and checks the complete output.
  for (int activation = 1; activation <= 8; ++activation) {
    void *plan = nullptr;
    if (fortml_cuda_dense_plan_create(weights, bias, n_inputs, n_outputs,
                                      activation, 0, &plan) != 0 ||
        plan == nullptr)
      return 3;
    cpu_oracle(weights, bias, n_inputs, n_outputs, activation, query_x,
               n_query, expected);
    if (fortml_cuda_dense_plan_predict(plan, query_x, n_query, actual) != 0)
      return 4;
    double activation_error = 0.0;
    if (!close(actual, expected, n_outputs * n_query, &activation_error))
      return 5;
    max_error = fmax(max_error, activation_error);
    cpu_jvp_oracle(weights, bias, weights_dot, bias_dot, n_inputs, n_outputs,
                   activation, query_x, query_x_dot, n_query, expected,
                   expected_dot);
    if (fortml_cuda_dense_plan_jvp(plan, query_x, query_x_dot, weights_dot,
                                   bias_dot, n_query, actual, actual_dot) != 0)
      return 6;
    double jvp_error = 0.0;
    if (!close(actual, expected, n_outputs * n_query, &jvp_error)) return 7;
    if (!close(actual_dot, expected_dot, n_outputs * n_query, &jvp_error))
      return 8;
    max_error = fmax(max_error, jvp_error);
    cpu_vjp_oracle(weights, bias, n_inputs, n_outputs, activation, query_x,
                   output_bar, n_query, expected_query_bar,
                   expected_weights_bar, expected_bias_bar);
    if (fortml_cuda_dense_plan_vjp(plan, query_x, output_bar, n_query,
                                   actual_query_bar, actual_weights_bar,
                                   actual_bias_bar) != 0)
      return 13;
    double vjp_error = 0.0;
    if (!close(actual_query_bar, expected_query_bar, n_inputs * n_query,
               &vjp_error))
      return 14;
    if (!close(actual_weights_bar, expected_weights_bar,
               n_inputs * n_outputs, &vjp_error))
      return 15;
    if (!close(actual_bias_bar, expected_bias_bar, n_outputs, &vjp_error))
      return 16;
    max_error = fmax(max_error, vjp_error);
    if (fortml_cuda_dense_plan_destroy(plan) != 0) return 6;
  }

  // A second batch on one resident model proves that prediction does not
  // rebuild or copy the immutable weights through a host fallback.
  void *plan = nullptr;
  if (fortml_cuda_dense_plan_create(weights, bias, n_inputs, n_outputs, 2,
                                    0, &plan) != 0 || plan == nullptr)
    return 9;
  const double repeat_query[n_inputs * 2] = {
      -2.0, 1.0, 0.5, 1.5, 2.0, -0.25};
  double repeat_expected[n_outputs * 2];
  double repeat_actual[n_outputs * 2];
  cpu_oracle(weights, bias, n_inputs, n_outputs, 2, repeat_query, 2,
             repeat_expected);
  if (fortml_cuda_dense_plan_predict(plan, repeat_query, 2, repeat_actual) !=
      0)
      return 10;
  double repeat_error = 0.0;
  const bool repeat_ok =
      close(repeat_actual, repeat_expected, n_outputs * 2, &repeat_error);
  max_error = fmax(max_error, repeat_error);
  if (fortml_cuda_dense_plan_destroy(plan) != 0 || !repeat_ok) return 11;

  // A single fixed MSE update forms the complete gradient on device and
  // mutates only the resident parameters.  The independent CPU recurrence
  // checks both the loss and the post-step snapshot, while transfer counters
  // prove that the model was uploaded once and the batch crossed the ABI
  // boundary explicitly.
  const double train_target[n_outputs * n_query] = {
      0.2, -0.1, 0.7, -0.3, 0.4,
      -0.5, 0.6, -0.2, 0.8, -0.15};
  double expected_train_weights[n_inputs * n_outputs];
  double expected_train_bias[n_outputs];
  std::copy(weights, weights + n_inputs * n_outputs, expected_train_weights);
  std::copy(bias, bias + n_outputs, expected_train_bias);
  const double expected_loss = cpu_train_mse_oracle(
      expected_train_weights, expected_train_bias, n_inputs, n_outputs, 2,
      query_x, train_target, n_query, 0.15);
  void *train_plan = nullptr;
  if (fortml_cuda_dense_plan_create(weights, bias, n_inputs, n_outputs, 2,
                                    0, &train_plan) != 0 ||
      train_plan == nullptr)
    return 17;
  double actual_loss = -1.0;
  if (fortml_cuda_dense_plan_train_mse(train_plan, query_x, train_target,
                                       n_query, 0.15, &actual_loss) != 0)
    return 18;
  if (std::fabs(actual_loss - expected_loss) > 3.0e-12) return 19;
  double actual_train_weights[n_inputs * n_outputs];
  double actual_train_bias[n_outputs];
  if (fortml_cuda_dense_plan_get_parameters(
          train_plan, actual_train_weights, actual_train_bias) != 0)
    return 20;
  double train_error = 0.0;
  if (!close(actual_train_weights, expected_train_weights,
             n_inputs * n_outputs, &train_error) ||
      !close(actual_train_bias, expected_train_bias, n_outputs, &train_error))
    return 21;
  std::uint64_t host_to_device_bytes = 0;
  std::uint64_t device_to_host_bytes = 0;
  std::uint64_t resident_bytes = 0;
  if (fortml_cuda_dense_plan_transfer_stats(
          train_plan, &host_to_device_bytes, &device_to_host_bytes,
          &resident_bytes) != 0 ||
      host_to_device_bytes < sizeof(double) *
          (n_inputs * n_outputs + n_outputs + n_inputs * n_query +
           n_outputs * n_query) ||
      device_to_host_bytes < sizeof(double) *
          (n_outputs + n_inputs * n_outputs + n_outputs) ||
      resident_bytes != sizeof(double) *
          (n_inputs * n_outputs + n_outputs))
    return 22;
  if (fortml_cuda_dense_plan_destroy(train_plan) != 0) return 23;

  // Non-finite host inputs are rejected before any CUDA allocation and cannot
  // be mistaken for a successful CPU fallback.
  const double bad_weights[n_inputs * n_outputs] = {
      0.5, -1.0, 0.25, -0.75, 0.4, NAN};
  void *bad_plan = nullptr;
  if (fortml_cuda_dense_plan_create(bad_weights, bias, n_inputs, n_outputs, 1,
                                    0, &bad_plan) == 0 || bad_plan != nullptr)
    return 12;

  std::printf("PASS CUDA resident dense affine oracle (activations 8, JVPs 8, VJPs 8, repeats 2, MSE update 1, max error %.3e)\n",
              fmax(max_error, train_error));
  return 0;
}
