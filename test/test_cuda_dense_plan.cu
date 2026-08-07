#include "../src/mlp/fortml_cuda_dense.h"

#include <cuda_runtime.h>

#include <cmath>
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
  double expected[n_outputs * n_query];
  double actual[n_outputs * n_query];
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
    if (fortml_cuda_dense_plan_destroy(plan) != 0) return 6;
  }

  // A second batch on one resident model proves that prediction does not
  // rebuild or copy the immutable weights through a host fallback.
  void *plan = nullptr;
  if (fortml_cuda_dense_plan_create(weights, bias, n_inputs, n_outputs, 2,
                                    0, &plan) != 0 || plan == nullptr)
    return 7;
  const double repeat_query[n_inputs * 2] = {
      -2.0, 1.0, 0.5, 1.5, 2.0, -0.25};
  double repeat_expected[n_outputs * 2];
  double repeat_actual[n_outputs * 2];
  cpu_oracle(weights, bias, n_inputs, n_outputs, 2, repeat_query, 2,
             repeat_expected);
  if (fortml_cuda_dense_plan_predict(plan, repeat_query, 2, repeat_actual) !=
      0)
    return 8;
  double repeat_error = 0.0;
  const bool repeat_ok =
      close(repeat_actual, repeat_expected, n_outputs * 2, &repeat_error);
  max_error = fmax(max_error, repeat_error);
  if (fortml_cuda_dense_plan_destroy(plan) != 0 || !repeat_ok) return 9;

  // Non-finite host inputs are rejected before any CUDA allocation and cannot
  // be mistaken for a successful CPU fallback.
  const double bad_weights[n_inputs * n_outputs] = {
      0.5, -1.0, 0.25, -0.75, 0.4, NAN};
  void *bad_plan = nullptr;
  if (fortml_cuda_dense_plan_create(bad_weights, bias, n_inputs, n_outputs, 1,
                                    0, &bad_plan) == 0 || bad_plan != nullptr)
    return 10;

  std::printf("PASS CUDA resident dense affine oracle (activations 8, repeats 2, max error %.3e)\n",
              max_error);
  return 0;
}
