#include "../src/mlp/fortml_cuda_mlp_chain.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

struct Fixture {
  int sizes[4] = {2, 3, 2, 1};
  int activations[3] = {2, 3, 1};
  int weight_offsets[3] = {0, 6, 12};
  int bias_offsets[3] = {0, 3, 5};
  int total_weights = 14;
  int total_biases = 6;
};

double activate(double x, int activation) {
  switch (activation) {
    case 2: return std::tanh(x);
    case 3: return std::max(x, 0.0);
    case 4: return 0.5 * x * (1.0 + std::tanh(
        0.79788456080286535588 * (x + 0.044715 * x * x * x)));
    case 5: return x / (1.0 + std::exp(-x));
    case 6: return x >= 0.0 ? x : std::exp(x) - 1.0;
    case 7: return x > 20.0 ? x : (x < -20.0 ? std::exp(x) :
        std::log1p(std::exp(x)));
    case 8: return x >= 0.0 ? x : 0.01 * x;
    default: return x;
  }
}

double derivative(double x, int activation) {
  switch (activation) {
    case 2: { const double t = std::tanh(x); return 1.0 - t * t; }
    case 3: return x >= 0.0 ? 1.0 : 0.0;
    case 4: {
      constexpr double c = 0.79788456080286535588;
      constexpr double a = 0.044715;
      const double x2 = x * x;
      const double t = std::tanh(c * (x + a * x * x2));
      return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * c *
          (1.0 + 3.0 * a * x2);
    }
    case 5: {
      const double s = 1.0 / (1.0 + std::exp(-x));
      return s + x * s * (1.0 - s);
    }
    case 6: return x >= 0.0 ? 1.0 : std::exp(x);
    case 7: return 1.0 / (1.0 + std::exp(-x));
    case 8: return x >= 0.0 ? 1.0 : 0.01;
    default: return 1.0;
  }
}

void forward(const Fixture &f, const std::vector<double> &weights,
             const std::vector<double> &biases, const std::vector<double> &x,
             int n_query, std::vector<std::vector<double>> &values,
             std::vector<std::vector<double>> &preacts) {
  values.assign(3, {});
  preacts.assign(3, {});
  std::vector<double> source = x;
  for (int layer = 0; layer < 3; ++layer) {
    const int in = f.sizes[layer];
    const int out = f.sizes[layer + 1];
    values[layer].resize(out * n_query);
    preacts[layer].resize(out * n_query);
    for (int j = 0; j < out; ++j) {
      for (int q = 0; q < n_query; ++q) {
        double z = biases[f.bias_offsets[layer] + j];
        for (int i = 0; i < in; ++i)
          z += weights[f.weight_offsets[layer] + j * in + i] *
              source[i * n_query + q];
        preacts[layer][j * n_query + q] = z;
        values[layer][j * n_query + q] = activate(z, f.activations[layer]);
      }
    }
    source = values[layer];
  }
}

void jvp(const Fixture &f, const std::vector<double> &weights,
         const std::vector<double> &biases, const std::vector<double> &wdot,
         const std::vector<double> &bdot, const std::vector<double> &x,
         const std::vector<double> &xdot, int n_query,
         std::vector<double> &out, std::vector<double> &out_dot) {
  std::vector<double> source = x, source_dot = xdot;
  for (int layer = 0; layer < 3; ++layer) {
    const int in = f.sizes[layer];
    const int width = f.sizes[layer + 1];
    std::vector<double> destination(width * n_query);
    std::vector<double> destination_dot(width * n_query);
    for (int j = 0; j < width; ++j) {
      for (int q = 0; q < n_query; ++q) {
        double z = biases[f.bias_offsets[layer] + j];
        double z_dot = bdot[f.bias_offsets[layer] + j];
        for (int i = 0; i < in; ++i) {
          const int k = f.weight_offsets[layer] + j * in + i;
          const double xv = source[i * n_query + q];
          z += weights[k] * xv;
          z_dot += weights[k] * source_dot[i * n_query + q] + wdot[k] * xv;
        }
        destination[j * n_query + q] = activate(z, f.activations[layer]);
        destination_dot[j * n_query + q] = derivative(z, f.activations[layer]) *
            z_dot;
      }
    }
    source = std::move(destination);
    source_dot = std::move(destination_dot);
  }
  out = std::move(source);
  out_dot = std::move(source_dot);
}

void vjp(const Fixture &f, const std::vector<double> &weights,
         const std::vector<double> &biases, const std::vector<double> &x,
         const std::vector<double> &output_bar, int n_query,
         std::vector<double> &x_bar, std::vector<double> &weights_bar,
         std::vector<double> &biases_bar) {
  std::vector<std::vector<double>> values, preacts;
  forward(f, weights, biases, x, n_query, values, preacts);
  std::vector<double> current = output_bar;
  weights_bar.assign(f.total_weights, 0.0);
  biases_bar.assign(f.total_biases, 0.0);
  for (int layer = 2; layer >= 0; --layer) {
    const int in = f.sizes[layer];
    const int out = f.sizes[layer + 1];
    std::vector<double> zbar(out * n_query);
    for (int j = 0; j < out; ++j)
      for (int q = 0; q < n_query; ++q)
        zbar[j * n_query + q] = current[j * n_query + q] *
            derivative(preacts[layer][j * n_query + q], f.activations[layer]);
    const std::vector<double> &source = layer == 0 ? x : values[layer - 1];
    for (int j = 0; j < out; ++j) {
      for (int i = 0; i < in; ++i) {
        double sum = 0.0;
        for (int q = 0; q < n_query; ++q)
          sum += zbar[j * n_query + q] * source[i * n_query + q];
        weights_bar[f.weight_offsets[layer] + j * in + i] = sum;
      }
      double sum = 0.0;
      for (int q = 0; q < n_query; ++q) sum += zbar[j * n_query + q];
      biases_bar[f.bias_offsets[layer] + j] = sum;
    }
    std::vector<double> previous(in * n_query, 0.0);
    for (int i = 0; i < in; ++i)
      for (int q = 0; q < n_query; ++q)
        for (int j = 0; j < out; ++j)
          previous[i * n_query + q] +=
              weights[f.weight_offsets[layer] + j * in + i] *
              zbar[j * n_query + q];
    current = std::move(previous);
  }
  x_bar = std::move(current);
}

double max_error(const std::vector<double> &actual,
                 const std::vector<double> &expected) {
  double result = 0.0;
  if (actual.size() != expected.size()) return 1.0e300;
  for (std::size_t i = 0; i < actual.size(); ++i)
    result = std::max(result, std::abs(actual[i] - expected[i]));
  return result;
}

}  // namespace

int main() {
  if (fortml_cuda_mlp_chain_available() == 0) {
    std::printf("CUDA MLP chain unavailable; test skipped\n");
    return 0;
  }
  if (cudaSetDevice(0) != cudaSuccess) return 2;
  constexpr int n_query = 4;
  const Fixture f;
  std::vector<double> weights(f.total_weights), biases(f.total_biases);
  std::vector<double> weights_dot(f.total_weights), biases_dot(f.total_biases);
  std::vector<double> x(2 * n_query), x_dot(2 * n_query), output_bar(n_query);
  for (int i = 0; i < f.total_weights; ++i) {
    weights[i] = 0.04 * (i + 1) - 0.25;
    weights_dot[i] = 0.03 * (i + 1) - 0.17;
  }
  for (int i = 0; i < f.total_biases; ++i) {
    biases[i] = 0.07 * (i + 1) - 0.19;
    biases_dot[i] = -0.02 * (i + 1) + 0.09;
  }
  for (int i = 0; i < 2 * n_query; ++i) {
    x[i] = 0.13 * (i + 1) - 0.41;
    x_dot[i] = -0.11 * (i + 1) + 0.23;
  }
  for (int i = 0; i < n_query; ++i) output_bar[i] = 0.17 * (i + 1) - 0.31;

  std::vector<std::vector<double>> values, preacts;
  forward(f, weights, biases, x, n_query, values, preacts);
  const std::vector<double> expected = values.back();
  std::vector<double> actual(1 * n_query, -7.0);
  void *plan = nullptr;
  if (fortml_cuda_mlp_chain_create(f.sizes, f.activations, weights.data(),
                                   biases.data(), 3, 0, &plan) != 0 ||
      plan == nullptr)
    return 3;
  if (fortml_cuda_mlp_chain_predict(plan, x.data(), n_query, actual.data()) != 0)
    return 4;
  double error = max_error(actual, expected);
  if (error > 3.0e-13) return 5;

  std::vector<double> expected_dot, actual_dot(n_query, -11.0);
  std::vector<double> expected_jvp;
  jvp(f, weights, biases, weights_dot, biases_dot, x, x_dot, n_query,
      expected_jvp, expected_dot);
  std::fill(actual.begin(), actual.end(), -13.0);
  if (fortml_cuda_mlp_chain_jvp(plan, x.data(), x_dot.data(), weights_dot.data(),
                                biases_dot.data(), n_query, actual.data(),
                                actual_dot.data()) != 0)
    return 6;
  if (max_error(actual, expected_jvp) > 3.0e-13 ||
      max_error(actual_dot, expected_dot) > 3.0e-13)
    return 7;

  std::vector<double> expected_x_bar, expected_weights_bar, expected_biases_bar;
  std::vector<double> actual_x_bar(2 * n_query, -17.0);
  std::vector<double> actual_weights_bar(f.total_weights, -19.0);
  std::vector<double> actual_biases_bar(f.total_biases, -23.0);
  vjp(f, weights, biases, x, output_bar, n_query, expected_x_bar,
      expected_weights_bar, expected_biases_bar);
  if (fortml_cuda_mlp_chain_vjp(plan, x.data(), output_bar.data(), n_query,
                                actual_x_bar.data(), actual_weights_bar.data(),
                                actual_biases_bar.data()) != 0)
    return 8;
  if (max_error(actual_x_bar, expected_x_bar) > 3.0e-13 ||
      max_error(actual_weights_bar, expected_weights_bar) > 3.0e-13 ||
      max_error(actual_biases_bar, expected_biases_bar) > 3.0e-13)
    return 9;

  // A second batch uses the same resident topology and model.  Counters must
  // report model upload plus explicit products, with nonzero permanent state.
  const double repeat_x[2 * 2] = {-0.2, 0.6, 0.9, -0.3};
  double repeat_y[2] = {-5.0, -5.0};
  if (fortml_cuda_mlp_chain_predict(plan, repeat_x, 2, repeat_y) != 0)
    return 10;
  std::uint64_t h2d = 0, d2h = 0, resident = 0;
  if (fortml_cuda_mlp_chain_transfer_stats(plan, &h2d, &d2h, &resident) != 0)
    return 11;
  const std::uint64_t model_bytes =
      sizeof(double) * (f.total_weights + f.total_biases) +
      sizeof(int) * (3 * 2 + 1);
  if (h2d <= model_bytes || d2h == 0 || resident <= model_bytes) return 12;
  if (fortml_cuda_mlp_chain_destroy(plan) != 0) return 13;
  std::printf("PASS CUDA MLP chain oracle (max error %.3e)\n", error);
  return 0;
}
