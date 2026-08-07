#include "../src/mlp/fortml_cuda_adamw.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>

int main() {
  if (fortml_cuda_adamw_available() == 0) {
    std::printf("CUDA AdamW unavailable; test skipped\n");
    return 0;
  }
  constexpr int n = 5;
  constexpr int steps = 7;
  const double learning_rate = 0.035;
  const double beta1 = 0.81;
  const double beta2 = 0.93;
  const double epsilon = 1.0e-6;
  const double weight_decay = 0.17;
  double parameters[n] = {0.2, -0.1, 0.3, -0.25, 0.05};
  double first[n] = {0.0, 0.0, 0.0, 0.0, 0.0};
  double second[n] = {0.0, 0.0, 0.0, 0.0, 0.0};
  void* plan = nullptr;
  if (fortml_cuda_adamw_plan_create(
          parameters, first, second, n, learning_rate, beta1, beta2,
          epsilon, weight_decay, 0, &plan) != 0 || plan == nullptr)
    return 2;

  double* device_gradient = nullptr;
  if (cudaMalloc(&device_gradient, sizeof(double) * n) != cudaSuccess) return 3;
  for (int step = 0; step < steps; ++step) {
    double gradient[n];
    for (int i = 0; i < n; ++i)
      gradient[i] = parameters[i] - 0.07 * (i + 1) + 0.01 * step;
    for (int i = 0; i < n; ++i) {
      first[i] = beta1 * first[i] + (1.0 - beta1) * gradient[i];
      second[i] = beta2 * second[i] +
          (1.0 - beta2) * gradient[i] * gradient[i];
      const double bias1 = 1.0 - std::pow(beta1, step + 1);
      const double bias2 = 1.0 - std::pow(beta2, step + 1);
      const double first_hat = first[i] / bias1;
      const double second_hat = second[i] / bias2;
      parameters[i] = (1.0 - learning_rate * weight_decay) * parameters[i] -
          learning_rate * first_hat / (std::sqrt(second_hat) + epsilon);
    }
    if (cudaMemcpy(device_gradient, gradient, sizeof(double) * n,
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        fortml_cuda_adamw_plan_step(plan, device_gradient) != 0)
      return 4;
  }

  double actual_parameters[n], actual_first[n], actual_second[n];
  int actual_steps = 0;
  const int status = fortml_cuda_adamw_plan_download(
      plan, actual_parameters, actual_first, actual_second, &actual_steps);
  cudaFree(device_gradient);
  fortml_cuda_adamw_plan_destroy(plan);
  double error = 0.0;
  for (int i = 0; i < n; ++i) {
    error = fmax(error, fabs(actual_parameters[i] - parameters[i]));
    error = fmax(error, fabs(actual_first[i] - first[i]));
    error = fmax(error, fabs(actual_second[i] - second[i]));
  }
  if (status != 0 || actual_steps != steps || error > 3.0e-13) return 5;
  std::printf("PASS CUDA AdamW resident-state oracle (max error %.3e)\n", error);
  return 0;
}
