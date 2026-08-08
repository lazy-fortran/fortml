#include "../src/mlp/fortml_cuda_adagrad.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>

int main() {
  if (fortml_cuda_adagrad_available() == 0) {
    std::printf("CUDA Adagrad unavailable; test skipped\n");
    return 0;
  }
  constexpr int n = 5;
  constexpr int steps = 8;
  const double learning_rate = 0.035;
  const double epsilon = 1.0e-6;
  double parameters[n] = {0.2, -0.1, 0.3, -0.25, 0.05};
  double accumulated[n] = {0.0, 0.0, 0.0, 0.0, 0.0};
  void* plan = nullptr;
  if (fortml_cuda_adagrad_plan_create(
          parameters, accumulated, n, learning_rate, epsilon, 0, &plan) != 0 ||
      plan == nullptr)
    return 2;

  double* device_gradient = nullptr;
  if (cudaMalloc(&device_gradient, sizeof(double) * n) != cudaSuccess) return 3;
  for (int step = 0; step < steps; ++step) {
    double gradient[n];
    for (int i = 0; i < n; ++i)
      gradient[i] = parameters[i] - 0.07 * (i + 1) + 0.01 * step;
    for (int i = 0; i < n; ++i) {
      accumulated[i] += gradient[i] * gradient[i];
      parameters[i] -= learning_rate * gradient[i] /
          (std::sqrt(accumulated[i]) + epsilon);
    }
    if (cudaMemcpy(device_gradient, gradient, sizeof(double) * n,
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        fortml_cuda_adagrad_plan_step(plan, device_gradient) != 0)
      return 4;
  }

  double actual_parameters[n], actual_accumulated[n];
  int actual_steps = 0;
  const int status = fortml_cuda_adagrad_plan_download(
      plan, actual_parameters, actual_accumulated, &actual_steps);
  cudaFree(device_gradient);
  fortml_cuda_adagrad_plan_destroy(plan);
  double error = 0.0;
  for (int i = 0; i < n; ++i) {
    error = fmax(error, fabs(actual_parameters[i] - parameters[i]));
    error = fmax(error, fabs(actual_accumulated[i] - accumulated[i]));
  }
  if (status != 0 || actual_steps != steps || error > 2.0e-13) return 5;
  std::printf("PASS CUDA Adagrad resident-state oracle (max error %.3e)\n",
              error);
  return 0;
}
