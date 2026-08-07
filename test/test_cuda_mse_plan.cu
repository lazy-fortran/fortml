#include "../src/validation/fortml_cuda_mse_plan.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>

int main() {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    std::printf("CUDA MSE resident plan unavailable; test skipped\n");
    return 0;
  }
  constexpr int n_samples = 4;
  constexpr int n_outputs = 2;
  const double target[n_samples * n_outputs] = {
      1.0, -2.0, 0.5, 4.0, 2.0, 1.0, -1.5, 3.0};
  const double prediction[n_samples * n_outputs] = {
      0.0, -1.0, 1.5, 2.0, 1.0, 2.0, -0.5, 4.0};
  const double weight[n_samples] = {1.0, 2.0, 0.5, 3.0};
  double expected = 0.0;
  double denominator = 0.0;
  for (int row = 0; row < n_samples; ++row) {
    denominator += weight[row];
    for (int output = 0; output < n_outputs; ++output) {
      const int index = row + n_samples * output;
      const double residual = target[index] - prediction[index];
      expected += weight[row] * residual * residual;
    }
  }
  expected /= denominator * n_outputs;

  void* plan = nullptr;
  if (fortml_cuda_mse_plan_create(target, prediction, weight, n_samples,
                                  n_outputs, 0, &plan) != 0 || plan == nullptr)
    return 2;
  double value = -17.0;
  double max_error = 0.0;
  for (int repetition = 0; repetition < 5; ++repetition) {
    if (fortml_cuda_mse_plan_execute(plan, &value) != 0) return 3;
    max_error = fmax(max_error, fabs(value - expected));
  }
  const int destroy_status = fortml_cuda_mse_plan_destroy(plan);
  if (destroy_status != 0 || max_error > 3.0e-13) return 4;
  std::printf("PASS CUDA resident MSE plan oracle (max error %.3e, repeats 5)\n",
              max_error);
  return 0;
}
