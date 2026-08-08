#include "../src/mlp/fortml_cuda_rmsprop.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <limits>

namespace {

bool refused_create(const double* parameters, const double* square,
                    const double* mean, const double* momentum, int n,
                    double learning_rate, double decay, double epsilon,
                    double momentum_decay, int centered, int device,
                    const char* description) {
  void* plan = reinterpret_cast<void*>(0x1);
  const int status = fortml_cuda_rmsprop_plan_create(
      parameters, square, mean, momentum, n, learning_rate, decay, epsilon,
      momentum_decay, centered, device, &plan);
  if (status == 0 || plan != nullptr) {
    std::fprintf(stderr, "RMSprop refusal missing: %s (status=%d)\n",
                 description, status);
    if (plan != nullptr) fortml_cuda_rmsprop_plan_destroy(plan);
    return false;
  }
  return true;
}

}  // namespace

int main() {
  int device_count = 0;
  const cudaError_t device_error = cudaGetDeviceCount(&device_count);
  const int expected_available =
      device_error == cudaSuccess && device_count > 0 ? 1 : 0;
  const int advertised_available = fortml_cuda_rmsprop_available();
  if (advertised_available != expected_available) {
    std::fprintf(stderr,
                 "RMSprop availability mismatch: advertised %d, expected %d\n",
                 advertised_available, expected_available);
    return 1;
  }

  constexpr int n = 4;
  double initial_parameters[n] = {0.2, -0.1, 0.3, -0.25};
  double initial_square[n] = {0.0, 0.0, 0.0, 0.0};
  double initial_mean[n] = {0.0, 0.0, 0.0, 0.0};
  double initial_buffer[n] = {0.0, 0.0, 0.0, 0.0};
  const double learning_rate = 0.08;
  const double decay = 0.8;
  const double epsilon = 1.0e-5;
  const double momentum = 0.2;
  if (!refused_create(nullptr, initial_square, initial_mean, initial_buffer,
                      n, learning_rate, decay, epsilon, momentum, 1, 0,
                      "null parameters") ||
      !refused_create(initial_parameters, initial_square, initial_mean,
                      initial_buffer, n, learning_rate, 1.0, epsilon,
                      momentum, 1, 0, "decay at one") ||
      !refused_create(initial_parameters, initial_square, initial_mean,
                      initial_buffer, n, learning_rate, decay, 0.0, momentum,
                      1, 0, "nonpositive epsilon") ||
      !refused_create(initial_parameters, initial_square, initial_mean,
                      initial_buffer, n, learning_rate, decay, epsilon,
                      momentum, 2, 0, "invalid centered flag"))
    return 2;
  double bad_parameter[n];
  for (int i = 0; i < n; ++i) bad_parameter[i] = initial_parameters[i];
  bad_parameter[1] = std::numeric_limits<double>::quiet_NaN();
  if (!refused_create(bad_parameter, initial_square, initial_mean,
                      initial_buffer, n, learning_rate, decay, epsilon,
                      momentum, 1, 0, "nonfinite parameter"))
    return 2;
  double bad_square[n] = {0.0, -1.0, 0.0, 0.0};
  if (!refused_create(initial_parameters, bad_square, initial_mean,
                      initial_buffer, n, learning_rate, decay, epsilon,
                      momentum, 1, 0, "negative square average"))
    return 2;
  if (advertised_available == 0) {
    std::printf("CUDA RMSprop unavailable; test skipped\n");
    return 0;
  }
  constexpr int steps = 5;
  double parameters[n] = {0.2, -0.1, 0.3, -0.25};
  double square[n] = {0.0, 0.0, 0.0, 0.0};
  double mean[n] = {0.0, 0.0, 0.0, 0.0};
  double buffer[n] = {0.0, 0.0, 0.0, 0.0};
  void* plan = nullptr;
  if (fortml_cuda_rmsprop_plan_create(
          parameters, square, mean, buffer, n, learning_rate, decay,
          epsilon, momentum, 1, 0, &plan) != 0 || plan == nullptr)
    return 2;
  int ignored_steps = 0;
  if (fortml_cuda_rmsprop_plan_step(plan, nullptr) == 0 ||
      fortml_cuda_rmsprop_plan_download(plan, nullptr, square, mean, buffer,
                                        &ignored_steps) == 0) {
    std::fprintf(stderr, "RMSprop invalid lifecycle arguments were accepted\n");
    fortml_cuda_rmsprop_plan_destroy(plan);
    return 3;
  }
  double* device_gradient = nullptr;
  if (cudaMalloc(&device_gradient, sizeof(double) * n) != cudaSuccess) return 4;
  for (int step = 0; step < steps; ++step) {
    double gradient[n];
    for (int i = 0; i < n; ++i) gradient[i] = parameters[i] - 0.1 * (i + 1);
    for (int i = 0; i < n; ++i) {
      square[i] = decay * square[i] + (1.0 - decay) * gradient[i] * gradient[i];
      mean[i] = decay * mean[i] + (1.0 - decay) * gradient[i];
      const double variance = fmax(square[i] - mean[i] * mean[i], 0.0);
      const double direction = gradient[i] / (sqrt(variance) + epsilon);
      buffer[i] = momentum * buffer[i] + direction;
      parameters[i] -= learning_rate * buffer[i];
    }
    if (cudaMemcpy(device_gradient, gradient, sizeof(double) * n,
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        fortml_cuda_rmsprop_plan_step(plan, device_gradient) != 0)
      return 5;
  }
  double actual_parameters[n], actual_square[n], actual_mean[n], actual_buffer[n];
  int actual_steps = 0;
  const int status = fortml_cuda_rmsprop_plan_download(
      plan, actual_parameters, actual_square, actual_mean, actual_buffer,
      &actual_steps);
  cudaFree(device_gradient);
  fortml_cuda_rmsprop_plan_destroy(plan);
  double error = 0.0;
  for (int i = 0; i < n; ++i) {
    error = fmax(error, fabs(actual_parameters[i] - parameters[i]));
    error = fmax(error, fabs(actual_square[i] - square[i]));
    error = fmax(error, fabs(actual_mean[i] - mean[i]));
    error = fmax(error, fabs(actual_buffer[i] - buffer[i]));
  }
  if (status != 0 || actual_steps != steps || error > 2.0e-12) return 6;
  std::printf("PASS CUDA RMSprop resident-state oracle (max error %.3e)\n", error);
  return 0;
}
