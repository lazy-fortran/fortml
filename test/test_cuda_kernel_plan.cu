#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <vector>

extern "C" void* fortml_cuda_kernel_plan_create(
    const double*, const int*, const double*, const double*, int, int, int,
    int*);
extern "C" int fortml_cuda_kernel_plan_destroy(void*);
extern "C" int fortml_cuda_kernel_plan_matvec(
    void*, const double*, double*, double);
extern "C" int fortml_cuda_kernel_plan_matmat(
    void*, const double*, double*, int, double);

namespace {

constexpr int n_samples = 5;
constexpr int n_features = 2;
constexpr double variance = 1.7;
constexpr double lengthscale = 0.9;
constexpr double constant_variance = 0.4;
constexpr double diagonal_shift = 0.2;

double kernel_value(const std::vector<double>& points, int left, int right) {
  const double dx = points[left] - points[right];
  const double dy = points[n_samples + left] - points[n_samples + right];
  const double distance = dx * dx + dy * dy;
  return variance * std::exp(-0.5 * distance / (lengthscale * lengthscale)) +
      constant_variance;
}

bool close(double left, double right) {
  return std::abs(left - right) < 2.0e-12;
}

}  // namespace

int main() {
  const std::vector<double> points = {
      0.0, 0.4, 0.9, 1.3, 1.8,
      -0.2, 0.1, 0.7, 1.1, 1.4};
  const std::vector<double> input = {1.0, -0.5, 0.25, 0.75, -1.2};
  const std::vector<double> input_matrix = {
      1.0, -0.5, 0.25, 0.75, -1.2,
      -0.3, 0.2, 0.8, -0.4, 1.1};
  const int program_kind[] = {1, 6, 8};
  const double program_variance[] = {variance, constant_variance, 0.0};
  const double program_lengthscale[] = {lengthscale, 1.0, 0.0};

  int status = 0;
  void* plan = fortml_cuda_kernel_plan_create(
      points.data(), program_kind, program_variance, program_lengthscale,
      n_samples, n_features, 3, &status);
  if (plan == nullptr || status != 0) return 1;

  double* device_input = nullptr;
  double* device_output = nullptr;
  double* device_matrix_input = nullptr;
  double* device_matrix_output = nullptr;
  if (cudaMalloc(&device_input, sizeof(double) * n_samples) != cudaSuccess ||
      cudaMalloc(&device_output, sizeof(double) * n_samples) != cudaSuccess ||
      cudaMalloc(&device_matrix_input, sizeof(double) * 2 * n_samples) !=
          cudaSuccess ||
      cudaMalloc(&device_matrix_output, sizeof(double) * 2 * n_samples) !=
          cudaSuccess) {
    fortml_cuda_kernel_plan_destroy(plan);
    return 2;
  }
  cudaMemcpy(device_input, input.data(), sizeof(double) * n_samples,
             cudaMemcpyHostToDevice);
  if (fortml_cuda_kernel_plan_matvec(
          plan, device_input, device_output, diagonal_shift) != 0) {
    fortml_cuda_kernel_plan_destroy(plan);
    return 3;
  }
  std::vector<double> output(n_samples);
  cudaMemcpy(output.data(), device_output, sizeof(double) * n_samples,
             cudaMemcpyDeviceToHost);
  for (int i = 0; i < n_samples; ++i) {
    double expected = diagonal_shift * input[i];
    for (int j = 0; j < n_samples; ++j) {
      expected += kernel_value(points, i, j) * input[j];
    }
    if (!close(output[i], expected)) {
      std::fprintf(stderr, "matvec mismatch i=%d actual=%.17g expected=%.17g\\n",
                   i, output[i], expected);
      return 4;
    }
  }

  cudaMemcpy(device_matrix_input, input_matrix.data(),
             sizeof(double) * 2 * n_samples, cudaMemcpyHostToDevice);
  if (fortml_cuda_kernel_plan_matmat(
          plan, device_matrix_input, device_matrix_output, 2,
          diagonal_shift) != 0) {
    fortml_cuda_kernel_plan_destroy(plan);
    return 5;
  }
  std::vector<double> matrix_output(2 * n_samples);
  cudaMemcpy(matrix_output.data(), device_matrix_output,
             sizeof(double) * 2 * n_samples, cudaMemcpyDeviceToHost);
  for (int rhs = 0; rhs < 2; ++rhs) {
    for (int i = 0; i < n_samples; ++i) {
      double expected = diagonal_shift * input_matrix[rhs * n_samples + i];
      for (int j = 0; j < n_samples; ++j) {
        expected += kernel_value(points, i, j) *
            input_matrix[rhs * n_samples + j];
      }
      if (!close(matrix_output[rhs * n_samples + i], expected)) {
        std::fprintf(stderr,
                     "matmat mismatch rhs=%d i=%d actual=%.17g expected=%.17g\\n",
                     rhs, i, matrix_output[rhs * n_samples + i], expected);
        return 6;
      }
    }
  }

  cudaFree(device_input);
  cudaFree(device_output);
  cudaFree(device_matrix_input);
  cudaFree(device_matrix_output);
  if (fortml_cuda_kernel_plan_destroy(plan) != 0) return 7;
  std::puts("test_cuda_kernel_plan: all checks passed");
  return 0;
}
