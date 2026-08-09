#include "../src/mlp/fortml_cuda_dense.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace {

constexpr int kOptimizerSgd = 1;
constexpr int kOptimizerAdam = 2;
constexpr int kOptimizerAdamW = 3;
constexpr int kInputs = 2;
constexpr int kOutputs = 1;
constexpr int kBatch = 4;
constexpr int kWeights = kInputs * kOutputs;
constexpr int kParameters = kWeights + kOutputs;

bool close(const double *actual, const double *expected, int count,
           double tolerance, double *max_error) {
  *max_error = 0.0;
  for (int i = 0; i < count; ++i)
    *max_error = fmax(*max_error, fabs(actual[i] - expected[i]));
  return *max_error <= tolerance;
}

double cpu_step(double *weights, double *bias, double *moment1,
                double *moment2, const double *query_x, const double *target,
                int optimizer_kind, double learning_rate, double beta1,
                double beta2, double epsilon, double weight_decay,
                int step) {
  double gradients[kParameters] = {};
  double loss = 0.0;
  for (int query = 0; query < kBatch; ++query) {
    const double value = bias[0] + weights[0] * query_x[query] +
        weights[1] * query_x[kBatch + query];
    const double residual = value - target[query];
    loss += 0.5 * residual * residual /
        static_cast<double>(kBatch * kOutputs);
    gradients[0] += residual * query_x[query] /
        static_cast<double>(kBatch * kOutputs);
    gradients[1] += residual * query_x[kBatch + query] /
        static_cast<double>(kBatch * kOutputs);
    gradients[2] += residual /
        static_cast<double>(kBatch * kOutputs);
  }
  for (int parameter = 0; parameter < kParameters; ++parameter) {
    double update = 0.0;
    if (optimizer_kind == kOptimizerSgd) {
      update = gradients[parameter];
    } else {
      moment1[parameter] = beta1 * moment1[parameter] +
          (1.0 - beta1) * gradients[parameter];
      moment2[parameter] = beta2 * moment2[parameter] +
          (1.0 - beta2) * gradients[parameter] * gradients[parameter];
      const double m_hat = moment1[parameter] /
          (1.0 - (beta1 == 0.0 ? 0.0 : std::pow(beta1, step)));
      const double v_hat = moment2[parameter] /
          (1.0 - (beta2 == 0.0 ? 0.0 : std::pow(beta2, step)));
      update = m_hat / (std::sqrt(v_hat) + epsilon);
    }
    if (optimizer_kind == kOptimizerAdamW)
      update += weight_decay * (parameter < kWeights ? weights[parameter]
                                                    : bias[0]);
    if (parameter < kWeights)
      weights[parameter] -= learning_rate * update;
    else
      bias[0] -= learning_rate * update;
  }
  return loss;
}

bool run_optimizer(int optimizer_kind, double learning_rate, double beta1,
                   double beta2, double epsilon, double weight_decay,
                   double *max_error) {
  const double initial_weights[kWeights] = {0.5, -0.25};
  const double initial_bias[kOutputs] = {0.1};
  const double query_x[kInputs * kBatch] = {
      -1.0, 0.0, 2.0, 0.5,
      0.75, -0.5, 1.25, -1.5};
  const double target[kOutputs * kBatch] = {0.2, -0.3, 0.6, 0.4};
  double expected_weights[kWeights];
  double expected_bias[kOutputs];
  double expected_moment1[kParameters] = {};
  double expected_moment2[kParameters] = {};
  std::memcpy(expected_weights, initial_weights, sizeof(initial_weights));
  std::memcpy(expected_bias, initial_bias, sizeof(initial_bias));

  void *plan = nullptr;
  if (fortml_cuda_dense_plan_create(initial_weights, initial_bias, kInputs,
                                    kOutputs, 1, 0, &plan) != 0 ||
      plan == nullptr)
    return false;
  if (fortml_cuda_dense_plan_upload_batch(plan, query_x, target, kBatch) != 0)
    return false;
  std::uint64_t h2d_before = 0;
  std::uint64_t d2h_before = 0;
  std::uint64_t resident = 0;
  if (fortml_cuda_dense_plan_transfer_stats(plan, &h2d_before, &d2h_before,
                                            &resident) != 0)
    return false;
  const std::uint64_t expected_resident = sizeof(double) *
      (kParameters + 3 * kParameters + kInputs * kBatch +
       kOutputs * kBatch + kOutputs);
  if (resident != expected_resident) return false;
  const std::uint64_t expected_h2d = sizeof(double) *
      (kParameters + kInputs * kBatch + kOutputs * kBatch);
  if (h2d_before != expected_h2d) return false;

  for (int step = 1; step <= 4; ++step) {
    const double expected_loss = cpu_step(
        expected_weights, expected_bias, expected_moment1, expected_moment2,
        query_x, target, optimizer_kind, learning_rate, beta1, beta2,
        epsilon, weight_decay, step);
    double actual_loss = -1.0;
    if (fortml_cuda_dense_plan_train_resident_mse(
            plan, learning_rate, beta1, beta2, epsilon, weight_decay,
            optimizer_kind, &actual_loss) != 0 ||
        fabs(actual_loss - expected_loss) > 2.0e-12)
      return false;
    double actual_weights[kWeights] = {};
    double actual_bias[kOutputs] = {};
    if (fortml_cuda_dense_plan_get_parameters(plan, actual_weights,
                                               actual_bias) != 0)
      return false;
    double error = 0.0;
    if (!close(actual_weights, expected_weights, kWeights, 3.0e-12, &error) ||
        !close(actual_bias, expected_bias, kOutputs, 3.0e-12, &error))
      return false;
    *max_error = fmax(*max_error, error);
  }

  std::uint64_t h2d_after = 0;
  std::uint64_t d2h_after = 0;
  if (fortml_cuda_dense_plan_transfer_stats(plan, &h2d_after, &d2h_after,
                                            &resident) != 0 ||
      h2d_after != h2d_before ||
      d2h_after < d2h_before + sizeof(double) * kOutputs)
    return false;
  if (fortml_cuda_dense_plan_destroy(plan) != 0) return false;
  return true;
}

}  // namespace

int main() {
  if (fortml_cuda_dense_available() == 0) {
    std::printf("CUDA resident dense training unavailable; test skipped\n");
    return 0;
  }
  if (cudaSetDevice(0) != cudaSuccess) return 2;
  double max_error = 0.0;
  if (!run_optimizer(kOptimizerSgd, 0.08, 0.9, 0.99, 1.0e-8, 0.01,
                     &max_error))
    return 3;
  if (!run_optimizer(kOptimizerAdam, 0.05, 0.8, 0.9, 1.0e-7, 0.2,
                     &max_error))
    return 4;
  if (!run_optimizer(kOptimizerAdamW, 0.03, 0.9, 0.99, 1.0e-8, 0.1,
                     &max_error))
    return 5;
  std::printf("PASS CUDA resident dense training oracle (SGD/Adam/AdamW, "
              "resident batch+gradients+moments, max error %.3e)\n",
              max_error);
  return 0;
}
