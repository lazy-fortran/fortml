#include "../src/classification/fortml_cuda_boosted_tree.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <limits>

namespace {

double cpu_route(const int *offsets, const int *features, const int *left,
                 const int *right, const int *missing_left,
                 const double *threshold, const double *weights,
                 const double *scales, int n_trees, int query, int n_query,
                 const double *x, double base, double rate) {
  double value = base;
  for (int tree = 0; tree < n_trees; ++tree) {
    int node = offsets[tree];
    while (features[node] >= 0) {
      const int feature = features[node];
      const double query_value = x[feature * n_query + query];
      if (std::isnan(query_value))
        node = missing_left[node] != 0 ? left[node] : right[node];
      else
        node = query_value < threshold[node] ? left[node] : right[node];
    }
    value += rate * scales[tree] * weights[node];
  }
  return value;
}

}  // namespace

int main() {
  if (fortml_cuda_boosted_tree_available() == 0) {
    std::printf("CUDA boosted tree unavailable; test skipped\n");
    return 0;
  }
  if (cudaSetDevice(0) != cudaSuccess) return 2;

  const int offsets[] = {0, 3, 6};
  const int features[] = {0, -1, -1, 1, -1, -1};
  const int left[] = {1, -1, -1, 4, -1, -1};
  const int right[] = {2, -1, -1, 5, -1, -1};
  const double threshold[] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0};
  const double weights[] = {0.0, -1.0, 2.0, 0.0, 0.5, -0.5};
  const int missing_left[] = {1, 0, 0, 0, 0, 0};
  const double scales[] = {1.0, 0.5};
  constexpr int n_query = 5;
  constexpr int n_inputs = 2;
  const double nan = std::numeric_limits<double>::quiet_NaN();
  // Column-major query matrix; the fifth row exercises the learned NaN route.
  const double query[] = {-1.0, 0.0, 0.5, 2.0, nan,
                           0.0, 1.0, 1.5, 2.0, 2.0};
  const double query_dot[] = {0.25, -0.5, 0.7, 0.3, 0.1,
                              -0.2, 0.4, -0.3, 0.9, 0.6};
  double expected[n_query] = {};
  for (int query_index = 0; query_index < n_query; ++query_index)
    expected[query_index] = cpu_route(offsets, features, left, right,
                                      missing_left, threshold, weights, scales,
                                      2, query_index, n_query, query, 0.2, 0.7);

  void *plan = nullptr;
  int status = fortml_cuda_boosted_tree_plan_create(
      offsets, features, left, right, threshold, weights, missing_left, scales,
      2, 6, n_inputs, 0.2, 0.7, 0, &plan);
  if (status != 0 || plan == nullptr) return 3;
  double margin[n_query];
  status = fortml_cuda_boosted_tree_plan_predict(plan, query, n_query, margin);
  if (status != 0) return 4;
  double max_error = 0.0;
  for (int i = 0; i < n_query; ++i)
    max_error = fmax(max_error, fabs(margin[i] - expected[i]));
  if (max_error > 1.0e-13) return 5;

  double margin_dot[n_query];
  status = fortml_cuda_boosted_tree_plan_predict_jvp(
      plan, query, query_dot, n_query, margin, margin_dot);
  // NaN routing is discrete and therefore rejected for input products.
  if (status == 0) return 6;
  const double boundary_query[] = {0.0, 2.0, 0.5, 2.0, 2.0,
                                   0.0, 1.0, 1.5, 2.0, 2.0};
  status = fortml_cuda_boosted_tree_plan_predict_jvp(
      plan, boundary_query, query_dot, n_query, margin, margin_dot);
  // The first query is exactly on tree 0's split, so the JVP is undefined.
  if (status == 0) return 7;

  const double smooth_query[] = {-1.0, -0.5, 0.5, 2.0, 3.0,
                                 0.0, 0.5, 1.5, 2.0, 2.0};
  status = fortml_cuda_boosted_tree_plan_predict_jvp(
      plan, smooth_query, query_dot, n_query, margin, margin_dot);
  if (status != 0) return 8;
  for (int i = 0; i < n_query; ++i) {
    const double expected_value = cpu_route(
        offsets, features, left, right, missing_left, threshold, weights,
        scales, 2, i, n_query, smooth_query, 0.2, 0.7);
    if (fabs(margin[i] - expected_value) > 1.0e-13 ||
        fabs(margin_dot[i]) > 1.0e-13)
      return 9;
  }
  std::uint64_t host_to_device = 0;
  std::uint64_t device_to_host = 0;
  std::uint64_t resident = 0;
  if (fortml_cuda_boosted_tree_plan_transfer_stats(
          plan, &host_to_device, &device_to_host, &resident) != 0 ||
      resident != 220 || host_to_device < 220 + 80 || device_to_host < 40)
    return 10;
  if (fortml_cuda_boosted_tree_plan_destroy(plan) != 0) return 11;
  std::printf("PASS CUDA boosted tree resident value/JVP oracle max_error %.3e\n",
              max_error);
  return 0;
}
