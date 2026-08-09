#include "fortml_cuda_boosted_tree.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <new>

namespace {

constexpr int kThreads = 128;
constexpr int kMaxTrees = 4096;

struct BoostedTreePlan {
  int *tree_offset = nullptr;
  int *node_feature = nullptr;
  int *node_left = nullptr;
  int *node_right = nullptr;
  double *node_threshold = nullptr;
  double *node_weight = nullptr;
  int *node_missing_left = nullptr;
  double *tree_scale = nullptr;
  int n_trees = 0;
  int n_nodes = 0;
  int n_inputs = 0;
  double base_score = 0.0;
  double learning_rate = 0.0;
  int device_index = 0;
  std::uint64_t host_to_device_bytes = 0;
  std::uint64_t device_to_host_bytes = 0;
  std::uint64_t resident_bytes = 0;
};

inline void add_h2d(BoostedTreePlan *plan, std::size_t bytes) {
  plan->host_to_device_bytes += static_cast<std::uint64_t>(bytes);
}

inline void add_d2h(BoostedTreePlan *plan, std::size_t bytes) {
  plan->device_to_host_bytes += static_cast<std::uint64_t>(bytes);
}

__device__ inline int route_query(const BoostedTreePlan &plan,
                                  const double *query_x, int n_query,
                                  int query, int tree, bool jvp,
                                  int *invalid) {
  int node = plan.tree_offset[tree];
  const int begin = node;
  const int end = plan.tree_offset[tree + 1];
  for (int step = 0; step <= plan.n_nodes; ++step) {
    if (node < begin || node >= end) {
      *invalid = 1;
      return -1;
    }
    const int feature = plan.node_feature[node];
    if (feature < 0) return node;
    const double value = query_x[feature * n_query + query];
    if (jvp && (isnan(value) || value == plan.node_threshold[node])) {
      *invalid = 1;
      return -1;
    }
    if (isnan(value)) {
      node = plan.node_missing_left[node] != 0 ? plan.node_left[node]
                                               : plan.node_right[node];
    } else {
      node = value < plan.node_threshold[node] ? plan.node_left[node]
                                               : plan.node_right[node];
    }
  }
  *invalid = 1;
  return -1;
}

__global__ void boosted_tree_predict_kernel(const BoostedTreePlan plan,
                                             const double *query_x,
                                             int n_query, double *margin,
                                             int *invalid) {
  const int query = blockIdx.x * blockDim.x + threadIdx.x;
  if (query >= n_query) return;
  double value = plan.base_score;
  for (int tree = 0; tree < plan.n_trees; ++tree) {
    int bad = 0;
    const int leaf = route_query(plan, query_x, n_query, query, tree, false,
                                 &bad);
    if (bad != 0 || leaf < 0) {
      atomicExch(invalid, 1);
      return;
    }
    value += plan.learning_rate * plan.tree_scale[tree] * plan.node_weight[leaf];
  }
  margin[query] = value;
}

__global__ void boosted_tree_jvp_kernel(const BoostedTreePlan plan,
                                        const double *query_x,
                                        const double *query_x_dot,
                                        int n_query, double *margin,
                                        double *margin_dot, int *invalid) {
  const int query = blockIdx.x * blockDim.x + threadIdx.x;
  if (query >= n_query) return;
  double value = plan.base_score;
  for (int tree = 0; tree < plan.n_trees; ++tree) {
    int bad = 0;
    const int leaf = route_query(plan, query_x, n_query, query, tree, true,
                                 &bad);
    if (bad != 0 || leaf < 0) {
      atomicExch(invalid, 1);
      return;
    }
    // A fixed tree topology is locally constant in the input.  The tangent
    // is read to keep the ABI explicit and to make accidental shape changes
    // visible to the host-side validator; it is zero on the valid branch.
    (void)query_x_dot;
    value += plan.learning_rate * plan.tree_scale[tree] * plan.node_weight[leaf];
  }
  margin[query] = value;
  margin_dot[query] = 0.0;
}

bool finite_array(const double *values, std::size_t count) {
  for (std::size_t i = 0; i < count; ++i)
    if (!std::isfinite(values[i])) return false;
  return true;
}

bool valid_query(const double *values, std::size_t count) {
  for (std::size_t i = 0; i < count; ++i)
    if (!std::isfinite(values[i]) && !std::isnan(values[i])) return false;
  return true;
}

bool valid_model(const int *tree_offset, const int *node_feature,
                 const int *node_left, const int *node_right,
                 const double *node_threshold, const double *node_weight,
                 const int *node_missing_left, const double *tree_scale,
                 int n_trees, int n_nodes, int n_inputs, double base_score,
                 double learning_rate) {
  if (tree_offset == nullptr || node_feature == nullptr || node_left == nullptr ||
      node_right == nullptr || node_threshold == nullptr || node_weight == nullptr ||
      node_missing_left == nullptr || tree_scale == nullptr || n_trees < 1 ||
      n_trees > kMaxTrees || n_nodes < n_trees || n_inputs < 1 ||
      !std::isfinite(base_score) || !std::isfinite(learning_rate) ||
      learning_rate <= 0.0 || tree_offset[0] != 0 ||
      tree_offset[n_trees] != n_nodes || !finite_array(node_threshold, n_nodes) ||
      !finite_array(node_weight, n_nodes) || !finite_array(tree_scale, n_trees))
    return false;
  for (int tree = 0; tree < n_trees; ++tree) {
    if (tree_offset[tree] < 0 || tree_offset[tree] >= tree_offset[tree + 1] ||
        tree_offset[tree + 1] > n_nodes)
      return false;
    const int begin = tree_offset[tree];
    const int end = tree_offset[tree + 1];
    for (int node = begin; node < end; ++node) {
      const int feature = node_feature[node];
      if (feature < -1 || feature >= n_inputs ||
          node_missing_left[node] < 0 || node_missing_left[node] > 1)
        return false;
      if (feature < 0) {
        if (node_left[node] != -1 || node_right[node] != -1) return false;
      } else if (node_left[node] < begin || node_left[node] >= end ||
                 node_right[node] < begin || node_right[node] >= end) {
        return false;
      }
    }
  }
  return true;
}

void destroy_plan(BoostedTreePlan *plan) {
  if (plan == nullptr) return;
  cudaSetDevice(plan->device_index);
  cudaFree(plan->tree_offset);
  cudaFree(plan->node_feature);
  cudaFree(plan->node_left);
  cudaFree(plan->node_right);
  cudaFree(plan->node_threshold);
  cudaFree(plan->node_weight);
  cudaFree(plan->node_missing_left);
  cudaFree(plan->tree_scale);
  delete plan;
}

template <typename T>
cudaError_t copy_to_device(T **destination, const T *source, std::size_t count) {
  cudaError_t error = cudaMalloc(destination, count * sizeof(T));
  if (error == cudaSuccess)
    error = cudaMemcpy(*destination, source, count * sizeof(T),
                       cudaMemcpyHostToDevice);
  return error;
}

int predict_common(BoostedTreePlan *plan, const double *query_x,
                   const double *query_x_dot, int n_query, double *margin,
                   double *margin_dot) {
  if (plan == nullptr || query_x == nullptr || margin == nullptr || n_query < 1 ||
      !valid_query(query_x, static_cast<std::size_t>(n_query) * plan->n_inputs) ||
      (query_x_dot != nullptr &&
       !finite_array(query_x_dot,
                    static_cast<std::size_t>(n_query) * plan->n_inputs)))
    return static_cast<int>(cudaErrorInvalidValue);
  if (query_x_dot != nullptr) {
    for (std::size_t i = 0; i < static_cast<std::size_t>(n_query) * plan->n_inputs;
         ++i)
      if (std::isnan(query_x[i])) return static_cast<int>(cudaErrorInvalidValue);
  }
  cudaSetDevice(plan->device_index);
  double *d_query = nullptr, *d_query_dot = nullptr, *d_margin = nullptr,
         *d_margin_dot = nullptr;
  int *d_invalid = nullptr;
  const std::size_t query_count = static_cast<std::size_t>(n_query) * plan->n_inputs;
  cudaError_t error = copy_to_device(&d_query, query_x, query_count);
  if (error == cudaSuccess) add_h2d(plan, sizeof(double) * query_count);
  if (error == cudaSuccess && query_x_dot != nullptr)
    error = copy_to_device(&d_query_dot, query_x_dot, query_count);
  if (error == cudaSuccess && query_x_dot != nullptr)
    add_h2d(plan, sizeof(double) * query_count);
  if (error == cudaSuccess) error = cudaMalloc(&d_margin, sizeof(double) * n_query);
  if (error == cudaSuccess && margin_dot != nullptr)
    error = cudaMalloc(&d_margin_dot, sizeof(double) * n_query);
  if (error == cudaSuccess) error = cudaMalloc(&d_invalid, sizeof(int));
  if (error == cudaSuccess) error = cudaMemset(d_invalid, 0, sizeof(int));
  if (error == cudaSuccess && margin_dot == nullptr) {
    boosted_tree_predict_kernel<<<(n_query + kThreads - 1) / kThreads, kThreads>>>(
        *plan, d_query, n_query, d_margin, d_invalid);
  } else if (error == cudaSuccess) {
    boosted_tree_jvp_kernel<<<(n_query + kThreads - 1) / kThreads, kThreads>>>(
        *plan, d_query, d_query_dot, n_query, d_margin, d_margin_dot, d_invalid);
  }
  if (error == cudaSuccess) error = cudaGetLastError();
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  int invalid = 1;
  if (error == cudaSuccess)
    error = cudaMemcpy(&invalid, d_invalid, sizeof(int), cudaMemcpyDeviceToHost);
  if (error == cudaSuccess && invalid == 0) {
    error = cudaMemcpy(margin, d_margin, sizeof(double) * n_query,
                       cudaMemcpyDeviceToHost);
    if (error == cudaSuccess) add_d2h(plan, sizeof(double) * n_query);
    if (error == cudaSuccess && margin_dot != nullptr)
      error = cudaMemcpy(margin_dot, d_margin_dot, sizeof(double) * n_query,
                         cudaMemcpyDeviceToHost);
    if (error == cudaSuccess && margin_dot != nullptr)
      add_d2h(plan, sizeof(double) * n_query);
  } else if (error == cudaSuccess) {
    error = cudaErrorInvalidValue;
  }
  cudaFree(d_query);
  cudaFree(d_query_dot);
  cudaFree(d_margin);
  cudaFree(d_margin_dot);
  cudaFree(d_invalid);
  return static_cast<int>(error);
}

}  // namespace

extern "C" int fortml_cuda_boosted_tree_available() {
  int count = 0;
  const cudaError_t error = cudaGetDeviceCount(&count);
  if (error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver)
    return 0;
  return error == cudaSuccess && count > 0 ? 1 : 0;
}

extern "C" int fortml_cuda_boosted_tree_plan_create(
    const int *tree_offset, const int *node_feature, const int *node_left,
    const int *node_right, const double *node_threshold,
    const double *node_weight, const int *node_missing_left,
    const double *tree_scale, int n_trees, int n_nodes, int n_inputs,
    double base_score, double learning_rate, int device_index,
    void **opaque_plan) {
  if (opaque_plan == nullptr) return static_cast<int>(cudaErrorInvalidValue);
  *opaque_plan = nullptr;
  if (device_index < 0 ||
      !valid_model(tree_offset, node_feature, node_left, node_right,
                   node_threshold, node_weight, node_missing_left, tree_scale,
                   n_trees, n_nodes, n_inputs, base_score, learning_rate))
    return static_cast<int>(cudaErrorInvalidValue);
  cudaError_t error = cudaSetDevice(device_index);
  if (error != cudaSuccess) return static_cast<int>(error);
  BoostedTreePlan *plan = new (std::nothrow) BoostedTreePlan();
  if (plan == nullptr) return static_cast<int>(cudaErrorMemoryAllocation);
  plan->n_trees = n_trees;
  plan->n_nodes = n_nodes;
  plan->n_inputs = n_inputs;
  plan->base_score = base_score;
  plan->learning_rate = learning_rate;
  plan->device_index = device_index;
  error = copy_to_device(&plan->tree_offset, tree_offset, n_trees + 1);
  if (error == cudaSuccess)
    add_h2d(plan, sizeof(int) * static_cast<std::size_t>(n_trees + 1));
  if (error == cudaSuccess) error = copy_to_device(&plan->node_feature, node_feature, n_nodes);
  if (error == cudaSuccess) add_h2d(plan, sizeof(int) * static_cast<std::size_t>(n_nodes));
  if (error == cudaSuccess) error = copy_to_device(&plan->node_left, node_left, n_nodes);
  if (error == cudaSuccess) add_h2d(plan, sizeof(int) * static_cast<std::size_t>(n_nodes));
  if (error == cudaSuccess) error = copy_to_device(&plan->node_right, node_right, n_nodes);
  if (error == cudaSuccess) add_h2d(plan, sizeof(int) * static_cast<std::size_t>(n_nodes));
  if (error == cudaSuccess) error = copy_to_device(&plan->node_threshold, node_threshold, n_nodes);
  if (error == cudaSuccess) add_h2d(plan, sizeof(double) * static_cast<std::size_t>(n_nodes));
  if (error == cudaSuccess) error = copy_to_device(&plan->node_weight, node_weight, n_nodes);
  if (error == cudaSuccess) add_h2d(plan, sizeof(double) * static_cast<std::size_t>(n_nodes));
  if (error == cudaSuccess) error = copy_to_device(&plan->node_missing_left, node_missing_left, n_nodes);
  if (error == cudaSuccess) add_h2d(plan, sizeof(int) * static_cast<std::size_t>(n_nodes));
  if (error == cudaSuccess) error = copy_to_device(&plan->tree_scale, tree_scale, n_trees);
  if (error == cudaSuccess) add_h2d(plan, sizeof(double) * static_cast<std::size_t>(n_trees));
  if (error != cudaSuccess) {
    destroy_plan(plan);
    return static_cast<int>(error);
  }
  plan->resident_bytes = sizeof(int) * static_cast<std::size_t>(n_trees + 1) +
      sizeof(int) * static_cast<std::size_t>(4 * n_nodes) +
      sizeof(double) * static_cast<std::size_t>(2 * n_nodes + n_trees);
  *opaque_plan = plan;
  return 0;
}

extern "C" int fortml_cuda_boosted_tree_plan_predict(
    void *opaque_plan, const double *query_x, int n_query, double *margin) {
  return predict_common(static_cast<BoostedTreePlan *>(opaque_plan), query_x,
                        nullptr, n_query, margin, nullptr);
}

extern "C" int fortml_cuda_boosted_tree_plan_predict_jvp(
    void *opaque_plan, const double *query_x, const double *query_x_dot,
    int n_query, double *margin, double *margin_dot) {
  if (margin_dot == nullptr)
    return static_cast<int>(cudaErrorInvalidValue);
  return predict_common(static_cast<BoostedTreePlan *>(opaque_plan), query_x,
                        query_x_dot, n_query, margin, margin_dot);
}

extern "C" int fortml_cuda_boosted_tree_plan_transfer_stats(
    void *opaque_plan, std::uint64_t *host_to_device_bytes,
    std::uint64_t *device_to_host_bytes, std::uint64_t *resident_bytes) {
  BoostedTreePlan *plan = static_cast<BoostedTreePlan *>(opaque_plan);
  if (plan == nullptr || host_to_device_bytes == nullptr ||
      device_to_host_bytes == nullptr || resident_bytes == nullptr)
    return static_cast<int>(cudaErrorInvalidValue);
  *host_to_device_bytes = plan->host_to_device_bytes;
  *device_to_host_bytes = plan->device_to_host_bytes;
  *resident_bytes = plan->resident_bytes;
  return 0;
}

extern "C" int fortml_cuda_boosted_tree_plan_destroy(void *opaque_plan) {
  BoostedTreePlan *plan = static_cast<BoostedTreePlan *>(opaque_plan);
  if (plan == nullptr) return 0;
  cudaSetDevice(plan->device_index);
  cudaError_t error = cudaSuccess;
  error = cudaFree(plan->tree_offset);
  if (error == cudaSuccess) error = cudaFree(plan->node_feature);
  if (error == cudaSuccess) error = cudaFree(plan->node_left);
  if (error == cudaSuccess) error = cudaFree(plan->node_right);
  if (error == cudaSuccess) error = cudaFree(plan->node_threshold);
  if (error == cudaSuccess) error = cudaFree(plan->node_weight);
  if (error == cudaSuccess) error = cudaFree(plan->node_missing_left);
  if (error == cudaSuccess) error = cudaFree(plan->tree_scale);
  delete plan;
  return static_cast<int>(error);
}
