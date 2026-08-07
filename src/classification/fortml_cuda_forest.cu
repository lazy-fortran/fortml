#include "fortml_cuda_forest.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <new>

namespace {

constexpr int kThreads = 128;
constexpr int kMaxTrees = 4096;

struct ForestPlan {
  int *tree_offset = nullptr;
  int *node_feature = nullptr;
  int *node_left = nullptr;
  int *node_right = nullptr;
  double *node_threshold = nullptr;
  double *node_probability = nullptr;
  int *class_label = nullptr;
  int n_trees = 0;
  int n_nodes = 0;
  int n_inputs = 0;
  int n_classes = 0;
  int device_index = 0;
};

__device__ inline int route_query_at(const ForestPlan &plan,
                                     const double *query_x, int n_query,
                                     int query, int tree) {
  int node = plan.tree_offset[tree];
  const int end = plan.tree_offset[tree + 1];
  // The host-side validator proves termination and bounds.  The guard keeps
  // malformed device memory from spinning forever if a plan is corrupted.
  for (int step = 0; step <= plan.n_nodes; ++step) {
    if (node < plan.tree_offset[tree] || node >= end) return -1;
    const int feature = plan.node_feature[node];
    if (feature < 0) return node;
    const double value = query_x[feature * n_query + query];
    node = value < plan.node_threshold[node] ? plan.node_left[node]
                                               : plan.node_right[node];
  }
  return -1;
}

__global__ void forest_probability_kernel(
    const ForestPlan plan, const double *query_x, int n_query,
    double *probabilities) {
  const int query = blockIdx.x * blockDim.x + threadIdx.x;
  if (query >= n_query) return;
  for (int class_index = 0; class_index < plan.n_classes; ++class_index)
    probabilities[class_index * n_query + query] = 0.0;
  for (int tree = 0; tree < plan.n_trees; ++tree) {
    const int leaf = route_query_at(plan, query_x, n_query, query, tree);
    if (leaf < 0) return;
    for (int class_index = 0; class_index < plan.n_classes; ++class_index) {
      probabilities[class_index * n_query + query] +=
          plan.node_probability[leaf * plan.n_classes + class_index];
    }
  }
  const double scale = 1.0 / static_cast<double>(plan.n_trees);
  for (int class_index = 0; class_index < plan.n_classes; ++class_index)
    probabilities[class_index * n_query + query] *= scale;
}

__global__ void forest_label_kernel(const ForestPlan plan,
                                     const double *query_x, int n_query,
                                     int *labels) {
  const int query = blockIdx.x * blockDim.x + threadIdx.x;
  if (query >= n_query) return;
  double best_probability = -1.0;
  int best_class = 0;
  for (int class_index = 0; class_index < plan.n_classes; ++class_index) {
    double probability = 0.0;
    for (int tree = 0; tree < plan.n_trees; ++tree) {
      const int leaf = route_query_at(plan, query_x, n_query, query, tree);
      if (leaf < 0) return;
      probability +=
          plan.node_probability[leaf * plan.n_classes + class_index];
    }
    probability /= static_cast<double>(plan.n_trees);
    // Strict comparison retains the smallest sorted class index on ties,
    // matching the CPU maxloc convention.
    if (class_index == 0 || probability > best_probability) {
      best_probability = probability;
      best_class = class_index;
    }
  }
  labels[query] = plan.class_label[best_class];
}

bool finite_array(const double *values, std::size_t count) {
  for (std::size_t i = 0; i < count; ++i)
    if (!std::isfinite(values[i])) return false;
  return true;
}

bool valid_model(const int *tree_offset, const int *node_feature,
                 const int *node_left, const int *node_right,
                 const double *node_threshold, const double *node_probability,
                 const int *class_label, int n_trees, int n_nodes,
                 int n_inputs, int n_classes) {
  if (tree_offset == nullptr || node_feature == nullptr || node_left == nullptr ||
      node_right == nullptr || node_threshold == nullptr ||
      node_probability == nullptr || class_label == nullptr || n_trees < 1 ||
      n_trees > kMaxTrees || n_nodes < n_trees || n_inputs < 1 ||
      n_classes < 1 || tree_offset[0] != 0 || tree_offset[n_trees] != n_nodes)
    return false;
  for (int tree = 0; tree < n_trees; ++tree) {
    if (tree_offset[tree] < 0 || tree_offset[tree] >= tree_offset[tree + 1] ||
        tree_offset[tree + 1] > n_nodes)
      return false;
  }
  if (!finite_array(node_threshold, static_cast<std::size_t>(n_nodes)) ||
      !finite_array(node_probability,
                    static_cast<std::size_t>(n_nodes) * n_classes))
    return false;
  for (int tree = 0; tree < n_trees; ++tree) {
    const int begin = tree_offset[tree];
    const int end = tree_offset[tree + 1];
    for (int node = begin; node < end; ++node) {
      const int feature = node_feature[node];
      if (feature < -1 || feature >= n_inputs) return false;
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

void destroy_plan(ForestPlan *plan) {
  if (plan == nullptr) return;
  cudaSetDevice(plan->device_index);
  cudaFree(plan->tree_offset);
  cudaFree(plan->node_feature);
  cudaFree(plan->node_left);
  cudaFree(plan->node_right);
  cudaFree(plan->node_threshold);
  cudaFree(plan->node_probability);
  cudaFree(plan->class_label);
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

}  // namespace

extern "C" int fortml_cuda_forest_available() {
  int count = 0;
  const cudaError_t error = cudaGetDeviceCount(&count);
  if (error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver)
    return 0;
  return error == cudaSuccess && count > 0 ? 1 : 0;
}

extern "C" int fortml_cuda_forest_plan_create(
    const int *tree_offset, const int *node_feature, const int *node_left,
    const int *node_right, const double *node_threshold,
    const double *node_probability, const int *class_label, int n_trees,
    int n_nodes, int n_inputs, int n_classes, int device_index,
    void **opaque_plan) {
  if (opaque_plan == nullptr) return static_cast<int>(cudaErrorInvalidValue);
  *opaque_plan = nullptr;
  if (device_index < 0 ||
      !valid_model(tree_offset, node_feature, node_left, node_right,
                   node_threshold, node_probability, class_label, n_trees,
                   n_nodes, n_inputs, n_classes))
    return static_cast<int>(cudaErrorInvalidValue);
  cudaError_t error = cudaSetDevice(device_index);
  if (error != cudaSuccess) return static_cast<int>(error);
  ForestPlan *plan = new (std::nothrow) ForestPlan();
  if (plan == nullptr) return static_cast<int>(cudaErrorMemoryAllocation);
  plan->n_trees = n_trees;
  plan->n_nodes = n_nodes;
  plan->n_inputs = n_inputs;
  plan->n_classes = n_classes;
  plan->device_index = device_index;
  error = copy_to_device(&plan->tree_offset, tree_offset, n_trees + 1);
  if (error == cudaSuccess)
    error = copy_to_device(&plan->node_feature, node_feature, n_nodes);
  if (error == cudaSuccess)
    error = copy_to_device(&plan->node_left, node_left, n_nodes);
  if (error == cudaSuccess)
    error = copy_to_device(&plan->node_right, node_right, n_nodes);
  if (error == cudaSuccess)
    error = copy_to_device(&plan->node_threshold, node_threshold, n_nodes);
  if (error == cudaSuccess)
    error = copy_to_device(&plan->node_probability, node_probability,
                           static_cast<std::size_t>(n_nodes) * n_classes);
  if (error == cudaSuccess)
    error = copy_to_device(&plan->class_label, class_label, n_classes);
  if (error != cudaSuccess) {
    destroy_plan(plan);
    return static_cast<int>(error);
  }
  *opaque_plan = plan;
  return 0;
}

extern "C" int fortml_cuda_forest_plan_predict_proba(
    void *opaque_plan, const double *query_x, int n_query,
    double *probabilities) {
  ForestPlan *plan = static_cast<ForestPlan *>(opaque_plan);
  if (plan == nullptr || query_x == nullptr || probabilities == nullptr ||
      n_query < 1)
    return static_cast<int>(cudaErrorInvalidValue);
  if (!finite_array(query_x, static_cast<std::size_t>(n_query) * plan->n_inputs))
    return static_cast<int>(cudaErrorInvalidValue);
  cudaSetDevice(plan->device_index);
  double *d_query = nullptr, *d_probability = nullptr;
  const std::size_t query_count = static_cast<std::size_t>(n_query) * plan->n_inputs;
  const std::size_t output_count = static_cast<std::size_t>(n_query) * plan->n_classes;
  cudaError_t error = copy_to_device(&d_query, query_x, query_count);
  if (error == cudaSuccess) error = cudaMalloc(&d_probability, output_count * sizeof(double));
  if (error == cudaSuccess) {
    forest_probability_kernel<<<(n_query + kThreads - 1) / kThreads, kThreads>>>(
        *plan, d_query, n_query, d_probability);
    error = cudaGetLastError();
  }
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  if (error == cudaSuccess)
    error = cudaMemcpy(probabilities, d_probability, output_count * sizeof(double),
                       cudaMemcpyDeviceToHost);
  cudaFree(d_query);
  cudaFree(d_probability);
  return static_cast<int>(error);
}

extern "C" int fortml_cuda_forest_plan_predict(void *opaque_plan,
                                                 const double *query_x,
                                                 int n_query, int *labels) {
  ForestPlan *plan = static_cast<ForestPlan *>(opaque_plan);
  if (plan == nullptr || query_x == nullptr || labels == nullptr || n_query < 1)
    return static_cast<int>(cudaErrorInvalidValue);
  if (!finite_array(query_x, static_cast<std::size_t>(n_query) * plan->n_inputs))
    return static_cast<int>(cudaErrorInvalidValue);
  cudaSetDevice(plan->device_index);
  double *d_query = nullptr;
  int *d_labels = nullptr;
  const std::size_t query_count = static_cast<std::size_t>(n_query) * plan->n_inputs;
  cudaError_t error = copy_to_device(&d_query, query_x, query_count);
  if (error == cudaSuccess) error = cudaMalloc(&d_labels, sizeof(int) * n_query);
  if (error == cudaSuccess) {
    forest_label_kernel<<<(n_query + kThreads - 1) / kThreads, kThreads>>>(
        *plan, d_query, n_query, d_labels);
    error = cudaGetLastError();
  }
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  if (error == cudaSuccess)
    error = cudaMemcpy(labels, d_labels, sizeof(int) * n_query,
                       cudaMemcpyDeviceToHost);
  cudaFree(d_query);
  cudaFree(d_labels);
  return static_cast<int>(error);
}

extern "C" int fortml_cuda_forest_plan_destroy(void *opaque_plan) {
  ForestPlan *plan = static_cast<ForestPlan *>(opaque_plan);
  if (plan == nullptr) return 0;
  cudaSetDevice(plan->device_index);
  cudaError_t error = cudaFree(plan->tree_offset);
  if (error == cudaSuccess) error = cudaFree(plan->node_feature);
  if (error == cudaSuccess) error = cudaFree(plan->node_left);
  if (error == cudaSuccess) error = cudaFree(plan->node_right);
  if (error == cudaSuccess) error = cudaFree(plan->node_threshold);
  if (error == cudaSuccess) error = cudaFree(plan->node_probability);
  if (error == cudaSuccess) error = cudaFree(plan->class_label);
  delete plan;
  return static_cast<int>(error);
}
