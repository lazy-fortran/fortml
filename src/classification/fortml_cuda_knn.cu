#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>

namespace {

// The kernel keeps the selected neighbors in a per-query global scratch row.
// This avoids a device-side sort allocation and gives deterministic ties by
// retaining the original training-row index as the secondary key.
constexpr int kThreads = 128;
constexpr int kMaxNeighbors = 1024;

__device__ inline bool better(double distance, int index, double other_distance,
                              int other_index) {
  return distance < other_distance ||
         (distance == other_distance && index < other_index);
}

__global__ void knn_predict_kernel(
    const double* __restrict__ train_x, const int* __restrict__ train_class,
    const double* __restrict__ sample_weight,
    const double* __restrict__ query_x, const int* __restrict__ class_label,
    double* __restrict__ top_distance, int* __restrict__ top_index,
    double* __restrict__ best_score, int* __restrict__ output, int n_train,
    int n_features, int n_query,
    int n_classes, int n_neighbors, int weighting_code) {
  const int query = blockIdx.x * blockDim.x + threadIdx.x;
  if (query >= n_query) return;

  double* query_distance = top_distance + query * n_neighbors;
  int* query_index = top_index + query * n_neighbors;
  for (int slot = 0; slot < n_neighbors; ++slot) {
    query_distance[slot] = INFINITY;
    query_index[slot] = n_train;
  }

  // The arrays use Fortran column-major storage: feature-major rows.
  for (int train = 0; train < n_train; ++train) {
    double distance = 0.0;
    for (int feature = 0; feature < n_features; ++feature) {
      const double difference = query_x[feature * n_query + query] -
                                train_x[feature * n_train + train];
      distance += difference * difference;
    }
    int insertion = n_neighbors;
    if (better(distance, train, query_distance[n_neighbors - 1],
               query_index[n_neighbors - 1])) {
      insertion = n_neighbors - 1;
      while (insertion > 0 &&
             better(distance, train, query_distance[insertion - 1],
                    query_index[insertion - 1])) {
        query_distance[insertion] = query_distance[insertion - 1];
        query_index[insertion] = query_index[insertion - 1];
        --insertion;
      }
      query_distance[insertion] = distance;
      query_index[insertion] = train;
    }
  }

  bool exact = false;
  if (weighting_code == 2) {
    for (int slot = 0; slot < n_neighbors; ++slot) {
      if (query_distance[slot] == 0.0) exact = true;
    }
  }

  // n_classes is intentionally not capped. The score row is a separate
  // global allocation, so large label sets do not consume per-thread stack.
  for (int class_index = 0; class_index < n_classes; ++class_index) {
    double score = 0.0;
    for (int slot = 0; slot < n_neighbors; ++slot) {
      const int train = query_index[slot];
      if (train >= n_train) continue;
      double weight = sample_weight[train];
      if (weighting_code == 2) {
        if (exact) {
          weight = query_distance[slot] == 0.0 ? weight : 0.0;
        } else {
          weight /= sqrt(query_distance[slot]);
        }
      }
      // FortML stores class indices one-based, while this loop is zero-based.
      if (train_class[train] == class_index + 1) score += weight;
    }
    // The output is initially the first class. A strict comparison preserves
    // the smallest sorted class label on an exact score tie.
    if (class_index == 0 || score > best_score[query]) {
      best_score[query] = score;
      output[query] = class_index;
    }
  }
  output[query] = class_label[output[query]];
}

bool valid_inputs(const double* train_x, const int* train_class,
                  const double* sample_weight, const double* query_x,
                  const int* class_label, int* output, int n_train,
                  int n_features, int n_query, int n_classes,
                  int n_neighbors, int weighting_code) {
  return train_x != nullptr && train_class != nullptr &&
         sample_weight != nullptr && query_x != nullptr &&
         class_label != nullptr && output != nullptr && n_train > 0 &&
         n_features > 0 && n_query > 0 && n_classes > 0 &&
         n_neighbors > 0 && n_neighbors <= n_train &&
         n_neighbors <= kMaxNeighbors &&
         (weighting_code == 1 || weighting_code == 2);
}

}  // namespace

extern "C" int fortml_cuda_knn_available() { return 1; }

struct FortmlCudaKnnPlan {
  double* train_x = nullptr;
  int* train_class = nullptr;
  double* sample_weight = nullptr;
  int* class_label = nullptr;
  int n_train = 0;
  int n_features = 0;
  int n_classes = 0;
  int n_neighbors = 0;
  int weighting_code = 0;
  int device_index = 0;
};

void destroy_plan(FortmlCudaKnnPlan* plan) {
  if (plan == nullptr) return;
  cudaFree(plan->train_x);
  cudaFree(plan->train_class);
  cudaFree(plan->sample_weight);
  cudaFree(plan->class_label);
  delete plan;
}

extern "C" int fortml_cuda_knn_plan_create(
    const double* train_x, const int* train_class,
    const double* sample_weight, const int* class_label, int n_train,
    int n_features, int n_classes, int n_neighbors, int weighting_code,
    int device_index, void** opaque_plan) {
  if (opaque_plan == nullptr || train_x == nullptr || train_class == nullptr ||
      sample_weight == nullptr || class_label == nullptr || n_train < 1 ||
      n_features < 1 || n_classes < 1 || n_neighbors < 1 ||
      n_neighbors > n_train || n_neighbors > kMaxNeighbors ||
      (weighting_code != 1 && weighting_code != 2) || device_index < 0) {
    if (opaque_plan != nullptr) *opaque_plan = nullptr;
    return static_cast<int>(cudaErrorInvalidValue);
  }
  cudaError_t error = cudaSetDevice(device_index);
  if (error != cudaSuccess) {
    *opaque_plan = nullptr;
    return static_cast<int>(error);
  }
  FortmlCudaKnnPlan* plan = new FortmlCudaKnnPlan;
  plan->n_train = n_train;
  plan->n_features = n_features;
  plan->n_classes = n_classes;
  plan->n_neighbors = n_neighbors;
  plan->weighting_code = weighting_code;
  plan->device_index = device_index;
  error = cudaMalloc(&plan->train_x,
                     sizeof(double) * static_cast<std::size_t>(n_train) *
                         static_cast<std::size_t>(n_features));
  if (error == cudaSuccess)
    error = cudaMalloc(&plan->train_class, sizeof(int) * n_train);
  if (error == cudaSuccess)
    error = cudaMalloc(&plan->sample_weight, sizeof(double) * n_train);
  if (error == cudaSuccess)
    error = cudaMalloc(&plan->class_label, sizeof(int) * n_classes);
  if (error == cudaSuccess)
    error = cudaMemcpy(plan->train_x, train_x,
                       sizeof(double) * static_cast<std::size_t>(n_train) *
                           static_cast<std::size_t>(n_features),
                       cudaMemcpyHostToDevice);
  if (error == cudaSuccess)
    error = cudaMemcpy(plan->train_class, train_class, sizeof(int) * n_train,
                       cudaMemcpyHostToDevice);
  if (error == cudaSuccess)
    error = cudaMemcpy(plan->sample_weight, sample_weight,
                       sizeof(double) * n_train, cudaMemcpyHostToDevice);
  if (error == cudaSuccess)
    error = cudaMemcpy(plan->class_label, class_label, sizeof(int) * n_classes,
                       cudaMemcpyHostToDevice);
  if (error != cudaSuccess) {
    destroy_plan(plan);
    *opaque_plan = nullptr;
    return static_cast<int>(error);
  }
  *opaque_plan = plan;
  return 0;
}

extern "C" int fortml_cuda_knn_plan_destroy(void* opaque_plan) {
  destroy_plan(static_cast<FortmlCudaKnnPlan*>(opaque_plan));
  return 0;
}

extern "C" int fortml_cuda_knn_plan_predict(void* opaque_plan,
                                             const double* query_x,
                                             int n_query, int* output) {
  FortmlCudaKnnPlan* plan = static_cast<FortmlCudaKnnPlan*>(opaque_plan);
  if (plan == nullptr || query_x == nullptr || output == nullptr ||
      n_query < 1) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  cudaSetDevice(plan->device_index);
  double* d_query_x = nullptr;
  double* d_top_distance = nullptr;
  int* d_top_index = nullptr;
  double* d_best_score = nullptr;
  int* d_output = nullptr;
  cudaError_t error = cudaMalloc(
      &d_query_x, sizeof(double) * static_cast<std::size_t>(n_query) *
                      static_cast<std::size_t>(plan->n_features));
  if (error == cudaSuccess)
    error = cudaMalloc(&d_top_distance,
                       sizeof(double) * static_cast<std::size_t>(n_query) *
                           static_cast<std::size_t>(plan->n_neighbors));
  if (error == cudaSuccess)
    error = cudaMalloc(&d_top_index,
                       sizeof(int) * static_cast<std::size_t>(n_query) *
                           static_cast<std::size_t>(plan->n_neighbors));
  if (error == cudaSuccess)
    error = cudaMalloc(&d_best_score, sizeof(double) * n_query);
  if (error == cudaSuccess)
    error = cudaMalloc(&d_output, sizeof(int) * n_query);
  if (error == cudaSuccess)
    error = cudaMemcpy(d_query_x, query_x,
                       sizeof(double) * static_cast<std::size_t>(n_query) *
                           static_cast<std::size_t>(plan->n_features),
                       cudaMemcpyHostToDevice);
  if (error == cudaSuccess) {
    const int grid = (n_query + kThreads - 1) / kThreads;
    knn_predict_kernel<<<grid, kThreads>>>(
        plan->train_x, plan->train_class, plan->sample_weight, d_query_x,
        plan->class_label, d_top_distance, d_top_index, d_best_score,
        d_output, plan->n_train, plan->n_features, n_query, plan->n_classes,
        plan->n_neighbors, plan->weighting_code);
    error = cudaGetLastError();
  }
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  if (error == cudaSuccess)
    error = cudaMemcpy(output, d_output, sizeof(int) * n_query,
                       cudaMemcpyDeviceToHost);
  cudaFree(d_query_x);
  cudaFree(d_top_distance);
  cudaFree(d_top_index);
  cudaFree(d_best_score);
  cudaFree(d_output);
  return static_cast<int>(error);
}

extern "C" int fortml_cuda_knn_predict(
    const double* train_x, const int* train_class,
    const double* sample_weight, const double* query_x, const int* class_label,
    int* output, int n_train, int n_features, int n_query, int n_classes,
    int n_neighbors, int weighting_code) {
  if (!valid_inputs(train_x, train_class, sample_weight, query_x, class_label,
                    output, n_train, n_features, n_query, n_classes,
                    n_neighbors, weighting_code)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  void* plan = nullptr;
  int status = fortml_cuda_knn_plan_create(
      train_x, train_class, sample_weight, class_label, n_train, n_features,
      n_classes, n_neighbors, weighting_code, 0, &plan);
  if (status != 0) return status;
  status = fortml_cuda_knn_plan_predict(plan, query_x, n_query, output);
  fortml_cuda_knn_plan_destroy(plan);
  return status;
}
