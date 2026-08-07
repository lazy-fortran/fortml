#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>

extern "C" int fortml_cuda_knn_available();
extern "C" int fortml_cuda_knn_plan_create(
    const double*, const int*, const double*, const int*, int, int, int, int,
    int, int, void**);
extern "C" int fortml_cuda_knn_plan_destroy(void*);
extern "C" int fortml_cuda_knn_plan_predict(void*, const double*, int, int*);

int main() {
  if (fortml_cuda_knn_available() == 0) {
    std::printf("CUDA kNN unavailable; test skipped\n");
    return 0;
  }
  if (cudaSetDevice(0) != cudaSuccess) return 2;
  // Fortran column-major storage: each feature is a contiguous column.
  const double train_x[] = {-2.0, -1.0, 1.0, 2.0};
  const int train_class[] = {1, 1, 2, 2};
  const double sample_weight[] = {1.0, 1.0, 1.0, 1.0};
  const int class_label[] = {-7, 11};
  const double query_x[] = {-1.5, 1.5};
  int output[] = {-99, -99};
  void* plan = nullptr;
  int status = fortml_cuda_knn_plan_create(
      train_x, train_class, sample_weight, class_label, 4, 1, 2, 1, 1, 0,
      &plan);
  if (status != 0 || plan == nullptr) return 3;
  status = fortml_cuda_knn_plan_predict(plan, query_x, 2, output);
  const bool correct = status == 0 && output[0] == -7 && output[1] == 11;
  const int destroy_status = fortml_cuda_knn_plan_destroy(plan);
  if (!correct || destroy_status != 0) return 4;
  std::printf("PASS CUDA kNN resident-plan nearest-neighbor oracle\n");
  return 0;
}
