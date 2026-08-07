#include "../src/classification/fortml_cuda_forest.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>

namespace {

void cpu_oracle(const int *tree_offset, const int *node_feature,
                const int *node_left, const int *node_right,
                const double *node_threshold, const double *node_probability,
                const int *class_label, int n_trees, int n_classes,
                int n_query, int n_inputs, const double *query_x,
                double *probabilities, int *labels) {
  for (int query = 0; query < n_query; ++query) {
    for (int c = 0; c < n_classes; ++c)
      probabilities[c * n_query + query] = 0.0;
    for (int tree = 0; tree < n_trees; ++tree) {
      int node = tree_offset[tree];
      while (node_feature[node] >= 0) {
        const int feature = node_feature[node];
        node = query_x[feature * n_query + query] < node_threshold[node]
                   ? node_left[node]
                   : node_right[node];
      }
      for (int c = 0; c < n_classes; ++c)
        probabilities[c * n_query + query] +=
            node_probability[node * n_classes + c];
    }
    int best = 0;
    for (int c = 0; c < n_classes; ++c) {
      probabilities[c * n_query + query] /= n_trees;
      if (probabilities[c * n_query + query] > probabilities[best * n_query + query])
        best = c;
    }
    labels[query] = class_label[best];
  }
}

}  // namespace

int main() {
  if (fortml_cuda_forest_available() == 0) {
    std::printf("CUDA forest unavailable; test skipped\n");
    return 0;
  }
  if (cudaSetDevice(0) != cudaSuccess) return 2;

  // Two trees, each with one split.  Nodes are zero-based and probabilities
  // are node-major.  Boundary queries exercise strict x < threshold routing.
  const int tree_offset[] = {0, 3, 6};
  const int node_feature[] = {0, -1, -1, 1, -1, -1};
  const int node_left[] = {1, -1, -1, 4, -1, -1};
  const int node_right[] = {2, -1, -1, 5, -1, -1};
  const double node_threshold[] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0};
  const double node_probability[] = {
      0.0, 0.0, 1.0, 0.0, 0.0, 1.0,  // tree 0
      0.0, 0.0, 1.0, 0.25, 0.75, 0.0  // tree 1
  };
  const int class_label[] = {-7, 11, 42};
  const double query_x[] = {
      -1.0, 0.0, 0.5, 1.0, 2.0,  // feature 0
       0.0, 1.0, 1.0, 1.0, 2.0   // feature 1
  };
  constexpr int n_query = 5;
  double expected_probability[3 * n_query] = {};
  int expected_label[n_query] = {};
  cpu_oracle(tree_offset, node_feature, node_left, node_right, node_threshold,
             node_probability, class_label, 2, 3, n_query, 2, query_x,
             expected_probability, expected_label);

  void *plan = nullptr;
  int status = fortml_cuda_forest_plan_create(
      tree_offset, node_feature, node_left, node_right, node_threshold,
      node_probability, class_label, 2, 6, 2, 3, 0, &plan);
  if (status != 0 || plan == nullptr) return 3;
  double probability[3 * n_query];
  int labels[n_query];
  status = fortml_cuda_forest_plan_predict_proba(plan, query_x, n_query,
                                                  probability);
  if (status != 0) return 4;
  status = fortml_cuda_forest_plan_predict(plan, query_x, n_query, labels);
  if (status != 0) return 5;
  double max_error = 0.0;
  for (int i = 0; i < 3 * n_query; ++i)
    max_error = fmax(max_error, fabs(probability[i] - expected_probability[i]));
  bool correct = max_error < 1e-13;
  for (int i = 0; i < n_query; ++i) correct = correct && labels[i] == expected_label[i];

  // A second execution confirms that model arrays remain resident and that
  // repeated batches do not depend on stale output or host-side routing.
  const double repeat_query[] = {-2.0, 2.0, 0.0, 2.0};
  double repeat_expected[3 * 2] = {};
  int repeat_expected_label[2] = {};
  cpu_oracle(tree_offset, node_feature, node_left, node_right, node_threshold,
             node_probability, class_label, 2, 3, 2, 2, repeat_query,
             repeat_expected, repeat_expected_label);
  double repeat_probability[3 * 2];
  int repeat_label[2];
  status = fortml_cuda_forest_plan_predict_proba(plan, repeat_query, 2,
                                                  repeat_probability);
  if (status != 0) return 6;
  status = fortml_cuda_forest_plan_predict(plan, repeat_query, 2, repeat_label);
  if (status != 0) return 7;
  for (int i = 0; i < 6; ++i)
    correct = correct && fabs(repeat_probability[i] - repeat_expected[i]) < 1e-13;
  for (int i = 0; i < 2; ++i) correct = correct && repeat_label[i] == repeat_expected_label[i];
  const int destroy_status = fortml_cuda_forest_plan_destroy(plan);
  if (!correct || destroy_status != 0) return 8;
  std::printf("PASS CUDA resident forest prediction oracle (max error %.3e, repeats 2)\n",
              max_error);
  return 0;
}
