#ifndef FORTML_CUDA_FOREST_H
#define FORTML_CUDA_FOREST_H

/*
 * Resident, prediction-only CUDA random-forest ABI.
 *
 * All indices in this C ABI are zero based.  tree_offset is a
 * n_trees+1-element half-open index array, node_feature is -1 for a leaf,
 * and node_probability is node-major (node*n_classes + class).  Query and
 * output matrices use Fortran column-major storage (feature*n_query+query
 * and class*n_query+query respectively).  The plan owns the model arrays on
 * the selected device; each prediction copies only the query batch in and
 * the requested result out.
 */

#ifdef __cplusplus
extern "C" {
#endif

int fortml_cuda_forest_available(void);

int fortml_cuda_forest_plan_create(
    const int *tree_offset, const int *node_feature, const int *node_left,
    const int *node_right, const double *node_threshold,
    const double *node_probability, const int *class_label, int n_trees,
    int n_nodes, int n_inputs, int n_classes, int device_index,
    void **opaque_plan);

int fortml_cuda_forest_plan_predict_proba(void *opaque_plan,
                                          const double *query_x, int n_query,
                                          double *probabilities);

int fortml_cuda_forest_plan_predict(void *opaque_plan, const double *query_x,
                                    int n_query, int *labels);

int fortml_cuda_forest_plan_destroy(void *opaque_plan);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // FORTML_CUDA_FOREST_H
