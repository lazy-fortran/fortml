#ifndef FORTML_CUDA_BOOSTED_TREE_H
#define FORTML_CUDA_BOOSTED_TREE_H

/*
 * Resident, fixed-topology CUDA additive-tree ABI.
 *
 * All indices are zero based.  tree_offset is a n_trees+1-element half-open
 * index array and node_feature is -1 for a leaf.  Query matrices use Fortran
 * column-major storage (feature*n_query + query).  A missing (NaN) query value
 * follows node_missing_left; infinities are rejected.  The plan owns every
 * model array on the selected device, so prediction only transfers a query
 * batch and the requested output.
 *
 * Leaf JVPs are zero for a fixed routing topology.  The JVP entry point
 * rejects NaN queries and exact split-boundary queries because their input
 * derivative is not defined.
 */

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

int fortml_cuda_boosted_tree_available(void);

int fortml_cuda_boosted_tree_plan_create(
    const int *tree_offset, const int *node_feature, const int *node_left,
    const int *node_right, const double *node_threshold,
    const double *node_weight, const int *node_missing_left,
    const double *tree_scale, int n_trees, int n_nodes, int n_inputs,
    double base_score, double learning_rate, int device_index,
    void **opaque_plan);

int fortml_cuda_boosted_tree_plan_predict(
    void *opaque_plan, const double *query_x, int n_query, double *margin);

int fortml_cuda_boosted_tree_plan_predict_jvp(
    void *opaque_plan, const double *query_x, const double *query_x_dot,
    int n_query, double *margin, double *margin_dot);

/* Return cumulative copy counters for the resident plan.  Model arrays are
 * uploaded exactly once by plan_create.  Each prediction adds only the query
 * and requested output bytes, so callers can distinguish resident execution
 * from a hidden host fallback. */
int fortml_cuda_boosted_tree_plan_transfer_stats(
    void *opaque_plan, uint64_t *host_to_device_bytes,
    uint64_t *device_to_host_bytes, uint64_t *resident_bytes);

int fortml_cuda_boosted_tree_plan_destroy(void *opaque_plan);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // FORTML_CUDA_BOOSTED_TREE_H
