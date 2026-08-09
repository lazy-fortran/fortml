#ifndef FORTML_CUDA_MLP_CHAIN_H
#define FORTML_CUDA_MLP_CHAIN_H

#include <stdint.h>

/*
 * Resident, no-autodiff CUDA ABI for a composed dense MLP.
 *
 * `layer_sizes` has n_layers + 1 entries.  Each layer's weights are packed
 * output-major (output*n_inputs+input), in layer order, followed by every
 * layer's bias block in layer order.  Queries and results are feature-major
 * (feature*n_query+query).  Activations use the fortml_mlp codes 1..8:
 * linear, tanh, ReLU, GELU, SiLU, ELU, softplus, and leaky-ReLU.
 *
 * The plan copies the immutable topology and parameters once and keeps them
 * on the selected device.  Calls explicitly upload batches/tangents and
 * download requested products; no call executes a host fallback.  Forward
 * and reverse products are fixed-state products over all layer parameters.
 */

#ifdef __cplusplus
extern "C" {
#endif

int fortml_cuda_mlp_chain_available(void);

int fortml_cuda_mlp_chain_create(
    const int *layer_sizes, const int *activations,
    const double *weights, const double *biases, int n_layers,
    int device_index, void **opaque_plan);

int fortml_cuda_mlp_chain_predict(void *opaque_plan, const double *query_x,
                                  int n_query, double *outputs);

/* Forward product.  The packed tangents use the same layer-major layouts as
 * weights and biases at creation. */
int fortml_cuda_mlp_chain_jvp(
    void *opaque_plan, const double *query_x, const double *query_x_dot,
    const double *weights_dot, const double *biases_dot, int n_query,
    double *outputs, double *outputs_dot);

/* Reverse product.  Results use the packed layouts documented above. */
int fortml_cuda_mlp_chain_vjp(
    void *opaque_plan, const double *query_x, const double *output_bar,
    int n_query, double *query_x_bar, double *weights_bar,
    double *biases_bar);

/* Counters include model/topology upload and every explicit batch/product
 * transfer.  resident_bytes includes the model and reusable device buffers. */
int fortml_cuda_mlp_chain_transfer_stats(
    void *opaque_plan, uint64_t *host_to_device_bytes,
    uint64_t *device_to_host_bytes, uint64_t *resident_bytes);

int fortml_cuda_mlp_chain_destroy(void *opaque_plan);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* FORTML_CUDA_MLP_CHAIN_H */
