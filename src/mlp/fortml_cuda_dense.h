#ifndef FORTML_CUDA_DENSE_H
#define FORTML_CUDA_DENSE_H

#include <stdint.h>

/*
 * Resident, no-autodiff CUDA dense-affine ABI.
 *
 * This is deliberately a bounded no-autodiff affine value/derivative and
 * MSE-update primitive.  The plan owns the weight and bias arrays on one
 * selected device.  Creation and every value/product/update call are
 * explicit host/device transfer boundaries; there is no host fallback.  All
 * indices use zero-based C conventions.
 *
 * `weights` is output-major (output*n_inputs + input), `query_x` is
 * feature-major (input*n_query + query), and `outputs` is output-major
 * (output*n_query + query).  The Fortran wrapper transposes its ordinary
 * `weights(n_inputs,n_outputs)` view at creation time.
 */

#ifdef __cplusplus
extern "C" {
#endif

int fortml_cuda_dense_available(void);

int fortml_cuda_dense_plan_create(
    const double *weights, const double *bias, int n_inputs, int n_outputs,
    int activation, int device_index, void **opaque_plan);

int fortml_cuda_dense_plan_predict(void *opaque_plan, const double *query_x,
                                   int n_query, double *outputs);

/* Forward-mode product through the resident affine layer and activation.
 * Tangent weights use the same output-major layout as `weights` at creation;
 * query and query tangent arrays use feature-major layout. */
int fortml_cuda_dense_plan_jvp(void *opaque_plan, const double *query_x,
                               const double *query_x_dot,
                               const double *weights_dot,
                               const double *bias_dot, int n_query,
                               double *outputs, double *outputs_dot);

/* Reverse-mode product through the resident affine layer and activation.
 * `output_bar` is output-major and represents the cotangent of the returned
 * values.  The three result arrays use feature-major, output-major, and
 * output order respectively: query_x_bar[input*n_query+query],
 * weights_bar[output*n_inputs+input], and bias_bar[output]. */
int fortml_cuda_dense_plan_vjp(void *opaque_plan, const double *query_x,
                               const double *output_bar, int n_query,
                               double *query_x_bar, double *weights_bar,
                               double *bias_bar);

/* One fixed, no-autodiff training step for the resident affine layer.
 *
 * The objective is mean 1/2 squared error over all output/query pairs.  The
 * query and target batches are copied to the selected device, the gradients
 * are formed there, and the resident weights and bias are updated in place:
 *
 *     theta <- theta - learning_rate * grad(mean(1/2 * (f_theta(x)-y)^2)).
 *
 * `query_x` is feature-major and `target` is output-major, following the
 * layouts documented above.  The scalar loss is copied back to the host.
 * This bounded primitive deliberately does not expose an optimizer or
 * autodiff graph; callers that need either should use the CPU objective APIs.
 */
int fortml_cuda_dense_plan_train_mse(
    void *opaque_plan, const double *query_x, const double *target, int n_query,
    double learning_rate, double *loss);

/* Upload a batch for repeated no-autodiff updates.  The query and target
 * arrays remain resident until the next upload or plan destruction. */
int fortml_cuda_dense_plan_upload_batch(void *opaque_plan,
                                        const double *query_x,
                                        const double *target, int n_query);

/* Run one resident-batch MSE update.  Gradients and both Adam moments are
 * device-resident.  optimizer_kind is 1=SGD, 2=Adam, 3=AdamW.  SGD ignores
 * beta1, beta2, epsilon, and weight_decay; Adam ignores weight_decay; AdamW
 * applies decoupled weight decay to all parameters.  Only the scalar loss is
 * copied back to the host. */
int fortml_cuda_dense_plan_train_resident_mse(
    void *opaque_plan, double learning_rate, double beta1, double beta2,
    double epsilon, double weight_decay, int optimizer_kind, double *loss);

/* Copy the resident parameters to the host.  This is an explicit snapshot,
 * not an implicit host fallback for prediction or training. */
int fortml_cuda_dense_plan_get_parameters(void *opaque_plan, double *weights,
                                          double *bias);

/* Explicit transfer and residency accounting for this plan.  Byte counters
 * include model upload and all batch/result copies made by the ABI calls;
 * resident_bytes is the model allocation that remains on the selected
 * device. */
int fortml_cuda_dense_plan_transfer_stats(void *opaque_plan,
                                          uint64_t *host_to_device_bytes,
                                          uint64_t *device_to_host_bytes,
                                          uint64_t *resident_bytes);

int fortml_cuda_dense_plan_destroy(void *opaque_plan);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* FORTML_CUDA_DENSE_H */
