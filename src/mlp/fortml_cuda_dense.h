#ifndef FORTML_CUDA_DENSE_H
#define FORTML_CUDA_DENSE_H

/*
 * Resident, no-autodiff CUDA dense-affine ABI.
 *
 * This is deliberately a no-autodiff affine value/derivative primitive.  The
 * plan owns the weight and bias arrays on one selected device.  Creation and
 * every value or product call are explicit host/device transfer boundaries;
 * there is no host fallback.  All indices use zero-based C conventions.
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

int fortml_cuda_dense_plan_destroy(void *opaque_plan);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* FORTML_CUDA_DENSE_H */
