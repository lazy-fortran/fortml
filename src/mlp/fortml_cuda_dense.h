#ifndef FORTML_CUDA_DENSE_H
#define FORTML_CUDA_DENSE_H

/*
 * Resident, prediction-only CUDA dense-affine ABI.
 *
 * This is deliberately a no-autodiff inference primitive.  The plan owns the
 * weight and bias arrays on one selected device.  Creation and every
 * prediction call are explicit host/device transfer boundaries; there is no
 * host fallback.  All indices use zero-based C conventions.
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

int fortml_cuda_dense_plan_destroy(void *opaque_plan);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* FORTML_CUDA_DENSE_H */
