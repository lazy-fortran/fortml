#ifndef FORTML_CUDA_RMSPROP_H
#define FORTML_CUDA_RMSPROP_H

/*
 * Native CUDA RMSprop state API.
 *
 * The plan owns parameters and running RMSprop state in device memory.
 * `fortml_cuda_rmsprop_plan_step` requires a gradient pointer already
 * resident on the selected device; it never performs an implicit host copy
 * or falls back to a host update.  Creation and download are the explicit
 * host transfer boundaries.  Return values are CUDA runtime error codes
 * (zero on success).
 */
#ifdef __cplusplus
extern "C" {
#endif

int fortml_cuda_rmsprop_available(void);

int fortml_cuda_rmsprop_plan_create(
    const double* parameters, const double* square_average,
    const double* gradient_average, const double* momentum_buffer,
    int n_parameters, double learning_rate, double decay, double epsilon,
    double momentum, int centered, int device_index, void** opaque_plan);

int fortml_cuda_rmsprop_plan_step(void* opaque_plan, const double* gradient);

int fortml_cuda_rmsprop_plan_download(
    void* opaque_plan, double* parameters, double* square_average,
    double* gradient_average, double* momentum_buffer, int* step_count);

int fortml_cuda_rmsprop_plan_destroy(void* opaque_plan);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* FORTML_CUDA_RMSPROP_H */
