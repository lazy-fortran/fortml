#ifndef FORTML_CUDA_ADAGRAD_H
#define FORTML_CUDA_ADAGRAD_H

/*
 * Native CUDA Adagrad state API.
 *
 * The plan owns parameters and accumulated-square state in device memory.
 * `fortml_cuda_adagrad_plan_step` requires a gradient pointer already resident
 * on the selected device; it never performs an implicit host-to-device copy or
 * falls back to a host update.  Creation and download are the explicit host
 * transfer boundaries.
 */
#ifdef __cplusplus
extern "C" {
#endif

int fortml_cuda_adagrad_available(void);

int fortml_cuda_adagrad_plan_create(
    const double* parameters, const double* accumulated_square,
    int n_parameters, double learning_rate, double epsilon, int device_index,
    void** opaque_plan);

int fortml_cuda_adagrad_plan_step(void* opaque_plan, const double* gradient);

int fortml_cuda_adagrad_plan_download(
    void* opaque_plan, double* parameters, double* accumulated_square,
    int* step_count);

int fortml_cuda_adagrad_plan_destroy(void* opaque_plan);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* FORTML_CUDA_ADAGRAD_H */
