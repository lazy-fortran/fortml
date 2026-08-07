#ifndef FORTML_CUDA_ADAMW_H
#define FORTML_CUDA_ADAMW_H

/*
 * Native CUDA AdamW state API.
 *
 * The plan owns parameters and first/second moments in device memory.  The
 * gradient passed to `fortml_cuda_adamw_plan_step` must point to device
 * memory on the plan's selected device; the function never performs an
 * implicit host-to-device copy.  Creation and download are the explicit host
 * transfer boundaries.  Return values are CUDA runtime error codes (zero on
 * success).
 */
#ifdef __cplusplus
extern "C" {
#endif

int fortml_cuda_adamw_available(void);

int fortml_cuda_adamw_plan_create(
    const double* parameters, const double* first_moment,
    const double* second_moment, int n_parameters, double learning_rate,
    double beta1, double beta2, double epsilon, double weight_decay,
    int device_index, void** opaque_plan);

int fortml_cuda_adamw_plan_step(void* opaque_plan, const double* gradient);

int fortml_cuda_adamw_plan_download(
    void* opaque_plan, double* parameters, double* first_moment,
    double* second_moment, int* step_count);

int fortml_cuda_adamw_plan_destroy(void* opaque_plan);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* FORTML_CUDA_ADAMW_H */
