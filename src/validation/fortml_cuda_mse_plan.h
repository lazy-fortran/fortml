#ifndef FORTML_CUDA_MSE_PLAN_H
#define FORTML_CUDA_MSE_PLAN_H

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Resident no-autodiff weighted-MSE reduction plan.
 *
 * create copies the immutable target/prediction/weight arrays to the selected
 * CUDA device.  execute may then be called repeatedly without re-uploading
 * those arrays; only the reduced block partials and final scalar leave the
 * device.  All functions return zero on success and a nonzero CUDA/argument
 * error otherwise.
 */
int fortml_cuda_mse_plan_create(const double* target,
                                const double* prediction,
                                const double* sample_weight,
                                int n_samples,
                                int n_outputs,
                                int device_index,
                                void** opaque_plan);

int fortml_cuda_mse_plan_execute(void* opaque_plan, double* value);

int fortml_cuda_mse_plan_destroy(void* opaque_plan);

#ifdef __cplusplus
}
#endif

#endif
