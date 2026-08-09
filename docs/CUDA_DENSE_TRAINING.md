# Resident CUDA dense training

`fortml_cuda_dense_api` exposes a deliberately bounded CUDA training primitive
for a single dense layer. It is useful as a low-level building block for a
larger MLP trainer while the general differentiable trainer remains the CPU
reference path.

The workflow is explicit:

1. Create a `cuda_dense_plan_t` with output-major parameters and a device
   index.
2. Call `upload_batch(query_x, target)` once for a feature-major batch. The
   batch remains on the selected device until the next upload or destruction.
3. Call `train_resident_mse(...)` for each update. The model, gradients, Adam
   moments, and batch stay resident; only the scalar loss crosses the ABI.
4. Call `parameters(...)` only when a host snapshot is needed.

Three no-autodiff optimizers are supported:

- `CUDA_DENSE_OPT_SGD` — plain gradient descent;
- `CUDA_DENSE_OPT_ADAM` — bias-corrected Adam;
- `CUDA_DENSE_OPT_ADAMW` — Adam with decoupled weight decay.

The loss is the mean `1/2 (f(x)-y)^2` over all output/query pairs. `beta1`,
`beta2`, and `epsilon` are validated by the typed wrapper. SGD ignores the
moment and decay options; Adam ignores `weight_decay`; AdamW applies
decoupled decay to both weights and bias. The optimizer step counter and
moments survive repeated updates and batch uploads.

The ordinary build links a typed unavailable stub. It returns
`FORTNUM_NOT_IMPLEMENTED` without silently running a CPU fallback. The native
ABI is intentionally no-autodiff and does not claim to replace the full MLP
training graph. `transfer_stats` reports host/device bytes and the complete
resident allocation, which makes accidental per-step batch copies observable.

Native checks:

```text
test/run_cuda_dense_resident_training.sh
test/run_cuda_dense_resident_training_sanitizer.sh
```

The independent CUDA oracle compares four SGD, Adam, and AdamW updates against
a CPU recurrence. The sanitizer gate runs the same test under
`compute-sanitizer --tool memcheck`.
