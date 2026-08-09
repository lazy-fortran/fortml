# MLP training precision contract

`mlp_training_options_t%precision_kind` makes numerical precision an explicit
part of the training state. The named modes are `MLP_PRECISION_FP64`,
`MLP_PRECISION_FP32`, `MLP_PRECISION_FP16`, and `MLP_PRECISION_BF16`; use
`mlp_precision_name` for stable diagnostics.

The trainer has a deterministic FP64 reference path and a CPU FP32
master-weight path. FP32 keeps optimizer state, checkpoints, and the public
model parameters in binary64. Each training input, target, model parameter,
and gradient crosses a binary32 rounding boundary before it is consumed by
the next product. Loss scaling checks the scaled gradient before the optimizer
mutates its state. A non-finite scaled gradient backs off the scale, skips the
update, and preserves the master parameters and optimizer moments.

FP16 and BF16 are recognized capability names and return
`FORTNUM_NOT_IMPLEMENTED` before model or optimizer state is mutated. They
remain typed refusals until storage and kernels for those formats are
available. Unknown precision values return `FORTNUM_DOMAIN_ERROR`.

Successful FP64 and FP32 training record the mode in
`mlp_training_state_t` and `mlp_training_checkpoint_t`, so a resumed
checkpoint cannot silently change precision. The checkpoint `parameters`
array is the binary64 master vector. After the final FP32 loss evaluation the
public model is restored to that master vector, and a resumed call reapplies
the binary32 boundary before evaluating a batch. No GPU path is inferred from
a precision request. Resident mixed-precision training remains an explicit
future CUDA capability.

```fortran
use fortml_mlp_training, only: mlp_training_options_t, MLP_PRECISION_FP32

options%precision_kind = MLP_PRECISION_FP32
call options%loss_scale%initialize(status, enabled=.true., initial_scale=128.0_dp)
call mlp_train(model, x, target, status, options, state, checkpoint=checkpoint)
```

`test_mlp_precision_contract` compares FP64 and FP32 SGD against independent
linear MSE recurrences, checks that the checkpoint stores the unrounded
master vector, verifies uninterrupted versus checkpoint/resume FP32
trajectories, and checks non-mutating typed refusals for FP16/BF16 plus a
domain error for an unknown mode. The benchmark records the FP32 master
trajectory, loss-scale state, and explicit unavailable FP16/BF16/CUDA rows.
