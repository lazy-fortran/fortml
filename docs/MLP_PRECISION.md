# MLP training precision contract

`mlp_training_options_t%precision_kind` makes numerical precision an explicit
part of the training state. The named modes are `MLP_PRECISION_FP64`,
`MLP_PRECISION_FP32`, `MLP_PRECISION_FP16`, and `MLP_PRECISION_BF16`; use
`mlp_precision_name` for stable diagnostics.

The current trainer is a deterministic FP64 reference path. FP32, FP16, and
BF16 are accepted as recognized capability names but return
`FORTNUM_NOT_IMPLEMENTED` before the model or optimizer state is mutated.
This boundary is intentional: lower precision is not claimed until master
weights, loss scaling, overflow recovery, and deterministic reduction policy
are implemented together. Unknown precision values return
`FORTNUM_DOMAIN_ERROR`.

Successful FP64 training records the same mode in `mlp_training_state_t` and
`mlp_training_checkpoint_t`, so a resumed checkpoint cannot silently change
precision. The existing exact FP64 trajectory and derivative products remain
unchanged. No GPU path is inferred from a precision request; resident mixed
precision remains an explicit future CUDA capability.

```fortran
use fortml_mlp_training, only: mlp_training_options_t, MLP_PRECISION_FP64

options%precision_kind = MLP_PRECISION_FP64
call mlp_train(model, x, target, status, options, state, checkpoint=checkpoint)
```

`test_mlp_precision_contract` compares FP64 SGD against an independent linear
MSE recurrence, checks state/checkpoint metadata and stable names, and verifies
non-mutating typed refusals for FP32/FP16/BF16 plus a domain error for an
unknown mode. The benchmark records the same CPU reference and explicit
unavailable lower-precision/CUDA rows.
