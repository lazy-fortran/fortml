# FortML device contract

`fortml_device` is the explicit control-plane boundary for accelerator-aware
models and operators. It describes the selected backend and records data-region
metadata; it does not allocate buffers, launch kernels, or silently move an
array to the host.

## Select and query a backend

```fortran
use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, &
    FORTML_DEVICE_CUDA
use fortnum_status, only: status_ok, fortnum_status_t

type(fortml_device_t) :: device
type(fortnum_status_t) :: status

call device%select(FORTML_DEVICE_CPU, status)
if (.not. status_ok(status)) error stop "CPU selection failed"

call device%select(FORTML_DEVICE_CUDA, status, device_index=0)
if (.not. status_ok(status)) then
    ! FORTNUM_NOT_IMPLEMENTED means that this build has no CUDA kernel or
    ! runtime.  Keep the CPU context and choose it explicitly.
    call device%select(FORTML_DEVICE_CPU, status)
end if
```

`FORTML_DEVICE_CPU` is always available. CUDA is available only when the
current build supplies at least one of FortML's CUDA kernel entry points. The
default Fortran build links the CUDA stubs, so a CUDA request returns
`FORTNUM_NOT_IMPLEMENTED`; this is a recoverable refusal, not a host fallback.
`fortml_query_device` and `fortml_device_available` expose the same runtime
probe without changing a context. An invalid kind returns
`FORTNUM_DOMAIN_ERROR`.

The context reports `kind`, `backend`, `device_index`, `stream_id`, and a
`fortml_device_capability_t`. The capability record distinguishes host
accessibility, persistent residency, asynchronous streams, and CUDA kernel
availability. Stream IDs other than zero are refused until a backend supplies
an ownership and synchronization implementation. Selecting a new backend
while a data region is active is also refused.

## Register residency and transfers

An operator that owns an explicit OpenACC or CUDA data region can register its
accounting boundary:

```fortran
call device%begin_residency(bytes, status, owns_data=.true.)
call device%record_host_to_device(bytes, status)
call operator%matvec_device(input, output, status)
call device%record_device_to_host(output_bytes, status)
call device%end_residency(status)
```

`begin_residency` stores the declared byte extent and ownership flag. It does
not allocate or copy memory. Transfer methods increment byte and event
counters, and are valid only for an active CUDA residency. A CPU context
refuses host-device transfer recording with `FORTNUM_NOT_IMPLEMENTED`, which
prevents a CPU timing from being reported as a GPU transfer. The counters are
`host_to_device_bytes`, `device_to_host_bytes`,
`host_to_device_transfers`, and `device_to_host_transfers`.

The accounting object is deliberately separate from an operator's data
ownership. `owns_residency=.true.` means that the caller registered ownership
for reporting; `fortml_device` still never deallocates the operator's arrays.
Call `end_residency` before `clear` or backend selection. Repeated create,
register, and destroy cycles therefore remain observable and recoverable.

Current operator APIs retain their existing explicit `enter_data`/`exit_data`
calls. They are not implicitly converted by this metadata layer, and a model
must not claim complete device execution until every operation and transfer is
registered and validated against an independent CPU oracle.

## Estimator capability example: kNN

`knn_classifier_t%device_supported(kind)` reports estimator-level support. The
default GNU build links a CUDA stub and therefore reports CPU support only.
Native CUDA builds link `src/classification/fortml_cuda_knn.cu`. In that build,
`knn_classifier_t%predict_device(device,x,labels,status)` creates one resident
training-set plan per CUDA device and copies each query batch explicitly. The
kernel keeps squared-distance ordering and original-row tie rules and returns
labels without a host neighbor-selection fallback. `test/run_cuda_knn_plan.sh`
checks the kernel directly, while `test/run_knn_classifier_cuda.sh` checks the
Fortran API against the same nearest-neighbor oracle. CUDA remains unavailable
when the native object is not linked, and JVP/VJP products remain refused at
the discrete neighbor boundary.

## Direct RMSprop state kernel

The no-autodiff optimizer recurrence has a separate native CUDA C API in
`src/mlp/fortml_cuda_rmsprop.cu`. `fortml_cuda_rmsprop_plan_create` keeps the
parameters, square average, centered mean, and momentum buffer resident.
`fortml_cuda_rmsprop_plan_step` accepts a device-resident gradient and performs
one update without a host state round trip. `plan_download` is an explicit
inspection boundary. `test/run_cuda_rmsprop_state.sh` checks centered momentum
updates against an independent CPU recurrence. This kernel does not provide
MLP gradient or hypergradient evaluation. Those autodiff-sensitive paths stay
on the FortAD/FortSym reference until a complete device graph exists.

## Direct AdamW state kernel

The corresponding no-autodiff AdamW recurrence is exposed by the native CUDA C
API in `src/mlp/fortml_cuda_adamw.cu` (declarations are in
`src/mlp/fortml_cuda_adamw.h`). `fortml_cuda_adamw_plan_create` selects an
explicit nonnegative CUDA device and copies the initial parameters and moment
state to that device. `fortml_cuda_adamw_plan_step` requires a gradient pointer
that is already resident on the selected device; it performs the bias-corrected
first/second-moment update and decoupled weight decay without a hidden host
copy. `plan_download` is the explicit inspection/checkpoint boundary and
`plan_destroy` releases the resident state. `test/run_cuda_adamw_state.sh`
compares a multi-step trajectory against an independent CPU AdamW recurrence.

This fixed recurrence is intentionally not an MLP autodiff implementation. It
does not evaluate network gradients, JVP/VJP/HVP products, or hypergradients.
Those paths must remain on the FortAD/FortSym graph until a complete resident
device graph and an independent derivative oracle are available. A caller
must therefore not report end-to-end GPU training from this state kernel alone.

## Transfer-inclusive CUDA MSE reduction

`fortml_cuda_metrics%cuda_mean_squared_error` is a small no-autodiff CUDA
building block for the shared weighted regression metric. It accepts
column-major host target and prediction matrices plus optional row weights,
copies them explicitly to a temporary allocation on the selected device, and
performs the squared-error and block reduction in
`src/validation/fortml_cuda_metrics.cu`. The block partials are copied back
for the final scalar accumulation, and the wrapper records the exact transfer
bytes/events; there is no hidden CPU metric fallback. `fortml_cuda_mse_available()` and
`fortml_device` capability probing report whether the native object is linked.
The ordinary Fortran build therefore gives a typed `FORTNUM_NOT_IMPLEMENTED`
refusal, while `test/run_cuda_metric.sh` builds the CUDA object with `nvcc`
and checks the result against an independent weighted NumPy-equivalent oracle
on the selected GPU. This is transfer-inclusive metric evidence, not a claim
that a complete estimator or trainer is resident.

## New estimator contracts

The weighted elastic-net regressor, OVO logistic classifier, Laplace GP
classifier, typed MLP learning-rate schedules, MLP classifier prediction
products, deterministic random-forest classification, and basis/pipeline HVP
products expose
`device_supported(kind)`. They currently report CPU support only. Their
`predict_device`/`predict_proba_device` (or GP latent-prediction/HVP) entry
points dispatch exactly to the selected CPU context and return
`FORTNUM_NOT_IMPLEMENTED` for CUDA because no resident model kernel is linked.
The scalar GP likelihood helper likewise reports CUDA derivative products as
unsupported. These typed refusals are intentional: an unrelated CUDA kernel
in the process cannot turn a host allocation into a GPU benchmark, and no
implicit host/device transfer is hidden behind a prediction call. The
independent `test_device_contract_new_features` fixture checks these refusal
boundaries without requiring a CUDA driver.

The same rule covers `basis_map_t%hvp` and the horizontal, column-selecting,
and sequential pipeline HVP methods: the analytic CPU products are tested by
finite-difference-of-VJP oracles, but no CUDA derivative kernel is claimed.
Random-forest prediction is a fixed, piecewise tree route. Its
`random_forest_cuda_plan_t` now exposes ABI version 1 and shape/device metadata
as a no-autodiff native-CUDA planning boundary; create and prediction still
return typed refusals until a resident-state kernel and oracle are linked.
CUDA requests therefore remain refusals rather than OpenACC host fallbacks.

The mixed value/first-derivative GP has the same explicit boundary. Its
`gp_derivative_regression_t%predict_device` method dispatches selected CPU
contexts to the reference covariance solve and returns
`FORTNUM_NOT_IMPLEMENTED` for CUDA until covariance assembly, factorization,
and derivative-query products are resident on the device. The companion
`device_supported` query reports this distinction; derivative parameter and
query-input JVP/VJP methods are not presented as GPU-capable by the CPU-only
build. `test_derivative_gp_device` verifies that CUDA refusal leaves output
buffers untouched and that CPU dispatch agrees with the ordinary prediction
path.
