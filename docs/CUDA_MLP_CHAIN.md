# Resident CUDA dense MLP-chain plan

`src/mlp/fortml_cuda_mlp_chain.{h,cu}` provides a bounded native-CUDA
inference plan for a sequential dense MLP.  The topology is supplied as
`layer_sizes(n_layers+1)` and `activations(n_layers)`; packed output-major
weights and layer-major biases are uploaded once and remain on the selected
device.  The eight activation codes shared with the resident dense plan are
supported: linear, tanh, ReLU, GELU, SiLU, ELU, softplus, and leaky-ReLU.

The C ABI uses feature-major batches (`feature*n_query+query`) and output-major
parameter blocks (`output*n_inputs+input`) within each layer.  `predict`
evaluates all layers in device workspaces.  `jvp` carries a primal and tangent
through every layer and accepts packed weight/bias tangents.  `vjp` stores
layer preactivations, propagates an output cotangent backwards, and returns
input, packed-weight, and packed-bias cotangents.  These are fixed-state
products; no host autodiff callback or CPU fallback is involved.

The plan reuses device workspaces for the largest batch seen so far.  Query and
product arrays cross an explicit ABI boundary, while model, topology,
intermediate activations, preactivations, and reusable product workspaces stay
resident.  `transfer_stats` reports successful host-to-device and
device-to-host bytes plus the current resident allocation, so a repeated batch
can be distinguished from model upload.  A malformed topology, nonfinite
parameter, or unsupported device returns a nonzero native code; the Fortran
wrapper maps unavailable native calls to `FORTNUM_NOT_IMPLEMENTED` and leaves
caller-owned result sentinels unchanged.

`cuda_mlp_chain_plan_t` is the typed Fortran wrapper.  Its `create` method
accepts normal packed arrays, and `predict`, `jvp`, `vjp`, `transfer_stats`,
`stage_count`, `input_count`, `output_count`, `parameter_count`, and `device`
expose the plan contract.  The ordinary GNU build links
`fortml_cuda_mlp_chain_stub.f90`; it reports a typed refusal rather than
silently running a CPU chain.

`test/run_cuda_mlp_chain.sh` compiles an independent CPU recurrence and runs
it against native CUDA when `nvcc` and a device are available.  The oracle
covers a three-layer tanh/ReLU/linear chain, all packed input/parameter
JVP/VJP products, repeated resident batches, transfer-counter lower bounds,
and compute-sanitizer memory checking.  `test_cuda_mlp_chain_api` covers the
ordinary-build refusal and sentinel preservation.  This slice deliberately
does not claim resident training, optimizer state, mixed precision, HVPs, or a
device-resident FortAD/FortSym graph.
