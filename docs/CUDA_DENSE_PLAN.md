# Resident CUDA dense-affine plan

`src/mlp/fortml_cuda_dense.{h,cu}` provides a small, explicit CUDA lowering
for one dense neural layer. It is an inference primitive, not a claim that
the complete MLP trainer or its FortAD/FortSym derivative graph is resident.
The plan owns the immutable weights and bias on one selected device and
performs an affine map followed by one of the eight `fortml_mlp` activations:
linear, `tanh`, ReLU, GELU, SiLU, ELU, softplus, or leaky-ReLU.

The C ABI uses output-major weights (`output*n_inputs+input`), feature-major
queries (`input*n_query+query`), and output-major results
(`output*n_query+query`). Creation validates dimensions, activation codes, and
finite host values before allocating. Prediction copies only the query batch
in and result out; it never calls a CPU implementation or transfers the
resident model back through the host. Destruction releases all model buffers
on the plan's selected device.

The typed `cuda_dense_plan_t` wrapper accepts the normal Fortran views
`weights(n_inputs,n_outputs)`, `query_x(n_query,n_inputs)`, and
`outputs(n_query,n_outputs)`, transposing only the weight layout required by
the C ABI. Its `jvp` method additionally accepts feature, weight, and bias
tangents and returns both the value and forward tangent. The native kernel
keeps the resident weights and evaluates the affine tangent followed by the
analytic derivative of each supported activation. An ordinary GNU build links
a stub and returns `FORTNUM_NOT_IMPLEMENTED` without changing output
sentinels. The `fortml_device` capability probe includes this native
availability symbol.

`test/run_cuda_dense_plan.sh` compiles the C ABI and an independent CPU oracle
when `nvcc` and a CUDA device are present. The gate checks every activation,
value and JVP products, two batches on a resident plan, finite-input
validation, and complete output arrays to a `3e-13` absolute tolerance.
Machines without a CUDA toolchain are reported as skipped, never as CPU
evidence. This primitive intentionally does not expose VJP, HVP, or optimizer
state; those remain on the FortAD/FortSym reference path until a full resident
derivative graph is available.
