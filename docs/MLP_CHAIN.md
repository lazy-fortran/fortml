# Composable MLP module trees

`fortml_mlp_chain` composes ordinary dense `mlp_t` stages without introducing
a second layer implementation. A chain owns its children and presents one
deterministic parameter tree:

```fortran
type(mlp_t) :: encoder, head
type(mlp_chain_t) :: network
type(fortnum_status_t) :: status

call encoder%initialize([4, 8], status)
call head%initialize([8, 1], status, output_activation=MLP_LINEAR)
call network%initialize(4, status)
call network%append(encoder, status, name="encoder")
call network%append(head, status, name="head")
```

The append operation checks every interface width and rejects duplicate or
empty names. `network%parameters()` packs all child weights and biases in
stage order; `parameter_range("head",first,last,found)` provides stable offsets
for parameter groups, diagnostics, and checkpoints. `set_parameters` updates
the live children in the same order.

`parameter_block_from_mlp_chain` and `parameter_products_from_mlp_chain` route
the same live tree through FortML's shared registry/product facades. A
pipeline or outer search can therefore name the whole network as one block
while retaining stage offsets and exact product implementations.

## Products and optimization

The chain forwards dense MLP products through each stage. `jvp` includes both
the packed stage direction and the input direction. `vjp` returns one packed
cotangent plus the input cotangent. `hvp` differentiates that reverse recurrence
for a fixed output cotangent, including the downstream-cotangent tangent; it is
not a finite-difference wrapper.

`mlp_chain_objective_t` defines

\[
  L(\theta,\lambda)=\frac{1}{2n}\|f_\theta(X)-Y\|_F^2
       +\frac{\lambda}{2}\|\theta\|_2^2.
\]

The optional non-negative `lambda` coordinate has an analytic derivative and
mixed HVP block. `mlp_chain_optimize_lbfgsb` passes the objective's analytic
value/gradient callback directly to FortOpt L-BFGS-B, with explicit bounds for
both network parameters and the optional L2 coordinate. This is the same
parameter layout returned by `parameters`; no optimizer-specific repacking or
finite differences are involved.

## Device boundary

The chain and its objective currently advertise CPU support only. Selecting a
CUDA device, or passing `device_kind=FORTML_DEVICE_CUDA` to the objective or
L-BFGS-B options, returns `FORTNUM_NOT_IMPLEMENTED`. This typed refusal is
intentional: no host fallback is timed or hidden behind a GPU request. A future
resident fused chain must keep every child parameter, activation, batch, and
optimizer buffer on the selected device and add an independent CPU oracle and
transfer-inclusive benchmark before the capability changes.
