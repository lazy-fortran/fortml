# Structure-aware finite-feature GP initialization

`fortml_mlp_structure_gp` provides a fixed-depth warm start for an ordinary
MLP with a linear output layer. It fits the existing finite-feature
last-layer GP posterior and records the complete hidden parameter prefix. The
apply and query methods validate that this prefix, its packed parameter layout,
and its initialization scale are unchanged. Only the final affine layer is
written.

```fortran
use fortml_mlp, only: mlp_t, MLP_TANH, MLP_LINEAR
use fortml_mlp_structure_gp, only: mlp_structure_gp_initializer_t

type(mlp_t) :: model
type(mlp_structure_gp_initializer_t) :: initializer
type(fortnum_status_t) :: status

call model%initialize([4, 32, 16, 1], status, &
    hidden_activation=MLP_TANH, output_activation=MLP_LINEAR)
call initializer%fit_apply(model, x, target, status, regularization=1.0e-4_dp)
```

The fitted hidden prefix is the packed vector returned by `model%parameters()`
up to the final weight and bias blocks. Its RMS scale and count are available
through `metadata()`, and the exact captured values are returned by
`hidden_parameters()`. A changed hidden parameter, topology, output
activation, or packed count returns `FORTNUM_DOMAIN_ERROR` before mutating the
model. `predict`, `jvp`, and posterior variance products use the same
validation. `apply_cuda` and `predict_cuda` return
`FORTNUM_NOT_IMPLEMENTED`. Host feature-map or solve work is never relabeled as
resident GPU execution.

The fitted coefficients solve

```text
(Z^T Z + lambda I) C = Z^T Y,
```

where `Z` is the frozen hidden feature map with a final intercept column. This
is a deterministic finite-width posterior-mean warm start. The metadata
therefore sets `exact_infinite_width=.false.`. NNGP covariance propagation,
sampled posterior weights, and structure-preserving Hamiltonian, symplectic,
or PINN mappings remain separate contracts.

`test_mlp_structure_gp` compares coefficients and predictions with an
independent dense normal-equation solve, checks hidden-state preservation and
mutation refusal, and exercises the typed CUDA boundary. The companion
benchmark uses an independent NumPy implementation and records CPU and typed
CUDA rows in `fortml-bench/results/MLP_STRUCTURE_GP.md`.
