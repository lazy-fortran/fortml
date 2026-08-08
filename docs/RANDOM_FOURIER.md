# Random Fourier feature basis

`make_random_fourier_basis` provides a deterministic random-Fourier feature
map for a stationary-kernel approximation. The caller owns sampling; pass a
frequency matrix `frequencies(m,d)` and phase vector `phases(m)` generated from
an explicit seed:

```fortran
use fortml_basis, only: basis_map_t, make_random_fourier_basis
real(dp) :: frequencies(128, 3), phases(128)
type(basis_map_t) :: features
features = make_random_fourier_basis(3, frequencies, phases, status)
```

For component `k`, the map is

```
phi_k(x) = sqrt(2/m) cos(phases_k + sum_j frequencies(k,j) x_j).
```

The frequencies and phases are fixed transform state. Consequently
`parameter_count()` is zero and `set_parameters` accepts only an empty vector;
this makes a seeded feature map safe to reuse across folds and predictions.
The optional intercept is prepended by `basis_map_t`, as for the other basis
families.

The implementation has analytic value, input JVP, input VJP, and input HVP
products. The HVP differentiates the VJP of a scalar contraction with a fixed
feature cotangent. Parameter blocks are explicit zero-size blocks rather than
silently pretending that random frequencies are trainable. The transform is
static-lowering eligible on CPU; CUDA residency remains governed by the basis
pipeline device contract and is not claimed by this constructor.

The independent oracle in
[`test_basis_random_fourier.f90`](../test/test_basis_random_fourier.f90)
checks the direct trigonometric formula, central-difference JVP and HVP
products, the VJP adjoint identity, intercept placement, and the fixed-state
parameter refusal. For a kernel approximation, sample frequencies from the
spectral density of the target stationary kernel and record the seed and
sampling distribution alongside the model manifest.
