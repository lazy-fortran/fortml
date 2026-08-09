# Scalar Lagrangian MLP

`fortml_lagrangian_mlp::lagrangian_mlp_t` represents a time-independent scalar
Lagrangian `L(q,v)` with a dense MLP whose input is `[q,v]` and whose output is
one scalar. `initialize(n_coordinates, layers, status)` prepends the
`2*n_coordinates` input width to the supplied hidden/output widths; the final
width must be one.

For a supplied acceleration `a`, the Euler--Lagrange residual is evaluated as

```
R(q,v,a) = L_vq(q,v) v + L_vv(q,v) a - L_q(q,v).
```

`mass_matrix` returns `L_vv` for every sample and refuses a numerically
singular velocity Hessian. Positive definiteness is not imposed by this
general-purpose model; a mechanical separable metric may add that check in a
higher-level adapter. The input Hessian is obtained from the analytic MLP
reverse-over-forward product with a zero parameter tangent. No finite
difference is used by the production methods.

```fortran
type(lagrangian_mlp_t) :: model
real(dp) :: q(n, d), v(n, d), acceleration(n, d), residual(n, d)
real(dp) :: mass(n, d, d)
call model%initialize(d, [32, 32, 1], status)
call model%euler_lagrange_residual(q, v, acceleration, residual, status)
call model%mass_matrix(q, v, mass, status)
```

The CPU derivative graph is the reference contract. `select_device(CUDA)` and
CUDA initialization return `FORTNUM_NOT_IMPLEMENTED`; there is no host
fallback hidden behind a device request. Parameter products through the MLP
remain available through the underlying parameter accessors, while
derivative-through-fit, gauge terms, explicit time dependence, and learned
constraints remain separate work.

`test_lagrangian_mlp` checks the state-gradient adjoint identity, mass-matrix
and Euler--Lagrange finite-difference oracles, singular-Hessian transaction,
and the typed CUDA refusal.
