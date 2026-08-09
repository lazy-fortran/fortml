# Local-periodic derivative-GP mixed HVP

`gp_derivative_regression_t%hyperparameter_hvp` now supports mixed
function-value and first-coordinate derivative observations for
`make_local_periodic_kernel`.

The four kernel coordinates are logarithmic variance, envelope length scale,
periodic length scale, and period.  For a lag `d = x1 - x2`, the covariance is

```
k(s) = v exp(-a s - b sin(c sqrt(s))^2),  s = dot(d,d),
```

with `a = 1/(2 ell_envelope^2)`, `b = 2/ell_periodic^2`, and
`c = pi/period`.  The implementation carries the exact radial jet through
`k`, `d k/ds`, and `d² k/ds²`, together with a parameter directional tangent.
The observation blocks are then assembled as

```
C00 = k
C10 = 2 k_s d_i
C01 = -2 k_s d_j
C11 = -2 k_s delta_ij - 4 k_ss d_i d_j.
```

At coincident points the `sin(c sqrt(s))²` series is evaluated directly, so
the removable radial limits remain finite.  No finite-difference fallback is
used by the production HVP path; unsupported leaves retain their typed
`FORTNUM_NOT_IMPLEMENTED` status.

The independent `test_derivative_gp_local_periodic` gate refactors a dense
Cholesky likelihood from its own covariance oracle and central-differences its
packed likelihood gradient along an arbitrary five-coordinate direction.  The
release app is `fortml_bench_derivative_gp_local_periodic_hvp`; the companion
benchmark and raw CSV are
[`fortml-bench/results/DERIVATIVE_GP_LOCAL_PERIODIC_HVP.md`](https://github.com/lazy-fortran/fortml-bench/blob/main/results/DERIVATIVE_GP_LOCAL_PERIODIC_HVP.md)
and `results/derivative_gp_local_periodic_hvp.csv`.

This remains a CPU reference path.  The derivative-GP resident covariance and
factorization graph is not linked for CUDA, so the device boundary is an
explicit typed refusal rather than an implicit host copy.
