# Binary Laplace-GP hyperparameter HVP

`gp_classification_t%hyperparameter_hvp(direction, product, status)` returns
the directional Hessian product of the binary Laplace mode-posterior envelope
with respect to the packed kernel log parameters. The fitted mode is
differentiated implicitly; this is distinct from `set_parameters`, whose
transactional fixed-state contract is used by prediction products.

```fortran
real(dp) :: direction(model%parameter_count())
real(dp) :: product(model%parameter_count())
direction = [0.07_dp, -0.04_dp]
call model%hyperparameter_hvp(direction, product, status)
```

For `K_dot = dK[direction]`, the implementation solves the mode tangent with
the resident posterior factorization,

\[
 (I + \sqrt W K\sqrt W)u = \sqrt W K_{dot}\alpha,
 \qquad f_{dot}=K_{dot}\alpha-K\sqrt W u,
\]

then differentiates `alpha = K^{-1} f_mode`. The kernel's analytic
`parameter_hvp` and `parameter_vjp` products contract the resulting outer
products. No finite-difference work is used in the library path.

`hyperparameter_hvp_device` dispatches a selected CPU context to this method.
Selected CUDA contexts return `FORTNUM_NOT_IMPLEMENTED`; there is no hidden
host fallback. The independent `test_gp_classification_hvp` fixture refits
logistic and probit models at central parameter probes, differentiates their
envelope gradients, checks transactional invalid updates, and checks the CUDA
boundary. The release workload and NumPy oracle are in
`fortml-bench/scripts/bench_gp_classification_hvp.py`.
