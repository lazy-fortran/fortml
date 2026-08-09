# Weighted Huber regression

`fortml_huber_regression` supplies a dense linear robust-regression estimator
and a FortOpt objective adapter. Inputs are row-oriented:
`x(n_samples,n_features)` and `y(n_samples,n_outputs)`; the vector overload
accepts `y(n_samples)`. Nonnegative sample weights are normalized by their
positive total mass. An intercept is fitted by default and is excluded from
the optional L2 penalty.

For residual `r = prediction - target` and positive `delta`, the per-row loss
is

```text
0.5*r*r                         if abs(r) <= delta
delta*(abs(r) - 0.5*delta)      otherwise.
```

`huber_training_objective_t` uses the shared parameter registry for the
coefficient block. With `optimize_l2` and/or `optimize_delta`, its packed
layout is `[coefficients, l2?, delta?]`; the optional coordinates are bounded
and have exact first and mixed second products. `value_gradient`, `jvp`, and
`vjp` are analytic. `hvp` differentiates the active residual branch and
returns `FORTNUM_NOT_IMPLEMENTED` when a nonzero-weight residual is at the
branch kink (within `kink_tolerance`).

`huber_optimize_lbfgsb` routes this exact callback through FortOpt L-BFGS-B.
The optimizer and its stopping state are fit-time boundaries; fixed-fit
prediction products remain available through `predict_jvp` and
`predict_vjp`. The current implementation is CPU-only. A selected CUDA
context returns a typed refusal and never falls back to host execution.

The independent `test_huber_regression` gate checks weighted value gradients,
JVP contraction, VJP duality, mixed HVP finite differences, FortOpt
convergence, the Huber-kink refusal, and the CUDA refusal. The release app is
`app/fortml_bench_huber_regression.f90`; the cross-engine benchmark and raw
provenance live in the sibling `fortml-bench` checkout.
