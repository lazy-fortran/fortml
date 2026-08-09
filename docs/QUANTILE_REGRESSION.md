# Weighted linear quantile regression

`fortml_quantile_regression` supplies a dense, weighted multi-output affine
quantile estimator. `quantile_regression_t%fit(x,y,status[,quantile_levels,
l2,fit_intercept,sample_weight,max_iterations,tolerance])` accepts row-
oriented samples and one target column per quantile. A vector overload uses a
single level (the default is `0.5`). Levels are finite, strictly inside
`(0,1)`, unique, and sorted deterministically; target columns are permuted to
the same order before fitting. Intercepts are fitted by default and excluded
from the optional feature L2 penalty. Sample weights must be finite,
nonnegative, and have positive mass.

Fitting routes a registry-backed pinball objective through FortOpt L-BFGS-B.
Because pinball is nonsmooth, the fit path uses an explicit C1 Huberized
continuation (`quantile_lbfgsb_options_t%fit_smoothing`, default `0.1`) for
the Armijo callback. The committed coefficients and reported objective are
then evaluated with the exact pinball loss. Set the continuation to zero only
when a caller accepts the line-search boundary behavior. The result records
the continuation width, smoothed optimizer gradient norm, and exact
post-fit gradient norm.

`quantile_training_objective_t%initialize(model,x,target,l2,status[,
sample_weight,kink_tolerance,device_kind])` exposes the exact objective to
generic FortOpt/search code. `parameters()` and `set_parameters()` use the
Fortran column-major coefficient vector. `value_gradient`, `jvp`, and `vjp`
are analytic fixed-design products. Pinball objective HVPs are zero away from
residual kinks except for feature L2 curvature; `hvp` returns
`FORTNUM_NOT_IMPLEMENTED` when a positive-weight residual is exactly (within
`kink_tolerance`) zero. This explicit refusal prevents a fabricated Hessian
at the nondifferentiable point.

Prediction `jvp`/`vjp` products differentiate packed coefficients and
continuous inputs while holding the fit, level ordering, and sample weights
fixed. `quantile_levels()`, `quantiles()`, `coefficients()`, and the feature,
output, regularization, and intercept accessors expose deployment metadata.
`device_supported` reports CPU only; `predict_device` and CUDA objective or
optimizer requests return `FORTNUM_NOT_IMPLEMENTED` until resident CUDA
affine/pinball kernels are linked, with no hidden host fallback.

The independent `test_quantile_regression` gate checks sorted metadata,
weighted finite-difference and JVP/VJP products, piecewise HVP behavior,
FortOpt convergence with continuation, exact-kink refusal, and the CUDA
boundary. The release app is `app/fortml_bench_quantile_regression.f90`; the
NumPy/SciPy oracle and pinned records are maintained in the sibling
`fortml-bench` checkout.
