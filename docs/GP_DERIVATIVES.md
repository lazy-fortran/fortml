# Gaussian-process derivative capability

`gp_derivative_regression_t` models a row-wise observation list. A component
of `0` is a function-value observation, and component `j > 0` is the derivative
with respect to feature `j`. `fit`, `predict`, and `joint_covariance` are CPU
reference paths. `predict_jvp`/`predict_vjp` differentiate with respect to the
packed log-kernel and log-noise parameters.  `predict_input_jvp` and
`predict_input_vjp` differentiate a fixed query batch and therefore require a
third input derivative of each covariance block. `predict_input_hvp` is the
value-query second-order companion: for one query point and direction it
returns Hessian-vector products of the posterior mean and variance. It uses
exact third-input covariance contractions and the quadratic-form solve;
derivative-component queries are refused because they would require fourth
input derivatives.

The table is deliberately narrower than the full kernel catalog. “Mixed
blocks” means value/first-derivative covariance blocks are available wherever
the requested pair is smooth and finite. A refusal is a typed
`fortnum_status_t`, and there is no hidden finite-difference fallback.

| Kernel family | Mixed value/first-derivative fit and prediction | Parameter JVP/VJP/HVP products | Query-input JVP/VJP | Explicit boundary |
| --- | --- | --- | --- | --- |
| RBF | Yes | Yes, including analytic mixed-observation HVPs | Yes | none |
| ARD RBF | Yes | Yes, including analytic mixed-observation HVPs for log variance and every log lengthscale | Yes | CUDA covariance/factorization remains `FORTNUM_NOT_IMPLEMENTED` |
| Matérn 1/2 | Value-only, derivative blocks are singular at coincident points | Value-only and noncoincident gradient/JVP/VJP; mixed HVP refusal | No at coincident query blocks | `FORTNUM_DOMAIN_ERROR` at coincidence |
| Matérn 3/2 | Yes | Gradient/JVP/VJP; mixed HVP refusal | Yes away from coincidence | Nonzero directional third derivative at coincidence is `FORTNUM_NOT_IMPLEMENTED` |
| Matérn 5/2 | Yes | Gradient/JVP/VJP; mixed HVP refusal | Yes | mixed HVP `FORTNUM_NOT_IMPLEMENTED` |
| Periodic, rational-quadratic, cosine | Yes | Gradient/JVP/VJP; mixed HVP refusal | Yes | mixed HVP `FORTNUM_NOT_IMPLEMENTED` |
| Linear, constant | Yes | Yes; mixed-observation HVPs are analytic | Yes | none |
| Polynomial | Yes when the positive polynomial base is finite | Gradient/JVP/VJP and analytic mixed HVP (all four logarithmic parameters) | Yes when the positive base is finite | `FORTNUM_DOMAIN_ERROR` for a nonpositive base |
| Spectral mixture | Yes | Gradient/JVP/VJP for packed log-weights, log-scales, and signed means; mixed HVP refusal | Yes | mixed HVP `FORTNUM_NOT_IMPLEMENTED` until fourth input/parameter products exist |
| Sum/product composites | Yes when every child supports the requested product | Gradient/JVP/VJP for every supported child; mixed HVP analytic only for RBF/linear/constant/polynomial-only trees | Yes when every child supports it | Unsupported mixed HVP child returns `FORTNUM_NOT_IMPLEMENTED` |
| Validated user formula | Yes for formulas with defined input derivatives | Variance and formula input products where defined; mixed HVP refusal | Not implemented | `push_distance` additionally refuses at coincident points |
| White noise | Value-only | Value-only | Not a differentiable query covariance | Any derivative observation is refused as nonsmooth |

The exact derivative-GP likelihood value, parameter gradient, JVP, and scalar
VJP are analytic for every smooth built-in leaf listed above, including the
four logarithmic polynomial parameters. `log_marginal_likelihood_vjp` and its
`hyperparameter_vjp` alias take an explicit scalar cotangent and return the
packed kernel/noise pullback. For value-only observation lists,
`hyperparameter_hvp` uses the analytic kernel parameter-HVP and differentiated
Cholesky solve. For mixed value/first-derivative lists, the HVP is analytic for
RBF, linear, constant, polynomial, and sums/products built entirely from those leaves;
the implementation differentiates the dense covariance, Cholesky solve, and
each parameter covariance block in one direction. Polynomial mixed HVPs now
use a closed-form positive-base expression, including the degree-log tangent
at degree one, and are checked against an independent finite-difference
likelihood oracle. Matérn, periodic, rational-quadratic, cosine,
spectral-mixture, user-formula, and other leaves return
`FORTNUM_NOT_IMPLEMENTED` for a mixed HVP until their required second
input/parameter products have generated kernels and independent oracles. A
mixed HVP never silently central-differences the likelihood gradient.
Second-derivative observations, operator-valued outputs, sparse/variational
derivative inference, and resident CUDA covariance/factorization kernels are
not implemented. `device_supported(FORTML_DEVICE_CUDA)` is false and the
device prediction/covariance entry points return `FORTNUM_NOT_IMPLEMENTED`
without a host fallback.

`joint_covariance_jvp` and `joint_covariance_vjp` provide the same packed
kernel-log/noise-log derivative contract for the dense latent posterior
covariance. The JVP differentiates the prior, train/query cross-covariance,
and Cholesky solve in one direction. The VJP propagates a symmetric cotangent
through those same blocks and satisfies the independent adjoint identity in
`test_derivative_gp_products`. These products are CPU-only and return the
ordinary covariance-block refusal for kernels that cannot form the requested
derivative observation; there is no finite-difference fallback. CUDA remains
an explicit `FORTNUM_NOT_IMPLEMENTED` boundary until the resident covariance
and factorization graph is linked.
`joint_covariance_jvp_device` and `joint_covariance_vjp_device` make the
backend boundary explicit: selected CPU contexts dispatch exactly to these
products, while selected CUDA contexts return `FORTNUM_NOT_IMPLEMENTED` before
touching their outputs. `test_derivative_gp_device` checks both refusal and
CPU-dispatch contracts.

The independent behavior gates are `test_derivative_gp_products`,
`test_derivative_gp_spectral_mixture`,
`test_derivative_gp_device`, and `test_derivative_gp_capabilities`. The
product test compares dense covariance, parameter products, query JVP/VJP,
and adjoint identities against independent finite-difference oracles. The
capability test checks the refusal statuses above and verifies that a refused
query product does not invalidate a fitted value-only model.
`test_derivative_gp_polynomial` additionally assembles polynomial covariance
blocks independently and checks the likelihood gradient, mixed HVP, and
query-input JVP/VJP products against finite differences and an adjoint identity.
`test_derivative_gp_ard` independently assembles anisotropic RBF value,
first-derivative, and mixed-Hessian covariance blocks, then checks the packed
kernel/noise likelihood gradient and mixed HVP against central differences. It
also reconstructs a value-query mean/variance Hessian-vector product from
coordinate JVP finite differences.
The polynomial path is intentionally kept as a short closed-form expression
(`b = offset + scale*dot(x1,x2)`, `k = variance*b**degree`) rather than a
generated FortSym leaf. The independent block oracle covers every packed
parameter and the query third derivative, while the general FortSym kernel
generation task remains tracked in the roadmap.
The spectral-mixture derivative-GP gate independently assembles its dense
two-dimensional value/first-derivative covariance blocks and checks packed
parameter gradients, posterior covariance, query JVP/VJP, and the typed HVP
refusal.
