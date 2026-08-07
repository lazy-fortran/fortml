# Gaussian-process derivative capability

`gp_derivative_regression_t` models a row-wise observation list.  A component
of `0` is a function-value observation; component `j > 0` is the derivative
with respect to feature `j`.  `fit`, `predict`, and `joint_covariance` are CPU
reference paths.  `predict_jvp`/`predict_vjp` differentiate with respect to the
packed log-kernel and log-noise parameters.  `predict_input_jvp` and
`predict_input_vjp` differentiate a fixed query batch and therefore require a
third input derivative of each covariance block.

The table is deliberately narrower than the full kernel catalog.  “Mixed
blocks” means value/first-derivative covariance blocks are available wherever
the requested pair is smooth and finite.  A refusal is a typed
`fortnum_status_t`; there is no hidden finite-difference fallback.

| Kernel family | Mixed value/first-derivative fit and prediction | Parameter JVP/VJP/HVP products | Query-input JVP/VJP | Explicit boundary |
| --- | --- | --- | --- | --- |
| RBF | Yes | Yes | Yes | — |
| Matérn 1/2 | Value-only; derivative blocks are singular at coincident points | Value-only and noncoincident products | No at coincident query blocks | `FORTNUM_DOMAIN_ERROR` at coincidence |
| Matérn 3/2 | Yes | Yes | Yes away from coincidence | Nonzero directional third derivative at coincidence is `FORTNUM_NOT_IMPLEMENTED` |
| Matérn 5/2 | Yes | Yes | Yes | — |
| Periodic, rational-quadratic, cosine | Yes | Yes | Yes | — |
| Linear, constant | Yes | Yes | Yes | — |
| Polynomial | Yes when the positive polynomial base is finite | Yes (all four logarithmic parameters) | Yes when the positive base is finite | `FORTNUM_DOMAIN_ERROR` for a nonpositive base |
| Sum/product composites | Yes when every child supports the requested product | Yes when every child supports it | Yes when every child supports it | The first unsupported child propagates its typed refusal |
| Validated user formula | Yes for formulas with defined input derivatives | Variance and formula input products where defined | Not implemented | `push_distance` additionally refuses at coincident points |
| White noise | Value-only | Value-only | Not a differentiable query covariance | Any derivative observation is refused as nonsmooth |

The exact derivative-GP likelihood gradient is analytic for every smooth
built-in leaf listed above, including the four logarithmic polynomial
parameters.  Its
`hyperparameter_hvp` is intentionally a deterministic central difference of
that gradient; it is not advertised as an analytic third-order product.
Second-derivative observations, operator-valued outputs, sparse/variational
derivative inference, and resident CUDA covariance/factorization kernels are
not implemented.  `device_supported(FORTML_DEVICE_CUDA)` is false and the
device prediction/covariance entry points return `FORTNUM_NOT_IMPLEMENTED`
without a host fallback.

The independent behavior gates are `test_derivative_gp_products`,
`test_derivative_gp_device`, and `test_derivative_gp_capabilities`.  The
product test compares dense covariance, parameter products, query JVP/VJP,
and adjoint identities against independent finite-difference oracles; the
capability test checks the refusal statuses above and verifies that a refused
query product does not invalidate a fitted value-only model.
`test_derivative_gp_polynomial` additionally assembles polynomial covariance
blocks independently and checks the likelihood gradient and query-input
JVP/VJP products against finite differences and an adjoint identity.
The polynomial path is intentionally kept as a short closed-form expression
(`b = offset + scale*dot(x1,x2)`, `k = variance*b**degree`) rather than a
generated FortSym leaf; the independent block oracle covers every packed
parameter and the query third derivative, while the general FortSym kernel
generation task remains tracked in the roadmap.
