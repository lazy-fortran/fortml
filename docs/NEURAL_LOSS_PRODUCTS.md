# Neural loss value and derivative products

`fortml_losses` is the shared loss facade used by the MLP objectives and by
standalone differentiable model code. This slice adds the missing second-order
products for softmax/log-softmax and focal BCE, and gives weighted
multiclass cross-entropy the same row-weight reduction contract as the other
neural losses.

## Stable softmax and log-softmax

The existing `softmax_value`, `softmax_jvp`, `softmax_vjp`,
`log_softmax_value`, `log_softmax_jvp`, and `log_softmax_vjp` routines subtract
the row maximum before exponentiating. The new second products are

```fortran
call softmax_hvp(logits, logits_dot, probabilities_bar, logits_hvp, status)
call log_softmax_hvp(logits, logits_dot, log_probabilities_bar, logits_hvp, status)
```

Each HVP is the directional derivative of the corresponding VJP for the
explicit output cotangent. This makes the vector-valued operation
unambiguous and composes with an outer FortOpt objective. Inputs, tangents,
cotangents, and outputs must have the same nonempty finite shape.

## Weighted multiclass cross-entropy

`softmax_cross_entropy_value`, `softmax_cross_entropy_jvp`,
`softmax_cross_entropy_vjp`, and `softmax_cross_entropy_hvp` accept optional
`sample_weight` and `reduction` arguments. Labels are one-based class columns.
With `LOSS_REDUCTION_MEAN` (the default), a row-weighted sum is divided by
positive weight mass. `LOSS_REDUCTION_SUM` leaves the weighted sum
unnormalized. Non-finite, negative, mismatched, or zero-support weights are a
typed domain refusal. The HVP is the exact weighted softmax Jacobian applied to
the logits tangent.

## Focal binary cross-entropy

`focal_binary_cross_entropy_with_logits_hvp` completes the analytic logits
value/JVP/VJP/HVP set for the existing stable focal BCE value/JVP/VJP routines:

```fortran
call focal_binary_cross_entropy_with_logits_hvp(logits, targets, alpha, gamma, &
    logits_dot, logits_hvp, status, sample_weight, reduction)
```

Targets may be relaxed in `[0,1]`, `alpha` is in `[0,1]`, and `gamma` is
nonnegative. The second derivative follows the exact focusing-factor product;
representationally saturated `1-p_t` rows use their finite limiting zero
products. The same positive row-weight mean/sum reductions apply.

## Multiclass focal cross-entropy

`focal_softmax_cross_entropy_value`, `focal_softmax_cross_entropy_jvp`,
`focal_softmax_cross_entropy_vjp`, and `focal_softmax_cross_entropy_hvp` apply
the focal factor to the true-class softmax probability:

```fortran
call focal_softmax_cross_entropy_hvp(logits, labels, gamma, logits_dot, &
    logits_hvp, status, sample_weight, reduction, class_weight)
```

Labels are one-based class columns. `gamma >= 0` is the focusing exponent;
`gamma=0` is exactly the weighted softmax cross-entropy contract. Optional
positive `class_weight` values multiply the true-class term, while
`sample_weight` controls the shared positive-mass mean/sum reduction. The
products are analytic scalar-composition derivatives through the true-class
probability and softmax JVP. A true-class probability at or below machine
tiny is refused because its focal derivatives are not representable in the
selected precision; a representationally saturated probability of one uses
the finite zero limiting products for `gamma>0`. The
`multiclass_focal_cross_entropy_*` names are generic aliases.

`mlp_classifier_options_t%focal_gamma` and
`mlp_classifier_lbfgsb_options_t%focal_gamma` select the same loss for
multiclass MLP fitting and the FortOpt objective. Their value/gradient and
parameter HVP paths share the standalone focal-softmax products, while the
default zero keeps the existing cross-entropy trajectory exact.

## Device boundary

`softmax_value_device`, `log_softmax_value_device`,
`softmax_cross_entropy_value_device`, and
`focal_binary_cross_entropy_with_logits_value_device` route CPU requests to the
reference implementation. CUDA requests return `FORTNUM_NOT_IMPLEMENTED`
without a host fallback or output mutation until resident loss kernels and
transfer accounting are available. Invalid device kinds return
`FORTNUM_DOMAIN_ERROR`.

`focal_softmax_cross_entropy_value_device` follows the same typed boundary;
CUDA requests leave the caller's scalar untouched.

The independent behavioral oracle is `test_neural_loss_products`; the release
benchmark is `fortml-bench/scripts/bench_neural_losses.py` and records CPU
softmax/log-softmax/focal products plus the typed CUDA capability row.

## Pairwise contrastive metric loss

`contrastive_loss_value`, `contrastive_loss_jvp`, `contrastive_loss_vjp`, and
`contrastive_loss_hvp` provide a shared primitive for Siamese/metric-learning
encoders. Matching rows use `0.5*d**2`; non-matching rows use
`0.5*max(0, margin-d)**2`, with optional nonnegative row weights and the same
mean/sum reduction convention. The two embedding matrices are differentiated
simultaneously, so the VJP returns equal-and-opposite row cotangents and the
HVP is the exact directional derivative of that VJP.

Value evaluation accepts zero distances and margin boundaries. Product calls
return a typed domain error for a non-matching zero distance (the Euclidean norm
is singular) and for a pair exactly at the margin (active-set second products
are not defined). Matching pairs remain smooth at zero because their branch is
the squared distance. `contrastive_loss_value_device` exposes the same CPU
path and a typed CUDA refusal until a resident pair-distance/reduction kernel is
available. The independent `test_contrastive_loss` covers formula parity,
JVP/VJP/HVP finite-difference oracles, reductions, boundaries, and dispatch.
