# Pairwise contrastive loss

`fortml_losses` now provides a differentiable pairwise contrastive objective for
Siamese and metric-learning heads. Samples are rows of two equal-shaped
embedding matrices. `labels(i)=1` denotes a matching pair and `labels(i)=0`
denotes a non-matching pair. For distance `d_i = ||a_i-b_i||_2` and positive
margin `m`, the per-pair value is

```text
0.5 d_i^2                         (label = 1)
0.5 max(0, m-d_i)^2               (label = 0).
```

The public routines are:

```fortran
use fortml_losses, only: contrastive_loss_value, contrastive_loss_jvp, &
    contrastive_loss_vjp, contrastive_loss_hvp

call contrastive_loss_value(a, b, labels, margin, value, status, weights)
call contrastive_loss_jvp(a, b, labels, margin, a_dot, b_dot, value, value_dot, &
    status, weights)
call contrastive_loss_vjp(a, b, labels, margin, value_bar, a_bar, b_bar, &
    status, weights)
call contrastive_loss_hvp(a, b, labels, margin, a_dot, b_dot, a_hvp, b_hvp, &
    status, weights)
```

Optional row weights are nonnegative and must have positive total mass. The
default `LOSS_REDUCTION_MEAN` divides by that mass; passing
`LOSS_REDUCTION_SUM` returns the unnormalised weighted sum. Input, tangent,
cotangent, and output shapes are checked transactionally and all numerical
inputs must be finite. Labels are restricted to zero and one and the margin
must be finite and strictly positive.

The derivative routines expose the exact Euclidean-distance products. They
return `FORTNUM_DOMAIN_ERROR` without leaving partial products when a
non-matching pair has zero distance (the norm is singular) or a pair is exactly
at the margin (the active-set branch is not twice differentiable). Matching
pairs use the squared-distance branch and remain smooth at zero distance.

`contrastive_loss_value_device(..., device_kind=FORTML_DEVICE_CUDA)` returns a
typed `FORTNUM_NOT_IMPLEMENTED` refusal until a resident pair-distance and
weighted-reduction CUDA kernel is linked. It never silently executes the CPU
path for a CUDA request. The release test
`test_contrastive_loss` checks the value against an independent pairwise
formula, JVP against a centered finite difference, VJP adjointness, HVP against
a finite difference of the VJP, both reductions, zero-distance and margin
refusals, and the CUDA boundary.

The routines are a reusable loss primitive rather than a hidden Siamese model:
an MLP/GP encoder can supply `a` and `b`, while a future resident paired
trainer can compose the same products through its model parameter callbacks.
