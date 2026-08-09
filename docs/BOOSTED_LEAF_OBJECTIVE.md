# Fixed boosted-tree leaf objectives

`fortml_boosted_leaf_objective` turns a fitted `xgboost_t` or `lightgbm_t`
ensemble into a smooth objective over its continuous fixed-structure
coordinates. The packed vector is

```text
[base_score, leaf weights in deterministic tree/node order]
```

Split thresholds, missing-value routes, categorical partitions, sampling
choices, tree scales, and the number of trees are discrete fitted state. They
are held fixed. A split boundary is therefore not silently differentiated.
The objective is instead the exact affine leaf map followed by a weighted
loss.

## API

```fortran
use fortml_boosted_leaf_objective, only: boosted_leaf_objective_t, &
    BOOSTED_LEAF_LOSS_SQUARED, BOOSTED_LEAF_LOSS_LOGISTIC

type(boosted_leaf_objective_t) :: objective
call objective%initialize_xgboost(model, x, target, &
    BOOSTED_LEAF_LOSS_LOGISTIC, status, sample_weight=weights, l2=0.1_dp)
call objective%value_gradient(theta, value, gradient, status)
call objective%jvp(theta, theta_dot, value, value_dot, status)
call objective%vjp(theta, value_bar, theta_bar, status)
call objective%hvp(theta, theta_dot, theta_hvp, status)
```

`initialize_lightgbm` has the same signature. The generic `%initialize`
binding accepts either tree type. `parameters()` returns the model's initial
packed coordinates and `parameter_count()` returns their length. The model is
not mutated by objective evaluations or by
`boosted_leaf_optimize_lbfgsb`; the optimizer result carries the optimized
coordinates so callers can apply their own persistence/update policy.

The normalized objective is

```text
sum_i w_i loss(A_i theta, target_i) / sum_i w_i
    + 0.5 * l2 * dot(theta, theta)
```

where `A` is the fixed tree-routing matrix. Squared loss is
`0.5*(margin-target)**2`; logistic loss is the stable binary
cross-entropy-with-logits expression. Gradients, directional JVPs, reverse
VJPs, and exact Hessian-vector products are analytic. The HVP includes the
L2 block and logistic curvature; no finite differences are used.

## FortOpt and devices

`boosted_leaf_optimize_lbfgsb` supplies the same objective callback to
FortOpt's bounded L-BFGS-B implementation. Bounds, memory, line-search, and
convergence tolerances are explicit in
`boosted_leaf_lbfgsb_options_t`; the result reports convergence, iteration
count, gradient norm, and optimized coordinates.

The complete operation graph is currently CPU-resident. A CUDA request returns
`FORTNUM_NOT_IMPLEMENTED` and never executes a host fallback. Split/routing
derivatives remain a separate typed boundary in the tree model APIs. The
independent oracle and release workload are
`test_boosted_leaf_objective` and
`fortml-bench/results/BOOSTED_LEAF_OBJECTIVE.md`.
