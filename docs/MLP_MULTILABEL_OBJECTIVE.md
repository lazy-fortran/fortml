# Multilabel MLP training objective

`mlp_multilabel_training_objective_t` exposes the shared multilabel MLP loss
through the same FortOpt callback used by the binary and multiclass neural
heads. The objective stores a copy of the feature rows, indicator targets, and
effective sample-by-label weights. Every evaluation updates the fitted MLP
parameter state, so direct products and a bounded L-BFGS-B solve use one loss
graph.

## Initialization

```fortran
use fortml_mlp_multilabel_classifier, only: &
    mlp_multilabel_training_objective_t

type(mlp_multilabel_training_objective_t) :: objective
call objective%initialize(model, x, indicators, 2.0e-3_dp, status, &
    sample_weight=sample_weight, class_weight=class_weight, &
    optimize_log_l2=.true.)
```

`model` must be a fitted `mlp_multilabel_classifier_t`. `x` has shape
`(n_samples,n_features)` and `indicators` has shape `(n_samples,n_labels)`.
Indicators are integer zero or one values. Sample weights are finite and
nonnegative. `class_weight` has shape `(2,n_labels)` and strictly positive
finite entries. Each label must retain positive effective weight after the two
weight arrays are multiplied. Inputs and the initial L2 coefficient must be
finite. A log-L2 objective requires a strictly positive initial coefficient.
Validation completes before the objective installs its model pointer or copies
data, so a failed initialization leaves the fitted model unchanged.

The packed vector is

```text
[network_parameters, l2]
```

when `optimize_l2` is selected, and

```text
[network_parameters, log(l2)]
```

when `optimize_log_l2` is selected. The two coordinate modes are mutually
exclusive. The logarithmic mode maps `z` to `exp(z)`, which keeps the physical
coefficient positive while retaining an unconstrained derivative coordinate.

## Products

The objective provides:

```fortran
parameters = objective%parameters()
call objective%value_gradient(parameters, value, gradient, status)
call objective%jvp(parameters, direction, value, value_dot, status)
call objective%vjp(parameters, value_bar, gradient_bar, status)
call objective%hvp(parameters, direction, product, status)
call objective%fortopt(fortopt_objective, status)
```

The data term is the mean of the independent weighted binary
cross-entropies. The regularizer is

```text
0.5 * l2 * dot(network_parameters, network_parameters).
```

For the log-L2 coordinate, the final gradient entry is
`0.5 * exp(z) * ||theta||^2`. The mixed HVP includes the exact cross term
`exp(z) * z_dot * theta` and the log-coordinate curvature term

```text
exp(z) * dot(theta, theta_dot)
  + 0.5 * exp(z) * ||theta||^2 * z_dot.
```

The network products use the MLP's analytic parameter JVP, VJP, and HVP
products. No finite differences are used by the objective or its FortOpt
callback.

## Bounded L-BFGS-B

`mlp_multilabel_optimize_lbfgsb` provides the direct training entry point:

```fortran
type(mlp_multilabel_lbfgsb_options_t) :: options
type(mlp_multilabel_lbfgsb_result_t) :: result

options%l2 = 2.0e-3_dp
options%optimize_log_l2 = .true.
options%log_l2_lower_bound = -12.0_dp
options%log_l2_upper_bound = 3.0_dp
call mlp_multilabel_optimize_lbfgsb(model, x, indicators, options, result, &
    status, sample_weight=sample_weight, class_weight=class_weight)
```

Network coordinates use `lower_bound` and `upper_bound`. Direct-L2 mode uses
`l2_lower_bound` and `l2_upper_bound`. Log-L2 mode uses the corresponding
`log_l2_*` bounds and reports the physical coefficient in `result%l2`. The
result records convergence, iterations, line-search evaluations, objective,
gradient norm, and the final physical L2 value. FortOpt convergence errors are
returned as `FORTNUM_CONVERGENCE_ERROR`. The wrapper preserves that failure
status.

The objective and optimizer are CPU contracts. A resident multi-layer CUDA
loss graph is not linked, so callers must keep device dispatch on the existing
typed `FORTNUM_NOT_IMPLEMENTED` boundary. The independent
`test_mlp_multilabel_objective` oracle checks weighted value products, central
finite-difference JVP and HVP identities, VJP duality, direct and positive
log-L2 coordinates, FortOpt callback routing, bounded solves, malformed
weights, and transactional model state.
