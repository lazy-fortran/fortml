# Multilabel Laplace Gaussian-process classification

`gp_multilabel_classification_t` fits one independent binary Laplace GP per
column of a dense `(n_samples,n_labels)` indicator matrix.  It is deliberately
different from `gp_multiclass_classification_t`: positive probabilities are
not normalized across labels, so a row may contain any combination of active
labels.  The label columns retain their input order and each head uses the
same kernel and selected logistic or probit likelihood.

The fit contract accepts finite nonnegative sample weights with positive total
mass and optional per-label decision thresholds in `(0,1)`.  Every label must
contain both indicator values.  `predict_proba` returns an
`(n_query,n_labels)` matrix of positive probabilities; `predict` applies the
stored thresholds and returns integer indicators.

The fitted model concatenates each binary head's packed kernel log parameters
in label order.  Latent and probability predictions expose query-input
JVP/VJP products and fixed-state packed kernel-parameter JVP/VJP products.
`hyperparameter_gradient` concatenates the exact binary Laplace envelope
gradients for the sum of the independent mode log posteriors.  As with the
binary GP contract, the Newton mode and curvature are held fixed in prediction
products; differentiating a full Laplace evidence or jointly correlated label
likelihood is a separate objective.

CPU device dispatch is exact.  CUDA latent/probability prediction and label calls return
`FORTNUM_NOT_IMPLEMENTED` until all binary Laplace states, covariance solves,
and the multilabel reduction are resident; no host fallback is implied.

The independent behavioral oracle is
[`test_gp_multilabel_classification`](../test/test_gp_multilabel_classification.f90).
The release benchmark compares the two-head weighted Laplace Newton recurrence,
posterior probabilities, and an input-JVP finite difference against NumPy:
[`fortml-bench/results/GP_MULTILABEL.md`](../../fortml-bench/results/GP_MULTILABEL.md).

## Shared kernel hyperparameter optimization

`shared_parameter_count()` and `shared_parameters()` expose the one common
kernel-log vector used by every label head. `set_shared_parameters(values,
status)` validates and factorizes candidate heads before committing any of
them, so failed values leave the fitted model unchanged. The smooth fixed-state
objective is available through
`fixed_state_value_gradient(values,value,gradient,status)`: it minimizes the
negative sum of the independent mode log posteriors while holding each fitted
Newton mode, likelihood curvature, and labels fixed. Its gradient is the exact
sum of binary prior-envelope contractions; no finite-difference gradient or
hidden refit is used.

For FortOpt callers, `gp_multilabel_training_objective_t` provides
`value_gradient`, `jvp`, `vjp`, and `fortopt`. The convenience
`gp_multilabel_optimize_lbfgsb(model,options,result,status)` applies bounded
FortOpt L-BFGS-B to this shared vector. Bounds, line-search controls, and
convergence diagnostics live in `gp_multilabel_lbfgsb_options_t` and
`gp_multilabel_lbfgsb_result_t`. This is intentionally a fixed-state outer
hyperparameter slice; call `fit` again when a fully recomputed Laplace mode is
required. The optimizer and products are CPU-only and return typed CUDA
refusals through the existing `device_supported` contract.
