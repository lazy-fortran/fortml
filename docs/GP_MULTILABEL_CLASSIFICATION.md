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
