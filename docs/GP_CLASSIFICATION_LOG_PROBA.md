# Binary Laplace-GP log probabilities

`gp_classification_t` now exposes the scikit-learn-style
`predict_log_proba` surface alongside `predict_proba`.  Both columns use the
ascending labels returned by `classes()`, so a consumer can pass either output
directly to a log-loss or calibration report without reconstructing the class
ordering.

The value path is the logarithm of the already integrated Laplace predictive
probability.  A finite `tiny(real64)` floor is applied only when a probit tail
rounds to zero. This keeps serialized benchmark rows finite and makes the
boundary explicit rather than returning `-Inf`.  The logistic and ordinary
probit regions therefore agree with the probability API to machine precision.

The input and fixed-state kernel-parameter products are available as
`predict_log_proba_jvp`, `predict_log_proba_vjp`,
`predict_log_proba_parameter_jvp`, and
`predict_log_proba_parameter_vjp`.  They compose the existing probability
products with `d log(p) = d p / p`.  At the floating-point floor the tangent
is reported as zero, matching the finite clipped value.  Reverse products
apply the corresponding `1 / max(p,tiny)` cotangent before dispatching to the
probability VJP.  `set_parameters` changes only the kernel coordinates and
rebuilds the covariance factorizations while retaining the converged Newton
mode, `alpha`, and likelihood curvature. It is the state update used for
central-difference and L-BFGS-B fixed-state checks.

CPU execution is complete for logistic and probit likelihoods.  A selected
CUDA device returns `FORTNUM_NOT_IMPLEMENTED` for log-probability prediction:
the covariance solve and Laplace state are not resident, and no hidden host
fallback is counted as GPU support.

The independent oracle is `test_gp_classification_log_proba`.  It checks
probability/log round trips, parameter and input JVP central differences,
parameter and input VJP adjoint identities, and the typed CUDA refusal.  The
release benchmark is `fortml-bench/results/GP_CLASSIFICATION_LOG_PROBA.md`.

## Multiclass OVR Laplace-GP

`gp_multiclass_classification_t` provides the same log-probability surface for
its sorted one-vs-rest simplex: `predict_log_proba` and
`predict_log_proba_device`, together with input and packed fixed-state kernel
products (`predict_log_proba_jvp`, `predict_log_proba_vjp`,
`predict_log_proba_parameter_jvp`, and `predict_log_proba_parameter_vjp`).
The implementation first applies the OVR normalization, then composes the
result with the finite logarithm and its exact `1 / max(p,tiny)` reverse rule.
Thus `exp(predict_log_proba(x))` agrees with `predict_proba(x)` while the
probability columns continue to sum to one.

The multiclass CPU path is covered by `test_gp_multiclass_log_proba`, which
uses central differences over query features and a JVP/VJP dot-product oracle
for both input and packed kernel coordinates.  CUDA remains an explicit
`FORTNUM_NOT_IMPLEMENTED` refusal until the OVR Laplace states and
normalization reduction are resident.
