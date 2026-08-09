# Variational multiclass GP log probabilities

`gp_variational_multiclass_classification_t` exposes stable one-vs-rest log
probabilities for the inducing-point Bernoulli variational GP classifier. The
columns follow the sorted integer labels returned by `classes()`.

```fortran
real(dp) :: log_probabilities(n_query, n_classes)
call model%predict_log_proba(x_query, log_probabilities, status)
```

Each positive Bernoulli link is evaluated in log space. A row-wise
log-sum-exp normalizes the columns, so `exp(log_probabilities(i, :))` sums to
one even when a positive link would underflow if it were formed first. The
logistic link uses a branch-stable log-sigmoid. The probit link uses a normal
log-CDF and an inverse-Mills derivative in the negative tail.

The fixed-state derivative surface has both packed variational-state and
query-input products:

```fortran
call model%predict_log_proba_parameter_jvp(x, direction, value, tangent, status)
call model%predict_log_proba_parameter_vjp(x, cotangent, parameter_bar, status)
call model%predict_log_proba_input_jvp(x, x_dot, value, tangent, status)
call model%predict_log_proba_input_vjp(x, cotangent, x_bar, status)
```

The packed parameter ordering is the concatenation of each binary model's
inducing mean, log-diagonal Cholesky entries, and strict lower-triangular
entries. The reverse products apply the simplex adjoint before accumulating the
per-class latent adjoints. The input products keep inducing locations,
variational state, and kernel hyperparameters fixed.

`predict_log_proba_device`, `predict_log_proba_parameter_vjp_device`, and
`predict_log_proba_input_vjp_device` provide explicit dispatch. CPU calls use
the reference implementation. CUDA calls return `FORTNUM_NOT_IMPLEMENTED`
until resident inducing solves and OVR log-sum-exp reductions are linked.
There is no host fallback behind a CUDA request.

The independent finite-difference and adjoint oracle is
[`test_gp_variational_multiclass_log_proba`](../test/test_gp_variational_multiclass_log_proba.f90).
The release workload is
[`fortml_bench_gp_variational_multiclass_log_proba`](../app/fortml_bench_gp_variational_multiclass_log_proba.f90).
