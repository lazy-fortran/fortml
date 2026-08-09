# Multiclass RBF SVM

`fortml_rbf_svm_multiclass` provides a deterministic one-vs-rest wrapper around
the finite-basis `rbf_svm_classifier_t`.  It accepts any finite integer labels,
sorts them once at fit time, and returns labels in that same order.  Each child
uses the weighted FortOpt L-BFGS-B squared-hinge RKHS objective; a fit is
transactional, so a failed child leaves an already fitted wrapper unchanged.

```fortran
type(rbf_svm_multiclass_t) :: model
call model%fit(x, labels, status, c=2.0_dp, gamma=0.6_dp, &
    max_iterations=2000, tolerance=1.0e-7_dp, sample_weight=weights)
call model%predict_proba(x_query, probabilities, status)
call model%predict(x_query, labels_query, status)
```

`decision_function` returns one signed margin per sorted class.  `predict_proba`
applies a stable sigmoid to each margin and normalizes each row to the
probability simplex.  `predict` selects the first class on an exact tie, making
the result deterministic.

The packed parameter vector is the concatenation of each child’s
`[coefficients, intercept, log(gamma)]` vector in sorted-class order.
`parameter_count`, `parameters`, and transactional `set_parameters` expose
this layout.  Fixed-state `predict_proba_jvp`/`predict_proba_vjp` differentiate
continuous query rows, while `predict_proba_parameter_jvp`/
`predict_proba_parameter_vjp` differentiate the packed child parameters.  The
fit active set, squared-hinge curvature boundary, hard labels, and label
encoding are discrete and are not differentiated.

CPU value and derivative dispatch is complete.  CUDA requests return typed
`FORTNUM_NOT_IMPLEMENTED` until a resident child-batch RBF kernel and reduction
are linked; no hidden host fallback is used.  Independent finite-difference
and adjoint tests live in `test/test_rbf_svm_multiclass.f90`.
