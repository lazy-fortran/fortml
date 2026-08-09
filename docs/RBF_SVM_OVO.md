# One-vs-one RBF SVM classification

`fortml_ovo_rbf_svm_classifier::ovo_rbf_svm_classifier_t` fits one dense
finite-basis [`rbf_svm_classifier_t`](API.md#fortml_rbf_svm_classifier) for
every pair of sorted integer classes. Pair `(i,j)` uses only rows belonging to
those classes, maps the lower class to `0` and the upper class to `1`, and
retains the pair's own training basis.

The deployment policy is explicit and deterministic: each pair contributes
its binary sigmoid probabilities to its two classes, and the accumulated
votes are divided by `n_pairs`. Thus `predict_proba` is a nonnegative simplex
row, while `predict` returns the original sorted integer label with the first
class winning exact ties. This is a normalized pairwise-vote policy, not a
claim to reproduce scikit-learn's private pairwise-coupling solver.

```fortran
use fortml_ovo_rbf_svm_classifier, only: ovo_rbf_svm_classifier_t
type(ovo_rbf_svm_classifier_t) :: model
type(fortnum_status_t) :: status
real(dp) :: x(n_samples,n_features), p(n_query,3)
integer :: labels(n_samples), prediction(n_query)

call model%fit(x, labels, status, c=2.0_dp, gamma=0.6_dp, &
    max_iterations=50000, tolerance=1.0e-6_dp)
call model%predict_proba(x_query, p, status)
call model%predict(x_query, prediction, status)
```

`classes()` and `pair_classes()` expose ascending class labels and the
deterministic `(negative,positive)` pair order. `decision_function` returns
one margin per pair. `parameters()` concatenates the child vectors
`[coefficients, intercept, log(gamma)]` in pair order; because each pair keeps
its own basis, `parameter_count()` and the packed offsets are allowed to vary
by pair.

Fixed-state derivatives are analytic. The input and packed-parameter JVP/VJP
families are available for both `decision_function` and `predict_proba`; fit,
active-set, squared-hinge boundary, and hard-label derivatives remain
explicitly outside the contract. `set_parameters` validates the whole packed
vector before publishing child updates.

Fit is transactional: malformed dimensions, weights, options, class counts,
or a failed pair solve leave a previously fitted model unchanged. CPU device
dispatch is supported. CUDA value and derivative execution currently returns
`FORTNUM_NOT_IMPLEMENTED` rather than silently copying data to the host.

The independent Fortran test is
`test/test_ovo_rbf_svm_classifier.f90`; the NumPy/SciPy benchmark and oracle
are published in the companion `fortml-bench` repository.
