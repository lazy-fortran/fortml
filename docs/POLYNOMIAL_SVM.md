# Polynomial-kernel SVM

`fortml_polynomial_svm_classifier` provides a dense binary polynomial-kernel
SVM for finite, weighted training sets. It is the first non-RBF kernel SVM
slice in FortML and deliberately keeps its fit and device boundaries explicit.

## Contract

```fortran
type(polynomial_svm_classifier_t) :: model
call model%fit(x, labels, status, c=3.0_dp, gamma=0.4_dp, degree=2, coef0=1.0_dp)
call model%predict_proba(query, probabilities, status)
```

The kernel is `(gamma * dot(x,z) + coef0)**degree`, with positive `gamma`,
nonnegative `coef0`, and integer `degree >= 1`. The finite training rows are
the RKHS basis and FortOpt L-BFGS-B minimizes the weighted squared-hinge
objective. Labels are sorted and retained rather than re-encoded in the public
result. Probability columns follow that sorted order and are the stable
sigmoid of the signed margin; they are not a fitted Platt calibration.

The fit is transactional. A malformed shape, nonfinite value, invalid kernel
parameter, zero effective weight mass, or failed optimizer convergence leaves
an already fitted object unchanged. The independent behavioral gate is
`test_polynomial_svm_classifier`.

The packed fixed-state parameter order is `[coefficients,intercept,log(gamma),coef0]`.
Score and probability JVP/VJP products differentiate that state and query
inputs analytically. Derivatives through the FortOpt fit, squared-hinge
active-set changes, and hard labels are not claimed.

CPU dispatch executes the same routines as the direct methods. CUDA requests
return `FORTNUM_NOT_IMPLEMENTED` until a resident polynomial kernel and its
workspace are linked; FortML never silently copies the request to the host.
The independent release benchmark is the `polynomial_svm` lane in
`fortml-bench`.
