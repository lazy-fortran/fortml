# Multiclass MLP probability parameter products

`mlp_classifier_t` now exposes fixed-input derivatives of the softmax
probability matrix with respect to the complete packed network state:

```fortran
call model%predict_proba_parameter_jvp(x, theta_dot, probabilities, &
    probabilities_dot, status)
call model%predict_proba_parameter_vjp(x, probabilities_bar, theta_bar, status)
```

The JVP is the ordinary classifier probability graph with `x_dot = 0`; the
VJP applies the stable softmax reverse rule and then the MLP reverse product.
Both routines validate fitted state, shapes, and finite values. The packed
parameter order is exactly `model%parameters()` and is shared with
`set_parameters`, `decision_function_jvp`, and the weighted multiclass
cross-entropy objective. This avoids a second parameter layout for
classification sensitivities.

Device wrappers make the capability boundary explicit. A selected CPU context
executes the host product. A selected CUDA context returns
`FORTNUM_NOT_IMPLEMENTED` because the resident multi-layer MLP classifier
graph is not yet linked; no hidden host fallback is attempted.

`test_mlp_classifier_parameter_products` checks the JVP against an independent
central finite-difference replay, verifies VJP/JVP Euclidean duality, checks CPU
dispatch, and checks the CUDA refusal. The release benchmark is
`fortml-bench/results/MLP_CLASSIFIER_PARAMETER_PRODUCTS.md`.
