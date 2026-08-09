# Derivative-observation kernel matrix

`gp_derivative_regression_t` uses one mixed-observation contract. A component
of `0` denotes a function value and component `j` denotes the first derivative
with respect to input feature `j`. The covariance path therefore exercises the
kernel value, both input gradients, and the mixed input Hessian without a
finite-difference fallback.

The catalog oracle in `test_derivative_gp_kernel_matrix` covers RBF, ARD RBF,
Matérn 3/2, Matérn 5/2, periodic, local-periodic, rational-quadratic, cosine,
polynomial, spectral-mixture, linear, constant, change-point, and sum/product
composites. It compares every analytic gradient and mixed Hessian entry with
an independent central-difference value oracle, then fits and predicts a
mixed value/first-derivative GP for each kernel.

White noise remains an intentional typed refusal because a derivative of its
delta covariance is not a finite observation model. Matérn 1/2 and selected
coincident higher derivatives have explicit nonsmooth boundaries. CUDA
prediction and covariance require a resident derivative-GP factorization and
are reported as typed capability rows until that plan exists.

Run the behavioral matrix with:

```bash
FO_SCAN_FALLBACK=regex FO_FC=gfortran \
fo test test_derivative_gp_kernel_matrix
```

The companion benchmark records the same catalog, independent oracle errors,
and CPU/device capability rows in `fortml-bench/results/GP_DERIVATIVE_KERNEL_MATRIX.md`.
