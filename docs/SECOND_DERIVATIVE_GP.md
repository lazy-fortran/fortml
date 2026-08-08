# RBF second-derivative Gaussian process

`second_derivative_gp_t` is a deliberately small exact-GP reference for scalar
one-dimensional RBF processes. It accepts one observation row per value,
first derivative, or second derivative observation. The `orders` vector uses
`0`, `1`, and `2` for those three operators; the same vector is used for
prediction and dense latent `joint_covariance` queries. One Cholesky factor is
shared by every mixed block, so value/gradient/Hessian observations can be
combined without finite-difference observations.

```fortran
type(second_derivative_gp_t) :: gp
type(kernel_t) :: kernel
real(dp) :: x(4,1), y(4), query(3,1), mean(3), variance(3)
integer :: orders(4), query_orders(3)
type(fortnum_status_t) :: status

kernel = make_rbf_kernel(1, 1.6_dp, 0.75_dp, status)
orders = [0, 1, 2, 0]
call gp%fit(x, orders, y, kernel, 0.035_dp, status)
query_orders = [0, 1, 2]
call gp%predict(query, query_orders, mean, variance, status)
```

The generated RBF covariance uses the exact distance derivatives through
order four. Query-coordinate JVPs and VJPs use the fifth distance derivative;
the VJP satisfies the ordinary cotangent identity for mean and latent
variance. `joint_covariance` returns the posterior latent covariance and does
not add observation noise. Parameters are the packed
`[log(variance), log(lengthscale), log(noise_variance)]` state for metadata
and interoperability; hyperparameter products are not yet exposed by this
bounded type.

The implementation is a CPU reference. `predict_device` and
`joint_covariance_device` dispatch selected CPU contexts and return the typed
`FORTNUM_NOT_IMPLEMENTED` status for selected CUDA contexts until a resident
derivative covariance/factorization kernel is linked. `device_supported` is
therefore true only for a fitted CPU model. Non-RBF kernels, dimensions other
than one, and order values outside `0:2` are explicit status errors.

`test_second_derivative_gp` independently assembles the RBF derivative blocks,
checks posterior means, variances, and joint covariance, compares input JVPs
with central differences, checks VJP duality, and verifies the CUDA and
non-RBF refusal boundaries. The release benchmark is
`fortml-bench/results/SECOND_DERIVATIVE_GP.md`.
