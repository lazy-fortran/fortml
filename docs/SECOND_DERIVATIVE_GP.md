# RBF/Matérn-5/2 second-derivative Gaussian process

`second_derivative_gp_t` is a deliberately small exact-GP reference for scalar
one-dimensional RBF or Matérn-5/2 processes. It accepts one observation row per value,
first derivative, second derivative, or (for RBF only) third derivative
observation. The `orders` vector uses `0`, `1`, `2`, and `3` for those operators.
The same vector is used for prediction and dense latent `joint_covariance`
queries. One Cholesky factor is shared by every mixed block, so value/
gradient/Hessian/third-derivative observations can be combined without
finite-difference observations.

```fortran
type(second_derivative_gp_t) :: gp
type(kernel_t) :: kernel
real(dp) :: x(4,1), y(4), query(3,1), mean(3), variance(3)
integer :: orders(4), query_orders(3)
type(fortnum_status_t) :: status

kernel = make_matern52_kernel(1, 1.6_dp, 0.75_dp, status) ! RBF is also supported
orders = [0, 1, 2, 0]
call gp%fit(x, orders, y, kernel, 0.035_dp, status)
query_orders = [0, 1, 2]
call gp%predict(query, query_orders, mean, variance, status)
```

RBF covariance blocks use exact distance derivatives through order six, and
RBF query-coordinate JVPs/VJPs use the seventh derivative. Matérn-5/2 remains
limited to orders `0:2`: its covariance reaches order four and its query
products use order five. A Matérn-5/2 fifth derivative requested exactly at a
training/query coincidence returns `FORTNUM_NOT_IMPLEMENTED` because that
derivative is discontinuous. The VJP satisfies the ordinary cotangent identity
for mean and latent variance. `joint_covariance` returns posterior latent
covariance and does not add observation noise.

The packed state is `[log(variance), log(lengthscale), log(noise_variance)]`.
For RBF states, `set_parameters` transactionally refactors the fitted model;
`log_marginal_likelihood`, its JVP/VJP, `hyperparameter_gradient`, and analytic
`hyperparameter_hvp` are available. The HVP differentiates the dense Cholesky
solve and covariance parameter blocks, including the mixed log-lengthscale
term. Matérn-5/2 hyperparameter products remain a typed
`FORTNUM_NOT_IMPLEMENTED` refusal until their order-four parameter jets are
generated. Invalid updates leave the fitted state unchanged.

The implementation is a CPU reference. `predict_device`,
`joint_covariance_device`, `predict_input_jvp_device`, and
`predict_input_vjp_device` dispatch selected CPU contexts and return the typed
`FORTNUM_NOT_IMPLEMENTED` status for selected CUDA contexts until a resident
derivative covariance/factorization kernel is linked. `device_supported` is
therefore true only for a fitted CPU model. Kernels other than RBF and
Matérn-5/2, dimensions other than one, and order values outside the
kernel-specific ranges (`0:3` for RBF, `0:2` for Matérn-5/2) are explicit
status errors. CUDA remains a typed refusal for prediction, covariance, and
all derivative products; no host fallback is used.

`test_second_derivative_gp` independently assembles both RBF and Matérn-5/2
derivative blocks, checks posterior means and variances, compares input JVPs
with central differences, checks VJP duality, and verifies the CUDA,
coincident-fifth-derivative, and non-RBF refusal boundaries. The dedicated
`test_second_derivative_gp_rbf_order3` gate adds an independent order-six
covariance oracle, likelihood-gradient finite differences, analytic HVP
finite differences, transactional parameter updates, and the order-three
refusal matrix. The RBF base expression is the FortSym-generated Gaussian leaf
(revision `26250ce`); the higher-order distance/HVP recurrence is independently
checked against the dense oracle before release. The release benchmark is
`fortml-bench/results/SECOND_DERIVATIVE_GP_RBF_ORDER3.md`.
