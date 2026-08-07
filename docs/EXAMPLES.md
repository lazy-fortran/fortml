# fortml examples

The programs in `app/` and `test/` are the executable examples. The tests add
independent numerical oracles, so they also state the tolerance and mathematical
contract behind each call sequence.

Run one with either driver:

```sh
fo test test_gaussian_process
fpm test --target test_gaussian_process
```

Benchmark applications use `fo exec`, for example:

```sh
fo exec fortml_bench_gp
```

## Dense regression with a basis map

Do not add the same intercept twice. This example lets the linear model add its
intercept and asks the basis map only for `x` and `x**2`.

```fortran
program basis_regression_example
    use, intrinsic :: iso_fortran_env, only: dp => real64
    use fortml_basis, only: basis_map_t, make_polynomial_basis
    use fortml_linear_regression, only: linear_regression_t
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    real(dp) :: x(5, 1), target(5), phi(5, 2), prediction(5)
    type(basis_map_t) :: basis
    type(linear_regression_t) :: model
    type(fortnum_status_t) :: status

    x(:, 1) = [-2.0_dp, -1.0_dp, 0.0_dp, 1.0_dp, 2.0_dp]
    target = 1.0_dp + 2.0_dp*x(:, 1) - 0.5_dp*x(:, 1)**2

    basis = make_polynomial_basis(1, 2, status)
    if (.not. status_ok(status)) error stop "basis construction failed"
    call basis%evaluate(x, phi, status)
    if (.not. status_ok(status)) error stop "basis evaluation failed"
    call model%fit(phi, target, status)
    if (.not. status_ok(status)) error stop "linear fit failed"
    call model%predict(phi, prediction, status)
    if (.not. status_ok(status)) error stop "linear prediction failed"

    if (maxval(abs(prediction - target)) > 1.0e-12_dp) &
        error stop "polynomial fit disagrees with the known target"
end program basis_regression_example
```

The full basis test covers polynomial, Fourier, radial, spline, and callback
maps: [test_basis.f90](../test/test_basis.f90). The vector and matrix regression
forms are in [test_linear_regression.f90](../test/test_linear_regression.f90).

## MLP products through one packed interface

An MLP must have the `target` attribute while a parameter adapter points to it.
The adapter keeps input data fixed and differentiates the packed model vector.

```fortran
use fortml_mlp, only: mlp_t
use fortml_parameter_products, only: parameter_products_t, &
    parameter_products_from_mlp

type(mlp_t), target :: model
type(parameter_products_t) :: products
real(dp) :: x(3, 2), y(3, 1), y_dot(3, 1), y_bar(3, 1)
real(dp), allocatable :: theta(:), direction(:), theta_bar(:), theta_hvp(:)

call model%initialize([2, 3, 1], status)
theta = model%parameters()
allocate(direction(size(theta)), theta_bar(size(theta)), &
    theta_hvp(size(theta)))
direction = 0.1_dp
y_bar = 1.0_dp

call parameter_products_from_mlp(products, "network", model, status)
call products%jvp(x, direction, y, y_dot, status)
call products%vjp(x, y_bar, theta_bar, status)
call products%hvp(x, y_bar, direction, theta_hvp, status)
```

[test_parameter_products.f90](../test/test_parameter_products.f90) checks that
sequence against finite differences, the adjoint identity, and Hessian
symmetry. Direct parameter and input products are in
[test_mlp.f90](../test/test_mlp.f90). The optimizer seam is exercised by
[test_mlp_adam.f90](../test/test_mlp_adam.f90).

## Neural probabilistic models

The BNN, VAE, and RNN use different objective interfaces:

```fortran
! Bayesian neural network: seeded Monte Carlo ELBO products.
call bnn%initialize([2, 4, 1], 16, 31415, status)
call bnn%elbo(x, target, value, status)
call bnn%elbo_vjp(x, target, 1.0_dp, gradient, status)

! VAE: the batch size is fixed at initialization.
call vae%initialize(3, 4, 2, 5, 31415, status, &
    likelihood_variance=0.4_dp)
call vae%elbo_gradient(batch_5_by_3, value, gradient, status)
call vae%reconstruct(batch_5_by_3, reconstruction, status)

! Vanilla RNN: time, batch, feature ordering and a zero initial state.
call rnn%initialize(2, 3, 1, status)
call rnn%forward(sequence_t_b_2, output_t_b_1, hidden_t_b_3, status)
call rnn%loss_gradient(sequence_t_b_2, target_t_b_1, value, gradient, status)
```

Complete seeded and finite-difference examples are
[test_bnn.f90](../test/test_bnn.f90) and
[test_vae_rnn.f90](../test/test_vae_rnn.f90). The generic Gaussian variational
family, minibatch scaling, likelihood callback, and `fortopt` update appear in
[test_variational.f90](../test/test_variational.f90).

## Derivative observations

The component array identifies the observation at each row. Zero denotes a
function value. A positive component is a one-based input dimension.

```fortran
use fortml_derivative_gaussian_process, only: gp_derivative_regression_t
use fortml_kernels, only: kernel_t, make_rbf_kernel

type(gp_derivative_regression_t) :: model
type(kernel_t) :: kernel
real(dp) :: x(4, 1), y(4, 1), query(2, 1)
real(dp) :: mean(2, 1), variance(2)
integer :: component(4), query_component(2)

x(:, 1) = [-1.0_dp, -0.25_dp, 0.5_dp, 1.25_dp]
component = [0, 1, 0, 1]       ! f, df/dx, f, df/dx
y(:, 1) = [1.0_dp, -0.5_dp, 0.25_dp, 2.5_dp]
query(:, 1) = [0.0_dp, 0.75_dp]
query_component = [0, 1]

kernel = make_rbf_kernel(1, 1.0_dp, 0.8_dp, status)
call model%fit(x, component, y, kernel, 1.0e-5_dp, status)
call model%predict(query, query_component, mean, variance, status)
```

[test_gaussian_process.f90](../test/test_gaussian_process.f90) contains the
dense mixed-covariance oracle and the Matérn and white-noise refusal cases.
Correlated output axes use a separate intrinsic-coregionalization example in
[test_multi_output_gp.f90](../test/test_multi_output_gp.f90).

## Approximate GP selection

The inference policy reports an algorithm. It does not construct or fit the
corresponding model.

```fortran
use fortml_inference_policy, only: inference_problem_t, inference_choice_t, &
    select_inference_policy, inference_policy_name

type(inference_problem_t) :: problem
type(inference_choice_t) :: choice

problem%n_samples = 20000
problem%n_features = 3
problem%n_outputs = 1
problem%inducing_budget = 256
call select_inference_policy(problem, choice, status)
write (*, '(a)') inference_policy_name(choice%policy)
```

The model examples are split by approximation because their state differs:

| Method | Executable example | Required caller choice |
| --- | --- | --- |
| Variational inducing GP | [test_sparse_gp.f90](../test/test_sparse_gp.f90) | Inducing points and the variational mean/factor |
| SoR, DTC, FITC, PITC | [test_sparse_prior_gp.f90](../test/test_sparse_prior_gp.f90) | Method, inducing points, and PITC block size |
| Local aggregations | [test_local_experts.f90](../test/test_local_experts.f90) | Partition, expert count, aggregation, and optional GRBCM communication seed |
| Tensor-grid SKI and SoD | [test_ski_gp.f90](../test/test_ski_gp.f90) | Grid budget, RBF restriction for `d > 1`, or subset size |

The alternate partition and two-dimensional grid-budget calls are:

```fortran
call experts%fit_clustered(x, y, 8, status)
call ski%initialize(points_2d, rbf, 256, noise, status)
call ski%cross_matvec(query_2d, coefficients, mean, status)
```

The review driver is
[fortml_bench_scalable_gp.f90](../app/fortml_bench_scalable_gp.f90).
`*_clustered`, including `grbcm_clustered`, selects clustered local fits. For
multidimensional RBF SKI, CLI `m` is the total grid budget. A budget below two
points per axis yields an all-NaN refusal row. Commands are in the
[benchmark reproduction instructions](../../fortml-bench/results/SCALABLE_GP.md#reproduce).

The [feature driver](../app/fortml_bench_gp_features.f90) covers `logdet`,
`predictive_variance`, `derivative`, `multi_output`, and `variational`, for
example:

```sh
fo exec fortml_bench_gp_features derivative 3
```

It prints one CSV row only after its direct oracle passes. Matched cross-engine
runs live in the sibling `fortml-bench` repository.

## User formulas and lazy CG

This postfix program defines
`variance * (1 + 3 r**2) * exp(-r)`. Validation is required before kernel
construction.

```fortran
use fortml_kernel_formula, only: kernel_formula_t
use fortml_kernel_operator, only: kernel_operator_t
use fortml_kernels, only: kernel_t, make_user_kernel

type(kernel_formula_t) :: formula
type(kernel_t) :: kernel
type(kernel_operator_t) :: operator

call formula%reset()
call formula%push_constant(1.0_dp)
call formula%push_squared_distance()
call formula%push_constant(3.0_dp)
call formula%multiply()
call formula%add()
call formula%push_distance()
call formula%negate()
call formula%exponential()
call formula%multiply()
call formula%validate(status)

kernel = make_user_kernel(size(points, 2), 1.3_dp, formula, status)
call operator%initialize(points, kernel, 0.05_dp, status)
solution = 0.0_dp
call operator%solve_cg(rhs, solution, 1.0e-10_dp, 500, &
    info, iterations, residual_norm)
```

Formula validation, composition, and direct pairwise checks are in
[test_kernel_formula.f90](../test/test_kernel_formula.f90). Host products,
multi-RHS solves, and block/Nystrom preconditioners are in
[test_kernel_operator.f90](../test/test_kernel_operator.f90).

## Resident device products

The operator owns reusable points, the lowered kernel program, and an optional
Krylov workspace. The caller owns the right-hand-side and output data region.

```fortran
call operator%enter_data(status, n_rhs=size(rhs, 2))
if (.not. status_ok(status)) error stop "device setup failed"

! Keep rhs and result present for all repeated calls in this region.
!$acc data copyin(rhs) create(result)
call operator%matmat_device(rhs, result, status)
!$acc update self(result)
!$acc end data

call operator%exit_data(status)
```

The complete OpenACC sequence is
[test_kernel_operator_device.f90](../test/test_kernel_operator_device.f90).
Structured and sparse resident paths use their own status types and are shown
in [test_structured_operator_device.f90](../test/test_structured_operator_device.f90)
and [test_sparse_operator_device.f90](../test/test_sparse_operator_device.f90).

## Operator examples

| Contract | Executable example |
| --- | --- |
| Sparse triplet products | [test_sparse_operator.f90](../test/test_sparse_operator.f90) |
| Tensor-product factors and derivative factors | [test_structured_multilevel.f90](../test/test_structured_multilevel.f90) |
| Toeplitz products and inherited CG | [test_toeplitz_operator.f90](../test/test_toeplitz_operator.f90) |
| Banded precision, solves, and determinants | [test_banded_precision.f90](../test/test_banded_precision.f90) |
| Lanczos log determinant and predictive variance | [test_lanczos.f90](../test/test_lanczos.f90) |
| Named parameter blocks | [test_parameter_registry.f90](../test/test_parameter_registry.f90) |
| Transform-aware hyperparameter vectors | [test_hyperparameter_registry.f90](../test/test_hyperparameter_registry.f90) |
