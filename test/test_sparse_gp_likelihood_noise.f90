program test_sparse_gp_likelihood_noise
    !! Independent finite-difference and adjoint oracles for sparse-GP noise.
    use, intrinsic :: ieee_arithmetic, only: ieee_quiet_nan, ieee_value
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    use fortml_kernels, only: kernel_t, make_rbf_kernel
    use fortml_sparse_gp, only: sparse_gp_t
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    implicit none

    integer, parameter :: n = 7, d = 1, m = 3
    real(dp) :: x(n, d), y(n), inducing(m, d), mean(m), factor(m, m)
    real(dp) :: theta(1), theta_plus(1), theta_minus(1), malformed(2)
    real(dp) :: value, value_plus, value_minus, tangent, h
    real(dp) :: vjp(1), hvp(1), gradient_plus(1), gradient_minus(1)
    type(sparse_gp_t) :: model
    type(kernel_t) :: kernel
    type(fortnum_status_t) :: status
    type(fortml_device_t) :: device
    integer :: failures, i

    do i = 1, n
        x(i, 1) = -1.1_dp + 0.31_dp*real(i - 1, dp)
        y(i) = sin(1.3_dp*x(i, 1)) + 0.08_dp*cos(real(i, dp))
    end do
    inducing(:, 1) = [-0.85_dp, -0.05_dp, 0.7_dp]
    mean = [0.21_dp, -0.14_dp, 0.09_dp]
    factor = 0.0_dp
    factor(1, 1) = 0.72_dp
    factor(2, 1) = -0.11_dp
    factor(2, 2) = 0.63_dp
    factor(3, 1) = 0.06_dp
    factor(3, 2) = 0.03_dp
    factor(3, 3) = 0.81_dp
    failures = 0
    h = 2.0e-6_dp

    kernel = make_rbf_kernel(d, 1.25_dp, 0.74_dp, status)
    call check(status_ok(status), "kernel constructor", failures)
    call model%initialize(inducing, kernel, 0.19_dp, status)
    call check(status_ok(status), "sparse initialization", failures)
    call model%set_variational(mean, factor, status)
    call check(status_ok(status), "variational state", failures)
    call check(model%likelihood_parameter_count() == 1, &
        "one transformed likelihood coordinate", failures)
    theta = model%likelihood_parameters()
    call check(abs(theta(1) - log(0.19_dp)) < 1.0e-14_dp, &
        "log-noise accessor", failures)

    call model%elbo_likelihood_parameter_jvp(x, y, [1.0_dp], value, tangent, status)
    call check(status_ok(status), "likelihood JVP status", failures)
    theta_plus = theta + h
    call model%set_likelihood_parameters(theta_plus, status)
    call model%elbo(x, y, value_plus, status)
    theta_minus = theta - h
    call model%set_likelihood_parameters(theta_minus, status)
    call model%elbo(x, y, value_minus, status)
    call model%set_likelihood_parameters(theta, status)
    call check(status_ok(status), "likelihood finite-difference restores state", failures)
    call check(abs(tangent - (value_plus - value_minus)/(2.0_dp*h)) < 3.0e-5_dp, &
        "likelihood JVP finite difference", failures)

    call model%elbo_likelihood_parameter_vjp(x, y, -1.7_dp, vjp, status)
    call check(status_ok(status), "likelihood VJP status", failures)
    call check(abs(vjp(1) + 1.7_dp*tangent) < 3.0e-12_dp, &
        "likelihood VJP/JVP adjoint identity", failures)

    call model%elbo_likelihood_parameter_hvp(x, y, [1.0_dp], hvp, status)
    call check(status_ok(status), "likelihood HVP status", failures)
    theta_plus = theta + h
    call model%set_likelihood_parameters(theta_plus, status)
    call model%elbo_likelihood_parameter_vjp(x, y, 1.0_dp, gradient_plus, status)
    theta_minus = theta - h
    call model%set_likelihood_parameters(theta_minus, status)
    call model%elbo_likelihood_parameter_vjp(x, y, 1.0_dp, gradient_minus, status)
    call model%set_likelihood_parameters(theta, status)
    call check(abs(hvp(1) - (gradient_plus(1) - gradient_minus(1))/(2.0_dp*h)) < 3.0e-5_dp, &
        "likelihood HVP finite difference", failures)

    ! Malformed and non-finite updates must not mutate the packed state.
    malformed = [theta(1), theta(1)]
    call model%set_likelihood_parameters(malformed, status)
    call check(.not. status_ok(status), "malformed likelihood update refused", failures)
    call check(maxval(abs(model%likelihood_parameters() - theta)) < 1.0e-14_dp, &
        "malformed update preserves noise", failures)
    call model%set_hyperparameters([ieee_value(0.0_dp, ieee_quiet_nan)], status)
    call check(.not. status_ok(status), "NaN likelihood update refused", failures)
    call check(maxval(abs(model%hyperparameters() - theta)) < 1.0e-14_dp, &
        "NaN update preserves noise", failures)
    call model%set_hyperparameters([huge(1.0_dp)], status)
    call check(.not. status_ok(status), "overflowing likelihood update refused", failures)
    call check(maxval(abs(model%hyperparameters() - theta)) < 1.0e-14_dp, &
        "overflow update preserves noise", failures)

    device%kind = FORTML_DEVICE_CPU
    device%selected = .true.
    device%available = .true.
    call model%elbo_likelihood_parameter_jvp_device(device, x, y, [1.0_dp], &
        value, tangent, status)
    call check(status_ok(status), "CPU likelihood JVP dispatch", failures)
    device%kind = FORTML_DEVICE_CUDA
    call model%elbo_likelihood_parameter_jvp_device(device, x, y, [1.0_dp], &
        value, tangent, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "CUDA likelihood JVP refusal", failures)
    call model%elbo_likelihood_parameter_vjp_device(device, x, y, 1.0_dp, vjp, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "CUDA likelihood VJP refusal", failures)
    call model%elbo_likelihood_parameter_hvp_device(device, x, y, [1.0_dp], hvp, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "CUDA likelihood HVP refusal", failures)

    if (failures /= 0) then
        write (error_unit, '(a,i0)') "FAIL sparse-GP likelihood-noise cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS sparse-GP likelihood-noise independent oracles"

contains

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "  FAIL [sparse-gp-noise] "//description
        end if
    end subroutine check

end program test_sparse_gp_likelihood_noise
