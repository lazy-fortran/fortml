program test_multi_output_gp_hypergradients
    !! Independent finite-difference oracle for ICM GP likelihood products.
    !! The oracle perturbs the packed state through the public transactional
    !! setter; it never calls the production contraction helpers.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    use fortml_kernels, only: kernel_t, make_rbf_kernel
    use fortml_multi_output_gp, only: multi_output_gp_t
    implicit none

    integer, parameter :: n = 5, d = 1, p = 2, q = 7
    real(dp), parameter :: signal = 1.3_dp, lengthscale = 0.7_dp, noise = 0.08_dp
    real(dp) :: x(n, d), y(n, p), mean(n, p), mean_before(n, p)
    real(dp) :: parameters(q), plus(q), minus(q), gradient(q), gradient_plus(q), gradient_minus(q)
    real(dp) :: hvp(q), direction(q), parameter_bar(q), value_dot, value, value_plus, value_minus
    type(multi_output_gp_t) :: model
    type(kernel_t) :: kernel
    type(fortnum_status_t) :: status
    type(fortml_device_t) :: cpu, cuda
    integer :: failures, i
    real(dp) :: h

    do i = 1, n
        x(i, 1) = -0.8_dp + 0.35_dp*real(i - 1, dp)
        y(i, 1) = sin(1.1_dp*x(i, 1))
        y(i, 2) = cos(0.8_dp*x(i, 1)) - 0.2_dp
    end do
    failures = 0
    kernel = make_rbf_kernel(d, signal, lengthscale, status)
    call model%initialize(kernel, reshape([0.75_dp, -0.35_dp], [p, 1]), &
        [0.25_dp, 0.4_dp], noise, status)
    call check(status_ok(status), "initialize", failures)
    call model%fit(x, y, status)
    call check(status_ok(status), "fit", failures)
    call check(model%sample_count() == n .and. model%feature_count() == d .and. &
        model%output_count() == p, "shape metadata", failures)

    parameters = model%parameters()
    direction = [0.07_dp, -0.11_dp, 0.13_dp, 0.05_dp, -0.04_dp, 0.06_dp, -0.03_dp]
    call model%hyperparameter_gradient(gradient, status)
    call check(status_ok(status), "likelihood gradient", failures)
    call model%log_marginal_likelihood_jvp(direction, value_dot, status)
    call check(status_ok(status) .and. abs(value_dot - dot_product(gradient, direction)) < 1.0e-12_dp, &
        "likelihood JVP composition", failures)
    call model%log_marginal_likelihood_vjp(1.0_dp, parameter_bar, status)
    call check(status_ok(status) .and. maxval(abs(parameter_bar - gradient)) < 1.0e-12_dp, &
        "likelihood VJP composition", failures)
    call model%log_marginal_likelihood(y, value, status)
    call check(status_ok(status), "likelihood value", failures)
    h = 2.0e-5_dp
    do i = 1, q
        plus = parameters
        minus = parameters
        plus(i) = plus(i) + h
        minus(i) = minus(i) - h
        call model%set_parameters(plus, status)
        call check(status_ok(status), "positive parameter perturbation", failures)
        call model%log_marginal_likelihood(y, value_plus, status)
        call check(status_ok(status), "positive likelihood", failures)
        call model%set_parameters(minus, status)
        call check(status_ok(status), "negative parameter perturbation", failures)
        call model%log_marginal_likelihood(y, value_minus, status)
        call check(status_ok(status), "negative likelihood", failures)
        call check(abs(gradient(i) - (value_plus - value_minus)/(2.0_dp*h)) < 2.0e-6_dp, &
            "gradient finite-difference oracle", failures)
    end do
    call model%set_parameters(parameters, status)
    call check(status_ok(status), "gradient state restore", failures)

    call model%hyperparameter_hvp(direction, hvp, status)
    call check(status_ok(status), "likelihood HVP", failures)
    call model%set_parameters(parameters + h*direction, status)
    call check(status_ok(status), "HVP positive direction", failures)
    call model%hyperparameter_gradient(gradient_plus, status)
    call check(status_ok(status), "positive gradient", failures)
    call model%set_parameters(parameters - h*direction, status)
    call check(status_ok(status), "HVP negative direction", failures)
    call model%hyperparameter_gradient(gradient_minus, status)
    call check(status_ok(status), "negative gradient", failures)
    if (maxval(abs(hvp - (gradient_plus - gradient_minus)/(2.0_dp*h))) >= 2.0e-5_dp) then
        write (error_unit, '(a,es12.4)') "HVP max error ", &
            maxval(abs(hvp - (gradient_plus - gradient_minus)/(2.0_dp*h)))
        write (error_unit, '(a,7es12.4)') "analytic ", hvp
        write (error_unit, '(a,7es12.4)') "finite-difference ", &
            (gradient_plus - gradient_minus)/(2.0_dp*h)
        failures = failures + 1
    end if
    call model%set_parameters(parameters, status)
    call check(status_ok(status), "HVP state restore", failures)

    call model%predict(x, mean_before, status)
    plus = parameters
    plus(q) = -1.0_dp
    call model%set_parameters(plus, status)
    call check(.not. status_ok(status), "invalid independent variance refused", failures)
    call check(maxval(abs(model%parameters() - parameters)) == 0.0_dp, &
        "failed parameter update rolled back", failures)
    call model%predict(x, mean, status)
    call check(maxval(abs(mean - mean_before)) == 0.0_dp, &
        "failed update preserved posterior", failures)

    cpu%kind = FORTML_DEVICE_CPU
    cpu%selected = .true.
    cpu%available = .true.
    call model%hyperparameter_gradient_device(cpu, gradient, status)
    call check(status_ok(status), "CPU gradient dispatch", failures)
    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    gradient = 8.0_dp
    call model%hyperparameter_gradient_device(cuda, gradient, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, "CUDA gradient refusal", failures)
    call check(maxval(abs(gradient)) == 0.0_dp, "CUDA gradient clears output", failures)
    hvp = 8.0_dp
    call model%hyperparameter_hvp_device(cuda, direction, hvp, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, "CUDA HVP refusal", failures)
    call check(maxval(abs(hvp)) == 0.0_dp, "CUDA HVP clears output", failures)

    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, " multi-output GP hypergradient test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS multi-output GP hypergradient independent oracle"

contains

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: description
        integer, intent(inout) :: failures

        if (condition) return
        write (error_unit, '(a)') "FAIL: "//description
        failures = failures + 1
    end subroutine check

end program test_multi_output_gp_hypergradients
