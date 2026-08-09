program test_gp_variational_multiclass_log_proba
    !! Independent finite-difference and adjoint oracles for stable OVR
    !! variational-GP log probabilities and their fixed-state products.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_gp_variational_multiclass_classification, only: &
        gp_variational_multiclass_classification_t
    use fortml_gp_variational_classification, only: GP_VARIATIONAL_PROBIT
    use fortml_kernels, only: kernel_t, make_rbf_kernel
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    implicit none

    type(gp_variational_multiclass_classification_t) :: model, probit_model
    type(kernel_t) :: kernel
    type(fortnum_status_t) :: status
    type(fortml_device_t) :: device
    real(dp) :: x(6, 1), x_dot(6, 1), inducing(2, 1), h
    real(dp) :: log_probabilities(6, 3), log_probabilities_dot(6, 3)
    real(dp) :: log_plus(6, 3), log_minus(6, 3), log_bar(6, 3)
    real(dp) :: parameter_bar(15), parameter_bar_fd(15), direction(15)
    real(dp) :: input_bar(6, 1), plus, minus, objective_plus, objective_minus
    real(dp) :: log_probit(6, 3), extreme_log(6, 3)
    real(dp), allocatable :: lambda(:), shifted(:)
    integer :: classes(3), failures, i

    x(:, 1) = [-1.1_dp, -0.55_dp, -0.05_dp, 0.30_dp, 0.80_dp, 1.25_dp]
    x_dot(:, 1) = [0.03_dp, -0.02_dp, 0.01_dp, 0.04_dp, -0.03_dp, 0.02_dp]
    inducing(:, 1) = [-0.8_dp, 0.75_dp]
    classes = [30, 10, 20]
    log_bar(:, 1) = [0.17_dp, -0.09_dp, 0.04_dp, 0.12_dp, -0.06_dp, 0.08_dp]
    log_bar(:, 2) = [-0.05_dp, 0.11_dp, -0.08_dp, 0.07_dp, 0.03_dp, -0.02_dp]
    log_bar(:, 3) = [0.02_dp, 0.06_dp, 0.09_dp, -0.04_dp, 0.05_dp, 0.01_dp]
    direction = [0.04_dp, -0.03_dp, 0.02_dp, 0.01_dp, -0.02_dp, &
        -0.02_dp, 0.05_dp, -0.01_dp, 0.03_dp, 0.02_dp, &
        0.01_dp, 0.02_dp, -0.04_dp, 0.01_dp, 0.03_dp]
    failures = 0
    h = 2.0e-6_dp

    kernel = make_rbf_kernel(1, 1.3_dp, 0.9_dp, status)
    call check(status_ok(status), "kernel constructor", failures)
    call model%initialize(inducing, classes, kernel, 16, 20260809, status)
    call check(status_ok(status), "logistic model initialization", failures)
    call model%predict_log_proba(x, log_probabilities, status)
    call check(status_ok(status), "stable logistic log probabilities", failures)
    call check(ieee_finite_matrix(log_probabilities), &
        "logistic log probabilities are finite", failures)
    call check(maxval(abs(sum(exp(log_probabilities), dim=2) - 1.0_dp)) < 2.0e-13_dp, &
        "logistic log probability simplex", failures)

    lambda = model%parameters()
    call model%predict_log_proba_parameter_jvp(x, direction, log_probabilities, &
        log_probabilities_dot, status)
    call check(status_ok(status), "packed log probability JVP", failures)
    allocate(shifted(size(lambda)))
    shifted = lambda + h*direction
    call model%set_parameters(shifted, status)
    call model%predict_log_proba(x, log_plus, status)
    shifted = lambda - h*direction
    call model%set_parameters(shifted, status)
    call model%predict_log_proba(x, log_minus, status)
    call model%set_parameters(lambda, status)
    call check(maxval(abs(log_probabilities_dot - (log_plus - log_minus)/(2.0_dp*h))) < 6.0e-5_dp, &
        "packed log probability JVP finite-difference oracle", failures)

    call model%predict_log_proba_parameter_vjp(x, log_bar, parameter_bar, status)
    call check(status_ok(status), "packed log probability VJP", failures)
    do i = 1, size(parameter_bar)
        shifted = lambda
        shifted(i) = shifted(i) + h
        call model%set_parameters(shifted, status)
        call model%predict_log_proba(x, log_plus, status)
        objective_plus = sum(log_plus*log_bar)
        shifted(i) = lambda(i) - h
        call model%set_parameters(shifted, status)
        call model%predict_log_proba(x, log_minus, status)
        objective_minus = sum(log_minus*log_bar)
        parameter_bar_fd(i) = (objective_plus - objective_minus)/(2.0_dp*h)
    end do
    call model%set_parameters(lambda, status)
    call check(maxval(abs(parameter_bar - parameter_bar_fd)) < 8.0e-5_dp, &
        "packed log probability VJP finite-difference oracle", failures)
    call check(abs(dot_product(parameter_bar, direction) - sum(log_probabilities_dot*log_bar)) &
        < 8.0e-5_dp, "packed log probability JVP/VJP identity", failures)

    call model%predict_log_proba_input_jvp(x, x_dot, log_probabilities, &
        log_probabilities_dot, status)
    call check(status_ok(status), "fixed-state input log probability JVP", failures)
    call model%predict_log_proba(x + h*x_dot, log_plus, status)
    call model%predict_log_proba(x - h*x_dot, log_minus, status)
    call check(maxval(abs(log_probabilities_dot - (log_plus - log_minus)/(2.0_dp*h))) < 8.0e-5_dp, &
        "input log probability JVP finite-difference oracle", failures)
    call model%predict_log_proba_input_vjp(x, log_bar, input_bar, status)
    call check(status_ok(status), "fixed-state input log probability VJP", failures)
    call check(abs(sum(input_bar*x_dot) - sum(log_probabilities_dot*log_bar)) < 8.0e-5_dp, &
        "input log probability JVP/VJP identity", failures)

    device%kind = FORTML_DEVICE_CPU
    device%selected = .true.
    device%available = .true.
    call model%predict_log_proba_device(device, x, log_plus, status)
    call check(status_ok(status), "CPU log probability dispatch", failures)
    call model%predict_log_proba_parameter_vjp_device(device, x, log_bar, parameter_bar, status)
    call check(status_ok(status), "CPU log probability parameter VJP dispatch", failures)
    call model%predict_log_proba_input_vjp_device(device, x, log_bar, input_bar, status)
    call check(status_ok(status), "CPU log probability input VJP dispatch", failures)
    device%kind = FORTML_DEVICE_CUDA
    call model%predict_log_proba_device(device, x, log_plus, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, "CUDA log probability refusal", failures)
    call model%predict_log_proba_parameter_vjp_device(device, x, log_bar, parameter_bar, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "CUDA log probability parameter VJP refusal", failures)
    call model%predict_log_proba_input_vjp_device(device, x, log_bar, input_bar, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "CUDA log probability input VJP refusal", failures)

    ! Large negative means exercise log-space normalization without forming
    ! underflowed positive probabilities first.
    lambda = model%parameters()
    lambda(1:2) = -1000.0_dp
    lambda(6:7) = -1000.0_dp
    lambda(11:12) = -1000.0_dp
    call model%set_parameters(lambda, status)
    call model%predict_log_proba(x, extreme_log, status)
    call check(status_ok(status), "extreme logistic log probability", failures)
    call check(ieee_finite_matrix(extreme_log), &
        "extreme logistic log probabilities remain finite", failures)
    call check(maxval(abs(sum(exp(extreme_log), dim=2) - 1.0_dp)) < 2.0e-13_dp, &
        "extreme logistic log probability simplex", failures)

    call probit_model%initialize(inducing, classes, kernel, 16, 20260811, status, &
        likelihood=GP_VARIATIONAL_PROBIT)
    call check(status_ok(status), "probit model initialization", failures)
    call probit_model%predict_log_proba(x, log_probit, status)
    call check(status_ok(status), "stable probit log probabilities", failures)
    call check(ieee_finite_matrix(log_probit), &
        "probit log probabilities are finite", failures)
    call check(maxval(abs(sum(exp(log_probit), dim=2) - 1.0_dp)) < 2.0e-13_dp, &
        "probit log probability simplex", failures)

    if (failures > 0) then
        write (error_unit, '(a,i0)') "FAIL variational GP multiclass log-probability cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS variational GP multiclass log-probability independent oracles"

contains

    logical function ieee_finite_matrix(values)
        real(dp), intent(in) :: values(:, :)
        integer :: i, j

        ieee_finite_matrix = .true.
        do j = 1, size(values, 2)
            do i = 1, size(values, 1)
                if (.not. ieee_is_finite_value(values(i, j))) ieee_finite_matrix = .false.
            end do
        end do
    end function ieee_finite_matrix

    logical function ieee_is_finite_value(value)
        real(dp), intent(in) :: value

        ieee_is_finite_value = (value == value) .and. abs(value) < huge(1.0_dp)
    end function ieee_is_finite_value

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "  FAIL [variational-gp-multiclass-log-proba] "//description
        end if
    end subroutine check

end program test_gp_variational_multiclass_log_proba
