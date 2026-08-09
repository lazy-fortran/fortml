program test_gp_variational_categorical_classification
    !! Independent behavioral oracles for coupled categorical variational GPs.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_gp_variational_categorical_classification, only: &
        gp_variational_categorical_classification_t, &
        gp_variational_categorical_options_t, gp_variational_categorical_state_t
    use fortml_kernels, only: kernel_t, make_rbf_kernel
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    implicit none

    type(gp_variational_categorical_classification_t) :: model
    type(gp_variational_categorical_classification_t) :: fit_model
    type(gp_variational_categorical_options_t) :: options
    type(gp_variational_categorical_state_t) :: fit_state
    type(kernel_t) :: kernel
    type(fortnum_status_t) :: status
    type(fortml_device_t) :: device
    real(dp) :: x(6, 1), x_dot(6, 1), inducing(3, 1)
    real(dp) :: probabilities(6, 3), probabilities_dot(6, 3)
    real(dp) :: probabilities_plus(6, 3), probabilities_minus(6, 3)
    real(dp) :: log_probabilities(6, 3), log_probabilities_dot(6, 3)
    real(dp) :: log_probabilities_plus(6, 3), log_probabilities_minus(6, 3)
    real(dp) :: log_probabilities_bar(6, 3), log_parameter_bar(27), log_parameter_bar_fd(27)
    real(dp) :: log_x_bar(6, 1), log_x_bar_fd(6, 1)
    real(dp) :: probabilities_bar(6, 3), parameter_bar(27), parameter_bar_fd(27)
    real(dp) :: x_bar(6, 1), x_bar_fd(6, 1)
    real(dp) :: means(6, 3), variances(6, 3), expected(6, 3), logits(3), total
    real(dp) :: gradient(27), gradient_fd(27), direction(27)
    real(dp) :: value, value_plus, value_minus, tangent, h
    real(dp), allocatable :: lambda(:), shifted(:)
    integer :: labels(6), classes(3), predicted(6), failures, i, j

    x(:, 1) = [-1.2_dp, -0.7_dp, -0.15_dp, 0.25_dp, 0.8_dp, 1.3_dp]
    x_dot(:, 1) = [0.03_dp, -0.02_dp, 0.01_dp, 0.04_dp, -0.03_dp, 0.02_dp]
    inducing(:, 1) = [-0.9_dp, 0.0_dp, 0.85_dp]
    labels = [30, 10, 20, 10, 30, 20]
    classes = [30, 10, 20]
    failures = 0
    h = 2.0e-6_dp

    kernel = make_rbf_kernel(1, 1.2_dp, 0.8_dp, status)
    call check(status_ok(status), "kernel constructor", failures)
    call model%initialize(inducing, classes, kernel, 8, 20260808, status)
    call check(status_ok(status), "categorical initialization", failures)
    call check(all(model%classes() == [10, 20, 30]), "sorted labels", failures)
    call check(model%parameter_count() == 27, "packed state size", failures)

    call model%predict_latent(x, means, variances, status)
    call check(status_ok(status), "latent prediction", failures)
    call model%predict_proba(x, probabilities, status)
    call check(status_ok(status), "coupled probability prediction", failures)
    call check(maxval(abs(sum(probabilities, dim=2) - 1.0_dp)) < 2.0e-13_dp, &
        "probability simplex", failures)
    do i = 1, size(x, 1)
        do j = 1, 3
            logits(j) = means(i, j)/sqrt(1.0_dp + 3.141592653589793_dp*variances(i, j)/8.0_dp)
        end do
        total = sum(exp(logits - maxval(logits)))
        expected(i, :) = exp(logits - maxval(logits))/total
    end do
    call check(maxval(abs(probabilities - expected)) < 2.0e-13_dp, &
        "hand softmax likelihood oracle", failures)
    call model%predict_log_proba(x, log_probabilities, status)
    call check(status_ok(status), "categorical log-probability prediction", failures)
    call check(maxval(abs(log_probabilities - log(expected))) < 2.0e-13_dp, &
        "hand log-softmax likelihood oracle", failures)

    lambda = model%parameters()
    lambda = lambda + [ &
        0.03_dp, -0.02_dp, 0.01_dp, 0.01_dp, -0.02_dp, 0.02_dp, 0.01_dp, &
        -0.01_dp, 0.02_dp, -0.03_dp, 0.04_dp, -0.02_dp, 0.03_dp, -0.01_dp, &
        0.02_dp, -0.02_dp, 0.01_dp, 0.03_dp, -0.01_dp, 0.02_dp, 0.01_dp, &
        -0.03_dp, 0.02_dp, 0.01_dp, -0.02_dp, 0.03_dp, -0.01_dp]
    call model%set_parameters(lambda, status)
    call check(status_ok(status), "set packed state", failures)
    call model%elbo_gradient(x, labels, value, gradient, status)
    call check(status_ok(status), "categorical ELBO gradient", failures)
    allocate(shifted(size(lambda)))
    do i = 1, size(lambda)
        shifted = lambda
        shifted(i) = shifted(i) + h
        call model%set_parameters(shifted, status)
        call model%elbo(x, labels, value_plus, status)
        shifted(i) = lambda(i) - h
        call model%set_parameters(shifted, status)
        call model%elbo(x, labels, value_minus, status)
        gradient_fd(i) = (value_plus - value_minus)/(2.0_dp*h)
    end do
    call model%set_parameters(lambda, status)
    call check(maxval(abs(gradient - gradient_fd)) < 1.5e-4_dp, &
        "ELBO finite-difference gradient oracle", failures)

    direction = [ &
        0.02_dp, -0.01_dp, 0.03_dp, 0.01_dp, -0.02_dp, 0.02_dp, -0.01_dp, &
        0.03_dp, -0.02_dp, 0.01_dp, -0.03_dp, 0.02_dp, 0.01_dp, 0.02_dp, &
        -0.01_dp, 0.02_dp, -0.02_dp, 0.01_dp, 0.03_dp, -0.01_dp, 0.02_dp, &
        -0.03_dp, 0.01_dp, 0.02_dp, -0.02_dp, 0.01_dp, 0.03_dp]
    call model%elbo_jvp(x, labels, direction, value, tangent, status)
    call check(status_ok(status), "ELBO JVP", failures)
    call model%set_parameters(lambda + h*direction, status)
    call model%elbo(x, labels, value_plus, status)
    call model%set_parameters(lambda - h*direction, status)
    call model%elbo(x, labels, value_minus, status)
    call model%set_parameters(lambda, status)
    call check(abs(tangent - (value_plus - value_minus)/(2.0_dp*h)) < 1.5e-4_dp, &
        "ELBO JVP finite-difference oracle", failures)

    call model%predict_proba_parameter_jvp(x, direction, probabilities, &
        probabilities_dot, status)
    call check(status_ok(status), "parameter probability JVP", failures)
    call model%set_parameters(lambda + h*direction, status)
    call model%predict_proba(x, probabilities_plus, status)
    call model%set_parameters(lambda - h*direction, status)
    call model%predict_proba(x, probabilities_minus, status)
    call model%set_parameters(lambda, status)
    call check(maxval(abs(probabilities_dot - (probabilities_plus - probabilities_minus)/(2.0_dp*h))) < 1.5e-4_dp, &
        "parameter probability JVP finite-difference oracle", failures)

    call model%predict_log_proba_parameter_jvp(x, direction, log_probabilities, &
        log_probabilities_dot, status)
    call check(status_ok(status), "log-probability parameter JVP", failures)
    call model%set_parameters(lambda + h*direction, status)
    call model%predict_log_proba(x, log_probabilities_plus, status)
    call model%set_parameters(lambda - h*direction, status)
    call model%predict_log_proba(x, log_probabilities_minus, status)
    call model%set_parameters(lambda, status)
    call check(maxval(abs(log_probabilities_dot - &
        (log_probabilities_plus - log_probabilities_minus)/(2.0_dp*h))) < 2.0e-4_dp, &
        "parameter log-probability JVP finite-difference oracle", failures)

    probabilities_bar = reshape([ &
        0.17_dp, -0.09_dp, 0.04_dp, 0.12_dp, -0.06_dp, 0.08_dp, &
        -0.05_dp, 0.11_dp, -0.08_dp, 0.07_dp, 0.03_dp, -0.02_dp, &
        0.02_dp, 0.06_dp, 0.09_dp, -0.04_dp, 0.05_dp, 0.01_dp], [6, 3])
    call model%predict_proba_parameter_vjp(x, probabilities_bar, parameter_bar, status)
    call check(status_ok(status), "parameter probability VJP", failures)
    do i = 1, size(parameter_bar)
        shifted = lambda
        shifted(i) = shifted(i) + h
        call model%set_parameters(shifted, status)
        call model%predict_proba(x, probabilities_plus, status)
        shifted(i) = lambda(i) - h
        call model%set_parameters(shifted, status)
        call model%predict_proba(x, probabilities_minus, status)
        parameter_bar_fd(i) = sum(probabilities_bar*(probabilities_plus - probabilities_minus))/(2.0_dp*h)
    end do
    call model%set_parameters(lambda, status)
    call check(maxval(abs(parameter_bar - parameter_bar_fd)) < 2.0e-4_dp, &
        "parameter probability VJP finite-difference oracle", failures)
    call check(abs(dot_product(parameter_bar, direction) - sum(probabilities_dot*probabilities_bar)) < 2.0e-4_dp, &
        "parameter JVP/VJP duality", failures)

    log_probabilities_bar = reshape([ &
        -0.07_dp, 0.11_dp, -0.05_dp, 0.09_dp, -0.03_dp, 0.08_dp, &
        0.04_dp, -0.12_dp, 0.06_dp, -0.02_dp, 0.10_dp, -0.09_dp, &
        0.05_dp, 0.01_dp, -0.08_dp, 0.03_dp, -0.06_dp, 0.07_dp], [6, 3])
    call model%predict_log_proba_parameter_vjp(x, log_probabilities_bar, log_parameter_bar, status)
    call check(status_ok(status), "log-probability parameter VJP", failures)
    do i = 1, size(log_parameter_bar)
        shifted = lambda
        shifted(i) = shifted(i) + h
        call model%set_parameters(shifted, status)
        call model%predict_log_proba(x, log_probabilities_plus, status)
        shifted(i) = lambda(i) - h
        call model%set_parameters(shifted, status)
        call model%predict_log_proba(x, log_probabilities_minus, status)
        log_parameter_bar_fd(i) = sum(log_probabilities_bar* &
            (log_probabilities_plus - log_probabilities_minus))/(2.0_dp*h)
    end do
    call model%set_parameters(lambda, status)
    call check(maxval(abs(log_parameter_bar - log_parameter_bar_fd)) < 2.0e-4_dp, &
        "log-probability parameter VJP finite-difference oracle", failures)
    call check(abs(dot_product(log_parameter_bar, direction) - &
        sum(log_probabilities_dot*log_probabilities_bar)) < 2.0e-4_dp, &
        "log-probability parameter JVP/VJP duality", failures)

    call model%predict_proba_input_jvp(x, x_dot, probabilities, probabilities_dot, status)
    call check(status_ok(status), "input probability JVP", failures)
    call model%predict_proba(x + h*x_dot, probabilities_plus, status)
    call model%predict_proba(x - h*x_dot, probabilities_minus, status)
    call check(maxval(abs(probabilities_dot - (probabilities_plus - probabilities_minus)/(2.0_dp*h))) < 2.0e-4_dp, &
        "input probability JVP finite-difference oracle", failures)
    call model%predict_proba_input_vjp(x, probabilities_bar, x_bar, status)
    call check(status_ok(status), "input probability VJP", failures)
    do i = 1, size(x, 1)
        do j = 1, size(x, 2)
            call model%predict_proba(x + unit_perturb(i, j, h), probabilities_plus, status)
            call model%predict_proba(x - unit_perturb(i, j, h), probabilities_minus, status)
            x_bar_fd(i, j) = sum(probabilities_bar*(probabilities_plus - probabilities_minus))/(2.0_dp*h)
        end do
    end do
    call check(maxval(abs(x_bar - x_bar_fd)) < 2.0e-4_dp, &
        "input probability VJP finite-difference oracle", failures)

    call model%predict_log_proba_input_jvp(x, x_dot, log_probabilities, &
        log_probabilities_dot, status)
    call check(status_ok(status), "input log-probability JVP", failures)
    call model%predict_log_proba(x + h*x_dot, log_probabilities_plus, status)
    call model%predict_log_proba(x - h*x_dot, log_probabilities_minus, status)
    call check(maxval(abs(log_probabilities_dot - &
        (log_probabilities_plus - log_probabilities_minus)/(2.0_dp*h))) < 2.0e-4_dp, &
        "input log-probability JVP finite-difference oracle", failures)
    call model%predict_log_proba_input_vjp(x, log_probabilities_bar, log_x_bar, status)
    call check(status_ok(status), "input log-probability VJP", failures)
    do i = 1, size(x, 1)
        do j = 1, size(x, 2)
            call model%predict_log_proba(x + unit_perturb(i, j, h), log_probabilities_plus, status)
            call model%predict_log_proba(x - unit_perturb(i, j, h), log_probabilities_minus, status)
            log_x_bar_fd(i, j) = sum(log_probabilities_bar*( &
                log_probabilities_plus - log_probabilities_minus))/(2.0_dp*h)
        end do
    end do
    call check(maxval(abs(log_x_bar - log_x_bar_fd)) < 2.0e-4_dp, &
        "input log-probability VJP finite-difference oracle", failures)

    call model%predict(x, predicted, status)
    call check(status_ok(status), "label prediction", failures)
    do i = 1, size(predicted)
        call check(any(predicted(i) == [10, 20, 30]), "sorted label prediction", failures)
    end do

    device%kind = FORTML_DEVICE_CUDA
    device%selected = .true.
    device%available = .true.
    call model%predict_proba_device(device, x, probabilities_plus, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, "CUDA probability refusal", failures)
    call check(maxval(abs(probabilities_plus)) == 0.0_dp, "CUDA probability output cleared", failures)
    call model%predict_log_proba_device(device, x, log_probabilities_plus, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, "CUDA log-probability refusal", failures)
    call check(maxval(abs(log_probabilities_plus)) == 0.0_dp, &
        "CUDA log-probability output cleared", failures)
    call model%predict_proba_parameter_vjp_device(device, x, probabilities_bar, parameter_bar, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, "CUDA parameter refusal", failures)
    call check(maxval(abs(parameter_bar)) == 0.0_dp, "CUDA parameter output cleared", failures)
    call model%predict_log_proba_parameter_vjp_device(device, x, log_probabilities_bar, &
        log_parameter_bar, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, "CUDA log-parameter refusal", failures)
    call check(maxval(abs(log_parameter_bar)) == 0.0_dp, &
        "CUDA log-parameter output cleared", failures)
    call model%predict_proba_input_vjp_device(device, x, probabilities_bar, x_bar, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, "CUDA input refusal", failures)
    call check(maxval(abs(x_bar)) == 0.0_dp, "CUDA input output cleared", failures)
    call model%predict_log_proba_input_vjp_device(device, x, log_probabilities_bar, log_x_bar, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, "CUDA log-input refusal", failures)
    call check(maxval(abs(log_x_bar)) == 0.0_dp, "CUDA log-input output cleared", failures)
    call model%elbo_device(device, x, labels, value, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, "CUDA ELBO refusal", failures)

    options%max_iterations = 80
    options%gradient_tolerance = 5.0e-5_dp
    call fit_model%fit(x, labels, inducing, kernel, status, options, fit_state)
    call check(status_ok(status), "FortOpt categorical fit", failures)
    call check(fit_state%converged, "FortOpt categorical convergence", failures)
    call fit_model%predict_proba(x, probabilities_plus, status)
    call check(status_ok(status), "fitted categorical prediction", failures)

    if (failures > 0) then
        write (error_unit, '(a,i0)') &
            "FAIL categorical variational GP cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS categorical variational GP independent oracles"

contains

    function unit_perturb(row, column, step) result(delta)
        integer, intent(in) :: row, column
        real(dp), intent(in) :: step
        real(dp) :: delta(6, 1)

        delta = 0.0_dp
        delta(row, column) = step
    end function unit_perturb

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "  FAIL [categorical-gp] "//description
        end if
    end subroutine check

end program test_gp_variational_categorical_classification
