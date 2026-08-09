program test_gp_ordinal_log_proba
    !! Independent oracle for ordinal-GP log-probability products.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortml_kernels, only: kernel_t, make_rbf_kernel
    use fortml_gp_ordinal_classification, only: &
        gp_ordinal_classification_t, gp_ordinal_classification_options_t
    implicit none

    type(gp_ordinal_classification_t) :: model, model_plus, model_minus
    type(gp_ordinal_classification_options_t) :: options
    type(kernel_t) :: kernel
    type(fortml_device_t) :: cuda
    type(fortnum_status_t) :: status
    real(dp) :: x(9, 1), query(5, 1), query_plus(5, 1), query_minus(5, 1)
    real(dp) :: query_dot(5, 1)
    real(dp) :: probabilities(5, 3), log_probabilities(5, 3)
    real(dp) :: log_probabilities_dot(5, 3)
    real(dp) :: log_probabilities_plus(5, 3), log_probabilities_minus(5, 3)
    real(dp) :: log_probabilities_bar(5, 3), query_bar(5, 1)
    real(dp), allocatable :: parameters(:), direction(:), parameter_bar(:)
    real(dp), allocatable :: parameters_plus(:), parameters_minus(:)
    real(dp) :: h, lhs, rhs
    integer :: labels(9), failures

    x(:, 1) = [-1.5_dp, -1.1_dp, -0.8_dp, -0.2_dp, 0.0_dp, 0.3_dp, &
        0.8_dp, 1.1_dp, 1.5_dp]
    labels = [-4, -4, -4, 7, 7, 7, 19, 19, 19]
    query(:, 1) = [-1.25_dp, -0.5_dp, 0.0_dp, 0.6_dp, 1.3_dp]
    query_dot(:, 1) = [0.2_dp, -0.3_dp, 0.1_dp, 0.5_dp, -0.2_dp]
    failures = 0
    kernel = make_rbf_kernel(1, 1.2_dp, 0.7_dp, status)
    options%noise_variance = 0.04_dp
    options%jitter = 1.0e-8_dp
    call model%fit(x, labels, kernel, status, options)
    call check(status_ok(status), "ordinal GP fit", failures)

    call model%predict_proba(query, probabilities, status)
    call model%predict_log_proba(query, log_probabilities, status)
    call check(status_ok(status) .and. ieee_finite(log_probabilities) .and. &
        maxval(abs(exp(log_probabilities) - probabilities)) < 2.0e-14_dp .and. &
        maxval(abs(sum(exp(log_probabilities), dim=2) - 1.0_dp)) < 2.0e-14_dp, &
        "stable log-probability simplex", failures)

    allocate(parameters(model%parameter_count()), direction(model%parameter_count()), &
        parameter_bar(model%parameter_count()), parameters_plus(model%parameter_count()), &
        parameters_minus(model%parameter_count()))
    parameters = model%parameters()
    direction = [0.11_dp, -0.07_dp, 0.04_dp]
    call model%predict_log_proba_parameter_jvp(query, direction, log_probabilities, &
        log_probabilities_dot, status)
    call check(status_ok(status), "log-probability parameter JVP status", failures)
    call model_plus%fit(x, labels, kernel, status, options)
    call model_minus%fit(x, labels, kernel, status, options)
    h = 1.0e-5_dp
    parameters_plus = parameters + h*direction
    parameters_minus = parameters - h*direction
    call model_plus%set_parameters(parameters_plus, status)
    call model_minus%set_parameters(parameters_minus, status)
    call model_plus%predict_log_proba(query, log_probabilities_plus, status)
    call model_minus%predict_log_proba(query, log_probabilities_minus, status)
    call check(maxval(abs(log_probabilities_dot - &
        (log_probabilities_plus - log_probabilities_minus)/(2.0_dp*h))) < 4.0e-6_dp, &
        "log-probability parameter JVP finite difference", failures)

    log_probabilities_bar = reshape([ &
        0.2_dp, -0.1_dp, 0.4_dp, -0.3_dp, 0.1_dp, &
        -0.2_dp, 0.5_dp, -0.4_dp, 0.3_dp, -0.1_dp, &
        0.1_dp, -0.2_dp, 0.3_dp, 0.4_dp, -0.5_dp], shape(log_probabilities_bar))
    call model%predict_log_proba_parameter_vjp(query, log_probabilities_bar, &
        parameter_bar, status)
    lhs = dot_product(parameter_bar, direction)
    rhs = sum(log_probabilities_bar*log_probabilities_dot)
    call check(status_ok(status) .and. abs(lhs - rhs) < 4.0e-6_dp, &
        "log-probability parameter VJP duality", failures)

    call model%predict_log_proba_input_jvp(query, query_dot, log_probabilities, &
        log_probabilities_dot, status)
    query_plus = query + h*query_dot
    query_minus = query - h*query_dot
    call model%predict_log_proba(query_plus, log_probabilities_plus, status)
    call model%predict_log_proba(query_minus, log_probabilities_minus, status)
    call check(status_ok(status) .and. maxval(abs(log_probabilities_dot - &
        (log_probabilities_plus - log_probabilities_minus)/(2.0_dp*h))) < 2.0e-5_dp, &
        "log-probability input JVP finite difference", failures)
    call model%predict_log_proba_input_vjp(query, log_probabilities_bar, query_bar, status)
    lhs = sum(query_bar*query_dot)
    rhs = sum(log_probabilities_bar*log_probabilities_dot)
    call check(status_ok(status) .and. abs(lhs - rhs) < 2.0e-5_dp, &
        "log-probability input VJP duality", failures)

    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    log_probabilities = 123.0_dp
    call model%predict_log_proba_device(cuda, query, log_probabilities, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED .and. &
        all(log_probabilities == 0.0_dp), "typed CUDA log-probability refusal", failures)

    if (failures > 0) then
        write (error_unit, '(a,i0)') "FAIL ordinal GP log-probability cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS ordinal GP log-probability independent derivative oracle"

contains

    logical function ieee_finite(values) result(ok)
        use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
        real(dp), intent(in) :: values(:, :)

        ok = all(ieee_is_finite(values))
    end function ieee_finite

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "  FAIL [gp-ordinal-log-proba] "//description
        end if
    end subroutine check

end program test_gp_ordinal_log_proba
