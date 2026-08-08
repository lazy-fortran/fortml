program test_gp_classification_log_proba
    !! Independent oracle for the sklearn-style binary GP log-probability API.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortml_kernels, only: kernel_t, make_rbf_kernel
    use fortml_gp_classification, only: gp_classification_t, &
        gp_classification_options_t
    implicit none

    type(gp_classification_t) :: model, model_plus, model_minus
    type(gp_classification_options_t) :: options
    type(kernel_t) :: kernel
    type(fortml_device_t) :: cuda
    type(fortnum_status_t) :: status
    real(dp) :: x(8, 1), query(4, 1), query_plus(4, 1), query_minus(4, 1)
    real(dp) :: query_dot(4, 1), probabilities(4, 2), probabilities_dot(4, 2)
    real(dp) :: log_probabilities(4, 2), log_probabilities_dot(4, 2)
    real(dp) :: log_plus(4, 2), log_minus(4, 2), log_bar(4, 2), query_bar(4, 1)
    real(dp), allocatable :: parameters(:), direction(:), parameters_plus(:), parameters_minus(:)
    real(dp) :: h, lhs, rhs
    integer :: labels(8), failures

    x(:, 1) = [-1.4_dp, -1.0_dp, -0.7_dp, -0.2_dp, 0.3_dp, 0.7_dp, 1.1_dp, 1.5_dp]
    labels = [-3, -3, -3, -3, 8, 8, 8, 8]
    query(:, 1) = [-1.15_dp, -0.25_dp, 0.55_dp, 1.2_dp]
    query_dot(:, 1) = [0.15_dp, -0.2_dp, 0.3_dp, -0.1_dp]
    failures = 0
    kernel = make_rbf_kernel(1, 1.1_dp, 0.8_dp, status)
    options%max_iterations = 100
    options%tolerance = 1.0e-9_dp
    call model%fit(x, labels, kernel, status, options)
    call check(status_ok(status) .and. model%fitted(), "fit", failures)

    call model%predict_proba(query, probabilities, status)
    call model%predict_log_proba(query, log_probabilities, status)
    call check(status_ok(status) .and. maxval(abs(log_probabilities - &
        log(max(probabilities, tiny(1.0_dp))))) < 2.0e-14_dp, &
        "log probability value oracle", failures)
    call check(maxval(abs(exp(log_probabilities) - probabilities)) < 2.0e-14_dp, &
        "log probability round trip", failures)

    allocate(parameters(model%parameter_count()), direction(model%parameter_count()), &
        parameters_plus(model%parameter_count()), parameters_minus(model%parameter_count()))
    parameters = model%parameters()
    direction = [0.09_dp, -0.05_dp]
    call model%predict_log_proba_parameter_jvp(query, direction, log_probabilities, &
        log_probabilities_dot, status)
    h = 1.0e-5_dp
    parameters_plus = parameters + h*direction
    parameters_minus = parameters - h*direction
    call model_plus%fit(x, labels, kernel, status, options)
    call model_minus%fit(x, labels, kernel, status, options)
    call model_plus%set_parameters(parameters_plus, status)
    call model_minus%set_parameters(parameters_minus, status)
    call model_plus%predict_log_proba(query, log_plus, status)
    call model_minus%predict_log_proba(query, log_minus, status)
    call check(maxval(abs(log_probabilities_dot - (log_plus - log_minus)/(2.0_dp*h))) < 3.0e-6_dp, &
        "log probability parameter JVP central difference", failures)

    log_bar = reshape([0.2_dp, -0.1_dp, 0.4_dp, -0.3_dp, 0.1_dp, -0.2_dp, &
        0.5_dp, -0.4_dp], shape(log_bar))
    call model%predict_log_proba_parameter_vjp(query, log_bar, parameters_plus, status)
    lhs = dot_product(parameters_plus, direction)
    rhs = sum(log_bar*log_probabilities_dot)
    call check(status_ok(status) .and. abs(lhs - rhs) < 3.0e-6_dp, &
        "log probability parameter VJP duality", failures)

    call model%predict_log_proba_jvp(query, query_dot, log_probabilities, &
        log_probabilities_dot, status)
    query_plus = query + h*query_dot
    query_minus = query - h*query_dot
    call model%predict_log_proba(query_plus, log_plus, status)
    call model%predict_log_proba(query_minus, log_minus, status)
    call check(maxval(abs(log_probabilities_dot - (log_plus - log_minus)/(2.0_dp*h))) < 2.0e-5_dp, &
        "log probability input JVP central difference", failures)
    call model%predict_log_proba_vjp(query, log_bar, query_bar, status)
    lhs = sum(query_bar*query_dot)
    rhs = sum(log_bar*log_probabilities_dot)
    call check(status_ok(status) .and. abs(lhs - rhs) < 2.0e-5_dp, &
        "log probability input VJP duality", failures)

    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call model%predict_log_proba_device(cuda, query, log_probabilities, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, "typed CUDA log-probability refusal", failures)

    if (failures > 0) then
        write (error_unit, '(a,i0)') "FAIL GP log-probability cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS GP log-probability independent value/derivative oracle"

contains

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "  FAIL [gp-log-proba] "//description
        end if
    end subroutine check

end program test_gp_classification_log_proba
