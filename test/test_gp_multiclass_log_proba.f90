program test_gp_multiclass_log_proba
    !! Independent oracle for the multiclass Laplace-GP log-probability API.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    use fortml_kernels, only: kernel_t, make_rbf_kernel
    use fortml_gp_multiclass_classification, only: &
        gp_multiclass_classification_t, gp_multiclass_classification_options_t
    implicit none

    type(gp_multiclass_classification_t) :: model
    type(gp_multiclass_classification_options_t) :: options
    type(kernel_t) :: kernel
    type(fortml_device_t) :: cpu, cuda
    type(fortnum_status_t) :: status
    real(dp) :: x(9, 2), query(5, 2), query_plus(5, 2), query_minus(5, 2)
    real(dp) :: query_dot(5, 2), probabilities(5, 3), probabilities_dot(5, 3)
    real(dp) :: log_probabilities(5, 3), log_probabilities_dot(5, 3)
    real(dp) :: log_plus(5, 3), log_minus(5, 3), log_bar(5, 3), query_bar(5, 2)
    real(dp) :: cpu_log_probabilities(5, 3)
    real(dp), allocatable :: direction(:), parameter_bar(:)
    real(dp) :: lhs, rhs, h
    integer :: labels(9), failures, k

    x = reshape([ &
        -1.0_dp, 1.0_dp, -0.8_dp, 1.1_dp, -1.1_dp, 0.9_dp, &
        0.0_dp, 0.0_dp, 0.1_dp, 0.1_dp, -0.1_dp, 0.0_dp, &
        1.0_dp, 1.0_dp, 0.9_dp, 1.1_dp, 1.1_dp, 0.9_dp], shape(x))
    labels = [42, 42, 42, -7, -7, -7, 10, 10, 10]
    query = reshape([ &
        -0.9_dp, 1.0_dp, 0.0_dp, 0.0_dp, 1.0_dp, 1.0_dp, &
        0.2_dp, 0.4_dp, 0.5_dp, 0.5_dp], shape(query))
    query_dot = reshape([ &
        0.15_dp, -0.10_dp, -0.20_dp, 0.25_dp, 0.30_dp, 0.10_dp, &
        -0.10_dp, 0.20_dp, 0.05_dp, -0.15_dp], shape(query_dot))
    log_bar = reshape([ &
        0.2_dp, -0.1_dp, 0.4_dp, -0.3_dp, 0.1_dp, -0.2_dp, &
        0.5_dp, -0.4_dp, 0.3_dp, 0.6_dp, -0.2_dp, 0.1_dp, &
        -0.5_dp, 0.4_dp, 0.2_dp], shape(log_bar))
    failures = 0
    kernel = make_rbf_kernel(2, 1.0_dp, 0.8_dp, status)
    options%max_iterations = 100
    options%tolerance = 1.0e-9_dp
    options%jitter = 1.0e-7_dp
    call model%fit(x, labels, kernel, status, options)
    call check(status_ok(status) .and. model%fitted(), "multiclass GP fit", failures)

    call model%predict_proba(query, probabilities, status)
    call model%predict_log_proba(query, log_probabilities, status)
    call check(status_ok(status) .and. maxval(abs(log_probabilities - &
        log(max(probabilities, tiny(1.0_dp))))) < 2.0e-14_dp, &
        "log probability value oracle", failures)
    call check(maxval(abs(exp(log_probabilities) - probabilities)) < 2.0e-14_dp, &
        "log probability round trip", failures)

    h = 1.0e-5_dp
    call model%predict_log_proba_jvp(query, query_dot, log_probabilities, &
        log_probabilities_dot, status)
    query_plus = query + h*query_dot
    query_minus = query - h*query_dot
    call model%predict_log_proba(query_plus, log_plus, status)
    call model%predict_log_proba(query_minus, log_minus, status)
    call check(status_ok(status) .and. maxval(abs(log_probabilities_dot - &
        (log_plus - log_minus)/(2.0_dp*h))) < 3.0e-5_dp, &
        "log probability input JVP central difference", failures)
    call model%predict_log_proba_vjp(query, log_bar, query_bar, status)
    lhs = sum(query_bar*query_dot)
    rhs = sum(log_bar*log_probabilities_dot)
    call check(status_ok(status) .and. abs(lhs - rhs) < 4.0e-5_dp, &
        "log probability input VJP duality", failures)

    allocate(direction(model%parameter_count()), parameter_bar(model%parameter_count()))
    do k = 1, size(direction)
        direction(k) = 0.04_dp*real(mod(k, 3) - 1, dp)
        if (direction(k) == 0.0_dp) direction(k) = -0.017_dp
    end do
    call model%predict_log_proba_parameter_jvp(query, direction, log_probabilities, &
        log_probabilities_dot, status)
    call model%predict_log_proba_parameter_vjp(query, log_bar, parameter_bar, status)
    lhs = dot_product(parameter_bar, direction)
    rhs = sum(log_bar*log_probabilities_dot)
    call check(status_ok(status) .and. abs(lhs - rhs) < 8.0e-7_dp, &
        "log probability parameter JVP/VJP duality", failures)

    call model%predict_log_proba(query(:4, :), log_plus, status)
    call check(.not. status_ok(status), "log probability shape refusal", failures)

    call cpu%select(FORTML_DEVICE_CPU, status)
    call model%predict_log_proba_device(cpu, query, cpu_log_probabilities, status)
    call check(status_ok(status) .and. maxval(abs(cpu_log_probabilities - &
        log_probabilities)) < 2.0e-14_dp, "CPU log probability dispatch", failures)
    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call model%predict_log_proba_device(cuda, query, cpu_log_probabilities, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "typed CUDA log probability refusal", failures)

    if (failures > 0) then
        write (error_unit, '(a,i0)') "FAIL GP multiclass log-probability cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS GP multiclass log-probability independent oracle"

contains

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "  FAIL [gp-multiclass-log-proba] "//description
        end if
    end subroutine check

end program test_gp_multiclass_log_proba
