program test_gp_multilabel_classification
    !! Independent behavioral and derivative oracles for multilabel Laplace GPs.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED, &
        FORTNUM_DOMAIN_ERROR
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    use fortml_kernels, only: kernel_t, make_rbf_kernel
    use fortml_gp_multilabel_classification, only: &
        gp_multilabel_classification_t, gp_multilabel_classification_options_t, &
        gp_multilabel_classification_state_t
    implicit none

    type(gp_multilabel_classification_t) :: model, probit_model
    type(gp_multilabel_classification_options_t) :: options
    type(gp_multilabel_classification_state_t) :: state
    type(kernel_t) :: kernel
    type(fortml_device_t) :: cpu, cuda
    type(fortnum_status_t) :: status
    real(dp) :: x(10, 1), query(5, 1), query_dot(5, 1)
    real(dp) :: probabilities(5, 2), probabilities_dot(5, 2)
    real(dp) :: probabilities_plus(5, 2), probabilities_minus(5, 2)
    real(dp) :: probabilities_bar(5, 2), x_bar(5, 1)
    real(dp) :: log_probabilities(5, 2), log_probabilities_dot(5, 2)
    real(dp) :: log_probabilities_plus(5, 2), log_probabilities_minus(5, 2)
    real(dp) :: log_probabilities_bar(5, 2)
    real(dp) :: latent(5, 2), latent_dot(5, 2), variance(5, 2), variance_dot(5, 2)
    real(dp), allocatable :: direction(:), parameter_bar(:), parameters(:)
    real(dp), allocatable :: shared_direction(:), shared_parameter_bar(:)
    real(dp), allocatable :: thresholds(:)
    integer :: indicators(10, 2), predicted(5, 2), failures, i
    real(dp) :: h, lhs, rhs

    x(:, 1) = [-2.0_dp, -1.5_dp, -1.0_dp, -0.5_dp, -0.1_dp, &
        0.1_dp, 0.5_dp, 1.0_dp, 1.5_dp, 2.0_dp]
    indicators(:, 1) = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    indicators(:, 2) = [1, 1, 1, 0, 0, 0, 0, 1, 1, 1]
    query(:, 1) = [-1.7_dp, -0.4_dp, 0.0_dp, 0.7_dp, 1.7_dp]
    query_dot(:, 1) = [0.2_dp, -0.3_dp, 0.1_dp, 0.4_dp, -0.2_dp]
    failures = 0

    kernel = make_rbf_kernel(1, 1.3_dp, 0.75_dp, status)
    options%max_iterations = 100
    options%tolerance = 1.0e-9_dp
    options%jitter = 1.0e-7_dp
    call model%fit(x, indicators, kernel, status, options, state, &
        sample_weight=[1.0_dp, 0.9_dp, 1.1_dp, 1.0_dp, 0.8_dp, 1.2_dp, &
        1.0_dp, 1.1_dp, 0.9_dp, 1.0_dp], thresholds=[0.5_dp, 0.6_dp])
    call check(status_ok(status) .and. state%converged .and. model%fitted(), &
        "multilabel GP fit", failures)
    call check(model%label_count() == 2 .and. model%feature_count() == 1, &
        "multilabel metadata", failures)
    thresholds = model%thresholds()
    call check(all(abs(thresholds - [0.5_dp, 0.6_dp]) < 1.0e-14_dp), &
        "per-label threshold metadata", failures)
    call model%set_thresholds([0.0_dp, 0.6_dp], status)
    call check(status%code == FORTNUM_DOMAIN_ERROR .and. &
        all(abs(model%thresholds() - thresholds) < 1.0e-14_dp), &
        "invalid threshold metadata is rejected transactionally", failures)

    call model%predict_proba(query, probabilities, status)
    call model%predict(query, predicted, status)
    call check(status_ok(status) .and. all(probabilities > 0.0_dp) .and. &
        all(probabilities < 1.0_dp), "independent positive probabilities", failures)
    call model%predict_log_proba(query, log_probabilities, status)
    call check(status_ok(status) .and. &
        maxval(abs(log_probabilities - log(probabilities))) < 2.0e-14_dp, &
        "finite log-probability values", failures)
    call check(all((predicted == 0) .or. (predicted == 1)), &
        "indicator prediction contract", failures)
    call model%predict_latent(query, latent, variance, status)
    call check(status_ok(status) .and. all(variance >= 0.0_dp), &
        "latent and uncertainty prediction", failures)

    h = 1.0e-5_dp
    call model%predict_proba_jvp(query, query_dot, probabilities, probabilities_dot, status)
    call model%predict_proba(query + h*query_dot, probabilities_plus, status)
    call model%predict_proba(query - h*query_dot, probabilities_minus, status)
    call check(status_ok(status) .and. maxval(abs(probabilities_dot - &
        (probabilities_plus - probabilities_minus)/(2.0_dp*h))) < 3.0e-6_dp, &
        "probability input JVP finite difference", failures)
    probabilities_bar = reshape([0.2_dp, -0.4_dp, 0.7_dp, -0.1_dp, 0.5_dp, &
        -0.3_dp, -0.6_dp, 0.1_dp, 0.2_dp, 0.8_dp], shape(probabilities_bar))
    call model%predict_proba_vjp(query, probabilities_bar, x_bar, status)
    lhs = sum(x_bar*query_dot)
    rhs = sum(probabilities_bar*probabilities_dot)
    call check(status_ok(status) .and. abs(lhs - rhs) < 4.0e-6_dp, &
        "probability input VJP dot-product identity", failures)

    call model%predict_log_proba_jvp(query, query_dot, log_probabilities, &
        log_probabilities_dot, status)
    call model%predict_log_proba(query + h*query_dot, log_probabilities_plus, status)
    call model%predict_log_proba(query - h*query_dot, log_probabilities_minus, status)
    call check(status_ok(status) .and. maxval(abs(log_probabilities_dot - &
        (log_probabilities_plus - log_probabilities_minus)/(2.0_dp*h))) < 4.0e-6_dp, &
        "log-probability input JVP finite difference", failures)
    log_probabilities_bar = reshape([0.2_dp, -0.4_dp, 0.7_dp, -0.1_dp, 0.5_dp, &
        -0.3_dp, -0.6_dp, 0.1_dp, 0.2_dp, 0.8_dp], shape(log_probabilities_bar))
    call model%predict_log_proba_vjp(query, log_probabilities_bar, x_bar, status)
    lhs = sum(x_bar*query_dot)
    rhs = sum(log_probabilities_bar*log_probabilities_dot)
    call check(status_ok(status) .and. abs(lhs - rhs) < 5.0e-6_dp, &
        "log-probability input VJP dot-product identity", failures)

    allocate(direction(model%parameter_count()), parameter_bar(model%parameter_count()))
    direction = [0.13_dp, -0.09_dp, 0.07_dp, -0.05_dp]
    call model%predict_latent_parameter_jvp(query, direction, latent, latent_dot, &
        variance, variance_dot, status)
    call model%predict_latent_parameter_vjp(query, latent, variance, parameter_bar, status)
    call check(status_ok(status) .and. abs(dot_product(parameter_bar, direction) - &
        (sum(latent*latent_dot) + sum(variance*variance_dot))) < 4.0e-6_dp, &
        "latent packed-parameter JVP/VJP duality", failures)
    call model%predict_proba_parameter_jvp(query, direction, probabilities, &
        probabilities_dot, status)
    call model%predict_proba_parameter_vjp(query, probabilities_bar, parameter_bar, status)
    call check(status_ok(status) .and. abs(dot_product(parameter_bar, direction) - &
        sum(probabilities_bar*probabilities_dot)) < 4.0e-6_dp, &
        "probability packed-parameter JVP/VJP duality", failures)
    call model%predict_log_proba_parameter_jvp(query, direction, log_probabilities, &
        log_probabilities_dot, status)
    call model%predict_log_proba_parameter_vjp(query, log_probabilities_bar, parameter_bar, status)
    call check(status_ok(status) .and. abs(dot_product(parameter_bar, direction) - &
        sum(log_probabilities_bar*log_probabilities_dot)) < 5.0e-6_dp, &
        "log-probability packed-parameter JVP/VJP duality", failures)
    parameters = model%parameters()
    call check(size(parameters) == model%parameter_count() .and. &
        all(ieee_is_finite(parameters)), "packed parameter metadata", failures)
    call model%hyperparameter_gradient(parameter_bar, status)
    call check(status_ok(status) .and. all(ieee_is_finite(parameter_bar)), &
        "sum-of-head hyperparameter gradient", failures)

    allocate(shared_direction(model%shared_parameter_count()), &
        shared_parameter_bar(model%shared_parameter_count()))
    shared_direction = [0.11_dp, -0.08_dp]
    call model%predict_log_proba_shared_parameter_jvp(query, shared_direction, &
        log_probabilities, log_probabilities_dot, status)
    parameters = model%shared_parameters()
    call model%set_shared_parameters(parameters + h*shared_direction, status)
    call model%predict_log_proba(query, log_probabilities_plus, status)
    call model%set_shared_parameters(parameters - h*shared_direction, status)
    call model%predict_log_proba(query, log_probabilities_minus, status)
    call model%set_shared_parameters(parameters, status)
    call check(status_ok(status) .and. maxval(abs(log_probabilities_dot - &
        (log_probabilities_plus - log_probabilities_minus)/(2.0_dp*h))) < 5.0e-6_dp, &
        "log-probability shared-kernel JVP finite difference", failures)
    call model%predict_log_proba_shared_parameter_vjp(query, log_probabilities_bar, &
        shared_parameter_bar, status)
    lhs = sum(shared_parameter_bar*shared_direction)
    rhs = sum(log_probabilities_bar*log_probabilities_dot)
    call check(status_ok(status) .and. abs(lhs - rhs) < 6.0e-6_dp, &
        "log-probability shared-kernel VJP dot-product identity", failures)

    options%likelihood = 2
    call probit_model%fit(x, indicators, kernel, status, options, state)
    call probit_model%predict_proba(query, probabilities, status)
    call check(status_ok(status) .and. all(probabilities > 0.0_dp) .and. &
        all(probabilities < 1.0_dp), "probit multilabel branch", failures)

    call cpu%select(FORTML_DEVICE_CPU, status)
    call model%predict_proba_device(cpu, query, probabilities, status)
    call check(status_ok(status), "CPU probability dispatch", failures)
    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call model%predict_proba_device(cuda, query, probabilities, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "typed CUDA probability refusal", failures)
    call model%predict_latent_device(cuda, query, latent, variance, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "typed CUDA latent refusal", failures)
    call model%predict_device(cuda, query, predicted, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "typed CUDA label refusal", failures)
    log_probabilities = 7.0_dp
    call model%predict_log_proba_device(cuda, query, log_probabilities, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED .and. &
        all(log_probabilities == 7.0_dp), "typed CUDA log-probability refusal preserves output", failures)
    log_probabilities = 8.0_dp
    log_probabilities_dot = 9.0_dp
    call model%predict_log_proba_shared_parameter_jvp_device(cuda, query, shared_direction, &
        log_probabilities, log_probabilities_dot, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED .and. &
        all(log_probabilities == 8.0_dp) .and. all(log_probabilities_dot == 9.0_dp), &
        "typed CUDA shared-kernel JVP refusal preserves outputs", failures)
    shared_parameter_bar = 10.0_dp
    call model%predict_log_proba_shared_parameter_vjp_device(cuda, query, &
        log_probabilities_bar, shared_parameter_bar, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED .and. &
        all(shared_parameter_bar == 10.0_dp), &
        "typed CUDA shared-kernel VJP refusal preserves output", failures)
    call check(model%device_supported(FORTML_DEVICE_CPU) .and. &
        .not. model%device_supported(FORTML_DEVICE_CUDA), &
        "device capability metadata", failures)

    if (failures > 0) then
        write (error_unit, '(a,i0)') "FAIL multilabel GP cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS multilabel GP independent behavioral and derivative oracle"

contains

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failures
        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "  FAIL [gp-multilabel] "//description
        end if
    end subroutine check

end program test_gp_multilabel_classification
