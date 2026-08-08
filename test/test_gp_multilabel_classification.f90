program test_gp_multilabel_classification
    !! Independent behavioral and derivative oracles for multilabel Laplace GPs.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
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
    real(dp) :: latent(5, 2), latent_dot(5, 2), variance(5, 2), variance_dot(5, 2)
    real(dp), allocatable :: direction(:), parameter_bar(:), parameters(:)
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

    call model%predict_proba(query, probabilities, status)
    call model%predict(query, predicted, status)
    call check(status_ok(status) .and. all(probabilities > 0.0_dp) .and. &
        all(probabilities < 1.0_dp), "independent positive probabilities", failures)
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
    parameters = model%parameters()
    call check(size(parameters) == model%parameter_count() .and. &
        all(ieee_is_finite(parameters)), "packed parameter metadata", failures)
    call model%hyperparameter_gradient(parameter_bar, status)
    call check(status_ok(status) .and. all(ieee_is_finite(parameter_bar)), &
        "sum-of-head hyperparameter gradient", failures)

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
    call model%predict_device(cuda, query, predicted, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "typed CUDA label refusal", failures)
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
