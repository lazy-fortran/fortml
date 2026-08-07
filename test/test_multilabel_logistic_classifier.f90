program test_multilabel_logistic_classifier
    !! Independent analytic and finite-difference checks for multilabel heads.
    use, intrinsic :: iso_fortran_env, only: dp => real64
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    use fortml_multilabel_logistic_classifier, only: &
        multilabel_logistic_classifier_t
    implicit none

    type(multilabel_logistic_classifier_t) :: model
    type(fortnum_status_t) :: status
    type(fortml_device_t) :: cpu, cuda
    real(dp) :: x(6, 2), x_dot(6, 2), scores(6, 2), probabilities(6, 2)
    real(dp) :: probabilities_plus(6, 2), probabilities_minus(6, 2)
    real(dp) :: probabilities_dot(6, 2), parameter_probabilities_dot(6, 2)
    real(dp) :: probabilities_bar(6, 2), x_bar(6, 2), theta_bar(6)
    real(dp) :: theta_dot(6), parameters(6), parameters_plus(6), parameters_minus(6)
    real(dp) :: expected_scores(6, 2), expected_probabilities(6, 2)
    real(dp) :: thresholds(2), h, lhs, rhs, score
    integer :: indicators(6, 2), predicted(6, 2), failures, i, j

    x = reshape([ &
        -2.0_dp, -1.0_dp, -0.5_dp, 0.0_dp, 0.75_dp, 1.5_dp, &
        1.0_dp, -0.5_dp, 0.25_dp, 1.0_dp, -1.25_dp, 0.5_dp], shape(x))
    indicators = reshape([ &
        0, 1, 0, 1, 1, 0, &
        1, 0, 1, 0, 1, 0], shape(indicators))
    x_dot = reshape([ &
        0.1_dp, -0.2_dp, 0.05_dp, 0.3_dp, -0.4_dp, 0.2_dp, &
        -0.3_dp, 0.25_dp, 0.15_dp, -0.1_dp, 0.2_dp, -0.05_dp], shape(x_dot))
    probabilities_bar = reshape([ &
        0.2_dp, -0.1_dp, 0.3_dp, -0.2_dp, 0.4_dp, -0.3_dp, &
        -0.4_dp, 0.5_dp, -0.6_dp, 0.7_dp, -0.8_dp, 0.9_dp], &
        shape(probabilities_bar))
    failures = 0
    h = 1.0e-5_dp

    call model%fit(x, indicators, status, l2=0.1_dp, max_iterations=1000, &
        tolerance=1.0e-7_dp)
    call check(status_ok(status) .and. model%fitted() .and. &
        model%label_count() == 2 .and. model%feature_count() == 2, &
        "multilabel fit metadata", failures)
    call check(model%parameter_count() == 6, "multilabel parameter count", failures)

    parameters = [0.7_dp, -0.2_dp, 0.1_dp, -0.4_dp, 0.3_dp, -0.25_dp]
    call model%set_parameters(parameters, status)
    call check(status_ok(status), "set known parameters", failures)
    call model%decision_function(x, scores, status)
    do i = 1, size(x, 1)
        expected_scores(i, 1) = x(i, 1)*parameters(1) + x(i, 2)*parameters(2) + &
            parameters(3)
        expected_scores(i, 2) = x(i, 1)*parameters(4) + x(i, 2)*parameters(5) + &
            parameters(6)
    end do
    call check(status_ok(status) .and. maxval(abs(scores-expected_scores)) < 1.0e-13_dp, &
        "analytic multilabel scores", failures)

    call model%predict_proba(x, probabilities, status)
    do j = 1, 2
        do i = 1, size(x, 1)
            score = expected_scores(i, j)
            expected_probabilities(i, j) = 1.0_dp/(1.0_dp + exp(-score))
        end do
    end do
    call check(status_ok(status) .and. &
        maxval(abs(probabilities-expected_probabilities)) < 1.0e-13_dp, &
        "analytic multilabel probabilities", failures)

    theta_dot = [0.03_dp, -0.04_dp, 0.02_dp, -0.01_dp, 0.05_dp, -0.03_dp]
    call model%predict_proba_parameter_jvp(x, theta_dot, probabilities, &
        parameter_probabilities_dot, status)
    parameters_plus = parameters + h*theta_dot
    call model%set_parameters(parameters_plus, status)
    call model%predict_proba(x, probabilities_plus, status)
    parameters_minus = parameters - h*theta_dot
    call model%set_parameters(parameters_minus, status)
    call model%predict_proba(x, probabilities_minus, status)
    call model%set_parameters(parameters, status)
    call check(status_ok(status) .and. maxval(abs(parameter_probabilities_dot - &
        (probabilities_plus-probabilities_minus)/(2.0_dp*h))) < 2.0e-8_dp, &
        "parameter JVP finite difference", failures)

    call model%predict_proba_jvp(x, x_dot, probabilities, probabilities_dot, status)
    call model%predict_proba(x+h*x_dot, probabilities_plus, status)
    call model%predict_proba(x-h*x_dot, probabilities_minus, status)
    call check(status_ok(status) .and. maxval(abs(probabilities_dot - &
        (probabilities_plus-probabilities_minus)/(2.0_dp*h))) < 2.0e-8_dp, &
        "input JVP finite difference", failures)

    call model%predict_proba_parameter_vjp(x, probabilities_bar, theta_bar, status)
    lhs = sum(probabilities_bar*parameter_probabilities_dot)
    rhs = sum(theta_bar*theta_dot)
    call check(status_ok(status) .and. abs(lhs-rhs) < 2.0e-8_dp, &
        "parameter VJP adjoint identity", failures)
    call model%predict_proba_vjp(x, probabilities_bar, x_bar, status)
    lhs = sum(probabilities_bar*probabilities_dot)
    rhs = sum(x_bar*x_dot)
    call check(status_ok(status) .and. abs(lhs-rhs) < 2.0e-8_dp, &
        "input VJP adjoint identity", failures)

    thresholds = [0.6_dp, 0.4_dp]
    call model%set_thresholds(thresholds, status)
    call model%predict(x, predicted, status)
    call check(status_ok(status), "multilabel prediction", failures)
    do j = 1, 2
        do i = 1, size(x, 1)
            if (expected_probabilities(i, j) >= thresholds(j)) then
                call check(predicted(i, j) == 1, "threshold positive prediction", failures)
            else
                call check(predicted(i, j) == 0, "threshold negative prediction", failures)
            end if
        end do
    end do

    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call model%predict_proba_device(cuda, x, probabilities, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED .and. &
        .not. model%device_supported(FORTML_DEVICE_CUDA), &
        "CUDA capability refusal", failures)
    call cpu%select(FORTML_DEVICE_CPU, status)
    call model%predict_proba_device(cpu, x, probabilities_plus, status)
    call model%predict_proba(x, probabilities_minus, status)
    call check(status_ok(status) .and. maxval(abs(probabilities_plus- &
        probabilities_minus)) < 1.0e-14_dp .and. &
        model%device_supported(FORTML_DEVICE_CPU), "CPU device dispatch", failures)

    if (failures /= 0) error stop failures
    print '(a)', "test_multilabel_logistic_classifier: PASS"

contains

    subroutine check(condition, description, failure_count)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failure_count

        if (.not. condition) then
            failure_count = failure_count + 1
            write (*, '(a)') "FAIL: "//trim(description)
        end if
    end subroutine check

end program test_multilabel_logistic_classifier
