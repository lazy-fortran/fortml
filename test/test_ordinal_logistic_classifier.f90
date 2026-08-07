program test_ordinal_logistic_classifier
    !! Independent cumulative-logit, derivative, and device-contract oracle.
    use, intrinsic :: iso_fortran_env, only: real64
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    use fortml_ordinal_logistic_classifier, only: ordinal_logistic_classifier_t
    implicit none

    type(ordinal_logistic_classifier_t) :: model
    type(fortnum_status_t) :: status
    type(fortml_device_t) :: cpu, cuda
    real(real64) :: x(6, 1), x_dot(6, 1), probabilities(6, 3)
    real(real64) :: probabilities_dot(6, 3), probabilities_plus(6, 3)
    real(real64) :: probabilities_minus(6, 3), probabilities_bar(6, 3)
    real(real64) :: theta(4), theta_dot(4), theta_plus(4), theta_minus(4)
    real(real64) :: theta_bar(4), x_bar(6, 1), expected(6, 3)
    real(real64) :: score, q1, q2, h, lhs, rhs
    real(real64) :: sample_weight(6)
    integer :: labels(6), predicted(6), classes(3), failures, i

    x(:, 1) = [-2.0_real64, -1.0_real64, 0.0_real64, 1.0_real64, &
        2.0_real64, 3.0_real64]
    x_dot(:, 1) = [0.07_real64, -0.11_real64, 0.13_real64, -0.17_real64, &
        0.19_real64, -0.23_real64]
    labels = [20, 10, 20, 30, 30, 10]
    probabilities_bar = reshape([0.2_real64, -0.3_real64, 0.4_real64, &
        -0.1_real64, 0.5_real64, -0.6_real64, 0.7_real64, -0.8_real64, &
        0.9_real64, -0.2_real64, 0.3_real64, -0.4_real64, 0.1_real64, &
        -0.5_real64, 0.6_real64, -0.7_real64, 0.8_real64, -0.9_real64], &
        shape(probabilities_bar))
    failures = 0
    h = 1.0e-6_real64

    sample_weight = [1.0_real64, 2.0_real64, 1.0_real64, 1.5_real64, &
        2.0_real64, 1.0_real64]
    call model%fit(x, labels, status, l2=0.1_real64, max_iterations=1000, &
        tolerance=1.0e-7_real64, sample_weight=sample_weight)
    call check(status_ok(status) .and. model%fitted(), &
        "weighted ordinal fit", failures)
    call check(model%class_count() == 3 .and. model%feature_count() == 1 .and. &
        model%parameter_count() == 4, "ordinal metadata", failures)
    classes = model%classes()
    call check(all(classes == [10, 20, 30]), "ordered arbitrary labels", failures)

    theta = [0.4_real64, 0.1_real64, 0.2_real64, 1.1_real64]
    call model%set_parameters(theta, status)
    call check(status_ok(status), "set known ordinal parameters", failures)
    call model%predict_proba(x, probabilities, status)
    do i = 1, size(x, 1)
        score = theta(1)*x(i, 1) + theta(2)
        q1 = stable_sigmoid(theta(3)-score)
        q2 = stable_sigmoid(theta(4)-score)
        expected(i, :) = [q1, q2-q1, 1.0_real64-q2]
    end do
    call check(status_ok(status) .and. maxval(abs(probabilities-expected)) < &
        2.0e-14_real64, "cumulative-logit probability oracle", failures)
    call model%predict(x, predicted, status)
    call check(status_ok(status) .and. all(predicted == [10, 10, 10, 10, 30, 30]), &
        "ordered argmax prediction", failures)

    theta_dot = [0.03_real64, -0.02_real64, 0.04_real64, -0.05_real64]
    call model%predict_proba_parameter_jvp(x, theta_dot, probabilities, &
        probabilities_dot, status)
    theta_plus = theta + h*theta_dot
    theta_minus = theta - h*theta_dot
    call model%set_parameters(theta_plus, status)
    call model%predict_proba(x, probabilities_plus, status)
    call model%set_parameters(theta_minus, status)
    call model%predict_proba(x, probabilities_minus, status)
    call model%set_parameters(theta, status)
    call check(status_ok(status) .and. maxval(abs(probabilities_dot - &
        (probabilities_plus-probabilities_minus)/(2.0_real64*h))) < 3.0e-8_real64, &
        "parameter JVP finite difference", failures)

    call model%predict_proba_jvp(x, x_dot, probabilities, probabilities_dot, status)
    call model%predict_proba(x+h*x_dot, probabilities_plus, status)
    call model%predict_proba(x-h*x_dot, probabilities_minus, status)
    call check(status_ok(status) .and. maxval(abs(probabilities_dot - &
        (probabilities_plus-probabilities_minus)/(2.0_real64*h))) < 3.0e-8_real64, &
        "input JVP finite difference", failures)
    call model%predict_proba_parameter_jvp(x, theta_dot, probabilities, &
        probabilities_dot, status)
    call model%predict_proba_parameter_vjp(x, probabilities_bar, theta_bar, status)
    lhs = sum(probabilities_bar*probabilities_dot)
    rhs = sum(theta_bar*theta_dot)
    call check(status_ok(status) .and. abs(lhs-rhs) < 3.0e-8_real64, &
        "parameter VJP adjoint identity", failures)
    call model%predict_proba_jvp(x, x_dot, probabilities, probabilities_dot, status)
    call model%predict_proba_vjp(x, probabilities_bar, x_bar, status)
    lhs = sum(probabilities_bar*probabilities_dot)
    rhs = sum(x_bar*x_dot)
    call check(status_ok(status) .and. abs(lhs-rhs) < 3.0e-8_real64, &
        "input VJP adjoint identity", failures)

    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call model%predict_proba_device(cuda, x, probabilities, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED .and. &
        .not. model%device_supported(FORTML_DEVICE_CUDA), &
        "CUDA typed refusal", failures)
    call cpu%select(FORTML_DEVICE_CPU, status)
    call model%predict_proba_device(cpu, x, probabilities_plus, status)
    call model%predict_proba(x, probabilities_minus, status)
    call check(status_ok(status) .and. maxval(abs(probabilities_plus- &
        probabilities_minus)) < 2.0e-14_real64 .and. &
        model%device_supported(FORTML_DEVICE_CPU), "CPU device dispatch", failures)

    if (failures /= 0) error stop failures
    print '(a)', "test_ordinal_logistic_classifier: PASS"

contains

    pure real(real64) function stable_sigmoid(value) result(probability)
        real(real64), intent(in) :: value
        if (value >= 0.0_real64) then
            probability = 1.0_real64/(1.0_real64+exp(-value))
        else
            probability = exp(value)/(1.0_real64+exp(value))
        end if
    end function stable_sigmoid

    subroutine check(condition, description, failure_count)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failure_count
        if (.not. condition) then
            failure_count = failure_count + 1
            write (*, '(a)') "FAIL: "//trim(description)
        end if
    end subroutine check

end program test_ordinal_logistic_classifier
