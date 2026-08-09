program test_classifier_chain
    !! Independent analytic and finite-difference oracle for a logistic chain.
    use, intrinsic :: iso_fortran_env, only: dp => real64
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_DOMAIN_ERROR, &
        FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    use fortml_classifier_chain, only: classifier_chain_t
    implicit none

    type(classifier_chain_t) :: model, clone, destination, unfitted
    type(fortnum_status_t) :: status
    type(fortml_device_t) :: cpu, cuda
    real(dp) :: x(6, 2), x_dot(6, 2), probabilities(6, 2)
    real(dp) :: probabilities_plus(6, 2), probabilities_minus(6, 2)
    real(dp) :: probabilities_dot(6, 2), parameter_dot(6, 2)
    real(dp) :: probabilities_bar(6, 2), x_bar(6, 2), theta_bar(7)
    real(dp) :: theta_hvp(7), x_hvp(6, 2), theta_bar_plus(7), theta_bar_minus(7)
    real(dp) :: x_bar_plus(6, 2), x_bar_minus(6, 2)
    real(dp) :: theta(7), theta_dot(7), theta_plus(7), theta_minus(7)
    real(dp) :: clone_probabilities(6, 2), source_probabilities(6, 2)
    real(dp) :: source_parameters(7), clone_parameters(7), changed_parameters(7)
    real(dp) :: expected(6, 2), expected_plus, h, lhs, rhs
    real(dp) :: thresholds(2)
    integer :: labels(6, 2), predicted(6, 2), classes(2, 2), failures, clone_code
    integer :: i

    x(:, 1) = [-1.0_dp, -0.5_dp, 0.0_dp, 0.5_dp, 1.0_dp, 1.5_dp]
    x(:, 2) = [1.0_dp, -1.0_dp, 0.5_dp, -0.5_dp, 1.0_dp, -1.5_dp]
    x_dot(:, 1) = [0.1_dp, -0.2_dp, 0.05_dp, 0.3_dp, -0.4_dp, 0.2_dp]
    x_dot(:, 2) = [-0.3_dp, 0.25_dp, 0.15_dp, -0.1_dp, 0.2_dp, -0.05_dp]
    labels(:, 1) = [-3, -3, 7, 7, -3, 7]
    labels(:, 2) = [10, 20, 20, 10, 20, 10]
    probabilities_bar(:, 1) = [0.2_dp, -0.1_dp, 0.3_dp, -0.2_dp, 0.4_dp, -0.3_dp]
    probabilities_bar(:, 2) = [-0.4_dp, 0.5_dp, -0.6_dp, 0.7_dp, -0.8_dp, 0.9_dp]
    theta = [0.7_dp, -0.2_dp, 0.1_dp, 0.4_dp, -0.3_dp, 0.8_dp, -0.25_dp]
    theta_dot = [0.03_dp, -0.04_dp, 0.02_dp, -0.01_dp, 0.05_dp, -0.03_dp, 0.04_dp]
    h = 1.0e-5_dp
    failures = 0

    call model%fit(x, labels, status, l2=0.1_dp, max_iterations=1000, &
        tolerance=1.0e-7_dp, class_weight=reshape([1.0_dp, 1.5_dp, 0.8_dp, 1.2_dp], [2, 2]))
    call check(status_ok(status) .and. model%fitted(), "chain fit", failures)
    call check(model%output_count() == 2 .and. model%feature_count() == 2 .and. &
        model%parameter_count() == 7, "chain metadata", failures)
    classes = model%classes()
    call check(all(classes == reshape([-3, 7, 10, 20], [2, 2])), &
        "sorted per-output classes", failures)
    call model%set_parameters(theta, status)
    call check(status_ok(status), "set chain parameters", failures)

    do i = 1, 6
        expected(i, 1) = sigmoid(x(i, 1)*theta(1) + x(i, 2)*theta(2) + theta(3))
        expected(i, 2) = sigmoid(x(i, 1)*theta(4) + x(i, 2)*theta(5) + &
            expected(i, 1)*theta(6) + theta(7))
    end do
    call model%predict_proba(x, probabilities, status)
    call check(status_ok(status) .and. maxval(abs(probabilities-expected)) < 1.0e-13_dp, &
        "independent chain probability oracle", failures)

    thresholds = [0.5_dp, 0.6_dp]
    call model%set_thresholds(thresholds, status)
    call model%predict(x, predicted, status)
    do i = 1, 6
        call check(predicted(i, 1) == merge(7, -3, expected(i, 1) >= thresholds(1)), &
            "first chain hard-label mapping", failures)
        call check(predicted(i, 2) == merge(20, 10, expected(i, 2) >= thresholds(2)), &
            "second chain hard-label mapping", failures)
    end do

    call model%clone(clone, status)
    call clone%predict_proba(x, clone_probabilities, status)
    source_parameters = model%parameters()
    clone_parameters = clone%parameters()
    call check(status_ok(status) .and. clone%fitted() .and. &
        clone%output_count() == model%output_count() .and. &
        clone%feature_count() == model%feature_count() .and. &
        clone%parameter_count() == model%parameter_count() .and. &
        maxval(abs(clone_probabilities - expected)) < 1.0e-13_dp .and. &
        maxval(abs(clone_parameters - source_parameters)) < 1.0e-14_dp .and. &
        maxval(abs(clone%thresholds() - thresholds)) < 1.0e-14_dp, &
        "clone reproduces independent chain behavior", failures)
    changed_parameters = clone_parameters
    changed_parameters(1) = changed_parameters(1) + 0.35_dp
    call clone%set_parameters(changed_parameters, status)
    call clone%predict_proba(x, clone_probabilities, status)
    call model%predict_proba(x, source_probabilities, status)
    call check(status_ok(status) .and. &
        maxval(abs(model%parameters() - source_parameters)) < 1.0e-14_dp .and. &
        maxval(abs(source_probabilities - clone_probabilities)) > 1.0e-8_dp, &
        "clone mutation does not alias source heads", failures)

    call model%predict_proba(x, source_probabilities, status)
    destination = model
    call unfitted%clone(destination, status)
    clone_code = status%code
    call destination%predict_proba(x, clone_probabilities, status)
    call check(clone_code == FORTNUM_DOMAIN_ERROR .and. status_ok(status) .and. &
        maxval(abs(destination%parameters() - source_parameters)) < 1.0e-14_dp .and. &
        maxval(abs(clone_probabilities - source_probabilities)) < 1.0e-14_dp, &
        "invalid clone source leaves destination unchanged", failures)

    call model%predict_proba_jvp(x, x_dot, probabilities, probabilities_dot, status)
    call model%predict_proba(x+h*x_dot, probabilities_plus, status)
    call model%predict_proba(x-h*x_dot, probabilities_minus, status)
    call check(status_ok(status) .and. maxval(abs(probabilities_dot - &
        (probabilities_plus-probabilities_minus)/(2.0_dp*h))) < 2.0e-8_dp, &
        "input JVP finite difference", failures)

    call model%predict_proba_parameter_jvp(x, theta_dot, probabilities, parameter_dot, status)
    theta_plus = theta + h*theta_dot
    call model%set_parameters(theta_plus, status)
    call model%predict_proba(x, probabilities_plus, status)
    theta_minus = theta - h*theta_dot
    call model%set_parameters(theta_minus, status)
    call model%predict_proba(x, probabilities_minus, status)
    call model%set_parameters(theta, status)
    call check(status_ok(status) .and. maxval(abs(parameter_dot - &
        (probabilities_plus-probabilities_minus)/(2.0_dp*h))) < 2.0e-8_dp, &
        "parameter JVP finite difference", failures)

    call model%predict_proba_parameter_vjp(x, probabilities_bar, theta_bar, status)
    lhs = sum(probabilities_bar*parameter_dot)
    rhs = sum(theta_bar*theta_dot)
    call check(status_ok(status) .and. abs(lhs-rhs) < 2.0e-8_dp, &
        "parameter VJP adjoint identity", failures)
    call model%predict_proba_vjp(x, probabilities_bar, x_bar, status)
    lhs = sum(probabilities_bar*probabilities_dot)
    rhs = sum(x_bar*x_dot)
    call check(status_ok(status) .and. abs(lhs-rhs) < 2.0e-8_dp, &
        "input VJP adjoint identity", failures)

    call model%predict_proba_hvp(x, probabilities_bar, theta_dot, x_dot, &
        theta_hvp, x_hvp, status)
    call check(status_ok(status), "HVP status", failures)
    call model%set_parameters(theta_plus, status)
    call model%predict_proba_parameter_vjp(x+h*x_dot, probabilities_bar, &
        theta_bar_plus, status)
    call model%predict_proba_vjp(x+h*x_dot, probabilities_bar, x_bar_plus, status)
    call model%set_parameters(theta_minus, status)
    call model%predict_proba_parameter_vjp(x-h*x_dot, probabilities_bar, &
        theta_bar_minus, status)
    call model%predict_proba_vjp(x-h*x_dot, probabilities_bar, x_bar_minus, status)
    call model%set_parameters(theta, status)
    call check(status_ok(status) .and. maxval(abs(theta_hvp - &
        (theta_bar_plus-theta_bar_minus)/(2.0_dp*h))) < 2.0e-7_dp .and. &
        maxval(abs(x_hvp - (x_bar_plus-x_bar_minus)/(2.0_dp*h))) < 2.0e-7_dp, &
        "joint parameter/input HVP finite difference", failures)

    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call model%predict_proba_device(cuda, x, probabilities, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED .and. &
        .not. model%device_supported(FORTML_DEVICE_CUDA), "CUDA refusal", failures)
    call cpu%select(FORTML_DEVICE_CPU, status)
    call model%predict_proba_device(cpu, x, probabilities_plus, status)
    call model%predict_proba(x, probabilities_minus, status)
    call check(status_ok(status) .and. maxval(abs(probabilities_plus- &
        probabilities_minus)) < 1.0e-14_dp .and. &
        model%device_supported(FORTML_DEVICE_CPU), "CPU dispatch", failures)
    call model%predict_proba_hvp_device(cuda, x, probabilities_bar, theta_dot, &
        x_dot, theta_hvp, x_hvp, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, "CUDA HVP refusal", failures)
    call model%clone_device(cuda, destination, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, "CUDA clone refusal", failures)
    call cpu%select(FORTML_DEVICE_CPU, status)
    call model%clone_device(cpu, destination, status)
    call destination%predict_proba(x, clone_probabilities, status)
    call check(status_ok(status) .and. maxval(abs(clone_probabilities - source_probabilities)) < &
        1.0e-14_dp, "CPU clone dispatch", failures)

    if (failures /= 0) error stop failures
    print '(a)', "test_classifier_chain: PASS"

contains

    real(dp) function sigmoid(value) result(probability)
        real(dp), intent(in) :: value
        if (value >= 0.0_dp) then
            probability = 1.0_dp/(1.0_dp + exp(-value))
        else
            probability = exp(value)/(1.0_dp + exp(value))
        end if
    end function sigmoid

    subroutine check(condition, description, failure_count)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failure_count

        if (.not. condition) then
            failure_count = failure_count + 1
            write (*, '(a)') "FAIL: "//trim(description)
        end if
    end subroutine check

end program test_classifier_chain
