program test_calibrated_logistic_classifier
    !! Independent finite-difference and fold-contract checks for calibrated OOF logistic.
    use, intrinsic :: iso_fortran_env, only: real64, error_unit
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortml_calibrated_logistic_classifier, only: &
        calibrated_logistic_classifier_t, calibrated_logistic_classifier_options_t
    use fortml_probability_calibration, only: CALIBRATION_TEMPERATURE, &
        CALIBRATION_ISOTONIC
    implicit none

    integer, parameter :: dp = real64
    type(calibrated_logistic_classifier_t) :: model, isotonic
    type(calibrated_logistic_classifier_options_t) :: options
    type(fortnum_status_t) :: status
    type(fortml_device_t) :: cuda
    real(dp) :: x(12, 2), x_dot(12, 2), probabilities(12, 2)
    real(dp) :: probabilities_dot(12, 2), probabilities_plus(12, 2)
    real(dp) :: probabilities_minus(12, 2), probabilities_bar(12, 2)
    real(dp), allocatable :: theta(:), theta_dot(:), theta_plus(:), theta_minus(:), theta_bar(:)
    real(dp) :: x_bar(12, 2), lhs, rhs, h
    integer :: labels(12), predicted(12), failures, i

    failures = 0
    h = 1.0e-6_dp
    x(1, :) = [-2.0_dp, -1.4_dp]
    x(2, :) = [-1.7_dp, -0.8_dp]
    x(3, :) = [-1.4_dp, -1.1_dp]
    x(4, :) = [-1.1_dp, -0.5_dp]
    x(5, :) = [-0.7_dp, -0.2_dp]
    x(6, :) = [-0.3_dp, 0.1_dp]
    x(7, :) = [0.2_dp, -0.1_dp]
    x(8, :) = [0.5_dp, 0.3_dp]
    x(9, :) = [0.8_dp, 0.7_dp]
    x(10, :) = [1.1_dp, 0.9_dp]
    x(11, :) = [1.5_dp, 1.2_dp]
    x(12, :) = [1.9_dp, 1.6_dp]
    labels = [-3, -3, -3, -3, -3, -3, 42, 42, 42, 42, 42, 42]
    x_dot = 0.0_dp
    do i = 1, size(x, 1)
        x_dot(i, 1) = 0.01_dp*sin(real(i, dp))
        x_dot(i, 2) = -0.02_dp*cos(real(i, dp))
    end do

    options = calibrated_logistic_classifier_options_t()
    options%l2 = 0.1_dp
    options%max_iterations = 300
    options%tolerance = 1.0e-8_dp
    options%cv_folds = 3
    options%cv_shuffle = .true.
    options%cv_seed = 19
    options%calibration%method = CALIBRATION_TEMPERATURE
    options%calibration%max_iterations = 300
    options%calibration%tolerance = 1.0e-10_dp
    call model%fit(x, labels, status, options=options)
    call check(status_ok(status) .and. model%fitted(), &
        "stratified OOF calibrated logistic fit", failures)
    call check(model%cv_folds() == 3 .and. model%oof_log_loss() > 0.0_dp .and. &
        model%calibrated_oof_log_loss() > 0.0_dp, &
        "OOF fold and finite log-loss diagnostics", failures)
    call check(all(model%classes() == [-3, 42]), &
        "sorted arbitrary class labels", failures)
    call model%predict_proba(x, probabilities, status)
    call model%predict(x, predicted, status)
    call check(status_ok(status) .and. maxval(abs(sum(probabilities, dim=2) - 1.0_dp)) < 1.0e-13_dp &
        .and. all(predicted(:6) == -3) .and. all(predicted(7:) == 42), &
        "calibrated probability simplex and labels", failures)

    theta = model%parameters()
    allocate(theta_dot(size(theta)), theta_plus(size(theta)), theta_minus(size(theta)), &
        theta_bar(size(theta)))
    theta_dot = [(0.01_dp*sin(real(i, dp)), i=1,size(theta))]
    probabilities_bar = reshape([ &
        0.1_dp, -0.2_dp, 0.3_dp, -0.4_dp, 0.5_dp, -0.1_dp, 0.2_dp, -0.3_dp, &
        -0.2_dp, 0.1_dp, -0.4_dp, 0.3_dp, -0.1_dp, 0.5_dp, -0.3_dp, 0.2_dp, &
        0.4_dp, -0.1_dp, 0.2_dp, -0.5_dp, 0.3_dp, -0.2_dp, 0.1_dp, -0.3_dp], &
        shape(probabilities_bar))
    call model%predict_proba_jvp(x, theta_dot, x_dot, probabilities, probabilities_dot, status)
    theta_plus = theta + h*theta_dot
    theta_minus = theta - h*theta_dot
    call model%set_parameters(theta_plus, status)
    call model%predict_proba(x+h*x_dot, probabilities_plus, status)
    call model%set_parameters(theta_minus, status)
    call model%predict_proba(x-h*x_dot, probabilities_minus, status)
    call model%set_parameters(theta, status)
    call check(status_ok(status) .and. maxval(abs(probabilities_dot - &
        (probabilities_plus - probabilities_minus)/(2.0_dp*h))) < 5.0e-5_dp, &
        "temperature OOF classifier joint JVP", failures)
    call model%predict_proba_vjp(x, probabilities_bar, theta_bar, x_bar, status)
    lhs = sum(probabilities_bar*probabilities_dot)
    rhs = sum(theta_bar*theta_dot) + sum(x_bar*x_dot)
    call check(status_ok(status) .and. abs(lhs - rhs) < 5.0e-5_dp, &
        "temperature OOF classifier VJP adjoint", failures)

    options%calibration%method = CALIBRATION_ISOTONIC
    call isotonic%fit(x, labels, status, options=options)
    call check(status_ok(status) .and. isotonic%fitted(), &
        "OOF isotonic calibration fit", failures)
    call isotonic%predict_proba_jvp(x, theta_dot(:isotonic%parameter_count()), x_dot, &
        probabilities, probabilities_dot, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "isotonic active-set derivative refusal", failures)

    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call model%predict_proba_device(cuda, x, probabilities, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED .and. &
        .not. model%device_supported(FORTML_DEVICE_CUDA), &
        "calibrated logistic CUDA refusal", failures)

    if (failures > 0) then
        write (error_unit, '(a,i0)') "FAIL calibrated logistic cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS calibrated logistic OOF independent oracles"

contains

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "  FAIL [calibrated-logistic] "//description
        end if
    end subroutine check

end program test_calibrated_logistic_classifier
