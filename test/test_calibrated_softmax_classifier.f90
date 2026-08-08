program test_calibrated_softmax_classifier
    !! Independent OOF, probability, derivative, and device-boundary checks.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortml_probability_calibration, only: &
        probability_calibration_options_t, CALIBRATION_TEMPERATURE, CALIBRATION_SIGMOID
    use fortml_calibrated_softmax_classifier, only: &
        calibrated_softmax_classifier_t, calibrated_softmax_classifier_options_t, &
        calibrated_softmax_classifier_state_t
    implicit none

    type(calibrated_softmax_classifier_t) :: model, repeated, invalid, unfitted
    type(calibrated_softmax_classifier_options_t) :: options
    type(calibrated_softmax_classifier_options_t) :: invalid_options
    type(calibrated_softmax_classifier_state_t) :: state
    type(fortnum_status_t) :: status
    type(fortml_device_t) :: cuda
    real(dp) :: x(12, 2), probabilities(12, 3), probabilities_dot(12, 3)
    real(dp) :: probabilities_plus(12, 3), probabilities_minus(12, 3)
    real(dp) :: probabilities_bar(12, 3), x_dot(12, 2), x_bar(12, 2)
    real(dp), allocatable :: parameters(:), parameters_dot(:), parameters_plus(:)
    real(dp), allocatable :: parameters_minus(:), parameters_bar(:)
    integer :: labels(12), predicted(12), class_labels(3), failures
    real(dp) :: h, lhs, rhs

    x = reshape([ &
        2.0_dp, 0.0_dp, 1.5_dp, 0.2_dp, 2.0_dp, -0.3_dp, 1.2_dp, 0.1_dp, &
        0.0_dp, 2.0_dp, 0.2_dp, 1.5_dp, -0.3_dp, 2.0_dp, 0.1_dp, 1.2_dp, &
        -2.0_dp, -2.0_dp, -1.5_dp, -1.8_dp, -2.0_dp, -1.5_dp, -1.2_dp, -1.3_dp], &
        shape(x), order=[2, 1])
    labels = [7, 7, 7, 7, 42, 42, 42, 42, 99, 99, 99, 99]
    probabilities_bar = reshape([ &
        0.2_dp, -0.1_dp, 0.3_dp, 0.1_dp, -0.2_dp, 0.4_dp, &
        -0.3_dp, 0.5_dp, -0.2_dp, 0.4_dp, -0.1_dp, 0.2_dp, &
        0.3_dp, -0.4_dp, 0.2_dp, -0.2_dp, 0.1_dp, -0.3_dp, &
        0.4_dp, -0.2_dp, 0.1_dp, -0.1_dp, 0.3_dp, -0.4_dp, &
        -0.2_dp, 0.3_dp, 0.5_dp, 0.1_dp, -0.4_dp, 0.2_dp, &
        0.2_dp, 0.1_dp, -0.3_dp, 0.3_dp, -0.2_dp, 0.4_dp], &
        shape(probabilities_bar), order=[2, 1])
    x_dot = reshape([ &
        0.01_dp, -0.02_dp, 0.03_dp, -0.01_dp, 0.02_dp, -0.03_dp, &
        -0.02_dp, 0.01_dp, 0.02_dp, -0.01_dp, 0.03_dp, -0.02_dp, &
        0.01_dp, 0.02_dp, -0.03_dp, 0.01_dp, -0.02_dp, 0.03_dp, &
        0.02_dp, -0.01_dp, 0.01_dp, -0.03_dp, 0.02_dp, -0.02_dp], &
        shape(x_dot), order=[2, 1])
    failures = 0
    h = 1.0e-6_dp
    options = calibrated_softmax_classifier_options_t()
    options%l2 = 0.2_dp
    options%max_iterations = 500
    options%tolerance = 1.0e-8_dp
    options%cv_folds = 3
    options%cv_shuffle = .true.
    options%cv_seed = 23
    options%calibration = probability_calibration_options_t( &
        method=CALIBRATION_TEMPERATURE, max_iterations=500, tolerance=1.0e-10_dp, &
        damping=1.0_dp, l2=0.05_dp)

    call model%fit(x, labels, status, options=options, state=state)
    call check(status_ok(status), "multiclass OOF fit status", failures)
    call check(model%fitted() .and. state%converged .and. state%cv_folds == 3, &
        "multiclass OOF convergence metadata", failures)
    class_labels = model%classes()
    call check(all(class_labels == [7, 42, 99]) .and. model%class_count() == 3, &
        "sorted class metadata", failures)
    call check(model%oof_log_loss() > 0.0_dp .and. &
        model%calibrated_oof_log_loss() > 0.0_dp .and. model%temperature() > 0.0_dp, &
        "OOF loss and positive temperature metadata", failures)

    call model%predict_proba(x, probabilities, status)
    call check(status_ok(status), "calibrated probability status", failures)
    call check(maxval(abs(sum(probabilities, dim=2) - 1.0_dp)) < 2.0e-14_dp, &
        "calibrated probability simplex", failures)
    call model%predict(x, predicted, status)
    call check(status_ok(status) .and. real(count(predicted == labels), dp)/12.0_dp > 0.8_dp, &
        "calibrated prediction accuracy", failures)

    parameters = model%parameters()
    call check(size(parameters) == model%parameter_count() .and. size(parameters) > 1, &
        "packed base and calibration parameters", failures)
    allocate(parameters_dot(size(parameters)), parameters_plus(size(parameters)), &
        parameters_minus(size(parameters)), parameters_bar(size(parameters)))
    parameters_dot = 0.0_dp
    if (size(parameters) >= 1) parameters_dot(1) = 0.02_dp
    if (size(parameters) >= 2) parameters_dot(2) = -0.01_dp
    if (size(parameters) >= 3) parameters_dot(3) = 0.03_dp
    if (size(parameters) >= 4) parameters_dot(4) = -0.02_dp
    parameters_dot(size(parameters)) = 0.07_dp
    parameters_bar = 0.0_dp
    if (size(parameters) >= 1) parameters_bar(1) = 0.2_dp
    if (size(parameters) >= 2) parameters_bar(2) = -0.1_dp
    if (size(parameters) >= 3) parameters_bar(3) = 0.3_dp
    if (size(parameters) >= 4) parameters_bar(4) = -0.2_dp
    parameters_bar(size(parameters)) = 0.17_dp

    call model%predict_proba_jvp(x, parameters_dot, x_dot, probabilities, probabilities_dot, status)
    call model%predict_proba_parameter_jvp(x, parameters_dot, probabilities_plus, &
        probabilities_minus, status, x_dot=x_dot)
    call check(status_ok(status) .and. maxval(abs(probabilities_dot - &
        probabilities_minus)) < 2.0e-14_dp, &
        "parameter JVP wrapper agrees with full JVP", failures)
    parameters_plus = parameters + h*parameters_dot
    parameters_minus = parameters - h*parameters_dot
    call model%set_parameters(parameters_plus, status)
    call model%predict_proba(x + h*x_dot, probabilities_plus, status)
    call model%set_parameters(parameters_minus, status)
    call model%predict_proba(x - h*x_dot, probabilities_minus, status)
    call model%set_parameters(parameters, status)
    call check(status_ok(status) .and. maxval(abs(probabilities_dot - &
        (probabilities_plus-probabilities_minus)/(2.0_dp*h))) < 2.0e-5_dp, &
        "full probability JVP finite difference", failures)

    call model%predict_proba_vjp(x, probabilities_bar, parameters_bar, x_bar, status)
    lhs = sum(probabilities_bar*probabilities_dot)
    rhs = sum(parameters_bar*parameters_dot) + sum(x_bar*x_dot)
    call check(status_ok(status) .and. abs(lhs-rhs) < 2.0e-5_dp, &
        "full probability VJP adjoint identity", failures)

    call repeated%fit(x, labels, status, options=options)
    call repeated%predict_proba(x, probabilities_plus, status)
    call model%predict_proba(x, probabilities_minus, status)
    call check(status_ok(status) .and. maxval(abs(probabilities_plus-probabilities_minus)) < &
        2.0e-13_dp, &
        "deterministic OOF calibration", failures)

    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call model%predict_proba_device(cuda, x, probabilities_plus, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED .and. &
        .not. model%device_supported(FORTML_DEVICE_CUDA), "multiclass CUDA refusal", failures)

    call invalid%fit(x, labels, status, options=calibrated_softmax_classifier_options_t( &
        cv_folds=5, calibration=options%calibration))
    call check(.not. status_ok(status), "insufficient per-class fold support refusal", failures)
    call invalid%fit(x, labels, status, options=options, sample_weight=[1.0_dp, 1.0_dp])
    call check(.not. status_ok(status), "sample-weight shape refusal", failures)
    invalid_options = options
    invalid_options%calibration = probability_calibration_options_t(method=CALIBRATION_SIGMOID)
    call invalid%fit(x, labels, status, options=invalid_options)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "multiclass non-temperature refusal", failures)
    call unfitted%predict_proba(x, probabilities, status)
    call check(.not. status_ok(status), "unfitted prediction refusal", failures)

    if (failures > 0) then
        write (error_unit, '(a,i0)') "FAIL calibrated softmax cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS calibrated softmax OOF independent oracles"

contains

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "  FAIL [calibrated-softmax] "//description
        end if
    end subroutine check

end program test_calibrated_softmax_classifier
