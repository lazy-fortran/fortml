program test_probability_calibration
    !! Independent analytic and finite-difference checks for calibration.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    use fortml_probability_calibration, only: &
        probability_calibrator_t, probability_calibration_options_t, &
        probability_calibration_state_t, CALIBRATION_SIGMOID, CALIBRATION_ISOTONIC
    implicit none

    type(probability_calibrator_t) :: sigmoid_model, isotonic_model, unfitted
    type(probability_calibration_options_t) :: options
    type(probability_calibration_state_t) :: state
    type(fortnum_status_t) :: status
    type(fortml_device_t) :: cpu, cuda
    real(dp) :: scores(8), scores_dot(8), probabilities(8, 2), probabilities_dot(8, 2)
    real(dp) :: probabilities_plus(8, 2), probabilities_minus(8, 2)
    real(dp) :: probabilities_bar(8, 2), scores_bar(8)
    real(dp) :: parameter_probabilities_dot(8, 2), parameters_dot(2), parameters_bar(2)
    real(dp) :: isotonic_scores(7), isotonic_query(3), isotonic_dot(3)
    real(dp) :: isotonic_probabilities(3, 2), isotonic_probabilities_dot(3, 2)
    real(dp) :: isotonic_plus(3, 2), isotonic_minus(3, 2)
    integer :: labels(8), isotonic_labels(7), predicted(8), classes(2), failures, i
    real(dp), allocatable :: parameters(:), parameters_plus(:), parameters_minus(:)
    real(dp) :: lhs, rhs, h

    scores = [-2.0_dp, -1.5_dp, -1.0_dp, -0.5_dp, 0.5_dp, 1.0_dp, 1.5_dp, 2.0_dp]
    scores_dot = [0.1_dp, -0.2_dp, 0.3_dp, -0.1_dp, 0.2_dp, -0.3_dp, 0.1_dp, 0.2_dp]
    labels = [10, 10, 10, 10, 42, 42, 42, 42]
    probabilities_bar = reshape([ &
        0.2_dp, -0.1_dp, 0.3_dp, -0.2_dp, 0.4_dp, -0.3_dp, 0.1_dp, 0.5_dp, &
        -0.2_dp, 0.3_dp, -0.4_dp, 0.2_dp, 0.5_dp, -0.1_dp, 0.6_dp, -0.2_dp], &
        shape(probabilities_bar))
    failures = 0
    h = 1.0e-5_dp

    options = probability_calibration_options_t(method=CALIBRATION_SIGMOID, &
        max_iterations=500, tolerance=1.0e-10_dp, damping=1.0_dp, l2=0.1_dp)
    call sigmoid_model%fit(scores, labels, status, options=options, state=state)
    call check(status_ok(status) .and. sigmoid_model%fitted() .and. state%converged, &
        "sigmoid fit convergence", failures)
    classes = sigmoid_model%classes()
    call check(all(classes == [10, 42]) .and. sigmoid_model%method() == CALIBRATION_SIGMOID &
        .and. sigmoid_model%parameter_count() == 2, "sigmoid metadata", failures)
    call sigmoid_model%predict_proba(scores, probabilities, status)
    call sigmoid_model%predict(scores, predicted, status)
    call check(status_ok(status) .and. maxval(abs(sum(probabilities, dim=2) - 1.0_dp)) < 1.0e-14_dp &
        .and. all(probabilities(:, 1) >= 0.0_dp) .and. all(probabilities(:, 2) <= 1.0_dp) &
        .and. all(predicted(:4) == 10) .and. all(predicted(5:) == 42), &
        "sigmoid probabilities and labels", failures)
    call sigmoid_model%predict_proba_jvp(scores, scores_dot, probabilities, probabilities_dot, status)
    call sigmoid_model%predict_proba(scores + h*scores_dot, probabilities_plus, status)
    call sigmoid_model%predict_proba(scores - h*scores_dot, probabilities_minus, status)
    call check(status_ok(status) .and. maxval(abs(probabilities_dot - &
        (probabilities_plus - probabilities_minus)/(2.0_dp*h))) < 2.0e-7_dp, &
        "sigmoid score JVP finite difference", failures)
    call sigmoid_model%predict_proba_vjp(scores, probabilities_bar, scores_bar, status)
    lhs = sum(probabilities_bar*probabilities_dot)
    rhs = sum(scores_bar*scores_dot)
    call check(status_ok(status) .and. abs(lhs-rhs) < 2.0e-7_dp, &
        "sigmoid score VJP adjoint identity", failures)

    parameters = sigmoid_model%parameters()
    parameters_dot = [0.03_dp, -0.02_dp]
    call sigmoid_model%predict_proba_parameter_jvp(scores, parameters_dot, probabilities, &
        parameter_probabilities_dot, status)
    parameters_plus = parameters + h*parameters_dot
    call sigmoid_model%set_parameters(parameters_plus, status)
    call sigmoid_model%predict_proba(scores, probabilities_plus, status)
    parameters_minus = parameters - h*parameters_dot
    call sigmoid_model%set_parameters(parameters_minus, status)
    call sigmoid_model%predict_proba(scores, probabilities_minus, status)
    call sigmoid_model%set_parameters(parameters, status)
    call check(status_ok(status) .and. maxval(abs(parameter_probabilities_dot - &
        (probabilities_plus - probabilities_minus)/(2.0_dp*h))) < 2.0e-7_dp, &
        "sigmoid parameter JVP finite difference", failures)
    call sigmoid_model%predict_proba_parameter_vjp(scores, probabilities_bar, parameters_bar, status)
    lhs = sum(probabilities_bar*parameter_probabilities_dot)
    rhs = sum(parameters_bar*parameters_dot)
    call check(status_ok(status) .and. abs(lhs-rhs) < 2.0e-7_dp, &
        "sigmoid parameter VJP adjoint identity", failures)
    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call sigmoid_model%predict_proba_device(cuda, scores, probabilities, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED .and. &
        .not. sigmoid_model%device_supported(FORTML_DEVICE_CUDA), &
        "sigmoid CUDA prediction refusal", failures)
    call cpu%select(FORTML_DEVICE_CPU, status)
    call sigmoid_model%predict_proba_device(cpu, scores, probabilities_plus, status)
    call sigmoid_model%predict_proba(scores, probabilities_minus, status)
    call check(status_ok(status) .and. maxval(abs(probabilities_plus - probabilities_minus)) < &
        1.0e-14_dp .and. sigmoid_model%device_supported(FORTML_DEVICE_CPU), &
        "sigmoid CPU device dispatch", failures)

    isotonic_scores = [-3.0_dp, -2.0_dp, -1.0_dp, 0.0_dp, 1.0_dp, 2.0_dp, 3.0_dp]
    isotonic_labels = [10, 42, 10, 42, 42, 10, 42]
    options = probability_calibration_options_t(method=CALIBRATION_ISOTONIC)
    call isotonic_model%fit(isotonic_scores, isotonic_labels, status, options=options, state=state)
    call check(status_ok(status) .and. isotonic_model%fitted() .and. state%converged .and. &
        state%knot_count >= 2 .and. isotonic_model%parameter_count() == 0, &
        "isotonic PAVA fit metadata", failures)
    isotonic_query = [-2.5_dp, 0.5_dp, 2.5_dp]
    isotonic_dot = [0.1_dp, -0.2_dp, 0.3_dp]
    call isotonic_model%predict_proba(isotonic_query, isotonic_probabilities, status)
    call isotonic_model%predict_proba_jvp(isotonic_query, isotonic_dot, &
        isotonic_probabilities, isotonic_probabilities_dot, status)
    call isotonic_model%predict_proba(isotonic_query + h*isotonic_dot, isotonic_plus, status)
    call isotonic_model%predict_proba(isotonic_query - h*isotonic_dot, isotonic_minus, status)
    call check(status_ok(status) .and. maxval(abs(sum(isotonic_probabilities, dim=2) - 1.0_dp)) < &
        1.0e-14_dp .and. maxval(abs(isotonic_probabilities_dot - &
        (isotonic_plus-isotonic_minus)/(2.0_dp*h))) < 2.0e-7_dp, &
        "isotonic interpolation and score JVP", failures)
    call isotonic_model%predict_proba_parameter_jvp(isotonic_query, [0.1_dp, 0.2_dp], &
        isotonic_probabilities, isotonic_probabilities_dot, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "isotonic parameter derivative refusal", failures)

    call unfitted%predict_proba(scores, probabilities, status)
    call check(.not. status_ok(status), "unfitted prediction refusal", failures)
    call sigmoid_model%fit(scores, labels, status, sample_weight=[1.0_dp, 1.0_dp, 1.0_dp, &
        1.0_dp, 0.0_dp, 0.0_dp, 0.0_dp, 0.0_dp])
    call check(.not. status_ok(status), "zero effective class weight refusal", failures)

    if (failures > 0) then
        write (error_unit, '(a,i0)') "FAIL probability calibration cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS probability calibration independent oracles"

contains

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "  FAIL [probability-calibration] "//description
        end if
    end subroutine check

end program test_probability_calibration
