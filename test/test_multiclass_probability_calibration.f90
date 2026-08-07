program test_multiclass_probability_calibration
    !! Independent softmax-temperature, product, and refusal oracles.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortml_probability_calibration, only: &
        multiclass_probability_calibrator_t, probability_calibration_options_t, &
        probability_calibration_state_t, CALIBRATION_TEMPERATURE, CALIBRATION_SIGMOID
    implicit none

    type(multiclass_probability_calibrator_t) :: model, repeated, invalid, unfitted
    type(probability_calibration_options_t) :: options
    type(probability_calibration_state_t) :: state
    type(fortnum_status_t) :: status
    type(fortml_device_t) :: cuda
    real(dp) :: scores(6, 3), scores_dot(6, 3), probabilities(6, 3)
    real(dp) :: probabilities_dot(6, 3), probabilities_plus(6, 3)
    real(dp) :: probabilities_minus(6, 3), probabilities_bar(6, 3), scores_bar(6, 3)
    real(dp) :: parameter_probabilities_dot(6, 3), parameters(1), parameters_plus(1)
    real(dp) :: parameters_minus(1), parameters_bar(1), parameter_dot(1)
    real(dp) :: lhs, rhs, h
    integer :: labels(6), predicted(6), classes(3), failures

    scores = reshape([ &
        3.0_dp, 0.0_dp, -1.0_dp, &
        2.0_dp, 1.0_dp, -1.0_dp, &
        -1.0_dp, 2.0_dp, 0.0_dp, &
        -1.0_dp, 3.0_dp, 0.0_dp, &
        0.0_dp, -1.0_dp, 3.0_dp, &
        1.0_dp, -1.0_dp, 2.0_dp], shape(scores), order=[2, 1])
    scores_dot = reshape([ &
        0.1_dp, -0.2_dp, 0.3_dp, &
        -0.2_dp, 0.4_dp, -0.1_dp, &
        0.3_dp, 0.1_dp, -0.2_dp, &
        -0.1_dp, 0.2_dp, 0.4_dp, &
        0.2_dp, -0.3_dp, 0.1_dp, &
        -0.4_dp, 0.2_dp, 0.3_dp], shape(scores_dot), order=[2, 1])
    labels = [7, 7, 42, 42, 99, 99]
    probabilities_bar = reshape([ &
        0.2_dp, -0.1_dp, 0.3_dp, -0.2_dp, 0.4_dp, -0.3_dp, &
        0.1_dp, 0.5_dp, -0.2_dp, 0.3_dp, -0.4_dp, 0.2_dp, &
        0.5_dp, -0.1_dp, 0.6_dp, -0.2_dp, 0.2_dp, -0.4_dp], &
        shape(probabilities_bar), order=[2, 1])
    h = 1.0e-6_dp
    failures = 0
    options = probability_calibration_options_t(method=CALIBRATION_TEMPERATURE, &
        max_iterations=500, tolerance=1.0e-11_dp, damping=1.0_dp, l2=0.05_dp)

    call model%fit(scores, labels, status, options=options, state=state)
    call check(status_ok(status), "multiclass temperature fit status", failures)
    call check(model%fitted() .and. state%converged, &
        "multiclass temperature convergence", failures)
    classes = model%classes()
    call check(all(classes == [7, 42, 99]) .and. model%parameter_count() == 1, &
        "sorted class and parameter metadata", failures)
    parameters = model%parameters()
    call check(parameters(1) > 0.0_dp .and. model%method() == CALIBRATION_TEMPERATURE, &
        "positive temperature metadata", failures)

    call model%predict_proba(scores, probabilities, status)
    call check(status_ok(status), "multiclass probability status", failures)
    call check(maxval(abs(probabilities - softmax_oracle(scores, parameters(1)))) < 2.0e-14_dp, &
        "multiclass softmax probability oracle", failures)
    call check(maxval(abs(sum(probabilities, dim=2) - 1.0_dp)) < 2.0e-14_dp, &
        "multiclass probability normalization", failures)
    call model%predict(scores, predicted, status)
    call check(status_ok(status) .and. all(predicted == labels), &
        "multiclass sorted prediction oracle", failures)

    call model%predict_proba_jvp(scores, scores_dot, probabilities, probabilities_dot, status)
    call model%predict_proba(scores + h*scores_dot, probabilities_plus, status)
    call model%predict_proba(scores - h*scores_dot, probabilities_minus, status)
    call check(status_ok(status) .and. maxval(abs(probabilities_dot - &
        (probabilities_plus-probabilities_minus)/(2.0_dp*h))) < 2.0e-6_dp, &
        "multiclass score JVP finite difference", failures)
    call model%predict_proba_vjp(scores, probabilities_bar, scores_bar, status)
    lhs = sum(probabilities_bar*probabilities_dot)
    rhs = sum(scores_bar*scores_dot)
    call check(status_ok(status) .and. abs(lhs-rhs) < 2.0e-6_dp, &
        "multiclass score VJP adjoint identity", failures)

    parameter_dot = [0.07_dp]
    call model%predict_proba_parameter_jvp(scores, parameter_dot, probabilities, &
        parameter_probabilities_dot, status)
    parameters_plus = parameters + h*parameter_dot
    call model%set_parameters(parameters_plus, status)
    call model%predict_proba(scores, probabilities_plus, status)
    parameters_minus = parameters - h*parameter_dot
    call model%set_parameters(parameters_minus, status)
    call model%predict_proba(scores, probabilities_minus, status)
    call model%set_parameters(parameters, status)
    call check(status_ok(status) .and. maxval(abs(parameter_probabilities_dot - &
        (probabilities_plus-probabilities_minus)/(2.0_dp*h))) < 2.0e-6_dp, &
        "multiclass temperature JVP finite difference", failures)
    call model%predict_proba_parameter_vjp(scores, probabilities_bar, parameters_bar, status)
    lhs = sum(probabilities_bar*parameter_probabilities_dot)
    rhs = sum(parameters_bar*parameter_dot)
    call check(status_ok(status) .and. abs(lhs-rhs) < 2.0e-6_dp, &
        "multiclass temperature VJP adjoint identity", failures)

    call repeated%fit(scores, labels, status, options=options)
    call repeated%predict_proba(scores, probabilities_plus, status)
    call check(status_ok(status) .and. maxval(abs(probabilities_plus-probabilities)) < 2.0e-14_dp, &
        "deterministic multiclass calibration", failures)
    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call model%predict_proba_device(cuda, scores, probabilities_plus, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED .and. &
        .not. model%device_supported(FORTML_DEVICE_CUDA), &
        "multiclass CUDA refusal", failures)
    call model%set_parameters([-1.0_dp], status)
    call check(.not. status_ok(status), "nonpositive multiclass temperature refusal", failures)
    call invalid%fit(scores, labels, status, options=probability_calibration_options_t( &
        method=CALIBRATION_SIGMOID))
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "multiclass sigmoid policy refusal", failures)
    call invalid%fit(scores(:, :2), labels, status, options=options)
    call check(.not. status_ok(status), "logit-class shape refusal", failures)
    call invalid%fit(scores, labels, status, options=options, &
        sample_weight=[1.0_dp, 1.0_dp, 1.0_dp, 1.0_dp, 0.0_dp, 0.0_dp])
    call check(.not. status_ok(status), "zero weighted class refusal", failures)
    call unfitted%predict_proba(scores, probabilities, status)
    call check(.not. status_ok(status), "unfitted multiclass prediction refusal", failures)

    if (failures > 0) then
        write (error_unit, '(a,i0)') "FAIL multiclass calibration cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS multiclass calibration independent oracles"

contains

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "  FAIL [multiclass-calibration] "//description
        end if
    end subroutine check

    pure function softmax_oracle(logits, temperature) result(probabilities)
        real(dp), intent(in) :: logits(:, :), temperature
        real(dp) :: probabilities(size(logits, 1), size(logits, 2))
        real(dp) :: shifted(size(logits, 2)), maximum, normalizer
        integer :: i

        do i = 1, size(logits, 1)
            shifted = logits(i, :)/temperature
            maximum = maxval(shifted)
            shifted = exp(shifted-maximum)
            normalizer = sum(shifted)
            probabilities(i, :) = shifted/normalizer
        end do
    end function softmax_oracle

end program test_multiclass_probability_calibration
