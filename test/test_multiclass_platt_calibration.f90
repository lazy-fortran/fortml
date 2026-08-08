program test_multiclass_platt_calibration
    !! Independent smooth-product and transactional multiclass Platt oracles.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortml_probability_calibration, only: &
        multiclass_probability_calibrator_t, probability_calibration_options_t, &
        CALIBRATION_SIGMOID
    implicit none

    type(multiclass_probability_calibrator_t) :: model, repeated, invalid
    type(probability_calibration_options_t) :: options
    type(fortnum_status_t) :: status
    type(fortml_device_t) :: cuda
    real(dp) :: scores(9, 3), scores_dot(9, 3), probabilities(9, 3)
    real(dp) :: probabilities_dot(9, 3), probabilities_plus(9, 3), probabilities_minus(9, 3)
    real(dp) :: probabilities_bar(9, 3), scores_bar(9, 3)
    real(dp), allocatable :: parameters(:), parameters_plus(:), parameters_minus(:)
    real(dp), allocatable :: parameter_dot(:), parameter_bar(:), saved(:)
    real(dp) :: weights(9), lhs, rhs, h
    integer :: labels(9), predicted(9), classes(3), failures

    scores = reshape([ &
        3.0_dp, 0.2_dp, -1.2_dp, 2.1_dp, 1.0_dp, -0.7_dp, -0.8_dp, 2.2_dp, 0.1_dp, &
        -0.4_dp, 2.8_dp, 0.4_dp, 0.2_dp, -0.9_dp, 3.1_dp, 1.2_dp, -0.5_dp, 2.3_dp, &
        2.4_dp, -0.6_dp, 0.3_dp, -0.2_dp, 1.8_dp, 0.9_dp, 0.7_dp, 0.1_dp, 2.0_dp], &
        shape(scores), order=[2, 1])
    labels = [7, 7, 42, 42, 99, 99, 7, 42, 99]
    weights = [1.0_dp, 0.7_dp, 1.2_dp, 0.8_dp, 1.4_dp, 0.9_dp, 1.1_dp, 0.6_dp, 1.3_dp]
    scores_dot = reshape([ &
        0.1_dp, -0.2_dp, 0.3_dp, -0.2_dp, 0.4_dp, -0.1_dp, 0.3_dp, 0.1_dp, -0.2_dp, &
        -0.1_dp, 0.2_dp, 0.4_dp, 0.2_dp, -0.3_dp, 0.1_dp, -0.4_dp, 0.2_dp, 0.3_dp, &
        0.3_dp, -0.1_dp, 0.2_dp, 0.1_dp, -0.2_dp, 0.4_dp, 0.2_dp, 0.3_dp, -0.1_dp], &
        shape(scores_dot), order=[2, 1])
    probabilities_bar = reshape([ &
        0.2_dp, -0.1_dp, 0.3_dp, -0.2_dp, 0.4_dp, -0.3_dp, 0.1_dp, 0.5_dp, -0.2_dp, &
        0.3_dp, -0.4_dp, 0.2_dp, 0.5_dp, -0.1_dp, 0.6_dp, -0.2_dp, 0.2_dp, -0.4_dp, &
        0.3_dp, -0.2_dp, 0.4_dp, -0.1_dp, 0.2_dp, 0.5_dp, -0.2_dp, 0.3_dp, 0.1_dp], &
        shape(probabilities_bar), order=[2, 1])
    h = 1.0e-6_dp
    failures = 0
    options = probability_calibration_options_t(method=CALIBRATION_SIGMOID, &
        max_iterations=500, tolerance=1.0e-11_dp, damping=1.0_dp, l2=0.05_dp)

    call model%fit(scores, labels, status, options=options, sample_weight=weights)
    call check(status_ok(status) .and. model%fitted(), "weighted multiclass Platt fit", failures)
    classes = model%classes()
    call check(all(classes == [7, 42, 99]) .and. model%method() == CALIBRATION_SIGMOID .and. &
        model%parameter_count() == 6, "sorted classes and packed map metadata", failures)
    parameters = model%parameters()
    call check(size(parameters) == 6 .and. all(parameters == parameters), &
        "interleaved slope/intercept parameter vector", failures)
    call model%predict_proba(scores, probabilities, status)
    call check(status_ok(status) .and. maxval(abs(sum(probabilities, dim=2)-1.0_dp)) < 2.0e-14_dp .and. &
        minval(probabilities) > 0.0_dp, "sigmoid simplex probability oracle", failures)
    call model%predict(scores, predicted, status)
    call check(status_ok(status) .and. all(predicted == [7, 7, 42, 42, 99, 99, 7, 42, 99]), &
        "strict first-class tie prediction policy", failures)

    call model%predict_proba_jvp(scores, scores_dot, probabilities, probabilities_dot, status)
    call model%predict_proba(scores+h*scores_dot, probabilities_plus, status)
    call model%predict_proba(scores-h*scores_dot, probabilities_minus, status)
    call check(status_ok(status) .and. maxval(abs(probabilities_dot- &
        (probabilities_plus-probabilities_minus)/(2.0_dp*h))) < 2.0e-6_dp, &
        "sigmoid score JVP central difference", failures)
    call model%predict_proba_vjp(scores, probabilities_bar, scores_bar, status)
    lhs = sum(probabilities_bar*probabilities_dot)
    rhs = sum(scores_bar*scores_dot)
    call check(status_ok(status) .and. abs(lhs-rhs) < 2.0e-6_dp, &
        "sigmoid score VJP adjoint identity", failures)

    allocate(parameter_dot(size(parameters)), parameter_bar(size(parameters)))
    parameter_dot = [0.03_dp, -0.02_dp, 0.04_dp, 0.01_dp, -0.02_dp, 0.05_dp]
    call model%predict_proba_parameter_jvp(scores, parameter_dot, probabilities, &
        probabilities_dot, status)
    parameters_plus = parameters+h*parameter_dot
    parameters_minus = parameters-h*parameter_dot
    call model%set_parameters(parameters_plus, status)
    call model%predict_proba(scores, probabilities_plus, status)
    call model%set_parameters(parameters_minus, status)
    call model%predict_proba(scores, probabilities_minus, status)
    call model%set_parameters(parameters, status)
    call check(status_ok(status) .and. maxval(abs(probabilities_dot- &
        (probabilities_plus-probabilities_minus)/(2.0_dp*h))) < 2.0e-6_dp, &
        "sigmoid parameter JVP central difference", failures)
    call model%predict_proba_parameter_vjp(scores, probabilities_bar, parameter_bar, status)
    lhs = sum(probabilities_bar*probabilities_dot)
    rhs = sum(parameter_bar*parameter_dot)
    call check(status_ok(status) .and. abs(lhs-rhs) < 2.0e-6_dp, &
        "sigmoid parameter VJP adjoint identity", failures)

    call repeated%fit(scores, labels, status, options=options, sample_weight=weights)
    call repeated%predict_proba(scores, probabilities_plus, status)
    call check(status_ok(status) .and. maxval(abs(probabilities_plus-probabilities)) < 2.0e-14_dp, &
        "deterministic weighted Platt fit", failures)
    allocate(saved(size(probabilities, 1)*size(probabilities, 2)))
    saved = reshape(probabilities, [size(saved)])
    call invalid%fit(scores(:, :2), labels, status, options=options)
    call check(.not. status_ok(status), "invalid-column transactional refusal", failures)
    call model%predict_proba(scores, probabilities_plus, status)
    call check(status_ok(status) .and. maxval(abs(reshape(probabilities_plus, [size(saved)])-saved)) < &
        2.0e-14_dp, "failed fit leaves fitted model unchanged", failures)

    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call model%predict_proba_device(cuda, scores, probabilities_plus, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED .and. &
        .not. model%device_supported(FORTML_DEVICE_CUDA), "multiclass Platt CUDA refusal", failures)

    if (failures > 0) then
        write (error_unit, '(a,i0)') "FAIL multiclass Platt cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS multiclass Platt independent oracles"

contains

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures+1
            write (error_unit, '(a)') "  FAIL [multiclass-platt] "//description
        end if
    end subroutine check

end program test_multiclass_platt_calibration
