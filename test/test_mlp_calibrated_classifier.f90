program test_mlp_calibrated_classifier
    !! Independent behavioral checks for calibrated neural heads.
    use, intrinsic :: iso_fortran_env, only: real64, error_unit
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortml_mlp_classifier, only: mlp_classifier_options_t
    use fortml_mlp_calibrated_classifier, only: &
        mlp_calibrated_classifier_t, mlp_calibrated_classifier_options_t, &
        MLP_CALIBRATION_SIGMOID, MLP_CALIBRATION_ISOTONIC, &
        MLP_CALIBRATION_TEMPERATURE
    implicit none

    integer, parameter :: dp = real64
    type(mlp_calibrated_classifier_t) :: binary, binary_repeat, isotonic, multiclass
    type(mlp_calibrated_classifier_options_t) :: options
    type(fortnum_status_t) :: status
    type(fortml_device_t) :: cuda
    real(dp) :: xb(8, 2), probabilities(8, 2), probabilities_dot(8, 2)
    real(dp) :: probabilities_plus(8, 2), probabilities_minus(8, 2), probabilities_bar(8, 2)
    real(dp) :: x_dot(8, 2), lhs, rhs, h
    real(dp), allocatable :: theta(:), theta_dot(:), theta_plus(:), theta_minus(:), theta_bar(:), x_bar(:, :)
    integer :: labels_b(8), predicted(8), classes(2), failures, i
    real(dp) :: xm(9, 2), pm(9, 3)
    integer :: labels_m(9), classes_m(3)

    failures = 0
    h = 1.0e-5_dp
    xb = reshape([ &
        -2.0_dp, -1.0_dp, -1.2_dp, -0.7_dp, -0.4_dp, -0.2_dp, &
         0.1_dp,  0.2_dp,  0.5_dp,  0.4_dp,  0.9_dp,  0.8_dp, &
         1.4_dp,  1.1_dp,  2.0_dp,  1.6_dp], shape(xb))
    labels_b = [42, 42, 42, 42, -3, -3, -3, -3]
    x_dot = reshape([ &
        0.03_dp, -0.02_dp, 0.02_dp, 0.01_dp, -0.01_dp, 0.03_dp, &
        0.01_dp, -0.04_dp, 0.02_dp, 0.02_dp, -0.03_dp, 0.01_dp, &
        0.04_dp, -0.01_dp, -0.02_dp, 0.03_dp], shape(x_dot))
    options = mlp_calibrated_classifier_options_t()
    options%classifier = mlp_classifier_options_t(max_epochs=80, batch_size=0, &
        initialization_seed=13, learning_rate=0.05_dp, tolerance=1.0e-8_dp)
    options%calibration%method = MLP_CALIBRATION_TEMPERATURE
    options%calibration%max_iterations = 300
    options%calibration%tolerance = 1.0e-10_dp
    options%calibration%l2 = 1.0e-6_dp
    call binary%fit(xb, labels_b, status, options=options)
    if (.not. status_ok(status)) write(error_unit,'(a,i0,2a)') 'binary fit status ', status%code, ': ', trim(status%msg)
    call check(status_ok(status) .and. binary%fitted(), "binary temperature fit", failures)
    classes = binary%classes()
    call check(all(classes == [-3, 42]) .and. binary%parameter_count() > 0 .and. &
        binary%temperature() > 0.0_dp, "sorted classes and positive temperature", failures)
    call binary%predict_proba(xb, probabilities, status)
    call binary%predict(xb, predicted, status)
    call check(status_ok(status) .and. maxval(abs(sum(probabilities, dim=2)-1.0_dp)) < 1.0e-13_dp .and. &
        all(predicted(:4) == 42) .and. all(predicted(5:) == -3), &
        "binary calibrated probabilities and labels", failures)

    theta = binary%parameters()
    allocate(theta_dot(size(theta)), theta_plus(size(theta)), theta_minus(size(theta)), &
        theta_bar(size(theta)), x_bar(size(xb, 1), size(xb, 2)))
    theta_dot = [(0.01_dp*sin(real(i, dp)), i=1,size(theta))]
    call binary%predict_proba_jvp(xb, theta_dot, x_dot, probabilities, probabilities_dot, status)
    theta_plus = theta + h*theta_dot
    theta_minus = theta - h*theta_dot
    call binary%set_parameters(theta_plus, status)
    call binary%predict_proba(xb+h*x_dot, probabilities_plus, status)
    call binary%set_parameters(theta_minus, status)
    call binary%predict_proba(xb-h*x_dot, probabilities_minus, status)
    call binary%set_parameters(theta, status)
    call check(status_ok(status) .and. maxval(abs(probabilities_dot - &
        (probabilities_plus-probabilities_minus)/(2.0_dp*h))) < 3.0e-5_dp, &
        "binary temperature joint JVP", failures)
    probabilities_bar = reshape([ &
        0.1_dp, -0.2_dp, 0.3_dp, -0.4_dp, 0.5_dp, -0.1_dp, 0.2_dp, -0.3_dp, &
        -0.2_dp, 0.1_dp, -0.4_dp, 0.3_dp, -0.1_dp, 0.5_dp, -0.3_dp, 0.2_dp], &
        shape(probabilities_bar))
    call binary%predict_proba_vjp(xb, probabilities_bar, theta_bar, x_bar, status)
    lhs = sum(probabilities_bar*probabilities_dot)
    rhs = sum(theta_bar*theta_dot) + sum(x_bar*x_dot)
    call check(status_ok(status) .and. abs(lhs-rhs) < 3.0e-5_dp, &
        "binary temperature VJP adjoint identity", failures)
    call binary_repeat%fit(xb, labels_b, status, options=options)
    call check(status_ok(status) .and. maxval(abs(binary_repeat%parameters()-theta)) < 1.0e-12_dp, &
        "deterministic binary fit", failures)

    options%calibration%method = MLP_CALIBRATION_ISOTONIC
    options%calibration%max_iterations = 100
    call isotonic%fit(xb, labels_b, status, options=options)
    if (.not. status_ok(status)) write(error_unit,'(a,i0,2a)') 'isotonic fit status ', status%code, ': ', trim(status%msg)
    call check(status_ok(status) .and. isotonic%fitted(), "binary isotonic fit", failures)
    call isotonic%predict_proba(xb, probabilities, status)
    call isotonic%predict_proba_jvp(xb, theta_dot(:isotonic%parameter_count()), x_dot, &
        probabilities, probabilities_dot, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, "isotonic active-set JVP refusal", failures)

    xm = reshape([ &
        -2.0_dp, -1.5_dp, -1.3_dp, -0.8_dp, -0.4_dp, -0.2_dp, &
         0.0_dp,  0.1_dp,  0.3_dp,  0.2_dp,  0.5_dp,  0.4_dp, &
         0.8_dp,  0.9_dp,  1.1_dp,  1.2_dp,  1.5_dp,  1.7_dp], shape(xm))
    labels_m = [100, 100, 100, 7, 7, 7, -4, -4, -4]
    options%calibration%method = MLP_CALIBRATION_TEMPERATURE
    options%classifier%max_epochs = 120
    options%classifier%learning_rate = 0.04_dp
    call multiclass%fit(xm, labels_m, status, options=options)
    if (.not. status_ok(status)) write(error_unit,'(a,i0,2a)') 'multiclass fit status ', status%code, ': ', trim(status%msg)
    call check(status_ok(status) .and. multiclass%fitted() .and. multiclass%class_count() == 3 .and. &
        multiclass%temperature() > 0.0_dp, "multiclass temperature fit", failures)
    call multiclass%predict_proba(xm, pm, status)
    classes_m = multiclass%classes()
    call check(status_ok(status) .and. all(classes_m == [-4, 7, 100]) .and. &
        maxval(abs(sum(pm, dim=2)-1.0_dp)) < 1.0e-13_dp, &
        "multiclass calibrated probabilities", failures)
    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call multiclass%predict_proba_device(cuda, xm, pm, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED .and. &
        .not. multiclass%device_supported(FORTML_DEVICE_CUDA), &
        "calibrated neural CUDA refusal", failures)

    if (failures > 0) then
        write (error_unit, '(a,i0)') "FAIL calibrated MLP cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS calibrated MLP independent oracles"

contains

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failures
        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "  FAIL [calibrated-mlp] "//description
        end if
    end subroutine check

end program test_mlp_calibrated_classifier
