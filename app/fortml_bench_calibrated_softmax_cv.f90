program fortml_bench_calibrated_softmax_cv
    !! Leakage-safe multiclass softmax OOF calibration workload.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortml_calibrated_softmax_classifier, only: &
        calibrated_softmax_classifier_t, calibrated_softmax_classifier_options_t
    use fortml_probability_calibration, only: CALIBRATION_SIGMOID, CALIBRATION_ISOTONIC, &
        CALIBRATION_TEMPERATURE
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer, parameter :: n_samples = 96, n_features = 2, n_classes = 3
    integer, parameter :: prediction_repetitions = 128
    real(dp) :: x(n_samples, n_features), probabilities(n_samples, n_classes)
    integer :: labels(n_samples), predicted(n_samples)
    real(dp), allocatable :: parameters(:)
    type(calibrated_softmax_classifier_t) :: model
    type(calibrated_softmax_classifier_options_t) :: options
    type(fortnum_status_t) :: status
    integer(int64) :: clock_start, clock_end, clock_rate
    real(dp) :: fit_seconds, predict_seconds
    character(len=1024) :: oracle_path
    character(len=32) :: method_name
    integer :: environment_status, unit, i, j, repetition

    call get_environment_variable("FORTML_BENCH_CALIBRATED_SOFTMAX_ORACLE", oracle_path, &
        status=environment_status)
    if (environment_status /= 0 .or. len_trim(oracle_path) == 0) then
        error stop "FORTML_BENCH_CALIBRATED_SOFTMAX_ORACLE is required"
    end if
    method_name = "temperature"
    call get_environment_variable("FORTML_BENCH_CALIBRATED_SOFTMAX_METHOD", method_name, &
        status=environment_status)
    if (environment_status /= 0 .or. len_trim(method_name) == 0) method_name = "temperature"
    call make_fixture(x, labels)
    options = calibrated_softmax_classifier_options_t()
    options%l2 = 0.2_dp
    options%max_iterations = 500
    options%tolerance = 1.0e-8_dp
    options%cv_folds = 3
    options%cv_shuffle = .true.
    options%cv_seed = 31
    select case (trim(adjustl(method_name)))
    case ("temperature")
        options%calibration%method = CALIBRATION_TEMPERATURE
    case ("sigmoid", "platt")
        method_name = "sigmoid"
        options%calibration%method = CALIBRATION_SIGMOID
    case ("isotonic")
        options%calibration%method = CALIBRATION_ISOTONIC
        options%calibration%max_iterations = 1
        options%calibration%l2 = 0.0_dp
    case default
        error stop "FORTML_BENCH_CALIBRATED_SOFTMAX_METHOD must be temperature, sigmoid, or isotonic"
    end select
    options%calibration%max_iterations = 500
    options%calibration%tolerance = 1.0e-10_dp
    options%calibration%l2 = 0.05_dp

    call system_clock(clock_start, clock_rate)
    call model%fit(x, labels, status, options=options)
    call system_clock(clock_end)
    if (.not. status_ok(status)) then
        write (*, '(a,i0,2a)') "calibrated softmax OOF fit failed ", status%code, ": ", &
            trim(status%msg)
        error stop 1
    end if
    fit_seconds = real(clock_end-clock_start, dp)/real(clock_rate, dp)
    call model%predict_proba(x, probabilities, status)
    call model%predict(x, predicted, status)
    if (.not. status_ok(status)) error stop "calibrated softmax prediction failed"
    call system_clock(clock_start, clock_rate)
    do repetition = 1, prediction_repetitions
        call model%predict_proba(x, probabilities, status)
    end do
    call system_clock(clock_end)
    predict_seconds = real(clock_end-clock_start, dp)/real(clock_rate, dp)/ &
        real(prediction_repetitions, dp)
    parameters = model%parameters()

    open (newunit=unit, file=trim(oracle_path), status="replace", action="write")
    write (unit, '(a)') "quantity,row,column,value"
    do j = 1, size(parameters)
        write (unit, '(a,i0,a,es26.17e3)') "parameter,", j, ",1,", parameters(j)
    end do
    do i = 1, n_samples
        write (unit, '(a,i0,a,i0)') "label,", i, ",1,", labels(i)
        write (unit, '(a,i0,a,i0)') "prediction,", i, ",1,", predicted(i)
        do j = 1, n_classes
            write (unit, '(a,i0,a,i0,a,es26.17e3)') &
                "probability,", i, ",", j, ",", probabilities(i, j)
        end do
    end do
    write (unit, '(a,i0,a,i0,a,es26.17e3)') &
        "oof_log_loss,", 1, ",", 1, ",", model%oof_log_loss()
    write (unit, '(a,i0,a,i0,a,es26.17e3)') &
        "calibrated_oof_log_loss,", 1, ",", 1, ",", model%calibrated_oof_log_loss()
    close (unit)
    write (*, '(a,a,a,es24.16)') "calibrated_softmax_cv_fit,", trim(method_name), ",", fit_seconds
    write (*, '(a,a,a,es24.16)') "calibrated_softmax_cv_predict,", trim(method_name), ",", predict_seconds

contains

    subroutine make_fixture(features, target)
        real(dp), intent(out) :: features(:, :)
        integer, intent(out) :: target(:)
        integer :: i, class_index
        real(dp) :: phase

        do i = 1, size(target)
            class_index = mod(i - 1, n_classes) + 1
            phase = real(i, dp)
            select case (class_index)
            case (1)
                features(i, :) = [2.0_dp + 0.1_dp*sin(phase), 0.1_dp*cos(phase)]
                target(i) = -4
            case (2)
                features(i, :) = [0.1_dp*cos(phase), 2.0_dp + 0.1_dp*sin(phase)]
                target(i) = 17
            case default
                features(i, :) = [-2.0_dp + 0.1_dp*sin(phase), -2.0_dp + 0.1_dp*cos(phase)]
                target(i) = 91
            end select
        end do
    end subroutine make_fixture

end program fortml_bench_calibrated_softmax_cv
