program fortml_bench_calibrated_logistic_cv
    !! Leakage-safe binary logistic OOF calibration workload.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortml_calibrated_logistic_classifier, only: &
        calibrated_logistic_classifier_t, calibrated_logistic_classifier_options_t
    use fortml_probability_calibration, only: CALIBRATION_TEMPERATURE
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer, parameter :: n_samples = 96, n_features = 2, prediction_repetitions = 128
    real(dp) :: x(n_samples, n_features), probabilities(n_samples, 2)
    integer :: labels(n_samples), predicted(n_samples)
    real(dp), allocatable :: parameters(:)
    type(calibrated_logistic_classifier_t) :: model
    type(calibrated_logistic_classifier_options_t) :: options
    type(fortnum_status_t) :: status
    integer(int64) :: clock_start, clock_end, clock_rate
    real(dp) :: fit_seconds, predict_seconds
    character(len=1024) :: oracle_path
    integer :: environment_status, unit, i, repetition

    call get_environment_variable("FORTML_BENCH_CALIBRATED_LOGISTIC_ORACLE", oracle_path, &
        status=environment_status)
    if (environment_status /= 0 .or. len_trim(oracle_path) == 0) then
        error stop "FORTML_BENCH_CALIBRATED_LOGISTIC_ORACLE is required"
    end if
    call make_fixture(x, labels)
    options = calibrated_logistic_classifier_options_t()
    options%l2 = 1.0_dp
    options%max_iterations = 1000
    options%tolerance = 1.0e-8_dp
    options%cv_folds = 3
    options%cv_shuffle = .false.
    options%cv_seed = 29
    options%calibration%method = CALIBRATION_TEMPERATURE
    options%calibration%max_iterations = 300
    options%calibration%tolerance = 1.0e-10_dp
    options%calibration%l2 = 1.0e-6_dp

    call system_clock(clock_start, clock_rate)
    call model%fit(x, labels, status, options=options)
    call system_clock(clock_end)
    if (.not. status_ok(status)) then
        write (*, '(a,i0,2a)') "calibrated logistic OOF fit failed ", status%code, ": ", trim(status%msg)
        error stop 1
    end if
    fit_seconds = real(clock_end - clock_start, dp)/real(clock_rate, dp)
    call model%predict_proba(x, probabilities, status)
    call model%predict(x, predicted, status)
    if (.not. status_ok(status)) error stop "calibrated logistic OOF prediction failed"
    call system_clock(clock_start, clock_rate)
    do repetition = 1, prediction_repetitions
        call model%predict_proba(x, probabilities, status)
    end do
    call system_clock(clock_end)
    predict_seconds = real(clock_end - clock_start, dp)/real(clock_rate, dp)/ &
        real(prediction_repetitions, dp)
    parameters = model%parameters()

    open (newunit=unit, file=trim(oracle_path), status="replace", action="write")
    write (unit, '(a)') "method,quantity,row,column,value"
    do i = 1, size(parameters)
        write (unit, '(a,i0,a,es26.17e3)') &
            "temperature,parameter,", i, ",1,", parameters(i)
    end do
    do i = 1, n_samples
        write (unit, '(a,i0,a,es26.17e3)') &
            "temperature,label,", i, ",1,", real(labels(i), dp)
        write (unit, '(a,i0,a,es26.17e3)') &
            "temperature,prediction,", i, ",1,", real(predicted(i), dp)
        write (unit, '(a,i0,a,es26.17e3)') &
            "temperature,probability,", i, ",1,", probabilities(i, 1)
        write (unit, '(a,i0,a,es26.17e3)') &
            "temperature,probability,", i, ",2,", probabilities(i, 2)
    end do
    write (unit, '(a)') "temperature,diagnostic,1,1,"// &
        trim(real_to_text(real(model%cv_folds(), dp)))
    write (unit, '(a)') "temperature,diagnostic,2,1,"// &
        trim(real_to_text(model%oof_log_loss()))
    write (unit, '(a)') "temperature,diagnostic,3,1,"// &
        trim(real_to_text(model%calibrated_oof_log_loss()))
    close (unit)
    write (*, '(a,es24.16)') "calibrated_logistic_cv_fit,temperature,", fit_seconds
    write (*, '(a,es24.16)') "calibrated_logistic_cv_predict,temperature,", predict_seconds

contains

    subroutine make_fixture(features, target)
        real(dp), intent(out) :: features(:, :)
        integer, intent(out) :: target(:)
        real(dp), parameter :: first(12) = [ &
            -2.0_dp, -1.7_dp, -1.4_dp, -1.1_dp, -0.7_dp, -0.3_dp, &
            0.2_dp,  0.5_dp,  0.8_dp,  1.1_dp,  1.5_dp,  1.9_dp]
        real(dp), parameter :: second(12) = [ &
            -1.4_dp, -0.8_dp, -1.1_dp, -0.5_dp, -0.2_dp,  0.1_dp, &
            -0.1_dp,  0.3_dp,  0.7_dp,  0.9_dp,  1.2_dp,  1.6_dp]
        integer :: i
        real(dp) :: phase

        do i = 1, size(target)
            phase = real(mod(i - 1, 12), dp)
            features(i, 1) = first(int(phase) + 1)
            features(i, 2) = second(int(phase) + 1)
            if (phase >= 6.0_dp) then
                target(i) = 42
            else
                target(i) = -3
            end if
        end do
    end subroutine make_fixture

    function real_to_text(value) result(text)
        real(dp), intent(in) :: value
        character(len=64) :: text

        write (text, '(es26.17e3)') value
    end function real_to_text

end program fortml_bench_calibrated_logistic_cv
