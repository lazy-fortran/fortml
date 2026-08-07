program fortml_bench_probability_calibration
    !! Complete-array probability calibration workload.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortml_probability_calibration, only: &
        probability_calibrator_t, probability_calibration_options_t, &
        CALIBRATION_SIGMOID, CALIBRATION_ISOTONIC
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer, parameter :: n_samples = 256, prediction_repetitions = 128
    real(dp) :: scores(n_samples), probabilities(n_samples, 2)
    integer :: labels(n_samples), predicted(n_samples)
    type(probability_calibrator_t) :: models(2)
    type(probability_calibration_options_t) :: options
    type(fortnum_status_t) :: status
    integer(int64) :: clock_start, clock_end, clock_rate
    real(dp) :: fit_seconds(2), predict_seconds(2)
    character(len=1024) :: oracle_path
    integer :: environment_status, unit, i, method_index, repetition
    character(len=12), parameter :: method_name(2) = ["sigmoid     ", "isotonic    "]

    call get_environment_variable("FORTML_BENCH_CALIBRATION_ORACLE", oracle_path, &
        status=environment_status)
    if (environment_status /= 0 .or. len_trim(oracle_path) == 0) then
        error stop "FORTML_BENCH_CALIBRATION_ORACLE is required"
    end if
    call make_fixture(scores, labels)
    options = probability_calibration_options_t(method=CALIBRATION_SIGMOID, &
        max_iterations=500, tolerance=1.0e-10_dp, damping=1.0_dp, l2=0.1_dp)
    call system_clock(clock_start, clock_rate)
    call models(1)%fit(scores, labels, status, options=options)
    call system_clock(clock_end)
    if (.not. status_ok(status)) error stop "calibration sigmoid benchmark fit failed"
    fit_seconds(1) = real(clock_end-clock_start, dp)/real(clock_rate, dp)
    options%method = CALIBRATION_ISOTONIC
    call system_clock(clock_start, clock_rate)
    call models(2)%fit(scores, labels, status, options=options)
    call system_clock(clock_end)
    if (.not. status_ok(status)) error stop "calibration isotonic benchmark fit failed"
    fit_seconds(2) = real(clock_end-clock_start, dp)/real(clock_rate, dp)

    open (newunit=unit, file=trim(oracle_path), status="replace", action="write")
    write (unit, '(a)') "method,quantity,row,column,value"
    do method_index = 1, 2
        call models(method_index)%predict_proba(scores, probabilities, status)
        call models(method_index)%predict(scores, predicted, status)
        if (.not. status_ok(status)) error stop "calibration benchmark prediction failed"
        call system_clock(clock_start, clock_rate)
        do repetition = 1, prediction_repetitions
            call models(method_index)%predict_proba(scores, probabilities, status)
        end do
        call system_clock(clock_end)
        predict_seconds(method_index) = real(clock_end-clock_start, dp)/ &
            real(clock_rate, dp)/real(prediction_repetitions, dp)
        write (*, '(a,a,a,es24.16)') "probability_calibration_fit,", &
            trim(method_name(method_index)), ",", fit_seconds(method_index)
        write (*, '(a,a,a,es24.16)') "probability_calibration_predict,", &
            trim(method_name(method_index)), ",", predict_seconds(method_index)
        do i = 1, n_samples
            write (unit, '(a,a,i0,a,es26.17e3)') trim(method_name(method_index)), &
                ",label,", i, ",1,", real(labels(i), dp)
            write (unit, '(a,a,i0,a,es26.17e3)') trim(method_name(method_index)), &
                ",prediction,", i, ",1,", real(predicted(i), dp)
            write (unit, '(a,a,i0,a,es26.17e3)') trim(method_name(method_index)), &
                ",probability,", i, ",1,", probabilities(i, 1)
            write (unit, '(a,a,i0,a,es26.17e3)') trim(method_name(method_index)), &
                ",probability,", i, ",2,", probabilities(i, 2)
        end do
    end do
    close (unit)

contains

    subroutine make_fixture(scores, labels)
        real(dp), intent(out) :: scores(:)
        integer, intent(out) :: labels(:)
        integer :: i
        real(dp) :: phase

        do i = 1, size(scores)
            phase = real(i, dp)
            scores(i) = 1.5_dp*sin(0.071_dp*phase) + &
                0.35_dp*cos(0.013_dp*phase) + 0.002_dp*phase
            if (scores(i) + 0.2_dp*sin(0.17_dp*phase) > 0.0_dp) then
                labels(i) = 42
            else
                labels(i) = 10
            end if
        end do
    end subroutine make_fixture

end program fortml_bench_probability_calibration
