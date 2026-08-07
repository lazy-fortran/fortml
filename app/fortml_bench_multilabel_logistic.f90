program fortml_bench_multilabel_logistic
    !! Correctness-gated multilabel-indicator logistic workload.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortml_multilabel_logistic_classifier, only: &
        multilabel_logistic_classifier_t
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer, parameter :: n_samples = 192, n_features = 4, n_labels = 3
    integer, parameter :: prediction_repetitions = 64
    real(dp) :: x(n_samples, n_features), probabilities(n_samples, n_labels)
    integer :: indicators(n_samples, n_labels), predicted(n_samples, n_labels)
    integer(int64) :: clock_start, clock_end, clock_rate
    real(dp) :: fit_seconds, predict_seconds
    character(len=1024) :: oracle_path
    integer :: environment_status, unit, i, j, repetition
    type(fortnum_status_t) :: status
    type(multilabel_logistic_classifier_t) :: model

    call get_environment_variable("FORTML_BENCH_MULTILABEL_ORACLE", oracle_path, &
        status=environment_status)
    if (environment_status /= 0 .or. len_trim(oracle_path) == 0) then
        error stop "FORTML_BENCH_MULTILABEL_ORACLE is required"
    end if
    call make_fixture(x, indicators)
    call system_clock(clock_start, clock_rate)
    call model%fit(x, indicators, status, l2=5.0e-2_dp, max_iterations=1000, &
        tolerance=1.0e-7_dp)
    call system_clock(clock_end)
    if (.not. status_ok(status)) error stop "multilabel benchmark fit failed"
    fit_seconds = real(clock_end-clock_start, dp)/real(clock_rate, dp)
    call model%predict_proba(x, probabilities, status)
    call model%predict(x, predicted, status)
    if (.not. status_ok(status)) error stop "multilabel benchmark prediction failed"
    call system_clock(clock_start, clock_rate)
    do repetition = 1, prediction_repetitions
        call model%predict_proba(x, probabilities, status)
    end do
    call system_clock(clock_end)
    predict_seconds = real(clock_end-clock_start, dp)/real(clock_rate, dp) &
        /real(prediction_repetitions, dp)
    write (*, '(a,i0,a,i0,a,i0,a,es24.16)') "multilabel_logistic_fit,", &
        n_samples, ",", n_features, ",", n_labels, ",", fit_seconds
    write (*, '(a,i0,a,i0,a,i0,a,es24.16)') "multilabel_logistic_predict,", &
        n_samples, ",", n_features, ",", n_labels, ",", predict_seconds

    open (newunit=unit, file=trim(oracle_path), status="replace", action="write")
    write (unit, '(a)') "quantity,row,column,value"
    do i = 1, n_samples
        do j = 1, n_labels
            write (unit, '(a,i0,a,i0,a,i0)') "label,", i, ",", j, ",", indicators(i, j)
            write (unit, '(a,i0,a,i0,a,i0)') "prediction,", i, ",", j, ",", &
                predicted(i, j)
            write (unit, '(a,i0,a,i0,a,es24.16)') "probability,", i, ",", j, ",", &
                probabilities(i, j)
        end do
    end do
    close (unit)

contains

    subroutine make_fixture(x, indicators)
        real(dp), intent(out) :: x(:, :)
        integer, intent(out) :: indicators(:, :)
        real(dp) :: phase, score
        integer :: i, j

        do i = 1, size(x, 1)
            phase = real(i, dp)
            do j = 1, size(x, 2)
                x(i, j) = sin(0.021_dp*phase + 0.083_dp*real(j, dp)) + &
                    0.15_dp*cos(0.011_dp*phase*real(j, dp))
            end do
            score = 0.8_dp*x(i, 1) - 0.45_dp*x(i, 2) + 0.15_dp*sin(0.13_dp*phase)
            indicators(i, 1) = merge(1, 0, score > 0.0_dp)
            score = -0.35_dp*x(i, 1) + 0.7_dp*x(i, 3) - &
                0.1_dp*cos(0.09_dp*phase + 0.3_dp)
            indicators(i, 2) = merge(1, 0, score > 0.0_dp)
            score = 0.3_dp*x(i, 2) + 0.55_dp*x(i, 4) + &
                0.2_dp*sin(0.07_dp*phase + 0.5_dp)
            indicators(i, 3) = merge(1, 0, score > 0.0_dp)
        end do
    end subroutine make_fixture

end program fortml_bench_multilabel_logistic
