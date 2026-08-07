program fortml_bench_ordinal_logistic
    !! Correctness-gated weighted ordinal cumulative-logit workload.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortml_ordinal_logistic_classifier, only: ordinal_logistic_classifier_t
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer, parameter :: n_samples = 192, n_features = 4, n_classes = 4
    integer, parameter :: prediction_repetitions = 64
    real(dp) :: x(n_samples, n_features), probabilities(n_samples, n_classes)
    real(dp) :: sample_weight(n_samples)
    integer :: labels(n_samples), predicted(n_samples), classes(n_classes)
    integer(int64) :: clock_start, clock_end, clock_rate
    real(dp) :: fit_seconds, predict_seconds
    character(len=1024) :: oracle_path
    integer :: environment_status, unit, i, j, repetition
    type(fortnum_status_t) :: status
    type(ordinal_logistic_classifier_t) :: model

    call get_environment_variable("FORTML_BENCH_ORDINAL_ORACLE", oracle_path, &
        status=environment_status)
    if (environment_status /= 0 .or. len_trim(oracle_path) == 0) then
        error stop "FORTML_BENCH_ORDINAL_ORACLE is required"
    end if
    call make_fixture(x, labels, sample_weight)
    call system_clock(clock_start, clock_rate)
    call model%fit(x, labels, status, l2=5.0e-2_dp, max_iterations=3000, &
        tolerance=1.0e-7_dp, sample_weight=sample_weight)
    call system_clock(clock_end)
    if (.not. status_ok(status)) then
        write (*, '(a)') "ordinal benchmark fit status: "//trim(status%msg)
        error stop "ordinal benchmark fit failed"
    end if
    fit_seconds = real(clock_end-clock_start, dp)/real(clock_rate, dp)
    call model%predict_proba(x, probabilities, status)
    call model%predict(x, predicted, status)
    classes = model%classes()
    if (.not. status_ok(status)) error stop "ordinal benchmark prediction failed"
    call system_clock(clock_start, clock_rate)
    do repetition = 1, prediction_repetitions
        call model%predict_proba(x, probabilities, status)
    end do
    call system_clock(clock_end)
    predict_seconds = real(clock_end-clock_start, dp)/real(clock_rate, dp) &
        /real(prediction_repetitions, dp)
    write (*, '(a,i0,a,i0,a,i0,a,es24.16)') "ordinal_logistic_fit,", &
        n_samples, ",", n_features, ",", n_classes, ",", fit_seconds
    write (*, '(a,i0,a,i0,a,i0,a,es24.16)') "ordinal_logistic_predict,", &
        n_samples, ",", n_features, ",", n_classes, ",", predict_seconds

    open (newunit=unit, file=trim(oracle_path), status="replace", action="write")
    write (unit, '(a)') "quantity,row,column,value"
    do i = 1, n_samples
        write (unit, '(a,i0,a,i0)') "label,", i, ",1,", labels(i)
        write (unit, '(a,i0,a,i0)') "prediction,", i, ",1,", predicted(i)
        do j = 1, n_classes
            write (unit, '(a,i0,a,i0,a,es24.16)') "probability,", i, ",", j, ",", &
                probabilities(i, j)
        end do
    end do
    do j = 1, n_classes
        write (unit, '(a,i0,a,i0)') "class,", j, ",", classes(j)
    end do
    close (unit)

contains

    subroutine make_fixture(x, labels, sample_weight)
        real(dp), intent(out) :: x(:, :), sample_weight(:)
        integer, intent(out) :: labels(:)
        real(dp) :: phase, score
        integer :: i, j
        do i = 1, size(x, 1)
            phase = real(i, dp)
            do j = 1, size(x, 2)
                x(i, j) = sin(0.021_dp*phase + 0.083_dp*real(j, dp)) + &
                    0.15_dp*cos(0.011_dp*phase*real(j, dp))
            end do
            score = 0.7_dp*x(i, 1) - 0.3_dp*x(i, 2) + &
                0.2_dp*sin(0.13_dp*phase)
            if (score < -0.25_dp) then
                labels(i) = -9
            else if (score < 0.05_dp) then
                labels(i) = 4
            else if (score < 0.30_dp) then
                labels(i) = 13
            else
                labels(i) = 42
            end if
            sample_weight(i) = 0.75_dp + 0.5_dp*real(mod(i, 7), dp)/6.0_dp
        end do
    end subroutine make_fixture

end program fortml_bench_ordinal_logistic
