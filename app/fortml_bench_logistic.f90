program fortml_bench_logistic
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortml_logistic_regression, only: logistic_regression_t
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer, parameter :: n_samples = 1024
    integer, parameter :: n_features = 8
    integer, parameter :: repetitions = 5
    real(dp) :: x(n_samples, n_features), probabilities(n_samples, 2)
    real(dp) :: scores(n_samples), true_score(n_samples)
    integer :: labels(n_samples), predicted(n_samples)
    real(dp) :: fit_elapsed, predict_elapsed, accuracy
    integer(int64) :: clock_start, clock_end, clock_rate
    integer :: i, j, k
    type(logistic_regression_t) :: model
    type(fortnum_status_t) :: status

    do j = 1, n_features
        do i = 1, n_samples
            x(i, j) = sin(0.013_dp*real(i, dp) + 0.071_dp*real(j, dp)) + &
                0.2_dp*cos(0.009_dp*real(i*j, dp))
        end do
    end do
    do i = 1, n_samples
        true_score(i) = -0.15_dp + sum(x(i, :)*sin(0.17_dp*real([(j, j=1,n_features)], dp)))
        if (true_score(i) >= 0.0_dp) then
            labels(i) = 7
        else
            labels(i) = -3
        end if
    end do

    call model%fit(x, labels, status, l2=0.1_dp, max_iterations=1000, tolerance=1.0e-6_dp)
    if (.not. status_ok(status)) then
        write (*, '(a)') "logistic fit status: "//trim(status%msg)
        error stop "logistic benchmark fit failed"
    end if
    call model%predict(x, predicted, status)
    call model%predict_proba(x, probabilities, status)
    if (.not. status_ok(status)) error stop "logistic benchmark prediction failed"
    accuracy = real(count(predicted == labels), dp)/real(n_samples, dp)
    if (accuracy < 0.95_dp .or. maxval(abs(sum(probabilities, dim=2) - 1.0_dp)) > 1.0e-14_dp) then
        error stop "logistic benchmark correctness oracle failed"
    end if
    call write_oracle(accuracy, maxval(abs(sum(probabilities, dim=2) - 1.0_dp)))
    if (oracle_only_requested()) stop

    call system_clock(clock_start, clock_rate)
    do k = 1, repetitions
        call model%fit(x, labels, status, l2=0.1_dp, max_iterations=1000, tolerance=1.0e-6_dp)
    end do
    call system_clock(clock_end)
    fit_elapsed = real(clock_end - clock_start, dp)/real(clock_rate, dp)/real(repetitions, dp)

    call system_clock(clock_start, clock_rate)
    do k = 1, repetitions*8
        call model%predict(x, predicted, status)
        call model%predict_proba(x, probabilities, status)
    end do
    call system_clock(clock_end)
    predict_elapsed = real(clock_end - clock_start, dp)/real(clock_rate, dp)/real(repetitions*8, dp)

    write (*, '(a,i0,a,i0,a,i0,a,es24.16)') &
        "logistic_fit,", n_samples, ",", n_features, ",", repetitions, ",", fit_elapsed
    write (*, '(a,i0,a,i0,a,i0,a,es24.16)') &
        "logistic_predict,", n_samples, ",", n_features, ",", repetitions*8, ",", predict_elapsed
    write (*, '(a,es24.16)') "logistic_accuracy,", accuracy

contains

    subroutine write_oracle(value, normalization_error)
        real(dp), intent(in) :: value, normalization_error
        character(len=1024) :: path
        integer :: unit, environment_status

        call get_environment_variable("FORTML_BENCH_ORACLE", path, status=environment_status)
        if (environment_status /= 0 .or. len_trim(path) == 0) return
        open (newunit=unit, file=trim(path), status="replace", action="write")
        write (unit, '(a,es26.17e3)') "accuracy,", value
        write (unit, '(a,es26.17e3)') "normalization_error,", normalization_error
        close (unit)
    end subroutine write_oracle

    logical function oracle_only_requested()
        character(len=16) :: value
        integer :: environment_status

        call get_environment_variable("FORTML_BENCH_ORACLE_ONLY", value, status=environment_status)
        oracle_only_requested = environment_status == 0 .and. trim(value) == "1"
    end function oracle_only_requested

end program fortml_bench_logistic
