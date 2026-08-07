program fortml_bench_linear_svr
    !! Correctness-gated weighted dense linear SVR workload.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortml_linear_svr, only: linear_svr_regression_t, SVR_LOSS_SQUARED_EPSILON
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer, parameter :: n_samples = 192, n_features = 4
    integer, parameter :: prediction_repetitions = 128
    real(dp) :: x(n_samples, n_features), targets(n_samples), prediction(n_samples)
    real(dp) :: sample_weight(n_samples)
    real(dp), allocatable :: packed(:)
    integer(int64) :: clock_start, clock_end, clock_rate
    real(dp) :: fit_seconds, predict_seconds
    character(len=1024) :: oracle_path
    integer :: environment_status, unit, i, j, repetition
    type(fortnum_status_t) :: status
    type(linear_svr_regression_t) :: model

    call get_environment_variable("FORTML_BENCH_LINEAR_SVR_ORACLE", oracle_path, &
        status=environment_status)
    if (environment_status /= 0 .or. len_trim(oracle_path) == 0) then
        error stop "FORTML_BENCH_LINEAR_SVR_ORACLE is required"
    end if
    call make_fixture(x, targets, sample_weight)
    call system_clock(clock_start, clock_rate)
    call model%fit(x, targets, status, l2=5.0e-2_dp, epsilon=8.0e-2_dp, &
        loss=SVR_LOSS_SQUARED_EPSILON, max_iterations=1500, &
        tolerance=1.0e-8_dp, sample_weight=sample_weight)
    call system_clock(clock_end)
    if (.not. status_ok(status)) then
        write (*, '(a)') "linear SVR benchmark fit status: "//trim(status%msg)
        error stop "linear SVR benchmark fit failed"
    end if
    fit_seconds = real(clock_end-clock_start, dp)/real(clock_rate, dp)
    call model%predict(x, prediction, status)
    if (.not. status_ok(status)) error stop "linear SVR benchmark prediction failed"
    packed = model%parameters()
    call system_clock(clock_start, clock_rate)
    do repetition = 1, prediction_repetitions
        call model%predict(x, prediction, status)
    end do
    call system_clock(clock_end)
    predict_seconds = real(clock_end-clock_start, dp)/real(clock_rate, dp) &
        /real(prediction_repetitions, dp)
    write (*, '(a,i0,a,i0,a,es24.16)') "linear_svr_fit,", n_samples, ",", &
        n_features, ",", fit_seconds
    write (*, '(a,i0,a,i0,a,es24.16)') "linear_svr_predict,", n_samples, ",", &
        n_features, ",", predict_seconds

    open (newunit=unit, file=trim(oracle_path), status="replace", action="write")
    write (unit, '(a)') "quantity,row,column,value"
    do i = 1, n_samples
        write (unit, '(a,i0,a,es24.16)') "target,", i, ",1,", targets(i)
        write (unit, '(a,i0,a,es24.16)') "prediction,", i, ",1,", prediction(i)
        write (unit, '(a,i0,a,es24.16)') "weight,", i, ",1,", sample_weight(i)
    end do
    do j = 1, model%parameter_count()
        write (unit, '(a,i0,a,es24.16)') "parameter,", j, ",1,", &
            packed(j)
    end do
    close (unit)

contains

    subroutine make_fixture(x, targets, sample_weight)
        real(dp), intent(out) :: x(:, :), targets(:), sample_weight(:)
        real(dp) :: phase, truth
        integer :: i, j

        do i = 1, size(x, 1)
            phase = real(i, dp)
            do j = 1, size(x, 2)
                x(i, j) = sin(0.017_dp*phase + 0.071_dp*real(j, dp)) + &
                    0.12_dp*cos(0.013_dp*phase*real(j, dp))
            end do
            truth = 0.8_dp*x(i, 1) - 0.45_dp*x(i, 2) + 0.25_dp*x(i, 3) - &
                0.15_dp*x(i, 4) + 0.10_dp*sin(0.11_dp*phase)
            targets(i) = truth + 0.08_dp*cos(0.07_dp*phase)
            sample_weight(i) = 0.75_dp + 0.5_dp*real(mod(i, 9), dp)/8.0_dp
        end do
    end subroutine make_fixture

end program fortml_bench_linear_svr
