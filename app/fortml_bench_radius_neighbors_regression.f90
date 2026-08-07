program fortml_bench_radius_neighbors_regression
    !! Release workload for dense radius-neighbor scalar regression.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortml_radius_neighbors_regression, only: &
        radius_neighbors_regressor_t, RADIUS_REGRESSION_WEIGHTS_DISTANCE
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer, parameter :: n_samples = 96, n_features = 2, n_query = 6
    integer, parameter :: prediction_repetitions = 256
    real(dp) :: x(n_samples, n_features), targets(n_samples)
    real(dp) :: query(n_query, n_features), predictions(n_query)
    real(dp) :: sample_weight(n_samples)
    integer(int64) :: clock_start, clock_end, clock_rate
    real(dp) :: fit_seconds, predict_seconds
    character(len=1024) :: output_path
    integer :: environment_status, unit, i, repetition
    type(fortnum_status_t) :: status
    type(radius_neighbors_regressor_t) :: model

    call get_environment_variable("FORTML_BENCH_RADIUS_REGRESSION_OUTPUT", &
        output_path, status=environment_status)
    if (environment_status /= 0 .or. len_trim(output_path) == 0) then
        error stop "FORTML_BENCH_RADIUS_REGRESSION_OUTPUT is required"
    end if
    call make_fixture(x, targets, sample_weight, query)
    call system_clock(clock_start, clock_rate)
    call model%fit(x, targets, status, radius=0.38_dp, &
        weights=RADIUS_REGRESSION_WEIGHTS_DISTANCE, sample_weight=sample_weight, &
        outlier_value=0.0_dp)
    call system_clock(clock_end)
    if (.not. status_ok(status)) error stop "radius regression fit failed"
    fit_seconds = real(clock_end-clock_start, dp)/real(clock_rate, dp)
    call model%predict(query, predictions, status)
    if (.not. status_ok(status)) error stop "radius regression prediction failed"
    call system_clock(clock_start, clock_rate)
    do repetition = 1, prediction_repetitions
        call model%predict(query, predictions, status)
    end do
    call system_clock(clock_end)
    predict_seconds = real(clock_end-clock_start, dp)/real(clock_rate, dp) &
        /real(prediction_repetitions, dp)
    write (*, '(a,i0,a,i0,a,es24.16)') &
        "radius_neighbors_regression_fit,", n_samples, ",", n_features, ",", &
        fit_seconds
    write (*, '(a,i0,a,i0,a,es24.16)') &
        "radius_neighbors_regression_predict,", n_query, ",", n_features, ",", &
        predict_seconds

    open (newunit=unit, file=trim(output_path), status="replace", action="write")
    write (unit, '(a)') "quantity,row,column,value"
    do i = 1, n_query
        write (unit, '(a,i0,a,es24.16)') "prediction,", i, ",1,", &
            predictions(i)
    end do
    close (unit)

contains

    subroutine make_fixture(x, targets, sample_weight, query)
        real(dp), intent(out) :: x(:, :), targets(:), sample_weight(:), query(:, :)
        integer :: i
        real(dp) :: a, b
        do i = 1, size(x, 1)
            a = -1.2_dp + 2.4_dp*real(i-1, dp)/real(size(x, 1)-1, dp)
            b = sin(0.19_dp*real(i, dp))
            x(i, :) = [a, b]
            targets(i) = 1.4_dp*a - 0.7_dp*b + 0.2_dp*a*a
            sample_weight(i) = 0.8_dp + 0.4_dp*real(mod(i, 5), dp)/4.0_dp
        end do
        query(1, :) = [-1.05_dp, 0.0_dp]
        query(2, :) = [-0.15_dp, 0.35_dp]
        query(3, :) = [0.25_dp, -0.45_dp]
        query(4, :) = [0.85_dp, 0.15_dp]
        query(5, :) = [1.18_dp, -0.2_dp]
        query(6, :) = [0.0_dp, 1.4_dp]
    end subroutine make_fixture

end program fortml_bench_radius_neighbors_regression
