program fortml_bench_radius_neighbors_multioutput
    !! Correctness-gated multi-output radius-neighbors workload.
    use, intrinsic :: iso_fortran_env, only: real64, int64
    use fortml_radius_neighbors_multioutput_regression, only: &
        radius_neighbors_multioutput_regressor_t
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    implicit none

    integer, parameter :: dp = real64
    integer, parameter :: n_samples = 320, n_features = 3, n_outputs = 2, n_query = 8
    real(dp) :: x(n_samples, n_features), y(n_samples, n_outputs)
    real(dp) :: query(n_query, n_features), predictions(n_query, n_outputs)
    integer(int64) :: started, finished, rate
    real(dp) :: fit_seconds, predict_seconds
    type(radius_neighbors_multioutput_regressor_t) :: model
    type(fortml_device_t) :: cuda
    type(fortnum_status_t) :: status
    integer :: i, repetitions

    do i = 1, n_samples
        x(i, 1) = -2.0_dp + 4.0_dp*real(mod(i - 1, 80), dp)/79.0_dp
        x(i, 2) = sin(0.13_dp*real(i, dp))
        x(i, 3) = cos(0.07_dp*real(i, dp))
        y(i, 1) = x(i, 1) + 0.1_dp*x(i, 2)
        y(i, 2) = x(i, 1)**2 + x(i, 3)
    end do
    query(:, 1) = [-1.5_dp, -1.0_dp, -0.5_dp, 0.0_dp, 0.5_dp, 1.0_dp, 1.5_dp, 3.5_dp]
    query(:, 2) = 0.0_dp
    query(:, 3) = 1.0_dp

    call system_clock(started, rate)
    call model%fit(x, y, status, radius=0.4_dp, outlier_value=[0.0_dp, 0.0_dp])
    call system_clock(finished)
    if (.not. status_ok(status)) error stop "multi-output radius benchmark fit failed"
    fit_seconds = real(finished - started, dp)/real(rate, dp)
    call model%predict(query, predictions, status)
    if (.not. status_ok(status)) error stop "multi-output radius benchmark prediction failed"
    if (maxval(abs(predictions(1:7, 1))) > 2.0e+1_dp .or. &
        maxval(abs(predictions(1:7, 2))) > 2.0e+1_dp) then
        error stop "multi-output radius benchmark prediction sanity failed"
    end if
    call system_clock(started, rate)
    do repetitions = 1, 128
        call model%predict(query(1:7, :), predictions(1:7, :), status)
    end do
    call system_clock(finished)
    predict_seconds = real(finished - started, dp)/real(rate, dp)/128.0_dp

    write (*, '(a,es24.16)') "radius_multioutput_fit_seconds,", fit_seconds
    write (*, '(a,es24.16)') "radius_multioutput_predict_seconds,", predict_seconds
    write (*, '(a,es24.16)') "radius_multioutput_max_abs_prediction,", &
        maxval(abs(predictions(1:7, :)))
    do i = 1, 7
        write (*, '(a,i0,a,i0,a,es24.16)') "radius_multioutput_prediction,", i, &
            ",", 1, ",", predictions(i, 1)
        write (*, '(a,i0,a,i0,a,es24.16)') "radius_multioutput_prediction,", i, &
            ",", 2, ",", predictions(i, 2)
    end do

    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call model%predict_device(cuda, query(1:2, :), predictions(1:2, :), status)
    if (status%code /= FORTNUM_NOT_IMPLEMENTED) then
        error stop "multi-output radius CUDA contract changed unexpectedly"
    end if
    write (*, '(a)') "radius_multioutput_cuda,unavailable"
end program fortml_bench_radius_neighbors_multioutput
