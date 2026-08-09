program fortml_bench_weighted_ols
    !! Release smoke workload for weighted ordinary least squares.
    use, intrinsic :: iso_fortran_env, only: real64, int64
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortml_weighted_ols, only: weighted_ols_regression_t
    implicit none
    integer, parameter :: dp = real64
    type(weighted_ols_regression_t) :: model
    type(fortnum_status_t) :: status
    type(fortml_device_t) :: cuda
    real(dp) :: x(64, 3), y(64, 2), weights(64), prediction(64, 2)
    real(dp) :: x_dot(64, 3), theta_dot(8), y_dot(64, 2)
    real(dp), allocatable :: theta(:)
    integer(int64) :: started, finished, rate
    real(dp) :: elapsed
    integer :: i

    do i = 1, 64
        x(i, 1) = real(i-1, dp)/16.0_dp
        x(i, 2) = sin(x(i, 1))
        x(i, 3) = cos(0.5_dp*x(i, 1))
        y(i, 1) = 0.3_dp + 1.1_dp*x(i, 1) - 0.2_dp*x(i, 2) + 0.5_dp*x(i, 3)
        y(i, 2) = -0.4_dp + 0.7_dp*x(i, 1) + 0.4_dp*x(i, 2) - 0.6_dp*x(i, 3)
        weights(i) = 0.5_dp + real(mod(i, 5), dp)/5.0_dp
    end do
    x_dot = 0.0_dp
    theta_dot = 0.0_dp
    call system_clock(started, rate)
    call model%fit(x, y, status, sample_weight=weights)
    if (.not. status_ok(status)) error stop "weighted OLS release fit failed"
    call model%predict(x, prediction, status)
    if (.not. status_ok(status)) error stop "weighted OLS release prediction failed"
    call model%predict_jvp(x, theta_dot, x_dot, prediction, y_dot, status)
    if (.not. status_ok(status)) error stop "weighted OLS release JVP failed"
    call system_clock(finished)
    elapsed = real(finished-started, dp)/real(rate, dp)
    theta = model%parameters()
    write (*, '(a,es24.16)') "weighted_ols_fit_predict_seconds,", elapsed
    write (*, '(a,es24.16)') "weighted_ols_prediction_mean,", &
        sum(prediction)/real(size(prediction), dp)
    write (*, '(a,es24.16)') "weighted_ols_parameter_checksum,", sum(theta)
    write (*, '(a,i0)') "weighted_ols_parameter_count,", size(theta)
    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call model%predict_device(cuda, x, prediction, status)
    if (status%code /= FORTNUM_NOT_IMPLEMENTED) error stop &
        "weighted OLS CUDA contract changed"
    write (*, '(a)') "weighted_ols_cuda,unavailable"
end program fortml_bench_weighted_ols
