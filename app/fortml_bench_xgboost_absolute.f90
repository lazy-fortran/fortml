program fortml_bench_xgboost_absolute
    !! Correctness-gated absolute-deviation XGBoost-style workload.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortml_xgboost, only: xgboost_t, xgboost_options_t
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    implicit none

    real(dp) :: x(4, 1), target(4), prediction(4), expected(4)
    type(xgboost_t) :: model
    type(xgboost_options_t) :: options
    type(fortnum_status_t) :: status
    type(fortml_device_t) :: cuda
    integer(int64) :: started, finished, rate
    real(dp) :: fit_seconds, predict_seconds

    x(:, 1) = [0.0_dp, 1.0_dp, 2.0_dp, 3.0_dp]
    target = [0.0_dp, 10.0_dp, 20.0_dp, 30.0_dp]
    expected = [9.0_dp, 9.0_dp, 12.0_dp, 12.0_dp]
    options%n_estimators = 1
    options%max_depth = 1
    options%learning_rate = 1.0_dp
    options%l2 = 1.0_dp
    options%min_child_weight = 0.0_dp

    call system_clock(started, rate)
    call model%fit_absolute(x, target, status, options)
    call system_clock(finished)
    if (.not. status_ok(status)) error stop "absolute benchmark fit failed"
    fit_seconds = real(finished - started, dp)/real(rate, dp)
    call model%predict(x, prediction, status)
    if (.not. status_ok(status)) error stop "absolute benchmark prediction failed"
    call system_clock(started, rate)
    call model%predict(x, prediction, status)
    call system_clock(finished)
    if (.not. status_ok(status)) error stop "absolute benchmark timing failed"
    predict_seconds = real(finished - started, dp)/real(rate, dp)
    write (*, '(a,es24.16)') "xgb_absolute_fit_seconds,", fit_seconds
    write (*, '(a,es24.16)') "xgb_absolute_predict_seconds,", predict_seconds
    write (*, '(a,es24.16)') "xgb_absolute_max_error,", &
        maxval(abs(prediction - expected))

    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call model%predict_device(cuda, x, prediction, status)
    if (status%code /= FORTNUM_NOT_IMPLEMENTED) then
        error stop "absolute XGBoost CUDA contract changed unexpectedly"
    end if
    write (*, '(a)') "xgb_absolute_cuda,unavailable"
end program fortml_bench_xgboost_absolute
