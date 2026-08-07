program fortml_bench_xgboost_robust
    !! Correctness-gated Huber and quantile XGBoost-style workloads.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortml_xgboost, only: xgboost_t, xgboost_options_t
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    implicit none

    real(dp) :: x(4, 1), target(4), prediction(4), expected(4)
    real(dp) :: huber_prediction(4), quantile_prediction(4)
    type(xgboost_t) :: huber_model, quantile_model
    type(xgboost_options_t) :: options
    type(fortnum_status_t) :: status
    type(fortml_device_t) :: cuda
    integer(int64) :: started, finished, rate
    real(dp) :: fit_seconds, predict_seconds

    x(:, 1) = [0.0_dp, 1.0_dp, 2.0_dp, 3.0_dp]
    target = [0.0_dp, 0.0_dp, 10.0_dp, 10.0_dp]
    options%n_estimators = 1
    options%max_depth = 1
    options%learning_rate = 1.0_dp
    options%l2 = 1.0_dp
    options%min_child_weight = 0.0_dp
    options%huber_delta = 1.0_dp
    expected = [3.0_dp, 3.0_dp, 7.0_dp, 7.0_dp]
    call system_clock(started, rate)
    call huber_model%fit_huber(x, target, status, options)
    call system_clock(finished)
    if (.not. status_ok(status)) error stop "Huber benchmark fit failed"
    fit_seconds = real(finished - started, dp)/real(rate, dp)
    call huber_model%predict(x, huber_prediction, status)
    if (.not. status_ok(status)) error stop "Huber benchmark prediction failed"
    call system_clock(started, rate)
    call huber_model%predict(x, prediction, status)
    call system_clock(finished)
    if (.not. status_ok(status)) error stop "Huber benchmark timing failed"
    predict_seconds = real(finished - started, dp)/real(rate, dp)
    write (*, '(a,es24.16)') "xgb_huber_fit_seconds,", fit_seconds
    write (*, '(a,es24.16)') "xgb_huber_predict_seconds,", predict_seconds
    write (*, '(a,es24.16)') "xgb_huber_max_error,", maxval(abs(huber_prediction - expected))

    options = xgboost_options_t()
    options%n_estimators = 1
    options%max_depth = 1
    options%learning_rate = 1.0_dp
    options%l2 = 1.0_dp
    options%min_child_weight = 0.0_dp
    options%quantile_alpha = 0.5_dp
    expected = [-1.0_dp, -1.0_dp, 1.0_dp, 1.0_dp]
    call system_clock(started, rate)
    call quantile_model%fit_quantile(x, target, status, options)
    call system_clock(finished)
    if (.not. status_ok(status)) error stop "quantile benchmark fit failed"
    fit_seconds = real(finished - started, dp)/real(rate, dp)
    call quantile_model%predict(x, quantile_prediction, status)
    if (.not. status_ok(status)) error stop "quantile benchmark prediction failed"
    call system_clock(started, rate)
    call quantile_model%predict(x, prediction, status)
    call system_clock(finished)
    if (.not. status_ok(status)) error stop "quantile benchmark timing failed"
    predict_seconds = real(finished - started, dp)/real(rate, dp)
    write (*, '(a,es24.16)') "xgb_quantile_fit_seconds,", fit_seconds
    write (*, '(a,es24.16)') "xgb_quantile_predict_seconds,", predict_seconds
    write (*, '(a,es24.16)') "xgb_quantile_max_error,", &
        maxval(abs(quantile_prediction - expected))

    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call huber_model%predict_device(cuda, x, prediction, status)
    if (status%code /= FORTNUM_NOT_IMPLEMENTED) then
        error stop "robust XGBoost CUDA contract changed unexpectedly"
    end if
    write (*, '(a)') "xgb_robust_cuda,unavailable"
end program fortml_bench_xgboost_robust
