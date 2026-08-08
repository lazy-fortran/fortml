program fortml_bench_xgboost_slice
    !! Release workload for prefix slicing of a fitted XGBoost ensemble.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortml_xgboost, only: xgboost_t, xgboost_options_t
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer, parameter :: n_samples = 8, n_features = 1
    real(dp) :: x(n_samples, n_features), target(n_samples)
    real(dp) :: staged(n_samples, 3), full_prediction(n_samples)
    real(dp) :: prefix_prediction(n_samples)
    integer(int64) :: clock_start, clock_end, clock_rate
    real(dp) :: elapsed_slice, elapsed_predict
    type(xgboost_t) :: source, prefix, invalid
    type(xgboost_options_t) :: options
    type(fortnum_status_t) :: status

    x(:, 1) = [0.0_dp, 1.0_dp, 2.0_dp, 3.0_dp, 4.0_dp, 5.0_dp, 6.0_dp, 7.0_dp]
    target = [0.0_dp, 0.0_dp, 0.0_dp, 0.0_dp, 10.0_dp, 10.0_dp, 10.0_dp, 10.0_dp]
    options%n_estimators = 3
    options%max_depth = 1
    options%min_samples_leaf = 1
    options%learning_rate = 0.5_dp
    options%l2 = 0.0_dp
    options%min_child_weight = 0.0_dp
    call source%fit_regression(x, target, status, options)
    if (.not. status_ok(status)) error stop "XGBoost slice fixture fit failed"
    call source%predict_staged(x, staged, status)
    if (.not. status_ok(status)) error stop "XGBoost slice staged prediction failed"
    call source%predict(x, full_prediction, status)
    if (.not. status_ok(status)) error stop "XGBoost slice full prediction failed"

    call system_clock(clock_start, clock_rate)
    call source%slice(2, prefix, status)
    call system_clock(clock_end)
    if (.not. status_ok(status)) error stop "XGBoost slice operation failed"
    elapsed_slice = real(clock_end - clock_start, dp)/real(clock_rate, dp)

    call system_clock(clock_start, clock_rate)
    call prefix%predict(x, prefix_prediction, status)
    call system_clock(clock_end)
    if (.not. status_ok(status)) error stop "XGBoost slice prediction failed"
    elapsed_predict = real(clock_end - clock_start, dp)/real(clock_rate, dp)

    call source%slice(0, invalid, status)
    write (*, '(a,i0)') "xgb_slice_invalid_status ", status%code
    write (*, '(a,*(1x,es24.16))') "xgb_slice_staged_1", staged(:, 1)
    write (*, '(a,*(1x,es24.16))') "xgb_slice_staged_2", staged(:, 2)
    write (*, '(a,*(1x,es24.16))') "xgb_slice_staged_3", staged(:, 3)
    write (*, '(a,*(1x,es24.16))') "xgb_slice_full_prediction", full_prediction
    write (*, '(a,*(1x,es24.16))') "xgb_slice_prefix_prediction", prefix_prediction
    write (*, '(a,es24.16)') "xgb_slice_seconds ", elapsed_slice
    write (*, '(a,es24.16)') "xgb_slice_predict_seconds ", elapsed_predict
end program fortml_bench_xgboost_slice
