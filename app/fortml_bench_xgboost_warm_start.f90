program fortml_bench_xgboost_warm_start
    !! Release workload for deterministic XGBoost suffix continuation.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortml_xgboost, only: xgboost_t, xgboost_options_t
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer, parameter :: n_samples = 8
    real(dp) :: x(n_samples, 1), target(n_samples)
    real(dp) :: warm_staged(n_samples, 4), full_staged(n_samples, 4)
    integer(int64) :: clock_start, clock_end, clock_rate
    real(dp) :: elapsed_warm
    type(xgboost_t) :: prefix, warm, full, invalid
    type(xgboost_options_t) :: prefix_options, full_options, bad_options
    type(fortnum_status_t) :: status

    x(:, 1) = [0.0_dp, 1.0_dp, 2.0_dp, 3.0_dp, 4.0_dp, 5.0_dp, 6.0_dp, 7.0_dp]
    target = [0.0_dp, 0.0_dp, 0.0_dp, 0.0_dp, 10.0_dp, 10.0_dp, 10.0_dp, 10.0_dp]
    prefix_options = xgboost_options_t()
    prefix_options%n_estimators = 2
    prefix_options%max_depth = 1
    prefix_options%learning_rate = 0.2_dp
    prefix_options%l2 = 1.0_dp
    full_options = prefix_options
    full_options%n_estimators = 4

    call prefix%fit_regression(x, target, status, prefix_options)
    if (.not. status_ok(status)) error stop "warm-start prefix fit failed"
    call full%fit_regression(x, target, status, full_options)
    if (.not. status_ok(status)) error stop "warm-start full fit failed"
    warm = prefix
    call system_clock(clock_start, clock_rate)
    call warm%fit_warm_start(x, target, status, full_options)
    call system_clock(clock_end)
    if (.not. status_ok(status)) error stop "warm-start continuation failed"
    elapsed_warm = real(clock_end - clock_start, dp)/real(clock_rate, dp)
    call warm%predict_staged_margin(x, warm_staged, status)
    if (.not. status_ok(status)) error stop "warm-start staged prediction failed"
    call full%predict_staged_margin(x, full_staged, status)
    if (.not. status_ok(status)) error stop "full staged prediction failed"

    bad_options = full_options
    bad_options%n_estimators = 2
    call warm%fit_warm_start(x, target, status, bad_options)
    write (*, '(a,i0)') "xgb_warm_invalid_target_status ", status%code
    bad_options = full_options
    bad_options%learning_rate = 0.3_dp
    call warm%fit_warm_start(x, target, status, bad_options)
    write (*, '(a,i0)') "xgb_warm_changed_control_status ", status%code
    call invalid%fit_warm_start(x, target, status, full_options)
    write (*, '(a,i0)') "xgb_warm_unfitted_status ", status%code
    write (*, '(a,i0)') "xgb_warm_estimator_count ", warm%estimator_count()
    write (*, '(a,i0)') "xgb_warm_requested_count ", warm%requested_estimator_count()
    write (*, '(a,*(1x,es24.16))') "xgb_warm_staged_4", warm_staged(:, 4)
    write (*, '(a,*(1x,es24.16))') "xgb_full_staged_4", full_staged(:, 4)
    write (*, '(a,es24.16)') "xgb_warm_seconds ", elapsed_warm
end program fortml_bench_xgboost_warm_start
