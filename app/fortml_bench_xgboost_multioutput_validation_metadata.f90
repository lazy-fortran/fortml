program fortml_bench_xgboost_multioutput_validation_metadata
    use, intrinsic :: iso_fortran_env, only: dp => real64
    use fortml_xgboost_multioutput, only: xgboost_multioutput_t, &
        lightgbm_multioutput_t
    use fortml_xgboost, only: xgboost_options_t
    use fortml_lightgbm, only: lightgbm_options_t
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    type(xgboost_multioutput_t) :: xgb
    type(lightgbm_multioutput_t) :: lgb
    type(xgboost_options_t) :: xo
    type(lightgbm_options_t) :: lo
    type(fortnum_status_t) :: status
    real(dp) :: x(8, 1), targets(8, 2), validation_targets(8, 2), elapsed
    real(dp) :: best_loss(2)
    integer, allocatable :: best_iteration(:)
    logical, allocatable :: stopped(:)
    integer :: i
    integer :: start_clock, end_clock, clock_rate

    x(:, 1) = real([(i, i = 0, 7)], dp)
    targets(:, 1) = [0.0_dp, 0.0_dp, 0.0_dp, 0.0_dp, 10.0_dp, 10.0_dp, 10.0_dp, 10.0_dp]
    targets(:, 2) = [1.0_dp, 1.0_dp, 1.0_dp, 1.0_dp, 5.0_dp, 5.0_dp, 5.0_dp, 5.0_dp]
    validation_targets(:, 1) = 10.0_dp - targets(:, 1)
    validation_targets(:, 2) = 6.0_dp - targets(:, 2)

    xo%n_estimators = 8
    xo%max_depth = 1
    xo%learning_rate = 1.0_dp
    xo%l2 = 1.0_dp
    xo%min_child_weight = 0.0_dp
    xo%early_stopping_rounds = 2
    call system_clock(start_clock, clock_rate)
    call xgb%fit(x, targets, status, xo, validation_x=x, &
        validation_targets=validation_targets)
    call system_clock(end_clock)
    if (.not. status_ok(status)) error stop "XGBoost multi-output validation fit failed"
    elapsed = real(end_clock - start_clock, dp)/real(clock_rate, dp)
    best_iteration = xgb%best_iteration()
    best_loss = xgb%best_validation_loss()
    stopped = xgb%early_stopped()
    write (*, '(a,2(i0,1x),2(es24.16,1x),2(l1,1x),es24.16)') &
        "xgb_multi_validation ", best_iteration(1), best_iteration(2), &
        best_loss(1), best_loss(2), stopped(1), stopped(2), elapsed

    lo%n_estimators = 6
    lo%num_leaves = 2
    lo%min_data_in_leaf = 1
    lo%max_bin = 16
    lo%learning_rate = 1.0_dp
    lo%l2 = 1.0_dp
    lo%early_stopping_rounds = 2
    call system_clock(start_clock)
    call lgb%fit(x, targets, status, lo, validation_x=x, &
        validation_targets=validation_targets)
    call system_clock(end_clock)
    if (.not. status_ok(status)) error stop "LightGBM multi-output validation fit failed"
    elapsed = real(end_clock - start_clock, dp)/real(clock_rate, dp)
    best_iteration = lgb%best_iteration()
    best_loss = lgb%best_validation_loss()
    stopped = lgb%early_stopped()
    write (*, '(a,2(i0,1x),2(es24.16,1x),2(l1,1x),es24.16)') &
        "lgb_multi_validation ", best_iteration(1), best_iteration(2), &
        best_loss(1), best_loss(2), stopped(1), stopped(2), elapsed
end program fortml_bench_xgboost_multioutput_validation_metadata
