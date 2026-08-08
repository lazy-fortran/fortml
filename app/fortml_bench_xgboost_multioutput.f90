program fortml_bench_xgboost_multioutput
    !! Release workload for deterministic multi-output XGBoost/LightGBM lanes.
    use, intrinsic :: iso_fortran_env, only: real64
    use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_quiet_nan
    use fortml_xgboost_multioutput, only: xgboost_multioutput_t, &
        lightgbm_multioutput_t
    use fortml_xgboost, only: xgboost_options_t
    use fortml_lightgbm, only: lightgbm_options_t
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortnum_status, only: fortnum_status_t
    implicit none

    integer, parameter :: dp = real64
    type(xgboost_multioutput_t) :: xgb
    type(lightgbm_multioutput_t) :: lgb
    type(xgboost_options_t) :: xo
    type(lightgbm_options_t) :: lo
    type(fortnum_status_t) :: status
    type(fortml_device_t) :: cuda
    real(dp) :: x(4, 1), targets(4, 2), values(4, 2), values_dot(4, 2)
    real(dp) :: staged(4, 1, 2), x_dot(4, 1), x_bar(4, 1), output_bar(4, 2)
    real(dp) :: parameter_dot(6), parameter_bar(6), before(4, 2), bad_targets(4, 2)
    real(dp) :: start, finish
    integer :: fit_status, predict_status, stage_status, jvp_status, vjp_status

    x(:, 1) = [0.0_dp, 1.0_dp, 2.0_dp, 3.0_dp]
    targets(:, 1) = [0.0_dp, 0.0_dp, 10.0_dp, 10.0_dp]
    targets(:, 2) = [1.0_dp, 1.0_dp, 5.0_dp, 5.0_dp]
    x_dot(:, 1) = 0.25_dp
    output_bar(:, 1) = [1.0_dp, 2.0_dp, 3.0_dp, 4.0_dp]
    output_bar(:, 2) = [2.0_dp, 3.0_dp, 4.0_dp, 5.0_dp]
    parameter_dot = [1.0_dp, 2.0_dp, 3.0_dp, 4.0_dp, 5.0_dp, 6.0_dp]
    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.

    xo%n_estimators = 1
    xo%max_depth = 1
    xo%min_child_weight = 0.0_dp
    xo%l2 = 1.0_dp
    xo%learning_rate = 1.0_dp
    call cpu_time(start)
    call xgb%fit(x, targets, status, xo)
    call cpu_time(finish)
    fit_status = status%code
    write(*,'(a,1x,es24.16)') 'xgb_fit_seconds', finish-start
    call xgb%predict(x, values, status)
    predict_status = status%code
    write(*,'(a,*(es24.16,1x))') 'xgb_predict', values
    call xgb%predict_staged_margin(x, staged, status)
    stage_status = status%code
    write(*,'(a,*(es24.16,1x))') 'xgb_staged', staged
    call xgb%predict_leaf_jvp(x, parameter_dot, values, values_dot, status)
    jvp_status = status%code
    write(*,'(a,*(es24.16,1x))') 'xgb_leaf_jvp', values_dot
    call xgb%predict_leaf_vjp(x, output_bar, parameter_bar, status)
    vjp_status = status%code
    write(*,'(a,*(es24.16,1x))') 'xgb_leaf_vjp', parameter_bar
    write(*,'(a,1x,i0)') 'xgb_fit_status', fit_status
    write(*,'(a,1x,i0)') 'xgb_predict_status', predict_status
    write(*,'(a,1x,i0)') 'xgb_stage_status', stage_status
    write(*,'(a,1x,i0)') 'xgb_jvp_status', jvp_status
    write(*,'(a,1x,i0)') 'xgb_vjp_status', vjp_status
    write(*,'(a,1x,i0)') 'xgb_output_count', xgb%output_count()
    write(*,'(a,1x,i0)') 'xgb_parameter_count', xgb%parameter_count()
    call xgb%predict_device(cuda, x, values, status)
    write(*,'(a,1x,i0)') 'xgb_cuda_status', status%code
    call xgb%predict_device_margin(cuda, x, values, status)
    write(*,'(a,1x,i0)') 'xgb_cuda_margin_status', status%code
    call xgb%predict(x, before, status)
    bad_targets = targets
    bad_targets(1, 1) = ieee_value(0.0_dp, ieee_quiet_nan)
    call xgb%fit(x, bad_targets, status, xo)
    write(*,'(a,1x,i0)') 'xgb_malformed_status', status%code
    call xgb%predict(x, values, status)
    write(*,'(a,1x,es24.16)') 'xgb_transaction_error', maxval(abs(values-before))

    lo%n_estimators = 1
    lo%num_leaves = 2
    lo%min_data_in_leaf = 1
    lo%max_bin = 16
    lo%l2 = 1.0_dp
    lo%learning_rate = 1.0_dp
    call cpu_time(start)
    call lgb%fit(x, targets, status, lo)
    call cpu_time(finish)
    fit_status = status%code
    write(*,'(a,1x,es24.16)') 'lgb_fit_seconds', finish-start
    call lgb%predict(x, values, status)
    predict_status = status%code
    write(*,'(a,*(es24.16,1x))') 'lgb_predict', values
    call lgb%predict_staged_margin(x, staged, status)
    stage_status = status%code
    write(*,'(a,*(es24.16,1x))') 'lgb_staged', staged
    call lgb%predict_leaf_jvp(x, parameter_dot, values, values_dot, status)
    jvp_status = status%code
    write(*,'(a,*(es24.16,1x))') 'lgb_leaf_jvp', values_dot
    call lgb%predict_leaf_vjp(x, output_bar, parameter_bar, status)
    vjp_status = status%code
    write(*,'(a,*(es24.16,1x))') 'lgb_leaf_vjp', parameter_bar
    write(*,'(a,1x,i0)') 'lgb_fit_status', fit_status
    write(*,'(a,1x,i0)') 'lgb_predict_status', predict_status
    write(*,'(a,1x,i0)') 'lgb_stage_status', stage_status
    write(*,'(a,1x,i0)') 'lgb_jvp_status', jvp_status
    write(*,'(a,1x,i0)') 'lgb_vjp_status', vjp_status
    write(*,'(a,1x,i0)') 'lgb_output_count', lgb%output_count()
    write(*,'(a,1x,i0)') 'lgb_parameter_count', lgb%parameter_count()
    call lgb%predict_device(cuda, x, values, status)
    write(*,'(a,1x,i0)') 'lgb_cuda_status', status%code
end program fortml_bench_xgboost_multioutput
