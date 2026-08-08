program fortml_bench_tree_leaf_products
    !! Small release workload for fixed-structure tree leaf products.
    use, intrinsic :: iso_fortran_env, only: real64
    use fortml_xgboost, only: xgboost_t, xgboost_options_t
    use fortml_lightgbm, only: lightgbm_t, lightgbm_options_t
    use fortnum_status, only: fortnum_status_t
    implicit none

    integer, parameter :: dp = real64
    type(xgboost_t) :: xgb
    type(lightgbm_t) :: lgb
    type(xgboost_options_t) :: xo
    type(lightgbm_options_t) :: lo
    type(fortnum_status_t) :: status
    real(dp) :: x(4, 2), target(4), dot(3), y(4), y_dot(4), bar(4), pbar(3)
    real(dp) :: parameters(3), start, finish, fit_seconds, predict_seconds

    x(:, 1) = [0.0_dp, 1.0_dp, 2.0_dp, 3.0_dp]
    x(:, 2) = 0.0_dp
    target = [0.0_dp, 0.0_dp, 10.0_dp, 10.0_dp]
    dot = [0.5_dp, 1.0_dp, 2.0_dp]
    bar = [1.0_dp, 2.0_dp, 3.0_dp, 4.0_dp]

    xo%n_estimators = 1
    xo%max_depth = 1
    xo%min_child_weight = 0.0_dp
    xo%l2 = 1.0_dp
    xo%learning_rate = 1.0_dp
    call cpu_time(start)
    call xgb%fit_regression(x, target, status, xo)
    call cpu_time(finish)
    fit_seconds = finish - start
    call cpu_time(start)
    parameters = xgb%leaf_parameters(status)
    call xgb%predict_leaf_jvp(x, dot, y, y_dot, status)
    call xgb%predict_leaf_vjp(x, bar, pbar, status)
    call cpu_time(finish)
    predict_seconds = finish - start
    write(*,'(a,1x,i0)') 'xgb_leaf_parameter_count', xgb%leaf_parameter_count()
    write(*,'(a,*(es24.16,1x))') 'xgb_leaf_parameters', parameters
    write(*,'(a,*(es24.16,1x))') 'xgb_leaf_jvp', y_dot
    write(*,'(a,*(es24.16,1x))') 'xgb_leaf_vjp', pbar
    write(*,'(a,1x,i0,1x,es24.16)') 'xgb_leaf_status_fit_seconds', status%code, fit_seconds
    write(*,'(a,1x,es24.16)') 'xgb_leaf_predict_seconds', predict_seconds

    lo%n_estimators = 1
    lo%num_leaves = 2
    lo%min_data_in_leaf = 1
    lo%max_bin = 16
    lo%l2 = 1.0_dp
    lo%learning_rate = 1.0_dp
    call cpu_time(start)
    call lgb%fit_regression(x, target, status, lo)
    call cpu_time(finish)
    fit_seconds = finish - start
    call cpu_time(start)
    parameters = lgb%leaf_parameters(status)
    call lgb%predict_leaf_jvp(x, dot, y, y_dot, status)
    call lgb%predict_leaf_vjp(x, bar, pbar, status)
    call cpu_time(finish)
    predict_seconds = finish - start
    write(*,'(a,1x,i0)') 'lgbm_leaf_parameter_count', lgb%leaf_parameter_count()
    write(*,'(a,*(es24.16,1x))') 'lgbm_leaf_parameters', parameters
    write(*,'(a,*(es24.16,1x))') 'lgbm_leaf_jvp', y_dot
    write(*,'(a,*(es24.16,1x))') 'lgbm_leaf_vjp', pbar
    write(*,'(a,1x,i0,1x,es24.16)') 'lgbm_leaf_status_fit_seconds', status%code, fit_seconds
    write(*,'(a,1x,es24.16)') 'lgbm_leaf_predict_seconds', predict_seconds
end program fortml_bench_tree_leaf_products
