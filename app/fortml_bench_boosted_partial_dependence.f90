program fortml_bench_boosted_partial_dependence
    !! Release workload for weighted boosted-tree PDP and ICE diagnostics.
    use, intrinsic :: iso_fortran_env, only: real64
    use fortml_boosted_partial_dependence, only: boosted_partial_dependence, &
        FORTML_TREE_RESPONSE_PREDICTION, FORTML_TREE_RESPONSE_MARGIN
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortml_xgboost, only: xgboost_t, xgboost_options_t
    use fortml_lightgbm, only: lightgbm_t, lightgbm_options_t
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    implicit none

    integer, parameter :: dp = real64, n = 4, d = 2, n_grid = 2
    real(dp) :: x(n, d), target(n), binary_target(n), grid(n_grid), weight(n)
    real(dp) :: average(n_grid), ice(n, n_grid), margin(n_grid), response(n_grid)
    real(dp) :: started, finished, link_error
    type(xgboost_t) :: xgb, xgb_binary
    type(lightgbm_t) :: lgb, lgb_binary
    type(xgboost_options_t) :: xgb_options, xgb_binary_options
    type(lightgbm_options_t) :: lgb_options, lgb_binary_options
    type(fortml_device_t) :: cuda
    type(fortnum_status_t) :: status

    x(:, 1) = [0.0_dp, 1.0_dp, 2.0_dp, 3.0_dp]
    x(:, 2) = [10.0_dp, 20.0_dp, 30.0_dp, 40.0_dp]
    target = [0.0_dp, 0.0_dp, 10.0_dp, 10.0_dp]
    binary_target = [0.0_dp, 0.0_dp, 1.0_dp, 1.0_dp]
    grid = [-1.0_dp, 1.0_dp]
    weight = [1.0_dp, 1.0_dp, 1.0_dp, 3.0_dp]

    xgb_options = xgboost_options_t()
    xgb_options%n_estimators = 1
    xgb_options%max_depth = 1
    xgb_options%min_child_weight = 0.0_dp
    xgb_options%l2 = 1.0_dp
    xgb_options%learning_rate = 1.0_dp
    call xgb%fit_regression(x, target, status, xgb_options)
    if (.not. status_ok(status)) error stop "XGBoost PDP fit failed"
    call cpu_time(started)
    call boosted_partial_dependence(xgb, x, 2, grid, average, status, weight, ice)
    call cpu_time(finished)
    if (.not. status_ok(status)) error stop "XGBoost PDP failed"
    write (*, '(a,es24.16)') "xgb_pdp_seconds ", max(0.0_dp, finished-started)
    write (*, '(a,*(es24.16,1x))') "xgb_pdp_values ", average, &
        reshape(ice, [n*n_grid])

    xgb_binary_options = xgb_options
    xgb_binary_options%objective = "logistic"
    call xgb_binary%fit_binary(x, binary_target, status, xgb_binary_options)
    if (.not. status_ok(status)) error stop "XGBoost binary PDP fit failed"
    grid = [0.0_dp, 3.0_dp]
    call boosted_partial_dependence(xgb_binary, x, 1, grid, margin, status, &
        response=FORTML_TREE_RESPONSE_MARGIN)
    if (.not. status_ok(status)) error stop "XGBoost margin PDP failed"
    call boosted_partial_dependence(xgb_binary, x, 1, grid, response, status, &
        response=FORTML_TREE_RESPONSE_PREDICTION)
    if (.not. status_ok(status)) error stop "XGBoost response PDP failed"
    link_error = maxval(abs(response-1.0_dp/(1.0_dp + exp(-margin))))
    write (*, '(a,es24.16)') "xgb_pdp_link_error ", link_error

    lgb_options = lightgbm_options_t()
    lgb_options%n_estimators = 1
    lgb_options%num_leaves = 2
    lgb_options%min_data_in_leaf = 1
    lgb_options%max_bin = 16
    lgb_options%l2 = 1.0_dp
    lgb_options%learning_rate = 1.0_dp
    call lgb%fit_regression(x, target, status, lgb_options)
    if (.not. status_ok(status)) error stop "LightGBM PDP fit failed"
    grid = [-1.0_dp, 1.0_dp]
    call cpu_time(started)
    call boosted_partial_dependence(lgb, x, 2, grid, average, status, weight, ice)
    call cpu_time(finished)
    if (.not. status_ok(status)) error stop "LightGBM PDP failed"
    write (*, '(a,es24.16)') "lgbm_pdp_seconds ", max(0.0_dp, finished-started)
    write (*, '(a,*(es24.16,1x))') "lgbm_pdp_values ", average, &
        reshape(ice, [n*n_grid])

    lgb_binary_options = lgb_options
    lgb_binary_options%objective = "binary"
    call lgb_binary%fit_binary(x, binary_target, status, lgb_binary_options)
    if (.not. status_ok(status)) error stop "LightGBM binary PDP fit failed"
    grid = [0.0_dp, 3.0_dp]
    call boosted_partial_dependence(lgb_binary, x, 1, grid, margin, status, &
        response=FORTML_TREE_RESPONSE_MARGIN)
    if (.not. status_ok(status)) error stop "LightGBM margin PDP failed"
    call boosted_partial_dependence(lgb_binary, x, 1, grid, response, status, &
        response=FORTML_TREE_RESPONSE_PREDICTION)
    if (.not. status_ok(status)) error stop "LightGBM response PDP failed"
    link_error = maxval(abs(response-1.0_dp/(1.0_dp + exp(-margin))))
    write (*, '(a,es24.16)') "lgbm_pdp_link_error ", link_error

    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call boosted_partial_dependence(xgb, x, 1, grid, average, status, device=cuda)
    if (status%code /= FORTNUM_NOT_IMPLEMENTED) error stop "XGBoost CUDA refusal failed"
    write (*, '(a,i0)') "xgb_pdp_cuda ", status%code
    call boosted_partial_dependence(lgb, x, 1, grid, average, status, device=cuda)
    if (status%code /= FORTNUM_NOT_IMPLEMENTED) error stop "LightGBM CUDA refusal failed"
    write (*, '(a,i0)') "lgbm_pdp_cuda ", status%code
end program fortml_bench_boosted_partial_dependence
