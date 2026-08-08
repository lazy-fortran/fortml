program fortml_bench_tree_shap
    !! Release workload for bounded per-feature SHAP-like tree attributions.
    use, intrinsic :: iso_fortran_env, only: real64
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortml_xgboost, only: xgboost_t, xgboost_options_t
    use fortml_lightgbm, only: lightgbm_t, lightgbm_options_t
    use fortnum_status, only: fortnum_status_t, FORTNUM_NOT_IMPLEMENTED, status_ok
    implicit none

    integer, parameter :: dp = real64, n = 4, d = 2
    real(dp) :: x(n, d), target(n), prediction(n), shap(n, d+1)
    real(dp) :: expected(n, d+1), started, finished, seconds, error
    type(xgboost_t) :: xgb
    type(lightgbm_t) :: lgb
    type(xgboost_options_t) :: xgb_options
    type(lightgbm_options_t) :: lgb_options
    type(fortml_device_t) :: cuda
    type(fortnum_status_t) :: status

    x(:, 1) = [0.0_dp, 1.0_dp, 2.0_dp, 3.0_dp]
    x(:, 2) = 0.0_dp
    target = [0.0_dp, 0.0_dp, 10.0_dp, 10.0_dp]
    expected = 0.0_dp
    expected(:, 1) = 5.0_dp
    expected(:2, 2) = -10.0_dp/3.0_dp
    expected(3:, 2) = 10.0_dp/3.0_dp

    xgb_options = xgboost_options_t()
    xgb_options%n_estimators = 1
    xgb_options%max_depth = 1
    xgb_options%min_child_weight = 0.0_dp
    xgb_options%l2 = 1.0_dp
    xgb_options%learning_rate = 1.0_dp
    call cpu_time(started)
    call xgb%fit_regression(x, target, status, xgb_options)
    call cpu_time(finished)
    if (.not. status_ok(status)) error stop "xgboost SHAP fit failed"
    seconds = max(0.0_dp, finished-started)
    call xgb%predict_shap(x, shap, status)
    if (.not. status_ok(status)) error stop "xgboost SHAP prediction failed"
    error = maxval(abs(shap-expected))
    call xgb%predict_margin(x, prediction, status)
    if (.not. status_ok(status)) error stop "xgboost margin prediction failed"
    write (*, '(a,i0,a,i0,a,i0,a,es24.16,a,es24.16)') &
        "xgb_shap_fit,", n, ",", d, ",", 1, ",", seconds, ",", error
    write (*, '(a,*(es24.16,1x))') "xgb_shap_values ", reshape(shap, [n*(d+1)])
    call cpu_time(started)
    call xgb%predict_shap(x, shap, status)
    call cpu_time(finished)
    if (.not. status_ok(status)) error stop "xgboost SHAP timing failed"
    write (*, '(a,es24.16)') "xgb_shap_predict_seconds ", max(0.0_dp, finished-started)
    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call xgb%predict_shap_device(cuda, x, shap, status)
    if (status%code /= FORTNUM_NOT_IMPLEMENTED) error stop "xgboost SHAP CUDA contract failed"
    write (*, '(a,i0)') "xgb_shap_cuda,", status%code

    lgb_options = lightgbm_options_t()
    lgb_options%n_estimators = 1
    lgb_options%num_leaves = 2
    lgb_options%min_data_in_leaf = 1
    lgb_options%max_bin = 16
    lgb_options%l2 = 1.0_dp
    lgb_options%learning_rate = 1.0_dp
    call cpu_time(started)
    call lgb%fit_regression(x, target, status, lgb_options)
    call cpu_time(finished)
    if (.not. status_ok(status)) error stop "lightgbm SHAP fit failed"
    seconds = max(0.0_dp, finished-started)
    call lgb%predict_shap(x, shap, status)
    if (.not. status_ok(status)) error stop "lightgbm SHAP prediction failed"
    error = maxval(abs(shap-expected))
    call lgb%predict_margin(x, prediction, status)
    if (.not. status_ok(status)) error stop "lightgbm margin prediction failed"
    write (*, '(a,i0,a,i0,a,i0,a,es24.16,a,es24.16)') &
        "lgbm_shap_fit,", n, ",", d, ",", 1, ",", seconds, ",", error
    write (*, '(a,*(es24.16,1x))') "lgbm_shap_values ", reshape(shap, [n*(d+1)])
    call cpu_time(started)
    call lgb%predict_shap(x, shap, status)
    call cpu_time(finished)
    if (.not. status_ok(status)) error stop "lightgbm SHAP timing failed"
    write (*, '(a,es24.16)') "lgbm_shap_predict_seconds ", max(0.0_dp, finished-started)
    call lgb%predict_shap_device(cuda, x, shap, status)
    if (status%code /= FORTNUM_NOT_IMPLEMENTED) error stop "lightgbm SHAP CUDA contract failed"
    write (*, '(a,i0)') "lgbm_shap_cuda,", status%code
end program fortml_bench_tree_shap
