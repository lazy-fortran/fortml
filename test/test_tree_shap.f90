program test_tree_shap
    !! Independent behavioral oracle for bounded per-feature tree attributions.
    !! The one-stump fixture has a hand-computed baseline and leaf correction;
    !! no private tree representation is consulted.
    use, intrinsic :: iso_fortran_env, only: real64, error_unit
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortml_xgboost, only: xgboost_t, xgboost_options_t
    use fortml_lightgbm, only: lightgbm_t, lightgbm_options_t
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    implicit none

    integer, parameter :: dp = real64
    type(xgboost_t) :: xgb
    type(lightgbm_t) :: lgb
    type(xgboost_options_t) :: xgb_options
    type(lightgbm_options_t) :: lgb_options
    type(fortml_device_t) :: cuda
    type(fortnum_status_t) :: status
    real(dp) :: x(4, 2), target(4), prediction(4), shap(4, 3)
    integer :: failures

    x(:, 1) = [0.0_dp, 1.0_dp, 2.0_dp, 3.0_dp]
    x(:, 2) = 0.0_dp
    target = [0.0_dp, 0.0_dp, 10.0_dp, 10.0_dp]
    failures = 0

    xgb_options = xgboost_options_t()
    xgb_options%n_estimators = 1
    xgb_options%max_depth = 1
    xgb_options%min_child_weight = 0.0_dp
    xgb_options%l2 = 1.0_dp
    xgb_options%learning_rate = 1.0_dp
    call xgb%fit_regression(x, target, status, xgb_options)
    call check(status_ok(status), "XGBoost stump fit", failures)
    call xgb%predict_margin(x, prediction, status)
    call xgb%predict_shap(x, shap, status)
    call check(status_ok(status), "XGBoost SHAP status", failures)
    call check(maxval(abs(sum(shap, dim=2)-prediction)) < 2.0e-13_dp, &
        "XGBoost SHAP additivity", failures)
    call check(maxval(abs(shap(:, 1)-5.0_dp)) < 2.0e-13_dp, &
        "XGBoost SHAP baseline oracle", failures)
    call check(maxval(abs(shap(:2, 2)+10.0_dp/3.0_dp)) < 2.0e-13_dp .and. &
        maxval(abs(shap(3:, 2)-10.0_dp/3.0_dp)) < 2.0e-13_dp, &
        "XGBoost SHAP one-feature leaf oracle", failures)
    call check(maxval(abs(shap(:, 3))) < 2.0e-13_dp, &
        "XGBoost unused-feature zero attribution", failures)

    lgb_options = lightgbm_options_t()
    lgb_options%n_estimators = 1
    lgb_options%num_leaves = 2
    lgb_options%min_data_in_leaf = 1
    lgb_options%max_bin = 16
    lgb_options%l2 = 1.0_dp
    lgb_options%learning_rate = 1.0_dp
    call lgb%fit_regression(x, target, status, lgb_options)
    call check(status_ok(status), "LightGBM stump fit", failures)
    call lgb%predict_margin(x, prediction, status)
    call lgb%predict_shap(x, shap, status)
    call check(status_ok(status), "LightGBM SHAP status", failures)
    call check(maxval(abs(sum(shap, dim=2)-prediction)) < 2.0e-13_dp, &
        "LightGBM SHAP additivity", failures)
    call check(maxval(abs(shap(:, 1)-5.0_dp)) < 2.0e-13_dp, &
        "LightGBM SHAP baseline oracle", failures)
    call check(maxval(abs(shap(:2, 2)+10.0_dp/3.0_dp)) < 2.0e-13_dp .and. &
        maxval(abs(shap(3:, 2)-10.0_dp/3.0_dp)) < 2.0e-13_dp, &
        "LightGBM SHAP one-feature leaf oracle", failures)
    call check(maxval(abs(shap(:, 3))) < 2.0e-13_dp, &
        "LightGBM unused-feature zero attribution", failures)

    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call xgb%predict_shap_device(cuda, x, shap, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "XGBoost CUDA SHAP refusal", failures)
    call lgb%predict_shap_device(cuda, x, shap, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "LightGBM CUDA SHAP refusal", failures)

    if (failures > 0) then
        write (error_unit, '(a,i0)') "FAIL tree SHAP cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS tree SHAP independent behavioral oracle"

contains

    subroutine check(condition, label, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: label
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "FAIL: " // trim(label)
        end if
    end subroutine check

end program test_tree_shap
