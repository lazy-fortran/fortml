program test_xgboost_multioutput
    !! Independent two-target stump oracle for the multi-output adapters.
    !! The expected values use only the closed-form one-split Newton update;
    !! no private child-tree state is inspected.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite, ieee_value, ieee_quiet_nan
    use, intrinsic :: iso_fortran_env, only: real64, error_unit
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortml_xgboost_multioutput, only: xgboost_multioutput_t, &
        lightgbm_multioutput_t
    use fortml_xgboost, only: xgboost_options_t
    use fortml_lightgbm, only: lightgbm_options_t
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_DOMAIN_ERROR, &
        FORTNUM_NOT_IMPLEMENTED
    implicit none

    integer, parameter :: dp = real64
    type(xgboost_multioutput_t) :: xgb
    type(lightgbm_multioutput_t) :: lgb
    type(xgboost_options_t) :: xo
    type(lightgbm_options_t) :: lo
    type(fortnum_status_t) :: status
    type(fortml_device_t) :: cuda
    real(dp) :: x(4, 1), x_missing(4, 1), targets(4, 2), values(4, 2), margin(4, 2)
    real(dp) :: staged(4, 1, 2), x_dot(4, 1), values_dot(4, 2), x_bar(4, 1)
    real(dp) :: output_bar(4, 2), parameter_dot(6), parameter_bar(6)
    real(dp) :: before(4, 2), bad_targets(4, 2)
    integer :: failures

    failures = 0
    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    x(:, 1) = [0.0_dp, 1.0_dp, 2.0_dp, 3.0_dp]
    targets(:, 1) = [0.0_dp, 0.0_dp, 10.0_dp, 10.0_dp]
    targets(:, 2) = [1.0_dp, 1.0_dp, 5.0_dp, 5.0_dp]
    x_dot(:, 1) = 0.25_dp
    output_bar(:, 1) = [1.0_dp, 2.0_dp, 3.0_dp, 4.0_dp]
    output_bar(:, 2) = [2.0_dp, 3.0_dp, 4.0_dp, 5.0_dp]
    parameter_dot = [1.0_dp, 2.0_dp, 3.0_dp, 4.0_dp, 5.0_dp, 6.0_dp]

    xo%n_estimators = 1
    xo%max_depth = 1
    xo%min_child_weight = 0.0_dp
    xo%l2 = 1.0_dp
    xo%learning_rate = 1.0_dp
    call xgb%fit(x, targets, status, xo)
    call check(status_ok(status) .and. xgb%fitted(), &
        "XGBoost multi-output fit", failures)
    call check(xgb%feature_count() == 1 .and. xgb%output_count() == 2 .and. &
        xgb%estimator_count() == 1 .and. xgb%parameter_count() == 6, &
        "XGBoost multi-output metadata", failures)
    call xgb%predict(x, values, status)
    call check(status_ok(status), "XGBoost multi-output predict status", failures)
    call check(maxval(abs(values(:, 1) - [5.0_dp/3.0_dp, 5.0_dp/3.0_dp, &
        25.0_dp/3.0_dp, 25.0_dp/3.0_dp])) < 2.0e-13_dp .and. &
        maxval(abs(values(:, 2) - [5.0_dp/3.0_dp, 5.0_dp/3.0_dp, &
        13.0_dp/3.0_dp, 13.0_dp/3.0_dp])) < 2.0e-13_dp, &
        "XGBoost two-output Newton stump oracle", failures)
    call xgb%predict_margin(x, margin, status)
    call check(status_ok(status) .and. maxval(abs(margin-values)) < 2.0e-13_dp, &
        "XGBoost regression margins", failures)
    call xgb%predict_staged_margin(x, staged, status)
    call check(status_ok(status) .and. maxval(abs(staged(:, 1, :) - values)) < 2.0e-13_dp, &
        "XGBoost staged multi-output margins", failures)
    call xgb%predict_jvp(x, x_dot, values, values_dot, status)
    call check(status_ok(status) .and. maxval(abs(values_dot)) < 2.0e-13_dp, &
        "XGBoost input JVP away from split", failures)
    call xgb%predict_vjp(x, output_bar, x_bar, status)
    call check(status_ok(status) .and. maxval(abs(x_bar)) < 2.0e-13_dp, &
        "XGBoost input VJP away from split", failures)
    call xgb%predict_leaf_jvp(x, parameter_dot, values, values_dot, status)
    call check(status_ok(status) .and. maxval(abs(values_dot(:, 1) - [3.0_dp, 3.0_dp, &
        4.0_dp, 4.0_dp])) < 2.0e-13_dp .and. &
        maxval(abs(values_dot(:, 2) - [9.0_dp, 9.0_dp, 10.0_dp, 10.0_dp])) < 2.0e-13_dp, &
        "XGBoost fixed-leaf parameter JVP", failures)
    call xgb%predict_leaf_vjp(x, output_bar, parameter_bar, status)
    call check(status_ok(status) .and. maxval(abs(parameter_bar - &
        [10.0_dp, 3.0_dp, 7.0_dp, 14.0_dp, 5.0_dp, 9.0_dp])) < 2.0e-13_dp, &
        "XGBoost fixed-leaf parameter VJP", failures)
    values = -91.0_dp
    call xgb%predict_device(cuda, x, values, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "XGBoost multi-output CUDA refusal", failures)
    call check(all(values == -91.0_dp), "XGBoost CUDA prediction is transactional", failures)
    values = -92.0_dp
    call xgb%predict_device_margin(cuda, x, values, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "XGBoost multi-output margin CUDA refusal", failures)
    call check(all(values == -92.0_dp), "XGBoost CUDA margin is transactional", failures)

    call xgb%predict(x, before, status)
    bad_targets = targets
    bad_targets(1, 1) = ieee_value(0.0_dp, ieee_quiet_nan)
    call xgb%fit(x, bad_targets, status, xo)
    call check(status%code == FORTNUM_DOMAIN_ERROR, &
        "XGBoost malformed fit refusal", failures)
    call xgb%predict(x, values, status)
    call check(status_ok(status) .and. maxval(abs(values-before)) < 2.0e-13_dp, &
        "XGBoost malformed fit is transactional", failures)

    x_missing = x
    x_missing(2, 1) = ieee_value(0.0_dp, ieee_quiet_nan)
    xo%missing_policy = "learn"
    call xgb%fit(x_missing, targets, status, xo)
    call check(status_ok(status), "XGBoost multi-output preserves missing policy", failures)
    xo%missing_policy = "error"

    lo%n_estimators = 1
    lo%num_leaves = 2
    lo%min_data_in_leaf = 1
    lo%max_bin = 16
    lo%l2 = 1.0_dp
    lo%learning_rate = 1.0_dp
    call lgb%fit(x, targets, status, lo)
    call check(status_ok(status) .and. lgb%fitted(), "LightGBM multi-output fit", failures)
    call check(lgb%feature_count() == 1 .and. lgb%output_count() == 2 .and. &
        lgb%estimator_count() == 1 .and. lgb%parameter_count() == 6, &
        "LightGBM multi-output metadata", failures)
    call lgb%predict(x, values, status)
    call check(status_ok(status) .and. maxval(abs(values(:, 1) - [5.0_dp/3.0_dp, &
        5.0_dp/3.0_dp, 25.0_dp/3.0_dp, 25.0_dp/3.0_dp])) < 2.0e-13_dp .and. &
        maxval(abs(values(:, 2) - [5.0_dp/3.0_dp, 5.0_dp/3.0_dp, &
        13.0_dp/3.0_dp, 13.0_dp/3.0_dp])) < 2.0e-13_dp, &
        "LightGBM two-output Newton stump oracle", failures)
    call lgb%predict_staged_margin(x, staged, status)
    call check(status_ok(status) .and. maxval(abs(staged(:, 1, :) - values)) < 2.0e-13_dp, &
        "LightGBM staged multi-output margins", failures)
    call lgb%predict_jvp(x, x_dot, values, values_dot, status)
    call check(status_ok(status) .and. maxval(abs(values_dot)) < 2.0e-13_dp, &
        "LightGBM input JVP away from split", failures)
    call lgb%predict_vjp(x, output_bar, x_bar, status)
    call check(status_ok(status) .and. maxval(abs(x_bar)) < 2.0e-13_dp, &
        "LightGBM input VJP away from split", failures)
    call lgb%predict_leaf_jvp(x, parameter_dot, values, values_dot, status)
    call check(status_ok(status) .and. maxval(abs(values_dot(:, 1) - [3.0_dp, 3.0_dp, &
        4.0_dp, 4.0_dp])) < 2.0e-13_dp .and. &
        maxval(abs(values_dot(:, 2) - [9.0_dp, 9.0_dp, 10.0_dp, 10.0_dp])) < 2.0e-13_dp, &
        "LightGBM fixed-leaf parameter JVP", failures)
    call lgb%predict_leaf_vjp(x, output_bar, parameter_bar, status)
    call check(status_ok(status) .and. maxval(abs(parameter_bar - &
        [10.0_dp, 3.0_dp, 7.0_dp, 14.0_dp, 5.0_dp, 9.0_dp])) < 2.0e-13_dp, &
        "LightGBM fixed-leaf parameter VJP", failures)
    values = -93.0_dp
    call lgb%predict_device(cuda, x, values, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "LightGBM multi-output CUDA refusal", failures)
    call check(all(values == -93.0_dp), "LightGBM CUDA prediction is transactional", failures)
    values = -94.0_dp
    call lgb%predict_device_margin(cuda, x, values, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "LightGBM multi-output margin CUDA refusal", failures)
    call check(all(values == -94.0_dp), "LightGBM CUDA margin is transactional", failures)

    if (failures > 0) then
        write (error_unit, '(a,i0)') "FAIL multi-output boosting cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS multi-output boosting independent behavioral oracle"

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

end program test_xgboost_multioutput
