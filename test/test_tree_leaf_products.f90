program test_tree_leaf_products
    !! Independent oracle for fixed-structure tree leaf products.
    !! The expected contractions use only the two leaves of a hand-selected
    !! stump; no private node arrays or fitted leaf values are inspected.
    use, intrinsic :: iso_fortran_env, only: real64, error_unit
    use fortml_xgboost, only: xgboost_t, xgboost_options_t
    use fortml_lightgbm, only: lightgbm_t, lightgbm_options_t
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_DOMAIN_ERROR
    implicit none

    integer, parameter :: dp = real64
    type(xgboost_t) :: xgb
    type(lightgbm_t) :: lgb
    type(xgboost_options_t) :: xgb_options
    type(lightgbm_options_t) :: lgb_options
    type(fortnum_status_t) :: status
    real(dp) :: x(4, 2), target(4), dot(3), y(4), y_dot(4), bar(4), pbar(3)
    real(dp) :: parameters(3), bad_y(4)
    integer :: failures

    x(:, 1) = [0.0_dp, 1.0_dp, 2.0_dp, 3.0_dp]
    x(:, 2) = 0.0_dp
    target = [0.0_dp, 0.0_dp, 10.0_dp, 10.0_dp]
    dot = [0.5_dp, 1.0_dp, 2.0_dp]
    bar = [1.0_dp, 2.0_dp, 3.0_dp, 4.0_dp]
    failures = 0

    xgb_options = xgboost_options_t()
    xgb_options%n_estimators = 1
    xgb_options%max_depth = 1
    xgb_options%min_child_weight = 0.0_dp
    xgb_options%l2 = 1.0_dp
    xgb_options%learning_rate = 1.0_dp
    call xgb%fit_regression(x, target, status, xgb_options)
    call check(status_ok(status), "XGBoost stump fit", failures)
    call check(xgb%leaf_parameter_count() == 3, &
        "XGBoost packed base plus two leaves", failures)
    parameters = xgb%leaf_parameters(status)
    call check(status_ok(status) .and. all(ieee_finite(parameters)), &
        "XGBoost packed leaf values", failures)
    call xgb%predict_leaf_jvp(x, dot, y, y_dot, status)
    call check(status_ok(status), "XGBoost leaf JVP status", failures)
    call check(maxval(abs(y_dot-[1.5_dp, 1.5_dp, 2.5_dp, 2.5_dp])) < 2.0e-13_dp, &
        "XGBoost leaf JVP hand oracle", failures)
    call xgb%predict_leaf_vjp(x, bar, pbar, status)
    call check(status_ok(status), "XGBoost leaf VJP status", failures)
    call check(maxval(abs(pbar-[10.0_dp, 3.0_dp, 7.0_dp])) < 2.0e-13_dp, &
        "XGBoost leaf VJP hand oracle", failures)
    call check(abs(dot(1)*pbar(1) + dot(2)*pbar(2) + dot(3)*pbar(3) - &
        sum(bar*y_dot)) < 2.0e-13_dp, "XGBoost leaf adjoint identity", failures)

    bad_y = -7.0_dp
    y = bad_y
    call xgb%predict_leaf_jvp(x, dot(:2), y, y_dot, status)
    call check(status%code == FORTNUM_DOMAIN_ERROR .and. all(y == bad_y), &
        "XGBoost malformed tangent refusal is transactional", failures)

    lgb_options = lightgbm_options_t()
    lgb_options%n_estimators = 1
    lgb_options%num_leaves = 2
    lgb_options%min_data_in_leaf = 1
    lgb_options%max_bin = 16
    lgb_options%l2 = 1.0_dp
    lgb_options%learning_rate = 1.0_dp
    call lgb%fit_regression(x, target, status, lgb_options)
    call check(status_ok(status), "LightGBM stump fit", failures)
    call check(lgb%leaf_parameter_count() == 3, &
        "LightGBM packed base plus two leaves", failures)
    parameters = lgb%leaf_parameters(status)
    call check(status_ok(status) .and. all(ieee_finite(parameters)), &
        "LightGBM packed leaf values", failures)
    call lgb%predict_leaf_jvp(x, dot, y, y_dot, status)
    call check(status_ok(status), "LightGBM leaf JVP status", failures)
    call check(maxval(abs(y_dot-[1.5_dp, 1.5_dp, 2.5_dp, 2.5_dp])) < 2.0e-13_dp, &
        "LightGBM leaf JVP hand oracle", failures)
    call lgb%predict_leaf_vjp(x, bar, pbar, status)
    call check(status_ok(status), "LightGBM leaf VJP status", failures)
    call check(maxval(abs(pbar-[10.0_dp, 3.0_dp, 7.0_dp])) < 2.0e-13_dp, &
        "LightGBM leaf VJP hand oracle", failures)
    call check(abs(dot(1)*pbar(1) + dot(2)*pbar(2) + dot(3)*pbar(3) - &
        sum(bar*y_dot)) < 2.0e-13_dp, "LightGBM leaf adjoint identity", failures)

    if (failures > 0) then
        write (error_unit, '(a,i0)') "FAIL tree leaf product cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS tree leaf products independent behavioral oracle"

contains

    elemental logical function ieee_finite(value)
        real(dp), intent(in) :: value
        ieee_finite = abs(value) < huge(1.0_dp)
    end function ieee_finite

    subroutine check(condition, label, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: label
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "FAIL: " // trim(label)
        end if
    end subroutine check

end program test_tree_leaf_products
