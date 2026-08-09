program test_xgboost_multioutput_validation_metadata
    !! Independent scalar-prefix oracle for multi-output validation metadata.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_xgboost, only: xgboost_t, xgboost_options_t
    use fortml_xgboost_multioutput, only: xgboost_multioutput_t, &
        lightgbm_multioutput_t
    use fortml_lightgbm, only: lightgbm_t, lightgbm_options_t
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    type(xgboost_multioutput_t) :: xgb_multi
    type(xgboost_t) :: xgb_scalar
    type(lightgbm_multioutput_t) :: lgb_multi
    type(lightgbm_t) :: lgb_scalar
    type(xgboost_options_t) :: xgb_options
    type(xgboost_options_t) :: xgb_oracle_options
    type(lightgbm_options_t) :: lgb_options
    type(lightgbm_options_t) :: lgb_oracle_options
    type(fortnum_status_t) :: status
    real(dp) :: x(8, 1), targets(8, 2), validation_targets(8, 2)
    real(dp) :: staged(8, 8), prediction(8), losses(8), best_losses(2)
    real(dp) :: lgb_staged(8, 6), lgb_losses(6), lgb_best_losses(2)
    integer, allocatable :: best_iterations(:), lgb_best_iterations(:)
    real(dp), allocatable :: multi_losses(:), lgb_multi_losses(:)
    logical, allocatable :: stopped(:), lgb_stopped(:)
    integer :: i, j, expected_iteration, lgb_expected_iteration, failures

    failures = 0
    x(:, 1) = real([(i, i = 0, 7)], dp)
    targets(:, 1) = [0.0_dp, 0.0_dp, 0.0_dp, 0.0_dp, 10.0_dp, 10.0_dp, 10.0_dp, 10.0_dp]
    targets(:, 2) = [1.0_dp, 1.0_dp, 1.0_dp, 1.0_dp, 5.0_dp, 5.0_dp, 5.0_dp, 5.0_dp]
    validation_targets(:, 1) = 10.0_dp - targets(:, 1)
    validation_targets(:, 2) = 6.0_dp - targets(:, 2)

    xgb_options%n_estimators = 8
    xgb_options%max_depth = 1
    xgb_options%learning_rate = 1.0_dp
    xgb_options%l2 = 1.0_dp
    xgb_options%min_child_weight = 0.0_dp
    xgb_options%early_stopping_rounds = 2
    xgb_oracle_options = xgb_options
    xgb_oracle_options%early_stopping_rounds = 0
    do j = 1, 2
        call xgb_scalar%fit_regression(x, targets(:, j), status, xgb_oracle_options)
        call check(status_ok(status), "XGBoost scalar oracle fit", failures)
        call xgb_scalar%predict_staged(x, staged, status)
        losses = 0.5_dp*sum((staged - spread(validation_targets(:, j), 2, 8))**2, dim=1)/8.0_dp
        expected_iteration = 1
        best_losses(j) = losses(1)
        do i = 2, 8
            if (losses(i) < best_losses(j)) then
                best_losses(j) = losses(i)
                expected_iteration = i
            end if
        end do
        call check(expected_iteration >= 1, "XGBoost scalar oracle has a best stage", failures)
    end do
    call xgb_multi%fit(x, targets, status, xgb_options, validation_x=x, &
        validation_targets=validation_targets)
    call check(status_ok(status), "XGBoost multi-output validation fit", failures)
    best_iterations = xgb_multi%best_iteration()
    stopped = xgb_multi%early_stopped()
    multi_losses = xgb_multi%best_validation_loss()
    do j = 1, 2
        call check(best_iterations(j) >= 1, "XGBoost multi-output best iteration", failures)
    end do
    call check(all(best_iterations >= 1) .and. all(stopped), &
        "XGBoost multi-output early-stop flags", failures)
    call check(maxval(abs(multi_losses - best_losses)) < 2.0e-12_dp, &
        "XGBoost multi-output validation loss vector", failures)

    lgb_options%n_estimators = 6
    lgb_options%num_leaves = 2
    lgb_options%min_data_in_leaf = 1
    lgb_options%max_bin = 16
    lgb_options%learning_rate = 1.0_dp
    lgb_options%l2 = 1.0_dp
    lgb_options%early_stopping_rounds = 2
    lgb_oracle_options = lgb_options
    lgb_oracle_options%early_stopping_rounds = 0
    do j = 1, 2
        call lgb_scalar%fit_regression(x, targets(:, j), status, lgb_oracle_options)
        call check(status_ok(status), "LightGBM scalar oracle fit", failures)
        call lgb_scalar%predict_staged(x, lgb_staged, status)
        lgb_losses = 0.5_dp*sum((lgb_staged - spread(validation_targets(:, j), 2, 6))**2, dim=1)/8.0_dp
        lgb_expected_iteration = 1
        lgb_best_losses(j) = lgb_losses(1)
        do i = 2, 6
            if (lgb_losses(i) < lgb_best_losses(j)) then
                lgb_best_losses(j) = lgb_losses(i)
                lgb_expected_iteration = i
            end if
        end do
        call check(lgb_expected_iteration >= 1, "LightGBM scalar oracle has a best stage", failures)
    end do
    call lgb_multi%fit(x, targets, status, lgb_options, validation_x=x, &
        validation_targets=validation_targets)
    call check(status_ok(status), "LightGBM multi-output validation fit", failures)
    lgb_best_iterations = lgb_multi%best_iteration()
    lgb_stopped = lgb_multi%early_stopped()
    lgb_multi_losses = lgb_multi%best_validation_loss()
    do j = 1, 2
        call check(lgb_best_iterations(j) >= 1, "LightGBM multi-output best iteration", failures)
    end do
    call check(all(lgb_best_iterations >= 1) .and. all(lgb_stopped), &
        "LightGBM multi-output early-stop flags", failures)
    call check(maxval(abs(lgb_multi_losses - lgb_best_losses)) < 2.0e-12_dp, &
        "LightGBM multi-output validation loss vector", failures)

    if (failures /= 0) then
        write (error_unit, '(a,i0)') "FAIL multi-output validation metadata: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS multi-output validation metadata independent oracle"

contains

    subroutine check(condition, description, failure_count)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failure_count

        if (.not. condition) then
            failure_count = failure_count + 1
            write (error_unit, '(a)') "  FAIL [multi-output validation] "//description
        end if
    end subroutine check

end program test_xgboost_multioutput_validation_metadata
