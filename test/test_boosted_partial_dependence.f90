program test_boosted_partial_dependence
    !! Independent hand and public-prediction replay oracles for PDP and ICE.
    use, intrinsic :: iso_fortran_env, only: real64, error_unit
    use fortml_boosted_partial_dependence, only: boosted_partial_dependence, &
        boosted_partial_dependence_device_supported, &
        FORTML_TREE_RESPONSE_PREDICTION, FORTML_TREE_RESPONSE_MARGIN
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, &
        FORTML_DEVICE_CUDA
    use fortml_xgboost, only: xgboost_t, xgboost_options_t
    use fortml_lightgbm, only: lightgbm_t, lightgbm_options_t
    use fortnum_status, only: fortnum_status_t, status_ok, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED
    implicit none

    integer, parameter :: dp = real64
    real(dp), parameter :: tolerance = 2.0e-13_dp
    real(dp) :: x(4, 2), target(4), binary_target(4), grid(2), weight(4)
    real(dp) :: average(2), ice(4, 2), expected_ice(4, 2), expected_prediction(2)
    real(dp) :: margin_average(2), prediction_average(2), sentinel_average(2)
    real(dp) :: sentinel_ice(4, 2), logistic_oracle(2)
    type(xgboost_t) :: xgb, xgb_binary, unfitted_xgb
    type(lightgbm_t) :: lgb, lgb_binary
    type(xgboost_options_t) :: xgb_options, xgb_binary_options
    type(lightgbm_options_t) :: lgb_options, lgb_binary_options
    type(fortml_device_t) :: cpu, cuda
    type(fortnum_status_t) :: status
    integer :: failures

    x(:, 1) = [0.0_dp, 1.0_dp, 2.0_dp, 3.0_dp]
    x(:, 2) = [10.0_dp, 20.0_dp, 30.0_dp, 40.0_dp]
    target = [0.0_dp, 0.0_dp, 10.0_dp, 10.0_dp]
    binary_target = [0.0_dp, 0.0_dp, 1.0_dp, 1.0_dp]
    weight = [1.0_dp, 1.0_dp, 1.0_dp, 3.0_dp]
    grid = [0.0_dp, 3.0_dp]
    failures = 0

    xgb_options = xgboost_options_t()
    xgb_options%n_estimators = 1
    xgb_options%max_depth = 1
    xgb_options%min_child_weight = 0.0_dp
    xgb_options%l2 = 1.0_dp
    xgb_options%learning_rate = 1.0_dp
    call xgb%fit_regression(x, target, status, xgb_options)
    call check(status_ok(status), "XGBoost regression fit", failures)

    call boosted_partial_dependence(xgb, x, 1, grid, average, status, &
        individual=ice)
    call check(status_ok(status), "XGBoost PDP status", failures)
    call check(maxval(abs(average-[5.0_dp/3.0_dp, 25.0_dp/3.0_dp])) < tolerance, &
        "XGBoost hand-computed PDP", failures)
    expected_ice(:, 1) = 5.0_dp/3.0_dp
    expected_ice(:, 2) = 25.0_dp/3.0_dp
    call check(maxval(abs(ice-expected_ice)) < tolerance, &
        "XGBoost hand-computed ICE", failures)

    grid = [-1.0_dp, 1.0_dp]
    expected_ice(:, 1) = [5.0_dp/3.0_dp, 5.0_dp/3.0_dp, &
        25.0_dp/3.0_dp, 25.0_dp/3.0_dp]
    expected_ice(:, 2) = expected_ice(:, 1)
    call boosted_partial_dependence(xgb, x, 2, grid, average, status, weight, ice)
    call check(status_ok(status), "XGBoost weighted PDP status", failures)
    call check(maxval(abs(ice-expected_ice)) < tolerance, &
        "XGBoost weighted ICE replay", failures)
    call check(maxval(abs(average-55.0_dp/9.0_dp)) < tolerance, &
        "XGBoost weighted mean oracle", failures)

    lgb_options = lightgbm_options_t()
    lgb_options%n_estimators = 1
    lgb_options%num_leaves = 2
    lgb_options%min_data_in_leaf = 1
    lgb_options%max_bin = 16
    lgb_options%l2 = 1.0_dp
    lgb_options%learning_rate = 1.0_dp
    call lgb%fit_regression(x, target, status, lgb_options)
    call check(status_ok(status), "LightGBM regression fit", failures)
    call boosted_partial_dependence(lgb, x, 2, grid, average, status, weight, ice)
    call check(status_ok(status), "LightGBM weighted PDP status", failures)
    call check(maxval(abs(ice-expected_ice)) < tolerance, &
        "LightGBM weighted ICE replay", failures)
    call check(maxval(abs(average-55.0_dp/9.0_dp)) < tolerance, &
        "LightGBM weighted mean oracle", failures)

    xgb_binary_options = xgb_options
    xgb_binary_options%objective = "logistic"
    call xgb_binary%fit_binary(x, binary_target, status, xgb_binary_options)
    call check(status_ok(status), "XGBoost binary fit", failures)
    grid = [0.0_dp, 3.0_dp]
    call boosted_partial_dependence(xgb_binary, x, 1, grid, margin_average, &
        status, response=FORTML_TREE_RESPONSE_MARGIN)
    call check(status_ok(status), "XGBoost margin PDP status", failures)
    call boosted_partial_dependence(xgb_binary, x, 1, grid, prediction_average, &
        status, response=FORTML_TREE_RESPONSE_PREDICTION)
    call check(status_ok(status), "XGBoost response PDP status", failures)
    logistic_oracle = 1.0_dp/(1.0_dp + exp(-margin_average))
    call check(maxval(abs(prediction_average-logistic_oracle)) < tolerance, &
        "XGBoost response-link oracle", failures)

    lgb_binary_options = lgb_options
    lgb_binary_options%objective = "binary"
    call lgb_binary%fit_binary(x, binary_target, status, lgb_binary_options)
    call check(status_ok(status), "LightGBM binary fit", failures)
    call boosted_partial_dependence(lgb_binary, x, 1, grid, margin_average, &
        status, response=FORTML_TREE_RESPONSE_MARGIN)
    call check(status_ok(status), "LightGBM margin PDP status", failures)
    call boosted_partial_dependence(lgb_binary, x, 1, grid, prediction_average, &
        status, response=FORTML_TREE_RESPONSE_PREDICTION)
    call check(status_ok(status), "LightGBM response PDP status", failures)
    logistic_oracle = 1.0_dp/(1.0_dp + exp(-margin_average))
    call check(maxval(abs(prediction_average-logistic_oracle)) < tolerance, &
        "LightGBM response-link oracle", failures)

    cpu%kind = FORTML_DEVICE_CPU
    cpu%selected = .true.
    cpu%available = .true.
    call boosted_partial_dependence(xgb, x, 1, grid, average, status, device=cpu)
    call check(status_ok(status), "explicit CPU PDP", failures)
    call check(boosted_partial_dependence_device_supported(FORTML_DEVICE_CPU), &
        "CPU capability query", failures)
    call check(.not. boosted_partial_dependence_device_supported(FORTML_DEVICE_CUDA), &
        "CUDA capability query", failures)

    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    sentinel_average = -123.0_dp
    sentinel_ice = -456.0_dp
    average = sentinel_average
    ice = sentinel_ice
    call boosted_partial_dependence(xgb, x, 1, grid, average, status, &
        individual=ice, device=cuda)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, "CUDA typed refusal", failures)
    call check(all(average == sentinel_average) .and. all(ice == sentinel_ice), &
        "CUDA refusal is transactional", failures)

    weight = 0.0_dp
    call boosted_partial_dependence(lgb, x, 1, grid, average, status, &
        sample_weight=weight)
    call check(status%code == FORTNUM_DOMAIN_ERROR, "zero-weight refusal", failures)
    call check(all(average == sentinel_average), &
        "weight refusal preserves output", failures)
    call boosted_partial_dependence(unfitted_xgb, x, 1, grid, average, status)
    call check(status%code == FORTNUM_DOMAIN_ERROR, "unfitted-model refusal", failures)
    call check(all(average == sentinel_average), &
        "unfitted refusal preserves output", failures)

    if (failures > 0) then
        write (error_unit, '(a,i0)') "FAIL boosted partial-dependence cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS boosted partial-dependence independent behavioral oracle"

contains

    subroutine check(condition, label, failure_count)
        logical, intent(in) :: condition
        character(*), intent(in) :: label
        integer, intent(inout) :: failure_count

        if (.not. condition) then
            failure_count = failure_count + 1
            write (error_unit, '(a)') "FAIL: " // trim(label)
        end if
    end subroutine check

end program test_boosted_partial_dependence
