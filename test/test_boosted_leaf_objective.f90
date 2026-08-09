program test_boosted_leaf_objective
    !! Independent weighted-loss oracle for fixed tree-coordinate objectives.
    !! The expected design matrix is the hand-stump routing matrix; fitted
    !! leaf values and private tree arrays are never used by the oracle.
    use, intrinsic :: iso_fortran_env, only: real64, error_unit
    use fortml_boosted_leaf_objective, only: boosted_leaf_objective_t, &
        boosted_leaf_lbfgsb_options_t, boosted_leaf_lbfgsb_result_t, &
        boosted_leaf_optimize_lbfgsb, BOOSTED_LEAF_LOSS_SQUARED, &
        BOOSTED_LEAF_LOSS_LOGISTIC, BOOSTED_LEAF_MODEL_XGBOOST, &
        BOOSTED_LEAF_MODEL_LIGHTGBM
    use fortml_device, only: FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    use fortml_xgboost, only: xgboost_t, xgboost_options_t
    use fortml_lightgbm, only: lightgbm_t, lightgbm_options_t
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_DOMAIN_ERROR, &
        FORTNUM_NOT_IMPLEMENTED
    implicit none

    integer, parameter :: dp = real64
    type(xgboost_t) :: xgb
    type(lightgbm_t) :: lgbm
    type(xgboost_options_t) :: xgb_options
    type(lightgbm_options_t) :: lgbm_options
    type(boosted_leaf_objective_t) :: objective
    type(boosted_leaf_lbfgsb_options_t) :: optimizer_options
    type(boosted_leaf_lbfgsb_result_t) :: optimizer_result
    type(fortnum_status_t) :: status
    real(dp) :: x(4, 2), y_squared(4), y_logistic(4), weights(4)
    real(dp) :: parameters(3), direction(3), gradient(3), hvp(3), vjp(3)
    real(dp) :: expected_gradient(3), expected_hvp(3), expected_value
    real(dp) :: value, tangent, finite_difference(3), plus(3), minus(3)
    real(dp) :: step, output_bar
    integer :: failures

    failures = 0
    x(:, 1) = [0.0_dp, 0.0_dp, 1.0_dp, 1.0_dp]
    x(:, 2) = 0.0_dp
    y_squared = [1.0_dp, 1.0_dp, 3.0_dp, 3.0_dp]
    y_logistic = [0.0_dp, 0.0_dp, 1.0_dp, 1.0_dp]
    weights = [1.0_dp, 2.0_dp, 3.0_dp, 4.0_dp]
    parameters = [0.4_dp, -0.2_dp, 0.7_dp]
    direction = [0.2_dp, -0.3_dp, 0.5_dp]
    output_bar = 1.7_dp
    step = 1.0e-6_dp

    xgb_options%n_estimators = 1
    xgb_options%max_depth = 1
    xgb_options%min_child_weight = 0.0_dp
    xgb_options%learning_rate = 1.0_dp
    xgb_options%l2 = 0.0_dp
    call xgb%fit_regression(x, y_squared, status, xgb_options)
    call check(status_ok(status), "XGBoost stump fit", failures)
    call objective%initialize_xgboost(xgb, x, y_squared, &
        BOOSTED_LEAF_LOSS_SQUARED, status, sample_weight=weights, l2=0.2_dp)
    call check(status_ok(status) .and. objective%initialized(), &
        "XGBoost squared objective initialization", failures)
    call check(objective%model_kind() == BOOSTED_LEAF_MODEL_XGBOOST, &
        "XGBoost model metadata", failures)
    call check(objective%parameter_count() == 3, &
        "XGBoost objective coordinate count", failures)
    call oracle_squared(parameters, y_squared, weights, 0.2_dp, expected_value, &
        expected_gradient, expected_hvp, direction)
    call objective%value_gradient(parameters, value, gradient, status)
    call check(status_ok(status) .and. abs(value-expected_value) < 2.0e-12_dp, &
        "XGBoost squared value oracle", failures)
    call check(status_ok(status) .and. maxval(abs(gradient-expected_gradient)) < &
        2.0e-12_dp, "XGBoost squared gradient oracle", failures)
    call objective%jvp(parameters, direction, value, tangent, status)
    call check(status_ok(status) .and. abs(tangent-dot_product(expected_gradient, &
        direction)) < 2.0e-12_dp, "XGBoost squared JVP oracle", failures)
    call objective%vjp(parameters, output_bar, vjp, status)
    call check(status_ok(status) .and. maxval(abs(vjp-output_bar*expected_gradient)) < &
        2.0e-12_dp, "XGBoost squared VJP oracle", failures)
    call objective%hvp(parameters, direction, hvp, status)
    call check(status_ok(status) .and. maxval(abs(hvp-expected_hvp)) < 2.0e-12_dp, &
        "XGBoost squared HVP oracle", failures)
    plus = parameters + step*direction
    minus = parameters - step*direction
    call objective%value_gradient(plus, value, gradient, status)
    call objective%value_gradient(minus, tangent, finite_difference, status)
    finite_difference = (gradient-finite_difference)/(2.0_dp*step)
    call check(maxval(abs(finite_difference-expected_hvp)) < 2.0e-7_dp, &
        "XGBoost squared finite-difference HVP", failures)

    call xgb%fit_binary(x, y_logistic, status, xgb_options)
    call check(status_ok(status), "XGBoost logistic stump fit", failures)
    call objective%initialize(xgb, x, y_logistic, BOOSTED_LEAF_LOSS_LOGISTIC, status, &
        sample_weight=weights, l2=0.1_dp)
    call check(status_ok(status), "XGBoost logistic objective initialization", failures)
    call oracle_logistic(parameters, y_logistic, weights, 0.1_dp, expected_value, &
        expected_gradient, expected_hvp, direction)
    call objective%value_gradient(parameters, value, gradient, status)
    call check(status_ok(status) .and. abs(value-expected_value) < 2.0e-12_dp, &
        "XGBoost logistic value oracle", failures)
    call check(maxval(abs(gradient-expected_gradient)) < 2.0e-12_dp, &
        "XGBoost logistic gradient oracle", failures)
    call objective%hvp(parameters, direction, hvp, status)
    call check(status_ok(status) .and. maxval(abs(hvp-expected_hvp)) < 2.0e-12_dp, &
        "XGBoost logistic HVP oracle", failures)

    lgbm_options%n_estimators = 1
    lgbm_options%num_leaves = 2
    lgbm_options%min_data_in_leaf = 1
    lgbm_options%max_bin = 16
    lgbm_options%learning_rate = 1.0_dp
    lgbm_options%l2 = 0.0_dp
    call lgbm%fit_regression(x, y_squared, status, lgbm_options)
    call check(status_ok(status), "LightGBM stump fit", failures)
    call objective%initialize_lightgbm(lgbm, x, y_squared, BOOSTED_LEAF_LOSS_SQUARED, &
        status, sample_weight=weights, l2=0.2_dp)
    call check(status_ok(status), "LightGBM squared objective initialization", failures)
    call oracle_squared(parameters, y_squared, weights, 0.2_dp, expected_value, &
        expected_gradient, expected_hvp, direction)
    call objective%value_gradient(parameters, value, gradient, status)
    call check(status_ok(status) .and. abs(value-expected_value) < 2.0e-12_dp, &
        "LightGBM squared value oracle", failures)
    call check(maxval(abs(gradient-expected_gradient)) < 2.0e-12_dp, &
        "LightGBM squared gradient oracle", failures)
    call objective%hvp(parameters, direction, hvp, status)
    call check(status_ok(status) .and. maxval(abs(hvp-expected_hvp)) < 2.0e-12_dp, &
        "LightGBM squared HVP oracle", failures)

    call lgbm%fit_binary(x, y_logistic, status, lgbm_options)
    call check(status_ok(status), "LightGBM logistic stump fit", failures)
    call objective%initialize(lgbm, x, y_logistic, BOOSTED_LEAF_LOSS_LOGISTIC, status, &
        sample_weight=weights, l2=0.1_dp)
    call check(status_ok(status), "LightGBM logistic objective initialization", failures)
    call oracle_logistic(parameters, y_logistic, weights, 0.1_dp, expected_value, &
        expected_gradient, expected_hvp, direction)
    call objective%value_gradient(parameters, value, gradient, status)
    call check(status_ok(status) .and. abs(value-expected_value) < 2.0e-12_dp, &
        "LightGBM logistic value oracle", failures)
    call check(maxval(abs(gradient-expected_gradient)) < 2.0e-12_dp, &
        "LightGBM logistic gradient oracle", failures)
    call objective%hvp(parameters, direction, hvp, status)
    call check(status_ok(status) .and. maxval(abs(hvp-expected_hvp)) < 2.0e-12_dp, &
        "LightGBM logistic HVP oracle", failures)

    optimizer_options%max_iterations = 200
    optimizer_options%gradient_tolerance = 1.0e-7_dp
    optimizer_options%l2 = 0.1_dp
    call boosted_leaf_optimize_lbfgsb(xgb, x, y_logistic, BOOSTED_LEAF_LOSS_LOGISTIC, &
        optimizer_options, optimizer_result, status, weights)
    call check(status_ok(status) .and. optimizer_result%converged .and. &
        allocated(optimizer_result%parameters), &
        "XGBoost fixed-leaf FortOpt optimization", failures)

    optimizer_options%device_kind = FORTML_DEVICE_CUDA
    call objective%initialize_xgboost(xgb, x, y_logistic, BOOSTED_LEAF_LOSS_LOGISTIC, &
        status, device_kind=FORTML_DEVICE_CUDA)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "XGBoost fixed-leaf CUDA refusal", failures)
    optimizer_options%device_kind = FORTML_DEVICE_CPU
    call objective%initialize_xgboost(xgb, x, y_logistic, BOOSTED_LEAF_LOSS_LOGISTIC, &
        status, sample_weight=[0.0_dp, 0.0_dp, 0.0_dp, 0.0_dp])
    call check(status%code == FORTNUM_DOMAIN_ERROR .and. .not. objective%initialized(), &
        "zero-weight objective refusal", failures)

    if (failures > 0) then
        write (error_unit, '(a,i0)') "FAIL boosted leaf objective cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS boosted leaf objective independent behavioral oracle"

contains

    subroutine oracle_squared(theta, target, weight, l2, value, gradient, hvp, direction)
        real(dp), intent(in) :: theta(:), target(:), weight(:), l2, direction(:)
        real(dp), intent(out) :: value, gradient(:), hvp(:)
        real(dp) :: design(4, 3), margin(4), residual(4), tangent(4), total

        design = reshape([1.0_dp, 1.0_dp, 1.0_dp, 1.0_dp, &
            1.0_dp, 1.0_dp, 0.0_dp, 0.0_dp, &
            0.0_dp, 0.0_dp, 1.0_dp, 1.0_dp], [4, 3])
        total = sum(weight)
        margin = matmul(design, theta)
        residual = margin-target
        value = 0.5_dp*sum(weight*residual*residual)/total + &
            0.5_dp*l2*dot_product(theta, theta)
        gradient = matmul(transpose(design), weight*residual)/total + l2*theta
        tangent = matmul(design, direction)
        hvp = matmul(transpose(design), weight*tangent)/total + l2*direction
    end subroutine oracle_squared

    subroutine oracle_logistic(theta, target, weight, l2, value, gradient, hvp, direction)
        real(dp), intent(in) :: theta(:), target(:), weight(:), l2, direction(:)
        real(dp), intent(out) :: value, gradient(:), hvp(:)
        real(dp) :: design(4, 3), margin(4), probability(4), tangent(4), total
        real(dp) :: loss(4), curvature(4)

        design = reshape([1.0_dp, 1.0_dp, 1.0_dp, 1.0_dp, &
            1.0_dp, 1.0_dp, 0.0_dp, 0.0_dp, &
            0.0_dp, 0.0_dp, 1.0_dp, 1.0_dp], [4, 3])
        total = sum(weight)
        margin = matmul(design, theta)
        probability = 1.0_dp/(1.0_dp + exp(-margin))
        loss = max(margin, 0.0_dp) + log(1.0_dp + exp(-abs(margin))) - &
            target*margin
        curvature = weight*probability*(1.0_dp-probability)
        value = sum(weight*loss)/total + 0.5_dp*l2*dot_product(theta, theta)
        gradient = matmul(transpose(design), weight*(probability-target))/total + l2*theta
        tangent = matmul(design, direction)
        hvp = matmul(transpose(design), curvature*tangent)/total + l2*direction
    end subroutine oracle_logistic

    subroutine check(condition, label, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: label
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "FAIL: " // trim(label)
        end if
    end subroutine check

end program test_boosted_leaf_objective
