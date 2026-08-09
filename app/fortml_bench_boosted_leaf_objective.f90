program fortml_bench_boosted_leaf_objective
    !! Release probe for fixed-structure XGBoost/LightGBM leaf objectives.
    use, intrinsic :: iso_fortran_env, only: real64
    use fortml_boosted_leaf_objective, only: boosted_leaf_objective_t, &
        boosted_leaf_lbfgsb_options_t, boosted_leaf_lbfgsb_result_t, &
        boosted_leaf_optimize_lbfgsb, BOOSTED_LEAF_LOSS_SQUARED, &
        BOOSTED_LEAF_LOSS_LOGISTIC
    use fortml_device, only: FORTML_DEVICE_CUDA
    use fortml_xgboost, only: xgboost_t, xgboost_options_t
    use fortml_lightgbm, only: lightgbm_t, lightgbm_options_t
    use fortnum_status, only: fortnum_status_t, status_ok
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
    real(dp) :: value, tangent, output_bar

    x(:, 1) = [0.0_dp, 0.0_dp, 1.0_dp, 1.0_dp]
    x(:, 2) = 0.0_dp
    y_squared = [1.0_dp, 1.0_dp, 3.0_dp, 3.0_dp]
    y_logistic = [0.0_dp, 0.0_dp, 1.0_dp, 1.0_dp]
    weights = [1.0_dp, 2.0_dp, 3.0_dp, 4.0_dp]
    parameters = [0.4_dp, -0.2_dp, 0.7_dp]
    direction = [0.2_dp, -0.3_dp, 0.5_dp]
    output_bar = 1.7_dp

    xgb_options%n_estimators = 1
    xgb_options%max_depth = 1
    xgb_options%min_child_weight = 0.0_dp
    xgb_options%learning_rate = 1.0_dp
    xgb_options%l2 = 0.0_dp
    call xgb%fit_regression(x, y_squared, status, xgb_options)
    if (.not. status_ok(status)) error stop "xgb squared fit failed"
    call objective%initialize_xgboost(xgb, x, y_squared, BOOSTED_LEAF_LOSS_SQUARED, &
        status, sample_weight=weights, l2=0.2_dp)
    if (.not. status_ok(status)) error stop "xgb squared objective failed"
    call oracle_squared(parameters, y_squared, weights, 0.2_dp, expected_value, &
        expected_gradient, expected_hvp, direction)
    call objective%value_gradient(parameters, value, gradient, status)
    call objective%jvp(parameters, direction, value, tangent, status)
    call objective%vjp(parameters, output_bar, vjp, status)
    call objective%hvp(parameters, direction, hvp, status)
    call emit("xgb_squared", value, gradient, tangent, vjp, hvp, expected_value, &
        expected_gradient, expected_hvp, status)

    call xgb%fit_binary(x, y_logistic, status, xgb_options)
    if (.not. status_ok(status)) error stop "xgb logistic fit failed"
    call objective%initialize_xgboost(xgb, x, y_logistic, BOOSTED_LEAF_LOSS_LOGISTIC, &
        status, sample_weight=weights, l2=0.1_dp)
    if (.not. status_ok(status)) error stop "xgb logistic objective failed"
    call oracle_logistic(parameters, y_logistic, weights, 0.1_dp, expected_value, &
        expected_gradient, expected_hvp, direction)
    call objective%value_gradient(parameters, value, gradient, status)
    call objective%jvp(parameters, direction, value, tangent, status)
    call objective%vjp(parameters, output_bar, vjp, status)
    call objective%hvp(parameters, direction, hvp, status)
    call emit("xgb_logistic", value, gradient, tangent, vjp, hvp, expected_value, &
        expected_gradient, expected_hvp, status)

    lgbm_options%n_estimators = 1
    lgbm_options%num_leaves = 2
    lgbm_options%min_data_in_leaf = 1
    lgbm_options%max_bin = 16
    lgbm_options%learning_rate = 1.0_dp
    lgbm_options%l2 = 0.0_dp
    call lgbm%fit_regression(x, y_squared, status, lgbm_options)
    if (.not. status_ok(status)) error stop "lightgbm squared fit failed"
    call objective%initialize_lightgbm(lgbm, x, y_squared, BOOSTED_LEAF_LOSS_SQUARED, &
        status, sample_weight=weights, l2=0.2_dp)
    if (.not. status_ok(status)) error stop "lightgbm squared objective failed"
    call oracle_squared(parameters, y_squared, weights, 0.2_dp, expected_value, &
        expected_gradient, expected_hvp, direction)
    call objective%value_gradient(parameters, value, gradient, status)
    call objective%jvp(parameters, direction, value, tangent, status)
    call objective%vjp(parameters, output_bar, vjp, status)
    call objective%hvp(parameters, direction, hvp, status)
    call emit("lgbm_squared", value, gradient, tangent, vjp, hvp, expected_value, &
        expected_gradient, expected_hvp, status)

    call lgbm%fit_binary(x, y_logistic, status, lgbm_options)
    if (.not. status_ok(status)) error stop "lightgbm logistic fit failed"
    call objective%initialize_lightgbm(lgbm, x, y_logistic, BOOSTED_LEAF_LOSS_LOGISTIC, &
        status, sample_weight=weights, l2=0.1_dp)
    if (.not. status_ok(status)) error stop "lightgbm logistic objective failed"
    call oracle_logistic(parameters, y_logistic, weights, 0.1_dp, expected_value, &
        expected_gradient, expected_hvp, direction)
    call objective%value_gradient(parameters, value, gradient, status)
    call objective%jvp(parameters, direction, value, tangent, status)
    call objective%vjp(parameters, output_bar, vjp, status)
    call objective%hvp(parameters, direction, hvp, status)
    call emit("lgbm_logistic", value, gradient, tangent, vjp, hvp, expected_value, &
        expected_gradient, expected_hvp, status)

    optimizer_options%max_iterations = 200
    optimizer_options%gradient_tolerance = 1.0e-7_dp
    optimizer_options%l2 = 0.1_dp
    call boosted_leaf_optimize_lbfgsb(xgb, x, y_logistic, BOOSTED_LEAF_LOSS_LOGISTIC, &
        optimizer_options, optimizer_result, status, weights)
    write (*, '(a,1x,i0)') "leaf_optimize_status", status%code
    write (*, '(a,1x,l1)') "leaf_optimize_converged", optimizer_result%converged
    write (*, '(a,1x,i0)') "leaf_optimize_iterations", optimizer_result%iterations
    write (*, '(a,1x,es24.16)') "leaf_optimize_gradient_norm", &
        optimizer_result%gradient_norm
    write (*, '(a,1x,i0)') "leaf_optimize_parameter_count", &
        size(optimizer_result%parameters)
    optimizer_options%device_kind = FORTML_DEVICE_CUDA
    call objective%initialize_xgboost(xgb, x, y_logistic, BOOSTED_LEAF_LOSS_LOGISTIC, &
        status, device_kind=FORTML_DEVICE_CUDA)
    write (*, '(a,1x,i0)') "leaf_cuda_status", status%code
contains

    subroutine emit(prefix, value, gradient, tangent, vjp, hvp, expected_value, &
            expected_gradient, expected_hvp, status)
        character(*), intent(in) :: prefix
        real(dp), intent(in) :: value, gradient(:), tangent, vjp(:), hvp(:)
        real(dp), intent(in) :: expected_value, expected_gradient(:), expected_hvp(:)
        type(fortnum_status_t), intent(in) :: status

        write (*, '(a,1x,i0)') trim(prefix)//"_status", status%code
        write (*, '(a,1x,es24.16)') trim(prefix)//"_value", value
        write (*, '(a,1x,es24.16)') trim(prefix)//"_value_error", &
            abs(value-expected_value)
        write (*, '(a,1x,es24.16)') trim(prefix)//"_gradient_error", &
            maxval(abs(gradient-expected_gradient))
        write (*, '(a,1x,es24.16)') trim(prefix)//"_jvp_error", &
            abs(tangent-dot_product(expected_gradient, direction))
        write (*, '(a,1x,es24.16)') trim(prefix)//"_vjp_error", &
            maxval(abs(vjp-output_bar*expected_gradient))
        write (*, '(a,1x,es24.16)') trim(prefix)//"_hvp_error", &
            maxval(abs(hvp-expected_hvp))
    end subroutine emit

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

end program fortml_bench_boosted_leaf_objective
