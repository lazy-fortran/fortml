program test_mlp_adamw_hypergradient
    !! Independent finite-difference and scalar-adjoint checks for AdamW
    !! trajectory hypergradients.
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: FORTML_DEVICE_CUDA
    use fortml_mlp, only: mlp_t
    use fortml_mlp_training, only: MLP_OPTIMIZER_SGD
    use fortml_mlp_hypergradient, only: &
        mlp_adamw_hypergradient_objective_t, mlp_adamw_hypergradient_options_t, &
        mlp_adamw_hypergradient_result_t, mlp_adamw_hypergradient_metadata_t, &
        mlp_optimize_adamw_hyperparameters, MLP_ADAMW_HYPERPARAMETER_COUNT, &
        MLP_ADAMW_LOG_LEARNING_RATE, MLP_ADAMW_LOG_L2, &
        MLP_ADAMW_LOG_WEIGHT_DECAY, mlp_adamw_full_hypergradient_objective_t, &
        mlp_adamw_full_hypergradient_options_t, mlp_adamw_full_hypergradient_metadata_t, &
        MLP_ADAMW_FULL_HYPERPARAMETER_COUNT, MLP_ADAMW_FULL_LOGIT_BETA1, &
        MLP_ADAMW_FULL_LOGIT_BETA2
    implicit none

    type(mlp_t), target :: model
    type(mlp_t), target :: full_model
    type(mlp_t), target :: nonlinear_model
    type(mlp_adamw_hypergradient_objective_t) :: objective
    type(mlp_adamw_full_hypergradient_objective_t) :: full_objective
    type(mlp_adamw_hypergradient_options_t) :: options, bad_options
    type(mlp_adamw_full_hypergradient_options_t) :: full_options, bad_full_options
    type(mlp_adamw_hypergradient_result_t) :: result
    type(mlp_adamw_hypergradient_metadata_t) :: metadata
    type(mlp_adamw_full_hypergradient_metadata_t) :: full_metadata
    type(fortnum_status_t) :: status
    real(dp) :: train_x(5, 1), train_target(5, 1)
    real(dp) :: validation_x(3, 1), validation_target(3, 1)
    real(dp) :: parameters(3), direction(3), gradient(3), vjp_gradient(3)
    real(dp) :: full_parameters(5), full_direction(5), full_gradient(5), full_vjp_gradient(5)
    real(dp) :: full_hvp(5), full_gradient_plus(5), full_gradient_minus(5)
    real(dp) :: value, value_plus, value_minus, tangent, h
    integer :: i, failures

    failures = 0
    train_x(:, 1) = [-2.0_dp, -1.0_dp, 0.0_dp, 1.0_dp, 2.0_dp]
    train_target(:, 1) = 0.7_dp*train_x(:, 1) - 0.2_dp
    validation_x(:, 1) = [-1.5_dp, 0.5_dp, 1.75_dp]
    validation_target(:, 1) = 0.7_dp*validation_x(:, 1) - 0.2_dp
    call model%initialize([1, 1], status, initialization_seed=23)
    call model%set_parameters([0.15_dp, -0.1_dp], status)
    options%steps = 4
    options%learning_rate = 0.12_dp
    options%l2 = 0.07_dp
    options%weight_decay = 0.03_dp
    options%lower_log_learning_rate = -4.0_dp
    options%upper_log_learning_rate = 0.0_dp
    options%lower_log_l2 = -5.0_dp
    options%upper_log_l2 = 0.0_dp
    options%lower_log_weight_decay = -6.0_dp
    options%upper_log_weight_decay = 0.0_dp
    call objective%initialize(model, train_x, train_target, validation_x, &
        validation_target, options, status)
    call check(status_ok(status), "AdamW hypergradient initialization", failures)
    call check(objective%parameter_count() == MLP_ADAMW_HYPERPARAMETER_COUNT, &
        "AdamW packed parameter count", failures)
    metadata = objective%metadata()
    call check(metadata%log_learning_rate_index == MLP_ADAMW_LOG_LEARNING_RATE .and. &
        metadata%log_l2_index == MLP_ADAMW_LOG_L2 .and. &
        metadata%log_weight_decay_index == MLP_ADAMW_LOG_WEIGHT_DECAY .and. &
        metadata%inner_steps == 4, "AdamW packed metadata", failures)
    parameters = objective%parameters()
    call objective%value_gradient(parameters, value, gradient, status)
    call check(status_ok(status), "AdamW reverse value/gradient", failures)
    h = 2.0e-6_dp
    do i = 1, MLP_ADAMW_HYPERPARAMETER_COUNT
        parameters(i) = parameters(i) + h
        call objective%value_gradient(parameters, value_plus, vjp_gradient, status)
        parameters(i) = parameters(i) - 2.0_dp*h
        call objective%value_gradient(parameters, value_minus, vjp_gradient, status)
        parameters(i) = parameters(i) + h
        call check(abs(gradient(i) - (value_plus - value_minus)/(2.0_dp*h)) &
            < 5.0e-7_dp, "AdamW hypergradient central difference", failures)
    end do
    direction = [0.31_dp, -0.27_dp, 0.19_dp]
    call objective%jvp(parameters, direction, value, tangent, status)
    call objective%value_gradient(parameters + h*direction, value_plus, &
        vjp_gradient, status)
    call objective%value_gradient(parameters - h*direction, value_minus, &
        vjp_gradient, status)
    call check(abs(tangent - (value_plus - value_minus)/(2.0_dp*h)) < 7.0e-7_dp, &
        "AdamW forward JVP central difference", failures)
    call objective%vjp(parameters, 1.7_dp, vjp_gradient, status)
    call objective%value_gradient(parameters, value, gradient, status)
    call check(maxval(abs(vjp_gradient - 1.7_dp*gradient)) < 2.0e-12_dp, &
        "AdamW scalar VJP adjoint", failures)
    options%max_iterations = 80
    options%gradient_tolerance = 1.0e-5_dp
    call mlp_optimize_adamw_hyperparameters(model, train_x, train_target, &
        validation_x, validation_target, options, result, status)
    call check(status_ok(status), "AdamW FortOpt L-BFGS-B solve", failures)
    call check(result%converged .and. result%learning_rate > 0.0_dp .and. &
        result%l2 > 0.0_dp .and. result%weight_decay > 0.0_dp, &
        "AdamW FortOpt result", failures)

    ! The five-parameter contract differentiates both moment coefficients
    ! through unconstrained beta logits.  The oracle below only evaluates
    ! the public objective twice per coordinate, so it is independent of the
    ! implementation's forward/reverse sensitivity code.
    call full_model%initialize([1, 1], status, initialization_seed=23)
    call full_model%set_parameters([0.15_dp, -0.1_dp], status)
    full_options%steps = 4
    full_options%learning_rate = 0.12_dp
    full_options%l2 = 0.07_dp
    full_options%weight_decay = 0.03_dp
    full_options%beta1 = 0.82_dp
    full_options%beta2 = 0.91_dp
    full_options%lower_log_learning_rate = -4.0_dp
    full_options%upper_log_learning_rate = 0.0_dp
    full_options%lower_log_l2 = -5.0_dp
    full_options%upper_log_l2 = 0.0_dp
    full_options%lower_log_weight_decay = -6.0_dp
    full_options%upper_log_weight_decay = 0.0_dp
    full_options%lower_logit_beta1 = -4.0_dp
    full_options%upper_logit_beta1 = 4.0_dp
    full_options%lower_logit_beta2 = -4.0_dp
    full_options%upper_logit_beta2 = 4.0_dp
    call full_objective%initialize(full_model, train_x, train_target, validation_x, &
        validation_target, full_options, status)
    call check(status_ok(status), "full AdamW hypergradient initialization", failures)
    full_metadata = full_objective%metadata()
    call check(full_objective%parameter_count() == MLP_ADAMW_FULL_HYPERPARAMETER_COUNT .and. &
        full_metadata%logit_beta1_index == MLP_ADAMW_FULL_LOGIT_BETA1 .and. &
        full_metadata%logit_beta2_index == MLP_ADAMW_FULL_LOGIT_BETA2 .and. &
        full_metadata%inner_steps == 4, "full AdamW packed metadata", failures)
    full_parameters = full_objective%parameters()
    call full_objective%value_gradient(full_parameters, value, full_gradient, status)
    call check(status_ok(status), "full AdamW value/gradient", failures)
    h = 2.0e-6_dp
    do i = 1, MLP_ADAMW_FULL_HYPERPARAMETER_COUNT
        full_parameters(i) = full_parameters(i) + h
        call full_objective%value_gradient(full_parameters, value_plus, full_vjp_gradient, status)
        full_parameters(i) = full_parameters(i) - 2.0_dp*h
        call full_objective%value_gradient(full_parameters, value_minus, full_vjp_gradient, status)
        full_parameters(i) = full_parameters(i) + h
        call check(abs(full_gradient(i) - (value_plus-value_minus)/(2.0_dp*h)) < 2.0e-6_dp, &
            "full AdamW beta hypergradient central difference", failures)
    end do
    full_direction = [0.31_dp, -0.27_dp, 0.19_dp, 0.13_dp, -0.22_dp]
    call full_objective%jvp(full_parameters, full_direction, value, tangent, status)
    call check(status_ok(status), "full AdamW forward JVP", failures)
    call full_objective%vjp(full_parameters, 1.7_dp, full_vjp_gradient, status)
    call full_objective%value_gradient(full_parameters, value, full_gradient, status)
    call check(maxval(abs(full_vjp_gradient-1.7_dp*full_gradient)) < 2.0e-12_dp, &
        "full AdamW scalar VJP adjoint", failures)
    call full_objective%hvp(full_parameters, full_direction, full_hvp, status)
    call check(status_ok(status), "full AdamW affine outer HVP", failures)
    call full_objective%value_gradient(full_parameters+h*full_direction, value_plus, &
        full_gradient_plus, status)
    call full_objective%value_gradient(full_parameters-h*full_direction, value_minus, &
        full_gradient_minus, status)
    call check(maxval(abs(full_hvp-(full_gradient_plus-full_gradient_minus)/(2.0_dp*h))) &
        < 3.0e-5_dp, "full AdamW outer HVP central difference", failures)
    full_direction = [0.31_dp, -0.27_dp, 0.19_dp, 0.13_dp, -0.22_dp]

    call nonlinear_model%initialize([1, 2, 1], status, initialization_seed=23)
    call full_objective%initialize(nonlinear_model, train_x, train_target, validation_x, &
        validation_target, full_options, status)
    call check(status_ok(status), "full AdamW nonlinear HVP setup", failures)
    full_parameters = full_objective%parameters()
    call full_objective%hvp(full_parameters, full_direction, full_hvp, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "full AdamW nonlinear HVP typed refusal", failures)
    bad_full_options = full_options
    bad_full_options%device_kind = FORTML_DEVICE_CUDA
    call full_objective%initialize(full_model, train_x, train_target, validation_x, &
        validation_target, bad_full_options, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "full AdamW CUDA typed refusal", failures)
    bad_options = options
    bad_options%optimizer = MLP_OPTIMIZER_SGD
    call objective%initialize(model, train_x, train_target, validation_x, &
        validation_target, bad_options, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "non-AdamW hypergradient refusal", failures)
    if (failures > 0) error stop 1
    write (*, '(a)') "PASS MLP AdamW hypergradient independent behavioral oracles"

contains

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: description
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (*, '(a)') "FAIL "//trim(description)
        end if
    end subroutine check

end program test_mlp_adamw_hypergradient
