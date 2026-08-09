program test_mlp_adagrad_hypergradient
    !! Independent finite-difference and adjoint checks for Adagrad trajectory
    !! hypergradients and the FortOpt L-BFGS-B adapter.
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_DOMAIN_ERROR, &
        FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: FORTML_DEVICE_CUDA
    use fortml_mlp, only: mlp_t, MLP_LINEAR
    use fortml_mlp_training, only: MLP_OPTIMIZER_ADAM
    use fortml_mlp_adagrad_hypergradient, only: &
        mlp_adagrad_hypergradient_objective_t, &
        mlp_adagrad_hypergradient_options_t, mlp_adagrad_hypergradient_result_t, &
        mlp_adagrad_hypergradient_metadata_t, mlp_optimize_adagrad_hyperparameters, &
        MLP_ADAGRAD_HYPERPARAMETER_COUNT, MLP_ADAGRAD_LOG_LEARNING_RATE, &
        MLP_ADAGRAD_LOG_L2, MLP_ADAGRAD_LOG_EPSILON
    implicit none

    type(mlp_t), target :: model, affine_model
    type(mlp_adagrad_hypergradient_objective_t) :: objective
    type(mlp_adagrad_hypergradient_options_t) :: options, bad_options
    type(mlp_adagrad_hypergradient_result_t) :: result
    type(mlp_adagrad_hypergradient_metadata_t) :: metadata
    type(fortnum_status_t) :: status
    real(dp) :: train_x(5, 1), train_target(5, 1)
    real(dp) :: validation_x(3, 1), validation_target(3, 1)
    real(dp) :: parameters(MLP_ADAGRAD_HYPERPARAMETER_COUNT)
    real(dp) :: direction(MLP_ADAGRAD_HYPERPARAMETER_COUNT)
    real(dp) :: gradient(MLP_ADAGRAD_HYPERPARAMETER_COUNT)
    real(dp) :: vjp_gradient(MLP_ADAGRAD_HYPERPARAMETER_COUNT)
    real(dp) :: value, value_plus, value_minus, tangent, h
    real(dp) :: hvp_product(MLP_ADAGRAD_HYPERPARAMETER_COUNT)
    real(dp) :: gradient_plus(MLP_ADAGRAD_HYPERPARAMETER_COUNT)
    real(dp) :: gradient_minus(MLP_ADAGRAD_HYPERPARAMETER_COUNT)
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
    options%epsilon = 0.03_dp
    options%lower_log_learning_rate = -4.0_dp
    options%upper_log_learning_rate = 0.0_dp
    options%lower_log_l2 = -5.0_dp
    options%upper_log_l2 = 0.0_dp
    options%lower_log_epsilon = -5.0_dp
    options%upper_log_epsilon = 0.0_dp
    call objective%initialize(model, train_x, train_target, validation_x, &
        validation_target, options, status)
    call check(status_ok(status), "Adagrad hypergradient initialization", failures)
    call check(objective%parameter_count() == MLP_ADAGRAD_HYPERPARAMETER_COUNT, &
        "Adagrad packed parameter count", failures)
    metadata = objective%metadata()
    call check(metadata%log_learning_rate_index == MLP_ADAGRAD_LOG_LEARNING_RATE .and. &
        metadata%log_l2_index == MLP_ADAGRAD_LOG_L2 .and. &
        metadata%log_epsilon_index == MLP_ADAGRAD_LOG_EPSILON .and. &
        metadata%inner_steps == 4, "Adagrad packed metadata", failures)

    parameters = objective%parameters()
    call objective%value_gradient(parameters, value, gradient, status)
    call check(status_ok(status), "Adagrad value/gradient", failures)
    h = 2.0e-6_dp
    do i = 1, MLP_ADAGRAD_HYPERPARAMETER_COUNT
        parameters(i) = parameters(i) + h
        call objective%value_gradient(parameters, value_plus, vjp_gradient, status)
        parameters(i) = parameters(i) - 2.0_dp*h
        call objective%value_gradient(parameters, value_minus, vjp_gradient, status)
        parameters(i) = parameters(i) + h
        call check(abs(gradient(i) - (value_plus-value_minus)/(2.0_dp*h)) < 2.0e-6_dp, &
            "Adagrad hypergradient central difference", failures)
    end do

    direction = [0.31_dp, -0.27_dp, 0.19_dp]
    call objective%jvp(parameters, direction, value, tangent, status)
    call check(status_ok(status), "Adagrad forward JVP", failures)
    call objective%value_gradient(parameters+h*direction, value_plus, vjp_gradient, status)
    call objective%value_gradient(parameters-h*direction, value_minus, vjp_gradient, status)
    call check(abs(tangent-(value_plus-value_minus)/(2.0_dp*h)) < 3.0e-6_dp, &
        "Adagrad forward JVP central difference", failures)
    call objective%vjp(parameters, 1.7_dp, vjp_gradient, status)
    call objective%value_gradient(parameters, value, gradient, status)
    call check(maxval(abs(vjp_gradient-1.7_dp*gradient)) < 2.0e-12_dp, &
        "Adagrad scalar VJP adjoint", failures)

    ! The affine branch has a parameter-independent MSE Hessian.  Its outer
    ! Adagrad Hessian-vector product is therefore analytic; compare it with
    ! an independent central difference of the tested gradient oracle.
    call affine_model%initialize([1, 1], status, hidden_activation=MLP_LINEAR, &
        output_activation=MLP_LINEAR, initialization_seed=23)
    call affine_model%set_parameters([0.15_dp, -0.1_dp], status)
    options%steps = 4
    call objective%initialize(affine_model, train_x, train_target, validation_x, &
        validation_target, options, status)
    call check(status_ok(status), "affine Adagrad HVP initialization", failures)
    parameters = objective%parameters()
    direction = [0.23_dp, -0.17_dp, 0.11_dp]
    h = 2.0e-6_dp
    call objective%hvp(parameters, direction, hvp_product, status)
    call check(status_ok(status), "affine Adagrad HVP product", failures)
    call objective%value_gradient(parameters+h*direction, value_plus, gradient_plus, status)
    call objective%value_gradient(parameters-h*direction, value_minus, gradient_minus, status)
    call check(status_ok(status) .and. maxval(abs(hvp_product - &
        (gradient_plus-gradient_minus)/(2.0_dp*h))) < 2.0e-5_dp, &
        "affine Adagrad HVP central-difference oracle", failures)

    options%max_iterations = 80
    options%gradient_tolerance = 1.0e-5_dp
    call mlp_optimize_adagrad_hyperparameters(model, train_x, train_target, &
        validation_x, validation_target, options, result, status)
    call check(status_ok(status), "Adagrad FortOpt L-BFGS-B solve", failures)
    call check(result%converged .and. result%learning_rate > 0.0_dp .and. &
        result%l2 > 0.0_dp .and. result%epsilon > 0.0_dp, &
        "Adagrad FortOpt result", failures)

    bad_options = options
    bad_options%optimizer = MLP_OPTIMIZER_ADAM
    call objective%initialize(model, train_x, train_target, validation_x, &
        validation_target, bad_options, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "non-Adagrad hypergradient refusal", failures)
    bad_options = options
    bad_options%device_kind = FORTML_DEVICE_CUDA
    call objective%initialize(model, train_x, train_target, validation_x, &
        validation_target, bad_options, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "CUDA Adagrad hypergradient refusal", failures)

    call model%initialize([1, 2, 1], status, initialization_seed=23)
    call objective%initialize(model, train_x, train_target, validation_x, &
        validation_target, options, status)
    call objective%hvp(objective%parameters(), direction, hvp_product, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "nonlinear Adagrad HVP typed refusal", failures)
    bad_options = options
    bad_options%learning_rate = -1.0_dp
    call objective%initialize(model, train_x, train_target, validation_x, &
        validation_target, bad_options, status)
    call check(status%code == FORTNUM_DOMAIN_ERROR, &
        "non-positive Adagrad learning-rate validation", failures)

    if (failures > 0) error stop 1
    write (*, '(a)') "PASS MLP Adagrad hypergradient independent behavioral oracles"

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

end program test_mlp_adagrad_hypergradient
