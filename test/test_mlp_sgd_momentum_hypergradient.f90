program test_mlp_sgd_momentum_hypergradient
    !! Independent finite-difference and scalar-adjoint checks for momentum SGD.
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: FORTML_DEVICE_CUDA
    use fortml_mlp, only: mlp_t
    use fortml_mlp_training, only: MLP_OPTIMIZER_ADAM
    use fortml_mlp_sgd_momentum_hypergradient, only: &
        mlp_sgd_momentum_hypergradient_objective_t, &
        mlp_sgd_momentum_hypergradient_options_t, &
        mlp_sgd_momentum_hypergradient_result_t, &
        mlp_sgd_momentum_hypergradient_metadata_t, &
        mlp_optimize_sgd_momentum_hyperparameters, &
        MLP_SGD_MOMENTUM_HYPERPARAMETER_COUNT, MLP_SGD_LOG_LEARNING_RATE, &
        MLP_SGD_LOG_L2, MLP_SGD_MOMENTUM
    implicit none

    type(mlp_t), target :: model, nesterov_model
    type(mlp_sgd_momentum_hypergradient_objective_t) :: objective
    type(mlp_sgd_momentum_hypergradient_options_t) :: options, bad_options
    type(mlp_sgd_momentum_hypergradient_result_t) :: result
    type(mlp_sgd_momentum_hypergradient_metadata_t) :: metadata
    type(fortnum_status_t) :: status
    real(dp) :: train_x(5, 1), train_target(5, 1)
    real(dp) :: validation_x(3, 1), validation_target(3, 1)
    real(dp) :: parameters(MLP_SGD_MOMENTUM_HYPERPARAMETER_COUNT)
    real(dp) :: direction(MLP_SGD_MOMENTUM_HYPERPARAMETER_COUNT)
    real(dp) :: gradient(MLP_SGD_MOMENTUM_HYPERPARAMETER_COUNT)
    real(dp) :: vjp_gradient(MLP_SGD_MOMENTUM_HYPERPARAMETER_COUNT)
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
    options%momentum = 0.31_dp
    options%lower_log_learning_rate = -4.0_dp
    options%upper_log_learning_rate = 0.0_dp
    options%lower_log_l2 = -5.0_dp
    options%upper_log_l2 = 0.0_dp
    options%lower_momentum = 0.05_dp
    options%upper_momentum = 0.8_dp
    call objective%initialize(model, train_x, train_target, validation_x, &
        validation_target, options, status)
    call check(status_ok(status), "momentum hypergradient initialization", failures)
    call check(objective%parameter_count() == MLP_SGD_MOMENTUM_HYPERPARAMETER_COUNT, &
        "momentum packed parameter count", failures)
    metadata = objective%metadata()
    call check(metadata%log_learning_rate_index == MLP_SGD_LOG_LEARNING_RATE .and. &
        metadata%log_l2_index == MLP_SGD_LOG_L2 .and. &
        metadata%momentum_index == MLP_SGD_MOMENTUM .and. metadata%inner_steps == 4 .and. &
        .not. metadata%nesterov, "momentum packed metadata", failures)
    parameters = objective%parameters()
    call objective%value_gradient(parameters, value, gradient, status)
    call check(status_ok(status), "momentum value/gradient", failures)
    h = 2.0e-6_dp
    do i = 1, MLP_SGD_MOMENTUM_HYPERPARAMETER_COUNT
        parameters(i) = parameters(i) + h
        call objective%value_gradient(parameters, value_plus, vjp_gradient, status)
        call check(status_ok(status), "momentum plus finite-difference evaluation", failures)
        parameters(i) = parameters(i) - 2.0_dp*h
        call objective%value_gradient(parameters, value_minus, vjp_gradient, status)
        call check(status_ok(status), "momentum minus finite-difference evaluation", failures)
        parameters(i) = parameters(i) + h
        call check(abs(gradient(i) - (value_plus-value_minus)/(2.0_dp*h)) < 2.0e-6_dp, &
            "momentum hypergradient central difference", failures)
    end do
    direction = [0.31_dp, -0.27_dp, 0.17_dp]
    call objective%jvp(parameters, direction, value, tangent, status)
    call check(status_ok(status), "momentum forward JVP", failures)
    call objective%value_gradient(parameters+h*direction, value_plus, vjp_gradient, status)
    call objective%value_gradient(parameters-h*direction, value_minus, vjp_gradient, status)
    call check(abs(tangent-(value_plus-value_minus)/(2.0_dp*h)) < 3.0e-6_dp, &
        "momentum forward JVP central difference", failures)
    call objective%vjp(parameters, 1.7_dp, vjp_gradient, status)
    call objective%value_gradient(parameters, value, gradient, status)
    call check(maxval(abs(vjp_gradient-1.7_dp*gradient)) < 2.0e-12_dp, &
        "momentum scalar VJP adjoint", failures)

    options%max_iterations = 80
    options%gradient_tolerance = 1.0e-5_dp
    call mlp_optimize_sgd_momentum_hyperparameters(model, train_x, train_target, &
        validation_x, validation_target, options, result, status)
    call check(status_ok(status), "momentum FortOpt L-BFGS-B solve", failures)
    call check(result%converged .and. result%learning_rate > 0.0_dp .and. &
        result%l2 > 0.0_dp .and. result%momentum >= options%lower_momentum .and. &
        result%momentum <= options%upper_momentum, "momentum FortOpt result", failures)

    ! Nesterov uses a distinct look-ahead direction but the same products.
    call nesterov_model%initialize([1, 1], status, initialization_seed=23)
    call nesterov_model%set_parameters([0.15_dp, -0.1_dp], status)
    options%momentum = 0.37_dp
    options%nesterov = .true.
    options%lower_momentum = 0.05_dp
    call objective%initialize(nesterov_model, train_x, train_target, validation_x, &
        validation_target, options, status)
    call check(status_ok(status), "Nesterov initialization", failures)
    metadata = objective%metadata()
    call check(metadata%nesterov, "Nesterov metadata", failures)
    parameters = objective%parameters()
    call objective%value_gradient(parameters, value, gradient, status)
    call check(status_ok(status), "Nesterov value/gradient", failures)
    do i = 1, MLP_SGD_MOMENTUM_HYPERPARAMETER_COUNT
        parameters(i) = parameters(i) + h
        call objective%value_gradient(parameters, value_plus, vjp_gradient, status)
        parameters(i) = parameters(i) - 2.0_dp*h
        call objective%value_gradient(parameters, value_minus, vjp_gradient, status)
        parameters(i) = parameters(i) + h
        call check(abs(gradient(i) - (value_plus-value_minus)/(2.0_dp*h)) < 3.0e-6_dp, &
            "Nesterov hypergradient central difference", failures)
    end do

    bad_options = options
    bad_options%optimizer = MLP_OPTIMIZER_ADAM
    call objective%initialize(nesterov_model, train_x, train_target, validation_x, &
        validation_target, bad_options, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "non-SGD hypergradient refusal", failures)
    bad_options = options
    bad_options%device_kind = FORTML_DEVICE_CUDA
    call objective%initialize(nesterov_model, train_x, train_target, validation_x, &
        validation_target, bad_options, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "CUDA SGD hypergradient refusal", failures)

    if (failures > 0) error stop 1
    write (*, '(a)') "PASS MLP SGD momentum hypergradient independent behavioral oracles"

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

end program test_mlp_sgd_momentum_hypergradient
