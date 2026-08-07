program test_mlp_hypergradient
    !! Independent finite-difference and adjoint checks for the fixed-trajectory
    !! MLP hyperparameter products.
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    use fortopt_objective, only: objective_t
    use fortml_device, only: FORTML_DEVICE_CUDA
    use fortml_mlp, only: mlp_t
    use fortml_mlp_training, only: MLP_OPTIMIZER_ADAM
    use fortml_mlp_hypergradient, only: &
        mlp_hypergradient_objective_t, mlp_hypergradient_options_t, &
        mlp_hypergradient_result_t, mlp_hyperparameter_metadata_t, &
        mlp_optimize_hyperparameters, MLP_HYPERPARAMETER_COUNT, &
        MLP_LOG_LEARNING_RATE, MLP_LOG_L2
    implicit none

    type(mlp_t), target :: model
    type(mlp_hypergradient_objective_t) :: objective
    type(mlp_hypergradient_options_t) :: options, bad_options
    type(mlp_hypergradient_result_t) :: result
    type(mlp_hyperparameter_metadata_t) :: metadata
    type(objective_t) :: fortopt_objective
    type(fortnum_status_t) :: status
    real(dp) :: train_x(5, 1), train_target(5, 1)
    real(dp) :: validation_x(3, 1), validation_target(3, 1)
    real(dp) :: parameters(2), direction(2), gradient(2), vjp_gradient(2)
    real(dp) :: value, value_plus, value_minus, tangent, h, output_bar
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
    options%lower_log_learning_rate = -4.0_dp
    options%upper_log_learning_rate = 0.0_dp
    options%lower_log_l2 = -5.0_dp
    options%upper_log_l2 = 0.0_dp
    call objective%initialize(model, train_x, train_target, validation_x, &
        validation_target, options, status)
    call check(status_ok(status), "hypergradient initialization", failures)
    call check(objective%is_initialized(), "initialized predicate", failures)
    call check(objective%parameter_count() == MLP_HYPERPARAMETER_COUNT, &
        "packed parameter count", failures)
    metadata = objective%metadata()
    call check(metadata%log_learning_rate_index == MLP_LOG_LEARNING_RATE .and. &
        metadata%log_l2_index == MLP_LOG_L2 .and. metadata%inner_steps == 4, &
        "packed metadata", failures)

    parameters = objective%parameters()
    call objective%value_gradient(parameters, value, gradient, status)
    call check(status_ok(status), "reverse value/gradient", failures)
    h = 2.0e-6_dp
    do i = 1, MLP_HYPERPARAMETER_COUNT
        parameters(i) = parameters(i) + h
        call objective%value_gradient(parameters, value_plus, vjp_gradient, status)
        call check(status_ok(status), "plus finite-difference evaluation", failures)
        parameters(i) = parameters(i) - 2.0_dp*h
        call objective%value_gradient(parameters, value_minus, vjp_gradient, status)
        call check(status_ok(status), "minus finite-difference evaluation", failures)
        parameters(i) = parameters(i) + h
        call check(abs(gradient(i) - (value_plus - value_minus)/(2.0_dp*h)) &
            < 2.0e-7_dp, "reverse hypergradient central difference", failures)
    end do

    direction = [0.31_dp, -0.27_dp]
    call objective%jvp(parameters, direction, value, tangent, status)
    call check(status_ok(status), "forward JVP", failures)
    call objective%value_gradient(parameters + h*direction, value_plus, &
        vjp_gradient, status)
    call check(status_ok(status), "JVP plus evaluation", failures)
    call objective%value_gradient(parameters - h*direction, value_minus, &
        vjp_gradient, status)
    call check(status_ok(status), "JVP minus evaluation", failures)
    call check(abs(tangent - (value_plus - value_minus)/(2.0_dp*h)) < 3.0e-7_dp, &
        "forward JVP central difference", failures)

    output_bar = 1.7_dp
    call objective%vjp(parameters, output_bar, vjp_gradient, status)
    call check(status_ok(status), "reverse VJP", failures)
    call objective%value_gradient(parameters, value, gradient, status)
    call check(maxval(abs(vjp_gradient - output_bar*gradient)) < 2.0e-12_dp, &
        "reverse VJP scalar adjoint", failures)

    call objective%fortopt(fortopt_objective, status)
    call check(status_ok(status), "FortOpt context adapter", failures)
    call fortopt_objective%value_gradient(parameters, value, vjp_gradient, status)
    call check(status_ok(status), "FortOpt value/gradient callback", failures)
    call check(abs(value - objective_value(objective, parameters, status)) < &
        2.0e-13_dp, "FortOpt callback value", failures)

    options%max_iterations = 100
    options%gradient_tolerance = 1.0e-5_dp
    call mlp_optimize_hyperparameters(model, train_x, train_target, &
        validation_x, validation_target, options, result, status)
    call check(status_ok(status), "FortOpt L-BFGS-B hyperparameter solve", failures)
    call check(result%converged .and. result%objective < huge(1.0_dp) .and. &
        result%learning_rate > 0.0_dp .and. result%l2 > 0.0_dp, &
        "FortOpt hyperparameter result", failures)

    bad_options = options
    bad_options%optimizer = MLP_OPTIMIZER_ADAM
    call objective%initialize(model, train_x, train_target, validation_x, &
        validation_target, bad_options, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "Adam hypergradient refusal", failures)
    bad_options = options
    bad_options%momentum = 0.3_dp
    call objective%initialize(model, train_x, train_target, validation_x, &
        validation_target, bad_options, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "momentum hypergradient refusal", failures)
    bad_options = options
    bad_options%device_kind = FORTML_DEVICE_CUDA
    call objective%initialize(model, train_x, train_target, validation_x, &
        validation_target, bad_options, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "CUDA hypergradient refusal", failures)

    if (failures > 0) error stop 1
    write (*, '(a)') "PASS MLP hypergradient independent behavioral oracles"

contains

    real(dp) function objective_value(adapter, parameters, status) result(value)
        type(mlp_hypergradient_objective_t), intent(inout) :: adapter
        real(dp), intent(in) :: parameters(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: gradient(MLP_HYPERPARAMETER_COUNT)

        call adapter%value_gradient(parameters, value, gradient, status)
    end function objective_value

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: description
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (*, '(a)') "FAIL "//trim(description)
        end if
    end subroutine check

end program test_mlp_hypergradient
