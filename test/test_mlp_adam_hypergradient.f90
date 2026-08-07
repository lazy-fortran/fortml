program test_mlp_adam_hypergradient
    !! Independent central-difference and scalar-adjoint checks for coupled-L2
    !! Adam trajectory hypergradients and the FortOpt adapter.
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: FORTML_DEVICE_CUDA
    use fortml_mlp, only: mlp_t
    use fortml_mlp_adam_hypergradient, only: &
        mlp_adam_hypergradient_objective_t, mlp_adam_hypergradient_options_t, &
        mlp_adam_hypergradient_result_t, mlp_adam_hypergradient_metadata_t, &
        mlp_optimize_adam_hyperparameters, MLP_ADAM_HYPERPARAMETER_COUNT, &
        MLP_ADAM_LOG_LEARNING_RATE, MLP_ADAM_LOG_L2, MLP_ADAM_LOGIT_BETA1, &
        MLP_ADAM_LOGIT_BETA2
    implicit none

    type(mlp_t), target :: model
    type(mlp_adam_hypergradient_objective_t) :: objective
    type(mlp_adam_hypergradient_options_t) :: options, bad_options
    type(mlp_adam_hypergradient_result_t) :: result
    type(mlp_adam_hypergradient_metadata_t) :: metadata
    type(fortnum_status_t) :: status
    real(dp) :: train_x(5, 1), train_target(5, 1)
    real(dp) :: validation_x(3, 1), validation_target(3, 1)
    real(dp) :: parameters(MLP_ADAM_HYPERPARAMETER_COUNT)
    real(dp) :: direction(MLP_ADAM_HYPERPARAMETER_COUNT)
    real(dp) :: gradient(MLP_ADAM_HYPERPARAMETER_COUNT)
    real(dp) :: vjp_gradient(MLP_ADAM_HYPERPARAMETER_COUNT)
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
    options%beta1 = 0.82_dp
    options%beta2 = 0.91_dp
    options%epsilon = 0.03_dp
    options%lower_log_learning_rate = -4.0_dp
    options%upper_log_learning_rate = 0.0_dp
    options%lower_log_l2 = -5.0_dp
    options%upper_log_l2 = 0.0_dp
    options%lower_logit_beta1 = -4.0_dp
    options%upper_logit_beta1 = 4.0_dp
    options%lower_logit_beta2 = -4.0_dp
    options%upper_logit_beta2 = 4.0_dp
    call objective%initialize(model, train_x, train_target, validation_x, &
        validation_target, options, status)
    call check(status_ok(status), "Adam hypergradient initialization", failures)
    call check(objective%parameter_count() == MLP_ADAM_HYPERPARAMETER_COUNT, &
        "Adam packed parameter count", failures)
    metadata = objective%metadata()
    call check(metadata%log_learning_rate_index == MLP_ADAM_LOG_LEARNING_RATE .and. &
        metadata%log_l2_index == MLP_ADAM_LOG_L2 .and. &
        metadata%logit_beta1_index == MLP_ADAM_LOGIT_BETA1 .and. &
        metadata%logit_beta2_index == MLP_ADAM_LOGIT_BETA2 .and. &
        metadata%inner_steps == 4, "Adam packed metadata", failures)

    parameters = objective%parameters()
    call objective%value_gradient(parameters, value, gradient, status)
    call check(status_ok(status), "Adam value/gradient", failures)
    h = 2.0e-6_dp
    do i = 1, MLP_ADAM_HYPERPARAMETER_COUNT
        parameters(i) = parameters(i) + h
        call objective%value_gradient(parameters, value_plus, vjp_gradient, status)
        parameters(i) = parameters(i) - 2.0_dp*h
        call objective%value_gradient(parameters, value_minus, vjp_gradient, status)
        parameters(i) = parameters(i) + h
        call check(abs(gradient(i) - (value_plus-value_minus)/(2.0_dp*h)) < 3.0e-6_dp, &
            "Adam hypergradient central difference", failures)
    end do

    direction = [0.31_dp, -0.27_dp, 0.13_dp, -0.22_dp]
    call objective%jvp(parameters, direction, value, tangent, status)
    call objective%value_gradient(parameters+h*direction, value_plus, vjp_gradient, status)
    call objective%value_gradient(parameters-h*direction, value_minus, vjp_gradient, status)
    call check(status_ok(status), "Adam forward JVP", failures)
    call check(abs(tangent-(value_plus-value_minus)/(2.0_dp*h)) < 5.0e-6_dp, &
        "Adam forward JVP central difference", failures)
    call objective%vjp(parameters, 1.7_dp, vjp_gradient, status)
    call objective%value_gradient(parameters, value, gradient, status)
    call check(maxval(abs(vjp_gradient-1.7_dp*gradient)) < 2.0e-12_dp, &
        "Adam scalar VJP adjoint", failures)

    options%max_iterations = 80
    options%gradient_tolerance = 1.0e-5_dp
    call mlp_optimize_adam_hyperparameters(model, train_x, train_target, &
        validation_x, validation_target, options, result, status)
    call check(status_ok(status), "Adam FortOpt L-BFGS-B solve", failures)
    call check(result%converged .and. result%learning_rate > 0.0_dp .and. &
        result%l2 > 0.0_dp .and. result%beta1 > 0.0_dp .and. result%beta1 < 1.0_dp .and. &
        result%beta2 > 0.0_dp .and. result%beta2 < 1.0_dp, &
        "Adam FortOpt result", failures)

    bad_options = options
    bad_options%device_kind = FORTML_DEVICE_CUDA
    call objective%initialize(model, train_x, train_target, validation_x, &
        validation_target, bad_options, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "CUDA Adam hypergradient refusal", failures)

    if (failures > 0) error stop 1
    write (*, '(a)') "PASS MLP coupled-L2 Adam hypergradient independent behavioral oracles"

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

end program test_mlp_adam_hypergradient
