program test_mlp_minibatch_hypergradient
    !! Independent finite-difference and adjoint checks for the deterministic
    !! mini-batch trajectory hypergradient and its FortOpt boundary.
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    use fortopt_objective, only: objective_t
    use fortml_device, only: FORTML_DEVICE_CUDA
    use fortml_mlp, only: mlp_t
    use fortml_mlp_minibatch_hypergradient, only: &
        mlp_minibatch_hypergradient_objective_t, &
        mlp_minibatch_hypergradient_options_t, &
        mlp_minibatch_hypergradient_result_t, &
        mlp_minibatch_hypergradient_metadata_t, &
        mlp_optimize_minibatch_hyperparameters, &
        MLP_MINIBATCH_HYPERPARAMETER_COUNT, &
        MLP_MINIBATCH_LOG_LEARNING_RATE, MLP_MINIBATCH_LOG_L2
    implicit none

    type(mlp_t), target :: model
    type(mlp_minibatch_hypergradient_objective_t) :: objective
    type(mlp_minibatch_hypergradient_options_t) :: options, bad_options
    type(mlp_minibatch_hypergradient_result_t) :: result
    type(mlp_minibatch_hypergradient_metadata_t) :: metadata
    type(objective_t) :: fortopt_objective
    type(fortnum_status_t) :: status
    real(dp) :: train_x(6, 1), train_target(6, 1)
    real(dp) :: validation_x(3, 1), validation_target(3, 1)
    real(dp) :: parameters(MLP_MINIBATCH_HYPERPARAMETER_COUNT)
    real(dp) :: direction(MLP_MINIBATCH_HYPERPARAMETER_COUNT)
    real(dp) :: gradient(MLP_MINIBATCH_HYPERPARAMETER_COUNT)
    real(dp) :: vjp_gradient(MLP_MINIBATCH_HYPERPARAMETER_COUNT)
    real(dp) :: product(MLP_MINIBATCH_HYPERPARAMETER_COUNT)
    real(dp) :: value, value_plus, value_minus, tangent, h
    integer :: i, failures

    failures = 0
    train_x(:, 1) = [-2.0_dp, -1.0_dp, -0.3_dp, 0.4_dp, 1.2_dp, 2.1_dp]
    train_target(:, 1) = 0.8_dp*train_x(:, 1) - 0.15_dp
    validation_x(:, 1) = [-1.7_dp, 0.25_dp, 1.8_dp]
    validation_target(:, 1) = 0.8_dp*validation_x(:, 1) - 0.15_dp

    call model%initialize([1, 1], status, initialization_seed=13)
    call model%set_parameters([0.17_dp, -0.08_dp], status)
    options%epochs = 3
    options%batch_size = 2
    options%shuffle = .true.
    options%shuffle_seed = 31
    options%learning_rate = 0.08_dp
    options%l2 = 0.04_dp
    options%lower_log_learning_rate = -4.0_dp
    options%upper_log_learning_rate = 0.0_dp
    options%lower_log_l2 = -5.0_dp
    options%upper_log_l2 = 0.0_dp
    call objective%initialize(model, train_x, train_target, validation_x, &
        validation_target, options, status)
    call check(status_ok(status), "mini-batch objective initialization", failures)
    call check(objective%is_initialized(), "initialized predicate", failures)
    metadata = objective%metadata()
    call check(metadata%epochs == 3 .and. metadata%batch_size == 2 .and. &
        metadata%steps == 9 .and. metadata%shuffle, "batch metadata", failures)

    parameters = objective%parameters()
    call objective%value_gradient(parameters, value, gradient, status)
    call check(status_ok(status), "reverse value/gradient", failures)
    h = 2.0e-6_dp
    do i = 1, MLP_MINIBATCH_HYPERPARAMETER_COUNT
        parameters(i) = parameters(i) + h
        call objective%value_gradient(parameters, value_plus, vjp_gradient, status)
        call check(status_ok(status), "plus finite-difference evaluation", failures)
        parameters(i) = parameters(i) - 2.0_dp*h
        call objective%value_gradient(parameters, value_minus, vjp_gradient, status)
        call check(status_ok(status), "minus finite-difference evaluation", failures)
        parameters(i) = parameters(i) + h
        call check(abs(gradient(i) - (value_plus - value_minus)/(2.0_dp*h)) < &
            2.0e-6_dp, "mini-batch hypergradient central difference", failures)
    end do

    direction = [0.29_dp, -0.23_dp]
    call objective%jvp(parameters, direction, value, tangent, status)
    call check(status_ok(status), "forward mini-batch JVP", failures)
    call objective%value_gradient(parameters + h*direction, value_plus, &
        vjp_gradient, status)
    call objective%value_gradient(parameters - h*direction, value_minus, &
        vjp_gradient, status)
    call check(abs(tangent - (value_plus-value_minus)/(2.0_dp*h)) < 3.0e-6_dp, &
        "mini-batch JVP central difference", failures)

    call objective%vjp(parameters, 1.7_dp, vjp_gradient, status)
    call check(status_ok(status), "reverse mini-batch VJP", failures)
    call objective%value_gradient(parameters, value, gradient, status)
    call check(maxval(abs(vjp_gradient - 1.7_dp*gradient)) < 2.0e-12_dp, &
        "mini-batch scalar VJP adjoint", failures)
    call objective%hvp(parameters, direction, product, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "outer mini-batch HVP typed refusal", failures)

    call objective%fortopt(fortopt_objective, status)
    call check(status_ok(status), "FortOpt mini-batch context adapter", failures)
    call fortopt_objective%value_gradient(parameters, value, vjp_gradient, status)
    call check(status_ok(status), "FortOpt mini-batch callback", failures)

    bad_options = options
    bad_options%device_kind = FORTML_DEVICE_CUDA
    call objective%initialize(model, train_x, train_target, validation_x, &
        validation_target, bad_options, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "CUDA mini-batch hypergradient refusal", failures)

    options%max_iterations = 500
    options%gradient_tolerance = 1.0e-3_dp
    call mlp_optimize_minibatch_hyperparameters(model, train_x, train_target, &
        validation_x, validation_target, options, result, status)
    call check(status_ok(status), "FortOpt mini-batch hyperparameter solve", failures)
    call check(result%converged .and. result%learning_rate > 0.0_dp .and. &
        result%l2 > 0.0_dp, "FortOpt mini-batch result", failures)

    if (failures > 0) error stop 1
    write (*, '(a)') "PASS MLP mini-batch hypergradient independent behavioral oracles"

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

end program test_mlp_minibatch_hypergradient
