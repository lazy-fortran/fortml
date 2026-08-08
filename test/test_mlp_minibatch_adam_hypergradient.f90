program test_mlp_minibatch_adam_hypergradient
    !! Independent finite-difference and adjoint checks for the deterministic
    !! mini-batch Adam trajectory hypergradient and FortOpt boundary.
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: FORTML_DEVICE_CUDA
    use fortml_mlp, only: mlp_t
    use fortml_mlp_minibatch_adam_hypergradient, only: &
        mlp_minibatch_adam_hypergradient_objective_t, &
        mlp_minibatch_adam_hypergradient_options_t, &
        mlp_minibatch_adam_hypergradient_result_t, &
        mlp_minibatch_adam_hypergradient_metadata_t, &
        mlp_optimize_minibatch_adam_hyperparameters, &
        MLP_MINIBATCH_ADAM_HYPERPARAMETER_COUNT, &
        MLP_MINIBATCH_ADAM_LOG_LEARNING_RATE, MLP_MINIBATCH_ADAM_LOG_L2
    implicit none

    type(mlp_t), target :: model
    type(mlp_t), target :: nonlinear_model
    type(mlp_minibatch_adam_hypergradient_objective_t) :: objective
    type(mlp_minibatch_adam_hypergradient_objective_t) :: nonlinear_objective
    type(mlp_minibatch_adam_hypergradient_options_t) :: options, bad_options
    type(mlp_minibatch_adam_hypergradient_result_t) :: result
    type(mlp_minibatch_adam_hypergradient_metadata_t) :: metadata
    type(fortnum_status_t) :: status
    real(dp) :: train_x(7, 1), train_target(7, 1)
    real(dp) :: validation_x(3, 1), validation_target(3, 1)
    real(dp) :: parameters(MLP_MINIBATCH_ADAM_HYPERPARAMETER_COUNT)
    real(dp) :: direction(MLP_MINIBATCH_ADAM_HYPERPARAMETER_COUNT)
    real(dp) :: gradient(MLP_MINIBATCH_ADAM_HYPERPARAMETER_COUNT)
    real(dp) :: vjp_gradient(MLP_MINIBATCH_ADAM_HYPERPARAMETER_COUNT)
    real(dp) :: value, value_plus, value_minus, tangent, h
    integer :: i, failures

    failures = 0
    train_x(:, 1) = [-2.2_dp, -1.1_dp, -0.35_dp, 0.2_dp, 0.9_dp, 1.6_dp, 2.3_dp]
    train_target(:, 1) = 0.65_dp*train_x(:, 1) - 0.11_dp
    validation_x(:, 1) = [-1.8_dp, 0.35_dp, 1.9_dp]
    validation_target(:, 1) = 0.65_dp*validation_x(:, 1) - 0.11_dp

    call model%initialize([1, 1], status, initialization_seed=29)
    call model%set_parameters([0.21_dp, -0.06_dp], status)
    options%epochs = 3
    options%batch_size = 3
    options%shuffle = .true.
    options%shuffle_seed = 43
    options%learning_rate = 0.08_dp
    options%l2 = 0.035_dp
    options%beta1 = 0.84_dp
    options%beta2 = 0.93_dp
    options%epsilon = 0.025_dp
    options%lower_log_learning_rate = -4.0_dp
    options%upper_log_learning_rate = 0.0_dp
    options%lower_log_l2 = -5.0_dp
    options%upper_log_l2 = 0.0_dp
    call objective%initialize(model, train_x, train_target, validation_x, &
        validation_target, options, status)
    call check(status_ok(status), "mini-batch Adam objective initialization", failures)
    call check(objective%is_initialized(), "initialized predicate", failures)
    metadata = objective%metadata()
    call check(metadata%epochs == 3 .and. metadata%batch_size == 3 .and. &
        metadata%steps == 9 .and. metadata%shuffle, "batch metadata", failures)

    parameters = objective%parameters()
    call objective%value_gradient(parameters, value, gradient, status)
    call check(status_ok(status), "value/gradient", failures)
    h = 2.0e-6_dp
    do i = 1, MLP_MINIBATCH_ADAM_HYPERPARAMETER_COUNT
        parameters(i) = parameters(i) + h
        call objective%value_gradient(parameters, value_plus, vjp_gradient, status)
        call check(status_ok(status), "plus finite-difference evaluation", failures)
        parameters(i) = parameters(i) - 2.0_dp*h
        call objective%value_gradient(parameters, value_minus, vjp_gradient, status)
        call check(status_ok(status), "minus finite-difference evaluation", failures)
        parameters(i) = parameters(i) + h
        call check(abs(gradient(i) - (value_plus-value_minus)/(2.0_dp*h)) < &
            8.0e-6_dp, "mini-batch Adam hypergradient central difference", failures)
    end do

    direction = [0.27_dp, -0.19_dp]
    call objective%jvp(parameters, direction, value, tangent, status)
    call check(status_ok(status), "forward mini-batch Adam JVP", failures)
    call objective%value_gradient(parameters + h*direction, value_plus, &
        vjp_gradient, status)
    call objective%value_gradient(parameters - h*direction, value_minus, &
        vjp_gradient, status)
    call check(abs(tangent - (value_plus-value_minus)/(2.0_dp*h)) < 1.0e-5_dp, &
        "mini-batch Adam JVP central difference", failures)

    call objective%vjp(parameters, 1.7_dp, vjp_gradient, status)
    call check(status_ok(status), "reverse mini-batch Adam VJP", failures)
    call objective%value_gradient(parameters, value, gradient, status)
    call check(maxval(abs(vjp_gradient - 1.7_dp*gradient)) < 2.0e-12_dp, &
        "mini-batch Adam scalar VJP adjoint", failures)

    ! Repeat the product checks on a nonlinear two-hidden-unit network.  The
    ! linear fixture above catches batch/moment conventions; this fixture keeps
    ! the behavioral oracle independent of a linear closed form.
    call nonlinear_model%initialize([1, 2, 1], status, initialization_seed=47)
    call nonlinear_model%set_parameters([0.14_dp, -0.09_dp, 0.07_dp, 0.12_dp, &
        -0.05_dp, 0.11_dp], status)
    bad_options = options
    bad_options%epochs = 2
    bad_options%batch_size = 4
    bad_options%shuffle = .false.
    bad_options%learning_rate = 0.035_dp
    bad_options%l2 = 0.018_dp
    bad_options%beta1 = 0.79_dp
    bad_options%beta2 = 0.9_dp
    bad_options%epsilon = 0.017_dp
    call nonlinear_objective%initialize(nonlinear_model, train_x, train_target, &
        validation_x, validation_target, bad_options, status)
    call check(status_ok(status), "nonlinear mini-batch Adam initialization", failures)
    parameters = nonlinear_objective%parameters()
    call nonlinear_objective%value_gradient(parameters, value, gradient, status)
    call check(status_ok(status), "nonlinear mini-batch Adam products", failures)
    do i = 1, MLP_MINIBATCH_ADAM_HYPERPARAMETER_COUNT
        parameters(i) = parameters(i) + h
        call nonlinear_objective%value_gradient(parameters, value_plus, vjp_gradient, status)
        parameters(i) = parameters(i) - 2.0_dp*h
        call nonlinear_objective%value_gradient(parameters, value_minus, vjp_gradient, status)
        parameters(i) = parameters(i) + h
        call check(abs(gradient(i) - (value_plus-value_minus)/(2.0_dp*h)) < &
            2.0e-5_dp, "nonlinear mini-batch Adam central difference", failures)
    end do

    call objective%hvp(parameters, direction, gradient, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "outer mini-batch Adam HVP typed refusal", failures)

    options%max_iterations = 120
    options%gradient_tolerance = 1.0e-3_dp
    call mlp_optimize_minibatch_adam_hyperparameters(model, train_x, train_target, &
        validation_x, validation_target, options, result, status)
    call check(status_ok(status), "FortOpt mini-batch Adam hyperparameter solve", failures)
    call check(result%converged .and. result%learning_rate > 0.0_dp .and. &
        result%l2 > 0.0_dp, "FortOpt mini-batch Adam result", failures)

    bad_options = options
    bad_options%device_kind = FORTML_DEVICE_CUDA
    call objective%initialize(model, train_x, train_target, validation_x, &
        validation_target, bad_options, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "CUDA mini-batch Adam hypergradient refusal", failures)

    if (failures > 0) error stop 1
    write (*, '(a)') "PASS MLP mini-batch Adam hypergradient independent behavioral oracles"

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

end program test_mlp_minibatch_adam_hypergradient
