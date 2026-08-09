program test_mlp_weighted_training
    !! Independent hand-oracle checks for weighted MSE training.
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_OK
    use fortml_mlp, only: mlp_t, MLP_LINEAR
    use fortml_mlp_training, only: mlp_training_options_t, mlp_training_state_t, &
        mlp_loss_value_gradient, mlp_train, MLP_OPTIMIZER_ADAM, MLP_OPTIMIZER_SGD, &
        MLP_OPTIMIZER_ADAMW, MLP_OPTIMIZER_ADAGRAD, MLP_OPTIMIZER_RMSPROP, &
        MLP_OPTIMIZER_ADAFACTOR, MLP_OPTIMIZER_AMSGRAD, MLP_OPTIMIZER_RADAM, &
        MLP_OPTIMIZER_LION
    use fortml_mlp_regressor, only: mlp_regressor_t, mlp_regressor_options_t, &
        mlp_regressor_state_t
    implicit none

    integer :: failures

    failures = 0
    call test_weighted_loss_oracle(failures)
    call test_weighted_sgd_recurrence(failures)
    call test_weighted_optimizer_dispatch(failures)
    call test_weighted_regressor(failures)
    if (failures > 0) error stop "weighted MLP training oracle failures"
    write (*, '(a)') "PASS weighted MLP training independent hand oracle"

contains

    subroutine test_weighted_loss_oracle(failures)
        integer, intent(inout) :: failures
        type(mlp_t) :: model
        type(fortnum_status_t) :: status
        real(dp) :: x(4, 1), target(4, 1), weight(4), theta(2), gradient(2)
        real(dp) :: value, l2_gradient, l2, prediction(4), residual(4)
        real(dp) :: mass, expected_value, expected_gradient(2)

        x(:, 1) = [-1.0_dp, -0.25_dp, 0.75_dp, 1.5_dp]
        target(:, 1) = [0.4_dp, -0.1_dp, 0.8_dp, 1.2_dp]
        weight = [1.0_dp, 0.0_dp, 2.0_dp, 0.5_dp]
        theta = [0.3_dp, -0.2_dp]
        l2 = 0.07_dp
        call model%initialize([1, 1], status, output_activation=MLP_LINEAR)
        call model%set_parameters(theta, status)
        call mlp_loss_value_gradient(model, x, target, l2, value, gradient, &
            l2_gradient, status, sample_weight=weight)
        prediction = x(:, 1)*theta(1) + theta(2)
        residual = prediction - target(:, 1)
        mass = sum(weight)
        expected_value = 0.5_dp*sum(weight*residual**2)/mass + &
            0.5_dp*l2*sum(theta**2)
        expected_gradient(1) = sum(weight*residual*x(:, 1))/mass + l2*theta(1)
        expected_gradient(2) = sum(weight*residual)/mass + l2*theta(2)
        call check(status_ok(status), "weighted loss status", failures)
        call check(abs(value - expected_value) < 1.0e-13_dp .and. &
            maxval(abs(gradient - expected_gradient)) < 1.0e-13_dp .and. &
            abs(l2_gradient - 0.5_dp*sum(theta**2)) < 1.0e-13_dp, &
            "weighted MSE+L2 hand oracle", failures)

        call model%set_parameters(theta, status)
        call mlp_train(model, x, target, status, sample_weight=[1.0_dp, -1.0_dp, &
            1.0_dp, 1.0_dp])
        call check(.not. status_ok(status) .and. &
            maxval(abs(model%parameters() - theta)) < 1.0e-14_dp, &
            "invalid weights are transactional", failures)
        call mlp_train(model, x, target, status, sample_weight=0.0_dp*weight)
        call check(.not. status_ok(status) .and. &
            maxval(abs(model%parameters() - theta)) < 1.0e-14_dp, &
            "zero-support weights are refused transactionally", failures)
    end subroutine test_weighted_loss_oracle

    subroutine test_weighted_sgd_recurrence(failures)
        integer, intent(inout) :: failures
        type(mlp_t) :: model
        type(mlp_training_options_t) :: options
        type(mlp_training_state_t) :: state
        type(fortnum_status_t) :: status
        real(dp) :: x(4, 1), target(4, 1), weight(4), theta(2), expected(2)
        real(dp) :: prediction(4), residual(4), gradient(2), mass, rate

        x(:, 1) = [-1.0_dp, -0.25_dp, 0.75_dp, 1.5_dp]
        target(:, 1) = [0.4_dp, -0.1_dp, 0.8_dp, 1.2_dp]
        weight = [1.0_dp, 0.0_dp, 2.0_dp, 0.5_dp]
        theta = [0.3_dp, -0.2_dp]
        call model%initialize([1, 1], status, output_activation=MLP_LINEAR)
        call model%set_parameters(theta, status)
        options%optimizer = MLP_OPTIMIZER_SGD
        options%max_epochs = 1
        options%batch_size = 2
        options%accumulation_steps = 2
        options%learning_rate = 0.025_dp
        options%tolerance = 0.0_dp
        options%restore_best = .false.
        call mlp_train(model, x, target, status, options, state, &
            sample_weight=weight)
        prediction = x(:, 1)*theta(1) + theta(2)
        residual = prediction - target(:, 1)
        mass = sum(weight)
        gradient(1) = sum(weight*residual*x(:, 1))/mass
        gradient(2) = sum(weight*residual)/mass
        rate = options%learning_rate
        expected = theta - rate*gradient
        call check(status_ok(status) .and. state%updates == 1, &
            "weighted accumulation status and update count", failures)
        call check(maxval(abs(model%parameters() - expected)) < 2.0e-13_dp, &
            "weighted minibatch accumulation hand recurrence", failures)
    end subroutine test_weighted_sgd_recurrence

    subroutine test_weighted_optimizer_dispatch(failures)
        integer, intent(inout) :: failures
        integer, parameter :: optimizers(9) = [MLP_OPTIMIZER_ADAM, MLP_OPTIMIZER_SGD, &
            MLP_OPTIMIZER_ADAMW, MLP_OPTIMIZER_ADAGRAD, MLP_OPTIMIZER_RMSPROP, &
            MLP_OPTIMIZER_ADAFACTOR, MLP_OPTIMIZER_AMSGRAD, MLP_OPTIMIZER_RADAM, &
            MLP_OPTIMIZER_LION]
        type(mlp_t) :: model
        type(mlp_training_options_t) :: options
        type(mlp_training_state_t) :: state
        type(fortnum_status_t) :: status
        real(dp) :: x(4, 1), target(4, 1), weight(4), theta(2)
        integer :: i

        x(:, 1) = [-1.0_dp, -0.25_dp, 0.75_dp, 1.5_dp]
        target(:, 1) = [0.4_dp, -0.1_dp, 0.8_dp, 1.2_dp]
        weight = [1.0_dp, 0.0_dp, 2.0_dp, 0.5_dp]
        theta = [0.3_dp, -0.2_dp]
        options%max_epochs = 1
        options%batch_size = 2
        options%accumulation_steps = 2
        options%learning_rate = 0.01_dp
        options%tolerance = 0.0_dp
        options%restore_best = .false.
        do i = 1, size(optimizers)
            call model%initialize([1, 1], status, output_activation=MLP_LINEAR)
            call model%set_parameters(theta, status)
            options%optimizer = optimizers(i)
            call mlp_train(model, x, target, status, options, state, &
                sample_weight=weight)
            call check(status_ok(status) .and. state%updates == 1 .and. &
                all(model%parameters() == model%parameters()), &
                "weighted current CPU optimizer dispatch", failures)
        end do
    end subroutine test_weighted_optimizer_dispatch

    subroutine test_weighted_regressor(failures)
        integer, intent(inout) :: failures
        type(mlp_regressor_t) :: model
        type(mlp_regressor_options_t) :: options
        type(mlp_regressor_state_t) :: state
        type(fortnum_status_t) :: status
        real(dp) :: x(4, 1), target(4, 1), weight(4), validation_weight(2)
        real(dp) :: validation_x(2, 1), validation_target(2, 1)

        x(:, 1) = [-1.0_dp, -0.25_dp, 0.75_dp, 1.5_dp]
        target(:, 1) = [0.4_dp, -0.1_dp, 0.8_dp, 1.2_dp]
        weight = [1.0_dp, 0.0_dp, 2.0_dp, 0.5_dp]
        validation_x(:, 1) = [-0.5_dp, 1.0_dp]
        validation_target(:, 1) = [0.0_dp, 0.9_dp]
        validation_weight = [0.25_dp, 1.5_dp]
        allocate(options%layer_sizes(2))
        options%layer_sizes = [1, 1]
        options%training%optimizer = MLP_OPTIMIZER_SGD
        options%training%max_epochs = 1
        options%training%learning_rate = 0.01_dp
        options%training%tolerance = 0.0_dp
        options%training%restore_best = .false.
        call model%fit(x, target, status, options, state, validation_x, &
            validation_target, sample_weight=weight, validation_weight=validation_weight)
        call check(status_ok(status) .and. model%fitted() .and. &
            state%training%updates == 1 .and. &
            state%training%final_validation_loss < huge(1.0_dp), &
            "weighted regressor fit and validation contract", failures)
    end subroutine test_weighted_regressor

    subroutine check(condition, message, failures)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: message
        integer, intent(inout) :: failures

        if (.not. condition) then
            write (*, '(a)') "FAIL: "//trim(message)
            failures = failures + 1
        end if
    end subroutine check

end program test_mlp_weighted_training
