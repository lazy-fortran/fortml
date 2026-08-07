program test_mlp_adagrad
    !! Independent recurrence and checkpoint oracles for MLP Adagrad training.
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_ok
    use fortml_mlp, only: mlp_t, MLP_LINEAR
    use fortml_mlp_training, only: mlp_training_options_t, &
        mlp_training_state_t, mlp_training_checkpoint_t, mlp_train, &
        MLP_OPTIMIZER_ADAGRAD
    implicit none

    integer :: failures

    failures = 0
    call test_two_step_oracle(failures)
    call test_checkpoint_resume(failures)
    call test_option_refusals(failures)
    if (failures > 0) then
        write (*, '(a,i0)') "FAIL MLP Adagrad cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS MLP Adagrad independent behavioral oracles"

contains

    subroutine test_two_step_oracle(failures)
        integer, intent(inout) :: failures
        type(mlp_t) :: model
        type(mlp_training_options_t) :: options
        type(mlp_training_state_t) :: state
        type(fortnum_status_t) :: status
        real(dp) :: x(3, 1), target(3, 1), theta(2), expected(2)
        real(dp) :: gradient(2), accumulated(2)
        real(dp), parameter :: rate = 0.2_dp, epsilon = 0.1_dp
        integer :: step

        x(:, 1) = [-1.0_dp, 0.0_dp, 1.0_dp]
        target(:, 1) = x(:, 1)
        theta = 0.0_dp
        expected = theta
        accumulated = 0.0_dp
        call model%initialize([1, 1], status, output_activation=MLP_LINEAR)
        call model%set_parameters(theta, status)
        options%optimizer = MLP_OPTIMIZER_ADAGRAD
        options%max_epochs = 2
        options%learning_rate = rate
        options%epsilon = epsilon
        options%tolerance = 0.0_dp
        options%restore_best = .false.
        call mlp_train(model, x, target, status, options, state)
        do step = 1, 2
            gradient = [2.0_dp*(expected(1) - 1.0_dp)/3.0_dp, expected(2)]
            accumulated = accumulated + gradient**2
            expected = expected - rate*gradient/(sqrt(accumulated) + epsilon)
        end do
        theta = model%parameters()
        call check(status_ok(status), "full-batch Adagrad status", failures)
        call check(state%updates == 2 .and. state%epochs == 2, &
            "full-batch Adagrad update count", failures)
        call check(maxval(abs(theta - expected)) < 2.0e-14_dp, &
            "full-batch Adagrad recurrence oracle", failures)
    end subroutine test_two_step_oracle

    subroutine test_checkpoint_resume(failures)
        integer, intent(inout) :: failures
        type(mlp_t) :: full_model, resumed_model
        type(mlp_training_options_t) :: full_options, split_options
        type(mlp_training_state_t) :: full_state, resumed_state
        type(mlp_training_checkpoint_t) :: full_checkpoint, resumed_checkpoint
        type(fortnum_status_t) :: status
        real(dp) :: x(4, 1), target(4, 1), full_theta(2), resumed_theta(2)

        x(:, 1) = [-2.0_dp, -1.0_dp, 1.0_dp, 2.0_dp]
        target(:, 1) = 0.5_dp*x(:, 1) + 0.25_dp
        call full_model%initialize([1, 1], status, output_activation=MLP_LINEAR)
        call resumed_model%initialize([1, 1], status, output_activation=MLP_LINEAR)
        call full_model%set_parameters([0.0_dp, 0.0_dp], status)
        call resumed_model%set_parameters([0.0_dp, 0.0_dp], status)
        full_options%optimizer = MLP_OPTIMIZER_ADAGRAD
        full_options%max_epochs = 4
        full_options%learning_rate = 0.08_dp
        full_options%epsilon = 1.0e-5_dp
        full_options%tolerance = 0.0_dp
        full_options%restore_best = .false.
        split_options = full_options
        split_options%max_epochs = 2
        call mlp_train(full_model, x, target, status, full_options, full_state, &
            checkpoint=full_checkpoint)
        call mlp_train(resumed_model, x, target, status, split_options, &
            checkpoint=resumed_checkpoint)
        call check(status_ok(status) .and. resumed_checkpoint%valid() .and. &
            resumed_checkpoint%optimizer == MLP_OPTIMIZER_ADAGRAD .and. &
            resumed_checkpoint%adam_step_count == 2, &
            "Adagrad checkpoint metadata", failures)
        call check(maxval(abs(resumed_checkpoint%second_moment)) < 1.0e-15_dp, &
            "Adagrad checkpoint has no second moment", failures)
        call mlp_train(resumed_model, x, target, status, full_options, resumed_state, &
            checkpoint=resumed_checkpoint)
        full_theta = full_model%parameters()
        resumed_theta = resumed_model%parameters()
        call check(status_ok(status), "Adagrad resumed status", failures)
        call check(resumed_state%updates == full_state%updates .and. &
            maxval(abs(full_theta - resumed_theta)) < 1.0e-14_dp .and. &
            maxval(abs(full_state%loss_history - resumed_state%loss_history)) < &
            1.0e-14_dp .and. maxval(abs(full_checkpoint%first_moment - &
            resumed_checkpoint%first_moment)) < 1.0e-14_dp, &
            "uninterrupted/resumed Adagrad trajectory", failures)
    end subroutine test_checkpoint_resume

    subroutine test_option_refusals(failures)
        integer, intent(inout) :: failures
        type(mlp_t) :: model
        type(mlp_training_options_t) :: options
        type(mlp_training_state_t) :: state
        type(fortnum_status_t) :: status
        real(dp) :: x(2, 1), target(2, 1)

        x(:, 1) = [-1.0_dp, 1.0_dp]
        target(:, 1) = x(:, 1)
        call model%initialize([1, 1], status, output_activation=MLP_LINEAR)
        options%optimizer = MLP_OPTIMIZER_ADAGRAD
        options%max_epochs = 1
        options%epsilon = 0.0_dp
        call mlp_train(model, x, target, status, options, state)
        call check(.not. status_ok(status), "zero Adagrad epsilon refusal", failures)
        options%epsilon = 1.0e-8_dp
        options%learning_rate = -1.0_dp
        call mlp_train(model, x, target, status, options, state)
        call check(.not. status_ok(status), "negative Adagrad rate refusal", failures)
    end subroutine test_option_refusals

    subroutine check(condition, label, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: label
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (*, '(a)') "  FAIL "//label
        end if
    end subroutine check

end program test_mlp_adagrad
