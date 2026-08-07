program test_mlp_adamw
    !! Independent value and trajectory oracles for AdamW MLP training.
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_ok
    use fortml_mlp, only: mlp_t, MLP_LINEAR
    use fortml_mlp_training, only: mlp_training_options_t, &
        mlp_training_state_t, mlp_training_checkpoint_t, mlp_train, &
        MLP_OPTIMIZER_ADAMW
    implicit none

    integer :: failures

    failures = 0
    call test_two_step_full_batch_oracle(failures)
    call test_minibatch_determinism_and_resume(failures)
    call test_option_refusals(failures)
    if (failures > 0) then
        write (*, '(a,i0)') "FAIL MLP AdamW cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS MLP AdamW independent behavioral oracles"

contains

    subroutine test_two_step_full_batch_oracle(failures)
        integer, intent(inout) :: failures
        type(mlp_t) :: model
        type(mlp_training_options_t) :: options
        type(mlp_training_state_t) :: state
        type(fortnum_status_t) :: status
        real(dp) :: x(1, 1), target(1, 1), theta(2), expected(2)
        real(dp) :: m(2), v(2), gradient(2), direction(2)
        integer :: step
        real(dp), parameter :: rate = 0.1_dp, beta1 = 0.8_dp
        real(dp), parameter :: beta2 = 0.9_dp, epsilon = 1.0e-7_dp
        real(dp), parameter :: weight_decay = 0.2_dp

        x = 1.0_dp
        target = 0.0_dp
        theta = [0.5_dp, 0.25_dp]
        expected = theta
        m = 0.0_dp
        v = 0.0_dp
        call model%initialize([1, 1], status, output_activation=MLP_LINEAR)
        call model%set_parameters(theta, status)
        options%optimizer = MLP_OPTIMIZER_ADAMW
        options%max_epochs = 2
        options%learning_rate = rate
        options%beta1 = beta1
        options%beta2 = beta2
        options%epsilon = epsilon
        options%weight_decay = weight_decay
        options%tolerance = 0.0_dp
        options%restore_best = .false.
        call mlp_train(model, x, target, status, options, state)
        do step = 1, 2
            gradient = [expected(1) + expected(2), expected(1) + expected(2)]
            m = beta1*m + (1.0_dp - beta1)*gradient
            v = beta2*v + (1.0_dp - beta2)*gradient**2
            direction = (m/(1.0_dp - beta1**step))/ &
                (sqrt(v/(1.0_dp - beta2**step)) + epsilon)
            expected = (1.0_dp - rate*weight_decay)*expected - rate*direction
        end do
        theta = model%parameters()
        call check(status_ok(status), "full-batch AdamW status", failures)
        call check(state%updates == 2 .and. state%epochs == 2, &
            "full-batch AdamW update count", failures)
        call check(maxval(abs(theta - expected)) < 2.0e-13_dp, &
            "full-batch AdamW two-step oracle", failures)
    end subroutine test_two_step_full_batch_oracle

    subroutine test_minibatch_determinism_and_resume(failures)
        integer, intent(inout) :: failures
        type(mlp_t) :: first_model, second_model, resumed_model
        type(mlp_training_options_t) :: options, split_options
        type(mlp_training_state_t) :: first_state, second_state, resumed_state
        type(mlp_training_checkpoint_t) :: checkpoint
        type(fortnum_status_t) :: status
        real(dp) :: x(6, 1), target(6, 1), first_theta(3), second_theta(3)
        real(dp) :: resumed_theta(3)

        x(:, 1) = [-2.0_dp, -1.0_dp, -0.5_dp, 0.5_dp, 1.0_dp, 2.0_dp]
        target(:, 1) = 0.6_dp*x(:, 1) - 0.15_dp
        call first_model%initialize([1, 2, 1], status, &
            output_activation=MLP_LINEAR, initialization_seed=73)
        call second_model%initialize([1, 2, 1], status, &
            output_activation=MLP_LINEAR, initialization_seed=73)
        call resumed_model%initialize([1, 2, 1], status, &
            output_activation=MLP_LINEAR, initialization_seed=73)
        options%optimizer = MLP_OPTIMIZER_ADAMW
        options%max_epochs = 5
        options%batch_size = 2
        options%shuffle = .true.
        options%shuffle_seed = 911
        options%learning_rate = 0.012_dp
        options%weight_decay = 0.03_dp
        options%tolerance = 0.0_dp
        options%restore_best = .false.
        split_options = options
        split_options%max_epochs = 2
        call mlp_train(first_model, x, target, status, options, first_state)
        call mlp_train(second_model, x, target, status, options, second_state)
        call mlp_train(resumed_model, x, target, status, split_options, &
            checkpoint=checkpoint)
        call check(status_ok(status) .and. checkpoint%valid(), &
            "AdamW partial checkpoint", failures)
        call mlp_train(resumed_model, x, target, status, options, resumed_state, &
            checkpoint=checkpoint)
        first_theta = first_model%parameters()
        second_theta = second_model%parameters()
        resumed_theta = resumed_model%parameters()
        call check(status_ok(status), "AdamW resumed status", failures)
        call check(maxval(abs(first_theta - second_theta)) < 1.0e-14_dp .and. &
            maxval(abs(first_theta - resumed_theta)) < 1.0e-14_dp .and. &
            maxval(abs(first_state%loss_history - resumed_state%loss_history)) < &
            1.0e-14_dp .and. resumed_state%updates == first_state%updates, &
            "deterministic minibatch and resumed AdamW trajectory", failures)
    end subroutine test_minibatch_determinism_and_resume

    subroutine test_option_refusals(failures)
        integer, intent(inout) :: failures
        type(mlp_t) :: model
        type(mlp_training_options_t) :: options
        type(mlp_training_state_t) :: state
        type(fortnum_status_t) :: status
        real(dp) :: x(2, 1), target(2, 1)

        x(:, 1) = [-1.0_dp, 1.0_dp]
        target(:, 1) = [-1.0_dp, 1.0_dp]
        call model%initialize([1, 1], status, output_activation=MLP_LINEAR)
        options%optimizer = MLP_OPTIMIZER_ADAMW
        options%weight_decay = -1.0_dp
        options%max_epochs = 1
        call mlp_train(model, x, target, status, options, state)
        call check(.not. status_ok(status), "negative AdamW decay refusal", failures)
        options%weight_decay = 0.01_dp
        options%learning_rate = 0.0_dp
        call mlp_train(model, x, target, status, options, state)
        call check(.not. status_ok(status), "zero AdamW learning-rate refusal", failures)
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

end program test_mlp_adamw
