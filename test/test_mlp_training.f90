program test_mlp_training
    !! Independent behavioral checks for deterministic MLP training products.
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_ok
    use fortml_mlp, only: mlp_t, MLP_LINEAR
    use fortml_mlp_training, only: mlp_training_options_t, &
        mlp_training_state_t, mlp_loss_value_gradient, mlp_train
    implicit none

    integer :: failures

    failures = 0
    call test_first_adam_step(failures)
    call test_reproducible_minibatch_and_callback(failures)
    call test_l2_hyperparameter_product(failures)
    if (failures > 0) then
        write (*, '(a,i0)') "FAIL MLP training cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS MLP training independent behavioral oracles"

contains

    subroutine test_first_adam_step(failures)
        integer, intent(inout) :: failures
        type(mlp_t) :: model
        type(mlp_training_options_t) :: options
        type(mlp_training_state_t) :: state
        type(fortnum_status_t) :: status
        real(dp) :: x(3, 1), target(3, 1), theta(2)

        x(:, 1) = [-1.0_dp, 0.0_dp, 1.0_dp]
        target(:, 1) = [-1.0_dp, 0.0_dp, 1.0_dp]
        call model%initialize([1, 1], status, output_activation=MLP_LINEAR)
        theta = 0.0_dp
        call model%set_parameters(theta, status)
        options%max_epochs = 1
        options%learning_rate = 0.1_dp
        options%tolerance = 0.0_dp
        call mlp_train(model, x, target, status, options, state)
        theta = model%parameters()
        ! For the zero-initialized affine model, dL/dw=-2/3 and dL/db=0.
        ! Adam's first bias-corrected step is therefore exactly +learning_rate
        ! for w and zero for b, an oracle independent of the implementation.
        call check(status_ok(status), "first Adam status", failures)
        call check(state%updates == 1 .and. state%epochs == 1, &
            "one full-batch update", failures)
        call check(abs(theta(1) - options%learning_rate*(2.0_dp/3.0_dp)/ &
            (2.0_dp/3.0_dp + options%epsilon)) < 2.0e-14_dp .and. &
            abs(theta(2)) < 2.0e-14_dp, "first Adam step oracle", failures)
        call check(size(state%loss_history) == 1 .and. &
            state%final_loss < state%initial_loss, "loss history and decrease", &
            failures)
    end subroutine test_first_adam_step

    subroutine test_reproducible_minibatch_and_callback(failures)
        integer, intent(inout) :: failures
        type(mlp_t) :: first_model, second_model
        type(mlp_training_options_t) :: options
        type(mlp_training_state_t) :: first_state, second_state
        type(fortnum_status_t) :: status
        real(dp) :: x(6, 1), target(6, 1), first_theta(3), second_theta(3)

        x(:, 1) = [-2.0_dp, -1.0_dp, -0.5_dp, 0.5_dp, 1.0_dp, 2.0_dp]
        target(:, 1) = 0.75_dp*x(:, 1) - 0.2_dp
        call first_model%initialize([1, 2, 1], status, &
            output_activation=MLP_LINEAR, initialization_seed=31)
        call second_model%initialize([1, 2, 1], status, &
            output_activation=MLP_LINEAR, initialization_seed=31)
        options%max_epochs = 12
        options%batch_size = 2
        options%shuffle = .true.
        options%shuffle_seed = 123
        options%learning_rate = 0.02_dp
        options%tolerance = 0.0_dp
        options%restore_best = .false.
        options%callback => stop_at_three
        call mlp_train(first_model, x, target, status, options, first_state)
        call mlp_train(second_model, x, target, status, options, second_state)
        first_theta = first_model%parameters()
        second_theta = second_model%parameters()
        call check(status_ok(status), "mini-batch status", failures)
        call check(first_state%early_stopped .and. first_state%epochs == 3, &
            "callback early stop", failures)
        call check(first_state%updates == 9, "mini-batch update count", failures)
        call check(second_state%epochs == first_state%epochs .and. &
            maxval(abs(first_theta - second_theta)) < 1.0e-14_dp .and. &
            maxval(abs(first_state%loss_history - second_state%loss_history)) < &
            1.0e-14_dp, "reproducible shuffled training", failures)
    end subroutine test_reproducible_minibatch_and_callback

    subroutine test_l2_hyperparameter_product(failures)
        integer, intent(inout) :: failures
        type(mlp_t) :: model
        type(fortnum_status_t) :: status
        real(dp) :: x(3, 1), target(3, 1), theta(2), gradient(2)
        real(dp) :: value, l2_bar, plus, minus, h
        real(dp) :: gradient_plus(2), gradient_minus(2), ignored

        x(:, 1) = [-1.0_dp, 0.0_dp, 2.0_dp]
        target(:, 1) = [0.5_dp, -0.25_dp, 1.5_dp]
        theta = [0.3_dp, -0.2_dp]
        call model%initialize([1, 1], status, output_activation=MLP_LINEAR)
        call model%set_parameters(theta, status)
        call mlp_loss_value_gradient(model, x, target, 0.4_dp, value, &
            gradient, l2_bar, status)
        h = 1.0e-6_dp
        call mlp_loss_value_gradient(model, x, target, 0.4_dp + h, plus, &
            gradient_plus, ignored, status)
        call mlp_loss_value_gradient(model, x, target, 0.4_dp - h, minus, &
            gradient_minus, ignored, status)
        call check(status_ok(status), "loss product status", failures)
        call check(abs(l2_bar - 0.5_dp*sum(theta*theta)) < 1.0e-14_dp, &
            "analytic L2 hyperparameter derivative", failures)
        call check(abs(l2_bar - (plus - minus)/(2.0_dp*h)) < 2.0e-10_dp, &
            "finite-difference L2 derivative", failures)
    end subroutine test_l2_hyperparameter_product

    subroutine stop_at_three(epoch, loss, gradient_norm, stop)
        integer, intent(in) :: epoch
        real(dp), intent(in) :: loss, gradient_norm
        logical, intent(out) :: stop

        stop = epoch >= 3
        if (loss < 0.0_dp .or. gradient_norm < 0.0_dp) stop = .false.
    end subroutine stop_at_three

    subroutine check(condition, label, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: label
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (*, '(a)') "  FAIL "//label
        end if
    end subroutine check

end program test_mlp_training
