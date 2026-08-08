program test_mlp_amsgrad
    !! Independent AMSGrad recurrence and checkpoint continuation oracles.
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_ok
    use fortml_mlp, only: mlp_t, MLP_LINEAR
    use fortml_mlp_training, only: mlp_training_options_t, mlp_training_state_t, &
        mlp_training_checkpoint_t, mlp_train, MLP_OPTIMIZER_AMSGRAD
    use fortml_mlp_checkpoint, only: mlp_checkpoint_save, mlp_checkpoint_load
    implicit none

    integer :: failures

    failures = 0
    call test_recurrence(failures)
    call test_checkpoint_resume(failures)
    if (failures > 0) then
        write (*, '(a,i0)') "FAIL MLP AMSGrad cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS MLP AMSGrad independent behavioral oracles"

contains

    subroutine test_recurrence(failures)
        integer, intent(inout) :: failures
        type(mlp_t) :: model
        type(mlp_training_options_t) :: options
        type(mlp_training_state_t) :: state
        type(fortnum_status_t) :: status
        real(dp) :: x(3, 1), target(3, 1), expected(2), first(2), second(2), vmax(2)
        real(dp) :: gradient(2), bias1, bias2
        integer :: step

        x(:, 1) = [-1.0_dp, 0.0_dp, 1.0_dp]
        target(:, 1) = x(:, 1)
        call model%initialize([1, 1], status, output_activation=MLP_LINEAR)
        call model%set_parameters([0.0_dp, 0.0_dp], status)
        options%optimizer = MLP_OPTIMIZER_AMSGRAD
        options%max_epochs = 4
        options%learning_rate = 0.15_dp
        options%beta1 = 0.7_dp
        options%beta2 = 0.8_dp
        options%epsilon = 0.03_dp
        options%tolerance = 0.0_dp
        options%restore_best = .false.
        call mlp_train(model, x, target, status, options, state)
        expected = [0.0_dp, 0.0_dp]
        first = 0.0_dp
        second = 0.0_dp
        vmax = 0.0_dp
        do step = 1, 4
            gradient = [2.0_dp*(expected(1) - 1.0_dp)/3.0_dp, expected(2)]
            first = options%beta1*first + (1.0_dp - options%beta1)*gradient
            second = options%beta2*second + (1.0_dp - options%beta2)*gradient**2
            vmax = max(vmax, second)
            bias1 = 1.0_dp - options%beta1**step
            bias2 = 1.0_dp - options%beta2**step
            expected = expected - options%learning_rate*(first/bias1)/ &
                (sqrt(vmax/bias2) + options%epsilon)
        end do
        call check(status_ok(status) .and. state%updates == 4, &
            "AMSGrad MLP update count", failures)
        call check(maxval(abs(model%parameters() - expected)) < 3.0e-13_dp, &
            "AMSGrad MLP recurrence matches independent oracle", failures)
    end subroutine test_recurrence

    subroutine test_checkpoint_resume(failures)
        integer, intent(inout) :: failures
        type(mlp_t) :: full_model, resumed_model
        type(mlp_training_options_t) :: full_options, split_options
        type(mlp_training_state_t) :: full_state, resumed_state
        type(mlp_training_checkpoint_t) :: full_checkpoint, resumed_checkpoint
        type(mlp_training_checkpoint_t) :: loaded_checkpoint
        type(fortnum_status_t) :: status
        real(dp) :: x(4, 1), target(4, 1), full_theta(2), resumed_theta(2)

        x(:, 1) = [-1.5_dp, -0.5_dp, 0.5_dp, 1.5_dp]
        target(:, 1) = 0.4_dp*x(:, 1) + 0.2_dp
        call full_model%initialize([1, 1], status, output_activation=MLP_LINEAR)
        call resumed_model%initialize([1, 1], status, output_activation=MLP_LINEAR)
        call full_model%set_parameters([0.0_dp, 0.0_dp], status)
        call resumed_model%set_parameters([0.0_dp, 0.0_dp], status)
        full_options%optimizer = MLP_OPTIMIZER_AMSGRAD
        full_options%max_epochs = 6
        full_options%learning_rate = 0.08_dp
        full_options%beta1 = 0.6_dp
        full_options%beta2 = 0.85_dp
        full_options%epsilon = 1.0e-4_dp
        full_options%tolerance = 0.0_dp
        full_options%restore_best = .false.
        split_options = full_options
        split_options%max_epochs = 2
        call mlp_train(full_model, x, target, status, full_options, full_state, &
            checkpoint=full_checkpoint)
        call mlp_train(resumed_model, x, target, status, split_options, &
            checkpoint=resumed_checkpoint)
        call check(status_ok(status) .and. resumed_checkpoint%valid() .and. &
            resumed_checkpoint%optimizer == MLP_OPTIMIZER_AMSGRAD .and. &
            allocated(resumed_checkpoint%max_second_moment) .and. &
            resumed_checkpoint%adam_step_count == 2, &
            "AMSGrad checkpoint metadata and max state", failures)
        call mlp_checkpoint_save(resumed_checkpoint, "test_mlp_amsgrad_checkpoint.txt", status)
        call mlp_checkpoint_load(loaded_checkpoint, "test_mlp_amsgrad_checkpoint.txt", status)
        call check(status_ok(status) .and. loaded_checkpoint%valid() .and. &
            allocated(loaded_checkpoint%max_second_moment), &
            "AMSGrad formatted checkpoint round trip", failures)
        call mlp_train(resumed_model, x, target, status, full_options, resumed_state, &
            checkpoint=loaded_checkpoint)
        full_theta = full_model%parameters()
        resumed_theta = resumed_model%parameters()
        call check(status_ok(status) .and. resumed_state%updates == full_state%updates .and. &
            maxval(abs(full_theta - resumed_theta)) < 3.0e-13_dp .and. &
            maxval(abs(full_checkpoint%max_second_moment - &
                loaded_checkpoint%max_second_moment)) < 3.0e-13_dp, &
            "AMSGrad checkpoint continuation matches uninterrupted trajectory", failures)
    end subroutine test_checkpoint_resume

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (*, '(a)') "  FAIL [mlp-amsgrad] "//description
        end if
    end subroutine check

end program test_mlp_amsgrad
