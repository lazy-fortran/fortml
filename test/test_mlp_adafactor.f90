program test_mlp_adafactor
    !! Independent vector-Adafactor recurrence and checkpoint oracle for MLPs.
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_ok
    use fortml_mlp, only: mlp_t, MLP_LINEAR
    use fortml_mlp_training, only: mlp_training_options_t, mlp_training_state_t, &
        mlp_training_checkpoint_t, mlp_train, MLP_OPTIMIZER_ADAFACTOR
    implicit none

    integer :: failures

    failures = 0
    call test_recurrence(failures)
    call test_checkpoint_resume(failures)
    if (failures > 0) then
        write (*, '(a,i0)') "FAIL MLP Adafactor cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS MLP Adafactor independent behavioral oracles"

contains

    subroutine test_recurrence(failures)
        integer, intent(inout) :: failures
        type(mlp_t) :: model
        type(mlp_training_options_t) :: options
        type(mlp_training_state_t) :: state
        type(fortnum_status_t) :: status
        real(dp) :: x(3, 1), target(3, 1), expected(2), moment(2), gradient(2)
        real(dp) :: rms, clip_scale
        integer :: step

        x(:, 1) = [-1.0_dp, 0.0_dp, 1.0_dp]
        target(:, 1) = x(:, 1)
        call model%initialize([1, 1], status, output_activation=MLP_LINEAR)
        call model%set_parameters([0.0_dp, 0.0_dp], status)
        options%optimizer = MLP_OPTIMIZER_ADAFACTOR
        options%max_epochs = 3
        options%learning_rate = 0.2_dp
        options%adafactor_decay = 0.75_dp
        options%adafactor_clip_threshold = 0.8_dp
        options%epsilon = 0.05_dp
        options%tolerance = 0.0_dp
        options%restore_best = .false.
        call mlp_train(model, x, target, status, options, state)
        expected = [0.0_dp, 0.0_dp]
        moment = 0.0_dp
        do step = 1, 3
            gradient = [2.0_dp*(expected(1) - 1.0_dp)/3.0_dp, expected(2)]
            moment = options%adafactor_decay*moment + &
                (1.0_dp - options%adafactor_decay)*gradient**2
            rms = sqrt(sum(moment)/2.0_dp)
            clip_scale = max(1.0_dp, rms/options%adafactor_clip_threshold)
            expected = expected - options%learning_rate*gradient/clip_scale/ &
                (sqrt(moment) + options%epsilon)
        end do
        call check(status_ok(status) .and. state%updates == 3, &
            "Adafactor MLP update count", failures)
        call check(maxval(abs(model%parameters() - expected)) < 2.0e-13_dp, &
            "Adafactor MLP recurrence matches independent oracle", failures)
    end subroutine test_recurrence

    subroutine test_checkpoint_resume(failures)
        integer, intent(inout) :: failures
        type(mlp_t) :: full_model, resumed_model
        type(mlp_training_options_t) :: full_options, split_options
        type(mlp_training_state_t) :: full_state, resumed_state
        type(mlp_training_checkpoint_t) :: full_checkpoint, resumed_checkpoint
        type(fortnum_status_t) :: status
        real(dp) :: x(3, 1), target(3, 1), full_theta(2), resumed_theta(2)

        x(:, 1) = [-1.0_dp, 0.0_dp, 1.0_dp]
        target(:, 1) = x(:, 1)
        call full_model%initialize([1, 1], status, output_activation=MLP_LINEAR)
        call resumed_model%initialize([1, 1], status, output_activation=MLP_LINEAR)
        call full_model%set_parameters([0.0_dp, 0.0_dp], status)
        call resumed_model%set_parameters([0.0_dp, 0.0_dp], status)
        full_options%optimizer = MLP_OPTIMIZER_ADAFACTOR
        full_options%max_epochs = 5
        full_options%learning_rate = 0.12_dp
        full_options%adafactor_decay = 0.8_dp
        full_options%adafactor_clip_threshold = 1.0_dp
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
            resumed_checkpoint%optimizer == MLP_OPTIMIZER_ADAFACTOR .and. &
            resumed_checkpoint%adam_step_count == 2, &
            "Adafactor MLP checkpoint metadata", failures)
        call mlp_train(resumed_model, x, target, status, full_options, resumed_state, &
            checkpoint=resumed_checkpoint)
        full_theta = full_model%parameters()
        resumed_theta = resumed_model%parameters()
        call check(status_ok(status) .and. resumed_state%updates == full_state%updates .and. &
            maxval(abs(full_theta - resumed_theta)) < 2.0e-13_dp .and. &
            maxval(abs(full_checkpoint%first_moment - resumed_checkpoint%first_moment)) < 2.0e-13_dp, &
            "Adafactor MLP checkpoint continuation matches uninterrupted trajectory", failures)
    end subroutine test_checkpoint_resume

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (*, '(a)') "  FAIL [mlp-adafactor] "//description
        end if
    end subroutine check

end program test_mlp_adafactor
