program test_mlp_loss_scaling
    !! Independent recurrence and checkpoint oracle for MLP loss scaling.
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    use fortml_mlp, only: mlp_t, MLP_LINEAR
    use fortml_mlp_training, only: mlp_loss_scale_state_t, mlp_training_options_t, &
        mlp_training_state_t, mlp_training_checkpoint_t, mlp_train, &
        MLP_OPTIMIZER_SGD, MLP_PRECISION_FP32
    use fortml_mlp_checkpoint, only: mlp_checkpoint_save, mlp_checkpoint_load
    implicit none

    integer :: failures

    failures = 0
    call test_recurrence(failures)
    call test_checkpoint_round_trip(failures)
    call test_lower_precision_refusal(failures)
    if (failures > 0) then
        write (*, '(a,i0)') "FAIL MLP loss-scaling cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS MLP loss-scaling independent behavioral oracles"

contains

    subroutine test_recurrence(failures)
        integer, intent(inout) :: failures
        type(mlp_loss_scale_state_t) :: scaler
        type(fortnum_status_t) :: status

        call scaler%initialize(status, enabled=.true., initial_scale=8.0_dp, &
            growth_factor=2.0_dp, backoff_factor=0.5_dp, growth_interval=2, &
            minimum_scale=1.0_dp, maximum_scale=32.0_dp)
        call check(status_ok(status) .and. scaler%valid() .and. &
            scaler%scale == 8.0_dp, "configured scaler is valid", failures)
        call scaler%observe(.true., .true., status)
        call check(status_ok(status) .and. scaler%scale == 8.0_dp .and. &
            scaler%good_steps == 1, "first finite update recurrence", failures)
        call scaler%observe(.true., .true., status)
        call check(status_ok(status) .and. scaler%scale == 16.0_dp .and. &
            scaler%good_steps == 0, "growth interval recurrence", failures)
        call scaler%observe(.false., .false., status)
        call check(status_ok(status) .and. scaler%scale == 8.0_dp .and. &
            scaler%overflow_count == 1 .and. scaler%skipped_updates == 1 .and. &
            scaler%good_steps == 0, "overflow backoff recurrence", failures)
        call scaler%observe(.false., .false., status)
        call check(status_ok(status) .and. scaler%scale == 4.0_dp .and. &
            scaler%overflow_count == 2, "repeated overflow backoff", failures)
        call scaler%observe(.true., .false., status)
        call check(status_ok(status) .and. scaler%scale == 4.0_dp .and. &
            scaler%good_steps == 0, "non-update does not grow scale", failures)
    end subroutine test_recurrence

    subroutine test_checkpoint_round_trip(failures)
        integer, intent(inout) :: failures
        type(mlp_t) :: model
        type(mlp_training_options_t) :: options
        type(mlp_training_state_t) :: state
        type(mlp_training_checkpoint_t) :: checkpoint, loaded
        type(fortnum_status_t) :: status
        real(dp) :: x(3, 1), target(3, 1)

        x(:, 1) = [-1.0_dp, 0.0_dp, 1.0_dp]
        target(:, 1) = 0.5_dp*x(:, 1) + 0.25_dp
        call model%initialize([1, 1], status, output_activation=MLP_LINEAR)
        call model%set_parameters([0.0_dp, 0.0_dp], status)
        options%optimizer = MLP_OPTIMIZER_SGD
        options%max_epochs = 2
        options%learning_rate = 0.1_dp
        options%tolerance = 0.0_dp
        options%restore_best = .false.
        call options%loss_scale%initialize(status, enabled=.true., initial_scale=4.0_dp, &
            growth_factor=2.0_dp, backoff_factor=0.5_dp, growth_interval=1, &
            minimum_scale=1.0_dp, maximum_scale=64.0_dp)
        call mlp_train(model, x, target, status, options, state, checkpoint=checkpoint)
        call check(status_ok(status) .and. checkpoint%valid() .and. &
            checkpoint%loss_scale%enabled .and. checkpoint%loss_scale%scale == 16.0_dp .and. &
            checkpoint%loss_scale%good_steps == 0, &
            "training captures dynamic loss-scale state", failures)
        call mlp_checkpoint_save(checkpoint, "test_mlp_loss_scaling_checkpoint.txt", status)
        call mlp_checkpoint_load(loaded, "test_mlp_loss_scaling_checkpoint.txt", status)
        call check(status_ok(status) .and. loaded%valid() .and. &
            loaded%loss_scale%scale == checkpoint%loss_scale%scale .and. &
            loaded%loss_scale%growth_interval == checkpoint%loss_scale%growth_interval .and. &
            loaded%loss_scale%overflow_count == checkpoint%loss_scale%overflow_count, &
            "formatted checkpoint preserves loss-scale state", failures)
    end subroutine test_checkpoint_round_trip

    subroutine test_lower_precision_refusal(failures)
        integer, intent(inout) :: failures
        type(mlp_t) :: model
        type(mlp_training_options_t) :: options
        type(mlp_training_state_t) :: state
        type(fortnum_status_t) :: status
        real(dp) :: x(2, 1), target(2, 1), before(2)

        x(:, 1) = [-1.0_dp, 1.0_dp]
        target(:, 1) = x(:, 1)
        call model%initialize([1, 1], status, output_activation=MLP_LINEAR)
        call model%set_parameters([0.25_dp, -0.5_dp], status)
        before = model%parameters()
        options%precision_kind = MLP_PRECISION_FP32
        options%max_epochs = 1
        options%loss_scale%enabled = .true.
        options%loss_scale%initial_scale = 16.0_dp
        options%loss_scale%scale = 16.0_dp
        call mlp_train(model, x, target, status, options, state)
        call check(status%code == FORTNUM_NOT_IMPLEMENTED .and. &
            maxval(abs(model%parameters() - before)) == 0.0_dp .and. &
            state%loss_scale%enabled, &
            "unsupported resident lower precision is typed and non-mutating", failures)
    end subroutine test_lower_precision_refusal

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (*, '(a)') "  FAIL [mlp-loss-scale] "//description
        end if
    end subroutine check

end program test_mlp_loss_scaling
