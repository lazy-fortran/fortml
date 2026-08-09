program fortml_bench_mlp_loss_scaling
    !! Release workload for the deterministic MLP loss-scale recurrence.
    use, intrinsic :: iso_fortran_env, only: real32
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_ok, status_set, FORTNUM_OK
    use fortml_mlp, only: mlp_t, MLP_LINEAR
    use fortml_mlp_training, only: mlp_loss_scale_state_t, mlp_train, &
        mlp_training_options_t, mlp_training_state_t, MLP_OPTIMIZER_SGD, &
        mlp_training_checkpoint_t, MLP_PRECISION_FP32, MLP_PRECISION_FP16, &
        MLP_PRECISION_BF16, MLP_EVENT_UPDATE_SKIPPED
    implicit none

    type(mlp_loss_scale_state_t) :: scaler
    type(mlp_t) :: model
    type(mlp_training_options_t) :: options
    type(mlp_training_state_t) :: state
    type(mlp_training_checkpoint_t) :: checkpoint
    type(fortnum_status_t) :: status
    real(dp) :: x(3, 1), target(3, 1), before(2), elapsed, started
    real(dp) :: gradient(2), scaled_gradient(2), recovered_gradient(2)
    real(dp) :: overflowing_gradient(2)
    real(dp) :: master_parameters(2), rounded_parameters(2)
    real(dp) :: overflow_x(2, 1), overflow_target(2, 1), overflow_before(2)
    integer :: i, skipped_events

    call cpu_time(started)
    call scaler%initialize(status, enabled=.true., initial_scale=8.0_dp, &
        growth_factor=2.0_dp, backoff_factor=0.5_dp, growth_interval=2, &
        minimum_scale=1.0_dp, maximum_scale=32.0_dp)
    if (.not. status_ok(status)) error stop "loss-scale initialization failed"
    write (*, '(a,",",es24.16)') "recurrence_initial_scale", scaler%scale
    do i = 1, 2
        call scaler%observe(.true., .true., status)
        if (.not. status_ok(status)) error stop "loss-scale finite update failed"
        write (*, '(a,i0,",",es24.16)') "recurrence_finite", i, scaler%scale
    end do
    call scaler%observe(.false., .false., status)
    if (.not. status_ok(status)) error stop "loss-scale overflow update failed"
    write (*, '(a,",",es24.16,",",i0,",",i0)') "recurrence_overflow", &
        scaler%scale, scaler%overflow_count, scaler%skipped_updates
    gradient = [1.25_dp, -2.5_dp]
    call scaler%scale_gradient(gradient, scaled_gradient, status)
    if (.not. status_ok(status)) error stop "loss-scale gradient scaling failed"
    call scaler%unscale_gradient(scaled_gradient, recovered_gradient, status)
    if (.not. status_ok(status)) error stop "loss-scale gradient unscaling failed"
    write (*, '(a,",",es24.16,",",i0)') "gradient_products", &
        maxval(abs(recovered_gradient - gradient)), &
        merge(1, 0, scaler%scaled_gradient_finite(scaled_gradient))
    overflowing_gradient = [huge(1.0_dp), 0.0_dp]
    call scaler%scale_gradient(overflowing_gradient, scaled_gradient, status)
    write (*, '(a,",",i0,",",i0)') "gradient_overflow", &
        merge(1, 0, .not. scaler%scaled_gradient_finite(scaled_gradient)), &
        scaler%overflow_count
    call scaler%unscale_gradient(scaled_gradient, recovered_gradient, status)
    write (*, '(a,",",i0)') "gradient_overflow_commit", status%code

    ! Exercise the production FP32 skip path.  The target/gradient are finite
    ! at the FP32 boundary, but the deliberately huge scale overflows the
    ! binary64 scaled vector.  The callback event makes the discarded update
    ! observable without changing the parameter trajectory.
    overflow_x(:, 1) = [1.0_dp, -1.0_dp]
    overflow_target(:, 1) = [1.0e38_dp, -1.0e38_dp]
    call model%initialize([1, 1], status, output_activation=MLP_LINEAR)
    call model%set_parameters([0.0_dp, 0.0_dp], status)
    overflow_before = model%parameters()
    options%optimizer = MLP_OPTIMIZER_SGD
    options%precision_kind = MLP_PRECISION_FP32
    options%max_epochs = 1
    options%learning_rate = 0.1_dp
    options%tolerance = 0.0_dp
    options%restore_best = .false.
    call options%loss_scale%initialize(status, enabled=.true., &
        initial_scale=1.0e300_dp, maximum_scale=1.0e300_dp, &
        backoff_factor=0.5_dp, growth_interval=1000)
    skipped_events = 0
    options%event_callback => record_training_event
    call mlp_train(model, overflow_x, overflow_target, status, options, state)
    if (.not. status_ok(status)) error stop "FP32 overflow skip failed"
    write (*, '(a,",",i0,",",i0,",",i0,",",i0,",",es24.16)') &
        "fp32_overflow_skip", state%updates, state%loss_scale%overflow_count, &
        state%loss_scale%skipped_updates, skipped_events, &
        maxval(abs(model%parameters() - overflow_before))

    options%event_callback => null()
    options%precision_kind = MLP_PRECISION_FP64
    options%loss_scale = scaler
    call scaler%initialize(status, enabled=.true., initial_scale=8.0_dp, &
        growth_factor=2.0_dp, backoff_factor=0.5_dp, growth_interval=2, &
        minimum_scale=1.0_dp, maximum_scale=32.0_dp)
    if (.not. status_ok(status)) error stop "loss-scale training state reset failed"

    x(:, 1) = [-1.0_dp, 0.0_dp, 1.0_dp]
    target(:, 1) = 0.5_dp*x(:, 1) + 0.25_dp
    call model%initialize([1, 1], status, output_activation=MLP_LINEAR)
    call model%set_parameters([0.0_dp, 0.0_dp], status)
    options%optimizer = MLP_OPTIMIZER_SGD
    options%max_epochs = 2
    options%learning_rate = 0.1_dp
    options%tolerance = 0.0_dp
    options%restore_best = .false.
    options%loss_scale = scaler
    call mlp_train(model, x, target, status, options, state)
    if (.not. status_ok(status)) error stop "FP64 loss-scale training failed"
    write (*, '(a,",",i0,",",es24.16,",",i0)') "fp64_training", &
        state%updates, state%loss_scale%scale, state%loss_scale%skipped_updates

    call model%set_parameters([0.123456789012345_dp, -0.234567890123456_dp], status)
    options%max_epochs = 3
    options%learning_rate = 0.05_dp
    options%loss_scale%enabled = .true.
    options%loss_scale%initial_scale = 2.0_dp
    options%loss_scale%scale = 2.0_dp
    options%loss_scale%growth_interval = 100
    options%precision_kind = MLP_PRECISION_FP32
    call mlp_train(model, x, target, status, options, state)
    if (.not. status_ok(status)) error stop "FP32 master training failed"
    master_parameters = model%parameters()
    rounded_parameters = real(real(master_parameters, real32), dp)
    write (*, '(a,",",i0,",",i0,",",es24.16,",",es24.16,",",es24.16,",",es24.16)') "fp32_training", &
        state%updates, state%precision_kind, state%loss_scale%scale, &
        maxval(abs(master_parameters - rounded_parameters)), master_parameters
    call model%set_parameters([0.123456789012345_dp, -0.234567890123456_dp], status)
    options%max_epochs = 2
    options%loss_scale%initial_scale = 2.0_dp
    options%loss_scale%scale = 2.0_dp
    call mlp_train(model, x, target, status, options, state, checkpoint=checkpoint)
    if (.not. status_ok(status) .or. .not. checkpoint%valid()) then
        error stop "FP32 checkpoint capture failed"
    end if
    write (*, '(a,",",i0,",",es24.16)') "fp32_checkpoint", &
        checkpoint%precision_kind, checkpoint%parameters(1)
    options%precision_kind = MLP_PRECISION_FP16
    before = model%parameters()
    call mlp_train(model, x, target, status, options, state)
    if (status_ok(status) .or. maxval(abs(model%parameters() - before)) /= 0.0_dp) then
        error stop "FP16 refusal contract failed"
    end if
    write (*, '(a,",",i0)') "fp16_typed_refusal", status%code
    options%precision_kind = MLP_PRECISION_BF16
    call mlp_train(model, x, target, status, options, state)
    if (status_ok(status) .or. maxval(abs(model%parameters() - before)) /= 0.0_dp) then
        error stop "BF16 refusal contract failed"
    end if
    write (*, '(a,",",i0)') "bf16_typed_refusal", status%code
    call cpu_time(elapsed)
    write (*, '(a,",",es24.16)') "elapsed_seconds", elapsed - started

contains

    subroutine record_training_event(event, epoch, update, loss, validation_loss, &
            gradient_norm, learning_rate, stop, callback_status)
        integer, intent(in) :: event, epoch, update
        real(dp), intent(in) :: loss, validation_loss, gradient_norm, learning_rate
        logical, intent(out) :: stop
        type(fortnum_status_t), intent(out) :: callback_status

        if (event == MLP_EVENT_UPDATE_SKIPPED) skipped_events = skipped_events + 1
        stop = .false.
        call status_set(callback_status, FORTNUM_OK, "")
    end subroutine record_training_event
end program fortml_bench_mlp_loss_scaling
