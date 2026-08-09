program fortml_bench_mlp_loss_scaling
    !! Release workload for the deterministic MLP loss-scale recurrence.
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_ok
    use fortml_mlp, only: mlp_t, MLP_LINEAR
    use fortml_mlp_training, only: mlp_loss_scale_state_t, mlp_train, &
        mlp_training_options_t, mlp_training_state_t, MLP_OPTIMIZER_SGD, &
        MLP_PRECISION_FP32
    implicit none

    type(mlp_loss_scale_state_t) :: scaler
    type(mlp_t) :: model
    type(mlp_training_options_t) :: options
    type(mlp_training_state_t) :: state
    type(fortnum_status_t) :: status
    real(dp) :: x(3, 1), target(3, 1), before(2), elapsed, started
    integer :: i

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

    options%precision_kind = MLP_PRECISION_FP32
    before = model%parameters()
    call mlp_train(model, x, target, status, options, state)
    if (status_ok(status) .or. maxval(abs(model%parameters() - before)) /= 0.0_dp) then
        error stop "lower-precision refusal contract failed"
    end if
    write (*, '(a,",",i0)') "fp32_typed_refusal", status%code
    call cpu_time(elapsed)
    write (*, '(a,",",es24.16)') "elapsed_seconds", elapsed - started
end program fortml_bench_mlp_loss_scaling
