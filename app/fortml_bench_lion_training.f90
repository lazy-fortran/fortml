program fortml_bench_lion_training
    !! Release benchmark app for the stateful CPU Lion trainer.
    use, intrinsic :: iso_fortran_env, only: int64
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_ok
    use fortml_mlp, only: mlp_t, MLP_LINEAR
    use fortml_mlp_training, only: mlp_training_options_t, mlp_training_state_t, &
        mlp_training_checkpoint_t, mlp_train, MLP_OPTIMIZER_LION
    implicit none

    type(mlp_t) :: full_model, split_model
    type(mlp_training_options_t) :: options, split_options
    type(mlp_training_state_t) :: state, split_state
    type(mlp_training_checkpoint_t) :: checkpoint, split_checkpoint
    type(fortnum_status_t) :: status
    real(dp) :: x(8, 1), target(8, 1), full_elapsed, split_elapsed
    real(dp) :: final_loss, trajectory_error, ema_norm
    integer(int64) :: tick_start, tick_end, ticks_per_second
    integer :: i

    do i = 1, size(x, 1)
        x(i, 1) = -1.0_dp + 2.0_dp*real(i-1, dp)/real(size(x, 1)-1, dp)
        target(i, 1) = 0.6_dp*x(i, 1) - 0.15_dp
    end do
    call system_clock(count_rate=ticks_per_second)
    call full_model%initialize([1, 1], status, output_activation=MLP_LINEAR, &
        initialization_seed=19)
    if (.not. status_ok(status)) error stop 1
    call split_model%initialize([1, 1], status, output_activation=MLP_LINEAR, &
        initialization_seed=19)
    if (.not. status_ok(status)) error stop 1
    call full_model%set_parameters([0.0_dp, 0.0_dp], status)
    if (.not. status_ok(status)) error stop 1
    call split_model%set_parameters([0.0_dp, 0.0_dp], status)
    if (.not. status_ok(status)) error stop 1
    options%optimizer = MLP_OPTIMIZER_LION
    options%max_epochs = 64
    options%batch_size = 4
    options%learning_rate = 2.0e-3_dp
    options%beta1 = 0.9_dp
    options%beta2 = 0.99_dp
    options%weight_decay = 1.0e-3_dp
    options%ema_decay = 0.9_dp
    options%tolerance = 0.0_dp
    options%restore_best = .false.

    call system_clock(tick_start)
    call mlp_train(full_model, x, target, status, options, state, &
        checkpoint=checkpoint)
    call system_clock(tick_end)
    if (.not. status_ok(status)) error stop 1
    full_elapsed = real(tick_end-tick_start, dp)/real(ticks_per_second, dp)
    final_loss = state%final_loss
    ema_norm = sqrt(sum(state%ema_parameters*state%ema_parameters))
    write (*, '(a,",pass,final_loss,",es24.16,",",es24.16)') &
        "lion_training", final_loss, full_elapsed
    write (*, '(a,",pass,ema_parameter_l2_norm,",es24.16,",",es24.16)') &
        "lion_ema", ema_norm, full_elapsed/real(options%max_epochs, dp)

    split_options = options
    split_options%max_epochs = 32
    call system_clock(tick_start)
    call mlp_train(split_model, x, target, status, split_options, split_state, &
        checkpoint=split_checkpoint)
    if (.not. status_ok(status)) error stop 1
    call mlp_train(split_model, x, target, status, options, split_state, &
        checkpoint=split_checkpoint)
    call system_clock(tick_end)
    if (.not. status_ok(status)) error stop 1
    split_elapsed = real(tick_end-tick_start, dp)/real(ticks_per_second, dp)
    trajectory_error = maxval(abs(full_model%parameters()-split_model%parameters()))
    write (*, '(a,",pass,resume_max_abs_error,",es24.16,",",es24.16)') &
        "lion_checkpoint_resume", trajectory_error, split_elapsed
end program fortml_bench_lion_training
