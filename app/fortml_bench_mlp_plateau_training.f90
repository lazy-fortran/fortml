program fortml_bench_mlp_plateau_training
    !! Release benchmark app for metric-aware typed plateau training.
    use, intrinsic :: iso_fortran_env, only: int64
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_ok
    use fortml_mlp, only: mlp_t, MLP_LINEAR
    use fortml_mlp_schedules, only: make_mlp_schedule_plateau
    use fortml_mlp_training, only: mlp_train, mlp_training_options_t, &
        mlp_training_state_t, MLP_OPTIMIZER_SGD
    implicit none

    type(mlp_t) :: model
    type(mlp_training_options_t) :: options
    type(mlp_training_state_t) :: state
    type(fortnum_status_t) :: status
    real(dp) :: x(3, 1), target(3, 1), elapsed
    integer(int64) :: tick_start, tick_end, ticks_per_second

    x(:, 1) = [-1.0_dp, 0.0_dp, 1.0_dp]
    target(:, 1) = 2.0_dp*x(:, 1)
    call system_clock(count_rate=ticks_per_second)
    call model%initialize([1, 1], status, output_activation=MLP_LINEAR)
    if (.not. status_ok(status)) error stop 1
    call model%set_parameters([0.0_dp, 0.0_dp], status)
    if (.not. status_ok(status)) error stop 1
    options%max_epochs = 4
    options%optimizer = MLP_OPTIMIZER_SGD
    options%learning_rate = 1.0e-12_dp
    options%tolerance = 0.0_dp
    options%restore_best = .false.
    options%use_typed_schedule = .true.
    options%typed_schedule = make_mlp_schedule_plateau(2, 0.02_dp, 0.4_dp)

    call system_clock(tick_start)
    call mlp_train(model, x, target, status, options, state)
    call system_clock(tick_end)
    if (.not. status_ok(status)) error stop 1
    elapsed = real(tick_end-tick_start, dp)/real(ticks_per_second, dp)
    write (*, '(a,",pass,final_learning_rate,",es24.16,",",es24.16)') &
        "mlp_plateau_training", state%last_learning_rate, elapsed
    write (*, '(a,",pass,updates,",i0,",",es24.16)') &
        "mlp_plateau_training", state%updates, elapsed/real(max(1, state%updates), dp)
end program fortml_bench_mlp_plateau_training
