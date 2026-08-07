program fortml_bench_rmsprop_training
    !! Release workload for the canonical FortOpt RMSprop recurrence and the
    !! MLP training integration.  The Python lane independently reconstructs
    !! both trajectories before retaining the timings below.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortopt_rmsprop, only: rmsprop_t
    use fortml_mlp, only: mlp_t, MLP_LINEAR
    use fortml_mlp_training, only: mlp_training_options_t, mlp_training_state_t, &
        mlp_train, MLP_OPTIMIZER_RMSPROP
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer, parameter :: n_parameters = 4096, steps = 128, repetitions = 16
    integer, parameter :: n_samples = 3, n_epochs = 32, mlp_repetitions = 8
    real(dp) :: parameter(n_parameters), target(n_parameters), gradient(n_parameters)
    real(dp) :: elapsed, final_norm
    real(dp) :: x(n_samples, 1), mlp_target(n_samples, 1), mlp_theta(2)
    real(dp) :: mlp_elapsed, mlp_loss
    integer(int64) :: clock_start, clock_end, clock_rate
    integer :: index, step, repetition
    type(rmsprop_t) :: optimizer
    type(mlp_t) :: model
    type(mlp_training_options_t) :: options
    type(mlp_training_state_t) :: state
    type(fortnum_status_t) :: status

    do index = 1, n_parameters
        parameter(index) = 0.1_dp*cos(0.003_dp*real(index, dp))
        target(index) = 0.25_dp*sin(0.0017_dp*real(index, dp))
    end do
    call optimizer%initialize(n_parameters, status, learning_rate=1.0e-2_dp, &
        decay=0.9_dp, epsilon=1.0e-8_dp, momentum=0.0_dp, centered=.false.)
    if (.not. status_ok(status)) error stop "RMSprop benchmark initialization failed"
    do step = 1, steps
        gradient = parameter - target
        call optimizer%step(parameter, gradient, status)
        if (.not. status_ok(status)) error stop "RMSprop benchmark reference failed"
    end do
    final_norm = sqrt(sum(parameter*parameter))

    call system_clock(clock_start, clock_rate)
    do repetition = 1, repetitions
        do index = 1, n_parameters
            parameter(index) = 0.1_dp*cos(0.003_dp*real(index, dp))
            target(index) = 0.25_dp*sin(0.0017_dp*real(index, dp))
        end do
        call optimizer%initialize(n_parameters, status, learning_rate=1.0e-2_dp, &
            decay=0.9_dp, epsilon=1.0e-8_dp, momentum=0.0_dp, centered=.false.)
        if (.not. status_ok(status)) error stop "RMSprop benchmark timing initialization failed"
        do step = 1, steps
            gradient = parameter - target
            call optimizer%step(parameter, gradient, status)
            if (.not. status_ok(status)) error stop "RMSprop benchmark timing failed"
        end do
    end do
    call system_clock(clock_end)
    elapsed = real(clock_end - clock_start, dp)/real(clock_rate, dp) &
        /real(repetitions, dp)
    write (*, '(a,i0,a,i0,a,es24.16,a,es24.16)') &
        "rmsprop_training,", n_parameters, ",", steps, ",", final_norm, ",", elapsed

    x(:, 1) = [-1.0_dp, 0.0_dp, 1.0_dp]
    mlp_target(:, 1) = x(:, 1)
    options%optimizer = MLP_OPTIMIZER_RMSPROP
    options%max_epochs = n_epochs
    options%learning_rate = 0.08_dp
    options%rmsprop_decay = 0.8_dp
    options%rmsprop_momentum = 0.2_dp
    options%rmsprop_centered = .true.
    options%epsilon = 1.0e-5_dp
    options%tolerance = 0.0_dp
    options%restore_best = .false.
    call model%initialize([1, 1], status, output_activation=MLP_LINEAR)
    call model%set_parameters([0.0_dp, 0.0_dp], status)
    call mlp_train(model, x, mlp_target, status, options, state)
    if (.not. status_ok(status)) error stop "MLP RMSprop benchmark reference failed"
    mlp_theta = model%parameters()
    mlp_loss = state%final_loss

    call system_clock(clock_start, clock_rate)
    do repetition = 1, mlp_repetitions
        call model%set_parameters([0.0_dp, 0.0_dp], status)
        call mlp_train(model, x, mlp_target, status, options, state)
        if (.not. status_ok(status)) error stop "MLP RMSprop benchmark timing failed"
    end do
    call system_clock(clock_end)
    mlp_elapsed = real(clock_end - clock_start, dp)/real(clock_rate, dp) &
        /real(mlp_repetitions, dp)
    write (*, '(a,i0,a,i0,a,es24.16,a,es24.16,a,es24.16)') &
        "rmsprop_mlp,", n_samples, ",", n_epochs, ",", mlp_loss, ",", &
        mlp_elapsed, ",", sqrt(sum(mlp_theta*mlp_theta))
end program fortml_bench_rmsprop_training
