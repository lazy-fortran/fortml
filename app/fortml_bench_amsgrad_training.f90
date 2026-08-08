program fortml_bench_amsgrad_training
    !! Release workload for AMSGrad state and MLP trainer integration.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortml_amsgrad, only: amsgrad_t
    use fortml_mlp, only: mlp_t, MLP_LINEAR
    use fortml_mlp_training, only: mlp_training_options_t, mlp_training_state_t, &
        mlp_train, MLP_OPTIMIZER_AMSGRAD
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
    type(amsgrad_t) :: optimizer
    type(mlp_t) :: model
    type(mlp_training_options_t) :: options
    type(mlp_training_state_t) :: state
    type(fortnum_status_t) :: status

    do index = 1, n_parameters
        parameter(index) = 0.1_dp*cos(0.003_dp*real(index, dp))
        target(index) = 0.25_dp*sin(0.0017_dp*real(index, dp))
    end do
    call optimizer%initialize(n_parameters, status, learning_rate=1.0e-2_dp, &
        beta1=0.9_dp, beta2=0.99_dp, epsilon=1.0e-8_dp)
    if (.not. status_ok(status)) error stop "AMSGrad benchmark initialization failed"
    do step = 1, steps
        gradient = parameter - target
        call optimizer%step(parameter, gradient, status)
        if (.not. status_ok(status)) error stop "AMSGrad benchmark reference failed"
    end do
    final_norm = sqrt(sum(parameter*parameter))

    call system_clock(clock_start, clock_rate)
    do repetition = 1, repetitions
        do index = 1, n_parameters
            parameter(index) = 0.1_dp*cos(0.003_dp*real(index, dp))
            target(index) = 0.25_dp*sin(0.0017_dp*real(index, dp))
        end do
        call optimizer%initialize(n_parameters, status, learning_rate=1.0e-2_dp, &
            beta1=0.9_dp, beta2=0.99_dp, epsilon=1.0e-8_dp)
        if (.not. status_ok(status)) error stop "AMSGrad benchmark timing initialization failed"
        do step = 1, steps
            gradient = parameter - target
            call optimizer%step(parameter, gradient, status)
            if (.not. status_ok(status)) error stop "AMSGrad benchmark timing failed"
        end do
    end do
    call system_clock(clock_end)
    elapsed = real(clock_end - clock_start, dp)/real(clock_rate, dp) &
        /real(repetitions, dp)
    write (*, '(a,i0,a,i0,a,es24.16,a,es24.16)') &
        "amsgrad_training,", n_parameters, ",", steps, ",", final_norm, ",", elapsed

    x(:, 1) = [-1.0_dp, 0.0_dp, 1.0_dp]
    mlp_target(:, 1) = x(:, 1)
    options%optimizer = MLP_OPTIMIZER_AMSGRAD
    options%max_epochs = n_epochs
    options%learning_rate = 0.08_dp
    options%beta1 = 0.85_dp
    options%beta2 = 0.95_dp
    options%epsilon = 1.0e-5_dp
    options%tolerance = 0.0_dp
    options%restore_best = .false.
    call model%initialize([1, 1], status, output_activation=MLP_LINEAR)
    call model%set_parameters([0.0_dp, 0.0_dp], status)
    call mlp_train(model, x, mlp_target, status, options, state)
    if (.not. status_ok(status)) error stop "MLP AMSGrad benchmark reference failed"
    mlp_theta = model%parameters()
    mlp_loss = state%final_loss

    call system_clock(clock_start, clock_rate)
    do repetition = 1, mlp_repetitions
        call model%set_parameters([0.0_dp, 0.0_dp], status)
        call mlp_train(model, x, mlp_target, status, options, state)
        if (.not. status_ok(status)) error stop "MLP AMSGrad benchmark timing failed"
    end do
    call system_clock(clock_end)
    mlp_elapsed = real(clock_end - clock_start, dp)/real(clock_rate, dp) &
        /real(mlp_repetitions, dp)
    write (*, '(a,i0,a,i0,a,es24.16,a,es24.16,a,es24.16)') &
        "amsgrad_mlp,", n_samples, ",", n_epochs, ",", mlp_loss, ",", &
        mlp_elapsed, ",", sqrt(sum(mlp_theta*mlp_theta))
end program fortml_bench_amsgrad_training
