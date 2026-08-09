program fortml_bench_rmsprop_weighted_hypergradient
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortnum_status, only: fortnum_status_t, status_ok
    use fortml_device, only: FORTML_DEVICE_CUDA
    use fortml_mlp, only: mlp_t
    use fortml_mlp_hypergradient, only: &
        mlp_rmsprop_hypergradient_objective_t, &
        mlp_rmsprop_hypergradient_options_t, mlp_rmsprop_hypergradient_metadata_t
    implicit none

    integer, parameter :: repetitions = 32
    real(dp) :: train_x(5, 1), train_target(5, 1), train_weight(5)
    real(dp) :: validation_x(3, 1), validation_target(3, 1), validation_weight(3)
    real(dp) :: parameters(5), direction(5), gradient(5), product(5)
    real(dp) :: value, tangent, gradient_seconds, hvp_seconds
    integer(int64) :: tick_start, tick_end, tick_rate
    integer :: i, repetition, cuda_status
    type(mlp_t), target :: model
    type(mlp_rmsprop_hypergradient_objective_t) :: adapter
    type(mlp_rmsprop_hypergradient_options_t) :: options
    type(mlp_rmsprop_hypergradient_metadata_t) :: metadata
    type(fortnum_status_t) :: status

    train_x(:, 1) = [-2.0_dp, -1.0_dp, 0.0_dp, 1.0_dp, 2.0_dp]
    train_target(:, 1) = 0.7_dp*train_x(:, 1) - 0.2_dp
    train_weight = [0.25_dp, 1.5_dp, 0.0_dp, 2.0_dp, 0.75_dp]
    validation_x(:, 1) = [-1.5_dp, 0.5_dp, 1.75_dp]
    validation_target(:, 1) = 0.7_dp*validation_x(:, 1) - 0.2_dp
    validation_weight = [2.0_dp, 0.5_dp, 1.25_dp]
    options%steps = 4
    options%learning_rate = 0.12_dp
    options%l2 = 0.07_dp
    options%rmsprop_decay = 0.78_dp
    options%epsilon = 0.03_dp
    options%momentum = 0.21_dp
    options%centered = .true.
    direction = [0.31_dp, -0.27_dp, 0.17_dp, -0.13_dp, 0.19_dp]

    call model%initialize([1, 1], status, initialization_seed=23)
    if (.not. status_ok(status)) error stop "weighted RMSprop model init failed"
    call model%set_parameters([0.15_dp, -0.1_dp], status)
    if (.not. status_ok(status)) error stop "weighted RMSprop parameter setup failed"
    call adapter%initialize(model, train_x, train_target, validation_x, &
        validation_target, options, status, train_weight, validation_weight)
    if (.not. status_ok(status)) error stop "weighted RMSprop adapter init failed"
    parameters = adapter%parameters()
    metadata = adapter%metadata()
    call adapter%value_gradient(parameters, value, gradient, status)
    if (.not. status_ok(status)) error stop "weighted RMSprop gradient failed"
    call adapter%jvp(parameters, direction, value, tangent, status)
    if (.not. status_ok(status)) error stop "weighted RMSprop JVP failed"
    call adapter%hvp(parameters, direction, product, status)
    if (.not. status_ok(status)) error stop "weighted RMSprop HVP failed"

    call system_clock(tick_start, tick_rate)
    do repetition = 1, repetitions
        call adapter%value_gradient(parameters, value, gradient, status)
        if (.not. status_ok(status)) error stop "weighted RMSprop timing failed"
    end do
    call system_clock(tick_end)
    gradient_seconds = real(tick_end - tick_start, dp)/real(tick_rate, dp) &
        /real(repetitions, dp)
    call system_clock(tick_start)
    do repetition = 1, repetitions
        call adapter%hvp(parameters, direction, product, status)
        if (.not. status_ok(status)) error stop "weighted RMSprop HVP timing failed"
    end do
    call system_clock(tick_end)
    hvp_seconds = real(tick_end - tick_start, dp)/real(tick_rate, dp) &
        /real(repetitions, dp)

    options%device_kind = FORTML_DEVICE_CUDA
    call adapter%initialize(model, train_x, train_target, validation_x, &
        validation_target, options, status, train_weight, validation_weight)
    cuda_status = status%code

    write (*, '(a,i0)') "rmsprop_weighted_repetitions,", repetitions
    write (*, '(a,es24.16)') "rmsprop_weighted_value,", value
    do i = 1, size(gradient)
        write (*, '(a,i0,a,es24.16)') "rmsprop_weighted_gradient_", i, ",", gradient(i)
        write (*, '(a,i0,a,es24.16)') "rmsprop_weighted_hvp_", i, ",", product(i)
    end do
    write (*, '(a,es24.16)') "rmsprop_weighted_jvp,", tangent
    write (*, '(a,es24.16)') "rmsprop_weighted_train_mass,", &
        metadata%training_weight_mass
    write (*, '(a,es24.16)') "rmsprop_weighted_validation_mass,", &
        metadata%validation_weight_mass
    write (*, '(a,es24.16)') "rmsprop_weighted_gradient_seconds,", gradient_seconds
    write (*, '(a,es24.16)') "rmsprop_weighted_hvp_seconds,", hvp_seconds
    write (*, '(a,i0)') "rmsprop_weighted_cuda_status,", cuda_status
end program fortml_bench_rmsprop_weighted_hypergradient
