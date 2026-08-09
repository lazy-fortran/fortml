program fortml_bench_mlp_clip_hypergradient
    !! Release workload for exact global-clip-threshold trajectory products.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    use fortml_mlp, only: mlp_t, MLP_LINEAR
    use fortml_mlp_clip_hypergradient, only: &
        mlp_clip_hypergradient_objective_t, mlp_clip_hypergradient_options_t
    implicit none

    integer, parameter :: repetitions = 32
    type(mlp_t), target :: model
    type(mlp_clip_hypergradient_objective_t) :: objective
    type(mlp_clip_hypergradient_options_t) :: options
    type(fortnum_status_t) :: status
    real(dp) :: train_x(5, 1), train_target(5, 1)
    real(dp) :: validation_x(3, 1), validation_target(3, 1)
    real(dp) :: parameters(3), direction(3), gradient(3), product(3)
    real(dp) :: value, tangent, elapsed
    integer(int64) :: clock_start, clock_end, clock_rate
    integer :: repetition, oracle_unit, environment_status
    character(len=1024) :: oracle_path

    train_x(:, 1) = [-2.0_dp, -1.0_dp, 0.0_dp, 1.0_dp, 2.0_dp]
    train_target(:, 1) = 0.7_dp*train_x(:, 1) - 0.2_dp
    validation_x(:, 1) = [-1.5_dp, 0.5_dp, 1.75_dp]
    validation_target(:, 1) = 0.7_dp*validation_x(:, 1) - 0.2_dp
    options%steps = 4
    options%learning_rate = 0.12_dp
    options%l2 = 0.07_dp
    options%gradient_clip_norm = 0.30_dp
    options%lower_log_learning_rate = log(0.03_dp)
    options%upper_log_learning_rate = log(0.25_dp)
    options%lower_log_l2 = log(0.01_dp)
    options%upper_log_l2 = log(0.20_dp)
    options%lower_log_clip_norm = log(0.05_dp)
    options%upper_log_clip_norm = log(0.40_dp)

    call model%initialize([1, 1], status, hidden_activation=MLP_LINEAR, &
        output_activation=MLP_LINEAR)
    if (.not. status_ok(status)) error stop "clip benchmark model initialization failed"
    call model%set_parameters([0.15_dp, -0.1_dp], status)
    if (.not. status_ok(status)) error stop "clip benchmark parameter setup failed"
    call objective%initialize(model, train_x, train_target, validation_x, &
        validation_target, options, status)
    if (.not. status_ok(status)) error stop "clip benchmark objective setup failed"
    parameters = objective%parameters()
    call objective%value_gradient(parameters, value, gradient, status)
    if (.not. status_ok(status)) error stop "clip benchmark gradient failed"
    direction = [0.23_dp, -0.17_dp, 0.31_dp]
    call objective%jvp(parameters, direction, value, tangent, status)
    if (.not. status_ok(status)) error stop "clip benchmark JVP failed"
    product = 1.0_dp
    call objective%hvp(parameters, direction, product, status)
    if (status%code /= FORTNUM_NOT_IMPLEMENTED .or. any(product /= 0.0_dp)) then
        error stop "clip benchmark HVP refusal failed"
    end if

    oracle_unit = -1
    call get_environment_variable("FORTML_BENCH_CLIP_HYPERGRADIENT_ORACLE", &
        oracle_path, status=environment_status)
    if (environment_status == 0 .and. len_trim(oracle_path) > 0) then
        open (newunit=oracle_unit, file=trim(oracle_path), status="replace", &
            action="write")
        write (oracle_unit, '(a)') "quantity,index,value"
        write (oracle_unit, '(a,es26.17e3)') "value,1,", value
        write (oracle_unit, '(a,es26.17e3)') "gradient,1,", gradient(1)
        write (oracle_unit, '(a,es26.17e3)') "gradient,2,", gradient(2)
        write (oracle_unit, '(a,es26.17e3)') "gradient,3,", gradient(3)
        write (oracle_unit, '(a,es26.17e3)') "jvp,1,", tangent
        write (oracle_unit, '(a,i0)') "hvp_status,1,", status%code
        close (oracle_unit)
    end if
    if (oracle_only_requested()) stop

    call system_clock(clock_start, clock_rate)
    do repetition = 1, repetitions
        call objective%value_gradient(parameters, value, gradient, status)
        if (.not. status_ok(status)) error stop "clip benchmark timing failed"
    end do
    call system_clock(clock_end)
    elapsed = real(clock_end-clock_start, dp)/real(clock_rate, dp)/ &
        real(repetitions, dp)
    write (*, '(a,es24.16)') "mlp_clip_hypergradient_value_gradient,", elapsed

contains

    logical function oracle_only_requested()
        character(len=16) :: value
        integer :: local_status

        call get_environment_variable("FORTML_BENCH_ORACLE_ONLY", value, &
            status=local_status)
        oracle_only_requested = local_status == 0 .and. trim(value) == "1"
    end function oracle_only_requested

end program fortml_bench_mlp_clip_hypergradient
