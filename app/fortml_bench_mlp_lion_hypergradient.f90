program fortml_bench_mlp_lion_hypergradient
    !! Release workload for piecewise-smooth fixed-trajectory Lion products.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortml_mlp, only: mlp_t
    use fortml_mlp_lion_hypergradient, only: &
        mlp_lion_hypergradient_objective_t, mlp_lion_hypergradient_options_t
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer, parameter :: n_train = 6, n_validation = 3, repetitions = 16
    real(dp) :: train_x(n_train, 1), train_target(n_train, 1)
    real(dp) :: validation_x(n_validation, 1), validation_target(n_validation, 1)
    real(dp) :: parameters(4), direction(4), gradient(4)
    real(dp) :: value, tangent, elapsed
    integer(int64) :: clock_start, clock_end, clock_rate
    integer :: oracle_unit, environment_status, repetition
    character(len=1024) :: oracle_path
    type(mlp_t), target :: model
    type(mlp_lion_hypergradient_objective_t) :: objective
    type(mlp_lion_hypergradient_options_t) :: options
    type(fortnum_status_t) :: status

    train_x(:, 1) = [-2.0_dp, -1.0_dp, -0.3_dp, 0.4_dp, 1.2_dp, 2.1_dp]
    train_target(:, 1) = 0.8_dp*train_x(:, 1) - 0.15_dp
    validation_x(:, 1) = [-1.7_dp, 0.25_dp, 1.8_dp]
    validation_target(:, 1) = 0.8_dp*validation_x(:, 1) - 0.15_dp
    options%steps = 4
    options%learning_rate = 2.0e-4_dp
    options%l2 = 0.04_dp
    options%beta1 = 0.9_dp
    options%beta2 = 0.99_dp
    options%lower_log_learning_rate = -10.0_dp
    options%upper_log_learning_rate = 0.0_dp
    options%lower_log_l2 = -8.0_dp
    options%upper_log_l2 = 0.0_dp

    call model%initialize([1, 1], status, initialization_seed=19)
    if (.not. status_ok(status)) error stop "Lion benchmark initialization failed"
    call model%set_parameters([0.17_dp, -0.08_dp], status)
    if (.not. status_ok(status)) error stop "Lion benchmark parameter setup failed"
    call objective%initialize(model, train_x, train_target, validation_x, &
        validation_target, options, status)
    if (.not. status_ok(status)) error stop "Lion benchmark setup failed"
    parameters = objective%parameters()
    call objective%value_gradient(parameters, value, gradient, status)
    if (.not. status_ok(status)) error stop "Lion benchmark value/gradient failed"
    direction = [0.29_dp, -0.23_dp, 0.17_dp, -0.13_dp]
    call objective%jvp(parameters, direction, value, tangent, status)
    if (.not. status_ok(status)) error stop "Lion benchmark JVP failed"

    oracle_unit = -1
    call get_environment_variable("FORTML_BENCH_LION_HYPERGRADIENT_ORACLE", &
        oracle_path, status=environment_status)
    if (environment_status == 0 .and. len_trim(oracle_path) > 0) then
        open (newunit=oracle_unit, file=trim(oracle_path), status="replace", &
            action="write")
        write (oracle_unit, '(a)') "quantity,index,value"
        write (oracle_unit, '(a,es26.17e3)') "value,1,", value
        write (oracle_unit, '(a,es26.17e3)') "gradient,1,", gradient(1)
        write (oracle_unit, '(a,es26.17e3)') "gradient,2,", gradient(2)
        write (oracle_unit, '(a,es26.17e3)') "gradient,3,", gradient(3)
        write (oracle_unit, '(a,es26.17e3)') "gradient,4,", gradient(4)
        write (oracle_unit, '(a,es26.17e3)') "jvp,1,", tangent
        close (oracle_unit)
    end if
    if (oracle_only_requested()) stop

    call system_clock(clock_start, clock_rate)
    do repetition = 1, repetitions
        call objective%value_gradient(parameters, value, gradient, status)
        if (.not. status_ok(status)) error stop "Lion benchmark timing failed"
    end do
    call system_clock(clock_end)
    elapsed = real(clock_end-clock_start, dp)/real(clock_rate, dp) &
        /real(repetitions, dp)
    write (*, '(a,es24.16)') "mlp_lion_hypergradient_value_gradient,", elapsed

contains

    logical function oracle_only_requested()
        character(len=16) :: value
        integer :: environment_status

        call get_environment_variable("FORTML_BENCH_ORACLE_ONLY", value, &
            status=environment_status)
        oracle_only_requested = environment_status == 0 .and. trim(value) == "1"
    end function oracle_only_requested

end program fortml_bench_mlp_lion_hypergradient
