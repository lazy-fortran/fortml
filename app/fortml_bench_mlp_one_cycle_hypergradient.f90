program fortml_bench_mlp_one_cycle_hypergradient
    !! Complete-array release workload for one-cycle schedule products.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortnum_status, only: fortnum_status_t, status_ok
    use fortml_mlp, only: mlp_t, MLP_LINEAR
    use fortml_mlp_schedules, only: make_mlp_schedule_one_cycle
    use fortml_mlp_schedule_hypergradient, only: &
        mlp_schedule_hypergradient_objective_t, &
        mlp_schedule_hypergradient_options_t
    implicit none

    integer, parameter :: n_train = 5, n_validation = 3, repetitions = 16
    integer, parameter :: steps = 5
    real(dp), parameter :: base_rate = 0.12_dp, l2 = 0.07_dp
    real(dp) :: train_x(n_train, 1), train_target(n_train, 1)
    real(dp) :: validation_x(n_validation, 1), validation_target(n_validation, 1)
    real(dp) :: parameters(4), direction(4), gradient(4)
    real(dp) :: value, tangent, elapsed
    integer(int64) :: clock_start, clock_end, clock_rate
    integer :: oracle_unit, environment_status, repetition
    character(len=1024) :: oracle_path
    type(mlp_t), target :: model
    type(mlp_schedule_hypergradient_objective_t) :: objective
    type(mlp_schedule_hypergradient_options_t) :: options
    type(fortnum_status_t) :: status

    train_x(:, 1) = [-2.0_dp, -1.0_dp, 0.0_dp, 1.0_dp, 2.0_dp]
    train_target(:, 1) = 0.7_dp*train_x(:, 1) - 0.2_dp
    validation_x(:, 1) = [-1.5_dp, 0.5_dp, 1.75_dp]
    validation_target(:, 1) = 0.7_dp*validation_x(:, 1) - 0.2_dp
    options%steps = steps
    options%base_rate = base_rate
    options%l2 = l2
    options%schedule = make_mlp_schedule_one_cycle(2, 8, 1.8_dp, 0.08_dp)
    options%lower_logit_min_fraction = -1.0_dp
    options%upper_logit_min_fraction = 1.0_dp
    options%lower_logit_decay_factor = -4.0_dp
    options%upper_logit_decay_factor = -1.0_dp

    call model%initialize([1, 1], status, hidden_activation=MLP_LINEAR, &
        output_activation=MLP_LINEAR, initialization_seed=23)
    if (.not. status_ok(status)) error stop "one-cycle benchmark initialization failed"
    call model%set_parameters([0.15_dp, -0.1_dp], status)
    if (.not. status_ok(status)) error stop "one-cycle benchmark parameter setup failed"
    call objective%initialize(model, train_x, train_target, validation_x, &
        validation_target, options, status)
    if (.not. status_ok(status)) error stop "one-cycle benchmark setup failed"
    parameters = objective%parameters()
    call objective%value_gradient(parameters, value, gradient, status)
    if (.not. status_ok(status)) error stop "one-cycle benchmark product failed"
    direction = [0.17_dp, -0.13_dp, 0.21_dp, -0.19_dp]
    call objective%jvp(parameters, direction, value, tangent, status)
    if (.not. status_ok(status)) error stop "one-cycle benchmark JVP failed"

    oracle_unit = -1
    call get_environment_variable("FORTML_BENCH_MLP_ONE_CYCLE_HYPERGRADIENT_ORACLE", &
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
        if (.not. status_ok(status)) error stop "one-cycle benchmark timing failed"
    end do
    call system_clock(clock_end)
    elapsed = real(clock_end-clock_start, dp)/real(clock_rate, dp) &
        /real(repetitions, dp)
    write (*, '(a,es24.16)') "mlp_one_cycle_hypergradient_value_gradient,", elapsed

contains

    logical function oracle_only_requested()
        character(len=16) :: value
        integer :: environment_code

        call get_environment_variable("FORTML_BENCH_ORACLE_ONLY", value, &
            status=environment_code)
        oracle_only_requested = environment_code == 0 .and. trim(value) == "1"
    end function oracle_only_requested

end program fortml_bench_mlp_one_cycle_hypergradient
