program fortml_bench_mlp_constant_schedule_hvp
    !! Release workload for the exact affine constant-schedule outer HVP.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortnum_status, only: fortnum_status_t, status_ok
    use fortml_mlp, only: mlp_t, MLP_LINEAR
    use fortml_mlp_schedules, only: make_mlp_schedule_constant
    use fortml_mlp_schedule_hypergradient, only: &
        mlp_schedule_hypergradient_objective_t, &
        mlp_schedule_hypergradient_options_t
    implicit none

    integer, parameter :: train_count = 96, validation_count = 32
    integer, parameter :: repetitions = 64
    integer, parameter :: n_parameters = 4
    real(dp), parameter :: base_rate = 0.08_dp, l2 = 0.03_dp
    real(dp) :: train_x(train_count, 1), train_target(train_count, 1)
    real(dp) :: validation_x(validation_count, 1), validation_target(validation_count, 1)
    real(dp) :: parameters(n_parameters), direction(n_parameters)
    real(dp) :: gradient(n_parameters), hvp(n_parameters)
    real(dp) :: value, tangent, elapsed_value, elapsed_hvp
    integer(int64) :: clock_start, clock_end, clock_rate
    integer :: i, repetition, oracle_unit, environment_status
    character(len=1024) :: oracle_path
    type(mlp_t), target :: model
    type(mlp_schedule_hypergradient_objective_t) :: objective
    type(mlp_schedule_hypergradient_options_t) :: options
    type(fortnum_status_t) :: status

    do i = 1, train_count
        train_x(i, 1) = -2.0_dp + 4.0_dp*real(i-1, dp)/real(train_count-1, dp)
        train_target(i, 1) = 0.7_dp*train_x(i, 1) - 0.2_dp
    end do
    do i = 1, validation_count
        validation_x(i, 1) = -1.8_dp + 3.6_dp*real(i-1, dp)/real(validation_count-1, dp)
        validation_target(i, 1) = 0.7_dp*validation_x(i, 1) - 0.2_dp
    end do
    call model%initialize([1, 1], status, initialization_seed=23, &
        hidden_activation=MLP_LINEAR, output_activation=MLP_LINEAR)
    if (.not. status_ok(status)) error stop "constant schedule benchmark model failed"
    call model%set_parameters([0.15_dp, -0.1_dp], status)
    options%steps = 8
    options%base_rate = base_rate
    options%l2 = l2
    options%schedule = make_mlp_schedule_constant()
    call objective%initialize(model, train_x, train_target, validation_x, &
        validation_target, options, status)
    if (.not. status_ok(status)) error stop "constant schedule benchmark setup failed"
    parameters = objective%parameters()
    direction = [0.31_dp, -0.27_dp, 0.18_dp, -0.22_dp]
    call objective%value_gradient(parameters, value, gradient, status)
    if (.not. status_ok(status)) error stop "constant schedule benchmark gradient failed"
    call objective%jvp(parameters, direction, value, tangent, status)
    if (.not. status_ok(status)) error stop "constant schedule benchmark JVP failed"
    call objective%hvp(parameters, direction, hvp, status)
    if (.not. status_ok(status)) error stop "constant schedule benchmark HVP failed"

    oracle_unit = -1
    call get_environment_variable("FORTML_BENCH_MLP_CONSTANT_SCHEDULE_HVP_ORACLE", &
        oracle_path, status=environment_status)
    if (environment_status == 0 .and. len_trim(oracle_path) > 0) then
        open (newunit=oracle_unit, file=trim(oracle_path), status="replace", &
            action="write")
        write (oracle_unit, '(a)') "quantity,index,value"
        write (oracle_unit, '(a,es26.17e3)') "value,1,", value
        do i = 1, n_parameters
            write (oracle_unit, '(a,i0,a,es26.17e3)') "gradient,", i, ",", gradient(i)
            write (oracle_unit, '(a,i0,a,es26.17e3)') "hvp,", i, ",", hvp(i)
        end do
        write (oracle_unit, '(a,es26.17e3)') "jvp,1,", tangent
        close (oracle_unit)
    end if
    if (oracle_only_requested()) stop

    call system_clock(clock_start, clock_rate)
    do repetition = 1, repetitions
        call objective%value_gradient(parameters, value, gradient, status)
    end do
    call system_clock(clock_end)
    elapsed_value = real(clock_end-clock_start, dp)/real(clock_rate, dp) &
        /real(repetitions, dp)
    call system_clock(clock_start, clock_rate)
    do repetition = 1, repetitions
        call objective%hvp(parameters, direction, hvp, status)
    end do
    call system_clock(clock_end)
    elapsed_hvp = real(clock_end-clock_start, dp)/real(clock_rate, dp) &
        /real(repetitions, dp)
    write (*, '(a,es24.16)') "mlp_constant_schedule_hvp_value_gradient,", elapsed_value
    write (*, '(a,es24.16)') "mlp_constant_schedule_hvp_hvp,", elapsed_hvp

contains

    logical function oracle_only_requested()
        character(len=16) :: value_text
        integer :: environment_code

        call get_environment_variable("FORTML_BENCH_ORACLE_ONLY", value_text, &
            status=environment_code)
        oracle_only_requested = environment_code == 0 .and. trim(value_text) == "1"
    end function oracle_only_requested

end program fortml_bench_mlp_constant_schedule_hvp
