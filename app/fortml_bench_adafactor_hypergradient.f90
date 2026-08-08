program fortml_bench_adafactor_hypergradient
    !! Release workload for the five-parameter unfactored Adafactor objective.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortml_mlp, only: mlp_t
    use fortml_mlp_adafactor_hypergradient, only: &
        mlp_adafactor_hypergradient_objective_t, mlp_adafactor_hypergradient_options_t
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer, parameter :: n_train = 5, n_validation = 3, repetitions = 32
    integer, parameter :: steps = 4, parameter_count = 5
    real(dp), parameter :: learning_rate = 0.12_dp, l2 = 0.07_dp
    real(dp), parameter :: decay = 0.75_dp, epsilon = 0.03_dp
    real(dp), parameter :: clip_threshold = 0.2_dp
    real(dp) :: train_x(n_train, 1), train_target(n_train, 1)
    real(dp) :: validation_x(n_validation, 1), validation_target(n_validation, 1)
    real(dp) :: parameters(parameter_count), direction(parameter_count)
    real(dp) :: gradient(parameter_count), value, tangent, elapsed
    integer(int64) :: clock_start, clock_end, clock_rate
    integer :: oracle_unit, environment_status, repetition, index
    character(len=1024) :: oracle_path
    type(mlp_t), target :: model
    type(mlp_adafactor_hypergradient_objective_t) :: objective
    type(mlp_adafactor_hypergradient_options_t) :: options
    type(fortnum_status_t) :: status

    train_x(:, 1) = [-2.0_dp, -1.0_dp, 0.0_dp, 1.0_dp, 2.0_dp]
    train_target(:, 1) = 0.7_dp*train_x(:, 1) - 0.2_dp
    validation_x(:, 1) = [-1.5_dp, 0.5_dp, 1.75_dp]
    validation_target(:, 1) = 0.7_dp*validation_x(:, 1) - 0.2_dp
    options%steps = steps
    options%learning_rate = learning_rate
    options%l2 = l2
    options%decay = decay
    options%epsilon = epsilon
    options%clip_threshold = clip_threshold
    options%lower_log_learning_rate = -4.0_dp
    options%upper_log_learning_rate = 0.0_dp
    options%lower_log_l2 = -5.0_dp
    options%upper_log_l2 = 0.0_dp
    options%lower_decay = 0.1_dp
    options%upper_decay = 0.95_dp
    options%lower_log_epsilon = -5.0_dp
    options%upper_log_epsilon = 0.0_dp
    options%lower_log_clip_threshold = -5.0_dp
    options%upper_log_clip_threshold = 0.0_dp

    call model%initialize([1, 1], status, initialization_seed=23)
    if (.not. status_ok(status)) error stop "Adafactor hypergradient benchmark initialization failed"
    call model%set_parameters([0.15_dp, -0.1_dp], status)
    if (.not. status_ok(status)) error stop "Adafactor hypergradient benchmark parameter setup failed"
    call objective%initialize(model, train_x, train_target, validation_x, &
        validation_target, options, status)
    if (.not. status_ok(status)) error stop "Adafactor hypergradient benchmark setup failed"
    parameters = objective%parameters()
    call objective%value_gradient(parameters, value, gradient, status)
    if (.not. status_ok(status)) error stop "Adafactor hypergradient value/gradient failed"
    direction = [0.31_dp, -0.27_dp, 0.19_dp, 0.13_dp, -0.22_dp]
    call objective%jvp(parameters, direction, value, tangent, status)
    if (.not. status_ok(status)) error stop "Adafactor hypergradient JVP failed"

    oracle_unit = -1
    call get_environment_variable("FORTML_BENCH_ADAFACTOR_HYPERGRADIENT_ORACLE", oracle_path, &
        status=environment_status)
    if (environment_status == 0 .and. len_trim(oracle_path) > 0) then
        open (newunit=oracle_unit, file=trim(oracle_path), status="replace", action="write")
        write (oracle_unit, '(a)') "quantity,index,value"
        write (oracle_unit, '(a,es26.17e3)') "value,1,", value
        do index = 1, parameter_count
            write (oracle_unit, '(a,i0,a,es26.17e3)') "gradient,", index, ",", gradient(index)
        end do
        write (oracle_unit, '(a,es26.17e3)') "jvp,1,", tangent
        close (oracle_unit)
    end if
    if (oracle_only_requested()) stop

    call system_clock(clock_start, clock_rate)
    do repetition = 1, repetitions
        call objective%value_gradient(parameters, value, gradient, status)
        if (.not. status_ok(status)) error stop "Adafactor hypergradient timing failed"
    end do
    call system_clock(clock_end)
    elapsed = real(clock_end-clock_start, dp)/real(clock_rate, dp)/real(repetitions, dp)
    write (*, '(a,es24.16)') "mlp_adafactor_hypergradient_value_gradient,", elapsed

contains

    logical function oracle_only_requested()
        character(len=16) :: value
        integer :: environment_code

        call get_environment_variable("FORTML_BENCH_ORACLE_ONLY", value, status=environment_code)
        oracle_only_requested = environment_code == 0 .and. trim(value) == "1"
    end function oracle_only_requested

end program fortml_bench_adafactor_hypergradient
