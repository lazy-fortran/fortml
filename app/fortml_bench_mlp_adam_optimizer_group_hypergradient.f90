program fortml_bench_mlp_adam_optimizer_group_hypergradient
    !! Release workload for grouped coupled-L2 Adam trajectory products.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortml_mlp, only: mlp_t, MLP_LINEAR
    use fortml_mlp_training, only: mlp_optimizer_group_t
    use fortml_mlp_optimizer_group_hypergradient, only: &
        mlp_optimizer_group_hypergradient_objective_t, &
        mlp_optimizer_group_hypergradient_options_t, MLP_OPTIMIZER_GROUP_ADAM
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer, parameter :: n_train = 4, n_validation = 3, repetitions = 32
    real(dp) :: train_x(n_train, 1), train_target(n_train, 1)
    real(dp) :: validation_x(n_validation, 1), validation_target(n_validation, 1)
    real(dp) :: parameters(6), direction(6), gradient(6)
    real(dp) :: value, tangent, elapsed
    integer(int64) :: clock_start, clock_end, clock_rate
    integer :: oracle_unit, environment_status, repetition
    character(len=1024) :: oracle_path
    type(mlp_t), target :: model
    type(mlp_optimizer_group_t) :: weight_group, bias_group
    type(mlp_optimizer_group_hypergradient_objective_t) :: objective
    type(mlp_optimizer_group_hypergradient_options_t) :: options
    type(fortnum_status_t) :: status

    train_x(:, 1) = [-1.0_dp, -0.2_dp, 0.7_dp, 1.5_dp]
    train_target(:, 1) = 0.6_dp*train_x(:, 1) - 0.15_dp
    validation_x(:, 1) = [-0.8_dp, 0.4_dp, 1.2_dp]
    validation_target(:, 1) = 0.6_dp*validation_x(:, 1) - 0.15_dp
    options%steps = 4
    options%learning_rate = 0.08_dp
    options%l2 = 0.03_dp
    options%beta1 = 0.85_dp
    options%beta2 = 0.97_dp
    options%epsilon = 1.0e-7_dp
    options%optimize_moment_parameters = .true.
    options%optimizer = MLP_OPTIMIZER_GROUP_ADAM
    options%lower_log_learning_rate = -6.0_dp
    options%upper_log_learning_rate = 0.0_dp
    options%lower_log_l2 = -7.0_dp
    options%upper_log_l2 = 0.0_dp
    options%lower_log_multiplier = -4.0_dp
    options%upper_log_multiplier = 4.0_dp
    call weight_group%initialize("weight", 1, 1, 0.7_dp, status)
    if (.not. status_ok(status)) error stop "grouped Adam benchmark weight setup failed"
    call bias_group%initialize("bias", 2, 2, 1.3_dp, status)
    if (.not. status_ok(status)) error stop "grouped Adam benchmark bias setup failed"
    allocate(options%groups(2))
    options%groups = [weight_group, bias_group]

    call model%initialize([1, 1], status, output_activation=MLP_LINEAR)
    if (.not. status_ok(status)) error stop "grouped Adam benchmark model initialization failed"
    call model%set_parameters([0.25_dp, 0.1_dp], status)
    if (.not. status_ok(status)) error stop "grouped Adam benchmark parameter setup failed"
    call objective%initialize(model, train_x, train_target, validation_x, &
        validation_target, options, status)
    if (.not. status_ok(status)) error stop "grouped Adam benchmark setup failed"
    parameters = objective%parameters()
    call objective%value_gradient(parameters, value, gradient, status)
    if (.not. status_ok(status)) error stop "grouped Adam benchmark product failed"
    direction = [0.17_dp, -0.13_dp, 0.07_dp, -0.05_dp, 0.11_dp, -0.09_dp]
    call objective%jvp(parameters, direction, value, tangent, status)
    if (.not. status_ok(status)) error stop "grouped Adam benchmark JVP failed"

    oracle_unit = -1
    call get_environment_variable("FORTML_BENCH_ADAM_GROUP_HYPERGRADIENT_ORACLE", &
        oracle_path, status=environment_status)
    if (environment_status == 0 .and. len_trim(oracle_path) > 0) then
        open (newunit=oracle_unit, file=trim(oracle_path), status="replace", action="write")
        write (oracle_unit, '(a)') "quantity,index,value"
        write (oracle_unit, '(a,es26.17e3)') "value,1,", value
        write (oracle_unit, '(a,es26.17e3)') "jvp,1,", tangent
        write (oracle_unit, '(a,es26.17e3)') "gradient,1,", gradient(1)
        write (oracle_unit, '(a,es26.17e3)') "gradient,2,", gradient(2)
        write (oracle_unit, '(a,es26.17e3)') "gradient,3,", gradient(3)
        write (oracle_unit, '(a,es26.17e3)') "gradient,4,", gradient(4)
        write (oracle_unit, '(a,es26.17e3)') "gradient,5,", gradient(5)
        write (oracle_unit, '(a,es26.17e3)') "gradient,6,", gradient(6)
        close (oracle_unit)
    end if
    if (oracle_only_requested()) stop

    call system_clock(clock_start, clock_rate)
    do repetition = 1, repetitions
        call objective%value_gradient(parameters, value, gradient, status)
        if (.not. status_ok(status)) error stop "grouped Adam benchmark timing failed"
    end do
    call system_clock(clock_end)
    elapsed = real(clock_end-clock_start, dp)/real(clock_rate, dp)/real(repetitions, dp)
    write (*, '(a,es24.16)') "mlp_adam_optimizer_group_hypergradient_value_gradient,", elapsed

contains

    logical function oracle_only_requested()
        character(len=16) :: value
        integer :: environment_status

        call get_environment_variable("FORTML_BENCH_ORACLE_ONLY", value, status=environment_status)
        oracle_only_requested = environment_status == 0 .and. trim(value) == "1"
    end function oracle_only_requested

end program fortml_bench_mlp_adam_optimizer_group_hypergradient
