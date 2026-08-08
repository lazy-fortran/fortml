program fortml_bench_mlp_weighted_validation_hypergradient
    !! Release workload for weighted validation in the SGD trajectory adapter.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortml_mlp, only: mlp_t, MLP_LINEAR
    use fortml_mlp_sgd_momentum_hypergradient, only: &
        mlp_sgd_momentum_hypergradient_objective_t, &
        mlp_sgd_momentum_hypergradient_options_t
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    implicit none

    integer, parameter :: n_train = 5, n_validation = 3, repetitions = 16
    real(dp) :: train_x(n_train, 1), train_target(n_train, 1)
    real(dp) :: validation_x(n_validation, 1), validation_target(n_validation, 1)
    real(dp) :: validation_weight(n_validation)
    real(dp) :: parameters(3), direction(3), gradient(3), hvp_product(3)
    real(dp) :: value, tangent, elapsed
    real(dp) :: weighted_value, weighted_tangent, weighted_gradient(3)
    integer :: oracle_unit, environment_status, repetition, nonuniform_hvp_status
    integer(int64) :: clock_start, clock_end, clock_rate
    character(len=1024) :: oracle_path
    type(mlp_t), target :: model
    type(mlp_sgd_momentum_hypergradient_objective_t) :: objective
    type(mlp_sgd_momentum_hypergradient_options_t) :: options
    type(fortnum_status_t) :: status

    train_x(:, 1) = [-2.0_dp, -1.0_dp, 0.0_dp, 1.0_dp, 2.0_dp]
    train_target(:, 1) = 0.65_dp*train_x(:, 1) - 0.1_dp
    validation_x(:, 1) = [-1.5_dp, 0.5_dp, 1.75_dp]
    validation_target(:, 1) = 0.65_dp*validation_x(:, 1) - 0.1_dp
    validation_weight = [1.0_dp, 2.0_dp, 4.0_dp]
    options%steps = 4
    options%learning_rate = 0.11_dp
    options%l2 = 0.06_dp
    options%momentum = 0.29_dp
    options%lower_log_learning_rate = -4.0_dp
    options%upper_log_learning_rate = 0.0_dp
    options%lower_log_l2 = -5.0_dp
    options%upper_log_l2 = 0.0_dp
    options%lower_momentum = 0.05_dp
    options%upper_momentum = 0.8_dp

    call model%initialize([1, 1], status, hidden_activation=MLP_LINEAR, &
        output_activation=MLP_LINEAR)
    if (.not. status_ok(status)) error stop "weighted validation benchmark initialization failed"
    call model%set_parameters([0.13_dp, -0.08_dp], status)
    if (.not. status_ok(status)) error stop "weighted validation benchmark parameters failed"
    call objective%initialize(model, train_x, train_target, validation_x, &
        validation_target, options, status, validation_weight)
    if (.not. status_ok(status)) error stop "weighted validation benchmark setup failed"
    parameters = objective%parameters()
    call objective%value_gradient(parameters, value, gradient, status)
    if (.not. status_ok(status)) error stop "weighted validation value-gradient failed"
    weighted_value = value
    weighted_gradient = gradient
    direction = [0.23_dp, -0.17_dp, 0.11_dp]
    call objective%jvp(parameters, direction, value, tangent, status)
    if (.not. status_ok(status)) error stop "weighted validation JVP failed"
    weighted_tangent = tangent
    call objective%hvp(parameters, direction, hvp_product, status)
    nonuniform_hvp_status = status%code
    if (status%code /= FORTNUM_NOT_IMPLEMENTED) then
        error stop "weighted validation non-uniform HVP boundary failed"
    end if

    call objective%initialize(model, train_x, train_target, validation_x, &
        validation_target, options, status)
    if (.not. status_ok(status)) error stop "uniform validation benchmark setup failed"
    parameters = objective%parameters()
    call objective%hvp(parameters, direction, hvp_product, status)
    if (.not. status_ok(status)) error stop "uniform validation HVP failed"

    oracle_unit = -1
    call get_environment_variable("FORTML_BENCH_MLP_WEIGHTED_VALIDATION_ORACLE", &
        oracle_path, status=environment_status)
    if (environment_status == 0 .and. len_trim(oracle_path) > 0) then
        open (newunit=oracle_unit, file=trim(oracle_path), status="replace", &
            action="write")
        write (oracle_unit, '(a)') "quantity,index,value"
        write (oracle_unit, '(a,es26.17e3)') "weighted_value,1,", weighted_value
        write (oracle_unit, '(a,es26.17e3)') "weighted_gradient,1,", weighted_gradient(1)
        write (oracle_unit, '(a,es26.17e3)') "weighted_gradient,2,", weighted_gradient(2)
        write (oracle_unit, '(a,es26.17e3)') "weighted_gradient,3,", weighted_gradient(3)
        write (oracle_unit, '(a,es26.17e3)') "weighted_jvp,1,", weighted_tangent
        write (oracle_unit, '(a,i0)') "nonuniform_hvp_status,1,", nonuniform_hvp_status
        write (oracle_unit, '(a,es26.17e3)') "uniform_hvp,1,", hvp_product(1)
        write (oracle_unit, '(a,es26.17e3)') "uniform_hvp,2,", hvp_product(2)
        write (oracle_unit, '(a,es26.17e3)') "uniform_hvp,3,", hvp_product(3)
        close (oracle_unit)
    end if
    if (oracle_only_requested()) stop

    call objective%initialize(model, train_x, train_target, validation_x, &
        validation_target, options, status, validation_weight)
    if (.not. status_ok(status)) error stop "weighted validation timing setup failed"
    parameters = objective%parameters()
    call system_clock(clock_start, clock_rate)
    do repetition = 1, repetitions
        call objective%value_gradient(parameters, value, gradient, status)
        if (.not. status_ok(status)) error stop "weighted validation timing failed"
    end do
    call system_clock(clock_end)
    elapsed = real(clock_end-clock_start, dp) / real(clock_rate, dp) / real(repetitions, dp)
    write (*, '(a,es24.16)') "mlp_weighted_validation_hypergradient_value_gradient,", elapsed

contains

    logical function oracle_only_requested()
        character(len=16) :: value
        integer :: environment_status

        call get_environment_variable("FORTML_BENCH_ORACLE_ONLY", value, &
            status=environment_status)
        oracle_only_requested = environment_status == 0 .and. trim(value) == "1"
    end function oracle_only_requested

end program fortml_bench_mlp_weighted_validation_hypergradient
