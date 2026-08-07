program fortml_bench_mlp_hypergradient
    !! Release workload for the exact fixed full-batch MLP SGD hypergradient.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortml_mlp, only: mlp_t
    use fortml_mlp_hypergradient, only: mlp_hypergradient_objective_t, &
        mlp_hypergradient_options_t
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer, parameter :: n_samples = 96, n_features = 3, n_hidden = 8
    integer, parameter :: n_outputs = 1, train_count = 72, validation_count = 24
    integer, parameter :: steps = 8, repetitions = 16
    real(dp), parameter :: learning_rate = 1.0e-2_dp, l2 = 1.0e-4_dp
    real(dp) :: x(n_samples, n_features), target(n_samples, n_outputs)
    real(dp) :: train_x(train_count, n_features), train_target(train_count, n_outputs)
    real(dp) :: validation_x(validation_count, n_features)
    real(dp) :: validation_target(validation_count, n_outputs)
    real(dp) :: parameters(2), direction(2), gradient(2), value, tangent, elapsed
    integer(int64) :: clock_start, clock_end, clock_rate
    integer :: oracle_unit, environment_status, repetition
    character(len=1024) :: oracle_path
    type(mlp_t), target :: model
    type(mlp_hypergradient_objective_t) :: objective
    type(mlp_hypergradient_options_t) :: options
    type(fortnum_status_t) :: status

    call make_fixture(x, target)
    train_x = x(:train_count, :)
    train_target = target(:train_count, :)
    validation_x = x(train_count+1:, :)
    validation_target = target(train_count+1:, :)
    options%steps = steps
    options%learning_rate = learning_rate
    options%l2 = l2
    options%lower_log_learning_rate = -12.0_dp
    options%upper_log_learning_rate = 2.0_dp
    options%lower_log_l2 = -20.0_dp
    options%upper_log_l2 = 2.0_dp
    call model%initialize([n_features, n_hidden, n_outputs], status, &
        hidden_activation=2, output_activation=1, initialization_seed=23)
    if (.not. status_ok(status)) error stop "hypergradient benchmark initialization failed"
    call objective%initialize(model, train_x, train_target, validation_x, &
        validation_target, options, status)
    if (.not. status_ok(status)) error stop "hypergradient benchmark setup failed"
    parameters = objective%parameters()
    call objective%value_gradient(parameters, value, gradient, status)
    if (.not. status_ok(status)) error stop "hypergradient benchmark product failed"
    direction = [0.7_dp, -0.3_dp]
    call objective%jvp(parameters, direction, value, tangent, status)
    if (.not. status_ok(status)) error stop "hypergradient benchmark JVP failed"

    oracle_unit = -1
    call get_environment_variable("FORTML_BENCH_HYPERGRADIENT_ORACLE", oracle_path, &
        status=environment_status)
    if (environment_status == 0 .and. len_trim(oracle_path) > 0) then
        open (newunit=oracle_unit, file=trim(oracle_path), status="replace", &
            action="write")
        write (oracle_unit, '(a)') "quantity,index,value"
        write (oracle_unit, '(a,es26.17e3)') "value,1,", value
        write (oracle_unit, '(a,es26.17e3)') "gradient,1,", gradient(1)
        write (oracle_unit, '(a,es26.17e3)') "gradient,2,", gradient(2)
        write (oracle_unit, '(a,es26.17e3)') "jvp,1,", tangent
        close (oracle_unit)
    end if
    if (oracle_only_requested()) stop

    call system_clock(clock_start, clock_rate)
    do repetition = 1, repetitions
        call objective%value_gradient(parameters, value, gradient, status)
        if (.not. status_ok(status)) error stop "hypergradient benchmark timing failed"
    end do
    call system_clock(clock_end)
    elapsed = real(clock_end-clock_start, dp)/real(clock_rate, dp)/real(repetitions, dp)
    write (*, '(a,es24.16)') "mlp_hypergradient_value_gradient,", elapsed

contains

    subroutine make_fixture(features, targets)
        real(dp), intent(out) :: features(:, :), targets(:, :)
        integer :: i, j

        do j = 1, size(features, 2)
            do i = 1, size(features, 1)
                features(i, j) = sin(0.017_dp*real(i, dp) + 0.13_dp*real(j, dp)) &
                    + 0.15_dp*cos(0.009_dp*real(i*j, dp))
            end do
        end do
        do i = 1, size(features, 1)
            targets(i, 1) = 0.4_dp*sin(features(i, 1)) + 0.2_dp*features(i, 2) &
                - 0.1_dp*features(i, 3) + 0.03_dp*cos(2.0_dp*features(i, 1))
        end do
    end subroutine make_fixture

    logical function oracle_only_requested()
        character(len=16) :: value
        integer :: environment_status

        call get_environment_variable("FORTML_BENCH_ORACLE_ONLY", value, &
            status=environment_status)
        oracle_only_requested = environment_status == 0 .and. trim(value) == "1"
    end function oracle_only_requested

end program fortml_bench_mlp_hypergradient
