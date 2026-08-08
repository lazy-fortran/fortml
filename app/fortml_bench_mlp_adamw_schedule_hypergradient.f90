program fortml_bench_mlp_adamw_schedule_hypergradient
    !! Release workload for scheduled AdamW trajectory products.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortnum_status, only: fortnum_status_t, status_ok
    use fortml_mlp, only: mlp_t, MLP_LINEAR
    use fortml_mlp_schedules, only: make_mlp_schedule_cosine_decay
    use fortml_mlp_adamw_schedule_hypergradient, only: &
        mlp_adamw_schedule_hypergradient_objective_t, &
        mlp_adamw_schedule_hypergradient_options_t, &
        MLP_ADAMW_SCHEDULE_HYPERPARAMETER_COUNT
    implicit none

    integer, parameter :: n_train = 6, n_validation = 3, repetitions = 16
    real(dp) :: train_x(n_train, 1), train_target(n_train, 1)
    real(dp) :: validation_x(n_validation, 1), validation_target(n_validation, 1)
    real(dp) :: parameters(MLP_ADAMW_SCHEDULE_HYPERPARAMETER_COUNT)
    real(dp) :: direction(MLP_ADAMW_SCHEDULE_HYPERPARAMETER_COUNT)
    real(dp) :: gradient(MLP_ADAMW_SCHEDULE_HYPERPARAMETER_COUNT)
    real(dp) :: value, tangent, elapsed
    integer(int64) :: clock_start, clock_end, clock_rate
    integer :: oracle_unit, environment_status, repetition, i
    character(len=1024) :: oracle_path
    type(mlp_t), target :: model
    type(mlp_adamw_schedule_hypergradient_objective_t) :: objective
    type(mlp_adamw_schedule_hypergradient_options_t) :: options
    type(fortnum_status_t) :: status

    train_x(:, 1) = [-2.5_dp, -1.5_dp, -0.5_dp, 0.5_dp, 1.5_dp, 2.5_dp]
    train_target(:, 1) = 0.6_dp*train_x(:, 1) - 0.35_dp
    validation_x(:, 1) = [-1.75_dp, 0.25_dp, 1.9_dp]
    validation_target(:, 1) = 0.6_dp*validation_x(:, 1) - 0.35_dp
    options%steps = 6
    options%schedule = make_mlp_schedule_cosine_decay(8, 0.2_dp)
    options%base_rate = 0.05_dp
    options%l2 = 0.03_dp
    options%weight_decay = 0.04_dp
    options%beta1 = 0.7_dp
    options%beta2 = 0.9_dp
    options%epsilon = 0.02_dp
    options%lower_log_base_rate = -5.0_dp
    options%upper_log_base_rate = 0.0_dp
    options%lower_log_l2 = -5.0_dp
    options%upper_log_l2 = 0.0_dp
    options%lower_log_weight_decay = -5.0_dp
    options%upper_log_weight_decay = 0.0_dp
    options%lower_log_epsilon = -5.0_dp
    options%upper_log_epsilon = 0.0_dp

    call model%initialize([1, 1], status, hidden_activation=MLP_LINEAR, &
        output_activation=MLP_LINEAR, initialization_seed=41)
    if (.not. status_ok(status)) error stop "AdamW schedule benchmark initialization failed"
    call model%set_parameters([0.12_dp, -0.08_dp], status)
    if (.not. status_ok(status)) error stop "AdamW schedule benchmark parameter setup failed"
    call objective%initialize(model, train_x, train_target, validation_x, &
        validation_target, options, status)
    if (.not. status_ok(status)) error stop "AdamW schedule benchmark setup failed"
    parameters = objective%parameters()
    call objective%value_gradient(parameters, value, gradient, status)
    if (.not. status_ok(status)) error stop "AdamW schedule benchmark product failed"
    direction = [0.21_dp, -0.17_dp, 0.23_dp, -0.19_dp, 0.13_dp, 0.11_dp, 0.07_dp, -0.05_dp]
    call objective%jvp(parameters, direction, value, tangent, status)
    if (.not. status_ok(status)) error stop "AdamW schedule benchmark JVP failed"

    oracle_unit = -1
    call get_environment_variable("FORTML_BENCH_MLP_ADAMW_SCHEDULE_HYPERGRADIENT_ORACLE", &
        oracle_path, status=environment_status)
    if (environment_status == 0 .and. len_trim(oracle_path) > 0) then
        open (newunit=oracle_unit, file=trim(oracle_path), status="replace", action="write")
        write (oracle_unit, '(a)') "quantity,index,value"
        write (oracle_unit, '(a,es26.17e3)') "value,1,", value
        do i = 1, MLP_ADAMW_SCHEDULE_HYPERPARAMETER_COUNT
            write (oracle_unit, '(a,i0,a,es26.17e3)') "gradient,", i, ",", gradient(i)
        end do
        write (oracle_unit, '(a,es26.17e3)') "jvp,1,", tangent
        close (oracle_unit)
    end if
    if (oracle_only_requested()) stop

    call system_clock(clock_start, clock_rate)
    do repetition = 1, repetitions
        call objective%value_gradient(parameters, value, gradient, status)
        if (.not. status_ok(status)) error stop "AdamW schedule benchmark timing failed"
    end do
    call system_clock(clock_end)
    elapsed = real(clock_end-clock_start, dp)/real(clock_rate, dp)/real(repetitions, dp)
    write (*, '(a,es24.16)') "mlp_adamw_schedule_hypergradient_value_gradient,", elapsed

contains

    logical function oracle_only_requested()
        character(len=16) :: value
        integer :: environment_code

        call get_environment_variable("FORTML_BENCH_ORACLE_ONLY", value, &
            status=environment_code)
        oracle_only_requested = environment_code == 0 .and. trim(value) == "1"
    end function oracle_only_requested

end program fortml_bench_mlp_adamw_schedule_hypergradient
