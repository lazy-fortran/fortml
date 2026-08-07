program fortml_bench_adamw_beta_hypergradient
    !! Release workload for the five-parameter AdamW beta-logit objective.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortml_mlp, only: mlp_t
    use fortml_mlp_hypergradient, only: &
        mlp_adamw_full_hypergradient_objective_t, &
        mlp_adamw_full_hypergradient_options_t
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer, parameter :: train_count = 5, validation_count = 3
    integer, parameter :: n_features = 1, n_outputs = 1, steps = 4
    integer, parameter :: parameter_count = 5, repetitions = 32
    real(dp), parameter :: learning_rate = 0.12_dp, l2 = 0.07_dp
    real(dp), parameter :: weight_decay = 0.03_dp, beta1 = 0.82_dp
    real(dp), parameter :: beta2 = 0.91_dp, epsilon = 1.0e-8_dp
    real(dp) :: train_x(train_count, n_features), train_target(train_count, n_outputs)
    real(dp) :: validation_x(validation_count, n_features)
    real(dp) :: validation_target(validation_count, n_outputs)
    real(dp) :: parameters(parameter_count), direction(parameter_count)
    real(dp) :: gradient(parameter_count), value, tangent, elapsed
    integer(int64) :: clock_start, clock_end, clock_rate
    integer :: oracle_unit, environment_status, repetition, index
    character(len=1024) :: oracle_path
    type(mlp_t), target :: model
    type(mlp_adamw_full_hypergradient_objective_t) :: objective
    type(mlp_adamw_full_hypergradient_options_t) :: options
    type(fortnum_status_t) :: status

    call make_fixture(train_x, train_target, validation_x, validation_target)
    options%steps = steps
    options%learning_rate = learning_rate
    options%l2 = l2
    options%weight_decay = weight_decay
    options%beta1 = beta1
    options%beta2 = beta2
    options%epsilon = epsilon
    options%lower_log_learning_rate = -4.0_dp
    options%upper_log_learning_rate = 0.0_dp
    options%lower_log_l2 = -5.0_dp
    options%upper_log_l2 = 0.0_dp
    options%lower_log_weight_decay = -6.0_dp
    options%upper_log_weight_decay = 0.0_dp
    options%lower_logit_beta1 = -4.0_dp
    options%upper_logit_beta1 = 4.0_dp
    options%lower_logit_beta2 = -4.0_dp
    options%upper_logit_beta2 = 4.0_dp

    call model%initialize([n_features, n_outputs], status, initialization_seed=23)
    if (.not. status_ok(status)) error stop "AdamW beta benchmark initialization failed"
    call model%set_parameters([0.15_dp, -0.1_dp], status)
    if (.not. status_ok(status)) error stop "AdamW beta benchmark parameter setup failed"
    call objective%initialize(model, train_x, train_target, validation_x, &
        validation_target, options, status)
    if (.not. status_ok(status)) error stop "AdamW beta benchmark objective setup failed"
    parameters = objective%parameters()
    call objective%value_gradient(parameters, value, gradient, status)
    if (.not. status_ok(status)) error stop "AdamW beta benchmark value/gradient failed"
    direction = [0.31_dp, -0.27_dp, 0.19_dp, 0.13_dp, -0.22_dp]
    call objective%jvp(parameters, direction, value, tangent, status)
    if (.not. status_ok(status)) error stop "AdamW beta benchmark JVP failed"

    oracle_unit = -1
    call get_environment_variable("FORTML_BENCH_ADAMW_BETA_ORACLE", oracle_path, &
        status=environment_status)
    if (environment_status == 0 .and. len_trim(oracle_path) > 0) then
        open (newunit=oracle_unit, file=trim(oracle_path), status="replace", &
            action="write")
        write (oracle_unit, '(a)') "quantity,index,value"
        write (oracle_unit, '(a,es26.17e3)') "value,1,", value
        do index = 1, parameter_count
            write (oracle_unit, '(a,i0,a,es26.17e3)') "gradient,", index, ",", &
                gradient(index)
        end do
        write (oracle_unit, '(a,es26.17e3)') "jvp,1,", tangent
        close (oracle_unit)
    end if
    if (oracle_only_requested()) stop

    call system_clock(clock_start, clock_rate)
    do repetition = 1, repetitions
        call objective%value_gradient(parameters, value, gradient, status)
        if (.not. status_ok(status)) error stop "AdamW beta benchmark timing failed"
    end do
    call system_clock(clock_end)
    elapsed = real(clock_end-clock_start, dp)/real(clock_rate, dp)/real(repetitions, dp)
    write (*, '(a,es24.16)') "mlp_adamw_beta_hypergradient_value_gradient,", elapsed

contains

    subroutine make_fixture(train_features, train_targets, validation_features, &
            validation_targets)
        real(dp), intent(out) :: train_features(:, :), train_targets(:, :)
        real(dp), intent(out) :: validation_features(:, :), validation_targets(:, :)

        train_features(:, 1) = [-2.0_dp, -1.0_dp, 0.0_dp, 1.0_dp, 2.0_dp]
        train_targets(:, 1) = 0.7_dp*train_features(:, 1) - 0.2_dp
        validation_features(:, 1) = [-1.5_dp, 0.5_dp, 1.75_dp]
        validation_targets(:, 1) = 0.7_dp*validation_features(:, 1) - 0.2_dp
    end subroutine make_fixture

    logical function oracle_only_requested()
        character(len=16) :: environment_value
        integer :: environment_code

        call get_environment_variable("FORTML_BENCH_ORACLE_ONLY", environment_value, &
            status=environment_code)
        oracle_only_requested = environment_code == 0 .and. trim(environment_value) == "1"
    end function oracle_only_requested

end program fortml_bench_adamw_beta_hypergradient
