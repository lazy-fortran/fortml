program fortml_bench_adamw_training
    !! Release workload for deterministic full-batch MLP AdamW training.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortml_mlp, only: mlp_t
    use fortml_mlp_training, only: mlp_train, mlp_training_options_t, &
        mlp_training_state_t, MLP_OPTIMIZER_ADAMW
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer, parameter :: n_samples = 96, n_features = 3, n_hidden = 8
    integer, parameter :: n_outputs = 1, epochs = 24, repetitions = 4
    real(dp), parameter :: learning_rate = 1.0e-2_dp, beta1 = 0.9_dp
    real(dp), parameter :: beta2 = 0.999_dp, epsilon = 1.0e-8_dp
    real(dp), parameter :: weight_decay = 1.0e-2_dp, l2 = 1.0e-4_dp
    real(dp) :: x(n_samples, n_features), target(n_samples, n_outputs)
    real(dp) :: prediction(n_samples, n_outputs), elapsed
    real(dp) :: initial_loss, final_loss
    integer(int64) :: clock_start, clock_end, clock_rate
    integer :: oracle_unit, environment_status, repetition
    character(len=1024) :: oracle_path
    type(mlp_t) :: model
    type(mlp_training_options_t) :: options
    type(mlp_training_state_t) :: state
    type(fortnum_status_t) :: status

    call make_fixture(x, target)
    call configure(options)
    call model%initialize([n_features, n_hidden, n_outputs], status, &
        hidden_activation=2, output_activation=1, initialization_seed=23)
    if (.not. status_ok(status)) error stop "AdamW benchmark initialization failed"
    call mlp_train(model, x, target, status, options, state)
    if (.not. status_ok(status)) error stop "AdamW benchmark training failed"
    call model%predict(x, prediction, status)
    if (.not. status_ok(status)) error stop "AdamW benchmark prediction failed"
    initial_loss = state%initial_loss
    final_loss = state%final_loss

    oracle_unit = -1
    call get_environment_variable("FORTML_BENCH_ADAMW_ORACLE", oracle_path, &
        status=environment_status)
    if (environment_status == 0 .and. len_trim(oracle_path) > 0) then
        open (newunit=oracle_unit, file=trim(oracle_path), status="replace", &
            action="write")
        write (oracle_unit, '(a)') "quantity,index,value"
        write (oracle_unit, '(a,es26.17e3)') "initial_loss,1,", initial_loss
        write (oracle_unit, '(a,es26.17e3)') "final_loss,1,", final_loss
        do repetition = 1, n_samples
            write (oracle_unit, '(a,i0,a,es26.17e3)') "prediction,", repetition, ",", &
                prediction(repetition, 1)
        end do
        close (oracle_unit)
    end if
    if (oracle_only_requested()) stop

    call system_clock(clock_start, clock_rate)
    do repetition = 1, repetitions
        call model%initialize([n_features, n_hidden, n_outputs], status, &
            hidden_activation=2, output_activation=1, initialization_seed=23)
        call mlp_train(model, x, target, status, options)
        if (.not. status_ok(status)) error stop "AdamW benchmark timing failed"
    end do
    call system_clock(clock_end)
    elapsed = real(clock_end-clock_start, dp)/real(clock_rate, dp)/real(repetitions, dp)
    write (*, '(a,es24.16)') "mlp_adamw_fit,", elapsed

    call system_clock(clock_start, clock_rate)
    do repetition = 1, repetitions
        call model%predict(x, prediction, status)
        if (.not. status_ok(status)) error stop "AdamW prediction timing failed"
    end do
    call system_clock(clock_end)
    elapsed = real(clock_end-clock_start, dp)/real(clock_rate, dp)/real(repetitions, dp)
    write (*, '(a,es24.16)') "mlp_adamw_predict,", elapsed

contains

    subroutine configure(config)
        type(mlp_training_options_t), intent(out) :: config

        config%optimizer = MLP_OPTIMIZER_ADAMW
        config%max_epochs = epochs
        config%batch_size = 0
        config%shuffle = .false.
        config%restore_best = .false.
        config%learning_rate = learning_rate
        config%beta1 = beta1
        config%beta2 = beta2
        config%epsilon = epsilon
        config%weight_decay = weight_decay
        config%l2 = l2
        config%tolerance = 0.0_dp
        config%patience = 0
    end subroutine configure

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

end program fortml_bench_adamw_training
