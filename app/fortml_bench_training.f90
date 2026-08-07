program fortml_bench_training
    !! Release workload for the production training and imputation products.
    !!
    !! The benchmark harness owns independent NumPy oracles.  This executable
    !! reports release-build timings and, when requested, writes complete
    !! prediction/transform/product arrays for a behavioral gate.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use, intrinsic :: ieee_arithmetic, only: ieee_quiet_nan, ieee_value
    use fortml_mlp, only: mlp_t
    use fortml_mlp_training, only: mlp_train, mlp_training_options_t, &
        mlp_training_state_t, MLP_OPTIMIZER_SGD
    use fortml_simple_imputer, only: simple_imputer_t
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer :: oracle_unit
    character(len=1024) :: oracle_path
    integer :: environment_status

    oracle_unit = -1
    call get_environment_variable("FORTML_BENCH_TRAINING_ORACLE", oracle_path, &
        status=environment_status)
    if (environment_status == 0 .and. len_trim(oracle_path) > 0) then
        open (newunit=oracle_unit, file=trim(oracle_path), status="replace", &
            action="write")
        write (oracle_unit, '(a)') "workload,variant,quantity,row,column,value"
    end if

    call benchmark_mlp(oracle_unit, .false.)
    call benchmark_mlp(oracle_unit, .true.)
    call benchmark_imputer(oracle_unit, "mean", 0.0_dp)
    call benchmark_imputer(oracle_unit, "median", 0.0_dp)
    call benchmark_imputer(oracle_unit, "constant", -0.25_dp)

    if (oracle_unit /= -1) close (oracle_unit)

contains

    subroutine benchmark_mlp(unit, nesterov)
        integer, intent(in) :: unit
        logical, intent(in) :: nesterov
        integer, parameter :: n_samples = 96, n_features = 3
        integer, parameter :: n_hidden = 8, n_outputs = 1, epochs = 24
        integer, parameter :: repetitions = 4
        real(dp) :: x(n_samples, n_features), target(n_samples, n_outputs)
        real(dp) :: prediction(n_samples, n_outputs), loss
        real(dp) :: elapsed, initial_loss, final_loss
        integer(int64) :: clock_start, clock_end, clock_rate
        integer :: i, repetition
        type(mlp_t) :: model
        type(mlp_training_options_t) :: options
        type(mlp_training_state_t) :: state
        type(fortnum_status_t) :: status
        character(len=16) :: variant

        call make_mlp_fixture(x, target)
        variant = "sgd"
        if (nesterov) variant = "nesterov"
        options%optimizer = MLP_OPTIMIZER_SGD
        options%max_epochs = epochs
        options%batch_size = 0
        options%shuffle = .false.
        options%restore_best = .false.
        options%learning_rate = 0.01_dp
        options%momentum = 0.8_dp
        options%nesterov = nesterov
        options%l2 = 1.0e-4_dp
        options%tolerance = 0.0_dp
        options%patience = 0

        call model%initialize([n_features, n_hidden, n_outputs], status, &
            hidden_activation=2, output_activation=1, initialization_seed=23)
        if (.not. status_ok(status)) error stop "MLP SGD benchmark initialization failed"
        call mlp_train(model, x, target, status, options, state)
        if (.not. status_ok(status)) error stop "MLP SGD benchmark training failed"
        call model%predict(x, prediction, status)
        if (.not. status_ok(status)) error stop "MLP SGD benchmark prediction failed"
        initial_loss = state%initial_loss
        final_loss = state%final_loss
        if (unit /= -1) call write_mlp_oracle(unit, variant, prediction, &
            initial_loss, final_loss)

        call system_clock(clock_start, clock_rate)
        do repetition = 1, repetitions
            call model%initialize([n_features, n_hidden, n_outputs], status, &
                hidden_activation=2, output_activation=1, initialization_seed=23)
            call mlp_train(model, x, target, status, options)
            if (.not. status_ok(status)) error stop "MLP SGD benchmark timing failed"
        end do
        call system_clock(clock_end)
        elapsed = real(clock_end - clock_start, dp)/real(clock_rate, dp) &
            /real(repetitions, dp)
        call mlp_loss(model, x, target, options%l2, loss, status)
        if (.not. status_ok(status)) error stop "MLP SGD benchmark loss failed"
        write (*, '(a,a,a,i0,a,i0,a,i0,a,i0,a,es24.16,a,es24.16,a,es24.16)') &
            "mlp_training,", trim(variant), ",", n_samples, ",", n_features, ",", &
            n_hidden, ",", epochs, ",", initial_loss, ",", final_loss, ",", elapsed
    end subroutine benchmark_mlp

    subroutine make_mlp_fixture(x, target)
        real(dp), intent(out) :: x(:, :), target(:, :)
        integer :: i, j

        do j = 1, size(x, 2)
            do i = 1, size(x, 1)
                x(i, j) = sin(0.017_dp*real(i, dp) + 0.13_dp*real(j, dp)) &
                    + 0.15_dp*cos(0.009_dp*real(i*j, dp))
            end do
        end do
        do i = 1, size(x, 1)
            target(i, 1) = 0.4_dp*sin(x(i, 1)) + 0.2_dp*x(i, 2) &
                - 0.1_dp*x(i, 3) + 0.03_dp*cos(2.0_dp*x(i, 1))
        end do
    end subroutine make_mlp_fixture

    subroutine mlp_loss(model, x, target, l2, value, status)
        type(mlp_t), intent(in) :: model
        real(dp), intent(in) :: x(:, :), target(:, :), l2
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: prediction(size(target, 1), size(target, 2))
        real(dp), allocatable :: theta(:)

        call model%predict(x, prediction, status)
        if (.not. status_ok(status)) return
        theta = model%parameters()
        value = 0.5_dp*sum((prediction - target)**2)/real(size(x, 1), dp) &
            + 0.5_dp*l2*sum(theta*theta)
    end subroutine mlp_loss

    subroutine write_mlp_oracle(unit, variant, prediction, initial_loss, final_loss)
        integer, intent(in) :: unit
        character(*), intent(in) :: variant
        real(dp), intent(in) :: prediction(:, :), initial_loss, final_loss
        integer :: i

        write (unit, '(a,a,a,es26.17e3)') &
            "mlp_training,", trim(variant), ",initial_loss,1,1,", initial_loss
        write (unit, '(a,a,a,es26.17e3)') &
            "mlp_training,", trim(variant), ",final_loss,1,1,", final_loss
        do i = 1, size(prediction, 1)
            write (unit, '(a,a,a,i0,a,es26.17e3)') &
                "mlp_training,", trim(variant), ",prediction,", i, ",1,", &
                prediction(i, 1)
        end do
    end subroutine write_mlp_oracle

    subroutine benchmark_imputer(unit, strategy, fill_value)
        integer, intent(in) :: unit
        character(*), intent(in) :: strategy
        real(dp), intent(in) :: fill_value
        integer, parameter :: n_samples = 12, n_features = 4
        integer, parameter :: fit_repetitions = 32, transform_repetitions = 128
        integer, parameter :: product_repetitions = 128
        real(dp) :: x(n_samples, n_features), x_dot(n_samples, n_features)
        real(dp) :: transformed(n_samples, n_features), transformed_dot(n_samples, n_features)
        real(dp) :: output_bar(n_samples, n_features), input_bar(n_samples, n_features)
        real(dp) :: statistics(n_features), elapsed
        integer(int64) :: clock_start, clock_end, clock_rate
        integer :: i, j, repetition
        type(simple_imputer_t) :: model
        type(fortnum_status_t) :: status

        call make_imputer_fixture(x, x_dot, output_bar)
        call model%fit(x, status, strategy=strategy, fill_value=fill_value)
        if (.not. status_ok(status)) error stop "simple imputer benchmark fit failed"
        statistics = model%statistics()
        call model%transform(x, transformed, status)
        if (.not. status_ok(status)) error stop "simple imputer benchmark transform failed"
        call model%transform_jvp(x, x_dot, transformed_dot, status)
        if (.not. status_ok(status)) error stop "simple imputer benchmark JVP failed"
        call model%transform_vjp(x, output_bar, input_bar, status)
        if (.not. status_ok(status)) error stop "simple imputer benchmark VJP failed"
        if (unit /= -1) call write_imputer_oracle(unit, strategy, statistics, &
            transformed, transformed_dot, input_bar)

        call system_clock(clock_start, clock_rate)
        do repetition = 1, fit_repetitions
            call model%fit(x, status, strategy=strategy, fill_value=fill_value)
            if (.not. status_ok(status)) error stop "simple imputer timed fit failed"
        end do
        call system_clock(clock_end)
        elapsed = real(clock_end - clock_start, dp)/real(clock_rate, dp) &
            /real(fit_repetitions, dp)
        call write_imputer_timing(strategy, "fit", elapsed)

        call system_clock(clock_start, clock_rate)
        do repetition = 1, transform_repetitions
            call model%transform(x, transformed, status)
            if (.not. status_ok(status)) error stop "simple imputer timed transform failed"
        end do
        call system_clock(clock_end)
        elapsed = real(clock_end - clock_start, dp)/real(clock_rate, dp) &
            /real(transform_repetitions, dp)
        call write_imputer_timing(strategy, "transform", elapsed)

        call system_clock(clock_start, clock_rate)
        do repetition = 1, product_repetitions
            call model%transform_jvp(x, x_dot, transformed_dot, status)
            if (.not. status_ok(status)) error stop "simple imputer timed JVP failed"
        end do
        call system_clock(clock_end)
        elapsed = real(clock_end - clock_start, dp)/real(clock_rate, dp) &
            /real(product_repetitions, dp)
        call write_imputer_timing(strategy, "jvp", elapsed)

        call system_clock(clock_start, clock_rate)
        do repetition = 1, product_repetitions
            call model%transform_vjp(x, output_bar, input_bar, status)
            if (.not. status_ok(status)) error stop "simple imputer timed VJP failed"
        end do
        call system_clock(clock_end)
        elapsed = real(clock_end - clock_start, dp)/real(clock_rate, dp) &
            /real(product_repetitions, dp)
        call write_imputer_timing(strategy, "vjp", elapsed)
    end subroutine benchmark_imputer

    subroutine make_imputer_fixture(x, x_dot, output_bar)
        real(dp), intent(out) :: x(:, :), x_dot(:, :), output_bar(:, :)
        integer :: i, j
        real(dp) :: nan

        nan = ieee_value(0.0_dp, ieee_quiet_nan)
        do j = 1, size(x, 2)
            do i = 1, size(x, 1)
                select case (j)
                case (1)
                    x(i, j) = 0.2_dp*real(i, dp)
                    if (i == 2 .or. i == 8) x(i, j) = nan
                case (2)
                    x(i, j) = -0.3_dp + 0.15_dp*real(i, dp)
                    if (i == 1 .or. i == 12) x(i, j) = nan
                case (3)
                    x(i, j) = sin(0.3_dp*real(i, dp))
                    if (i == 4 .or. i == 7) x(i, j) = nan
                case default
                    x(i, j) = 0.1_dp*real(i*i, dp)
                    if (i == 3 .or. i == 5 .or. i == 10) x(i, j) = nan
                end select
                x_dot(i, j) = 0.01_dp*cos(0.2_dp*real(i, dp) &
                    + 0.1_dp*real(j, dp))
                output_bar(i, j) = 0.2_dp*real(i, dp) - 0.03_dp*real(j, dp)
            end do
        end do
    end subroutine make_imputer_fixture

    subroutine write_imputer_oracle(unit, strategy, statistics, transformed, &
            transformed_dot, input_bar)
        integer, intent(in) :: unit
        character(*), intent(in) :: strategy
        real(dp), intent(in) :: statistics(:), transformed(:, :), transformed_dot(:, :)
        real(dp), intent(in) :: input_bar(:, :)
        integer :: i, j

        do j = 1, size(statistics)
            write (unit, '(a,a,a,i0,a,es26.17e3)') &
                "simple_imputer,", trim(strategy), ",statistic,1,", j, ",", &
                statistics(j)
        end do
        do j = 1, size(transformed, 2)
            do i = 1, size(transformed, 1)
                write (unit, '(a,a,a,i0,a,i0,a,es26.17e3)') &
                    "simple_imputer,", trim(strategy), ",transform,", i, ",", j, ",", &
                    transformed(i, j)
                write (unit, '(a,a,a,i0,a,i0,a,es26.17e3)') &
                    "simple_imputer,", trim(strategy), ",jvp,", i, ",", j, ",", &
                    transformed_dot(i, j)
                write (unit, '(a,a,a,i0,a,i0,a,es26.17e3)') &
                    "simple_imputer,", trim(strategy), ",vjp,", i, ",", j, ",", &
                    input_bar(i, j)
            end do
        end do
    end subroutine write_imputer_oracle

    subroutine write_imputer_timing(strategy, phase, elapsed)
        character(*), intent(in) :: strategy, phase
        real(dp), intent(in) :: elapsed

        write (*, '(a,a,a,a,a,es24.16)') "simple_imputer,", trim(strategy), &
            ",", trim(phase), ",12,4,", elapsed
    end subroutine write_imputer_timing

end program fortml_bench_training
