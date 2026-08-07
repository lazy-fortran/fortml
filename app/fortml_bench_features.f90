program fortml_bench_features
    !! Reproducible release workload for the newly added model primitives.
    !!
    !! The executable keeps the mathematical fixture deliberately small.  The
    !! Python harness supplies independent NumPy oracles; this program only
    !! reports values and timings after checking the FortML status contract.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortml_mlp, only: mlp_t
    use fortml_mlp_training, only: mlp_train, mlp_training_options_t, &
        mlp_training_state_t, mlp_loss_value_gradient
    use fortml_basis, only: basis_map_t, make_polynomial_basis, &
        make_fourier_basis
    use fortml_pipeline, only: basis_pipeline_t, make_basis_pipeline
    use fortml_tree, only: decision_stump_t, gradient_boosting_regressor_t, &
        cart_regressor_t
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    call benchmark_mlp_training()
    call benchmark_basis_pipeline()
    call benchmark_tree_models()

contains

    subroutine benchmark_mlp_training()
        integer, parameter :: n_samples = 96, n_features = 3
        integer, parameter :: n_hidden = 8, n_outputs = 1, repetitions = 4
        integer, parameter :: epochs = 24
        real(dp) :: x(n_samples, n_features), target(n_samples, n_outputs)
        real(dp) :: prediction(n_samples, n_outputs)
        real(dp), allocatable :: gradient(:)
        real(dp) :: loss, l2_gradient, elapsed
        integer(int64) :: clock_start, clock_end, clock_rate
        integer :: i, j, repetition
        type(mlp_t) :: model
        type(mlp_training_options_t) :: options
        type(mlp_training_state_t) :: state
        type(fortnum_status_t) :: status

        do j = 1, n_features
            do i = 1, n_samples
                x(i, j) = sin(0.017_dp*real(i, dp) + 0.13_dp*real(j, dp)) &
                    + 0.15_dp*cos(0.009_dp*real(i*j, dp))
            end do
        end do
        do i = 1, n_samples
            target(i, 1) = 0.4_dp*sin(x(i, 1)) + 0.2_dp*x(i, 2) &
                - 0.1_dp*x(i, 3) + 0.03_dp*cos(2.0_dp*x(i, 1))
        end do

        options%max_epochs = epochs
        options%batch_size = 0
        options%shuffle = .false.
        options%restore_best = .false.
        options%learning_rate = 0.01_dp
        options%l2 = 1.0e-4_dp
        options%tolerance = 0.0_dp
        options%patience = 0

        call model%initialize([n_features, n_hidden, n_outputs], status, &
            hidden_activation=2, output_activation=1, initialization_seed=23)
        call mlp_train(model, x, target, status, options, state)
        if (.not. status_ok(status)) error stop "MLP training benchmark failed"
        call model%predict(x, prediction, status)
        if (.not. status_ok(status)) error stop "MLP training prediction failed"
        allocate(gradient(model%parameter_count()))
        call mlp_loss_value_gradient(model, x, target, options%l2, loss, &
            gradient, l2_gradient, status)
        if (.not. status_ok(status)) error stop "MLP training loss failed"

        call system_clock(clock_start, clock_rate)
        do repetition = 1, repetitions
            call model%initialize([n_features, n_hidden, n_outputs], status, &
                hidden_activation=2, output_activation=1, initialization_seed=23)
            call mlp_train(model, x, target, status, options)
            if (.not. status_ok(status)) error stop "MLP timed training failed"
        end do
        call system_clock(clock_end)
        elapsed = real(clock_end - clock_start, dp)/real(clock_rate, dp)

        write (*, '(a,i0,a,i0,a,i0,a,i0,a,i0,a,es24.16,a,es24.16,a,es24.16)') &
            "mlp_train,", n_samples, ",", n_features, ",", n_hidden, ",", &
            n_outputs, ",", state%epochs, ",", state%initial_loss, ",", &
            loss, ",", elapsed/real(repetitions, dp)
        do i = 1, min(8, n_samples)
            write (*, '(a,i0,a,es24.16)') "mlp_train_prediction,", i, ",", &
                prediction(i, 1)
        end do
    end subroutine benchmark_mlp_training

    subroutine benchmark_basis_pipeline()
        integer, parameter :: n_samples = 256, n_inputs = 2
        integer, parameter :: repetitions = 32
        real(dp) :: x(n_samples, n_inputs), x_dot(n_samples, n_inputs)
        real(dp) :: frequencies(2, n_inputs)
        real(dp), allocatable :: phi(:, :), phi_dot(:, :), u(:, :)
        real(dp), allocatable :: theta_dot(:), theta_bar(:), x_bar(:, :)
        real(dp) :: transform_sum, jvp_sum, theta_bar_sum, x_bar_sum, elapsed
        integer(int64) :: clock_start, clock_end, clock_rate
        integer :: i, j, repetition, n_features
        type(basis_map_t) :: polynomial, fourier
        type(basis_pipeline_t) :: pipeline
        type(fortnum_status_t) :: status

        do j = 1, n_inputs
            do i = 1, n_samples
                x(i, j) = sin(0.011_dp*real(i, dp) + 0.17_dp*real(j, dp))
                x_dot(i, j) = cos(0.007_dp*real(i + 2*j, dp))
            end do
        end do
        frequencies = reshape([1.2_dp, 0.55_dp, 0.7_dp, 1.1_dp], &
            shape(frequencies))
        polynomial = make_polynomial_basis(n_inputs, 3, status, &
            include_intercept=.true.)
        if (.not. status_ok(status)) error stop "polynomial basis construction failed"
        fourier = make_fourier_basis(n_inputs, frequencies, status, &
            include_intercept=.true.)
        if (.not. status_ok(status)) error stop "Fourier basis construction failed"
        pipeline = make_basis_pipeline(n_inputs, status)
        call pipeline%append(polynomial, status)
        call pipeline%append(fourier, status)
        call pipeline%fit(x, status)
        if (.not. status_ok(status)) error stop "basis pipeline construction failed"

        n_features = pipeline%feature_count()
        allocate(phi(n_samples, n_features), phi_dot(n_samples, n_features))
        allocate(u(n_samples, n_features), theta_dot(pipeline%parameter_count()))
        allocate(theta_bar(size(theta_dot)), x_bar(n_samples, n_inputs))
        do j = 1, n_features
            do i = 1, n_samples
                u(i, j) = 0.13_dp*sin(0.013_dp*real(i + j, dp))
            end do
        end do
        theta_dot = 0.07_dp

        call pipeline%transform(x, phi, status)
        transform_sum = sum(phi)
        call pipeline%jvp(x, theta_dot, x_dot, phi, phi_dot, status)
        call pipeline%vjp(x, u, theta_bar, x_bar, status)
        if (.not. status_ok(status)) error stop "basis pipeline products failed"
        jvp_sum = sum(phi_dot)
        theta_bar_sum = sum(theta_bar)
        x_bar_sum = sum(x_bar)

        call system_clock(clock_start, clock_rate)
        do repetition = 1, repetitions
            call pipeline%transform(x, phi, status)
            if (.not. status_ok(status)) error stop "basis transform timing failed"
        end do
        call system_clock(clock_end)
        elapsed = real(clock_end - clock_start, dp)/real(clock_rate, dp)
        write (*, '(a,i0,a,i0,a,i0,a,es24.16,a,es24.16)') "basis_transform,", &
            n_samples, ",", n_inputs, ",", n_features, ",", &
            elapsed/real(repetitions, dp), ",", transform_sum

        call system_clock(clock_start, clock_rate)
        do repetition = 1, repetitions
            call pipeline%jvp(x, theta_dot, x_dot, phi, phi_dot, status)
            if (.not. status_ok(status)) error stop "basis JVP timing failed"
        end do
        call system_clock(clock_end)
        elapsed = real(clock_end - clock_start, dp)/real(clock_rate, dp)
        write (*, '(a,i0,a,i0,a,i0,a,es24.16,a,es24.16)') "basis_jvp,", &
            n_samples, ",", n_inputs, ",", n_features, ",", &
            elapsed/real(repetitions, dp), ",", jvp_sum

        call system_clock(clock_start, clock_rate)
        do repetition = 1, repetitions
            call pipeline%vjp(x, u, theta_bar, x_bar, status)
            if (.not. status_ok(status)) error stop "basis VJP timing failed"
        end do
        call system_clock(clock_end)
        elapsed = real(clock_end - clock_start, dp)/real(clock_rate, dp)
        write (*, '(a,i0,a,i0,a,i0,a,es24.16,a,es24.16,a,es24.16)') "basis_vjp,", &
            n_samples, ",", n_inputs, ",", n_features, ",", &
            elapsed/real(repetitions, dp), ",", theta_bar_sum, ",", x_bar_sum
    end subroutine benchmark_basis_pipeline

    subroutine benchmark_tree_models()
        integer, parameter :: n_samples = 128, n_features = 2
        integer, parameter :: n_estimators = 16, repetitions = 8
        real(dp) :: x(n_samples, n_features), y(n_samples)
        real(dp) :: prediction(n_samples), x_dot(n_samples, n_features)
        real(dp) :: prediction_dot(n_samples), elapsed_fit, elapsed_predict
        real(dp) :: mse, elapsed
        real(dp) :: threshold, left_value, right_value
        integer(int64) :: clock_start, clock_end, clock_rate
        integer :: i, j, repetition
        type(decision_stump_t) :: stump
        type(gradient_boosting_regressor_t) :: booster
        type(cart_regressor_t) :: cart
        type(fortnum_status_t) :: status

        do i = 1, n_samples
            x(i, 1) = -1.0_dp + 2.0_dp*real(i - 1, dp)/real(n_samples - 1, dp)
            x(i, 2) = sin(0.09_dp*real(i, dp))
            y(i) = merge(1.7_dp + 0.2_dp*x(i, 2), -0.8_dp + &
                0.1_dp*x(i, 2), x(i, 1) >= 0.1_dp)
        end do
        x_dot = 0.0_dp
        x_dot(:, 1) = 0.3_dp

        call system_clock(clock_start, clock_rate)
        do repetition = 1, repetitions
            call stump%fit(x, y, status, min_samples_leaf=3)
            if (.not. status_ok(status)) error stop "stump fit timing failed"
        end do
        call system_clock(clock_end)
        elapsed_fit = real(clock_end - clock_start, dp)/real(clock_rate, dp) &
            /real(repetitions, dp)
        call stump%predict(x, prediction, status)
        call stump%jvp(x, x_dot, prediction, prediction_dot, status)
        if (.not. status_ok(status)) error stop "stump prediction failed"
        threshold = stump%split_threshold()
        left_value = sum(prediction, mask=x(:, stump%split_feature()) < threshold) &
            /real(count(x(:, stump%split_feature()) < threshold), dp)
        right_value = sum(prediction, mask=x(:, stump%split_feature()) >= threshold) &
            /real(count(x(:, stump%split_feature()) >= threshold), dp)
        mse = sum((prediction - y)**2)/real(n_samples, dp)
        call system_clock(clock_start, clock_rate)
        do repetition = 1, repetitions*8
            call stump%predict(x, prediction, status)
            if (.not. status_ok(status)) error stop "stump prediction timing failed"
        end do
        call system_clock(clock_end)
        elapsed_predict = real(clock_end - clock_start, dp)/real(clock_rate, dp) &
            /real(repetitions*8, dp)
        write (*, '(a,i0,a,i0,a,es24.16,a,es24.16,a,i0,a,es24.16,a,es24.16,a,es24.16,a,es24.16,a,es24.16)') &
            "stump,", n_samples, ",", n_features, ",", elapsed_fit, ",", &
            elapsed_predict, ",", stump%split_feature(), ",", threshold, ",", &
            left_value, ",", right_value, ",", mse, ",", sum(prediction)

        call system_clock(clock_start, clock_rate)
        do repetition = 1, repetitions
            call cart%fit(x, y, status, max_depth=3, min_samples_leaf=3)
            if (.not. status_ok(status)) error stop "CART fit timing failed"
        end do
        call system_clock(clock_end)
        elapsed_fit = real(clock_end - clock_start, dp)/real(clock_rate, dp) &
            /real(repetitions, dp)
        call cart%predict(x, prediction, status)
        if (.not. status_ok(status)) error stop "CART prediction failed"
        mse = sum((prediction - y)**2)/real(n_samples, dp)
        call system_clock(clock_start, clock_rate)
        do repetition = 1, repetitions*8
            call cart%predict(x, prediction, status)
            if (.not. status_ok(status)) error stop "CART prediction timing failed"
        end do
        call system_clock(clock_end)
        elapsed_predict = real(clock_end - clock_start, dp)/real(clock_rate, dp) &
            /real(repetitions*8, dp)
        write (*, '(a,i0,a,i0,a,i0,a,es24.16,a,es24.16,a,es24.16,a,es24.16,a,i0)') &
            "cart,", n_samples, ",", n_features, ",", cart%depth(), ",", &
            elapsed_fit, ",", elapsed_predict, ",", mse, ",", sum(prediction), &
            ",", cart%node_count()

        call system_clock(clock_start, clock_rate)
        do repetition = 1, repetitions
            call booster%fit(x, y, status, n_estimators=n_estimators, &
                learning_rate=0.1_dp, min_samples_leaf=3)
            if (.not. status_ok(status)) error stop "boosting fit timing failed"
        end do
        call system_clock(clock_end)
        elapsed_fit = real(clock_end - clock_start, dp)/real(clock_rate, dp) &
            /real(repetitions, dp)
        call booster%predict(x, prediction, status)
        if (.not. status_ok(status)) error stop "boosting prediction failed"
        mse = sum((prediction - y)**2)/real(n_samples, dp)
        call system_clock(clock_start, clock_rate)
        do repetition = 1, repetitions*8
            call booster%predict(x, prediction, status)
            if (.not. status_ok(status)) error stop "boosting prediction timing failed"
        end do
        call system_clock(clock_end)
        elapsed_predict = real(clock_end - clock_start, dp)/real(clock_rate, dp) &
            /real(repetitions*8, dp)
        elapsed = maxval(abs(prediction_dot))
        write (*, '(a,i0,a,i0,a,i0,a,es24.16,a,es24.16,a,es24.16,a,es24.16)') &
            "boosting,", n_samples, ",", n_features, ",", n_estimators, ",", &
            elapsed_fit, ",", elapsed_predict, ",", mse, ",", sum(prediction)
    end subroutine benchmark_tree_models

end program fortml_bench_features
