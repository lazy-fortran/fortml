program fortml_bench_classification
    !! Correctness-gated release workload for classifiers, scalers, and XGB.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortml_preprocessing, only: standard_scaler_t, minmax_scaler_t
    use fortml_xgboost, only: xgboost_t, xgboost_options_t
    use fortml_gp_classification, only: gp_classification_t, &
        gp_classification_options_t, GP_LIKELIHOOD_LOGISTIC, &
        GP_LIKELIHOOD_PROBIT
    use fortml_gp_multiclass_classification, only: &
        gp_multiclass_classification_t, gp_multiclass_classification_options_t, &
        gp_multiclass_classification_state_t
    use fortml_kernels, only: kernel_t, make_rbf_kernel
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    call benchmark_scalers()
    call benchmark_xgboost()
    call benchmark_gp_classification()

contains

    subroutine benchmark_scalers()
        integer, parameter :: n_samples = 128, n_features = 3
        integer, parameter :: repetitions = 32
        real(dp) :: x(n_samples, n_features), transformed(n_samples, n_features)
        real(dp) :: tangent(n_samples, n_features), tangent_out(n_samples, n_features)
        real(dp) :: elapsed_fit, elapsed_transform, elapsed_jvp
        real(dp) :: checksum, jvp_checksum
        integer(int64) :: clock_start, clock_end, clock_rate
        integer :: i, j, repetition
        type(standard_scaler_t) :: standard
        type(minmax_scaler_t) :: minmax
        type(fortnum_status_t) :: status

        do j = 1, n_features
            do i = 1, n_samples
                x(i, j) = sin(0.017_dp*real(i, dp) + 0.13_dp*real(j, dp))
                tangent(i, j) = cos(0.011_dp*real(i + 2*j, dp))
            end do
        end do
        x(:, 3) = 2.0_dp
        call system_clock(clock_start, clock_rate)
        call standard%fit(x, status)
        call minmax%fit(x, status, feature_range=[-1.0_dp, 1.0_dp])
        call system_clock(clock_end)
        elapsed_fit = real(clock_end - clock_start, dp)/real(clock_rate, dp)
        call standard%transform(x, transformed, status)
        checksum = sum(transformed)
        call system_clock(clock_start, clock_rate)
        do repetition = 1, repetitions
            call standard%transform(x, transformed, status)
        end do
        call system_clock(clock_end)
        elapsed_transform = real(clock_end - clock_start, dp)/real(clock_rate, dp) &
            /real(repetitions, dp)
        call standard%transform_jvp(tangent, tangent_out, status)
        call system_clock(clock_start, clock_rate)
        do repetition = 1, repetitions
            call standard%transform_jvp(tangent, tangent_out, status)
        end do
        call system_clock(clock_end)
        elapsed_jvp = real(clock_end - clock_start, dp)/real(clock_rate, dp) &
            /real(repetitions, dp)
        jvp_checksum = sum(tangent_out)
        if (.not. status_ok(status)) error stop "scaler benchmark failed"
        write (*, '(a,i0,a,i0,a,es24.16,a,es24.16,a,es24.16,a,es24.16,a,es24.16)') &
            "standard_scaler,", n_samples, ",", n_features, ",", elapsed_fit, &
            ",", elapsed_transform, ",", elapsed_jvp, ",", checksum, ",", &
            jvp_checksum

        call minmax%transform(x, transformed, status)
        checksum = sum(transformed)
        if (.not. status_ok(status)) error stop "minmax transform failed"
        write (*, '(a,i0,a,i0,a,es24.16)') "minmax_scaler,", n_samples, ",", &
            n_features, ",", checksum
    end subroutine benchmark_scalers

    subroutine benchmark_xgboost()
        integer, parameter :: n_samples = 128, n_features = 2
        integer, parameter :: n_estimators = 12, repetitions = 6
        real(dp) :: x(n_samples, n_features), y(n_samples), labels(n_samples)
        real(dp) :: prediction(n_samples), probability(n_samples)
        real(dp) :: elapsed_fit, elapsed_predict, mse, accuracy, gain
        integer(int64) :: clock_start, clock_end, clock_rate
        integer :: i, repetition
        type(xgboost_t) :: squared, logistic
        type(xgboost_options_t) :: options
        type(fortnum_status_t) :: status

        do i = 1, n_samples
            x(i, 1) = -1.0_dp + 2.0_dp*real(i - 1, dp)/real(n_samples - 1, dp)
            x(i, 2) = sin(0.09_dp*real(i, dp))
            y(i) = merge(1.7_dp + 0.2_dp*x(i, 2), -0.8_dp + &
                0.1_dp*x(i, 2), x(i, 1) >= 0.1_dp)
            labels(i) = merge(1.0_dp, 0.0_dp, x(i, 1) >= 0.1_dp)
        end do
        options%n_estimators = n_estimators
        options%learning_rate = 0.2_dp
        options%min_child_weight = 0.0_dp
        call system_clock(clock_start, clock_rate)
        do repetition = 1, repetitions
            call squared%fit_regression(x, y, status, options)
        end do
        call system_clock(clock_end)
        elapsed_fit = real(clock_end - clock_start, dp)/real(clock_rate, dp) &
            /real(repetitions, dp)
        call squared%predict(x, prediction, status)
        mse = sum((prediction - y)**2)/real(n_samples, dp)
        gain = squared%split_gain(1)
        call system_clock(clock_start, clock_rate)
        do repetition = 1, repetitions*8
            call squared%predict(x, prediction, status)
        end do
        call system_clock(clock_end)
        elapsed_predict = real(clock_end - clock_start, dp)/real(clock_rate, dp) &
            /real(repetitions*8, dp)
        if (.not. status_ok(status)) error stop "XGB regression benchmark failed"
        write (*, '(a,i0,a,i0,a,i0,a,es24.16,a,es24.16,a,es24.16,a,es24.16)') &
            "xgboost_squared,", n_samples, ",", n_features, ",", n_estimators, &
            ",", elapsed_fit, ",", elapsed_predict, ",", mse, ",", gain

        call logistic%fit_binary(x, labels, status, options)
        call logistic%predict(x, probability, status)
        accuracy = real(count((probability >= 0.5_dp) .eqv. (labels >= 0.5_dp)), dp) &
            /real(n_samples, dp)
        if (.not. status_ok(status)) error stop "XGB logistic benchmark failed"
        write (*, '(a,i0,a,i0,a,es24.16,a,es24.16)') "xgboost_logistic,", &
            n_samples, ",", n_features, ",", accuracy, ",", sum(probability)
    end subroutine benchmark_xgboost

    subroutine benchmark_gp_classification()
        integer, parameter :: n_samples = 32, n_features = 1
        integer, parameter :: n_query = 32, repetitions = 3
        real(dp) :: x(n_samples, n_features), query(n_query, n_features)
        real(dp) :: probabilities(n_query, 2), mean(n_query), variance(n_query)
        real(dp) :: multiclass_probabilities(n_query, 3)
        real(dp) :: elapsed_fit, elapsed_predict, elapsed_multiclass
        real(dp) :: accuracy, checksum
        integer :: labels(n_samples), predicted(n_samples), multiclass_labels(n_samples)
        integer :: multiclass_predicted(n_samples), i, repetition
        integer(int64) :: clock_start, clock_end, clock_rate
        type(kernel_t) :: kernel
        type(gp_classification_t) :: model
        type(gp_classification_options_t) :: options
        type(gp_multiclass_classification_t) :: multiclass_model
        type(gp_multiclass_classification_options_t) :: multiclass_options
        type(gp_multiclass_classification_state_t) :: multiclass_state
        type(fortnum_status_t) :: status

        do i = 1, n_samples
            x(i, 1) = -1.5_dp + 3.0_dp*real(i - 1, dp)/real(n_samples - 1, dp)
            labels(i) = merge(11, -7, x(i, 1) >= 0.0_dp)
            query(i, 1) = x(i, 1)
        end do
        kernel = make_rbf_kernel(n_features, 1.2_dp, 0.7_dp, status)
        options%max_iterations = 80
        options%tolerance = 1.0e-8_dp
        options%jitter = 1.0e-7_dp
        call system_clock(clock_start, clock_rate)
        do repetition = 1, repetitions
            call model%fit(x, labels, kernel, status, options)
        end do
        call system_clock(clock_end)
        elapsed_fit = real(clock_end - clock_start, dp)/real(clock_rate, dp) &
            /real(repetitions, dp)
        call model%predict(x, predicted, status)
        call model%predict_latent(query, mean, variance, status)
        call model%predict_proba(query, probabilities, status)
        accuracy = real(count(predicted == labels), dp)/real(n_samples, dp)
        checksum = sum(probabilities(:, 2))
        call system_clock(clock_start, clock_rate)
        do repetition = 1, repetitions*8
            call model%predict_proba(query, probabilities, status)
        end do
        call system_clock(clock_end)
        elapsed_predict = real(clock_end - clock_start, dp)/real(clock_rate, dp) &
            /real(repetitions*8, dp)
        if (.not. status_ok(status)) error stop "GP classification benchmark failed"
        write (*, '(a,i0,a,i0,a,es24.16,a,es24.16,a,es24.16,a,es24.16)') &
            "gp_classification_logistic,", n_samples, ",", n_features, ",", &
            elapsed_fit, ",", elapsed_predict, ",", accuracy, ",", checksum

        options%likelihood = GP_LIKELIHOOD_PROBIT
        call model%fit(x, labels, kernel, status, options)
        call model%predict_proba(query, probabilities, status)
        if (.not. status_ok(status)) error stop "GP probit benchmark failed"
        write (*, '(a,i0,a,i0,a,es24.16)') "gp_classification_probit,", &
            n_samples, ",", n_features, ",", sum(probabilities(:, 2))

        do i = 1, n_samples
            multiclass_labels(i) = merge(-7, merge(3, 11, x(i, 1) < 0.5_dp), &
                x(i, 1) < -0.5_dp)
        end do
        multiclass_options%max_iterations = 80
        multiclass_options%tolerance = 1.0e-8_dp
        multiclass_options%jitter = 1.0e-7_dp
        call system_clock(clock_start, clock_rate)
        do repetition = 1, repetitions
            call multiclass_model%fit(x, multiclass_labels, kernel, status, &
                multiclass_options, multiclass_state)
        end do
        call system_clock(clock_end)
        elapsed_multiclass = real(clock_end - clock_start, dp) &
            /real(clock_rate, dp)/real(repetitions, dp)
        call multiclass_model%predict(x, multiclass_predicted, status)
        call multiclass_model%predict_proba(x, multiclass_probabilities, status)
        accuracy = real(count(multiclass_predicted == multiclass_labels), dp) &
            /real(n_samples, dp)
        if (.not. status_ok(status)) error stop "GP multiclass benchmark failed"
        write (*, '(a,i0,a,i0,a,es24.16,a,es24.16,a,es24.16,a,es24.16)') &
            "gp_classification_multiclass,", n_samples, ",", n_features, ",", &
            elapsed_multiclass, ",", accuracy, ",", sum(multiclass_probabilities), ",", &
            real(multiclass_state%total_iterations, dp)
    end subroutine benchmark_gp_classification

end program fortml_bench_classification
