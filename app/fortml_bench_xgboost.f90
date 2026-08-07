program fortml_bench_xgboost
    !! Release workload for the exact depth-one second-order boosting lane.
    !!
    !! NumPy reconstructs this fixture and the split formulas independently.
    !! This executable reports only values and release-build timings.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortml_xgboost, only: xgboost_t, xgboost_options_t
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    call benchmark_regression()
    call benchmark_logistic()

contains

    subroutine benchmark_regression()
        integer, parameter :: n_samples = 192, n_features = 3
        integer, parameter :: n_estimators = 12, repetitions = 4
        integer, parameter :: prediction_repetitions = 32
        real(dp) :: x(n_samples, n_features), y(n_samples)
        real(dp) :: prediction(n_samples), weights(2)
        real(dp) :: mse, elapsed_fit, elapsed_predict, split_gain
        integer(int64) :: clock_start, clock_end, clock_rate
        integer :: i, repetition
        type(xgboost_t) :: model
        type(xgboost_options_t) :: options
        type(fortnum_status_t) :: status

        call make_fixture(x, y)
        options%n_estimators = n_estimators
        options%max_depth = 1
        options%min_samples_leaf = 2
        options%learning_rate = 0.25_dp
        options%l1 = 0.15_dp
        options%l2 = 1.5_dp
        options%gamma = 0.01_dp
        options%min_child_weight = 0.1_dp

        call system_clock(clock_start, clock_rate)
        do repetition = 1, repetitions
            call model%fit_regression(x, y, status, options)
            if (.not. status_ok(status)) error stop "XGBoost regression fit failed"
        end do
        call system_clock(clock_end)
        elapsed_fit = real(clock_end - clock_start, dp)/real(clock_rate, dp) &
            /real(repetitions, dp)
        call model%fit_regression(x, y, status, options)
        if (.not. status_ok(status)) error stop "XGBoost regression reference fit failed"
        call model%predict(x, prediction, status)
        if (.not. status_ok(status)) error stop "XGBoost regression prediction failed"
        mse = sum((prediction - y)**2)/real(n_samples, dp)
        split_gain = model%split_gain(1)
        call model%leaf_weights(1, weights, status)
        if (.not. status_ok(status)) error stop "XGBoost regression leaf query failed"
        write (*, '(a,i0,a,i0,a,i0,a,es24.16,a,es24.16,a,es24.16,a,es24.16)') &
            "xgb_regression_fit,", n_samples, ",", n_features, ",", &
            n_estimators, ",", elapsed_fit, ",", mse, ",", split_gain, ",", &
            sum(weights)

        call system_clock(clock_start, clock_rate)
        do repetition = 1, prediction_repetitions
            call model%predict(x, prediction, status)
            if (.not. status_ok(status)) error stop "XGBoost regression timing failed"
        end do
        call system_clock(clock_end)
        elapsed_predict = real(clock_end - clock_start, dp)/real(clock_rate, dp) &
            /real(prediction_repetitions, dp)
        write (*, '(a,i0,a,i0,a,i0,a,es24.16,a,es24.16,a,es24.16)') &
            "xgb_regression_predict,", n_samples, ",", n_features, ",", &
            n_estimators, ",", elapsed_predict, ",", mse, ",", sum(prediction)
    end subroutine benchmark_regression

    subroutine benchmark_logistic()
        integer, parameter :: n_samples = 192, n_features = 3
        integer, parameter :: n_estimators = 12, repetitions = 4
        integer, parameter :: prediction_repetitions = 32
        real(dp) :: x(n_samples, n_features), labels(n_samples)
        real(dp) :: probabilities(n_samples), probability_matrix(n_samples, 2)
        real(dp) :: logloss, elapsed_fit, elapsed_predict, split_gain
        real(dp) :: accuracy
        integer(int64) :: clock_start, clock_end, clock_rate
        integer :: i, repetition
        type(xgboost_t) :: model
        type(xgboost_options_t) :: options
        type(fortnum_status_t) :: status

        call make_fixture(x, labels)
        do i = 1, n_samples
            labels(i) = merge(1.0_dp, 0.0_dp, &
                x(i, 1) + 0.2_dp*x(i, 2) >= 0.0_dp)
        end do
        options%n_estimators = n_estimators
        options%max_depth = 1
        options%min_samples_leaf = 2
        options%learning_rate = 0.25_dp
        options%l1 = 0.0_dp
        options%l2 = 1.0_dp
        options%gamma = 0.01_dp
        options%min_child_weight = 0.1_dp

        call system_clock(clock_start, clock_rate)
        do repetition = 1, repetitions
            call model%fit_binary(x, labels, status, options)
            if (.not. status_ok(status)) error stop "XGBoost logistic fit failed"
        end do
        call system_clock(clock_end)
        elapsed_fit = real(clock_end - clock_start, dp)/real(clock_rate, dp) &
            /real(repetitions, dp)
        call model%fit_binary(x, labels, status, options)
        if (.not. status_ok(status)) error stop "XGBoost logistic reference fit failed"
        call model%predict(x, probabilities, status)
        if (.not. status_ok(status)) error stop "XGBoost logistic prediction failed"
        call model%predict_proba(x, probability_matrix, status)
        if (.not. status_ok(status)) error stop "XGBoost probability prediction failed"
        logloss = -sum(labels*log(max(probabilities, 1.0e-15_dp)) + &
            (1.0_dp - labels)*log(max(1.0_dp - probabilities, 1.0e-15_dp))) &
            /real(n_samples, dp)
        accuracy = real(count((probabilities >= 0.5_dp) .eqv. &
            (labels >= 0.5_dp)), dp)/real(n_samples, dp)
        split_gain = model%split_gain(1)
        write (*, '(a,i0,a,i0,a,i0,a,es24.16,a,es24.16,a,es24.16,a,es24.16)') &
            "xgb_logistic_fit,", n_samples, ",", n_features, ",", &
            n_estimators, ",", elapsed_fit, ",", logloss, ",", split_gain, ",", &
            sum(probability_matrix(:, 2))

        call system_clock(clock_start, clock_rate)
        do repetition = 1, prediction_repetitions
            call model%predict(x, probabilities, status)
            if (.not. status_ok(status)) error stop "XGBoost logistic timing failed"
        end do
        call system_clock(clock_end)
        elapsed_predict = real(clock_end - clock_start, dp)/real(clock_rate, dp) &
            /real(prediction_repetitions, dp)
        write (*, '(a,i0,a,i0,a,i0,a,es24.16,a,es24.16,a,es24.16)') &
            "xgb_logistic_predict,", n_samples, ",", n_features, ",", &
            n_estimators, ",", elapsed_predict, ",", logloss, ",", accuracy
    end subroutine benchmark_logistic

    subroutine make_fixture(x, y)
        real(dp), intent(out) :: x(:, :), y(:)
        integer :: i

        do i = 1, size(x, 1)
            x(i, 1) = -1.0_dp + 2.0_dp*real(i - 1, dp)/ &
                real(size(x, 1) - 1, dp)
            x(i, 2) = sin(0.09_dp*real(i, dp))
            x(i, 3) = cos(0.04_dp*real(i, dp))
            y(i) = merge(1.5_dp + 0.25_dp*x(i, 2), &
                -0.7_dp + 0.12_dp*x(i, 3), x(i, 1) >= 0.08_dp)
        end do
    end subroutine make_fixture

end program fortml_bench_xgboost
