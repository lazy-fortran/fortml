program fortml_bench_xgboost_interaction
    !! Release workload for path-local XGBoost interaction constraints.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortml_xgboost, only: xgboost_t, xgboost_options_t
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer, parameter :: n_samples = 192, n_features = 3
    integer, parameter :: n_estimators = 8, repetitions = 8
    real(dp) :: x(n_samples, n_features), y(n_samples), prediction(n_samples)
    real(dp) :: elapsed_fit, elapsed_predict
    integer(int64) :: clock_start, clock_end, clock_rate
    integer :: repetition, i
    type(xgboost_t) :: unconstrained, constrained
    type(xgboost_options_t) :: unconstrained_options, constrained_options
    type(fortnum_status_t) :: status

    do i = 1, n_samples
        x(i, 1) = real((i - 1)/96, dp)
        x(i, 2) = real(mod(i - 1, 4), dp)
        x(i, 3) = 0.0_dp
        y(i) = 10.0_dp*x(i, 1) + 4.0_dp*floor(x(i, 2)/2.0_dp)
    end do
    unconstrained_options = xgboost_options_t()
    unconstrained_options%n_estimators = n_estimators
    unconstrained_options%max_depth = 2
    unconstrained_options%learning_rate = 1.0_dp
    unconstrained_options%l2 = 0.0_dp
    unconstrained_options%min_child_weight = 0.0_dp
    constrained_options = unconstrained_options
    constrained_options%interaction_groups = [1, 2, 0]

    call system_clock(clock_start, clock_rate)
    do repetition = 1, repetitions
        call unconstrained%fit_regression(x, y, status, unconstrained_options)
        if (.not. status_ok(status)) error stop "unconstrained interaction fit failed"
    end do
    call system_clock(clock_end)
    elapsed_fit = real(clock_end - clock_start, dp)/real(clock_rate, dp)/real(repetitions, dp)
    call unconstrained%predict(x, prediction, status)
    if (.not. status_ok(status)) error stop "unconstrained interaction prediction failed"
    write (*, '(a,i0,a,i0,a,i0,a,es24.16,a,i0,a,es24.16)') &
        "xgb_interaction_unconstrained_fit,", n_samples, ",", n_features, ",", &
        n_estimators, ",", elapsed_fit, ",", unconstrained%tree_node_count(1), ",", &
        maxval(abs(prediction - y))

    call system_clock(clock_start, clock_rate)
    do repetition = 1, repetitions
        call constrained%fit_regression(x, y, status, constrained_options)
        if (.not. status_ok(status)) error stop "constrained interaction fit failed"
    end do
    call system_clock(clock_end)
    elapsed_fit = real(clock_end - clock_start, dp)/real(clock_rate, dp)/real(repetitions, dp)
    call system_clock(clock_start, clock_rate)
    do repetition = 1, repetitions*8
        call constrained%predict(x, prediction, status)
        if (.not. status_ok(status)) error stop "constrained interaction prediction failed"
    end do
    call system_clock(clock_end)
    elapsed_predict = real(clock_end - clock_start, dp)/real(clock_rate, dp)/real(repetitions*8, dp)
    write (*, '(a,i0,a,i0,a,i0,a,es24.16,a,i0,a,es24.16)') &
        "xgb_interaction_constrained_fit,", n_samples, ",", n_features, ",", &
        n_estimators, ",", elapsed_fit, ",", constrained%tree_node_count(1), ",", &
        maxval(abs(prediction - y))
    call constrained%predict(x, prediction, status)
    write (*, '(a,*(1x,es24.16))') "xgb_interaction_constrained_values", prediction
    write (*, '(a,es24.16)') "xgb_interaction_predict_seconds ", elapsed_predict
end program fortml_bench_xgboost_interaction
