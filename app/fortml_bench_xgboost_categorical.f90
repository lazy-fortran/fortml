program fortml_bench_xgboost_categorical
    !! Release workload for ordered-gradient integer categorical partitions.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortml_xgboost, only: xgboost_t, xgboost_options_t
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer, parameter :: n_samples = 256, n_features = 2
    integer, parameter :: n_estimators = 1, repetitions = 16
    real(dp) :: x(n_samples, n_features), y(n_samples), prediction(n_samples)
    real(dp) :: elapsed_fit, elapsed_predict
    integer(int64) :: clock_start, clock_end, clock_rate
    integer :: repetition, i, category
    type(xgboost_t) :: model
    type(xgboost_options_t) :: options
    type(fortnum_status_t) :: status

    do i = 1, n_samples
        category = mod((i - 1)/64, 4)
        x(i, 1) = real(category, dp)
        x(i, 2) = real(mod(i - 1, 7), dp)
        if (category < 2) then
            y(i) = 0.0_dp
        else
            y(i) = 4.0_dp
        end if
    end do
    options = xgboost_options_t()
    options%n_estimators = n_estimators
    options%max_depth = 1
    options%learning_rate = 1.0_dp
    options%l2 = 0.0_dp
    options%min_child_weight = 0.0_dp
    options%categorical_policy = "ordered"
    options%categorical_max_categories = 4
    options%categorical_features = [1]

    call system_clock(clock_start, clock_rate)
    do repetition = 1, repetitions
        call model%fit_regression(x, y, status, options)
        if (.not. status_ok(status)) error stop "categorical fit failed"
    end do
    call system_clock(clock_end)
    elapsed_fit = real(clock_end - clock_start, dp)/real(clock_rate, dp)/real(repetitions, dp)
    call model%predict(x, prediction, status)
    if (.not. status_ok(status)) error stop "categorical prediction failed"
    write (*, '(a,i0,a,i0,a,i0,a,es24.16,a,i0,a,es24.16)') &
        "xgb_categorical_fit,", n_samples, ",", n_features, ",", n_estimators, ",", &
        elapsed_fit, ",", model%tree_node_count(1), ",", maxval(abs(prediction - y))

    call system_clock(clock_start, clock_rate)
    do repetition = 1, repetitions*8
        call model%predict(x, prediction, status)
        if (.not. status_ok(status)) error stop "categorical prediction failed"
    end do
    call system_clock(clock_end)
    elapsed_predict = real(clock_end - clock_start, dp)/real(clock_rate, dp)/real(repetitions*8, dp)
    call model%predict(x, prediction, status)
    write (*, '(a,*(1x,es24.16))') "xgb_categorical_values", prediction
    write (*, '(a,es24.16)') "xgb_categorical_predict_seconds ", elapsed_predict
end program fortml_bench_xgboost_categorical
