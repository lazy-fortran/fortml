program fortml_bench_xgboost_poisson
    !! Release workload for the Poisson XGBoost objective.
    !!
    !! The benchmark reports a deterministic count fixture, fit and prediction
    !! timings, and Poisson deviance. Correctness is checked independently by
    !! scripts/bench_xgboost_poisson.py; this executable only measures the
    !! release build and never labels the CPU path as CUDA.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortml_xgboost, only: xgboost_t, xgboost_options_t
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer, parameter :: n_samples = 256, n_features = 3
    integer, parameter :: n_estimators = 16, repetitions = 3
    integer, parameter :: prediction_repetitions = 24
    real(dp) :: x(n_samples, n_features), target(n_samples), prediction(n_samples)
    real(dp) :: elapsed_fit, elapsed_predict, deviance
    integer(int64) :: clock_start, clock_end, clock_rate
    integer :: i, repetition
    type(xgboost_t) :: model
    type(xgboost_options_t) :: options, hist_options
    type(fortnum_status_t) :: status
    real(dp) :: oracle_x(4, 1), oracle_target(4), oracle_prediction(4)
    type(xgboost_options_t) :: oracle_options

    call make_fixture(x, target)
    oracle_x(:, 1) = [0.0_dp, 1.0_dp, 2.0_dp, 3.0_dp]
    oracle_target = [1.0_dp, 1.0_dp, 9.0_dp, 9.0_dp]
    oracle_options%n_estimators = 1
    oracle_options%max_depth = 1
    oracle_options%learning_rate = 1.0_dp
    oracle_options%l2 = 0.0_dp
    oracle_options%min_child_weight = 0.0_dp
    call model%fit_poisson(oracle_x, oracle_target, status, oracle_options)
    if (.not. status_ok(status)) error stop "Poisson oracle fit failed"
    call model%predict(oracle_x, oracle_prediction, status)
    if (.not. status_ok(status)) error stop "Poisson oracle prediction failed"
    write (*, '(a,es24.16)') "xgb_poisson_oracle_max_error,", &
        maxval(abs(oracle_prediction - [5.0_dp*exp(-0.8_dp), &
            5.0_dp*exp(-0.8_dp), 5.0_dp*exp(0.8_dp), 5.0_dp*exp(0.8_dp)]))

    options%n_estimators = n_estimators
    options%max_depth = 2
    options%min_samples_leaf = 2
    options%learning_rate = 0.25_dp
    options%l1 = 0.05_dp
    options%l2 = 1.0_dp
    options%gamma = 0.0_dp
    options%min_child_weight = 0.1_dp

    call system_clock(clock_start, clock_rate)
    do repetition = 1, repetitions
        call model%fit_poisson(x, target, status, options)
        if (.not. status_ok(status)) error stop "Poisson fit failed"
    end do
    call system_clock(clock_end)
    elapsed_fit = real(clock_end - clock_start, dp)/real(clock_rate, dp) &
        /real(repetitions, dp)
    call model%fit_poisson(x, target, status, options)
    if (.not. status_ok(status)) error stop "Poisson reference fit failed"
    call model%predict(x, prediction, status)
    if (.not. status_ok(status)) error stop "Poisson prediction failed"
    deviance = poisson_deviance(target, prediction)
    write (*, '(a,i0,a,i0,a,i0,a,es24.16,a,es24.16,a,es24.16)') &
        "xgb_poisson_fit,", n_samples, ",", n_features, ",", n_estimators, ",", &
        elapsed_fit, ",", deviance, ",", sum(prediction)/real(n_samples, dp)

    call system_clock(clock_start, clock_rate)
    do repetition = 1, prediction_repetitions
        call model%predict(x, prediction, status)
        if (.not. status_ok(status)) error stop "Poisson timing failed"
    end do
    call system_clock(clock_end)
    elapsed_predict = real(clock_end - clock_start, dp)/real(clock_rate, dp) &
        /real(prediction_repetitions, dp)
    write (*, '(a,i0,a,i0,a,i0,a,es24.16,a,es24.16,a,es24.16)') &
        "xgb_poisson_predict,", n_samples, ",", n_features, ",", n_estimators, ",", &
        elapsed_predict, ",", deviance, ",", sum(prediction)/real(n_samples, dp)

    hist_options = options
    hist_options%tree_method = "hist"
    hist_options%max_bin = 16
    call model%fit_poisson(x, target, status, hist_options)
    if (.not. status_ok(status)) error stop "Poisson histogram fit failed"
    call model%predict(x, prediction, status)
    if (.not. status_ok(status)) error stop "Poisson histogram prediction failed"
    write (*, '(a,i0,a,i0,a,i0,a,es24.16,a,es24.16)') &
        "xgb_poisson_hist,", n_samples, ",", n_features, ",", n_estimators, ",", &
        poisson_deviance(target, prediction), ",", &
        sum(prediction)/real(n_samples, dp)

contains

    subroutine make_fixture(x, target)
        real(dp), intent(out) :: x(:, :), target(:)
        integer :: i
        real(dp) :: z

        do i = 1, size(x, 1)
            z = real(i - 1, dp)/real(size(x, 1) - 1, dp)
            x(i, 1) = -1.0_dp + 2.0_dp*z
            x(i, 2) = sin(0.11_dp*real(i, dp))
            x(i, 3) = cos(0.07_dp*real(i, dp))
            target(i) = real(mod(7*i + 3, 12), dp)
        end do
    end subroutine make_fixture

    real(dp) function poisson_deviance(target, mean) result(value)
        real(dp), intent(in) :: target(:), mean(:)
        integer :: i
        real(dp) :: term

        value = 0.0_dp
        do i = 1, size(target)
            if (target(i) == 0.0_dp) then
                term = mean(i)
            else
                term = target(i)*log(target(i)/mean(i)) - (target(i) - mean(i))
            end if
            value = value + 2.0_dp*term
        end do
        value = value/real(size(target), dp)
    end function poisson_deviance

end program fortml_bench_xgboost_poisson
