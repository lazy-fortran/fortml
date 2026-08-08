program fortml_bench_xgboost_tweedie
    !! Release workload for the bounded XGBoost Tweedie objective.
    !!
    !! The app emits a deterministic one-tree compound-Poisson oracle and
    !! exact/histogram CPU timings.  The Python harness reconstructs the
    !! objective independently and records the explicit CUDA refusal.
    use, intrinsic :: iso_fortran_env, only: dp => real64
    use fortml_xgboost, only: xgboost_t, xgboost_options_t
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer, parameter :: n_samples = 256, n_features = 3
    integer, parameter :: n_estimators = 16, repetitions = 3
    integer, parameter :: prediction_repetitions = 24
    real(dp) :: x(n_samples, n_features), target(n_samples), prediction(n_samples)
    real(dp) :: elapsed_fit, elapsed_predict, loss
    integer :: clock_start, clock_end, clock_rate, repetition
    type(xgboost_t) :: model
    type(xgboost_options_t) :: options, hist_options, oracle_options
    type(fortnum_status_t) :: status
    real(dp) :: oracle_x(4, 1), oracle_target(4), oracle_prediction(4)
    real(dp) :: expected(4), base, power, first_term, second_term
    real(dp) :: gradient(4), hessian(4), left_weight, right_weight

    power = 1.5_dp
    oracle_x(:, 1) = [0.0_dp, 1.0_dp, 2.0_dp, 3.0_dp]
    oracle_target = [1.0_dp, 1.0_dp, 9.0_dp, 9.0_dp]
    oracle_options%n_estimators = 1
    oracle_options%max_depth = 1
    oracle_options%learning_rate = 1.0_dp
    oracle_options%l2 = 0.0_dp
    oracle_options%min_child_weight = 0.0_dp
    oracle_options%tweedie_variance_power = power
    call model%fit_tweedie(oracle_x, oracle_target, status, oracle_options)
    if (.not. status_ok(status)) error stop "Tweedie oracle fit failed"
    call model%predict(oracle_x, oracle_prediction, status)
    if (.not. status_ok(status)) error stop "Tweedie oracle prediction failed"
    base = log(5.0_dp)
    first_term = exp((1.0_dp - power)*base)
    second_term = exp((2.0_dp - power)*base)
    gradient = -oracle_target*first_term + second_term
    hessian = oracle_target*(power - 1.0_dp)*first_term + &
        (2.0_dp - power)*second_term
    left_weight = -sum(gradient(:2))/(sum(hessian(:2)) + oracle_options%l2)
    right_weight = -sum(gradient(3:))/(sum(hessian(3:)) + oracle_options%l2)
    expected = exp([base + left_weight, base + left_weight, &
        base + right_weight, base + right_weight])
    write (*, '(a,es24.16)') "xgb_tweedie_oracle_max_error,", &
        maxval(abs(oracle_prediction - expected))

    call make_fixture(x, target)
    options%n_estimators = n_estimators
    options%max_depth = 2
    options%min_samples_leaf = 2
    options%learning_rate = 0.25_dp
    options%l1 = 0.05_dp
    options%l2 = 1.0_dp
    options%min_child_weight = 0.1_dp
    options%tweedie_variance_power = power
    call system_clock(clock_start, clock_rate)
    do repetition = 1, repetitions
        call model%fit_tweedie(x, target, status, options)
        if (.not. status_ok(status)) error stop "Tweedie fit failed"
    end do
    call system_clock(clock_end)
    elapsed_fit = real(clock_end - clock_start, dp)/real(clock_rate, dp)/real(repetitions, dp)
    call model%fit_tweedie(x, target, status, options)
    if (.not. status_ok(status)) error stop "Tweedie reference fit failed"
    call model%predict(x, prediction, status)
    if (.not. status_ok(status)) error stop "Tweedie prediction failed"
    loss = tweedie_loss(target, prediction, power)
    write (*, '(a,i0,a,i0,a,i0,a,es24.16,a,es24.16,a,es24.16)') &
        "xgb_tweedie_fit,", n_samples, ",", n_features, ",", n_estimators, ",", &
        elapsed_fit, ",", loss, ",", sum(prediction)/real(n_samples, dp)

    call system_clock(clock_start, clock_rate)
    do repetition = 1, prediction_repetitions
        call model%predict(x, prediction, status)
        if (.not. status_ok(status)) error stop "Tweedie timing failed"
    end do
    call system_clock(clock_end)
    elapsed_predict = real(clock_end - clock_start, dp)/real(clock_rate, dp)/ &
        real(prediction_repetitions, dp)
    write (*, '(a,i0,a,i0,a,i0,a,es24.16,a,es24.16,a,es24.16)') &
        "xgb_tweedie_predict,", n_samples, ",", n_features, ",", n_estimators, ",", &
        elapsed_predict, ",", loss, ",", sum(prediction)/real(n_samples, dp)

    hist_options = options
    hist_options%tree_method = "hist"
    hist_options%max_bin = 16
    call model%fit_tweedie(x, target, status, hist_options)
    if (.not. status_ok(status)) error stop "Tweedie histogram fit failed"
    call model%predict(x, prediction, status)
    if (.not. status_ok(status)) error stop "Tweedie histogram prediction failed"
    write (*, '(a,i0,a,i0,a,i0,a,es24.16,a,es24.16)') &
        "xgb_tweedie_hist,", n_samples, ",", n_features, ",", n_estimators, ",", &
        tweedie_loss(target, prediction, power), ",", sum(prediction)/real(n_samples, dp)

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

    real(dp) function tweedie_loss(target, mean, variance_power) result(value)
        real(dp), intent(in) :: target(:), mean(:), variance_power
        integer :: i
        real(dp) :: margin

        value = 0.0_dp
        do i = 1, size(target)
            margin = log(max(mean(i), tiny(1.0_dp)))
            value = value + target(i)*exp((1.0_dp - variance_power)*margin)/ &
                (variance_power - 1.0_dp) + &
                exp((2.0_dp - variance_power)*margin)/(2.0_dp - variance_power)
        end do
        value = value/real(size(target), dp)
    end function tweedie_loss

end program fortml_bench_xgboost_tweedie
