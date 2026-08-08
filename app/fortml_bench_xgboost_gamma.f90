program fortml_bench_xgboost_gamma
    !! Release workload for the bounded fixed-shape Gamma XGBoost objective.
    !!
    !! The app emits a deterministic one-tree log-link oracle and
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
    real(dp) :: expected(4), base, shape
    real(dp) :: gradient(4), hessian(4), left_weight, right_weight

    shape = 2.0_dp
    oracle_x(:, 1) = [0.0_dp, 1.0_dp, 2.0_dp, 3.0_dp]
    oracle_target = [1.0_dp, 1.0_dp, 9.0_dp, 9.0_dp]
    oracle_options%n_estimators = 1
    oracle_options%max_depth = 1
    oracle_options%learning_rate = 1.0_dp
    oracle_options%l2 = 0.0_dp
    oracle_options%min_child_weight = 0.0_dp
    oracle_options%gamma_shape = shape
    call model%fit_gamma(oracle_x, oracle_target, status, oracle_options)
    if (.not. status_ok(status)) error stop "Gamma oracle fit failed"
    call model%predict(oracle_x, oracle_prediction, status)
    if (.not. status_ok(status)) error stop "Gamma oracle prediction failed"
    base = log(5.0_dp)
    gradient = shape*(1.0_dp - oracle_target*exp(-base))
    hessian = shape*oracle_target*exp(-base)
    left_weight = -sum(gradient(:2))/(sum(hessian(:2)) + oracle_options%l2)
    right_weight = -sum(gradient(3:))/(sum(hessian(3:)) + oracle_options%l2)
    expected = exp([base + left_weight, base + left_weight, &
        base + right_weight, base + right_weight])
    write (*, '(a,es24.16)') "xgb_gamma_oracle_max_error,", &
        maxval(abs(oracle_prediction - expected))

    call make_fixture(x, target)
    options%n_estimators = n_estimators
    options%max_depth = 2
    options%min_samples_leaf = 2
    options%learning_rate = 0.25_dp
    options%l1 = 0.05_dp
    options%l2 = 1.0_dp
    options%min_child_weight = 0.1_dp
    options%gamma_shape = shape
    call system_clock(clock_start, clock_rate)
    do repetition = 1, repetitions
        call model%fit_gamma(x, target, status, options)
        if (.not. status_ok(status)) error stop "Gamma fit failed"
    end do
    call system_clock(clock_end)
    elapsed_fit = real(clock_end - clock_start, dp)/real(clock_rate, dp)/real(repetitions, dp)
    call model%fit_gamma(x, target, status, options)
    if (.not. status_ok(status)) error stop "Gamma reference fit failed"
    call model%predict(x, prediction, status)
    if (.not. status_ok(status)) error stop "Gamma prediction failed"
    loss = gamma_loss(target, prediction, shape)
    write (*, '(a,i0,a,i0,a,i0,a,es24.16,a,es24.16,a,es24.16)') &
        "xgb_gamma_fit,", n_samples, ",", n_features, ",", n_estimators, ",", &
        elapsed_fit, ",", loss, ",", sum(prediction)/real(n_samples, dp)

    call system_clock(clock_start, clock_rate)
    do repetition = 1, prediction_repetitions
        call model%predict(x, prediction, status)
        if (.not. status_ok(status)) error stop "Gamma timing failed"
    end do
    call system_clock(clock_end)
    elapsed_predict = real(clock_end - clock_start, dp)/real(clock_rate, dp)/ &
        real(prediction_repetitions, dp)
    write (*, '(a,i0,a,i0,a,i0,a,es24.16,a,es24.16,a,es24.16)') &
        "xgb_gamma_predict,", n_samples, ",", n_features, ",", n_estimators, ",", &
        elapsed_predict, ",", loss, ",", sum(prediction)/real(n_samples, dp)

    hist_options = options
    hist_options%tree_method = "hist"
    hist_options%max_bin = 16
    call model%fit_gamma(x, target, status, hist_options)
    if (.not. status_ok(status)) error stop "Gamma histogram fit failed"
    call model%predict(x, prediction, status)
    if (.not. status_ok(status)) error stop "Gamma histogram prediction failed"
    write (*, '(a,i0,a,i0,a,i0,a,es24.16,a,es24.16)') &
        "xgb_gamma_hist,", n_samples, ",", n_features, ",", n_estimators, ",", &
        gamma_loss(target, prediction, shape), ",", sum(prediction)/real(n_samples, dp)

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
            target(i) = 0.5_dp + real(mod(7*i + 3, 12), dp)
        end do
    end subroutine make_fixture

    real(dp) function gamma_loss(target, mean, shape) result(value)
        real(dp), intent(in) :: target(:), mean(:), shape
        integer :: i

        value = 0.0_dp
        do i = 1, size(target)
            value = value + shape*(log(max(mean(i), tiny(1.0_dp))) + &
                target(i)/max(mean(i), tiny(1.0_dp)))
        end do
        value = value/real(size(target), dp)
    end function gamma_loss

end program fortml_bench_xgboost_gamma
