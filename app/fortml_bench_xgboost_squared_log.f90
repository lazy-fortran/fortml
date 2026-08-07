program fortml_bench_xgboost_squared_log
    !! Correctness-gated XGBoost squared-log/RMSLE workload.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortml_xgboost, only: xgboost_t, xgboost_options_t
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    implicit none

    integer, parameter :: n_samples = 256, n_features = 3
    integer, parameter :: n_estimators = 16, fit_repetitions = 3
    integer, parameter :: predict_repetitions = 24
    real(dp) :: x(n_samples, n_features), target(n_samples), prediction(n_samples)
    real(dp) :: oracle_x(4, 1), oracle_target(4), oracle_prediction(4)
    real(dp) :: expected_oracle(4), elapsed_fit, elapsed_predict
    real(dp) :: hist_prediction(n_samples), hist_error
    integer(int64) :: clock_start, clock_end, clock_rate
    integer :: repetition
    type(xgboost_t) :: model, hist_model
    type(xgboost_options_t) :: options, oracle_options, hist_options
    type(fortnum_status_t) :: status
    type(fortml_device_t) :: cuda

    call make_fixture(x, target)

    oracle_x(:, 1) = [0.0_dp, 1.0_dp, 2.0_dp, 3.0_dp]
    oracle_target = [0.0_dp, 0.0_dp, 3.0_dp, 3.0_dp]
    oracle_options%n_estimators = 1
    oracle_options%max_depth = 1
    oracle_options%learning_rate = 1.0_dp
    oracle_options%l2 = 0.7_dp
    oracle_options%min_child_weight = 0.0_dp
    call model%fit_squared_log(oracle_x, oracle_target, status, oracle_options)
    if (.not. status_ok(status)) error stop "squared-log oracle fit failed"
    call model%predict(oracle_x, oracle_prediction, status)
    if (.not. status_ok(status)) error stop "squared-log oracle prediction failed"
    call independent_oracle(oracle_target, oracle_options%l2, expected_oracle)
    write (*, '(a,es24.16)') "xgb_squared_log_oracle_max_error,", &
        maxval(abs(oracle_prediction - expected_oracle))

    options%n_estimators = n_estimators
    options%max_depth = 2
    options%min_samples_leaf = 2
    options%learning_rate = 0.25_dp
    options%l1 = 0.05_dp
    options%l2 = 1.0_dp
    options%min_child_weight = 0.1_dp

    call system_clock(clock_start, clock_rate)
    do repetition = 1, fit_repetitions
        call model%fit_squared_log(x, target, status, options)
        if (.not. status_ok(status)) error stop "squared-log fit failed"
    end do
    call system_clock(clock_end)
    elapsed_fit = real(clock_end - clock_start, dp)/real(clock_rate, dp) &
        /real(fit_repetitions, dp)
    call model%fit_squared_log(x, target, status, options)
    if (.not. status_ok(status)) error stop "squared-log reference fit failed"
    call model%predict(x, prediction, status)
    if (.not. status_ok(status)) error stop "squared-log prediction failed"
    write (*, '(a,i0,a,i0,a,i0,a,es24.16,a,es24.16)') &
        "xgb_squared_log_fit,", n_samples, ",", n_features, ",", &
        n_estimators, ",", elapsed_fit, ",", sum(prediction)/real(n_samples, dp)

    call system_clock(clock_start, clock_rate)
    do repetition = 1, predict_repetitions
        call model%predict(x, prediction, status)
        if (.not. status_ok(status)) error stop "squared-log timing failed"
    end do
    call system_clock(clock_end)
    elapsed_predict = real(clock_end - clock_start, dp)/real(clock_rate, dp) &
        /real(predict_repetitions, dp)
    write (*, '(a,i0,a,i0,a,i0,a,es24.16,a,es24.16)') &
        "xgb_squared_log_predict,", n_samples, ",", n_features, ",", &
        n_estimators, ",", elapsed_predict, ",", sum(prediction)/real(n_samples, dp)

    hist_options = options
    hist_options%tree_method = "hist"
    hist_options%max_bin = 16
    call hist_model%fit_squared_log(x, target, status, hist_options)
    if (.not. status_ok(status)) error stop "squared-log histogram fit failed"
    call hist_model%predict(x, hist_prediction, status)
    if (.not. status_ok(status)) error stop "squared-log histogram prediction failed"
    hist_error = maxval(abs(hist_prediction - prediction))
    write (*, '(a,es24.16)') "xgb_squared_log_hist_max_error,", hist_error

    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call model%predict_device(cuda, x, prediction, status)
    if (status%code /= FORTNUM_NOT_IMPLEMENTED) then
        error stop "squared-log XGBoost CUDA contract changed unexpectedly"
    end if
    write (*, '(a)') "xgb_squared_log_cuda,unavailable"

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
            target(i) = exp(0.45_dp + 0.9_dp*z + 0.15_dp*sin(0.23_dp*real(i, dp))) - 1.0_dp
        end do
    end subroutine make_fixture

    subroutine independent_oracle(target, l2, expected)
        real(dp), intent(in) :: target(:), l2
        real(dp), intent(out) :: expected(:)
        real(dp) :: transformed(4), residual(4), gradient(4), hessian(4)
        real(dp) :: base, left_correction, right_correction

        transformed = log(1.0_dp + target)
        base = sum(transformed)/real(size(target), dp)
        residual = base - transformed
        gradient = residual/exp(base)
        hessian = max((1.0_dp - residual)/exp(base), 1.0e-12_dp)
        left_correction = -sum(gradient(1:2))/(sum(hessian(1:2)) + l2)
        right_correction = -sum(gradient(3:4))/(sum(hessian(3:4)) + l2)
        expected = exp([base + left_correction, base + left_correction, &
            base + right_correction, base + right_correction]) - 1.0_dp
    end subroutine independent_oracle

end program fortml_bench_xgboost_squared_log
