program fortml_bench_xgboost_classifier
    !! Release workload for the public binary XGBoost classifier facade.
    !!
    !! Rows contain deterministic CPU fit/predict timings and an independent
    !! probability/label invariant.  CUDA is intentionally reported as a
    !! typed refusal until a resident tree kernel is linked.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortml_xgboost, only: xgboost_options_t
    use fortml_xgboost_classifier, only: xgboost_classifier_t
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, &
        FORTML_DEVICE_CUDA
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    implicit none

    integer, parameter :: n_samples = 192, n_features = 3
    integer, parameter :: n_estimators = 12, fit_repetitions = 4
    integer, parameter :: predict_repetitions = 32
    real(dp) :: x(n_samples, n_features), probabilities(n_samples, 2)
    real(dp) :: staged(n_samples, 2, n_estimators), importance(n_features)
    real(dp) :: weights(n_samples), positive_logloss, accuracy
    real(dp) :: fit_seconds, predict_seconds, invariant_error
    integer :: labels(n_samples), predicted(n_samples), i, repetition
    integer(int64) :: clock_start, clock_end, clock_rate
    type(xgboost_classifier_t) :: model
    type(xgboost_options_t) :: options
    type(fortnum_status_t) :: status
    type(fortml_device_t) :: cuda

    do i = 1, n_samples
        x(i, 1) = -3.0_dp + 6.0_dp*real(i - 1, dp)/real(n_samples - 1, dp)
        x(i, 2) = sin(0.7_dp*real(i, dp))
        x(i, 3) = cos(0.37_dp*real(i, dp))
        labels(i) = merge(17, -5, x(i, 1) + 0.25_dp*x(i, 2) >= 0.0_dp)
        weights(i) = 1.0_dp + 0.5_dp*real(mod(i, 4), dp)
    end do
    options%n_estimators = n_estimators
    options%max_depth = 2
    options%min_samples_leaf = 2
    options%learning_rate = 0.25_dp
    options%l2 = 1.0_dp
    options%gamma = 0.01_dp
    options%min_child_weight = 0.1_dp
    options%seed = 1729_int64

    call system_clock(clock_start, clock_rate)
    do repetition = 1, fit_repetitions
        call model%fit(x, labels, status, options, sample_weight=weights)
        if (.not. status_ok(status)) error stop "XGBoost classifier fit failed"
    end do
    call system_clock(clock_end)
    fit_seconds = real(clock_end - clock_start, dp)/real(clock_rate, dp)/ &
        real(fit_repetitions, dp)

    call model%fit(x, labels, status, options, sample_weight=weights)
    if (.not. status_ok(status)) error stop "XGBoost classifier reference fit failed"
    call model%predict_proba(x, probabilities, status)
    if (.not. status_ok(status)) error stop "XGBoost classifier probability failed"
    call model%predict(x, predicted, status)
    if (.not. status_ok(status)) error stop "XGBoost classifier label failed"
    call model%predict_proba_staged(x, staged, status)
    if (.not. status_ok(status)) error stop "XGBoost classifier staged failed"
    call model%feature_importance(importance, status, "gain", .true.)
    if (.not. status_ok(status)) error stop "XGBoost classifier importance failed"
    positive_logloss = -sum(log(max(probabilities(:, 1), 1.0e-15_dp)), &
        mask=labels == -5) - sum(log(max(probabilities(:, 2), 1.0e-15_dp)), &
        mask=labels == 17)
    positive_logloss = positive_logloss/real(n_samples, dp)
    accuracy = real(count(predicted == labels), dp)/real(n_samples, dp)
    invariant_error = maxval(abs(staged(:, :, n_estimators) - probabilities))
    write (*, '(a,i0,a,i0,a,i0,a,es24.16,a,es24.16,a,es24.16,a,es24.16)') &
        "xgb_classifier_fit,", n_samples, ",", n_features, ",", n_estimators, &
        ",", fit_seconds, ",", positive_logloss, ",", accuracy, ",", &
        invariant_error

    call system_clock(clock_start, clock_rate)
    do repetition = 1, predict_repetitions
        call model%predict_proba(x, probabilities, status)
        if (.not. status_ok(status)) error stop "XGBoost classifier prediction timing failed"
    end do
    call system_clock(clock_end)
    predict_seconds = real(clock_end - clock_start, dp)/real(clock_rate, dp)/ &
        real(predict_repetitions, dp)
    write (*, '(a,i0,a,i0,a,i0,a,es24.16,a,es24.16,a,es24.16)') &
        "xgb_classifier_predict,", n_samples, ",", n_features, ",", &
        n_estimators, ",", predict_seconds, ",", accuracy, ",", sum(importance)

    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call model%predict_proba_device(cuda, x, probabilities, status)
    if (status%code /= FORTNUM_NOT_IMPLEMENTED) error stop &
        "XGBoost classifier CUDA contract changed unexpectedly"
    write (*, '(a,i0)') "xgb_classifier_cuda_refusal,", status%code
end program fortml_bench_xgboost_classifier
