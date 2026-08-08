program fortml_bench_lightgbm_goss
    !! Correctness-gated release workload for LightGBM GOSS.
    use, intrinsic :: iso_fortran_env, only: real64
    use fortml_lightgbm, only: lightgbm_t, lightgbm_options_t
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortnum_status, only: fortnum_status_t, FORTNUM_OK
    implicit none

    real(real64) :: x(4, 1), target(4), prediction(4), replay(4)
    type(lightgbm_t) :: model, repeated
    type(lightgbm_options_t) :: options, invalid
    type(fortnum_status_t) :: status
    type(fortml_device_t) :: cuda
    real(real64) :: started, finished, oracle_error, replay_error

    x(:, 1) = [0.0_real64, 1.0_real64, 2.0_real64, 3.0_real64]
    target = [0.0_real64, 0.0_real64, 10.0_real64, 10.0_real64]
    options = lightgbm_options_t()
    options%n_estimators = 1
    options%num_leaves = 2
    options%min_data_in_leaf = 1
    options%max_bin = 16
    options%learning_rate = 1.0_real64
    options%l2 = 0.0_real64
    options%boosting_type = "goss"
    options%top_rate = 0.5_real64
    options%other_rate = 0.25_real64
    options%seed = 1729

    call cpu_time(started)
    call model%fit_regression(x, target, status, options)
    call cpu_time(finished)
    if (status%code /= FORTNUM_OK) error stop "GOSS fit failed"
    write (*, '(a,es24.16)') "lightgbm_goss_fit_seconds,", max(0.0_real64, finished-started)

    call cpu_time(started)
    call model%predict(x, prediction, status)
    call cpu_time(finished)
    if (status%code /= FORTNUM_OK) error stop "GOSS prediction failed"
    oracle_error = maxval(abs(prediction-target))
    write (*, '(a,es24.16)') "lightgbm_goss_predict_seconds,", max(0.0_real64, finished-started)
    write (*, '(a,es24.16)') "lightgbm_goss_oracle_error,", oracle_error
    write (*, '(a,a)') "lightgbm_goss_type,", trim(model%boosting_type())
    write (*, '(a,es24.16)') "lightgbm_goss_top_rate,", model%top_rate()
    write (*, '(a,es24.16)') "lightgbm_goss_other_rate,", model%other_rate()

    call repeated%fit_regression(x, target, status, options)
    call repeated%predict(x, replay, status)
    if (status%code /= FORTNUM_OK) error stop "GOSS replay failed"
    replay_error = maxval(abs(replay-prediction))
    write (*, '(a,es24.16)') "lightgbm_goss_replay_error,", replay_error

    invalid = options
    invalid%top_rate = 0.8_real64
    invalid%other_rate = 0.25_real64
    call repeated%fit_regression(x, target, status, invalid)
    write (*, '(a,i0)') "lightgbm_goss_invalid_status,", status%code

    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call model%predict_device(cuda, x, prediction, status)
    write (*, '(a,i0)') "lightgbm_goss_cuda_status,", status%code
end program fortml_bench_lightgbm_goss
