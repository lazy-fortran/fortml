program fortml_bench_xgboost_dart
    !! Correctness-gated release workload for bounded XGBoost DART.
    use, intrinsic :: iso_fortran_env, only: real64, int64
    use fortml_xgboost, only: xgboost_t, xgboost_options_t
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortnum_status, only: fortnum_status_t, FORTNUM_OK
    implicit none

    character(*), parameter :: snapshot = "fortml_bench_xgboost_dart.txt"
    real(real64) :: x(4, 1), target(4), prediction(4), replay(4), restored(4)
    real(real64) :: expected(4), started, finished
    real(real64) :: oracle_error, replay_error, restore_error, warm_error
    type(xgboost_t) :: model, repeated, restored_model, prefix, full
    type(xgboost_options_t) :: options, prefix_options, invalid
    type(fortnum_status_t) :: status
    type(fortml_device_t) :: cuda
    integer :: unit, ios

    x(:, 1) = [0.0_real64, 1.0_real64, 2.0_real64, 3.0_real64]
    target = [0.0_real64, 0.0_real64, 10.0_real64, 10.0_real64]
    ! Independent depth-one Newton/tree-walk oracle for l2=1, eta=1/2:
    ! DART scales are [1/4,1/2,1/2] under seed 1729, drop=.99, cap=1.
    expected = [3.0555555555555556_real64, 3.0555555555555556_real64, &
        6.9444444444444444_real64, 6.9444444444444444_real64]
    options = xgboost_options_t()
    options%n_estimators = 3
    options%max_depth = 1
    options%min_samples_leaf = 1
    options%learning_rate = 0.5_real64
    options%l2 = 1.0_real64
    options%booster = "dart"
    options%dart_drop_rate = 0.99_real64
    options%dart_max_drop = 1
    options%seed = 1729_int64

    call cpu_time(started)
    call model%fit_regression(x, target, status, options)
    call cpu_time(finished)
    if (status%code /= FORTNUM_OK) error stop "DART fit failed"
    write (*, '(a,es24.16)') "xgboost_dart_fit_seconds,", max(0.0_real64, finished-started)
    write (*, '(a,a)') "xgboost_dart_booster,", trim(model%booster())
    write (*, '(a,es24.16)') "xgboost_dart_drop_rate,", model%dart_drop_rate()
    write (*, '(a,i0)') "xgboost_dart_max_drop,", model%dart_max_drop()
    write (*, '(a,es24.16)') "xgboost_dart_scale_1,", model%tree_scale(1)
    write (*, '(a,es24.16)') "xgboost_dart_scale_2,", model%tree_scale(2)
    write (*, '(a,es24.16)') "xgboost_dart_scale_3,", model%tree_scale(3)

    call cpu_time(started)
    call model%predict(x, prediction, status)
    call cpu_time(finished)
    if (status%code /= FORTNUM_OK) error stop "DART prediction failed"
    oracle_error = maxval(abs(prediction-expected))
    write (*, '(a,4(es24.16,:,a))') "xgboost_dart_prediction,", prediction(1), ",", &
        prediction(2), ",", prediction(3), ",", prediction(4)
    write (*, '(a,es24.16)') "xgboost_dart_predict_seconds,", max(0.0_real64, finished-started)
    write (*, '(a,es24.16)') "xgboost_dart_oracle_error,", oracle_error

    call repeated%fit_regression(x, target, status, options)
    call repeated%predict(x, replay, status)
    if (status%code /= FORTNUM_OK) error stop "DART replay failed"
    replay_error = maxval(abs(replay-prediction))
    write (*, '(a,es24.16)') "xgboost_dart_replay_error,", replay_error

    call model%save_text(snapshot, status)
    if (status%code /= FORTNUM_OK) error stop "DART save failed"
    call restored_model%load_text(snapshot, status)
    if (status%code /= FORTNUM_OK) error stop "DART load failed"
    call restored_model%predict(x, restored, status)
    if (status%code /= FORTNUM_OK) error stop "DART restored prediction failed"
    restore_error = maxval(abs(restored-prediction))
    write (*, '(a,es24.16)') "xgboost_dart_restore_error,", restore_error

    prefix_options = options
    prefix_options%n_estimators = 2
    call prefix%fit_regression(x, target, status, prefix_options)
    full = prefix
    prefix_options%n_estimators = 3
    call prefix%fit_warm_start(x, target, status, prefix_options)
    if (status%code /= FORTNUM_OK) error stop "DART warm-start failed"
    call prefix%predict(x, replay, status)
    call full%fit_regression(x, target, status, options)
    call full%predict(x, restored, status)
    if (status%code /= FORTNUM_OK) error stop "DART warm-start prediction failed"
    warm_error = maxval(abs(replay-restored))
    write (*, '(a,es24.16)') "xgboost_dart_warm_start_error,", warm_error

    invalid = options
    invalid%dart_drop_rate = 1.0_real64
    call repeated%fit_regression(x, target, status, invalid)
    write (*, '(a,i0)') "xgboost_dart_invalid_status,", status%code

    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call model%predict_device(cuda, x, prediction, status)
    write (*, '(a,i0)') "xgboost_dart_cuda_status,", status%code

    open(newunit=unit, file=snapshot, status="old", action="read", iostat=ios)
    if (ios == 0) close(unit, status="delete")
end program fortml_bench_xgboost_dart
