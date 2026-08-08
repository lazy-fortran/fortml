program fortml_bench_lightgbm_dart
    !! Correctness-gated release workload for bounded LightGBM DART.
    use, intrinsic :: iso_fortran_env, only: real64
    use fortml_lightgbm, only: lightgbm_t, lightgbm_options_t
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortnum_status, only: fortnum_status_t, FORTNUM_OK
    implicit none

    character(*), parameter :: snapshot = "fortml_bench_lightgbm_dart.txt"
    real(real64) :: x(4, 1), target(4), prediction(4), replay(4), restored(4)
    real(real64) :: expected(4), started, finished
    real(real64) :: oracle_error, replay_error, restore_error, warm_error
    type(lightgbm_t) :: model, repeated, restored_model, prefix
    type(lightgbm_options_t) :: options, prefix_options, invalid
    type(fortnum_status_t) :: status
    type(fortml_device_t) :: cuda
    integer :: unit, ios

    x(:, 1) = [0.0_real64, 1.0_real64, 2.0_real64, 3.0_real64]
    target = [0.0_real64, 0.0_real64, 10.0_real64, 10.0_real64]
    ! Independent depth-one Newton/tree-walk oracle for l2=1, eta=1/2:
    ! DART scales are [1/4,1/2,1/2] under seed 1729, drop=.99, cap=1.
    expected = [3.0555555555555556_real64, 3.0555555555555556_real64, &
        6.9444444444444444_real64, 6.9444444444444444_real64]
    options = lightgbm_options_t()
    options%n_estimators = 3
    options%num_leaves = 2
    options%min_data_in_leaf = 1
    options%max_bin = 16
    options%learning_rate = 0.5_real64
    options%l2 = 1.0_real64
    options%boosting_type = "dart"
    options%dart_drop_rate = 0.99_real64
    options%dart_max_drop = 1
    options%seed = 1729

    call cpu_time(started)
    call model%fit_regression(x, target, status, options)
    call cpu_time(finished)
    if (status%code /= FORTNUM_OK) error stop "DART fit failed"
    write (*, '(a,es24.16)') "lightgbm_dart_fit_seconds,", max(0.0_real64, finished-started)
    write (*, '(a,a)') "lightgbm_dart_type,", trim(model%boosting_type())
    write (*, '(a,es24.16)') "lightgbm_dart_drop_rate,", model%dart_drop_rate()
    write (*, '(a,i0)') "lightgbm_dart_max_drop,", model%dart_max_drop()
    write (*, '(a,es24.16)') "lightgbm_dart_scale_1,", model%tree_scale(1)
    write (*, '(a,es24.16)') "lightgbm_dart_scale_2,", model%tree_scale(2)
    write (*, '(a,es24.16)') "lightgbm_dart_scale_3,", model%tree_scale(3)

    call cpu_time(started)
    call model%predict(x, prediction, status)
    call cpu_time(finished)
    if (status%code /= FORTNUM_OK) error stop "DART prediction failed"
    oracle_error = maxval(abs(prediction-expected))
    write (*, '(a,es24.16)') "lightgbm_dart_predict_seconds,", max(0.0_real64, finished-started)
    write (*, '(a,es24.16)') "lightgbm_dart_oracle_error,", oracle_error

    call repeated%fit_regression(x, target, status, options)
    call repeated%predict(x, replay, status)
    if (status%code /= FORTNUM_OK) error stop "DART replay failed"
    replay_error = maxval(abs(replay-prediction))
    write (*, '(a,es24.16)') "lightgbm_dart_replay_error,", replay_error

    call model%save_text(snapshot, status)
    if (status%code /= FORTNUM_OK) error stop "DART save failed"
    call restored_model%load_text(snapshot, status)
    if (status%code /= FORTNUM_OK) error stop "DART load failed"
    call restored_model%predict(x, restored, status)
    if (status%code /= FORTNUM_OK) error stop "DART restored prediction failed"
    restore_error = maxval(abs(restored-prediction))
    write (*, '(a,es24.16)') "lightgbm_dart_restore_error,", restore_error

    prefix_options = options
    prefix_options%n_estimators = 2
    call prefix%fit_regression(x, target, status, prefix_options)
    prefix_options%n_estimators = 3
    call prefix%fit_warm_start(x, target, status, prefix_options)
    if (status%code /= FORTNUM_OK) error stop "DART warm-start failed"
    call prefix%predict(x, replay, status)
    if (status%code /= FORTNUM_OK) error stop "DART warm-start prediction failed"
    warm_error = maxval(abs(replay-prediction))
    write (*, '(a,es24.16)') "lightgbm_dart_warm_start_error,", warm_error

    invalid = options
    invalid%dart_drop_rate = 1.0_real64
    call repeated%fit_regression(x, target, status, invalid)
    write (*, '(a,i0)') "lightgbm_dart_invalid_status,", status%code

    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call model%predict_device(cuda, x, prediction, status)
    write (*, '(a,i0)') "lightgbm_dart_cuda_status,", status%code

    open(newunit=unit, file=snapshot, status="old", action="read", iostat=ios)
    if (ios == 0) close(unit, status="delete")
end program fortml_bench_lightgbm_dart
