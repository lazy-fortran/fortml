program fortml_bench_lightgbm_validation_warm_start
    !! Release-app protocol for validation-aware LightGBM continuation.
    use, intrinsic :: iso_fortran_env, only: real64
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortml_lightgbm, only: lightgbm_t, lightgbm_options_t
    use fortnum_status, only: fortnum_status_t, FORTNUM_OK
    implicit none

    integer, parameter :: n = 8
    real(real64) :: x(n, 1), target(n), validation_target(n), validation_weight(n)
    real(real64) :: prediction(n), before(n), after(n)
    type(lightgbm_t) :: prefix, retained, invalid
    type(lightgbm_options_t) :: prefix_options, warm_options
    type(fortnum_status_t) :: status
    type(fortml_device_t) :: cuda
    integer :: i, clock_start, clock_finish, clock_rate, invalid_status
    real(real64) :: elapsed, preservation_error

    x(:, 1) = real([(i, i = 0, n-1)], real64)
    target = [0.0_real64, 0.0_real64, 0.0_real64, 0.0_real64, &
        10.0_real64, 10.0_real64, 10.0_real64, 10.0_real64]
    validation_target = 10.0_real64-target
    validation_weight = [1.0_real64, 2.0_real64, 1.0_real64, 2.0_real64, &
        1.0_real64, 2.0_real64, 1.0_real64, 2.0_real64]
    prefix_options = lightgbm_options_t()
    prefix_options%n_estimators = 1
    prefix_options%num_leaves = 2
    prefix_options%min_data_in_leaf = 1
    prefix_options%max_bin = 16
    prefix_options%learning_rate = 1.0_real64
    prefix_options%l2 = 1.0_real64
    call prefix%fit_regression(x, target, status, prefix_options)
    if (status%code /= FORTNUM_OK) error stop "lightgbm warm prefix fit failed"

    warm_options = prefix_options
    warm_options%n_estimators = 4
    warm_options%early_stopping_rounds = 2
    warm_options%restore_best = .true.
    call system_clock(clock_start, clock_rate)
    call prefix%fit_warm_start(x, target, status, warm_options, &
        validation_x=x, validation_y=validation_target, &
        validation_weight=validation_weight)
    call system_clock(clock_finish)
    elapsed = real(clock_finish-clock_start, real64)/real(clock_rate, real64)
    if (status%code /= FORTNUM_OK) error stop "lightgbm restore-best warm fit failed"
    call prefix%predict(x, prediction, status)
    if (status%code /= FORTNUM_OK) error stop "lightgbm restore-best prediction failed"
    write (*, '(a,i0,a,i0,a,l1,a,es24.16,a,es24.16)') "lgbm_warm_restore,", &
        prefix%best_iteration(), ",", prefix%estimator_count(), ",", &
        prefix%early_stopped(), ",", prefix%best_validation_loss(), ",", elapsed

    call retained%fit_regression(x, target, status, prefix_options)
    if (status%code /= FORTNUM_OK) error stop "lightgbm retain prefix fit failed"
    warm_options%restore_best = .false.
    call system_clock(clock_start)
    call retained%fit_warm_start(x, target, status, warm_options, &
        validation_x=x, validation_y=validation_target, &
        validation_weight=validation_weight)
    call system_clock(clock_finish)
    elapsed = real(clock_finish-clock_start, real64)/real(clock_rate, real64)
    if (status%code /= FORTNUM_OK) error stop "lightgbm retain-all warm fit failed"
    write (*, '(a,i0,a,i0,a,l1,a,es24.16,a,es24.16)') "lgbm_warm_retain,", &
        retained%best_iteration(), ",", retained%estimator_count(), ",", &
        retained%early_stopped(), ",", retained%best_validation_loss(), ",", elapsed

    invalid = prefix
    call invalid%predict(x, before, status)
    call invalid%fit_warm_start(x, target, status, warm_options, validation_x=x, &
        validation_y=validation_target(:3))
    invalid_status = status%code
    call invalid%predict(x, after, status)
    preservation_error = maxval(abs(after-before))
    write (*, '(a,i0,a,i0,a,es24.16)') "lgbm_warm_invalid,", invalid_status, ",", &
        invalid%estimator_count(), ",", preservation_error

    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call prefix%predict_device(cuda, x, prediction, status)
    write (*, '(a,i0)') "lgbm_warm_cuda,", status%code
end program fortml_bench_lightgbm_validation_warm_start
