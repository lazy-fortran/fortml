program test_xgboost_cuda_policy
    !! Independent device-policy oracle for the bounded XGBoost resident path.
    !! Numeric finite gbtree models may execute on native CUDA, while every
    !! discrete or unsupported policy must refuse before touching outputs.
    use, intrinsic :: iso_fortran_env, only: real64, error_unit
    use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_quiet_nan
    use fortml_xgboost, only: xgboost_t, xgboost_options_t
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortnum_status, only: fortnum_status_t, FORTNUM_OK, FORTNUM_NOT_IMPLEMENTED
    implicit none

    type(fortml_device_t) :: cuda
    type(fortnum_status_t) :: status
    type(xgboost_t) :: model
    type(xgboost_options_t) :: options
    real(real64) :: x(8, 2), y(8), prediction(8), host_prediction(8)
    real(real64) :: x_missing(8, 2), y_missing(8)
    integer :: i, failures
    integer :: group(6)
    real(real64) :: ranking_x(6, 1), relevance(6)

    failures = 0
    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.

    do i = 1, 8
        x(i, 1) = real(i - 1, real64) / 7.0_real64
        x(i, 2) = 1.0_real64 - x(i, 1)
        y(i) = merge(-1.0_real64, 2.0_real64, i <= 4)
    end do
    options%n_estimators = 3
    options%max_depth = 2
    options%learning_rate = 0.4_real64
    options%min_child_weight = 0.0_real64
    call model%fit_regression(x, y, status, options)
    call check(status%code == FORTNUM_OK, "finite numeric fit", failures)
    call model%predict(x, host_prediction, status)
    call check(status%code == FORTNUM_OK, "finite numeric host prediction", failures)
    prediction = -huge(1.0_real64)
    call model%predict_device(cuda, x, prediction, status)
    if (status%code == FORTNUM_OK) then
        call check(model%device_supported(FORTML_DEVICE_CUDA), &
            "native numeric CUDA capability", failures)
        call check(maxval(abs(prediction - host_prediction)) < 2.0e-11_real64, &
            "numeric CUDA/CPU prediction parity", failures)
    else
        call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
            "numeric unavailable plan is typed", failures)
        call check(all(prediction == -huge(1.0_real64)), &
            "numeric refusal preserves output", failures)
    end if

    options%categorical_policy = "ordered"
    allocate(options%categorical_features(1))
    options%categorical_features(1) = 1
    options%categorical_max_categories = 4
    options%n_estimators = 1
    options%max_depth = 1
    options%learning_rate = 1.0_real64
    options%l2 = 0.0_real64
    do i = 1, 8
        x(i, 1) = real((i - 1) / 2, real64)
        x(i, 2) = 0.0_real64
        y(i) = merge(0.0_real64, 4.0_real64, i <= 4)
    end do
    prediction = -huge(1.0_real64)
    call model%fit(x, y, status, options)
    call check(status%code == FORTNUM_OK, "categorical fit", failures)
    call model%predict_device(cuda, x, prediction, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "categorical CUDA refusal", failures)
    call check(all(prediction == -huge(1.0_real64)), &
        "categorical refusal preserves output", failures)
    deallocate(options%categorical_features)

    options%categorical_policy = "none"
    options%missing_policy = "learn"
    x_missing = x
    y_missing = y
    x_missing(2, 2) = ieee_value(0.0_real64, ieee_quiet_nan)
    prediction = -huge(1.0_real64)
    call model%fit(x_missing, y_missing, status, options)
    call check(status%code == FORTNUM_OK, "missing-default fit", failures)
    call model%predict_device(cuda, x_missing, prediction, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "missing-default CUDA refusal", failures)
    call check(all(prediction == -huge(1.0_real64)), &
        "missing-default refusal preserves output", failures)

    options%missing_policy = "error"
    options%booster = "dart"
    options%n_estimators = 3
    prediction = -huge(1.0_real64)
    call model%fit(x, y, status, options)
    call check(status%code == FORTNUM_OK, "DART fit", failures)
    call model%predict_device(cuda, x, prediction, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, "DART CUDA refusal", failures)
    call check(all(prediction == -huge(1.0_real64)), &
        "DART refusal preserves output", failures)

    ranking_x(:, 1) = [0.0_real64, 1.0_real64, 2.0_real64, 3.0_real64, 4.0_real64, 5.0_real64]
    relevance = [0.0_real64, 1.0_real64, 0.0_real64, 2.0_real64, 1.0_real64, 0.0_real64]
    group = [1, 1, 2, 2, 3, 3]
    options%booster = "gbtree"
    options%n_estimators = 2
    prediction(1:6) = -huge(1.0_real64)
    call model%fit_ranking(ranking_x, relevance, group, status, options)
    call check(status%code == FORTNUM_OK, "ranking fit", failures)
    call model%predict_device(cuda, ranking_x, prediction(1:6), status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, "ranking CUDA refusal", failures)
    call check(all(prediction(1:6) == -huge(1.0_real64)), &
        "ranking refusal preserves output", failures)

    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, " XGBoost CUDA policy test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS XGBoost resident CUDA policy/refusal oracle"

contains

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failures
        if (.not. condition) then
            write (error_unit, '(a)') "FAIL ["//trim(description)//"]"
            failures = failures + 1
        end if
    end subroutine check

end program test_xgboost_cuda_policy
