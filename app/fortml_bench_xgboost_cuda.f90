program fortml_bench_xgboost_cuda
    !! Release probe for the finite numeric XGBoost resident CUDA contract.
    !! Unsupported tree policies are expected to refuse without host fallback.
    use, intrinsic :: iso_fortran_env, only: real64
    use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_quiet_nan
    use fortml_xgboost, only: xgboost_t, xgboost_options_t
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortnum_status, only: fortnum_status_t, FORTNUM_OK
    implicit none

    type(xgboost_t) :: model
    type(xgboost_options_t) :: options
    type(fortml_device_t) :: cuda
    type(fortnum_status_t) :: status
    real(real64) :: x(8, 2), y(8), host_prediction(8), prediction(8)
    real(real64) :: ranking_x(6, 1), relevance(6), x_missing(8, 2)
    integer :: group(6), i
    integer :: numeric_status, categorical_status, missing_status, dart_status
    integer :: ranking_status
    real(real64) :: numeric_error, preserved_error

    do i = 1, 8
        x(i, 1) = real(i - 1, real64) / 7.0_real64
        x(i, 2) = 1.0_real64 - x(i, 1)
        y(i) = merge(-1.0_real64, 2.0_real64, i <= 4)
    end do
    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.

    options%n_estimators = 3
    options%max_depth = 2
    options%learning_rate = 0.4_real64
    options%min_child_weight = 0.0_real64
    call model%fit_regression(x, y, status, options)
    call model%predict(x, host_prediction, status)
    prediction = -huge(1.0_real64)
    call model%predict_device(cuda, x, prediction, status)
    numeric_status = status%code
    if (status%code == FORTNUM_OK) then
        numeric_error = maxval(abs(prediction - host_prediction))
    else
        numeric_error = 0.0_real64
    end if

    options%categorical_policy = "ordered"
    allocate(options%categorical_features(1))
    options%categorical_features(1) = 1
    do i = 1, 8
        x(i, 1) = real(mod(i - 1, 3), real64)
    end do
    call model%fit(x, y, status, options)
    prediction = -huge(1.0_real64)
    call model%predict_device(cuda, x, prediction, status)
    categorical_status = status%code
    preserved_error = merge(0.0_real64, 1.0_real64, &
        all(prediction == -huge(1.0_real64)))
    deallocate(options%categorical_features)

    options%categorical_policy = "none"
    options%missing_policy = "learn"
    x_missing = x
    x_missing(2, 2) = ieee_value(0.0_real64, ieee_quiet_nan)
    call model%fit(x_missing, y, status, options)
    prediction = -huge(1.0_real64)
    call model%predict_device(cuda, x_missing, prediction, status)
    missing_status = status%code
    options%missing_policy = "error"

    options%booster = "dart"
    call model%fit(x, y, status, options)
    prediction = -huge(1.0_real64)
    call model%predict_device(cuda, x, prediction, status)
    dart_status = status%code

    ranking_x(:, 1) = [0.0_real64, 1.0_real64, 2.0_real64, 3.0_real64, 4.0_real64, 5.0_real64]
    relevance = [0.0_real64, 1.0_real64, 0.0_real64, 2.0_real64, 1.0_real64, 0.0_real64]
    group = [1, 1, 2, 2, 3, 3]
    options%booster = "gbtree"
    options%n_estimators = 2
    call model%fit_ranking(ranking_x, relevance, group, status, options)
    prediction(1:6) = -huge(1.0_real64)
    call model%predict_device(cuda, ranking_x, prediction(1:6), status)
    ranking_status = status%code

    write (*, '(a,i0)') "xgb_cuda_numeric_status,", numeric_status
    write (*, '(a,es24.16)') "xgb_cuda_numeric_error,", numeric_error
    write (*, '(a,i0)') "xgb_cuda_categorical_status,", categorical_status
    write (*, '(a,es24.16)') "xgb_cuda_categorical_sentinel_error,", preserved_error
    write (*, '(a,i0)') "xgb_cuda_missing_status,", missing_status
    write (*, '(a,i0)') "xgb_cuda_dart_status,", dart_status
    write (*, '(a,i0)') "xgb_cuda_ranking_status,", ranking_status
end program fortml_bench_xgboost_cuda
