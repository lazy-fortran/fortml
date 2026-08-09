program fortml_bench_random_forest_regression
    !! Correctness-gated deterministic weighted random-forest regression workload.
    use, intrinsic :: iso_fortran_env, only: real64, int64
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortml_random_forest_regressor, only: random_forest_regressor_t
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    implicit none

    integer, parameter :: dp = real64
    integer, parameter :: n_samples = 8, n_features = 2, n_outputs = 2
    integer, parameter :: n_query = 4, n_trees = 25
    real(dp) :: x(n_samples, n_features), targets(n_samples, n_outputs)
    real(dp) :: query(n_query, n_features), predictions(n_query, n_outputs)
    real(dp) :: staged(n_query, n_trees, n_outputs), importances(n_features)
    real(dp) :: x_dot(n_query, n_features), predictions_dot(n_query, n_outputs)
    real(dp) :: fit_seconds, predict_seconds
    real(dp) :: weights(n_samples), cuda_predictions(n_query, n_outputs)
    integer(int64) :: started, finished, rate
    integer :: i, repetitions
    type(random_forest_regressor_t) :: model
    type(fortml_device_t) :: cuda
    type(fortnum_status_t) :: status

    x(:, 1) = [-3.0_dp, -2.0_dp, -1.0_dp, 1.0_dp, 2.0_dp, 3.0_dp, 4.0_dp, 5.0_dp]
    x(:, 2) = 0.25_dp
    targets(:, 1) = [0.0_dp, 0.0_dp, 0.0_dp, 1.0_dp, 1.0_dp, 1.0_dp, 1.0_dp, 1.0_dp]
    targets(:, 2) = 0.5_dp + 2.0_dp*targets(:, 1)
    query(:, 1) = [-2.7_dp, -0.123_dp, 0.123_dp, 2.7_dp]
    query(:, 2) = 0.25_dp
    weights = [1.0_dp, 2.0_dp, 1.0_dp, 3.0_dp, 1.0_dp, 1.0_dp, 2.0_dp, 1.0_dp]
    x_dot = 1.0_dp

    call system_clock(started, rate)
    call model%fit(x, targets, status, n_trees=n_trees, max_depth=3, &
        min_samples_leaf=1, seed=5489, sample_weight=weights)
    call system_clock(finished)
    if (.not. status_ok(status)) error stop "random forest regression fit failed"
    fit_seconds = real(finished - started, dp)/real(rate, dp)

    call system_clock(started, rate)
    do repetitions = 1, 128
        call model%predict(query, predictions, status)
    end do
    call system_clock(finished)
    if (.not. status_ok(status)) error stop "random forest regression prediction failed"
    predict_seconds = real(finished - started, dp)/real(rate, dp)/128.0_dp

    call model%predict_staged(query, staged, status)
    call model%predict_jvp(query, x_dot, predictions, predictions_dot, status)
    call model%feature_importances(importances, status)
    if (.not. status_ok(status)) error stop "random forest regression diagnostics failed"

    write (*, '(a,es24.16)') "random_forest_regression_fit_seconds,", fit_seconds
    write (*, '(a,es24.16)') "random_forest_regression_predict_seconds,", predict_seconds
    write (*, '(a,es24.16)') "random_forest_regression_stage_error,", &
        maxval(abs(staged(:, n_trees, :) - predictions))
    write (*, '(a,es24.16)') "random_forest_regression_jvp_max,", &
        maxval(abs(predictions_dot))
    write (*, '(a,es24.16)') "random_forest_regression_importance_sum,", sum(importances)
    do i = 1, n_query
        write (*, '(a,i0,a,es24.16)') &
            "random_forest_regression_prediction,", i, ",1,", predictions(i, 1)
        write (*, '(a,i0,a,es24.16)') &
            "random_forest_regression_prediction,", i, ",2,", predictions(i, 2)
    end do

    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    cuda_predictions = -37.0_dp
    call model%predict_device(cuda, query, cuda_predictions, status)
    if (status%code /= FORTNUM_NOT_IMPLEMENTED .or. any(cuda_predictions /= -37.0_dp)) then
        error stop "random forest regression CUDA contract changed unexpectedly"
    end if
    write (*, '(a)') "random_forest_regression_cuda,unavailable"
end program fortml_bench_random_forest_regression
