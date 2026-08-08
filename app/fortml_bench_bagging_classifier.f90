program fortml_bench_bagging_classifier
    !! Correctness-gated bagging classification workload.
    use, intrinsic :: iso_fortran_env, only: real64, int64
    use fortml_bagging_classifier, only: bagging_classifier_t
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    implicit none

    integer, parameter :: dp = real64
    integer, parameter :: n_samples = 240, n_features = 3, n_query = 6
    real(dp) :: x(n_samples, n_features), query(n_query, n_features)
    real(dp) :: probabilities(n_query, 3)
    integer :: labels(n_samples), predictions(n_query)
    integer(int64) :: started, finished, rate
    real(dp) :: fit_seconds, predict_seconds
    type(bagging_classifier_t) :: model
    type(fortml_device_t) :: cuda
    type(fortnum_status_t) :: status
    integer :: i, repetitions

    do i = 1, n_samples
        x(i, 1) = -2.0_dp + 4.0_dp*real(mod(i - 1, 80), dp)/79.0_dp
        x(i, 2) = sin(0.17_dp*real(i, dp))
        x(i, 3) = cos(0.11_dp*real(i, dp))
        if (x(i, 1) < -0.65_dp) then
            labels(i) = -3
        else if (x(i, 1) > 0.65_dp) then
            labels(i) = 11
        else
            labels(i) = 4
        end if
    end do
    query(:, 1) = [-1.5_dp, -0.7_dp, -0.1_dp, 0.1_dp, 0.7_dp, 1.5_dp]
    query(:, 2) = 0.0_dp
    query(:, 3) = 1.0_dp

    call system_clock(started, rate)
    call model%fit(x, labels, status, n_trees=32, max_depth=6, max_samples=180, &
        bootstrap=.true., seed=1729)
    call system_clock(finished)
    if (.not. status_ok(status)) error stop "bagging benchmark fit failed"
    fit_seconds = real(finished - started, dp)/real(rate, dp)
    call model%predict_proba(query, probabilities, status)
    call model%predict(query, predictions, status)
    if (.not. status_ok(status)) error stop "bagging benchmark prediction failed"
    if (maxval(abs(sum(probabilities, dim=2) - 1.0_dp)) > 2.0e-13_dp) then
        error stop "bagging benchmark probability simplex failed"
    end if
    call system_clock(started, rate)
    do repetitions = 1, 128
        call model%predict_proba(query, probabilities, status)
    end do
    call system_clock(finished)
    predict_seconds = real(finished - started, dp)/real(rate, dp)/128.0_dp

    write (*, '(a,es24.16)') "bagging_fit_seconds,", fit_seconds
    write (*, '(a,es24.16)') "bagging_predict_seconds,", predict_seconds
    write (*, '(a,es24.16)') "bagging_probability_sum_error,", &
        maxval(abs(sum(probabilities, dim=2) - 1.0_dp))
    write (*, '(a,i0)') "bagging_query_correct,", &
        count(predictions == [-3, -3, 4, 4, 11, 11])

    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call model%predict_proba_device(cuda, query, probabilities, status)
    if (status%code /= FORTNUM_NOT_IMPLEMENTED) then
        error stop "bagging CUDA contract changed unexpectedly"
    end if
    write (*, '(a)') "bagging_cuda,unavailable"
end program fortml_bench_bagging_classifier
