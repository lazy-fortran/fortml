program fortml_bench_gaussian_nb_partial_fit
    !! Release workload for GaussianNB partial-fit replay and device boundary.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortml_gaussian_naive_bayes, only: gaussian_naive_bayes_t
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer, parameter :: n_samples = 6, n_features = 2, n_classes = 3
    integer, parameter :: n_query = 2
    real(dp) :: x(n_samples, n_features), query(n_query, n_features)
    real(dp) :: probabilities(n_query, n_classes)
    real(dp) :: reference_probabilities(n_query, n_classes)
    integer :: labels(n_samples), classes(n_classes), split, i
    integer(int64) :: tick_start, tick_end, ticks_per_second
    real(dp) :: elapsed, replay_error
    type(gaussian_naive_bayes_t) :: chunked, reference
    type(fortnum_status_t) :: status
    type(fortml_device_t) :: cuda

    x(:, 1) = [8.0_dp, 0.0_dp, 10.0_dp, 2.0_dp, 4.0_dp, 6.0_dp]
    x(:, 2) = [4.0_dp, 0.0_dp, 6.0_dp, 2.0_dp, 1.0_dp, 3.0_dp]
    labels = [9, -3, 9, -3, 4, 4]
    query(1, :) = [1.0_dp, 1.0_dp]
    query(2, :) = [5.0_dp, 2.0_dp]
    classes = [-3, 4, 9]
    split = 3

    call reference%fit(x, labels, status, var_smoothing=0.0_dp)
    if (.not. status_ok(status)) error stop "reference GaussianNB fit failed"
    call chunked%partial_fit(x(:split, :), labels(:split), status, &
        classes=classes, var_smoothing=0.0_dp)
    if (.not. status_ok(status)) error stop "prefix GaussianNB partial fit failed"
    call system_clock(tick_start, ticks_per_second)
    call chunked%partial_fit(x(split + 1:, :), labels(split + 1:), status)
    if (.not. status_ok(status)) error stop "suffix GaussianNB partial fit failed"
    call system_clock(tick_end)
    elapsed = real(tick_end - tick_start, dp)/real(ticks_per_second, dp)
    call chunked%predict_proba(query, probabilities, status)
    if (.not. status_ok(status)) error stop "chunked GaussianNB prediction failed"
    call reference%predict_proba(query, reference_probabilities, status)
    if (.not. status_ok(status)) error stop "reference GaussianNB prediction failed"
    replay_error = maxval(abs(probabilities - reference_probabilities))

    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call chunked%partial_fit_device(cuda, x(1:1, :), labels(1:1), status)
    write (*, '(a,",",i0,",",es24.16,",",es24.16,",",i0)') &
        "gaussian_nb_partial_fit", chunked%batch_count(), replay_error, &
        elapsed, status%code
end program fortml_bench_gaussian_nb_partial_fit
