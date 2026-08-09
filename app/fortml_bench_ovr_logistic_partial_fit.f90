program fortml_bench_ovr_logistic_partial_fit
    !! Release workload for deterministic OVR partial-fit replay.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortml_ovr_logistic_classifier, only: ovr_logistic_classifier_t
    use fortml_classification_state, only: classification_state_t
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer, parameter :: n_samples = 12, n_features = 3, n_classes = 3
    integer, parameter :: n_query = 6
    real(dp) :: x(n_samples, n_features), query(n_query, n_features)
    real(dp) :: probabilities(n_query, n_classes)
    real(dp) :: reference_probabilities(n_query, n_classes)
    integer :: labels(n_samples), classes(n_classes), prediction(n_query)
    integer :: split, repetitions, i
    integer(int64) :: tick_start, tick_end, ticks_per_second
    real(dp) :: elapsed, replay_error
    type(ovr_logistic_classifier_t) :: chunked, reference
    type(classification_state_t) :: metadata
    type(fortnum_status_t) :: status
    type(fortml_device_t) :: cuda

    do i = 1, n_samples
        x(i, :) = [real(i - 1, dp), real(mod(i, 3) - 1, dp), &
            real(mod(i + 1, 2), dp)]
        labels(i) = merge(-7, merge(10, 42, mod(i, 3) == 0), mod(i, 3) == 1)
    end do
    query = reshape([ &
        0.0_dp, -1.0_dp, 0.0_dp, 2.0_dp, 0.0_dp, 1.0_dp, &
        4.0_dp, 1.0_dp, 0.0_dp, 7.0_dp, -1.0_dp, 1.0_dp, &
        9.0_dp, 0.0_dp, 0.0_dp, 11.0_dp, 1.0_dp, 1.0_dp], shape(query))
    classes = [-7, 10, 42]
    split = 4
    repetitions = 1

    call reference%fit(x, labels, status, l2=0.1_dp, max_iterations=1000, &
        tolerance=1.0e-7_dp)
    if (.not. status_ok(status)) error stop "reference fit failed"
    call chunked%partial_fit(x(:split, :), labels(:split), status, classes=classes, &
        l2=0.1_dp, max_iterations=1000, tolerance=1.0e-7_dp)
    if (.not. status_ok(status)) error stop "prefix partial fit failed"
    call system_clock(tick_start, ticks_per_second)
    do i = 1, repetitions
        call chunked%partial_fit(x(split + 1:, :), labels(split + 1:), status)
        if (.not. status_ok(status)) error stop "suffix partial fit failed"
    end do
    call system_clock(tick_end)
    elapsed = real(tick_end - tick_start, dp)/real(ticks_per_second, dp)
    call chunked%predict_proba(query, probabilities, status)
    if (.not. status_ok(status)) error stop "chunked prediction failed"
    call reference%predict_proba(query, reference_probabilities, status)
    if (.not. status_ok(status)) error stop "reference prediction failed"
    replay_error = maxval(abs(probabilities - reference_probabilities))
    call chunked%predict(query, prediction, status)
    if (.not. status_ok(status)) error stop "prediction failed"
    metadata = chunked%metadata()
    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call chunked%predict_proba_device(cuda, query, probabilities, status)
    write (*, '(a,",",i0,",",es24.16,",",es24.16,",",i0)') &
        "ovr_logistic_partial_fit", metadata%batch_count(), replay_error, &
        elapsed, status%code
end program fortml_bench_ovr_logistic_partial_fit
