program fortml_bench_adaboost_samme_r
    !! Release workload for the multiclass SAMME.R probability-update contract.
    use, intrinsic :: iso_fortran_env, only: real64, int64
    use fortml_adaboost_classifier, only: adaboost_classifier_t, &
        ADABOOST_ALGORITHM_SAMME_R
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer, parameter :: dp = real64
    integer, parameter :: n_samples = 9, n_query = 5, repetitions = 128
    real(dp) :: x(n_samples, 1), query(n_query, 1), probabilities(n_query, 3)
    integer :: labels(n_samples), predicted(n_query), i, row, cuda_code
    integer(int64) :: clock_start, clock_end, clock_rate
    real(dp) :: fit_seconds, predict_seconds, stage_weight(1)
    type(adaboost_classifier_t) :: model
    type(fortml_device_t) :: cuda
    type(fortnum_status_t) :: status

    do i = 1, n_samples
        x(i, 1) = real(i-1, dp)
    end do
    labels = [-7, -7, -7, 4, 4, 4, 99, 99, 99]
    query(:, 1) = [0.5_dp, 2.5_dp, 3.5_dp, 5.5_dp, 7.5_dp]

    call system_clock(clock_start, clock_rate)
    do row = 1, repetitions
        call model%fit(x, labels, status, n_estimators=1, max_depth=1, &
            min_samples_leaf=1, seed=7, algorithm=ADABOOST_ALGORITHM_SAMME_R)
        if (.not. status_ok(status)) error stop "SAMME.R benchmark fit failed"
    end do
    call system_clock(clock_end)
    fit_seconds = real(clock_end-clock_start, dp)/real(clock_rate, dp)/real(repetitions, dp)

    call system_clock(clock_start, clock_rate)
    do row = 1, repetitions*16
        call model%predict_proba(query, probabilities, status)
        if (.not. status_ok(status)) error stop "SAMME.R benchmark prediction failed"
    end do
    call system_clock(clock_end)
    predict_seconds = real(clock_end-clock_start, dp)/real(clock_rate, dp)/real(repetitions*16, dp)
    call model%predict(query, predicted, status)
    if (.not. status_ok(status)) error stop "SAMME.R benchmark labels failed"
    stage_weight = model%stage_weights()

    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call model%predict_proba_device(cuda, query, probabilities, status)
    cuda_code = status%code

    write (*, '(a,es24.16)') "sammer_fit,", fit_seconds
    write (*, '(a,es24.16)') "sammer_predict,", predict_seconds
    write (*, '(a,*(1x,es24.16))') "sammer_probability_values", probabilities
    write (*, '(a,*(1x,i0))') "sammer_prediction_values", predicted
    write (*, '(a,i0)') "sammer_stage_count,", model%estimator_count()
    write (*, '(a,es24.16)') "sammer_stage_weight,", stage_weight(1)
    write (*, '(a,i0)') "sammer_cuda_code,", cuda_code
end program fortml_bench_adaboost_samme_r
