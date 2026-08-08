program fortml_bench_adaboost_classifier
    !! Release workload for the binary AdaBoost weighted-stump contract.
    use, intrinsic :: iso_fortran_env, only: real64, int64
    use fortml_adaboost_classifier, only: adaboost_classifier_t
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer, parameter :: dp = real64
    integer, parameter :: n_samples = 192, n_features = 1, repetitions = 32
    real(dp) :: x(n_samples, n_features), probabilities(n_samples, 2)
    integer :: labels(n_samples), predicted(n_samples), i, row, pattern
    integer(int64) :: clock_start, clock_end, clock_rate
    real(dp) :: fit_seconds, predict_seconds
    type(adaboost_classifier_t) :: model
    type(fortnum_status_t) :: status

    do i = 1, n_samples
        pattern = mod(i - 1, 6)
        x(i, 1) = real(pattern, dp)
        select case (pattern)
        case (0, 1, 3)
            labels(i) = -3
        case default
            labels(i) = 8
        end select
    end do

    call system_clock(clock_start, clock_rate)
    do row = 1, repetitions
        call model%fit(x, labels, status, n_estimators=1, max_depth=1, &
            min_samples_leaf=1)
        if (.not. status_ok(status)) error stop "AdaBoost benchmark fit failed"
    end do
    call system_clock(clock_end)
    fit_seconds = real(clock_end-clock_start, dp)/real(clock_rate, dp)/real(repetitions, dp)

    call system_clock(clock_start, clock_rate)
    do row = 1, repetitions*8
        call model%predict_proba(x, probabilities, status)
        if (.not. status_ok(status)) error stop "AdaBoost benchmark prediction failed"
    end do
    call system_clock(clock_end)
    predict_seconds = real(clock_end-clock_start, dp)/real(clock_rate, dp)/real(repetitions*8, dp)
    call model%predict(x, predicted, status)
    if (.not. status_ok(status)) error stop "AdaBoost benchmark labels failed"

    write (*, '(a,i0,a,i0,a,i0,a,es24.16)') &
        "adaboost_fit,", n_samples, ",", n_features, ",", &
        model%estimator_count(), ",", fit_seconds
    write (*, '(a,i0,a,i0,a,i0,a,es24.16)') &
        "adaboost_predict,", n_samples, ",", n_features, ",", &
        model%estimator_count(), ",", predict_seconds
    write (*, '(a,*(1x,es24.16))') "adaboost_probability_values", probabilities(:, 2)
    write (*, '(a,*(1x,i0))') "adaboost_prediction_values", predicted
end program fortml_bench_adaboost_classifier
