program fortml_bench_random_forest_permutation
    !! Correctness-gated random-forest permutation-importance workload.
    use, intrinsic :: iso_fortran_env, only: real64, int64
    use fortml_random_forest_classifier, only: random_forest_classifier_t
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    implicit none

    integer, parameter :: dp = real64
    integer, parameter :: n_samples = 240, n_features = 3, n_trees = 64
    integer, parameter :: n_repeats = 24, permutation_seed = 991
    real(dp) :: x(n_samples, n_features), importance(n_features), importance_std(n_features)
    real(dp) :: fit_seconds, permutation_seconds, baseline
    integer :: labels(n_samples), predictions(n_samples), expected(n_samples)
    integer :: i, oracle_correct
    integer(int64) :: started, finished, rate
    type(random_forest_classifier_t) :: model
    type(fortml_device_t) :: cuda
    type(fortnum_status_t) :: status

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

    call system_clock(started, rate)
    call model%fit(x, labels, status, n_trees=n_trees, max_depth=6, seed=1729)
    call system_clock(finished)
    if (.not. status_ok(status)) error stop "random forest permutation benchmark fit failed"
    fit_seconds = real(finished - started, dp)/real(rate, dp)

    importance = -17.0_dp
    importance_std = -19.0_dp
    baseline = -23.0_dp
    call system_clock(started)
    call model%permutation_importance(x, labels, importance, status, n_repeats=n_repeats, &
        seed=permutation_seed, importance_std=importance_std, baseline_score=baseline)
    call system_clock(finished)
    if (.not. status_ok(status)) error stop "random forest permutation benchmark failed"
    call model%predict(x, predictions, status)
    if (.not. status_ok(status)) error stop "random forest permutation baseline failed"
    do i = 1, n_samples
        if (x(i, 1) < -0.65_dp) then
            expected(i) = -3
        else if (x(i, 1) > 0.65_dp) then
            expected(i) = 11
        else
            expected(i) = 4
        end if
    end do
    oracle_correct = count(predictions == expected)
    permutation_seconds = real(finished - started, dp)/real(rate, dp)
    if (baseline < 0.95_dp .or. importance(1) <= importance(2) .or. &
        importance(1) <= importance(3) .or. importance_std(1) < 0.0_dp .or. &
        oracle_correct /= n_samples) then
        error stop "random forest permutation independent oracle failed"
    end if

    write (*, '(a,i0,a,i0,a,i0,a,i0,a,es24.16,a,es24.16,a,es24.16,a,es24.16,a,es24.16,a,es24.16,a,es24.16,a,es24.16,a,es24.16,a,i0)') &
        "rf_permutation,metrics,", n_samples, ",", n_features, ",", n_trees, ",", &
        n_repeats, ",", fit_seconds, ",", permutation_seconds, ",", baseline, ",", &
        importance(1), ",", importance(2), ",", importance(3), ",", importance_std(1), ",", &
        importance_std(2), ",", importance_std(3), ",", oracle_correct

    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    importance = -29.0_dp
    importance_std = -31.0_dp
    baseline = -37.0_dp
    call model%permutation_importance_device(cuda, x, labels, importance, status, &
        n_repeats=n_repeats, seed=permutation_seed, importance_std=importance_std, &
        baseline_score=baseline)
    if (status%code /= FORTNUM_NOT_IMPLEMENTED .or. any(importance /= -29.0_dp) .or. &
        any(importance_std /= -31.0_dp) .or. baseline /= -37.0_dp) then
        error stop "random forest permutation CUDA contract changed unexpectedly"
    end if
    write (*, '(a)') "rf_permutation,cuda,unavailable"
end program fortml_bench_random_forest_permutation
