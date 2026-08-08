program fortml_bench_random_forest_oob
    !! Correctness-gated random-forest out-of-bag workload.
    use, intrinsic :: iso_fortran_env, only: real64, int64
    use fortml_random_forest_classifier, only: random_forest_classifier_t
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    implicit none

    integer, parameter :: dp = real64
    integer, parameter :: n_samples = 240, n_features = 3, n_trees = 64
    real(dp) :: x(n_samples, n_features), probabilities(n_samples, 3)
    real(dp) :: fit_seconds, oob_seconds, oob_score, coverage, simplex_error
    integer :: labels(n_samples), predictions(n_samples), expected(n_samples)
    integer :: i, min_oob_count, oracle_correct
    integer, allocatable :: oob_counts(:), classes(:)
    logical, allocatable :: inclusion(:, :)
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
    if (.not. status_ok(status)) error stop "random forest OOB benchmark fit failed"
    fit_seconds = real(finished - started, dp)/real(rate, dp)

    probabilities = -17.0_dp
    call system_clock(started, rate)
    call model%oob_decision_function(x, probabilities, status)
    call system_clock(finished)
    if (.not. status_ok(status)) error stop "random forest OOB benchmark prediction failed"
    oob_seconds = real(finished - started, dp)/real(rate, dp)
    oob_score = -1.0_dp
    call model%oob_score(x, labels, oob_score, status)
    if (.not. status_ok(status)) error stop "random forest OOB benchmark score failed"

    inclusion = model%bootstrap_inclusion()
    classes = model%classes()
    allocate(oob_counts(n_samples))
    do i = 1, n_samples
        oob_counts(i) = count(.not. inclusion(i, :))
        if (x(i, 1) < -0.65_dp) then
            expected(i) = -3
        else if (x(i, 1) > 0.65_dp) then
            expected(i) = 11
        else
            expected(i) = 4
        end if
        predictions(i) = classes(maxloc(probabilities(i, :), dim=1))
    end do
    min_oob_count = minval(oob_counts)
    coverage = model%oob_coverage()
    oracle_correct = count(predictions == expected)
    simplex_error = maxval(abs(sum(probabilities, dim=2) - 1.0_dp))
    if (coverage < 1.0_dp .or. min_oob_count < 1 .or. simplex_error > 5.0e-13_dp .or. &
        oracle_correct /= n_samples .or. abs(oob_score - real(oracle_correct, dp)/ &
        real(n_samples, dp)) > 5.0e-13_dp) then
        error stop "random forest OOB independent oracle failed"
    end if

    write (*, '(a,i0,a,i0,a,i0,a,es24.16,a,es24.16,a,es24.16,a,es24.16,a,es24.16,a,i0,a,i0)') &
        "rf_oob,metrics,", n_samples, ",", n_features, ",", n_trees, ",", fit_seconds, ",", &
        oob_seconds, ",", oob_score, ",", coverage, ",", simplex_error, ",", oracle_correct, ",", &
        min_oob_count

    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    probabilities = -29.0_dp
    call model%oob_decision_function_device(cuda, x, probabilities, status)
    if (status%code /= FORTNUM_NOT_IMPLEMENTED .or. any(probabilities /= -29.0_dp)) then
        error stop "random forest OOB CUDA contract changed unexpectedly"
    end if
    write (*, '(a)') "rf_oob,cuda,unavailable"
end program fortml_bench_random_forest_oob
