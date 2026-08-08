program test_random_forest_permutation_importance
    !! Independent behavioral oracle for fixed-state forest permutation scores.
    use, intrinsic :: iso_fortran_env, only: real64, error_unit
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_DOMAIN_ERROR, &
        FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortml_random_forest_classifier, only: random_forest_classifier_t
    implicit none

    integer, parameter :: dp = real64
    integer, parameter :: n_samples = 30
    real(dp) :: x(n_samples, 2), importance(2), repeat_importance(2), importance_std(2), &
        repeat_std(2), cuda_importance(2), cuda_std(2), baseline, repeat_baseline, &
        cuda_baseline
    integer :: labels(n_samples), failures, i
    type(random_forest_classifier_t) :: model, repeat_model
    type(fortnum_status_t) :: status
    type(fortml_device_t) :: cuda

    do i = 1, n_samples
        x(i, 1) = -1.0_dp + 2.0_dp*real(i - 1, dp)/real(n_samples - 1, dp)
        x(i, 2) = sin(0.71_dp*real(i, dp))
        if (x(i, 1) < -0.34_dp) then
            labels(i) = -4
        else if (x(i, 1) > 0.34_dp) then
            labels(i) = 9
        else
            labels(i) = 2
        end if
    end do
    failures = 0

    call model%fit(x, labels, status, n_trees=31, max_depth=4, seed=73)
    call check(status_ok(status), "permutation fixture fit", failures)
    importance = -31.0_dp
    importance_std = -37.0_dp
    baseline = -41.0_dp
    call model%permutation_importance(x, labels, importance, status, n_repeats=9, seed=19, &
        importance_std=importance_std, baseline_score=baseline)
    call check(status_ok(status) .and. baseline > 0.99_dp .and. baseline <= 1.0_dp, &
        "baseline accuracy oracle", failures)
    call check(importance(1) > importance(2) + 0.20_dp .and. importance(1) > 0.20_dp .and. &
        importance_std(1) >= 0.0_dp .and. importance_std(2) >= 0.0_dp, &
        "informative-column permutation oracle", failures)

    call repeat_model%fit(x, labels, status, n_trees=31, max_depth=4, seed=73)
    repeat_importance = -53.0_dp
    repeat_std = -59.0_dp
    repeat_baseline = -61.0_dp
    call repeat_model%permutation_importance(x, labels, repeat_importance, status, &
        n_repeats=9, seed=19, importance_std=repeat_std, baseline_score=repeat_baseline)
    call check(status_ok(status) .and. maxval(abs(repeat_importance - importance)) < 2.0e-14_dp .and. &
        maxval(abs(repeat_std - importance_std)) < 2.0e-14_dp .and. &
        abs(repeat_baseline - baseline) < 2.0e-14_dp, &
        "permutation stream determinism", failures)

    importance = -67.0_dp
    call model%permutation_importance(x, labels, importance, status, n_repeats=0)
    call check(status%code == FORTNUM_DOMAIN_ERROR .and. all(importance == -67.0_dp), &
        "invalid repeat refusal is transactional", failures)

    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    cuda_importance = -71.0_dp
    cuda_std = -73.0_dp
    cuda_baseline = -79.0_dp
    call model%permutation_importance_device(cuda, x, labels, cuda_importance, status, &
        n_repeats=9, seed=19, importance_std=cuda_std, baseline_score=cuda_baseline)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED .and. all(cuda_importance == -71.0_dp) .and. &
        all(cuda_std == -73.0_dp) .and. cuda_baseline == -79.0_dp, &
        "CUDA permutation refusal preserves outputs", failures)

    if (failures > 0) then
        write (error_unit, '(a,i0)') "FAIL random forest permutation cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS random forest permutation independent behavioral oracles"

contains

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failures
        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "FAIL [random-forest-permutation] "//description
        end if
    end subroutine check

end program test_random_forest_permutation_importance
