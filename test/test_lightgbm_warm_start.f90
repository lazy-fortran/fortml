program test_lightgbm_warm_start
    !! Independent staged-margin oracle for transactional LightGBM continuation.
    !! The warm path is compared with a separately fitted full ensemble; failed
    !! controls must leave every staged prefix unchanged.
    use, intrinsic :: iso_fortran_env, only: real64, error_unit
    use fortml_lightgbm, only: lightgbm_t, lightgbm_options_t
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_DOMAIN_ERROR, &
        FORTNUM_NOT_IMPLEMENTED
    implicit none

    type(lightgbm_t) :: prefix, warm, full, invalid
    type(lightgbm_options_t) :: prefix_options, full_options, bad_options
    type(fortnum_status_t) :: status
    real(real64) :: x(10, 2), target(10), prefix_staged(10, 2), warm_staged(10, 5)
    real(real64) :: full_staged(10, 5), before_invalid(10, 2), after_invalid(10, 2)
    real(real64) :: weights(10)
    integer :: failures, i

    failures = 0
    do i = 1, size(x, 1)
        x(i, 1) = real(i-1, real64)
        x(i, 2) = real(mod(i+1, 4), real64) - 1.5_real64
        target(i) = 0.5_real64*x(i, 1) + 2.0_real64*x(i, 2)
        weights(i) = 1.0_real64 + real(mod(i, 3), real64)*0.25_real64
    end do
    prefix_options = lightgbm_options_t()
    prefix_options%n_estimators = 2
    prefix_options%num_leaves = 3
    prefix_options%min_data_in_leaf = 1
    prefix_options%max_bin = 16
    prefix_options%learning_rate = 0.2_real64
    prefix_options%l2 = 0.75_real64
    full_options = prefix_options
    full_options%n_estimators = 5
    call prefix%fit_regression(x, target, status, prefix_options, weights)
    call check(status_ok(status), "prefix fit", failures)
    call prefix%predict_staged_margin(x, prefix_staged, status)
    call check(status_ok(status), "prefix staged output", failures)
    call full%fit_regression(x, target, status, full_options, weights)
    call check(status_ok(status), "independent full fit", failures)
    call full%predict_staged_margin(x, full_staged, status)
    call check(status_ok(status), "full staged output", failures)
    call check(maxval(abs(prefix_staged-full_staged(:, :2))) < 2.0e-13_real64, &
        "independent prefix staged oracle", failures)

    warm = prefix
    call warm%fit_warm_start(x, target, status, full_options, weights)
    call check(status_ok(status) .and. warm%estimator_count() == 5, &
        "warm continuation status and count", failures)
    call warm%predict_staged_margin(x, warm_staged, status)
    call check(status_ok(status), "warm staged output", failures)
    call check(maxval(abs(warm_staged-full_staged)) < 2.0e-13_real64, &
        "warm continuation equals independent full staged oracle", failures)
    call check(maxval(abs(warm_staged(:, :2)-prefix_staged)) < 2.0e-13_real64, &
        "warm continuation preserves prefix", failures)

    invalid = prefix
    call invalid%predict_staged_margin(x, before_invalid, status)
    bad_options = prefix_options
    bad_options%n_estimators = 2
    call invalid%fit_warm_start(x, target, status, bad_options, weights)
    call check(status%code == FORTNUM_DOMAIN_ERROR, "non-growing target refusal", failures)
    call invalid%predict_staged_margin(x, after_invalid, status)
    call check(status_ok(status) .and. maxval(abs(after_invalid-before_invalid)) < &
        2.0e-13_real64, "non-growing refusal is transactional", failures)

    bad_options = full_options
    bad_options%num_leaves = 2
    call invalid%fit_warm_start(x, target, status, bad_options, weights)
    call check(status%code == FORTNUM_DOMAIN_ERROR, "mismatched tree option refusal", failures)
    call invalid%predict_staged_margin(x, after_invalid, status)
    call check(status_ok(status) .and. maxval(abs(after_invalid-before_invalid)) < &
        2.0e-13_real64, "mismatched option refusal is transactional", failures)

    bad_options = full_options
    bad_options%early_stopping_rounds = 1
    call invalid%fit_warm_start(x, target, status, bad_options, weights)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "validation early-stopping boundary", failures)
    call invalid%predict_staged_margin(x, after_invalid, status)
    call check(status_ok(status) .and. maxval(abs(after_invalid-before_invalid)) < &
        2.0e-13_real64, "early-stopping refusal is transactional", failures)

    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, " LightGBM warm-start test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS LightGBM warm-start independent staged oracle"

contains

    subroutine check(condition, label, failures)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: label
        integer, intent(inout) :: failures

        if (.not. condition) then
            write (error_unit, '(a)') "FAIL ["//trim(label)//"]"
            failures = failures + 1
        end if
    end subroutine check

end program test_lightgbm_warm_start
