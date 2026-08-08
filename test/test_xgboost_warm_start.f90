program test_xgboost_warm_start
    !! Independent oracle for deterministic XGBoost suffix continuation.
    use, intrinsic :: iso_fortran_env, only: real64, error_unit
    use fortnum_status, only: fortnum_status_t, FORTNUM_OK
    use fortml_xgboost, only: xgboost_t, xgboost_options_t
    implicit none

    type(xgboost_t) :: prefix, warm, full, invalid
    type(xgboost_options_t) :: prefix_options, full_options, bad_options
    type(fortnum_status_t) :: status
    real(real64) :: x(8, 2), y(8), warm_staged(8, 4), full_staged(8, 4)
    integer :: failures

    failures = 0
    x = reshape([ &
        -3.0_real64, -2.0_real64, -1.0_real64, 0.0_real64, &
         1.0_real64,  2.0_real64,  3.0_real64,  4.0_real64, &
         2.0_real64,  1.0_real64,  0.0_real64,  1.0_real64, &
         2.0_real64,  3.0_real64,  4.0_real64,  5.0_real64], [8, 2])
    y = [ -2.0_real64, -1.0_real64, 0.5_real64, 1.0_real64, &
           2.0_real64,  2.5_real64, 3.5_real64,  4.0_real64 ]

    prefix_options = xgboost_options_t()
    prefix_options%n_estimators = 2
    prefix_options%max_depth = 2
    prefix_options%learning_rate = 0.2_real64
    prefix_options%subsample = 0.75_real64
    prefix_options%colsample_bytree = 0.5_real64
    prefix_options%seed = 9871
    call prefix%fit_regression(x, y, status, prefix_options)
    call check(status%code == FORTNUM_OK .and. prefix%estimator_count() == 2, &
        "prefix fit", failures)

    full_options = prefix_options
    full_options%n_estimators = 4
    call full%fit_regression(x, y, status, full_options)
    call check(status%code == FORTNUM_OK .and. full%estimator_count() == 4, &
        "reference fit", failures)

    warm = prefix
    call warm%fit_warm_start(x, y, status, full_options)
    call check(status%code == FORTNUM_OK .and. warm%estimator_count() == 4, &
        "warm continuation fit", failures)
    call warm%predict_staged_margin(x, warm_staged, status)
    call full%predict_staged_margin(x, full_staged, status)
    call check(status%code == FORTNUM_OK .and. &
        maxval(abs(warm_staged - full_staged)) < 2.0e-13_real64, &
        "warm continuation matches independent full fit", failures)
    call check(warm%requested_estimator_count() == 4, &
        "warm continuation records requested total", failures)

    bad_options = full_options
    bad_options%n_estimators = 2
    call warm%fit_warm_start(x, y, status, bad_options)
    call check(status%code /= FORTNUM_OK .and. warm%estimator_count() == 4, &
        "non-increasing warm target is refused transactionally", failures)

    bad_options = full_options
    bad_options%learning_rate = 0.3_real64
    call warm%fit_warm_start(x, y, status, bad_options)
    call check(status%code /= FORTNUM_OK .and. warm%estimator_count() == 4, &
        "changed controls are refused transactionally", failures)

    call invalid%fit_warm_start(x, y, status, full_options)
    call check(status%code /= FORTNUM_OK .and. .not. invalid%fitted(), &
        "unfitted warm-start source is refused", failures)

    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, " xgboost warm-start test(s) failed"
        error stop 1
    end if
contains
    subroutine check(condition, label, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: label
        integer, intent(inout) :: failures
        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "FAIL [xgb warm] "//label
        end if
    end subroutine check
end program test_xgboost_warm_start
