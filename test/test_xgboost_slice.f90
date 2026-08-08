program test_xgboost_slice
    !! Independent staged-prefix oracle for XGBoost model slicing.
    use, intrinsic :: iso_fortran_env, only: real64, error_unit
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_DOMAIN_ERROR
    use fortml_xgboost, only: xgboost_t, xgboost_options_t
    implicit none

    integer :: failures

    failures = 0
    call test_prefix_equivalence(failures)
    call test_refusals_leave_destination(failures)
    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, " xgboost slice test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS"

contains

    subroutine test_prefix_equivalence(failures)
        integer, intent(inout) :: failures
        type(xgboost_t) :: full, sliced
        type(xgboost_options_t) :: options
        type(fortnum_status_t) :: status
        real(real64) :: x(12, 2), y(12), query(7, 2)
        real(real64) :: staged(7, 4), full_prediction(7), sliced_prediction(7)
        real(real64) :: full_importance(2), sliced_importance(2)
        integer :: i

        do i = 1, 12
            x(i, 1) = real(i - 1, real64)
            x(i, 2) = real(mod(2*i + 1, 5), real64)
            y(i) = merge(6.0_real64 + 0.25_real64*x(i, 2), &
                -2.0_real64 + 0.5_real64*x(i, 2), i > 6)
        end do
        query(:, 1) = [-1.0_real64, 0.5_real64, 2.5_real64, 5.5_real64, &
            7.5_real64, 9.5_real64, 12.0_real64]
        query(:, 2) = [-1.0_real64, 0.0_real64, 1.0_real64, 2.0_real64, &
            3.0_real64, 4.0_real64, 5.0_real64]
        options = xgboost_options_t()
        options%n_estimators = 4
        options%max_depth = 2
        options%learning_rate = 0.4_real64
        options%l2 = 0.75_real64
        options%missing_policy = "learn"
        options%monotone_constraints = [1, 0]
        call full%fit_regression(x, y, status, options)
        call check(status_ok(status), "fit source", failures)
        call full%predict_staged(query, staged, status)
        call check(status_ok(status), "staged source prediction", failures)
        call full%slice(2, sliced, status)
        call check(status_ok(status) .and. sliced%fitted(), "slice succeeds", failures)
        call sliced%predict(query, sliced_prediction, status)
        call check(status_ok(status) .and. maxval(abs(sliced_prediction - staged(:, 2))) < &
            2.0e-13_real64, "slice equals second staged prefix", failures)
        call full%predict(query, full_prediction, status)
        call check(status_ok(status) .and. maxval(abs(full_prediction - sliced_prediction)) > &
            1.0e-8_real64, "slice is not the complete source", failures)
        call full%feature_importance(full_importance, status, kind="gain")
        call sliced%feature_importance(sliced_importance, status, kind="gain")
        call check(status_ok(status) .and. sum(sliced_importance) <= sum(full_importance) + &
            2.0e-13_real64, "slice diagnostics retain prefix gains", failures)
        call check(sliced%estimator_count() == 2 .and. sliced%best_iteration() == 2 .and. &
            sliced%feature_count() == full%feature_count() .and. &
            sliced%objective_name() == full%objective_name() .and. &
            sliced%tree_method() == full%tree_method() .and. &
            sliced%missing_policy() == full%missing_policy() .and. &
            sliced%monotone_constraint(1) == 1 .and. sliced%monotone_constraint(2) == 0 .and. &
            abs(sliced%base_margin() - full%base_margin()) < 2.0e-13_real64, &
            "slice preserves model metadata", failures)
    end subroutine test_prefix_equivalence

    subroutine test_refusals_leave_destination(failures)
        integer, intent(inout) :: failures
        type(xgboost_t) :: source, destination, unfitted, invalid
        type(xgboost_options_t) :: options
        type(fortnum_status_t) :: status
        real(real64) :: x(4, 1), y(4), before(4), after(4)

        x(:, 1) = [0.0_real64, 1.0_real64, 2.0_real64, 3.0_real64]
        y = [0.0_real64, 0.0_real64, 1.0_real64, 1.0_real64]
        options = xgboost_options_t()
        options%n_estimators = 2
        call source%fit_binary(x, y, status, options)
        call destination%fit_binary(x, y, status, options)
        call destination%predict(x, before, status)
        call source%slice(0, destination, status)
        call check(status%code == FORTNUM_DOMAIN_ERROR, "zero prefix refusal", failures)
        call destination%predict(x, after, status)
        call check(status_ok(status) .and. maxval(abs(before - after)) < 2.0e-13_real64, &
            "zero prefix leaves destination", failures)
        call source%slice(3, destination, status)
        call check(status%code == FORTNUM_DOMAIN_ERROR, "oversized prefix refusal", failures)
        call source%slice(1, invalid, status)
        call check(status_ok(status) .and. invalid%fitted(), "valid replacement after refusal", failures)
        call unfitted%slice(1, destination, status)
        call check(status%code == FORTNUM_DOMAIN_ERROR, "unfitted source refusal", failures)
    end subroutine test_refusals_leave_destination

    subroutine check(condition, label, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: label
        integer, intent(inout) :: failures

        if (.not. condition) then
            write (error_unit, '(a)') "FAIL [xgb slice] "//label
            failures = failures + 1
        end if
    end subroutine check

end program test_xgboost_slice
