program test_missing_indicator
    use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_quiet_nan, &
        ieee_positive_inf
    use fortml_missing_indicator, only: missing_indicator_t, &
        MISSING_INDICATOR_ALL, MISSING_INDICATOR_MISSING_ONLY
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer :: failures

    failures = 0
    call test_modes(failures)
    call test_products(failures)
    call test_refusals(failures)
    if (failures > 0) then
        write (*, '(a,i0)') "FAIL missing indicator cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS missing indicator independent behavioral oracles"

contains

    subroutine check(condition, label, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: label
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (*, '(a)') "  FAIL "//label
        end if
    end subroutine check

    subroutine test_modes(failures)
        integer, intent(inout) :: failures
        type(missing_indicator_t) :: all_features, missing_only
        type(fortnum_status_t) :: status
        real(dp) :: x(3, 3), mask_all(3, 3), mask_missing(3, 2), nan

        nan = ieee_value(0.0_dp, ieee_quiet_nan)
        x = 1.0_dp
        x(:, 1) = [1.0_dp, nan, 3.0_dp]
        x(:, 2) = [4.0_dp, 5.0_dp, 6.0_dp]
        x(:, 3) = [7.0_dp, 8.0_dp, nan]
        call all_features%fit(x, status, features="all")
        call check(status_ok(status), "all fit", failures)
        call check(all_features%mode() == MISSING_INDICATOR_ALL, "all mode", failures)
        call check(all_features%output_count() == 3, "all output count", failures)
        call all_features%transform(x, mask_all, status)
        call check(status_ok(status), "all transform", failures)
        call check(maxval(abs(mask_all - reshape([0.0_dp, 1.0_dp, 0.0_dp, &
            0.0_dp, 0.0_dp, 0.0_dp, 0.0_dp, 0.0_dp, 1.0_dp], [3, 3]))) < 1.0e-14_dp, &
            "all mask oracle", failures)

        call missing_only%fit(x, status, features="missing-only")
        call check(status_ok(status), "missing-only fit", failures)
        call check(missing_only%mode() == MISSING_INDICATOR_MISSING_ONLY, &
            "missing-only mode", failures)
        call check(missing_only%output_count() == 2, "missing-only output count", failures)
        call check(all(missing_only%feature_indices() == [1, 3]), &
            "missing-only feature selection", failures)
        call missing_only%transform(x, mask_missing, status)
        call check(status_ok(status), "missing-only transform", failures)
        call check(maxval(abs(mask_missing - reshape([0.0_dp, 1.0_dp, 0.0_dp, &
            0.0_dp, 0.0_dp, 1.0_dp], [3, 2]))) < 1.0e-14_dp, &
            "missing-only mask oracle", failures)
    end subroutine test_modes

    subroutine test_products(failures)
        integer, intent(inout) :: failures
        type(missing_indicator_t) :: indicator
        type(fortnum_status_t) :: status
        real(dp) :: x(2, 2), x_dot(2, 2), y_dot(2, 1), y_bar(2, 1), x_bar(2, 2), nan

        nan = ieee_value(0.0_dp, ieee_quiet_nan)
        x = reshape([nan, 2.0_dp, 3.0_dp, 4.0_dp], [2, 2])
        x_dot = 1.0_dp
        y_dot = 4.0_dp
        y_bar = reshape([2.0_dp, -3.0_dp], [2, 1])
        call indicator%fit(x, status, features="missing-only")
        call check(status_ok(status), "product fit", failures)
        call indicator%transform_jvp(x, x_dot, y_dot, status)
        call check(status_ok(status), "zero JVP", failures)
        call check(maxval(abs(y_dot)) == 0.0_dp, "zero JVP oracle", failures)
        call indicator%transform_vjp(x, y_bar, x_bar, status)
        call check(status_ok(status), "zero VJP", failures)
        call check(maxval(abs(x_bar)) == 0.0_dp, "zero VJP oracle", failures)
    end subroutine test_products

    subroutine test_refusals(failures)
        integer, intent(inout) :: failures
        type(missing_indicator_t) :: indicator
        type(fortnum_status_t) :: status
        real(dp) :: x(2, 1), y(2, 1), inf

        x = 1.0_dp
        y = 0.0_dp
        call indicator%transform(x, y, status)
        call check(.not. status_ok(status), "unfitted transform refusal", failures)
        inf = ieee_value(0.0_dp, ieee_positive_inf)
        x(1, 1) = inf
        call indicator%fit(x, status)
        call check(.not. status_ok(status), "infinity fit refusal", failures)
        x = 1.0_dp
        call indicator%fit(x, status, features="unknown")
        call check(.not. status_ok(status), "unknown mode refusal", failures)
    end subroutine test_refusals

end program test_missing_indicator
