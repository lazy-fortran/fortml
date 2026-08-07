program test_simple_imputer
    use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_quiet_nan
    use fortml_simple_imputer, only: simple_imputer_t, SIMPLE_IMPUTER_MEAN, &
        SIMPLE_IMPUTER_MEDIAN, SIMPLE_IMPUTER_CONSTANT
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer :: failures

    failures = 0
    call test_mean_and_products(failures)
    call test_median_and_constant(failures)
    call test_refusals(failures)
    if (failures > 0) then
        write (*, '(a,i0)') "FAIL simple imputer cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS simple imputer independent behavioral oracles"

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

    subroutine test_mean_and_products(failures)
        integer, intent(inout) :: failures
        type(simple_imputer_t) :: imputer
        type(fortnum_status_t) :: status
        real(dp) :: x(3, 3), expected(3, 3), transformed(3, 3)
        real(dp) :: x_dot(3, 3), transformed_dot(3, 3), output_bar(3, 3)
        real(dp) :: input_bar(3, 3), statistics(3), lhs, rhs
        real(dp) :: nan

        nan = ieee_value(0.0_dp, ieee_quiet_nan)
        x = reshape([1.0_dp, 3.0_dp, 5.0_dp, nan, 5.0_dp, 7.0_dp, &
            3.0_dp, nan, 9.0_dp], [3, 3])
        expected = reshape([1.0_dp, 3.0_dp, 5.0_dp, 6.0_dp, 5.0_dp, 7.0_dp, &
            3.0_dp, 6.0_dp, 9.0_dp], [3, 3])
        call imputer%fit(x, status, strategy="mean")
        call check(status_ok(status), "mean fit", failures)
        call check(imputer%fitted(), "mean fitted flag", failures)
        call check(imputer%strategy() == SIMPLE_IMPUTER_MEAN, &
            "mean strategy code", failures)
        statistics = imputer%statistics()
        call check(maxval(abs(statistics - [3.0_dp, 6.0_dp, 6.0_dp])) < &
            1.0e-14_dp, "mean statistic oracle", failures)
        call imputer%transform(x, transformed, status)
        call check(status_ok(status), "mean transform", failures)
        call check(maxval(abs(transformed - expected)) < 1.0e-14_dp, &
            "mean transform oracle", failures)

        x_dot = reshape([0.5_dp, -1.0_dp, 2.0_dp, 3.0_dp, -4.0_dp, 5.0_dp, &
            -6.0_dp, 7.0_dp, -8.0_dp], [3, 3])
        expected = reshape([0.5_dp, -1.0_dp, 2.0_dp, 0.0_dp, -4.0_dp, 5.0_dp, &
            -6.0_dp, 0.0_dp, -8.0_dp], [3, 3])
        call imputer%transform_jvp(x, x_dot, transformed_dot, status)
        call check(status_ok(status), "mean JVP", failures)
        call check(maxval(abs(transformed_dot - expected)) < 1.0e-14_dp, &
            "piecewise JVP oracle", failures)

        output_bar = reshape([1.0_dp, -2.0_dp, 3.0_dp, -4.0_dp, 5.0_dp, -6.0_dp, &
            7.0_dp, -8.0_dp, 9.0_dp], [3, 3])
        expected = output_bar
        expected(1, 2) = 0.0_dp
        expected(2, 3) = 0.0_dp
        call imputer%transform_vjp(x, output_bar, input_bar, status)
        call check(status_ok(status), "mean VJP", failures)
        call check(maxval(abs(input_bar - expected)) < 1.0e-14_dp, &
            "piecewise VJP oracle", failures)
        lhs = sum(output_bar*transformed_dot)
        rhs = sum(x_dot*input_bar)
        call check(abs(lhs - rhs) < 1.0e-14_dp, "JVP/VJP adjoint", failures)
    end subroutine test_mean_and_products

    subroutine test_median_and_constant(failures)
        integer, intent(inout) :: failures
        type(simple_imputer_t) :: median, constant
        type(fortnum_status_t) :: status
        real(dp) :: x(4, 2), transformed(4, 2), expected(4, 2), nan
        real(dp) :: all_missing(2, 2), constant_output(2, 2)

        nan = ieee_value(0.0_dp, ieee_quiet_nan)
        x = reshape([9.0_dp, 1.0_dp, nan, 5.0_dp, 4.0_dp, 8.0_dp, &
            2.0_dp, nan], [4, 2])
        expected = reshape([9.0_dp, 1.0_dp, 5.0_dp, 5.0_dp, 4.0_dp, 8.0_dp, &
            2.0_dp, 4.0_dp], [4, 2])
        call median%fit(x, status, strategy="median")
        call check(status_ok(status), "median fit", failures)
        call check(median%strategy() == SIMPLE_IMPUTER_MEDIAN, &
            "median strategy code", failures)
        call median%transform(x, transformed, status)
        call check(status_ok(status), "median transform", failures)
        call check(maxval(abs(transformed - expected)) < 1.0e-14_dp, &
            "median transform oracle", failures)

        all_missing = nan
        call constant%fit(all_missing, status, strategy="constant", fill_value=-2.5_dp)
        call check(status_ok(status), "constant fit with all missing", failures)
        call constant%transform(all_missing, constant_output, status)
        call check(status_ok(status), "constant transform", failures)
        call check(maxval(abs(constant_output + 2.5_dp)) < 1.0e-14_dp, &
            "constant transform oracle", failures)
        call check(constant%feature_count() == 2, "constant feature count", failures)
        call check(constant%strategy() == SIMPLE_IMPUTER_CONSTANT, &
            "constant strategy code", failures)
    end subroutine test_median_and_constant

    subroutine test_refusals(failures)
        integer, intent(inout) :: failures
        type(simple_imputer_t) :: imputer
        type(fortnum_status_t) :: status
        real(dp) :: x(2, 2), output(2, 2), nan

        nan = ieee_value(0.0_dp, ieee_quiet_nan)
        x = 1.0_dp
        call imputer%transform(x, output, status)
        call check(.not. status_ok(status), "unfitted transform refusal", failures)
        call imputer%fit(reshape([nan, nan, 1.0_dp, 2.0_dp], [2, 2]), status, &
            strategy="mean")
        call check(.not. status_ok(status), "all-missing mean refusal", failures)
        call imputer%fit(x, status, strategy="unknown")
        call check(.not. status_ok(status), "unknown strategy refusal", failures)
        x(1, 1) = ieee_value(0.0_dp, ieee_quiet_nan)
        x(2, 1) = ieee_value(0.0_dp, ieee_quiet_nan)
        call imputer%fit(x, status, strategy="mean")
        call check(.not. status_ok(status), "all-missing feature refusal", failures)
    end subroutine test_refusals

end program test_simple_imputer
