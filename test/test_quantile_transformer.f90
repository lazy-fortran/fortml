program test_quantile_transformer
    !! Independent order-statistic oracle for the uniform quantile map.
    use, intrinsic :: iso_fortran_env, only: error_unit
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_DOMAIN_ERROR, &
        FORTNUM_NOT_IMPLEMENTED
    use fortml_preprocessing, only: quantile_transformer_t, &
        QUANTILE_OUTPUT_NORMAL
    implicit none

    type(quantile_transformer_t) :: transformer, unsupported
    type(fortnum_status_t) :: status
    real(dp) :: x(5, 2), query(2, 2), transformed(2, 2), restored(2, 2)
    real(dp) :: x_dot(2, 2), transformed_dot(2, 2), knot(1, 2), knot_dot(1, 2)
    integer :: failures

    failures = 0
    x(:, 1) = [0.0_dp, 1.0_dp, 2.0_dp, 3.0_dp, 4.0_dp]
    x(:, 2) = [0.0_dp, 10.0_dp, 20.0_dp, 30.0_dp, 40.0_dp]
    call transformer%fit(x, status, n_quantiles=5)
    call check(status_ok(status) .and. transformer%fitted() .and. &
        transformer%feature_count() == 2 .and. transformer%quantile_count() == 5, &
        "fit stores the independent order-statistic state", failures)

    query = 0.0_dp
    query(1, :) = [0.5_dp, 5.0_dp]
    query(2, :) = [2.5_dp, 25.0_dp]
    call transformer%transform(query, transformed, status)
    call check(status_ok(status) .and. maxval(abs(transformed - &
        reshape([0.125_dp, 0.625_dp, 0.125_dp, 0.625_dp], [2, 2]))) < 1.0e-14_dp, &
        "piecewise-linear CDF matches the analytic uniform oracle", failures)

    call transformer%inverse_transform(transformed, restored, status)
    call check(status_ok(status) .and. maxval(abs(restored - query)) < 1.0e-14_dp, &
        "inverse interpolation recovers interior query values", failures)

    x_dot = 0.0_dp
    x_dot(1, :) = [2.0_dp, -4.0_dp]
    x_dot(2, :) = [-3.0_dp, 8.0_dp]
    call transformer%transform_jvp(query, x_dot, transformed_dot, status)
    call check(status_ok(status) .and. maxval(abs(transformed_dot - &
        reshape([0.5_dp, -0.75_dp, -0.1_dp, 0.2_dp], [2, 2]))) < 1.0e-14_dp, &
        "input JVP matches the fixed open-segment slope oracle", failures)

    query(1, :) = [-1.0_dp, 50.0_dp]
    call transformer%transform(query(1:1, :), transformed(1:1, :), status)
    call check(status_ok(status) .and. maxval(abs(transformed(1, :) - &
        [0.0_dp, 1.0_dp])) < 1.0e-14_dp, &
        "out-of-range values clamp to the empirical CDF endpoints", failures)

    knot(1, :) = [1.0_dp, 10.0_dp]
    knot_dot = 1.0_dp
    call transformer%transform_jvp(knot, knot_dot, transformed_dot(1:1, :), status)
    call check(status%code == FORTNUM_DOMAIN_ERROR .and. &
        maxval(abs(transformed_dot(1, :))) == 0.0_dp, &
        "knot JVP is an explicit nonsmooth refusal", failures)

    call unsupported%fit(x, status, n_quantiles=5, &
        output_distribution=QUANTILE_OUTPUT_NORMAL)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED .and. &
        .not. unsupported%fitted(), "normal output remains a typed boundary", failures)

    if (failures /= 0) then
        write (error_unit, '(a,i0)') "FAIL quantile transformer cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS quantile transformer independent order-statistic oracle"

contains

    subroutine check(condition, description, failure_count)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failure_count

        if (.not. condition) then
            failure_count = failure_count + 1
            write (error_unit, '(a)') "  FAIL [quantile] "//description
        end if
    end subroutine check

end program test_quantile_transformer
