program test_power_transformer
    !! Independent Yeo-Johnson and Box-Cox transform oracle.
    use, intrinsic :: iso_fortran_env, only: error_unit
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_DOMAIN_ERROR
    use fortml_preprocessing, only: power_transformer_t, &
        POWER_METHOD_YEO_JOHNSON, POWER_METHOD_BOX_COX
    implicit none

    type(power_transformer_t) :: yeo, box, fitted
    type(fortnum_status_t) :: status
    real(dp) :: yj_input(4, 1), yj_output(4, 1), yj_restored(4, 1)
    real(dp) :: yj_dot(4, 1), yj_output_dot(4, 1), yj_expected(4)
    real(dp) :: box_input(3, 1), box_output(3, 1), box_restored(3, 1)
    real(dp) :: box_expected(3), lambdas(1), invalid(2, 1)
    integer :: failures

    failures = 0
    lambdas = 0.0_dp
    yj_input(:, 1) = [-0.5_dp, 0.0_dp, 1.0_dp, 3.0_dp]
    yj_expected = [-0.625_dp, 0.0_dp, log(2.0_dp), log(4.0_dp)]
    call yeo%fit(yj_input, status, method=POWER_METHOD_YEO_JOHNSON, &
        standardize=.false., lambdas=lambdas)
    call check(status_ok(status) .and. yeo%fitted() .and. &
        yeo%method() == POWER_METHOD_YEO_JOHNSON, &
        "Yeo-Johnson fixed-lambda fit", failures)
    call yeo%transform(yj_input, yj_output, status)
    call check(status_ok(status) .and. maxval(abs(yj_output(:, 1) - yj_expected)) < 1.0e-13_dp, &
        "Yeo-Johnson values match independent branch oracle", failures)
    call yeo%inverse_transform(yj_output, yj_restored, status)
    call check(status_ok(status) .and. maxval(abs(yj_restored - yj_input)) < 1.0e-13_dp, &
        "Yeo-Johnson inverse roundtrip", failures)
    yj_dot(:, 1) = [2.0_dp, 1.0_dp, -3.0_dp, 4.0_dp]
    call yeo%transform_jvp(yj_input, yj_dot, yj_output_dot, status)
    call check(status%code == FORTNUM_DOMAIN_ERROR .and. maxval(abs(yj_output_dot)) == 0.0_dp, &
        "branch-point JVP is an explicit refusal", failures)
    call yeo%transform_jvp(yj_input([1, 3, 4], :), yj_dot([1, 3, 4], :), &
        yj_output_dot(1:3, :), status)
    call check(status_ok(status) .and. maxval(abs(yj_output_dot(1:3, 1) - &
        [2.0_dp*1.5_dp, -3.0_dp/2.0_dp, 4.0_dp/4.0_dp])) < 1.0e-13_dp, &
        "Yeo-Johnson JVP matches analytic derivative away from zero", failures)

    lambdas = 0.5_dp
    box_input(:, 1) = [1.0_dp, 4.0_dp, 9.0_dp]
    box_expected = [0.0_dp, 2.0_dp, 4.0_dp]
    call box%fit(box_input, status, method=POWER_METHOD_BOX_COX, &
        standardize=.false., lambdas=lambdas)
    call box%transform(box_input, box_output, status)
    call check(status_ok(status) .and. maxval(abs(box_output(:, 1) - box_expected)) < 1.0e-13_dp, &
        "Box-Cox values match square-root oracle", failures)
    call box%inverse_transform(box_output, box_restored, status)
    call check(status_ok(status) .and. maxval(abs(box_restored - box_input)) < 1.0e-13_dp, &
        "Box-Cox inverse roundtrip", failures)

    invalid(:, 1) = [0.0_dp, 1.0_dp]
    call fitted%fit(invalid, status, method=POWER_METHOD_BOX_COX, lambdas=lambdas)
    call check(status%code == FORTNUM_DOMAIN_ERROR .and. .not. fitted%fitted(), &
        "Box-Cox rejects nonpositive input transactionally", failures)
    call fitted%fit(yj_input, status)
    call check(status_ok(status) .and. fitted%fitted() .and. &
        all(fitted%lambdas() >= -2.0_dp) .and. all(fitted%lambdas() <= 2.0_dp), &
        "automatic lambda search is deterministic and bounded", failures)

    if (failures /= 0) then
        write (error_unit, '(a,i0)') "FAIL power transformer cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS power transformer independent Yeo-Johnson/Box-Cox oracle"

contains

    subroutine check(condition, description, failure_count)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failure_count

        if (.not. condition) then
            failure_count = failure_count + 1
            write (error_unit, '(a)') "  FAIL [power] "//description
        end if
    end subroutine check

end program test_power_transformer
