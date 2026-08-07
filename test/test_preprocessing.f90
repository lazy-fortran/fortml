program test_preprocessing
    use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_quiet_nan
    use fortml_preprocessing, only: standard_scaler_t, minmax_scaler_t
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer :: failures

    failures = 0
    call test_standard_scaler(failures)
    call test_minmax_scaler(failures)
    call test_refusals(failures)
    if (failures > 0) then
        write (*, '(a,i0)') "FAIL preprocessing cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS preprocessing independent behavioral oracles"

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

    subroutine test_standard_scaler(failures)
        integer, intent(inout) :: failures
        type(standard_scaler_t) :: scaler
        type(fortnum_status_t) :: status
        real(dp) :: x(3, 2), transformed(3, 2), recovered(3, 2), tangent(3, 2)
        real(dp) :: transformed_tangent(3, 2)
        real(dp), allocatable :: means(:), scales(:)

        x = reshape([1.0_dp, 3.0_dp, 5.0_dp, 2.0_dp, 2.0_dp, 2.0_dp], [3, 2])
        tangent = 1.0_dp
        call scaler%fit(x, status)
        call check(status_ok(status), "standard fit", failures)
        call check(scaler%fitted(), "standard fitted flag", failures)
        means = scaler%means()
        scales = scaler%scales()
        call check(maxval(abs(means - [3.0_dp, 2.0_dp])) < 1.0e-14_dp, &
            "standard means", failures)
        call check(maxval(abs(scales - [sqrt(8.0_dp/3.0_dp), 1.0_dp])) < &
            1.0e-14_dp, "standard scales", failures)
        call scaler%transform(x, transformed, status)
        call scaler%inverse_transform(transformed, recovered, status)
        call check(status_ok(status), "standard transforms", failures)
        call check(maxval(abs(recovered - x)) < 1.0e-14_dp, &
            "standard inverse oracle", failures)
        call scaler%transform_jvp(tangent, transformed_tangent, status)
        call check(status_ok(status), "standard JVP", failures)
        call check(maxval(abs(transformed_tangent(:, 1) - &
            tangent(:, 1)/sqrt(8.0_dp/3.0_dp))) < 1.0e-14_dp, &
            "standard JVP scale", failures)
        call check(maxval(abs(transformed_tangent(:, 2) - tangent(:, 2))) < &
            1.0e-14_dp, "constant-feature JVP", failures)
    end subroutine test_standard_scaler

    subroutine test_minmax_scaler(failures)
        integer, intent(inout) :: failures
        type(minmax_scaler_t) :: scaler
        type(fortnum_status_t) :: status
        real(dp) :: x(3, 2), transformed(3, 2), recovered(3, 2)
        real(dp) :: tangent(3, 2), transformed_tangent(3, 2)

        x = reshape([1.0_dp, 3.0_dp, 5.0_dp, 2.0_dp, 2.0_dp, 2.0_dp], [3, 2])
        tangent = 1.0_dp
        call scaler%fit(x, status, feature_range=[-1.0_dp, 1.0_dp])
        call check(status_ok(status), "minmax fit", failures)
        call scaler%transform(x, transformed, status)
        call scaler%inverse_transform(transformed, recovered, status)
        call check(status_ok(status), "minmax transforms", failures)
        call check(maxval(abs(transformed(:, 1) - [-1.0_dp, 0.0_dp, 1.0_dp])) < &
            1.0e-14_dp, "minmax range oracle", failures)
        call check(maxval(abs(transformed(:, 2) + 1.0_dp)) < 1.0e-14_dp, &
            "minmax constant feature", failures)
        call check(maxval(abs(recovered - x)) < 1.0e-14_dp, &
            "minmax inverse oracle", failures)
        call scaler%transform_jvp(tangent, transformed_tangent, status)
        call check(status_ok(status), "minmax JVP", failures)
        call check(maxval(abs(transformed_tangent(:, 1) - 0.5_dp)) < &
            1.0e-14_dp, "minmax JVP scale", failures)
        call check(maxval(abs(transformed_tangent(:, 2) - 2.0_dp)) < &
            1.0e-14_dp, "minmax constant JVP", failures)
    end subroutine test_minmax_scaler

    subroutine test_refusals(failures)
        integer, intent(inout) :: failures
        type(standard_scaler_t) :: standard
        type(minmax_scaler_t) :: minmax
        type(fortnum_status_t) :: status
        real(dp) :: x(2, 1), output(2, 1)

        x = 0.0_dp
        call standard%transform(x, output, status)
        call check(.not. status_ok(status), "unfitted standard refusal", failures)
        call minmax%fit(x, status, feature_range=[1.0_dp, 1.0_dp])
        call check(.not. status_ok(status), "invalid minmax range refusal", failures)
        x(1, 1) = ieee_value(0.0_dp, ieee_quiet_nan)
        call standard%fit(x, status)
        call check(.not. status_ok(status), "nonfinite standard refusal", failures)
    end subroutine test_refusals

end program test_preprocessing
