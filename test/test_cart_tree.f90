program test_cart_tree
    use, intrinsic :: ieee_arithmetic, only: ieee_quiet_nan, ieee_value
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_tree, only: cart_regressor_t
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_DOMAIN_ERROR
    implicit none

    integer :: failures

    failures = 0
    call test_depth_one_oracle(failures)
    call test_depth_two_and_jvp(failures)
    call test_weighted_leaf_oracle(failures)
    call test_refusals(failures)
    if (failures /= 0) then
        write (error_unit, '(a,i0)') "FAIL CART regression cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS CART regression independent behavioral oracles"

contains

    subroutine test_depth_one_oracle(failures)
        integer, intent(inout) :: failures
        type(cart_regressor_t) :: model
        type(fortnum_status_t) :: status
        real(dp) :: x(6, 1), y(6), prediction(6)

        x(:, 1) = [0.0_dp, 1.0_dp, 2.0_dp, 3.0_dp, 4.0_dp, 5.0_dp]
        y = [0.0_dp, 0.0_dp, 0.0_dp, 10.0_dp, 10.0_dp, 10.0_dp]
        call model%fit(x, y, status, max_depth=1)
        call model%predict(x, prediction, status)
        if (.not. status_ok(status) .or. model%node_count() /= 3 .or. &
            model%input_count() /= 1 .or. maxval(abs(prediction - y)) > 1.0e-12_dp) then
            write (error_unit, '(a)') "FAIL [CART depth one] exhaustive split oracle"
            failures = failures + 1
        end if
    end subroutine test_depth_one_oracle

    subroutine test_depth_two_and_jvp(failures)
        integer, intent(inout) :: failures
        type(cart_regressor_t) :: model
        type(fortnum_status_t) :: status
        real(dp) :: x(8, 1), y(8), query(4, 1), tangent(4, 1)
        real(dp) :: prediction(4), prediction_dot(4), expected(4)

        x(:, 1) = [0.0_dp, 1.0_dp, 2.0_dp, 3.0_dp, 4.0_dp, 5.0_dp, &
            6.0_dp, 7.0_dp]
        y = [0.0_dp, 0.0_dp, 2.0_dp, 2.0_dp, 8.0_dp, 8.0_dp, 10.0_dp, &
            10.0_dp]
        call model%fit(x, y, status, max_depth=2)
        query(:, 1) = [0.25_dp, 1.75_dp, 4.25_dp, 6.75_dp]
        expected = [0.0_dp, 2.0_dp, 8.0_dp, 10.0_dp]
        tangent(:, 1) = [0.1_dp, -0.2_dp, 0.3_dp, -0.4_dp]
        call model%predict_jvp(query, tangent, prediction, prediction_dot, status)
        if (.not. status_ok(status) .or. model%node_count() /= 7 .or. &
            maxval(abs(prediction - expected)) > 1.0e-12_dp .or. &
            maxval(abs(prediction_dot)) > 1.0e-14_dp) then
            write (error_unit, '(a)') "FAIL [CART depth two] recursive split oracle"
            failures = failures + 1
        end if

        query(1, 1) = 3.5_dp
        call model%predict_jvp(query, tangent(1:1, :), prediction(1:1), &
            prediction_dot(1:1), status)
        if (status%code /= FORTNUM_DOMAIN_ERROR) then
            write (error_unit, '(a)') "FAIL [CART JVP] split-boundary refusal"
            failures = failures + 1
        end if
    end subroutine test_depth_two_and_jvp

    subroutine test_weighted_leaf_oracle(failures)
        integer, intent(inout) :: failures
        type(cart_regressor_t) :: model
        type(fortnum_status_t) :: status
        real(dp) :: x(4, 1), y(4), weights(4), query(2, 1), prediction(2)
        real(dp) :: expected(2)

        x(:, 1) = [0.0_dp, 1.0_dp, 2.0_dp, 3.0_dp]
        y = [0.0_dp, 2.0_dp, 10.0_dp, 12.0_dp]
        weights = [1.0_dp, 3.0_dp, 1.0_dp, 1.0_dp]
        call model%fit(x, y, status, max_depth=1, sample_weight=weights)
        query(:, 1) = [0.5_dp, 2.5_dp]
        expected = [1.5_dp, 11.0_dp]
        call model%predict(query, prediction, status)
        if (.not. status_ok(status) .or. maxval(abs(prediction - expected)) > 1.0e-12_dp) then
            write (error_unit, '(a)') "FAIL [CART weighted] weighted mean oracle"
            failures = failures + 1
        end if
    end subroutine test_weighted_leaf_oracle

    subroutine test_refusals(failures)
        integer, intent(inout) :: failures
        type(cart_regressor_t) :: model
        type(fortnum_status_t) :: status
        real(dp) :: x(4, 1), y(4), query(2, 1), prediction(2)
        real(dp) :: nan_value

        nan_value = ieee_value(0.0_dp, ieee_quiet_nan)
        x(:, 1) = [0.0_dp, 1.0_dp, 2.0_dp, 3.0_dp]
        y = [0.0_dp, 0.0_dp, 1.0_dp, 1.0_dp]
        query(:, 1) = [0.25_dp, 2.25_dp]
        x(2, 1) = nan_value
        call model%fit(x, y, status)
        if (status%code /= FORTNUM_DOMAIN_ERROR) then
            write (error_unit, '(a)') "FAIL [CART refusal] nonfinite fit"
            failures = failures + 1
        end if
        x(2, 1) = 1.0_dp
        call model%fit(x, y, status)
        query(1, 1) = nan_value
        call model%predict(query, prediction, status)
        if (status%code /= FORTNUM_DOMAIN_ERROR) then
            write (error_unit, '(a)') "FAIL [CART refusal] nonfinite prediction"
            failures = failures + 1
        end if
    end subroutine test_refusals

end program test_cart_tree
