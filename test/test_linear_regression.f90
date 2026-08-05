program test_linear_regression
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_linear_regression, only: linear_regression_t, linear_predict_jvp, &
        linear_predict_vjp
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer :: failures
    failures = 0
    call test_fit_and_predict(failures)
    call test_fit_without_intercept(failures)
    call test_rank_deficient_fit(failures)
    call test_products(failures)
    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, " regression test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS"

contains

    subroutine test_fit_and_predict(failures)
        integer, intent(inout) :: failures
        real(dp) :: x(5, 2), y(5, 2), prediction(5, 2)
        type(linear_regression_t) :: model
        type(fortnum_status_t) :: status

        x = reshape([ &
            0.0_dp, 0.0_dp, &
            1.0_dp, 0.0_dp, &
            0.0_dp, 1.0_dp, &
            1.0_dp, 1.0_dp, &
            2.0_dp, -1.0_dp], shape(x))
        y(:, 1) = 2.0_dp + 3.0_dp*x(:, 1) - 4.0_dp*x(:, 2)
        y(:, 2) = -1.0_dp + 0.5_dp*x(:, 1) + 2.0_dp*x(:, 2)
        call model%fit(x, y, status)
        if (.not. status_ok(status)) then
            write (error_unit, '(a,a)') "FAIL [fit] ", trim(status%msg)
            failures = failures + 1
            return
        end if
        call model%predict(x, prediction, status)
        if (.not. status_ok(status) .or. maxval(abs(prediction - y)) > 1.0e-12_dp) then
            write (error_unit, '(a)') "FAIL [predict] exact multi-output fit"
            failures = failures + 1
        end if
        if (maxval(abs(model%coef(:, 1) - [2.0_dp, 3.0_dp, -4.0_dp])) > 1.0e-12_dp .or. &
            maxval(abs(model%coef(:, 2) - [-1.0_dp, 0.5_dp, 2.0_dp])) > 1.0e-12_dp) then
            write (error_unit, '(a)') "FAIL [fit] coefficients"
            failures = failures + 1
        end if
    end subroutine test_fit_and_predict

    subroutine test_fit_without_intercept(failures)
        integer, intent(inout) :: failures
        real(dp) :: x(4, 2), y(4), prediction(4)
        type(linear_regression_t) :: model
        type(fortnum_status_t) :: status

        x = reshape([ &
            1.0_dp, 0.0_dp, &
            0.0_dp, 1.0_dp, &
            2.0_dp, -1.0_dp, &
            -1.0_dp, 3.0_dp], shape(x))
        y = 2.0_dp*x(:, 1) - 3.0_dp*x(:, 2)
        call model%fit(x, y, status, fit_intercept=.false.)
        call model%predict(x, prediction, status)
        if (.not. status_ok(status) .or. maxval(abs(prediction - y)) > 1.0e-12_dp .or. &
            maxval(abs(model%coef(:, 1) - [0.0_dp, 2.0_dp, -3.0_dp])) > 1.0e-12_dp) then
            write (error_unit, '(a)') "FAIL [fit] no-intercept regression"
            failures = failures + 1
        end if
    end subroutine test_fit_without_intercept

    subroutine test_rank_deficient_fit(failures)
        integer, intent(inout) :: failures
        real(dp) :: x(6, 2), y(6), prediction(6)
        type(linear_regression_t) :: model
        type(fortnum_status_t) :: status

        x(:, 1) = [-2.0_dp, -1.0_dp, 0.0_dp, 1.0_dp, 2.0_dp, 3.0_dp]
        x(:, 2) = x(:, 1)
        y = 3.0_dp + 5.0_dp*x(:, 1)
        call model%fit(x, y, status)
        call model%predict(x, prediction, status)
        if (.not. status_ok(status) .or. maxval(abs(prediction - y)) > 1.0e-11_dp) then
            write (error_unit, '(a)') "FAIL [fit] rank-deficient least squares"
            failures = failures + 1
        end if
    end subroutine test_rank_deficient_fit

    subroutine test_products(failures)
        integer, intent(inout) :: failures
        real(dp) :: coef(3, 2), x(4, 2), dcoef(3, 2), dx(4, 2)
        real(dp) :: y(4, 2), dy(4, 2), yp(4, 2), ym(4, 2), fd(4, 2)
        real(dp) :: u(4, 2), coef_bar(3, 2), x_bar(4, 2)
        real(dp) :: lhs, rhs, h

        coef = reshape([ &
            1.0_dp, 2.0_dp, -1.0_dp, &
            -0.5_dp, 0.25_dp, 3.0_dp], shape(coef))
        x = reshape([ &
            0.1_dp, -0.2_dp, &
            0.4_dp, 0.3_dp, &
            -0.5_dp, 0.8_dp, &
            1.1_dp, -0.7_dp], shape(x))
        dcoef = reshape([ &
            0.2_dp, -0.1_dp, 0.3_dp, &
            -0.4_dp, 0.5_dp, -0.2_dp], shape(dcoef))
        dx = reshape([ &
            -0.3_dp, 0.2_dp, &
            0.1_dp, -0.4_dp, &
            0.5_dp, 0.6_dp, &
            -0.2_dp, 0.3_dp], shape(dx))
        h = 1.0e-6_dp
        call linear_predict_jvp(coef, x, dcoef, dx, y, dy)
        call reference_predict(coef + h*dcoef, x + h*dx, yp)
        call reference_predict(coef - h*dcoef, x - h*dx, ym)
        fd = (yp - ym)/(2.0_dp*h)
        if (maxval(abs(dy - fd)) > 1.0e-9_dp) then
            write (error_unit, '(a,es12.4)') "FAIL [jvp] finite difference=", &
                maxval(abs(dy - fd))
            failures = failures + 1
        end if

        u = reshape([ &
            0.4_dp, -0.2_dp, &
            0.3_dp, 0.5_dp, &
            -0.7_dp, 0.1_dp, &
            0.2_dp, -0.6_dp], shape(u))
        call linear_predict_vjp(coef, x, u, coef_bar, x_bar)
        lhs = sum(u*dy)
        rhs = sum(coef_bar*dcoef) + sum(x_bar*dx)
        if (abs(lhs - rhs) > 1.0e-12_dp) then
            write (error_unit, '(a,es12.4)') "FAIL [vjp] adjoint identity=", &
                abs(lhs - rhs)
            failures = failures + 1
        end if
    end subroutine test_products

    subroutine reference_predict(coef, x, y)
        real(dp), intent(in) :: coef(:, :), x(:, :)
        real(dp), intent(out) :: y(:, :)
        integer :: i, k

        do k = 1, size(y, 2)
            do i = 1, size(y, 1)
                y(i, k) = coef(1, k) + sum(x(i, :)*coef(2:, k))
            end do
        end do
    end subroutine reference_predict

end program test_linear_regression
