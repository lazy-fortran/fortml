program test_ridge_regression
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_ridge_regression, only: ridge_regression_t
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer :: failures

    failures = 0
    call test_weighted_oracle(failures)
    call test_products_and_packing(failures)
    call test_refusal_contract(failures)
    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, " ridge test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS"

contains

    subroutine test_weighted_oracle(failures)
        integer, intent(inout) :: failures
        real(dp) :: x(4, 2), y(4, 2), weights(4), prediction(4, 2)
        real(dp) :: design(4, 3), gram(3, 3), rhs(3, 2), expected(3, 2)
        real(dp), allocatable :: theta(:), coefficients(:, :)
        real(dp) :: alpha
        type(ridge_regression_t) :: model
        type(fortnum_status_t) :: status
        logical :: solved
        integer :: i, j

        x(:, 1) = [-1.0_dp, 0.0_dp, 1.0_dp, 2.0_dp]
        x(:, 2) = [0.0_dp, 1.0_dp, 0.0_dp, 2.0_dp]
        y(:, 1) = 1.0_dp + 2.0_dp*x(:, 1) - 0.5_dp*x(:, 2)
        y(:, 2) = -0.5_dp + 0.25_dp*x(:, 1) + 1.5_dp*x(:, 2)
        weights = [1.0_dp, 2.0_dp, 0.5_dp, 3.0_dp]
        alpha = 0.7_dp

        design(:, 1) = 1.0_dp
        design(:, 2:) = x
        gram = 0.0_dp
        rhs = 0.0_dp
        do i = 1, size(x, 1)
            do j = 1, 3
                rhs(j, :) = rhs(j, :) + weights(i)*design(i, j)*y(i, :)
                gram(j, :) = gram(j, :) + weights(i)*design(i, j)*design(i, :)
            end do
        end do
        gram(2, 2) = gram(2, 2) + alpha
        gram(3, 3) = gram(3, 3) + alpha
        call solve_reference(gram, rhs, expected, solved)
        if (.not. solved) then
            write (error_unit, '(a)') "FAIL [oracle] reference system is singular"
            failures = failures + 1
            return
        end if

        call model%fit(x, y, status, alpha=alpha, sample_weight=weights)
        if (.not. status_ok(status)) then
            write (error_unit, '(a,a)') "FAIL [fit] ", trim(status%msg)
            failures = failures + 1
            return
        end if
        theta = model%parameters()
        if (size(theta) /= 6 .or. maxval(abs(theta - reshape(expected, [6]))) &
            > 2.0e-11_dp) then
            write (error_unit, '(a,es12.4)') "FAIL [fit] weighted oracle error=", &
                maxval(abs(theta - reshape(expected, [6])))
            failures = failures + 1
        end if
        call model%predict(x, prediction, status)
        if (.not. status_ok(status) .or. maxval(abs(prediction - &
            matmul(design, expected))) > 2.0e-11_dp) then
            write (error_unit, '(a)') "FAIL [predict] weighted ridge prediction"
            failures = failures + 1
        end if
        coefficients = model%coefficients()
        if (any(shape(coefficients) /= [3, 2]) .or. &
            maxval(abs(coefficients - expected)) > 2.0e-11_dp .or. &
            model%feature_count() /= 2 .or. model%output_count() /= 2 .or. &
            model%parameter_count() /= 6 .or. &
            abs(model%regularization() - alpha) > tiny(alpha) .or. &
            .not. model%fit_intercept() .or. .not. model%fitted()) then
            write (error_unit, '(a)') "FAIL [metadata] fitted ridge state"
            failures = failures + 1
        end if
    end subroutine test_weighted_oracle

    subroutine test_products_and_packing(failures)
        integer, intent(inout) :: failures
        real(dp) :: x(4, 2), x_dot(4, 2), weights(4), y(4, 2)
        real(dp) :: prediction(4, 2), y_dot(4, 2), y_plus(4, 2), y_minus(4, 2)
        real(dp) :: theta_dot(6), theta_bar(6), x_bar(4, 2), u(4, 2)
        real(dp) :: design(4, 3), shifted_prediction(4, 2)
        real(dp), allocatable :: theta(:), shifted(:)
        real(dp) :: lhs, rhs, h, fd_error
        type(ridge_regression_t) :: model
        type(fortnum_status_t) :: status

        x(:, 1) = [-1.0_dp, 0.0_dp, 1.0_dp, 2.0_dp]
        x(:, 2) = [0.0_dp, 1.0_dp, 0.0_dp, 2.0_dp]
        x_dot(:, 1) = [0.2_dp, -0.1_dp, 0.3_dp, -0.4_dp]
        x_dot(:, 2) = [-0.3_dp, 0.5_dp, 0.2_dp, 0.1_dp]
        y(:, 1) = 1.0_dp + 2.0_dp*x(:, 1) - 0.5_dp*x(:, 2)
        y(:, 2) = -0.5_dp + 0.25_dp*x(:, 1) + 1.5_dp*x(:, 2)
        weights = [1.0_dp, 2.0_dp, 0.5_dp, 3.0_dp]
        call model%fit(x, y, status, alpha=0.7_dp, sample_weight=weights)
        if (.not. status_ok(status)) then
            write (error_unit, '(a,a)') "FAIL [products fit] ", trim(status%msg)
            failures = failures + 1
            return
        end if
        theta = model%parameters()
        theta_dot = [0.2_dp, -0.1_dp, 0.3_dp, -0.4_dp, 0.5_dp, -0.2_dp]
        u = reshape([0.4_dp, -0.2_dp, 0.3_dp, 0.5_dp, -0.7_dp, 0.1_dp, &
            0.2_dp, -0.6_dp], shape(u))
        h = 1.0e-6_dp
        call model%predict_jvp(x, theta_dot, x_dot, prediction, y_dot, status)
        if (.not. status_ok(status)) then
            write (error_unit, '(a,a)') "FAIL [jvp] ", trim(status%msg)
            failures = failures + 1
            return
        end if
        shifted = theta + h*theta_dot
        call model%set_parameters(shifted, status)
        call model%predict(x + h*x_dot, y_plus, status)
        shifted = theta - h*theta_dot
        call model%set_parameters(shifted, status)
        call model%predict(x - h*x_dot, y_minus, status)
        fd_error = maxval(abs(y_dot - (y_plus - y_minus)/(2.0_dp*h)))
        if (fd_error > 2.0e-9_dp) then
            write (error_unit, '(a,es12.4)') "FAIL [jvp] finite difference=", fd_error
            failures = failures + 1
        end if
        call model%set_parameters(theta, status)
        call model%predict_vjp(x, u, theta_bar, x_bar, status)
        lhs = sum(u*y_dot)
        rhs = sum(theta_bar*theta_dot) + sum(x_bar*x_dot)
        if (.not. status_ok(status) .or. abs(lhs - rhs) > 2.0e-12_dp) then
            write (error_unit, '(a,es12.4)') "FAIL [vjp] adjoint error=", abs(lhs-rhs)
            failures = failures + 1
        end if

        shifted = theta
        shifted(1) = shifted(1) + 0.25_dp
        call model%set_parameters(shifted, status)
        call model%predict(x, prediction, status)
        design(:, 1) = 1.0_dp
        design(:, 2:) = x
        shifted_prediction = matmul(design, reshape(shifted, [3, 2]))
        if (.not. status_ok(status) .or. maxval(abs(prediction - &
            shifted_prediction)) > 2.0e-12_dp) then
            write (error_unit, '(a)') "FAIL [packing] set_parameters"
            failures = failures + 1
        end if
    end subroutine test_products_and_packing

    subroutine test_refusal_contract(failures)
        integer, intent(inout) :: failures
        real(dp) :: x(3, 1), y(3), weights(3), prediction(3)
        type(ridge_regression_t) :: model
        type(fortnum_status_t) :: status

        x(:, 1) = [-1.0_dp, 0.0_dp, 1.0_dp]
        y = [1.0_dp, 0.0_dp, 1.0_dp]
        weights = 0.0_dp
        call model%fit(x, y, status, sample_weight=weights)
        if (status_ok(status)) then
            write (error_unit, '(a)') "FAIL [refusal] zero sample mass accepted"
            failures = failures + 1
        end if
        weights = [1.0_dp, -1.0_dp, 1.0_dp]
        call model%fit(x, y, status, sample_weight=weights)
        if (status_ok(status)) then
            write (error_unit, '(a)') "FAIL [refusal] negative sample weight accepted"
            failures = failures + 1
        end if
        call model%predict(x, prediction, status)
        if (status_ok(status)) then
            write (error_unit, '(a)') "FAIL [refusal] unfitted prediction accepted"
            failures = failures + 1
        end if
    end subroutine test_refusal_contract

    subroutine solve_reference(matrix, right, solution, solved)
        real(dp), intent(in) :: matrix(:, :), right(:, :)
        real(dp), intent(out) :: solution(:, :)
        logical, intent(out) :: solved
        real(dp) :: a(size(matrix, 1), size(matrix, 2))
        real(dp) :: b(size(right, 1), size(right, 2))
        real(dp) :: row_a(size(matrix, 2)), row_b(size(right, 2))
        real(dp) :: factor, pivot_value
        integer :: n, m, k, i, pivot

        n = size(matrix, 1)
        m = size(right, 2)
        a = matrix
        b = right
        solved = .true.
        do k = 1, n
            pivot = k - 1 + maxloc(abs(a(k:n, k)), dim=1)
            pivot_value = a(pivot, k)
            if (abs(pivot_value) <= 1.0e-13_dp) then
                solved = .false.
                solution = 0.0_dp
                return
            end if
            if (pivot /= k) then
                row_a = a(k, :)
                a(k, :) = a(pivot, :)
                a(pivot, :) = row_a
                row_b = b(k, :)
                b(k, :) = b(pivot, :)
                b(pivot, :) = row_b
            end if
            pivot_value = a(k, k)
            a(k, :) = a(k, :)/pivot_value
            b(k, :) = b(k, :)/pivot_value
            do i = 1, n
                if (i == k) cycle
                factor = a(i, k)
                a(i, :) = a(i, :) - factor*a(k, :)
                b(i, :) = b(i, :) - factor*b(k, :)
            end do
        end do
        solution = b(:n, :m)
    end subroutine solve_reference

end program test_ridge_regression
