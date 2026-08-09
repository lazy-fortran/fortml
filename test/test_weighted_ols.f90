program test_weighted_ols
    !! Independent weighted normal-equation and fixed-state product oracle.
    use, intrinsic :: iso_fortran_env, only: real64, error_unit
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, &
        FORTML_DEVICE_CUDA
    use fortml_weighted_ols, only: weighted_ols_regression_t
    implicit none
    integer, parameter :: dp = real64
    integer :: failures

    failures = 0
    call test_weighted_multioutput_oracle(failures)
    call test_fixed_state_products(failures)
    call test_validation_and_device_contract(failures)
    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, " weighted-OLS test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS weighted OLS independent behavioral oracle"

contains

    subroutine test_weighted_multioutput_oracle(failures)
        integer, intent(inout) :: failures
        real(dp) :: x(6, 2), y(6, 2), weights(6), design(6, 3)
        real(dp) :: gram(3, 3), rhs(3, 2), expected(3, 2), prediction(6, 2)
        real(dp), allocatable :: theta(:), coefficients(:, :)
        logical :: solved
        type(weighted_ols_regression_t) :: model
        type(fortnum_status_t) :: status
        integer :: i, j

        x(:, 1) = [-1.0_dp, 0.0_dp, 1.0_dp, 2.0_dp, -2.0_dp, 0.5_dp]
        x(:, 2) = [0.0_dp, 1.0_dp, 0.0_dp, 2.0_dp, 1.5_dp, -1.0_dp]
        y(:, 1) = 0.7_dp + 1.3_dp*x(:, 1) - 0.4_dp*x(:, 2)
        y(:, 2) = -0.2_dp + 0.25_dp*x(:, 1) + 1.6_dp*x(:, 2)
        weights = [0.5_dp, 1.0_dp, 1.4_dp, 0.8_dp, 1.2_dp, 0.7_dp]
        design(:, 1) = 1.0_dp
        design(:, 2:) = x
        gram = 0.0_dp
        rhs = 0.0_dp
        do i = 1, size(x, 1)
            do j = 1, size(design, 2)
                gram(j, :) = gram(j, :) + weights(i)*design(i, j)*design(i, :)
                rhs(j, :) = rhs(j, :) + weights(i)*design(i, j)*y(i, :)
            end do
        end do
        call solve_reference(gram, rhs, expected, solved)
        call check(solved, "full-rank independent normal equation", failures)
        if (.not. solved) return

        call model%fit(x, y, status, sample_weight=weights)
        call check(status_ok(status), "weighted multi-output fit", failures)
        if (.not. status_ok(status)) return
        theta = model%parameters()
        call check(size(theta) == 6, "packed parameter count", failures)
        call check(maxval(abs(theta - reshape(expected, [6]))) < 3.0e-11_dp, &
            "weighted normal-equation coefficients", failures)
        call model%predict(x, prediction, status)
        call check(status_ok(status), "multi-output prediction", failures)
        call check(maxval(abs(prediction - matmul(design, expected))) < &
            3.0e-11_dp, "multi-output prediction oracle", failures)
        coefficients = model%coefficients()
        call check(all(shape(coefficients) == [3, 2]) .and. &
            maxval(abs(coefficients - expected)) < 3.0e-11_dp, &
            "coefficient matrix metadata", failures)
        call check(model%feature_count() == 2 .and. model%output_count() == 2 &
            .and. model%fit_intercept() .and. model%fitted(), &
            "fitted metadata", failures)
    end subroutine test_weighted_multioutput_oracle

    subroutine test_fixed_state_products(failures)
        integer, intent(inout) :: failures
        real(dp) :: x(6, 2), x_dot(6, 2), y(6, 2), weights(6)
        real(dp) :: prediction(6, 2), y_dot(6, 2), y_plus(6, 2), y_minus(6, 2)
        real(dp) :: theta_dot(6), theta_bar(6), x_bar(6, 2), u(6, 2)
        real(dp), allocatable :: theta(:), shifted(:)
        real(dp) :: h, lhs, rhs, fd_error
        type(weighted_ols_regression_t) :: model
        type(fortnum_status_t) :: status

        x(:, 1) = [-1.0_dp, 0.0_dp, 1.0_dp, 2.0_dp, -2.0_dp, 0.5_dp]
        x(:, 2) = [0.0_dp, 1.0_dp, 0.0_dp, 2.0_dp, 1.5_dp, -1.0_dp]
        x_dot(:, 1) = [0.2_dp, -0.1_dp, 0.3_dp, -0.4_dp, 0.1_dp, 0.5_dp]
        x_dot(:, 2) = [-0.3_dp, 0.5_dp, 0.2_dp, 0.1_dp, -0.2_dp, 0.4_dp]
        y(:, 1) = 0.7_dp + 1.3_dp*x(:, 1) - 0.4_dp*x(:, 2)
        y(:, 2) = -0.2_dp + 0.25_dp*x(:, 1) + 1.6_dp*x(:, 2)
        weights = [0.5_dp, 1.0_dp, 1.4_dp, 0.8_dp, 1.2_dp, 0.7_dp]
        call model%fit(x, y, status, sample_weight=weights)
        call check(status_ok(status), "products fit", failures)
        if (.not. status_ok(status)) return
        theta = model%parameters()
        theta_dot = [0.03_dp, -0.02_dp, 0.01_dp, 0.04_dp, -0.05_dp, 0.02_dp]
        u = reshape([0.02_dp, -0.03_dp, 0.04_dp, 0.01_dp, -0.05_dp, 0.06_dp, &
            0.07_dp, -0.08_dp, 0.03_dp, 0.02_dp, -0.01_dp, 0.05_dp], shape(u))
        h = 2.0e-6_dp
        call model%predict_jvp(x, theta_dot, x_dot, prediction, y_dot, status)
        call check(status_ok(status), "prediction JVP status", failures)
        shifted = theta + h*theta_dot
        call model%set_parameters(shifted, status)
        call model%predict(x + h*x_dot, y_plus, status)
        shifted = theta - h*theta_dot
        call model%set_parameters(shifted, status)
        call model%predict(x - h*x_dot, y_minus, status)
        fd_error = maxval(abs(y_dot - (y_plus-y_minus)/(2.0_dp*h)))
        call check(fd_error < 3.0e-8_dp, "prediction JVP finite difference", failures)
        call model%set_parameters(theta, status)
        call model%predict_vjp(x, u, theta_bar, x_bar, status)
        lhs = sum(u*y_dot)
        rhs = sum(theta_bar*theta_dot) + sum(x_bar*x_dot)
        call check(status_ok(status) .and. abs(lhs-rhs) < 2.0e-11_dp, &
            "prediction VJP duality", failures)
    end subroutine test_fixed_state_products

    subroutine test_validation_and_device_contract(failures)
        integer, intent(inout) :: failures
        real(dp) :: x(4, 1), y(4), weights(4), prediction(4, 1)
        type(weighted_ols_regression_t) :: model
        type(fortml_device_t) :: cuda
        type(fortnum_status_t) :: status

        x(:, 1) = [-1.0_dp, 0.0_dp, 1.0_dp, 2.0_dp]
        y = [1.0_dp, 0.0_dp, 1.0_dp, 2.0_dp]
        weights = 0.0_dp
        call model%fit(x, y, status, sample_weight=weights)
        call check(.not. status_ok(status), "zero sample mass refusal", failures)
        weights = [1.0_dp, -1.0_dp, 1.0_dp, 1.0_dp]
        call model%fit(x, y, status, sample_weight=weights)
        call check(.not. status_ok(status), "negative sample weight refusal", failures)
        call model%predict(x, prediction, status)
        call check(.not. status_ok(status), "unfitted prediction refusal", failures)
        weights = 1.0_dp
        call model%fit(x, y, status, sample_weight=weights)
        cuda%kind = FORTML_DEVICE_CUDA
        cuda%selected = .true.
        cuda%available = .true.
        call model%predict_device(cuda, x, prediction, status)
        call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
            "typed CUDA refusal", failures)
        call check(model%device_supported(FORTML_DEVICE_CUDA) .eqv. .false. &
            .and. model%device_supported(FORTML_DEVICE_CPU), &
            "device capability metadata", failures)
    end subroutine test_validation_and_device_contract

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

    subroutine check(condition, label, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: label
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "FAIL: "//trim(label)
        end if
    end subroutine check

end program test_weighted_ols
