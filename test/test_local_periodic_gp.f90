program test_local_periodic_gp
    !! Independent NumPy-style oracle for the locally-periodic GP contract.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_gaussian_process, only: gp_regression_t
    use fortml_kernel_operator, only: kernel_operator_t
    use fortml_kernels, only: kernel_t, make_local_periodic_kernel
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer :: failures

    failures = 0
    call test_kernel_products(failures)
    call test_exact_gp_integration(failures)
    call test_refusals(failures)
    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, " local-periodic test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS local-periodic GP independent behavioral oracles"

contains

    subroutine test_kernel_products(failures)
        integer, intent(inout) :: failures
        type(kernel_t) :: kernel
        type(fortnum_status_t) :: status
        real(dp) :: x1(3, 2), x2(2, 2), matrix(3, 2), expected(3, 2)
        real(dp) :: matrix_dot(3, 2), matrix_plus(3, 2), matrix_minus(3, 2)
        real(dp) :: matrix_bar(3, 2), parameter_bar(4), parameter_bar_dot(4)
        real(dp) :: parameter_plus(4), parameter_minus(4), parameters(4)
        real(dp) :: direction(4), h, lhs, rhs
        real(dp) :: value, gradient_x1(2), gradient_x2(2), hessian(2, 2)
        real(dp) :: mixed_reference(2, 2)
        real(dp) :: gradient_plus(2), gradient_minus(2), x_plus(2), x_minus(2)
        integer :: i, j, k

        x1 = reshape([0.0_dp, 0.5_dp, -0.4_dp, 1.0_dp, 1.2_dp, -0.7_dp], shape(x1))
        x2 = reshape([0.2_dp, -0.1_dp, 0.8_dp, 0.4_dp], shape(x2))
        kernel = make_local_periodic_kernel(2, 1.7_dp, 1.1_dp, 0.65_dp, 1.3_dp, status)
        call check(status_ok(status), "constructor status", failures)
        call check(kernel%parameter_count() == 4, "parameter count", failures)
        call check(maxval(abs(kernel%parameters() - [log(1.7_dp), log(1.1_dp), &
            log(0.65_dp), log(1.3_dp)])) < 2.0e-14_dp, "log parameter packing", failures)
        do j = 1, size(x2, 1)
            do i = 1, size(x1, 1)
                expected(i, j) = local_reference(1.7_dp, 1.1_dp, 0.65_dp, 1.3_dp, &
                    x1(i, :), x2(j, :))
            end do
        end do
        call kernel%matrix(x1, x2, matrix, status)
        call check(status_ok(status), "matrix status", failures)
        call check(maxval(abs(matrix - expected)) < 3.0e-14_dp, "dense value oracle", failures)

        call kernel%input_derivatives(x1(1, :), x2(1, :), value, gradient_x1, &
            gradient_x2, hessian, status)
        call check(status_ok(status), "input derivative status", failures)
        mixed_reference = hessian
        call kernel%input_derivatives(x1(1, :), x1(1, :), value, gradient_x1, &
            gradient_x2, hessian, status)
        call check(status_ok(status), "coincident input derivative status", failures)
        call check(maxval(abs(gradient_x1)) < 2.0e-13_dp .and. &
            maxval(abs(gradient_x2)) < 2.0e-13_dp .and. &
            all(hessian == hessian), "coincident input derivative limit", failures)
        call kernel%input_derivatives(x1(1, :), x2(1, :), value, gradient_x1, &
            gradient_x2, hessian, status)
        h = 1.0e-6_dp
        do k = 1, 2
            x_plus = x1(1, :)
            x_minus = x1(1, :)
            x_plus(k) = x_plus(k) + h
            x_minus(k) = x_minus(k) - h
            call check(abs(gradient_x1(k) - (kernel%value(x_plus, x2(1, :)) - &
                kernel%value(x_minus, x2(1, :)))/(2.0_dp*h)) < 4.0e-9_dp, &
                "input gradient finite difference", failures)
        end do
        do k = 1, 2
            x_plus = x2(1, :)
            x_minus = x2(1, :)
            x_plus(k) = x_plus(k) + h
            x_minus(k) = x_minus(k) - h
            call kernel%input_derivatives(x1(1, :), x_plus, value, gradient_plus, &
                gradient_x2, hessian, status)
            call kernel%input_derivatives(x1(1, :), x_minus, value, gradient_minus, &
                gradient_x2, hessian, status)
            call check(maxval(abs(mixed_reference(:, k) - &
                (gradient_plus - gradient_minus)/(2.0_dp*h))) &
                < 4.0e-6_dp, "mixed Hessian finite difference", failures)
        end do

        direction = [0.17_dp, -0.23_dp, 0.11_dp, 0.29_dp]
        parameters = kernel%parameters()
        call kernel%matrix_jvp(x1, x2, direction, matrix, matrix_dot, status)
        h = 2.0e-6_dp
        call kernel%set_parameters(parameters + h*direction, status)
        call kernel%matrix(x1, x2, matrix_plus, status)
        call kernel%set_parameters(parameters - h*direction, status)
        call kernel%matrix(x1, x2, matrix_minus, status)
        call kernel%set_parameters(parameters, status)
        call check(status_ok(status), "matrix JVP status", failures)
        call check(maxval(abs(matrix_dot - (matrix_plus - matrix_minus)/(2.0_dp*h))) < &
            4.0e-9_dp, "matrix JVP finite difference", failures)

        matrix_bar = reshape([0.4_dp, -0.2_dp, 0.3_dp, 0.5_dp, -0.7_dp, 0.1_dp], &
            shape(matrix_bar))
        call kernel%parameter_vjp(x1, x2, matrix_bar, parameter_bar, status)
        lhs = sum(matrix_bar*matrix_dot)
        rhs = sum(parameter_bar*direction)
        call check(status_ok(status), "parameter VJP status", failures)
        call check(abs(lhs - rhs) < 3.0e-11_dp, "parameter VJP adjoint identity", failures)
        call kernel%parameter_hvp(x1, x2, matrix_bar, direction, parameter_bar, &
            parameter_bar_dot, status)
        call kernel%set_parameters(parameters + h*direction, status)
        call kernel%parameter_vjp(x1, x2, matrix_bar, parameter_plus, status)
        call kernel%set_parameters(parameters - h*direction, status)
        call kernel%parameter_vjp(x1, x2, matrix_bar, parameter_minus, status)
        call kernel%set_parameters(parameters, status)
        call check(status_ok(status), "parameter HVP status", failures)
        call check(maxval(abs(parameter_bar_dot - (parameter_plus - parameter_minus)/(2.0_dp*h))) < &
            3.0e-7_dp, "parameter HVP finite difference", failures)
    end subroutine test_kernel_products

    subroutine test_exact_gp_integration(failures)
        integer, intent(inout) :: failures
        type(kernel_t) :: kernel
        type(gp_regression_t) :: gp
        type(fortnum_status_t) :: status
        real(dp) :: x_train(2, 1), y_train(2, 1), x_query(1, 1)
        real(dp) :: mean(1, 1), variance(1), expected_mean, expected_variance
        real(dp) :: k00, k01, kqq, kq0, kq1, a11, a22, det, alpha0, alpha1
        real(dp), parameter :: noise = 0.04_dp

        x_train(:, 1) = [0.0_dp, 0.7_dp]
        y_train(:, 1) = [1.2_dp, -0.3_dp]
        x_query(:, 1) = [0.25_dp]
        kernel = make_local_periodic_kernel(1, 1.8_dp, 0.95_dp, 0.7_dp, 1.4_dp, status)
        call gp%fit(x_train, y_train, kernel, noise, status)
        call check(status_ok(status), "exact GP fit status", failures)
        call gp%predict(x_query, mean, variance, status)
        call check(status_ok(status), "exact GP prediction status", failures)

        k00 = local_reference(1.8_dp, 0.95_dp, 0.7_dp, 1.4_dp, [0.0_dp], [0.0_dp])
        k01 = local_reference(1.8_dp, 0.95_dp, 0.7_dp, 1.4_dp, [0.0_dp], [0.7_dp])
        kq0 = local_reference(1.8_dp, 0.95_dp, 0.7_dp, 1.4_dp, [0.25_dp], [0.0_dp])
        kq1 = local_reference(1.8_dp, 0.95_dp, 0.7_dp, 1.4_dp, [0.25_dp], [0.7_dp])
        kqq = local_reference(1.8_dp, 0.95_dp, 0.7_dp, 1.4_dp, [0.25_dp], [0.25_dp])
        a11 = k00 + noise
        a22 = k00 + noise
        det = a11*a22 - k01*k01
        alpha0 = (a22*y_train(1, 1) - k01*y_train(2, 1))/det
        alpha1 = (-k01*y_train(1, 1) + a11*y_train(2, 1))/det
        expected_mean = kq0*alpha0 + kq1*alpha1
        expected_variance = kqq - (kq0*(a22*kq0 - k01*kq1) + &
            kq1*(-k01*kq0 + a11*kq1))/det
        call check(abs(mean(1, 1) - expected_mean) < 2.0e-11_dp, &
            "exact GP mean oracle", failures)
        call check(abs(variance(1) - expected_variance) < 2.0e-11_dp, &
            "exact GP variance oracle", failures)
    end subroutine test_exact_gp_integration

    subroutine test_refusals(failures)
        integer, intent(inout) :: failures
        type(kernel_t) :: kernel
        type(kernel_operator_t) :: operator
        type(fortnum_status_t) :: status
        real(dp) :: points(2, 1)

        points(:, 1) = [0.0_dp, 0.5_dp]
        kernel = make_local_periodic_kernel(1, 1.0_dp, 1.0_dp, 0.8_dp, 1.2_dp, status)
        call operator%initialize(points, kernel, 0.01_dp, status)
        call check(.not. status_ok(status), "typed CUDA/operator refusal", failures)
        kernel = make_local_periodic_kernel(0, 1.0_dp, 1.0_dp, 0.8_dp, 1.2_dp, status)
        call check(.not. status_ok(status), "invalid dimension refusal", failures)
    end subroutine test_refusals

    real(dp) function local_reference(variance, envelope_lengthscale, periodic_lengthscale, &
            period, x1, x2) result(value)
        real(dp), intent(in) :: variance, envelope_lengthscale, periodic_lengthscale, period
        real(dp), intent(in) :: x1(:), x2(:)
        real(dp) :: squared_distance, distance, argument, pi

        pi = acos(-1.0_dp)
        squared_distance = sum((x1 - x2)**2)
        distance = sqrt(squared_distance)
        argument = pi*distance/period
        value = variance*exp(-squared_distance/(2.0_dp*envelope_lengthscale**2) - &
            2.0_dp*sin(argument)**2/periodic_lengthscale**2)
    end function local_reference

    function finite_gradient(kernel, x1, x2, h) result(gradient)
        type(kernel_t), intent(in) :: kernel
        real(dp), intent(in) :: x1(:), x2(:), h
        real(dp) :: gradient(size(x1)), plus(size(x1)), minus(size(x1))
        integer :: i

        do i = 1, size(x1)
            plus = x1
            minus = x1
            plus(i) = plus(i) + h
            minus(i) = minus(i) - h
            gradient(i) = (kernel%value(plus, x2) - kernel%value(minus, x2))/(2.0_dp*h)
        end do
    end function finite_gradient

    subroutine check(condition, message, failures)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: message
        integer, intent(inout) :: failures

        if (.not. condition) then
            write (error_unit, '(a)') "FAIL ["//trim(message)//"]"
            failures = failures + 1
        end if
    end subroutine check

end program test_local_periodic_gp
