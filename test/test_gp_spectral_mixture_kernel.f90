program test_gp_spectral_mixture_kernel
    !! Independent behavioral oracle for the spectral-mixture GP kernel.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_gaussian_process, only: gp_regression_t
    use fortml_kernels, only: kernel_t, make_spectral_mixture_kernel
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer :: failures

    failures = 0
    call test_values_and_products(failures)
    call test_exact_gp_fit(failures)
    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, " spectral-mixture test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS spectral-mixture independent behavioral oracles"

contains

    subroutine test_values_and_products(failures)
        integer, intent(inout) :: failures
        type(kernel_t) :: kernel
        type(fortnum_status_t) :: status
        real(dp) :: weights(2), means(2, 2), scales(2, 2)
        real(dp) :: x1(3, 2), x2(2, 2), matrix(3, 2), matrix_dot(3, 2)
        real(dp) :: expected(3, 2), matrix_plus(3, 2), matrix_minus(3, 2)
        real(dp) :: matrix_bar(3, 2), parameter_bar(10), parameter_bar_dot(10)
        real(dp) :: parameter_plus(10), parameter_minus(10), direction(10)
        real(dp) :: value, value_plus, value_minus, gradient_x1(2), gradient_x2(2)
        real(dp) :: mixed_hessian(2, 2), mixed_reference(2, 2)
        real(dp) :: gradient_plus(2), gradient_minus(2), gradient_x2_reference(2)
        real(dp) :: gradient_x2_dummy(2)
        real(dp) :: h, lhs, rhs, parameters(10)
        integer :: i, j, d

        weights = [1.2_dp, 0.7_dp]
        means = reshape([0.25_dp, -0.4_dp, 0.8_dp, 0.15_dp], shape(means))
        scales = reshape([0.35_dp, 0.6_dp, 0.2_dp, 0.45_dp], shape(scales))
        x1 = reshape([0.0_dp, 0.4_dp, -0.7_dp, 1.1_dp, 0.5_dp, -0.3_dp], shape(x1))
        x2 = reshape([0.2_dp, -0.1_dp, 0.9_dp, 0.6_dp], shape(x2))
        kernel = make_spectral_mixture_kernel(2, 2, weights, means, scales, status)
        call check(status_ok(status), "constructor status", failures)
        call check(kernel%parameter_count() == 10, "parameter count", failures)
        parameters = kernel%parameters()
        call check(abs(parameters(1) - log(weights(1))) < 2.0e-14_dp, &
            "weight packing", failures)
        call check(abs(parameters(2) - log(scales(1, 1))) < 2.0e-14_dp, &
            "scale packing", failures)
        call check(abs(parameters(4) - means(1, 1)) < 2.0e-14_dp, &
            "frequency packing", failures)
        do j = 1, size(x2, 1)
            do i = 1, size(x1, 1)
                expected(i, j) = spectral_reference(x1(i, :), x2(j, :), weights, means, scales)
            end do
        end do
        call kernel%matrix(x1, x2, matrix, status)
        call check(status_ok(status), "matrix status", failures)
        call check(maxval(abs(matrix - expected)) < 3.0e-14_dp, &
            "dense value oracle", failures)
        call check(abs(kernel%value(x1(2, :), x2(1, :)) - expected(2, 1)) < 3.0e-14_dp, &
            "scalar value oracle", failures)

        call kernel%input_derivatives(x1(2, :), x2(1, :), value, gradient_x1, gradient_x2, &
            mixed_hessian, status)
        call check(status_ok(status), "input derivative status", failures)
        mixed_reference = mixed_hessian
        gradient_x2_reference = gradient_x2
        h = 1.0e-6_dp
        do d = 1, 2
            x1(2, d) = x1(2, d) + h
            value_plus = kernel%value(x1(2, :), x2(1, :))
            x1(2, d) = x1(2, d) - 2.0_dp*h
            value_minus = kernel%value(x1(2, :), x2(1, :))
            x1(2, d) = x1(2, d) + h
            call check(abs(gradient_x1(d) - (value_plus - value_minus)/(2.0_dp*h)) < 3.0e-8_dp, &
                "input gradient finite difference", failures)
            x2(1, d) = x2(1, d) + h
            call kernel%input_derivatives(x1(2, :), x2(1, :), value_plus, gradient_plus, &
                gradient_x2_dummy, mixed_hessian, status)
            x2(1, d) = x2(1, d) - 2.0_dp*h
            call kernel%input_derivatives(x1(2, :), x2(1, :), value_minus, gradient_minus, &
                gradient_x2_dummy, mixed_hessian, status)
            x2(1, d) = x2(1, d) + h
            call check(abs(gradient_x2_reference(d) - (value_plus - value_minus)/(2.0_dp*h)) < &
                3.0e-8_dp, "second input value finite difference", failures)
            call check(maxval(abs(mixed_reference(:, d) - (gradient_plus - gradient_minus) / &
                (2.0_dp*h))) < 2.0e-7_dp, "mixed input Hessian finite difference", failures)
        end do

        direction = [0.13_dp, -0.21_dp, 0.08_dp, 0.17_dp, -0.12_dp, &
            0.19_dp, -0.16_dp, 0.11_dp, 0.07_dp, -0.09_dp]
        call kernel%matrix_jvp(x1, x2, direction, matrix, matrix_dot, status)
        parameters = kernel%parameters()
        call kernel%set_parameters(parameters + h*direction, status)
        call kernel%matrix(x1, x2, matrix_plus, status)
        call kernel%set_parameters(parameters - h*direction, status)
        call kernel%matrix(x1, x2, matrix_minus, status)
        call kernel%set_parameters(parameters, status)
        call check(status_ok(status), "matrix JVP status", failures)
        call check(maxval(abs(matrix_dot - (matrix_plus - matrix_minus)/(2.0_dp*h))) < 4.0e-8_dp, &
            "matrix JVP finite difference", failures)

        matrix_bar = reshape([0.4_dp, -0.2_dp, 0.3_dp, 0.5_dp, -0.7_dp, 0.1_dp], shape(matrix_bar))
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
            2.0e-6_dp, "parameter HVP finite difference", failures)
    end subroutine test_values_and_products

    subroutine test_exact_gp_fit(failures)
        integer, intent(inout) :: failures
        type(kernel_t) :: kernel
        type(gp_regression_t) :: model
        type(fortnum_status_t) :: status
        real(dp) :: weights(1), means(1, 1), scales(1, 1)
        real(dp) :: x_train(3, 1), y_train(3, 1), x_query(2, 1)
        real(dp) :: mean(2, 1), variance(2)

        weights = [1.1_dp]
        means = reshape([0.35_dp], shape(means))
        scales = reshape([0.2_dp], shape(scales))
        kernel = make_spectral_mixture_kernel(1, 1, weights, means, scales, status)
        x_train = reshape([-0.5_dp, 0.0_dp, 0.7_dp], shape(x_train))
        y_train = reshape([0.3_dp, -0.2_dp, 0.8_dp], shape(y_train))
        x_query = reshape([-0.2_dp, 0.4_dp], shape(x_query))
        call model%fit(x_train, y_train, kernel, 0.08_dp, status, jitter=0.0_dp)
        call check(status_ok(status), "exact GP spectral-mixture fit status", failures)
        call model%predict(x_query, mean, variance, status)
        call check(status_ok(status), "exact GP spectral-mixture predict status", failures)
        call check(all(variance >= -1.0e-12_dp), "exact GP nonnegative variance", failures)
    end subroutine test_exact_gp_fit

    real(dp) function spectral_reference(x1, x2, weights, means, scales) result(value)
        real(dp), intent(in) :: x1(:), x2(:), weights(:), means(:, :), scales(:, :)
        real(dp) :: tau, component
        real(dp) :: two_pi
        integer :: q, d

        two_pi = 2.0_dp*acos(-1.0_dp)
        value = 0.0_dp
        do q = 1, size(weights)
            component = weights(q)
            do d = 1, size(x1)
                tau = x1(d) - x2(d)
                component = component*exp(-0.5_dp*two_pi*two_pi*tau*tau*scales(q, d)**2)* &
                    cos(two_pi*tau*means(q, d))
            end do
            value = value + component
        end do
    end function spectral_reference

    subroutine check(condition, label, failures)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: label
        integer, intent(inout) :: failures

        if (.not. condition) then
            write (error_unit, '(a)') "FAIL [spectral-mixture] "//label
            failures = failures + 1
        end if
    end subroutine check

end program test_gp_spectral_mixture_kernel
