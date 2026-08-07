program test_gp_ard_kernel
    !! Independent behavioral oracle for anisotropic squared-exponential kernels.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_gaussian_process, only: gp_regression_t
    use fortml_kernels, only: kernel_t, make_rbf_ard_kernel, make_ard_rbf_kernel
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer :: failures

    failures = 0
    call test_constructor_and_values(failures)
    call test_input_products(failures)
    call test_parameter_products(failures)
    call test_exact_gp_integration(failures)
    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, " ARD RBF test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS ARD RBF independent behavioral oracles"

contains

    subroutine test_constructor_and_values(failures)
        integer, intent(inout) :: failures
        type(kernel_t) :: kernel, alias_kernel
        type(fortnum_status_t) :: status
        real(dp) :: x1(3, 3), x2(2, 3), matrix(3, 2), expected(3, 2)
        real(dp), parameter :: variance = 2.5_dp
        real(dp), parameter :: lengthscales(3) = [0.7_dp, 1.3_dp, 2.0_dp]
        integer :: i, j

        x1 = reshape([0.0_dp, 0.5_dp, -0.4_dp, 1.0_dp, 1.2_dp, -0.7_dp, &
            -0.2_dp, 0.9_dp, 0.4_dp], shape(x1))
        x2 = reshape([0.2_dp, -0.1_dp, 0.8_dp, 0.4_dp, -0.6_dp, 0.3_dp], shape(x2))
        kernel = make_rbf_ard_kernel(3, variance, lengthscales, status)
        call check(status_ok(status), "constructor status", failures)
        call check(kernel%parameter_count() == 4, "parameter count", failures)
        call check(maxval(abs(kernel%parameters() - [log(variance), log(lengthscales)])) < &
            2.0e-14_dp, "log parameter packing", failures)
        do j = 1, size(x2, 1)
            do i = 1, size(x1, 1)
                expected(i, j) = ard_reference(variance, lengthscales, x1(i, :), x2(j, :))
            end do
        end do
        call kernel%matrix(x1, x2, matrix, status)
        call check(status_ok(status), "matrix status", failures)
        call check(maxval(abs(matrix - expected)) < 2.0e-14_dp, "dense value oracle", failures)
        call check(abs(kernel%value(x1(2, :), x2(1, :)) - expected(2, 1)) < 2.0e-14_dp, &
            "scalar value oracle", failures)

        alias_kernel = make_ard_rbf_kernel(3, variance, lengthscales, status)
        call alias_kernel%matrix(x1, x2, matrix, status)
        call check(status_ok(status), "alias constructor status", failures)
        call check(maxval(abs(matrix - expected)) < 2.0e-14_dp, "alias value oracle", failures)

        kernel = make_rbf_ard_kernel(3, variance, [1.0_dp, 2.0_dp], status)
        call check(.not. status_ok(status), "invalid lengthscale shape refusal", failures)
    end subroutine test_constructor_and_values

    subroutine test_input_products(failures)
        integer, intent(inout) :: failures
        type(kernel_t) :: kernel
        type(fortnum_status_t) :: status
        real(dp), parameter :: variance = 1.7_dp
        real(dp), parameter :: lengthscales(3) = [0.6_dp, 1.1_dp, 1.8_dp]
        real(dp) :: x1(3), x2(3), value, value_plus, value_minus, h
        real(dp) :: gradient_x1(3), gradient_x2(3), mixed_hessian(3, 3)
        real(dp) :: mixed_reference(3, 3)
        real(dp) :: gradient_plus(3), gradient_minus(3), dummy_hessian(3, 3)
        integer :: i

        kernel = make_rbf_ard_kernel(3, variance, lengthscales, status)
        x1 = [0.2_dp, -0.3_dp, 0.7_dp]
        x2 = [-0.1_dp, 0.8_dp, -0.4_dp]
        call kernel%input_derivatives(x1, x2, value, gradient_x1, gradient_x2, &
            mixed_hessian, status)
        mixed_reference = mixed_hessian
        call check(status_ok(status), "input derivative status", failures)
        h = 1.0e-6_dp
        do i = 1, 3
            x1(i) = x1(i) + h
            value_plus = kernel%value(x1, x2)
            x1(i) = x1(i) - 2.0_dp*h
            value_minus = kernel%value(x1, x2)
            x1(i) = x1(i) + h
            call check(abs(gradient_x1(i) - (value_plus - value_minus)/(2.0_dp*h)) < &
                3.0e-9_dp, "input gradient finite difference", failures)

            x2(i) = x2(i) + h
            call kernel%input_derivatives(x1, x2, value_plus, gradient_plus, gradient_x2, &
                dummy_hessian, status)
            x2(i) = x2(i) - 2.0_dp*h
            call kernel%input_derivatives(x1, x2, value_minus, gradient_minus, gradient_x2, &
                dummy_hessian, status)
            x2(i) = x2(i) + h
            call check(maxval(abs(mixed_reference(:, i) - (gradient_plus - gradient_minus) / &
                (2.0_dp*h))) < 2.0e-7_dp, "mixed input Hessian finite difference", failures)
        end do
    end subroutine test_input_products

    subroutine test_parameter_products(failures)
        integer, intent(inout) :: failures
        type(kernel_t) :: kernel
        type(fortnum_status_t) :: status
        real(dp), parameter :: variance = 2.1_dp
        real(dp), parameter :: lengthscales(3) = [0.8_dp, 1.2_dp, 1.6_dp]
        real(dp) :: x1(3, 3), x2(2, 3), matrix(3, 2), matrix_dot(3, 2)
        real(dp) :: matrix_plus(3, 2), matrix_minus(3, 2), matrix_bar(3, 2)
        real(dp) :: direction(4), parameters(4), parameter_bar(4), parameter_bar_dot(4)
        real(dp) :: parameter_plus(4), parameter_minus(4), h, lhs, rhs

        x1 = reshape([0.0_dp, 0.5_dp, -0.4_dp, 1.0_dp, 1.2_dp, -0.7_dp, &
            -0.2_dp, 0.9_dp, 0.4_dp], shape(x1))
        x2 = reshape([0.2_dp, -0.1_dp, 0.8_dp, 0.4_dp, -0.6_dp, 0.3_dp], shape(x2))
        matrix_bar = reshape([0.4_dp, -0.2_dp, 0.3_dp, 0.5_dp, -0.7_dp, 0.1_dp], &
            shape(matrix_bar))
        direction = [0.17_dp, -0.23_dp, 0.11_dp, 0.29_dp]
        kernel = make_rbf_ard_kernel(3, variance, lengthscales, status)
        call kernel%matrix_jvp(x1, x2, direction, matrix, matrix_dot, status)
        parameters = kernel%parameters()
        h = 1.0e-6_dp
        call kernel%set_parameters(parameters + h*direction, status)
        call kernel%matrix(x1, x2, matrix_plus, status)
        call kernel%set_parameters(parameters - h*direction, status)
        call kernel%matrix(x1, x2, matrix_minus, status)
        call kernel%set_parameters(parameters, status)
        call check(status_ok(status), "matrix JVP status", failures)
        call check(maxval(abs(matrix_dot - (matrix_plus - matrix_minus)/(2.0_dp*h))) < &
            3.0e-9_dp, "matrix JVP finite difference", failures)

        call kernel%parameter_vjp(x1, x2, matrix_bar, parameter_bar, status)
        lhs = sum(matrix_bar*matrix_dot)
        rhs = sum(parameter_bar*direction)
        call check(status_ok(status), "parameter VJP status", failures)
        call check(abs(lhs - rhs) < 2.0e-12_dp, "parameter VJP adjoint identity", failures)

        call kernel%parameter_hvp(x1, x2, matrix_bar, direction, parameter_bar, &
            parameter_bar_dot, status)
        call kernel%set_parameters(parameters + h*direction, status)
        call kernel%parameter_vjp(x1, x2, matrix_bar, parameter_plus, status)
        call kernel%set_parameters(parameters - h*direction, status)
        call kernel%parameter_vjp(x1, x2, matrix_bar, parameter_minus, status)
        call kernel%set_parameters(parameters, status)
        call check(status_ok(status), "parameter HVP status", failures)
        call check(maxval(abs(parameter_bar_dot - (parameter_plus - parameter_minus) / &
            (2.0_dp*h))) < 4.0e-8_dp, "parameter HVP finite difference", failures)
    end subroutine test_parameter_products

    subroutine test_exact_gp_integration(failures)
        integer, intent(inout) :: failures
        type(kernel_t) :: kernel
        type(gp_regression_t) :: model
        type(fortnum_status_t) :: status
        real(dp), parameter :: variance = 1.5_dp
        real(dp), parameter :: noise_variance = 0.2_dp
        real(dp), parameter :: lengthscales(3) = [0.8_dp, 1.4_dp, 2.1_dp]
        real(dp) :: x_train(1, 3), y_train(1, 1), x_query(1, 3)
        real(dp) :: mean(1, 1), posterior_variance(1), cross_value, expected_mean
        real(dp) :: expected_variance, denominator, lml, lml_plus, lml_minus, h
        real(dp) :: gp_parameters(5), gp_gradient(5), plus_parameters(5), minus_parameters(5)
        integer :: i

        x_train = 0.0_dp
        y_train = 2.0_dp
        x_query = reshape([1.0_dp, -0.5_dp, 0.25_dp], shape(x_query))
        kernel = make_rbf_ard_kernel(3, variance, lengthscales, status)
        call model%fit(x_train, y_train, kernel, noise_variance, status, jitter=0.0_dp)
        call check(status_ok(status), "exact GP ARD fit status", failures)
        call model%predict(x_query, mean, posterior_variance, status)
        call check(status_ok(status), "exact GP ARD prediction status", failures)
        cross_value = ard_reference(variance, lengthscales, x_train(1, :), x_query(1, :))
        denominator = variance + noise_variance
        expected_mean = cross_value/denominator*y_train(1, 1)
        expected_variance = variance - cross_value*cross_value/denominator
        call check(abs(mean(1, 1) - expected_mean) < 2.0e-12_dp, &
            "exact GP ARD posterior mean oracle", failures)
        call check(abs(posterior_variance(1) - expected_variance) < 2.0e-12_dp, &
            "exact GP ARD posterior variance oracle", failures)

        call model%log_marginal_likelihood(lml, status)
        call model%hyperparameter_gradient(gp_gradient, status)
        gp_parameters = model%parameters()
        h = 1.0e-6_dp
        do i = 1, size(gp_parameters)
            plus_parameters = gp_parameters
            minus_parameters = gp_parameters
            plus_parameters(i) = plus_parameters(i) + h
            minus_parameters(i) = minus_parameters(i) - h
            call model%set_parameters(plus_parameters, status)
            call model%log_marginal_likelihood(lml_plus, status)
            call model%set_parameters(minus_parameters, status)
            call model%log_marginal_likelihood(lml_minus, status)
            call check(abs(gp_gradient(i) - (lml_plus - lml_minus)/(2.0_dp*h)) < &
                3.0e-7_dp, "exact GP ARD LML gradient finite difference", failures)
        end do
        call model%set_parameters(gp_parameters, status)
    end subroutine test_exact_gp_integration

    real(dp) function ard_reference(variance, lengthscales, x1, x2) result(value)
        real(dp), intent(in) :: variance, lengthscales(:), x1(:), x2(:)
        real(dp) :: weighted_squared_distance
        integer :: i

        weighted_squared_distance = 0.0_dp
        do i = 1, size(lengthscales)
            weighted_squared_distance = weighted_squared_distance + &
                (x1(i) - x2(i))**2/(lengthscales(i)*lengthscales(i))
        end do
        value = variance*exp(-0.5_dp*weighted_squared_distance)
    end function ard_reference

    subroutine check(condition, label, failures)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: label
        integer, intent(inout) :: failures

        if (.not. condition) then
            write (error_unit, '(a)') "FAIL [ARD RBF] "//label
            failures = failures + 1
        end if
    end subroutine check

end program test_gp_ard_kernel
