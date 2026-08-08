program test_kernel_change_point
    use, intrinsic :: iso_fortran_env, only: real64, error_unit
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortml_kernels, only: kernel_t, make_change_point_kernel, make_constant_kernel, &
        make_rbf_kernel
    use fortml_gaussian_process, only: gp_regression_t
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer :: failures
    type(kernel_t) :: left, right, kernel
    type(fortnum_status_t) :: status

    failures = 0
    left = make_rbf_kernel(2, 1.4_real64, 0.6_real64, status)
    if (.not. status_ok(status)) error stop "RBF constructor failed"
    right = make_constant_kernel(2, 0.4_real64, status)
    if (.not. status_ok(status)) error stop "constant constructor failed"
    kernel = make_change_point_kernel(left, right, 2, 0.15_real64, 0.7_real64, status)
    if (.not. status_ok(status)) error stop "change-point constructor failed"
    call check_kernel(kernel, failures)
    call check_gp_path(kernel, failures)
    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, " change-point test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS"

contains

    subroutine check_kernel(kernel, failures)
        type(kernel_t), intent(inout) :: kernel
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: local_status
        real(real64) :: x1(3, 2), x2(2, 2), matrix(3, 2), expected(3, 2)
        real(real64) :: matrix_dot(3, 2), matrix_plus(3, 2), matrix_minus(3, 2)
        real(real64) :: matrix_bar(3, 2), parameters(5), direction(5)
        real(real64) :: parameter_bar(5), parameter_bar_dot(5)
        real(real64) :: parameter_plus(5), parameter_minus(5)
        real(real64) :: value, gradient_x1(2), gradient_x2(2), mixed_hessian(2, 2)
        real(real64) :: x_plus(2), x_minus(2), gradient_plus(2), gradient_minus(2)
        real(real64) :: h, r2, s1, s2, gate_left, gate_right, reference
        integer :: i, j, k

        x1 = reshape([0.20_real64, -0.10_real64, 0.45_real64, 0.30_real64, &
            -0.35_real64, 0.75_real64], shape(x1))
        x2 = reshape([0.30_real64, 0.40_real64, -0.20_real64, 0.15_real64], shape(x2))
        matrix_bar = reshape([0.4_real64, -0.2_real64, 0.3_real64, 0.5_real64, &
            -0.7_real64, 0.1_real64], shape(matrix_bar))
        direction = [0.17_real64, -0.23_real64, 0.11_real64, -0.07_real64, 0.13_real64]
        parameters = kernel%parameters()
        if (kernel%parameter_count() /= 5) then
            call fail("parameter count", failures)
            return
        end if

        call kernel%matrix(x1, x2, matrix, local_status)
        if (.not. status_ok(local_status)) then
            call fail("matrix status", failures)
            return
        end if
        do j = 1, size(x2, 1)
            do i = 1, size(x1, 1)
                s1 = 0.5_real64*(1.0_real64 + tanh((x1(i, 2) - 0.15_real64)/0.7_real64))
                s2 = 0.5_real64*(1.0_real64 + tanh((x2(j, 2) - 0.15_real64)/0.7_real64))
                gate_left = s1*s2
                gate_right = (1.0_real64 - s1)*(1.0_real64 - s2)
                r2 = sum((x1(i, :) - x2(j, :))**2)
                reference = gate_left*1.4_real64*exp(-0.5_real64*r2/(0.6_real64**2)) + &
                    gate_right*0.4_real64
                expected(i, j) = reference
            end do
        end do
        if (maxval(abs(matrix - expected)) > 3.0e-13_real64) then
            call fail("matrix oracle", failures)
        end if

        call kernel%input_derivatives(x1(1, :), x2(1, :), value, gradient_x1, &
            gradient_x2, mixed_hessian, local_status)
        if (.not. status_ok(local_status)) then
            call fail("input derivative status", failures)
        else
            h = 2.0e-5_real64
            do k = 1, 2
                x_plus = x1(1, :)
                x_minus = x1(1, :)
                x_plus(k) = x_plus(k) + h
                x_minus(k) = x_minus(k) - h
                if (abs(gradient_x1(k) - (kernel%value(x_plus, x2(1, :)) - &
                    kernel%value(x_minus, x2(1, :)))/(2.0_real64*h)) > 5.0e-6_real64) then
                    call fail("input gradient", failures)
                end if
            end do
            do k = 1, 2
                x_plus = x2(1, :)
                x_minus = x2(1, :)
                x_plus(k) = x_plus(k) + h
                x_minus(k) = x_minus(k) - h
                gradient_plus = finite_gradient(kernel, x1(1, :), x_plus, h/4.0_real64)
                gradient_minus = finite_gradient(kernel, x1(1, :), x_minus, h/4.0_real64)
                if (maxval(abs(mixed_hessian(:, k) - &
                    (gradient_plus - gradient_minus)/(2.0_real64*h))) > 1.0e-4_real64) then
                    call fail("mixed hessian", failures)
                end if
            end do
        end if

        call kernel%matrix_jvp(x1, x2, direction, matrix, matrix_dot, local_status)
        h = 2.0e-6_real64
        call kernel%set_parameters(parameters + h*direction, local_status)
        call kernel%matrix(x1, x2, matrix_plus, local_status)
        call kernel%set_parameters(parameters - h*direction, local_status)
        call kernel%matrix(x1, x2, matrix_minus, local_status)
        call kernel%set_parameters(parameters, local_status)
        if (.not. status_ok(local_status) .or. maxval(abs(matrix_dot - &
            (matrix_plus - matrix_minus)/(2.0_real64*h))) > 2.0e-6_real64) then
            call fail("parameter jvp", failures)
        end if

        call kernel%parameter_vjp(x1, x2, matrix_bar, parameter_bar, local_status)
        if (.not. status_ok(local_status) .or. abs(sum(matrix_bar*matrix_dot) - &
            sum(parameter_bar*direction)) > 2.0e-9_real64) then
            call fail("parameter vjp", failures)
        end if
        call kernel%parameter_hvp(x1, x2, matrix_bar, direction, parameter_bar, &
            parameter_bar_dot, local_status)
        call kernel%set_parameters(parameters + h*direction, local_status)
        call kernel%parameter_vjp(x1, x2, matrix_bar, parameter_plus, local_status)
        call kernel%set_parameters(parameters - h*direction, local_status)
        call kernel%parameter_vjp(x1, x2, matrix_bar, parameter_minus, local_status)
        call kernel%set_parameters(parameters, local_status)
        if (.not. status_ok(local_status) .or. maxval(abs(parameter_bar_dot - &
            (parameter_plus - parameter_minus)/(2.0_real64*h))) > 2.0e-5_real64) then
            call fail("parameter hvp", failures)
        end if
    end subroutine check_kernel

    subroutine check_gp_path(kernel, failures)
        type(kernel_t), intent(in) :: kernel
        integer, intent(inout) :: failures
        type(gp_regression_t) :: model
        type(fortnum_status_t) :: local_status
        real(real64) :: x_train(4, 2), y_train(4, 1), x_query(2, 2)
        real(real64) :: mean(2, 1), variance(2), gradient(6), gradient_plus(6)
        real(real64) :: gradient_minus(6), theta(6), direction(6), h
        real(real64) :: lml_plus, lml_minus, lml

        x_train = reshape([-1.0_real64, -0.2_real64, 0.4_real64, 1.2_real64, &
            0.0_real64, 0.5_real64, 0.9_real64, 1.4_real64], shape(x_train))
        y_train(:, 1) = [0.2_real64, 0.7_real64, 1.6_real64, 1.9_real64]
        x_query = reshape([-0.5_real64, 0.6_real64, 0.2_real64, 1.0_real64], shape(x_query))
        call model%fit(x_train, y_train, kernel, 0.05_real64, local_status)
        call model%predict(x_query, mean, variance, local_status)
        if (.not. status_ok(local_status) .or. any(.not. ieee_is_finite(mean)) .or. &
            any(.not. ieee_is_finite(variance)) .or. any(variance < 0.0_real64)) then
            call fail("exact GP path", failures)
            return
        end if
        theta = model%parameters()
        direction = [0.07_real64, -0.13_real64, 0.11_real64, -0.05_real64, &
            0.09_real64, -0.08_real64]
        call model%hyperparameter_gradient(gradient, local_status)
        call model%log_marginal_likelihood(lml, local_status)
        h = 2.0e-6_real64
        call model%set_parameters(theta + h*direction, local_status)
        call model%hyperparameter_gradient(gradient_plus, local_status)
        call model%log_marginal_likelihood(lml_plus, local_status)
        call model%set_parameters(theta - h*direction, local_status)
        call model%hyperparameter_gradient(gradient_minus, local_status)
        call model%log_marginal_likelihood(lml_minus, local_status)
        call model%set_parameters(theta, local_status)
        if (.not. status_ok(local_status) .or. maxval(abs(gradient - &
            (gradient_plus + gradient_minus)/2.0_real64)) > 1.0e-4_real64 .or. &
            abs(dot_product(gradient, direction) - (lml_plus - lml_minus)/(2.0_real64*h)) > &
            5.0e-6_real64 .or. .not. ieee_is_finite(lml)) then
            call fail("GP hyperparameter gradient", failures)
        end if
    end subroutine check_gp_path

    function finite_gradient(kernel, x1, x2, h) result(gradient)
        type(kernel_t), intent(in) :: kernel
        real(real64), intent(in) :: x1(:), x2(:), h
        real(real64) :: gradient(size(x1)), plus(size(x1)), minus(size(x1))
        integer :: i

        do i = 1, size(x1)
            plus = x1
            minus = x1
            plus(i) = plus(i) + h
            minus(i) = minus(i) - h
            gradient(i) = (kernel%value(plus, x2) - kernel%value(minus, x2))/(2.0_real64*h)
        end do
    end function finite_gradient

    subroutine fail(message, failures)
        character(len=*), intent(in) :: message
        integer, intent(inout) :: failures

        write (error_unit, '(a)') "FAIL ["//trim(message)//"]"
        failures = failures + 1
    end subroutine fail

end program test_kernel_change_point
