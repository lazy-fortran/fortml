program test_kernel_cosine_polynomial
    use, intrinsic :: iso_fortran_env, only: real64, error_unit
    use fortml_kernels, only: kernel_t, make_cosine_kernel, make_polynomial_kernel
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer :: failures
    type(kernel_t) :: cosine, polynomial
    type(fortnum_status_t) :: status

    failures = 0
    cosine = make_cosine_kernel(2, 1.4_real64, 0.9_real64, status)
    if (.not. status_ok(status)) error stop "cosine constructor failed"
    polynomial = make_polynomial_kernel(2, 1.1_real64, 0.7_real64, 1.5_real64, &
        2.3_real64, status)
    if (.not. status_ok(status)) error stop "polynomial constructor failed"
    call check_kernel(cosine, "cosine", 2, failures)
    call check_kernel(polynomial, "polynomial", 4, failures)
    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, " cosine/polynomial test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS"

contains

    subroutine check_kernel(kernel, label, n_parameters, failures)
        type(kernel_t), intent(inout) :: kernel
        character(len=*), intent(in) :: label
        integer, intent(in) :: n_parameters
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: local_status
        real(real64) :: x1(3, 2), x2(2, 2), matrix(3, 2), matrix_dot(3, 2)
        real(real64) :: matrix_plus(3, 2), matrix_minus(3, 2), matrix_bar(3, 2)
        real(real64) :: parameters(n_parameters), direction(n_parameters)
        real(real64) :: parameter_bar(n_parameters), parameter_bar_dot(n_parameters)
        real(real64) :: parameter_plus(n_parameters), parameter_minus(n_parameters)
        real(real64) :: value, gradient_x1(2), gradient_x2(2), mixed_hessian(2, 2)
        real(real64) :: gradient_plus(2), gradient_minus(2), x_plus(2), x_minus(2)
        real(real64) :: expected, dot_value, h, r2, inner_product
        integer :: i, j, k

        x1 = reshape([0.20_real64, -0.10_real64, 0.45_real64, 0.30_real64, &
            -0.35_real64, 0.75_real64], shape(x1))
        x2 = reshape([0.30_real64, 0.40_real64, -0.20_real64, 0.15_real64], shape(x2))
        matrix_bar = reshape([0.4_real64, -0.2_real64, 0.3_real64, 0.5_real64, &
            -0.7_real64, 0.1_real64], shape(matrix_bar))
        direction = 0.0_real64
        direction(1:2) = [0.17_real64, -0.23_real64]
        if (n_parameters > 2) direction(3:n_parameters) = [0.11_real64, -0.07_real64]
        parameters = kernel%log_parameters
        if (kernel%parameter_count() /= n_parameters) then
            call fail(label//" parameter count", failures)
            return
        end if

        call kernel%matrix(x1, x2, matrix, local_status)
        if (.not. status_ok(local_status)) then
            call fail(label//" matrix status", failures)
            return
        end if
        r2 = sum((x1(1, :) - x2(1, :))**2)
        inner_product = dot_product(x1(1, :), x2(1, :))
        if (index(label, "cosine") == 1) then
            expected = 1.4_real64*cos(sqrt(r2)/0.9_real64)
        else
            expected = 1.1_real64*(1.5_real64 + 0.7_real64*inner_product)**2.3_real64
        end if
        if (abs(matrix(1, 1) - expected) > 3.0e-13_real64) then
            call fail(label//" matrix value", failures)
        end if

        call kernel%input_derivatives(x1(1, :), x2(1, :), value, gradient_x1, &
            gradient_x2, mixed_hessian, local_status)
        if (.not. status_ok(local_status)) then
            call fail(label//" input derivative status", failures)
        else
            h = 2.0e-5_real64
            do k = 1, 2
                x_plus = x1(1, :)
                x_minus = x1(1, :)
                x_plus(k) = x_plus(k) + h
                x_minus(k) = x_minus(k) - h
                if (abs(gradient_x1(k) - (kernel%value(x_plus, x2(1, :)) - &
                    kernel%value(x_minus, x2(1, :)))/(2.0_real64*h)) > 4.0e-6_real64) then
                    call fail(label//" input gradient", failures)
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
                    (gradient_plus - gradient_minus)/(2.0_real64*h))) > 7.0e-5_real64) then
                    call fail(label//" mixed hessian", failures)
                end if
            end do
        end if

        call kernel%matrix_jvp(x1, x2, direction(:n_parameters), matrix, matrix_dot, local_status)
        h = 2.0e-6_real64
        call kernel%set_parameters(parameters + h*direction(:n_parameters), local_status)
        call kernel%matrix(x1, x2, matrix_plus, local_status)
        call kernel%set_parameters(parameters - h*direction(:n_parameters), local_status)
        call kernel%matrix(x1, x2, matrix_minus, local_status)
        call kernel%set_parameters(parameters, local_status)
        if (.not. status_ok(local_status) .or. maxval(abs(matrix_dot - &
            (matrix_plus - matrix_minus)/(2.0_real64*h))) > 8.0e-7_real64) then
            call fail(label//" parameter jvp", failures)
        end if

        call kernel%parameter_vjp(x1, x2, matrix_bar, parameter_bar, local_status)
        dot_value = sum(matrix_bar*matrix_dot)
        if (.not. status_ok(local_status) .or. abs(dot_value - &
            sum(parameter_bar*direction(:n_parameters))) > 5.0e-10_real64) then
            call fail(label//" parameter vjp", failures)
        end if

        call kernel%parameter_hvp(x1, x2, matrix_bar, direction(:n_parameters), &
            parameter_bar, parameter_bar_dot, local_status)
        call kernel%set_parameters(parameters + h*direction(:n_parameters), local_status)
        call kernel%parameter_vjp(x1, x2, matrix_bar, parameter_plus, local_status)
        call kernel%set_parameters(parameters - h*direction(:n_parameters), local_status)
        call kernel%parameter_vjp(x1, x2, matrix_bar, parameter_minus, local_status)
        call kernel%set_parameters(parameters, local_status)
        if (.not. status_ok(local_status) .or. maxval(abs(parameter_bar_dot - &
            (parameter_plus - parameter_minus)/(2.0_real64*h))) > 1.5e-6_real64) then
            call fail(label//" parameter hvp", failures)
        end if
    end subroutine check_kernel

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

end program test_kernel_cosine_polynomial
