program test_kernel_catalog
    use, intrinsic :: iso_fortran_env, only: real64, error_unit
    use fortml_kernels, only: kernel_t, make_periodic_kernel, &
        make_rational_quadratic_kernel
    use fortml_kernel_operator, only: kernel_operator_t
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer :: failures
    type(kernel_t) :: periodic, rational_quadratic
    type(fortnum_status_t) :: status

    failures = 0
    periodic = make_periodic_kernel(2, 1.7_real64, 0.8_real64, 1.3_real64, status)
    if (.not. status_ok(status)) error stop "periodic constructor failed"
    rational_quadratic = make_rational_quadratic_kernel(2, 1.7_real64, 0.8_real64, &
        1.4_real64, status)
    if (.not. status_ok(status)) error stop "rational quadratic constructor failed"
    call check_kernel(periodic, "periodic", failures)
    call check_kernel(rational_quadratic, "rational_quadratic", failures)
    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, " kernel catalog test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS"

contains

    subroutine check_kernel(kernel, label, failures)
        type(kernel_t), intent(inout) :: kernel
        character(len=*), intent(in) :: label
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status
        real(real64) :: x1(3, 2), x2(2, 2), matrix(3, 2), expected(3, 2)
        real(real64) :: value, gradient_x1(2), gradient_x2(2), hessian(2, 2)
        real(real64) :: gradient_plus(2), gradient_minus(2), x_plus(2), x_minus(2)
        real(real64) :: direction(3), matrix_dot(3, 2), matrix_plus(3, 2), matrix_minus(3, 2)
        real(real64) :: matrix_bar(3, 2), parameter_bar(3), parameter_bar_dot(3)
        real(real64) :: parameter_plus(3), parameter_minus(3), parameters(3)
        real(real64) :: h, lhs, rhs, r2
        type(kernel_operator_t) :: operator
        integer :: i, j, k

        x1 = reshape([0.0_real64, 0.5_real64, -0.4_real64, 1.0_real64, 1.2_real64, -0.7_real64], &
            shape(x1))
        x2 = reshape([0.2_real64, -0.1_real64, 0.8_real64, 0.4_real64], shape(x2))
        call kernel%matrix(x1, x2, matrix, status)
        if (.not. status_ok(status)) then
            call fail(label//" matrix status", failures)
            return
        end if
        do j = 1, size(x2, 1)
            do i = 1, size(x1, 1)
                r2 = sum((x1(i, :) - x2(j, :))**2)
                if (index(label, "periodic") == 1) then
                    expected(i, j) = 1.7_real64*exp(-2.0_real64* &
                        sin(acos(-1.0_real64)*sqrt(r2)/1.3_real64)**2/0.8_real64**2)
                else
                    expected(i, j) = 1.7_real64*(1.0_real64 + r2/(2.0_real64*1.4_real64*0.8_real64**2))** &
                        (-1.4_real64)
                end if
            end do
        end do
        if (maxval(abs(matrix - expected)) > 3.0e-13_real64) then
            call fail(label//" matrix value", failures)
        end if
        if (kernel%parameter_count() /= 3) call fail(label//" parameter count", failures)
        call operator%initialize(x1, kernel, 0.01_real64, status)
        if (status_ok(status)) call fail(label//" operator device refusal", failures)

        call kernel%input_derivatives(x1(1, :), x2(1, :), value, gradient_x1, &
            gradient_x2, hessian, status)
        if (.not. status_ok(status)) then
            call fail(label//" input derivative status", failures)
        else
            call kernel%input_derivatives(x1(1, :), x1(1, :), value, gradient_x1, &
                gradient_x2, hessian, status)
            if (.not. status_ok(status) .or. maxval(abs(gradient_x1)) > 2.0e-13_real64 .or. &
                maxval(abs(gradient_x2)) > 2.0e-13_real64 .or. &
                any(.not. (hessian == hessian))) then
                call fail(label//" coincident input limit", failures)
            end if
            call kernel%input_derivatives(x1(1, :), x2(1, :), value, gradient_x1, &
                gradient_x2, hessian, status)
            h = 2.0e-4_real64
            do k = 1, 2
                x_plus = x1(1, :)
                x_minus = x1(1, :)
                x_plus(k) = x_plus(k) + h
                x_minus(k) = x_minus(k) - h
                if (abs(gradient_x1(k) - (kernel%value(x_plus, x2(1, :)) - &
                    kernel%value(x_minus, x2(1, :)))/(2.0_real64*h)) > 2.0e-6_real64) then
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
                if (maxval(abs(hessian(:, k) - (gradient_plus - gradient_minus)/(2.0_real64*h))) > &
                    3.0e-5_real64) then
                    call fail(label//" mixed hessian", failures)
                end if
            end do
            if (abs(value - kernel%value(x1(1, :), x2(1, :))) > 2.0e-13_real64) then
                call fail(label//" derivative value", failures)
            end if
        end if

        direction = [0.17_real64, -0.23_real64, 0.11_real64]
        call kernel%matrix_jvp(x1, x2, direction, matrix, matrix_dot, status)
        parameters = kernel%parameters()
        h = 2.0e-6_real64
        call kernel%set_parameters(parameters + h*direction, status)
        call kernel%matrix(x1, x2, matrix_plus, status)
        call kernel%set_parameters(parameters - h*direction, status)
        call kernel%matrix(x1, x2, matrix_minus, status)
        call kernel%set_parameters(parameters, status)
        if (.not. status_ok(status) .or. maxval(abs(matrix_dot - &
            (matrix_plus - matrix_minus)/(2.0_real64*h))) > 4.0e-7_real64) then
            call fail(label//" parameter jvp", failures)
        end if

        matrix_bar = reshape([0.4_real64, -0.2_real64, 0.3_real64, 0.5_real64, &
            -0.7_real64, 0.1_real64], shape(matrix_bar))
        call kernel%parameter_vjp(x1, x2, matrix_bar, parameter_bar, status)
        lhs = sum(matrix_bar*matrix_dot)
        rhs = sum(parameter_bar*direction)
        if (.not. status_ok(status) .or. abs(lhs - rhs) > 3.0e-11_real64) then
            call fail(label//" parameter vjp", failures)
        end if
        call kernel%parameter_hvp(x1, x2, matrix_bar, direction, parameter_bar, &
            parameter_bar_dot, status)
        call kernel%set_parameters(parameters + h*direction, status)
        call kernel%parameter_vjp(x1, x2, matrix_bar, parameter_plus, status)
        call kernel%set_parameters(parameters - h*direction, status)
        call kernel%parameter_vjp(x1, x2, matrix_bar, parameter_minus, status)
        call kernel%set_parameters(parameters, status)
        if (.not. status_ok(status) .or. maxval(abs(parameter_bar_dot - &
            (parameter_plus - parameter_minus)/(2.0_real64*h))) > 4.0e-6_real64) then
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

end program test_kernel_catalog
