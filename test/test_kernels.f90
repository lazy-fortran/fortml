program test_kernels
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_kernels, only: kernel_t, make_rbf_kernel, make_matern32_kernel, &
        make_linear_kernel, make_constant_kernel, make_white_noise_kernel, &
        kernel_add, kernel_multiply
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer :: failures
    failures = 0
    call test_leaf_values(failures)
    call test_composition(failures)
    call test_parameter_products(failures)
    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, " kernel test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS"

contains

    subroutine test_leaf_values(failures)
        integer, intent(inout) :: failures
        type(kernel_t) :: rbf, matern, linear, constant, white_noise
        type(fortnum_status_t) :: status
        real(dp) :: x1(3, 2), x2(2, 2), matrix(3, 2), expected(3, 2)
        real(dp) :: white_matrix(3, 3), white_expected(3, 3)
        real(dp) :: r2
        integer :: i, j

        x1 = reshape([0.0_dp, 0.5_dp, -0.4_dp, 1.0_dp, 1.2_dp, -0.7_dp], shape(x1))
        x2 = reshape([0.2_dp, -0.1_dp, 0.8_dp, 0.4_dp], shape(x2))
        rbf = make_rbf_kernel(2, 2.0_dp, 1.0_dp, status)
        call rbf%matrix(x1, x2, matrix, status)
        do j = 1, size(x2, 1)
            do i = 1, size(x1, 1)
                r2 = sum((x1(i, :) - x2(j, :))**2)
                expected(i, j) = 2.0_dp*exp(-0.5_dp*r2)
            end do
        end do
        if (.not. status_ok(status) .or. maxval(abs(matrix - expected)) > 2.0e-14_dp) then
            write (error_unit, '(a)') "FAIL [rbf] value"
            failures = failures + 1
        end if
        if (abs(rbf%value(x1(1, :), x2(1, :)) - matrix(1, 1)) > 2.0e-14_dp) then
            write (error_unit, '(a)') "FAIL [rbf] scalar value"
            failures = failures + 1
        end if

        matern = make_matern32_kernel(2, 1.5_dp, 0.8_dp, status)
        call matern%matrix(x1, x2, matrix, status)
        do j = 1, size(x2, 1)
            do i = 1, size(x1, 1)
                r2 = sqrt(sum((x1(i, :) - x2(j, :))**2))/0.8_dp
                expected(i, j) = 1.5_dp*(1.0_dp + sqrt(3.0_dp)*r2)* &
                    exp(-sqrt(3.0_dp)*r2)
            end do
        end do
        if (.not. status_ok(status) .or. maxval(abs(matrix - expected)) > 2.0e-14_dp) then
            write (error_unit, '(a)') "FAIL [matern32] value"
            failures = failures + 1
        end if
        if (abs(matern%value(x1(1, :), x2(1, :)) - matrix(1, 1)) > 2.0e-14_dp) then
            write (error_unit, '(a)') "FAIL [matern32] scalar value"
            failures = failures + 1
        end if

        linear = make_linear_kernel(2, 0.7_dp, status)
        call linear%matrix(x1, x2, matrix, status)
        expected = 0.7_dp*matmul(x1, transpose(x2))
        if (.not. status_ok(status) .or. maxval(abs(matrix - expected)) > 2.0e-14_dp) then
            write (error_unit, '(a)') "FAIL [linear] value"
            failures = failures + 1
        end if
        if (abs(linear%value(x1(1, :), x2(1, :)) - matrix(1, 1)) > 2.0e-14_dp) then
            write (error_unit, '(a)') "FAIL [linear] scalar value"
            failures = failures + 1
        end if

        constant = make_constant_kernel(2, 0.3_dp, status)
        call constant%matrix(x1, x2, matrix, status)
        expected = 0.3_dp
        if (.not. status_ok(status) .or. maxval(abs(matrix - expected)) > 2.0e-14_dp) then
            write (error_unit, '(a)') "FAIL [constant] value"
            failures = failures + 1
        end if
        if (abs(constant%value(x1(1, :), x2(1, :)) - matrix(1, 1)) > 2.0e-14_dp) then
            write (error_unit, '(a)') "FAIL [constant] scalar value"
            failures = failures + 1
        end if

        white_noise = make_white_noise_kernel(2, 0.2_dp, status)
        call white_noise%matrix(x1, x1, white_matrix, status)
        white_expected = 0.0_dp
        do i = 1, size(x1, 1)
            white_expected(i, i) = 0.2_dp
        end do
        if (.not. status_ok(status) .or. maxval(abs(white_matrix - white_expected)) > &
            2.0e-14_dp) then
            write (error_unit, '(a)') "FAIL [white_noise] value"
            failures = failures + 1
        end if
        if (abs(white_noise%value(x1(1, :), x1(1, :)) - 0.2_dp) > 2.0e-14_dp .or. &
            abs(white_noise%value(x1(1, :), x1(2, :))) > 2.0e-14_dp) then
            write (error_unit, '(a)') "FAIL [white_noise] scalar value"
            failures = failures + 1
        end if
    end subroutine test_leaf_values

    subroutine test_composition(failures)
        integer, intent(inout) :: failures
        type(kernel_t) :: rbf, constant, sum_kernel, product_kernel
        type(fortnum_status_t) :: status
        real(dp) :: x(3, 2), matrix(3, 3), rbf_matrix(3, 3), constant_matrix(3, 3)

        x = reshape([0.0_dp, 0.5_dp, -0.4_dp, 1.0_dp, 1.2_dp, -0.7_dp], shape(x))
        rbf = make_rbf_kernel(2, 1.2_dp, 0.9_dp, status)
        constant = make_constant_kernel(2, 0.4_dp, status)
        sum_kernel = kernel_add(rbf, constant, status)
        product_kernel = kernel_multiply(rbf, constant, status)
        call rbf%matrix(x, x, rbf_matrix, status)
        call constant%matrix(x, x, constant_matrix, status)
        call sum_kernel%matrix(x, x, matrix, status)
        if (.not. status_ok(status) .or. maxval(abs(matrix - rbf_matrix - &
            constant_matrix)) > 2.0e-14_dp .or. sum_kernel%parameter_count() /= 3) then
            write (error_unit, '(a)') "FAIL [sum] composition"
            failures = failures + 1
        end if
        call product_kernel%matrix(x, x, matrix, status)
        if (.not. status_ok(status) .or. maxval(abs(matrix - rbf_matrix* &
            constant_matrix)) > 2.0e-14_dp) then
            write (error_unit, '(a)') "FAIL [product] composition"
            failures = failures + 1
        end if
        if (abs(sum_kernel%value(x(1, :), x(2, :)) - &
            (rbf_matrix(1, 2) + constant_matrix(1, 2))) > 2.0e-14_dp .or. &
            abs(product_kernel%value(x(1, :), x(2, :)) - &
            rbf_matrix(1, 2)*constant_matrix(1, 2)) > 2.0e-14_dp) then
            write (error_unit, '(a)') "FAIL [composition] scalar value"
            failures = failures + 1
        end if
    end subroutine test_composition

    subroutine test_parameter_products(failures)
        integer, intent(inout) :: failures
        type(kernel_t) :: rbf
        type(fortnum_status_t) :: status
        real(dp) :: x1(3, 2), x2(2, 2), direction(2), matrix(3, 2)
        real(dp) :: matrix_dot(3, 2), matrix_plus(3, 2), matrix_minus(3, 2)
        real(dp) :: matrix_bar(3, 2), parameter_bar(2), lhs, rhs, h
        real(dp) :: parameter_bar_dot(2), parameter_plus(2), parameter_minus(2)
        real(dp) :: parameters(2)

        x1 = reshape([0.0_dp, 0.5_dp, -0.4_dp, 1.0_dp, 1.2_dp, -0.7_dp], shape(x1))
        x2 = reshape([0.2_dp, -0.1_dp, 0.8_dp, 0.4_dp], shape(x2))
        rbf = make_rbf_kernel(2, 2.0_dp, 1.1_dp, status)
        direction = [0.17_dp, -0.23_dp]
        call rbf%matrix_jvp(x1, x2, direction, matrix, matrix_dot, status)
        h = 1.0e-6_dp
        call reference_rbf(rbf%parameters() + h*direction, x1, x2, matrix_plus)
        call reference_rbf(rbf%parameters() - h*direction, x1, x2, matrix_minus)
        if (.not. status_ok(status) .or. maxval(abs(matrix_dot - &
            (matrix_plus - matrix_minus)/(2.0_dp*h))) > 2.0e-9_dp) then
            write (error_unit, '(a)') "FAIL [jvp] kernel finite difference"
            failures = failures + 1
        end if

        matrix_bar = reshape([0.4_dp, -0.2_dp, 0.3_dp, 0.5_dp, -0.7_dp, 0.1_dp], &
            shape(matrix_bar))
        call rbf%parameter_vjp(x1, x2, matrix_bar, parameter_bar, status)
        lhs = sum(matrix_bar*matrix_dot)
        rhs = sum(parameter_bar*direction)
        if (.not. status_ok(status) .or. abs(lhs - rhs) > 2.0e-12_dp) then
            write (error_unit, '(a)') "FAIL [vjp] kernel adjoint identity"
            failures = failures + 1
        end if

        parameters = rbf%parameters()
        call rbf%parameter_hvp(x1, x2, matrix_bar, direction, parameter_bar, &
            parameter_bar_dot, status)
        h = 1.0e-6_dp
        call rbf%set_parameters(parameters + h*direction, status)
        call rbf%parameter_vjp(x1, x2, matrix_bar, parameter_plus, status)
        call rbf%set_parameters(parameters - h*direction, status)
        call rbf%parameter_vjp(x1, x2, matrix_bar, parameter_minus, status)
        call rbf%set_parameters(parameters, status)
        if (.not. status_ok(status) .or. maxval(abs(parameter_bar_dot - &
            (parameter_plus - parameter_minus)/(2.0_dp*h))) > 3.0e-8_dp) then
            write (error_unit, '(a)') "FAIL [hvp] kernel finite difference"
            failures = failures + 1
        end if
    end subroutine test_parameter_products

    subroutine reference_rbf(parameters, x1, x2, matrix)
        real(dp), intent(in) :: parameters(:), x1(:, :), x2(:, :)
        real(dp), intent(out) :: matrix(:, :)
        real(dp) :: variance, lengthscale, r2
        integer :: i, j

        variance = exp(parameters(1))
        lengthscale = exp(parameters(2))
        do j = 1, size(x2, 1)
            do i = 1, size(x1, 1)
                r2 = sum((x1(i, :) - x2(j, :))**2)
                matrix(i, j) = variance*exp(-0.5_dp*r2/(lengthscale**2))
            end do
        end do
    end subroutine reference_rbf

end program test_kernels
