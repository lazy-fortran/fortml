program test_parameter_products
    use fortml_parameter_products, only: parameter_products_t, &
        parameter_products_from_mlp, parameter_products_from_gp
    use fortml_mlp, only: mlp_t
    use fortml_gaussian_process, only: gp_regression_t
    use fortml_kernels, only: kernel_t, make_rbf_kernel
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_DOMAIN_ERROR
    implicit none

    integer :: failures

    failures = 0
    call test_mlp_parameter_products(failures)
    call test_gp_parameter_products(failures)
    if (failures == 0) then
        write (*, '(a)') "PASS parameter products independent full-vector oracles"
    else
        write (*, '(a,i0)') "FAIL parameter products cases: ", failures
        error stop 1
    end if

contains

    subroutine check(condition, label, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: label
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (*, '(a)') "  FAIL "//label
        end if
    end subroutine check

    subroutine test_mlp_parameter_products(failures)
        integer, intent(inout) :: failures
        type(mlp_t), target :: model
        type(parameter_products_t) :: products
        type(fortnum_status_t) :: status
        real(dp) :: x(3, 2), y(3, 1), y_dot(3, 1), y_bar(3, 1)
        real(dp) :: y_plus(3, 1), y_minus(3, 1)
        real(dp) :: theta(13), direction(13), direction_two(13)
        real(dp) :: theta_bar(13), theta_plus(13), theta_minus(13)
        real(dp) :: theta_hvp(13), theta_hvp_fd(13), theta_hvp_two(13)
        real(dp) :: h, lhs, rhs

        x = reshape([0.2_dp, -0.3_dp, 0.7_dp, 0.4_dp, -0.6_dp, 0.5_dp], &
            shape(x))
        theta = [0.3_dp, -0.2_dp, 0.4_dp, 0.1_dp, -0.5_dp, 0.6_dp, &
            -0.7_dp, 0.2_dp, 0.8_dp, -0.1_dp, 0.5_dp, -0.4_dp, 0.9_dp]
        direction = [-0.2_dp, 0.1_dp, 0.3_dp, -0.4_dp, 0.5_dp, 0.2_dp, &
            -0.1_dp, 0.4_dp, -0.3_dp, 0.6_dp, -0.2_dp, 0.7_dp, -0.5_dp]
        direction_two = [0.6_dp, -0.5_dp, 0.4_dp, -0.3_dp, 0.2_dp, -0.1_dp, &
            0.7_dp, -0.6_dp, 0.5_dp, -0.4_dp, 0.3_dp, -0.2_dp, 0.1_dp]
        y_bar = reshape([0.8_dp, -0.4_dp, 0.6_dp], shape(y_bar))
        h = 1.0e-6_dp

        call model%initialize([2, 3, 1], status)
        call model%set_parameters(theta, status)
        call check(status_ok(status), "MLP setup", failures)
        call parameter_products_from_mlp(products, "mlp", model, status)
        call check(status_ok(status), "MLP product construction", failures)
        call check(products%parameter_count() == size(theta), &
            "MLP product parameter count", failures)

        call products%value(x, y, status)
        call check(status_ok(status), "MLP product value", failures)
        call products%jvp(x, direction, y, y_dot, status)
        call check(status_ok(status), "MLP product JVP", failures)

        call products%unpack(theta + h*direction, status)
        call products%value(x, y_plus, status)
        call products%unpack(theta - h*direction, status)
        call products%value(x, y_minus, status)
        call products%unpack(theta, status)
        call check(maxval(abs(y_dot - (y_plus - y_minus)/(2.0_dp*h))) < 2.0e-8_dp, &
            "MLP JVP finite difference", failures)

        call products%vjp(x, y_bar, theta_bar, status)
        call check(status_ok(status), "MLP product VJP", failures)
        lhs = sum(y_bar*y_dot)
        rhs = sum(direction*theta_bar)
        call check(abs(lhs - rhs) < 2.0e-12_dp, &
            "MLP product VJP adjoint identity", failures)

        call products%hvp(x, y_bar, direction, theta_hvp, status)
        call check(status_ok(status) .and. products%has_hvp(), &
            "MLP product HVP", failures)
        call products%unpack(theta + h*direction, status)
        call products%vjp(x, y_bar, theta_plus, status)
        call products%unpack(theta - h*direction, status)
        call products%vjp(x, y_bar, theta_minus, status)
        call products%unpack(theta, status)
        theta_hvp_fd = (theta_plus - theta_minus)/(2.0_dp*h)
        call check(maxval(abs(theta_hvp - theta_hvp_fd)) < 3.0e-7_dp, &
            "MLP HVP finite difference", failures)

        call products%hvp(x, y_bar, direction_two, theta_hvp_two, status)
        call check(abs(sum(direction_two*theta_hvp) - &
            sum(direction*theta_hvp_two)) < 3.0e-10_dp, &
            "MLP HVP Hessian symmetry", failures)
    end subroutine test_mlp_parameter_products

    subroutine test_gp_parameter_products(failures)
        integer, intent(inout) :: failures
        type(gp_regression_t), target :: model
        type(parameter_products_t) :: products
        type(kernel_t) :: kernel
        type(fortnum_status_t) :: status
        real(dp) :: x_train(4, 1), y_train(4, 1), x_test(2, 1)
        real(dp) :: y(2, 1), y_dot(2, 1), y_plus(2, 1), y_minus(2, 1)
        real(dp) :: y_bar(2, 1), theta(3), direction(3), theta_bar(3)
        real(dp) :: h, lhs, rhs

        x_train(:, 1) = [-1.0_dp, -0.2_dp, 0.4_dp, 1.1_dp]
        y_train(:, 1) = [0.7_dp, -0.1_dp, 0.6_dp, 1.2_dp]
        x_test(:, 1) = [-0.5_dp, 0.8_dp]
        y_bar(:, 1) = [0.4_dp, -0.7_dp]
        direction = [0.2_dp, -0.3_dp, 0.5_dp]
        h = 1.0e-6_dp

        kernel = make_rbf_kernel(1, 1.3_dp, 0.8_dp, status)
        call model%fit(x_train, y_train, kernel, 0.05_dp, status, &
            jitter=1.0e-10_dp)
        call check(status_ok(status), "GP setup", failures)
        call parameter_products_from_gp(products, "gp", model, status)
        call check(status_ok(status), "GP product construction", failures)
        call check(products%parameter_count() == 3, &
            "GP packed kernel/noise parameter count", failures)
        call products%pack(theta, status)
        call check(status_ok(status), "GP product pack", failures)

        call products%jvp(x_test, direction, y, y_dot, status)
        call check(status_ok(status), "GP product JVP", failures)
        call products%unpack(theta + h*direction, status)
        call products%value(x_test, y_plus, status)
        call products%unpack(theta - h*direction, status)
        call products%value(x_test, y_minus, status)
        call products%unpack(theta, status)
        call check(maxval(abs(y_dot - (y_plus - y_minus)/(2.0_dp*h))) < 2.0e-6_dp, &
            "GP JVP finite difference", failures)

        call products%vjp(x_test, y_bar, theta_bar, status)
        call check(status_ok(status), "GP product VJP", failures)
        lhs = sum(y_bar*y_dot)
        rhs = sum(direction*theta_bar)
        call check(abs(lhs - rhs) < 3.0e-9_dp, &
            "GP product VJP adjoint identity", failures)

        call products%hvp(x_test, y_bar, direction, theta_bar, status)
        call check(status%code == FORTNUM_DOMAIN_ERROR .and. &
            .not. products%has_hvp(), "GP undeclared HVP is refused", failures)
    end subroutine test_gp_parameter_products

end program test_parameter_products
