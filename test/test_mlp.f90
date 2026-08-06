program test_mlp
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_mlp, only: mlp_t, MLP_TANH, MLP_RELU
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer :: failures
    failures = 0
    call test_value_and_packing(failures)
    call test_products_against_reference(failures)
    call test_hvp_against_vjp_finite_difference(failures)
    call test_relu(failures)
    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, " MLP test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS"

contains

    subroutine test_value_and_packing(failures)
        integer, intent(inout) :: failures
        type(mlp_t) :: model
        type(fortnum_status_t) :: status
        real(dp) :: theta(13), recovered(13), x(3, 2), y(3, 1), reference(3, 1)

        theta = [ &
            0.20_dp, -0.10_dp, 0.40_dp, 0.30_dp, -0.50_dp, 0.60_dp, &
            0.05_dp, -0.15_dp, 0.25_dp, 0.70_dp, -0.30_dp, 0.80_dp, 0.10_dp]
        x = reshape([ &
            0.2_dp, -0.4_dp, &
            0.7_dp, 0.1_dp, &
            -0.5_dp, 0.8_dp], shape(x))
        call model%initialize([2, 3, 1], status, hidden_activation=MLP_TANH)
        call model%set_parameters(theta, status)
        call model%predict(x, y, status)
        recovered = model%parameters()
        call reference_predict(theta, x, reference)
        if (.not. status_ok(status) .or. model%parameter_count() /= 13 .or. &
            maxval(abs(recovered - theta)) > 1.0e-14_dp .or. &
            maxval(abs(y - reference)) > 1.0e-14_dp) then
            write (error_unit, '(a)') "FAIL [value] MLP forward or parameter layout"
            failures = failures + 1
        end if
    end subroutine test_value_and_packing

    subroutine test_products_against_reference(failures)
        integer, intent(inout) :: failures
        type(mlp_t) :: model
        type(fortnum_status_t) :: status
        real(dp) :: theta(13), dtheta(13), x(3, 2), dx(3, 2)
        real(dp) :: y(3, 1), dy(3, 1), yp(3, 1), ym(3, 1), fd(3, 1)
        real(dp) :: u(3, 1), parameter_bar(13), x_bar(3, 2)
        real(dp) :: lhs, rhs, h

        theta = [ &
            0.20_dp, -0.10_dp, 0.40_dp, 0.30_dp, -0.50_dp, 0.60_dp, &
            0.05_dp, -0.15_dp, 0.25_dp, 0.70_dp, -0.30_dp, 0.80_dp, 0.10_dp]
        dtheta = [ &
            -0.30_dp, 0.20_dp, 0.10_dp, -0.40_dp, 0.50_dp, -0.20_dp, &
            0.30_dp, -0.25_dp, 0.15_dp, -0.10_dp, 0.35_dp, -0.45_dp, 0.25_dp]
        x = reshape([ &
            0.2_dp, -0.4_dp, &
            0.7_dp, 0.1_dp, &
            -0.5_dp, 0.8_dp], shape(x))
        dx = reshape([ &
            -0.3_dp, 0.2_dp, &
            0.1_dp, -0.4_dp, &
            0.5_dp, 0.6_dp], shape(dx))
        u = reshape([0.4_dp, -0.2_dp, 0.3_dp], shape(u))
        call model%initialize([2, 3, 1], status, hidden_activation=MLP_TANH)
        call model%set_parameters(theta, status)
        call model%jvp(x, dtheta, dx, y, dy, status)
        h = 1.0e-6_dp
        call reference_predict(theta + h*dtheta, x + h*dx, yp)
        call reference_predict(theta - h*dtheta, x - h*dx, ym)
        fd = (yp - ym)/(2.0_dp*h)
        if (.not. status_ok(status) .or. maxval(abs(dy - fd)) > 2.0e-9_dp) then
            write (error_unit, '(a,es12.4)') "FAIL [jvp] finite difference=", &
                maxval(abs(dy - fd))
            failures = failures + 1
        end if

        call model%vjp(x, u, parameter_bar, x_bar, status)
        lhs = sum(u*dy)
        rhs = sum(parameter_bar*dtheta) + sum(x_bar*dx)
        if (.not. status_ok(status) .or. abs(lhs - rhs) > 2.0e-12_dp) then
            write (error_unit, '(a,es12.4)') "FAIL [vjp] adjoint identity=", &
                abs(lhs - rhs)
            failures = failures + 1
        end if
        call test_directional_gradient(theta, x, u, parameter_bar, x_bar, failures)
    end subroutine test_products_against_reference

    subroutine test_hvp_against_vjp_finite_difference(failures)
        integer, intent(inout) :: failures
        type(mlp_t) :: model
        type(fortnum_status_t) :: status
        real(dp) :: theta(13), dtheta(13), x(3, 2), dx(3, 2), u(3, 1)
        real(dp) :: parameter_hvp(13), x_hvp(3, 2)
        real(dp) :: parameter_plus(13), parameter_minus(13)
        real(dp) :: x_plus(3, 2), x_minus(3, 2)
        real(dp) :: gradient_plus(13), gradient_minus(13)
        real(dp) :: x_bar_plus(3, 2), x_bar_minus(3, 2)
        real(dp) :: finite_parameter_hvp(13), finite_x_hvp(3, 2), h

        theta = [ &
            0.20_dp, -0.10_dp, 0.40_dp, 0.30_dp, -0.50_dp, 0.60_dp, &
            0.05_dp, -0.15_dp, 0.25_dp, 0.70_dp, -0.30_dp, 0.80_dp, 0.10_dp]
        dtheta = [ &
            -0.30_dp, 0.20_dp, 0.10_dp, -0.40_dp, 0.50_dp, -0.20_dp, &
            0.30_dp, -0.25_dp, 0.15_dp, -0.10_dp, 0.35_dp, -0.45_dp, 0.25_dp]
        x = reshape([ &
            0.2_dp, -0.4_dp, 0.7_dp, 0.1_dp, -0.5_dp, 0.8_dp], shape(x))
        dx = reshape([ &
            -0.3_dp, 0.2_dp, 0.1_dp, -0.4_dp, 0.5_dp, 0.6_dp], shape(dx))
        u = reshape([0.4_dp, -0.2_dp, 0.3_dp], shape(u))
        call model%initialize([2, 3, 1], status, hidden_activation=MLP_TANH)
        call model%set_parameters(theta, status)
        call model%hvp(x, u, dtheta, dx, parameter_hvp, x_hvp, status)
        h = 1.0e-6_dp
        parameter_plus = theta + h*dtheta
        parameter_minus = theta - h*dtheta
        x_plus = x + h*dx
        x_minus = x - h*dx
        call model%set_parameters(parameter_plus, status)
        call model%vjp(x_plus, u, gradient_plus, x_bar_plus, status)
        call model%set_parameters(parameter_minus, status)
        call model%vjp(x_minus, u, gradient_minus, x_bar_minus, status)
        call model%set_parameters(theta, status)
        finite_parameter_hvp = (gradient_plus - gradient_minus)/(2.0_dp*h)
        finite_x_hvp = (x_bar_plus - x_bar_minus)/(2.0_dp*h)
        if (.not. status_ok(status) .or. &
            maxval(abs(parameter_hvp - finite_parameter_hvp)) > 2.0e-8_dp .or. &
            maxval(abs(x_hvp - finite_x_hvp)) > 2.0e-8_dp) then
            write (error_unit, '(a,2es12.4)') &
                "FAIL [hvp] finite difference parameter/input=", &
                maxval(abs(parameter_hvp - finite_parameter_hvp)), &
                maxval(abs(x_hvp - finite_x_hvp))
            failures = failures + 1
        end if
    end subroutine test_hvp_against_vjp_finite_difference

    subroutine test_directional_gradient(theta, x, target, parameter_bar, x_bar, failures)
        real(dp), intent(in) :: theta(:), x(:, :), target(:, :)
        real(dp), intent(in) :: parameter_bar(:), x_bar(:, :)
        integer, intent(inout) :: failures
        real(dp) :: theta_plus(size(theta)), theta_minus(size(theta))
        real(dp) :: x_plus(size(x, 1), size(x, 2)), x_minus(size(x, 1), size(x, 2))
        real(dp) :: y_plus(size(target, 1), size(target, 2))
        real(dp) :: y_minus(size(target, 1), size(target, 2))
        real(dp) :: direction(size(theta)), x_direction(size(x, 1), size(x, 2))
        real(dp) :: gradient_dot, finite_difference, h
        integer :: i

        direction = [(0.07_dp*real(i, dp), i=1, size(theta))]
        x_direction = 0.0_dp
        x_direction(:, 1) = [0.2_dp, -0.1_dp, 0.3_dp]
        x_direction(:, 2) = [-0.4_dp, 0.5_dp, -0.2_dp]
        h = 1.0e-6_dp
        theta_plus = theta + h*direction
        theta_minus = theta - h*direction
        x_plus = x + h*x_direction
        x_minus = x - h*x_direction
        call reference_predict(theta_plus, x_plus, y_plus)
        call reference_predict(theta_minus, x_minus, y_minus)
        finite_difference = (sum(y_plus*target) - sum(y_minus*target))/(2.0_dp*h)
        gradient_dot = sum(parameter_bar*direction) + sum(x_bar*x_direction)
        if (abs(finite_difference - gradient_dot) > 3.0e-9_dp) then
            write (error_unit, '(a,es12.4)') "FAIL [backprop] directional gradient=", &
                abs(finite_difference - gradient_dot)
            failures = failures + 1
        end if
    end subroutine test_directional_gradient

    subroutine test_relu(failures)
        integer, intent(inout) :: failures
        type(mlp_t) :: model
        type(fortnum_status_t) :: status
        real(dp) :: theta(9), dtheta(9), x(2, 2), dx(2, 2)
        real(dp) :: y(2, 1), dy(2, 1), parameter_bar(9), x_bar(2, 2)

        theta = [1.0_dp, 0.0_dp, 0.0_dp, 1.0_dp, 0.1_dp, -0.2_dp, &
            0.4_dp, -0.3_dp, 0.4_dp]
        dtheta = [0.2_dp, -0.3_dp, 0.1_dp, 0.4_dp, -0.2_dp, 0.3_dp, &
            -0.1_dp, 0.2_dp, -0.15_dp]
        x = reshape([1.0_dp, 2.0_dp, -0.5_dp, 0.7_dp], shape(x))
        dx = reshape([0.1_dp, -0.2_dp, 0.3_dp, 0.4_dp], shape(dx))
        call model%initialize([2, 2, 1], status, hidden_activation=MLP_RELU)
        call model%set_parameters(theta, status)
        call model%jvp(x, dtheta, dx, y, dy, status)
        call model%vjp(x, reshape([1.0_dp, -0.5_dp], [2, 1]), parameter_bar, &
            x_bar, status)
        if (.not. status_ok(status) .or. any(y < 0.0_dp)) then
            write (error_unit, '(a)') "FAIL [relu] positive branch"
            failures = failures + 1
        end if
    end subroutine test_relu

    subroutine reference_predict(theta, x, y)
        real(dp), intent(in) :: theta(:), x(:, :)
        real(dp), intent(out) :: y(:, :)
        real(dp) :: weight_1(2, 3), bias_1(3), weight_2(3, 1), bias_2(1)
        real(dp) :: hidden(3), preactivation(3)
        integer :: i, j

        weight_1 = reshape(theta(1:6), shape(weight_1))
        bias_1 = theta(7:9)
        weight_2 = reshape(theta(10:12), shape(weight_2))
        bias_2 = theta(13:13)
        do i = 1, size(x, 1)
            do j = 1, 3
                preactivation(j) = bias_1(j) + sum(x(i, :)*weight_1(:, j))
                hidden(j) = tanh(preactivation(j))
            end do
            y(i, 1) = bias_2(1) + sum(hidden*weight_2(:, 1))
        end do
    end subroutine reference_predict

end program test_mlp
