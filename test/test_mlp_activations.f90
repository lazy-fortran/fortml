program test_mlp_activations
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_mlp, only: mlp_t, MLP_LINEAR, MLP_TANH, MLP_RELU, MLP_GELU, &
        MLP_SILU, MLP_ELU, MLP_SOFTPLUS, MLP_LEAKY_RELU, MLP_SIGMOID, MLP_MISH
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer :: failures, i
    integer, parameter :: kinds(10) = [MLP_LINEAR, MLP_TANH, MLP_RELU, MLP_GELU, &
        MLP_SILU, MLP_ELU, MLP_SOFTPLUS, MLP_LEAKY_RELU, MLP_SIGMOID, MLP_MISH]

    failures = 0
    do i = 1, size(kinds)
        call test_activation(kinds(i), failures)
    end do
    call test_sigmoid_extremes(failures)
    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, " MLP activation test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS"

contains

    subroutine test_activation(kind, failures)
        integer, intent(in) :: kind
        integer, intent(inout) :: failures
        type(mlp_t) :: model
        type(fortnum_status_t) :: status
        real(dp) :: theta(2), dtheta(2), x(5, 1), dx(5, 1)
        real(dp) :: y(5, 1), expected(5, 1), dy(5, 1), finite_dy(5, 1)
        real(dp) :: u(5, 1), parameter_bar(2), x_bar(5, 1)
        real(dp) :: parameter_hvp(2), x_hvp(5, 1)
        real(dp) :: gradient_plus(2), gradient_minus(2)
        real(dp) :: x_bar_plus(5, 1), x_bar_minus(5, 1)
        real(dp) :: yp(5, 1), ym(5, 1), h, h_hvp, lhs, rhs
        integer :: i

        theta = [1.15_dp, -0.37_dp]
        dtheta = [0.23_dp, -0.19_dp]
        x = reshape([-1.3_dp, -0.2_dp, 0.4_dp, 1.7_dp, 0.9_dp], shape(x))
        dx = reshape([0.11_dp, -0.07_dp, 0.05_dp, -0.13_dp, 0.08_dp], shape(dx))
        u = reshape([0.4_dp, -0.2_dp, 0.7_dp, -0.3_dp, 0.5_dp], shape(u))

        call model%initialize([1, 1], status, output_activation=kind)
        call model%set_parameters(theta, status)
        call model%predict(x, y, status)
        do i = 1, size(x, 1)
            expected(i, 1) = reference_activation(theta(1)*x(i, 1) + theta(2), kind)
        end do
        if (.not. status_ok(status) .or. maxval(abs(y - expected)) > 2.0e-14_dp) then
            write (error_unit, '(a,i0)') "FAIL [activation value] kind=", kind
            failures = failures + 1
            return
        end if

        call model%jvp(x, dtheta, dx, y, dy, status)
        h = 1.0e-6_dp
        call model%set_parameters(theta + h*dtheta, status)
        call model%predict(x + h*dx, yp, status)
        call model%set_parameters(theta - h*dtheta, status)
        call model%predict(x - h*dx, ym, status)
        finite_dy = (yp - ym)/(2.0_dp*h)
        call model%set_parameters(theta, status)
        if (.not. status_ok(status) .or. maxval(abs(dy - finite_dy)) > 3.0e-8_dp) then
            write (error_unit, '(a,i0,es12.4)') "FAIL [activation JVP] kind=", &
                kind, maxval(abs(dy - finite_dy))
            failures = failures + 1
        end if

        call model%vjp(x, u, parameter_bar, x_bar, status)
        lhs = sum(u*dy)
        rhs = sum(parameter_bar*dtheta) + sum(x_bar*dx)
        if (.not. status_ok(status) .or. abs(lhs - rhs) > 2.0e-11_dp) then
            write (error_unit, '(a,i0,es12.4)') "FAIL [activation VJP] kind=", &
                kind, abs(lhs - rhs)
            failures = failures + 1
        end if

        call model%hvp(x, u, dtheta, dx, parameter_hvp, x_hvp, status)
        h_hvp = 1.0e-5_dp
        call model%set_parameters(theta + h_hvp*dtheta, status)
        call model%vjp(x + h_hvp*dx, u, gradient_plus, x_bar_plus, status)
        call model%set_parameters(theta - h_hvp*dtheta, status)
        call model%vjp(x - h_hvp*dx, u, gradient_minus, x_bar_minus, status)
        call model%set_parameters(theta, status)
        if (.not. status_ok(status) .or. &
            maxval(abs(parameter_hvp - (gradient_plus - gradient_minus)/(2.0_dp*h_hvp))) > &
            2.0e-6_dp .or. maxval(abs(x_hvp - (x_bar_plus - x_bar_minus)/(2.0_dp*h_hvp))) > &
            2.0e-6_dp) then
            write (error_unit, '(a,i0,2es12.4)') "FAIL [activation HVP] kind=", kind, &
                maxval(abs(parameter_hvp - (gradient_plus - gradient_minus)/(2.0_dp*h_hvp))), &
                maxval(abs(x_hvp - (x_bar_plus - x_bar_minus)/(2.0_dp*h_hvp)))
            failures = failures + 1
        end if
    end subroutine test_activation

    subroutine test_sigmoid_extremes(failures)
        integer, intent(inout) :: failures
        type(mlp_t) :: model
        type(fortnum_status_t) :: status
        real(dp) :: theta(2), x(3, 1), y(3, 1), dy(3, 1)

        theta = [1.0_dp, 0.0_dp]
        x(:, 1) = [-1000.0_dp, 0.0_dp, 1000.0_dp]
        call model%initialize([1, 1], status, output_activation=MLP_SIGMOID)
        call model%set_parameters(theta, status)
        call model%predict(x, y, status)
        if (.not. status_ok(status) .or. .not. is_finite(y) .or. &
            y(1, 1) /= 0.0_dp .or. y(2, 1) /= 0.5_dp .or. &
            y(3, 1) /= 1.0_dp) then
            write (error_unit, '(a)') "FAIL [sigmoid extreme values]"
            failures = failures + 1
            return
        end if
        call model%jvp(x, [0.0_dp, 1.0_dp], 0.0_dp*x, y, dy, status)
        if (.not. status_ok(status) .or. .not. is_finite(dy) .or. &
            dy(1, 1) /= 0.0_dp .or. abs(dy(2, 1) - 0.25_dp) > 2.0e-15_dp .or. &
            dy(3, 1) /= 0.0_dp) then
            write (error_unit, '(a)') "FAIL [sigmoid extreme derivatives]"
            failures = failures + 1
        end if
    end subroutine test_sigmoid_extremes

    logical function is_finite(values) result(valid)
        real(dp), intent(in) :: values(:, :)

        valid = all(values == values) .and. all(abs(values) < huge(1.0_dp))
    end function is_finite

    real(dp) function reference_activation(x, kind) result(value)
        real(dp), intent(in) :: x
        integer, intent(in) :: kind
        real(dp) :: s, u, t
        real(dp), parameter :: c = 0.797884560802865355879_dp
        real(dp), parameter :: k = 0.044715_dp

        select case (kind)
        case (MLP_LINEAR)
            value = x
        case (MLP_TANH)
            value = tanh(x)
        case (MLP_RELU)
            value = max(0.0_dp, x)
        case (MLP_GELU)
            u = c*(x + k*x*x*x)
            t = tanh(u)
            value = 0.5_dp*x*(1.0_dp + t)
        case (MLP_SILU)
            if (x >= 0.0_dp) then
                s = 1.0_dp/(1.0_dp + exp(-x))
            else
                s = exp(x)/(1.0_dp + exp(x))
            end if
            value = x*s
        case (MLP_ELU)
            value = merge(x, exp(x) - 1.0_dp, x > 0.0_dp)
        case (MLP_SOFTPLUS)
            if (x > 0.0_dp) then
                value = x + log(1.0_dp + exp(-x))
            else
                value = log(1.0_dp + exp(x))
            end if
        case (MLP_LEAKY_RELU)
            value = merge(x, 0.01_dp*x, x >= 0.0_dp)
        case (MLP_SIGMOID)
            if (x >= 0.0_dp) then
                value = 1.0_dp/(1.0_dp + exp(-x))
            else
                value = exp(x)/(1.0_dp + exp(x))
            end if
        case (MLP_MISH)
            if (x > 0.0_dp) then
                value = x*tanh(x + log(1.0_dp + exp(-x)))
            else
                value = x*tanh(log(1.0_dp + exp(x)))
            end if
        end select
    end function reference_activation

end program test_mlp_activations
