program test_basis
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_basis, only: basis_map_t
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer :: failures

    failures = 0
    call test_polynomial(failures)
    call test_fourier(failures)
    call test_radial(failures)
    call test_spline(failures)
    call test_callback(failures)
    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, " basis test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS"

contains

    subroutine test_polynomial(failures)
        integer, intent(inout) :: failures
        type(basis_map_t) :: map
        type(fortnum_status_t) :: status
        real(dp) :: x(4, 2), x_dot(4, 2), phi(4, 5), phi_dot(4, 5)
        real(dp) :: phi_plus(4, 5), phi_minus(4, 5)
        real(dp) :: u(4, 5), x_bar(4, 2), lhs, rhs
        real(dp), allocatable :: theta_dot(:), theta_bar(:)
        real(dp) :: h

        x = reshape([0.1_dp, -0.2_dp, 0.4_dp, 0.3_dp, &
            -0.5_dp, 0.8_dp, 0.7_dp, -0.6_dp], shape(x))
        x_dot = reshape([-0.3_dp, 0.2_dp, 0.1_dp, -0.4_dp, &
            0.5_dp, 0.6_dp, -0.2_dp, 0.3_dp], shape(x_dot))
        call map%initialize_polynomial(2, 2, status, include_intercept=.true.)
        allocate(theta_dot(0), theta_bar(0))
        call map%evaluate(x, phi, status)
        call map%jvp(x, theta_dot, x_dot, phi, phi_dot, status)
        h = 1.0e-6_dp
        call map%evaluate(x + h*x_dot, phi_plus, status)
        call map%evaluate(x - h*x_dot, phi_minus, status)
        if (.not. status_ok(status) .or. map%feature_count() /= 5 .or. &
            maxval(abs(phi_dot - (phi_plus - phi_minus)/(2.0_dp*h))) > 2.0e-10_dp) then
            write (error_unit, '(a)') "FAIL [polynomial] value or JVP"
            failures = failures + 1
        end if

        u = reshape([ &
            0.4_dp, -0.2_dp, 0.3_dp, 0.5_dp, -0.7_dp, &
            0.1_dp, 0.2_dp, -0.6_dp, 0.8_dp, -0.4_dp, &
            -0.3_dp, 0.9_dp, -0.5_dp, 0.2_dp, 0.6_dp, &
            0.7_dp, -0.1_dp, 0.4_dp, -0.8_dp, 0.3_dp], shape(u))
        call map%vjp(x, u, theta_bar, x_bar, status)
        lhs = sum(u*phi_dot)
        rhs = sum(x_bar*x_dot)
        if (.not. status_ok(status) .or. abs(lhs - rhs) > 1.0e-12_dp) then
            write (error_unit, '(a,es12.4)') "FAIL [polynomial] VJP identity=", &
                abs(lhs - rhs)
            failures = failures + 1
        end if
    end subroutine test_polynomial

    subroutine test_fourier(failures)
        integer, intent(inout) :: failures
        type(basis_map_t) :: map
        type(fortnum_status_t) :: status
        real(dp) :: x(3, 2), x_dot(3, 2), phi(3, 5), phi_dot(3, 5)
        real(dp) :: phi_plus(3, 5), phi_minus(3, 5)
        real(dp) :: theta(2), theta_dot(2), theta_plus(2), theta_minus(2)
        real(dp) :: u(3, 5), theta_bar(2), x_bar(3, 2), lhs, rhs, h

        x = reshape([0.2_dp, -0.4_dp, 0.7_dp, 0.1_dp, -0.5_dp, 0.8_dp], shape(x))
        x_dot = reshape([-0.3_dp, 0.2_dp, 0.1_dp, -0.4_dp, 0.5_dp, 0.6_dp], &
            shape(x_dot))
        call map%initialize_fourier(2, reshape([1.2_dp, 0.7_dp], [1, 2]), &
            status, include_intercept=.true.)
        theta = map%parameters()
        theta_dot = [0.17_dp, -0.23_dp]
        call map%jvp(x, theta_dot, x_dot, phi, phi_dot, status)
        h = 1.0e-6_dp
        theta_plus = theta + h*theta_dot
        theta_minus = theta - h*theta_dot
        call map%set_parameters(theta_plus, status)
        call map%evaluate(x + h*x_dot, phi_plus, status)
        call map%set_parameters(theta_minus, status)
        call map%evaluate(x - h*x_dot, phi_minus, status)
        call map%set_parameters(theta, status)
        if (.not. status_ok(status) .or. &
            maxval(abs(phi_dot - (phi_plus - phi_minus)/(2.0_dp*h))) > 2.0e-10_dp) then
            write (error_unit, '(a)') "FAIL [Fourier] JVP finite difference"
            failures = failures + 1
        end if

        u = reshape([ &
            0.4_dp, -0.2_dp, 0.3_dp, 0.5_dp, -0.7_dp, &
            0.1_dp, 0.2_dp, -0.6_dp, 0.8_dp, -0.4_dp, &
            -0.3_dp, 0.9_dp, -0.5_dp, 0.2_dp, 0.6_dp], shape(u))
        call map%vjp(x, u, theta_bar, x_bar, status)
        lhs = sum(u*phi_dot)
        rhs = sum(theta_bar*theta_dot) + sum(x_bar*x_dot)
        if (.not. status_ok(status) .or. abs(lhs - rhs) > 1.0e-12_dp) then
            write (error_unit, '(a,es12.4)') "FAIL [Fourier] VJP identity=", &
                abs(lhs - rhs)
            failures = failures + 1
        end if
    end subroutine test_fourier

    subroutine test_radial(failures)
        integer, intent(inout) :: failures
        type(basis_map_t) :: map
        type(fortnum_status_t) :: status
        real(dp) :: x(3, 2), x_dot(3, 2), centers(2, 2), scales(2, 2)
        real(dp) :: phi(3, 3), phi_dot(3, 3), phi_plus(3, 3), phi_minus(3, 3)
        real(dp) :: theta(8), theta_dot(8), theta_plus(8), theta_minus(8)
        real(dp) :: u(3, 3), theta_bar(8), x_bar(3, 2), lhs, rhs, h

        x = reshape([0.2_dp, -0.4_dp, 0.7_dp, 0.1_dp, -0.5_dp, 0.8_dp], shape(x))
        x_dot = reshape([-0.3_dp, 0.2_dp, 0.1_dp, -0.4_dp, 0.5_dp, 0.6_dp], &
            shape(x_dot))
        centers = reshape([0.1_dp, -0.2_dp, 0.7_dp, 0.4_dp], shape(centers))
        scales = reshape([0.8_dp, 1.1_dp, 0.6_dp, 0.9_dp], shape(scales))
        call map%initialize_radial(2, centers, scales, status, include_intercept=.true.)
        theta = map%parameters()
        theta_dot = [0.07_dp, -0.11_dp, 0.13_dp, -0.17_dp, &
            0.19_dp, -0.23_dp, 0.29_dp, -0.31_dp]
        call map%jvp(x, theta_dot, x_dot, phi, phi_dot, status)
        h = 1.0e-6_dp
        theta_plus = theta + h*theta_dot
        theta_minus = theta - h*theta_dot
        call map%set_parameters(theta_plus, status)
        call map%evaluate(x + h*x_dot, phi_plus, status)
        call map%set_parameters(theta_minus, status)
        call map%evaluate(x - h*x_dot, phi_minus, status)
        call map%set_parameters(theta, status)
        if (.not. status_ok(status) .or. &
            maxval(abs(phi_dot - (phi_plus - phi_minus)/(2.0_dp*h))) > 3.0e-10_dp) then
            write (error_unit, '(a)') "FAIL [radial] JVP finite difference"
            failures = failures + 1
        end if

        u = reshape([0.4_dp, -0.2_dp, 0.3_dp, 0.5_dp, -0.7_dp, 0.1_dp, &
            0.2_dp, -0.6_dp, 0.8_dp], shape(u))
        call map%vjp(x, u, theta_bar, x_bar, status)
        lhs = sum(u*phi_dot)
        rhs = sum(theta_bar*theta_dot) + sum(x_bar*x_dot)
        if (.not. status_ok(status) .or. abs(lhs - rhs) > 2.0e-12_dp) then
            write (error_unit, '(a,es12.4)') "FAIL [radial] VJP identity=", &
                abs(lhs - rhs)
            failures = failures + 1
        end if
    end subroutine test_radial

    subroutine test_spline(failures)
        integer, intent(inout) :: failures
        type(basis_map_t) :: map
        type(fortnum_status_t) :: status
        real(dp) :: breakpoints(3, 1), x(3, 1), x_dot(3, 1)
        real(dp) :: phi(3, 4), phi_dot(3, 4), phi_plus(3, 4), phi_minus(3, 4)
        real(dp) :: u(3, 4), x_bar(3, 1), lhs, rhs, h
        real(dp), allocatable :: theta_dot(:), theta_bar(:)

        breakpoints(:, 1) = [0.0_dp, 1.0_dp, 2.0_dp]
        x(:, 1) = [0.3_dp, 0.7_dp, 1.4_dp]
        x_dot(:, 1) = [0.1_dp, -0.2_dp, 0.3_dp]
        call map%initialize_spline(1, 3, breakpoints, status)
        allocate(theta_dot(0), theta_bar(0))
        call map%jvp(x, theta_dot, x_dot, phi, phi_dot, status)
        h = 1.0e-6_dp
        call map%evaluate(x + h*x_dot, phi_plus, status)
        call map%evaluate(x - h*x_dot, phi_minus, status)
        if (.not. status_ok(status) .or. &
            maxval(abs(phi_dot - (phi_plus - phi_minus)/(2.0_dp*h))) > 3.0e-9_dp) then
            write (error_unit, '(a)') "FAIL [spline] JVP finite difference"
            failures = failures + 1
        end if

        u = reshape([0.4_dp, -0.2_dp, 0.3_dp, 0.5_dp, -0.7_dp, 0.1_dp, &
            0.2_dp, -0.6_dp, 0.8_dp, -0.4_dp, 0.9_dp, -0.5_dp], shape(u))
        call map%vjp(x, u, theta_bar, x_bar, status)
        lhs = sum(u*phi_dot)
        rhs = sum(x_bar*x_dot)
        if (.not. status_ok(status) .or. abs(lhs - rhs) > 2.0e-12_dp) then
            write (error_unit, '(a,es12.4)') "FAIL [spline] VJP identity=", &
                abs(lhs - rhs)
            failures = failures + 1
        end if
    end subroutine test_spline

    subroutine test_callback(failures)
        integer, intent(inout) :: failures
        type(basis_map_t) :: map
        type(fortnum_status_t) :: status
        real(dp) :: x(3, 2), x_dot(3, 2), phi(3, 2), phi_dot(3, 2)
        real(dp) :: phi_plus(3, 2), phi_minus(3, 2)
        real(dp) :: theta(3), theta_dot(3), theta_plus(3), theta_minus(3)
        real(dp) :: u(3, 2), theta_bar(3), x_bar(3, 2), lhs, rhs, h

        x = reshape([0.2_dp, -0.4_dp, 0.7_dp, 0.1_dp, -0.5_dp, 0.8_dp], shape(x))
        x_dot = reshape([-0.3_dp, 0.2_dp, 0.1_dp, -0.4_dp, 0.5_dp, 0.6_dp], &
            shape(x_dot))
        theta = [1.2_dp, -0.7_dp, 0.4_dp]
        theta_dot = [0.17_dp, -0.23_dp, 0.31_dp]
        call map%initialize_callback(2, 2, theta, callback_value, callback_jvp, &
            callback_vjp, status)
        call map%evaluate(x, phi, status)
        call map%jvp(x, theta_dot, x_dot, phi, phi_dot, status)
        h = 1.0e-6_dp
        theta_plus = theta + h*theta_dot
        theta_minus = theta - h*theta_dot
        call map%set_parameters(theta_plus, status)
        call map%evaluate(x + h*x_dot, phi_plus, status)
        call map%set_parameters(theta_minus, status)
        call map%evaluate(x - h*x_dot, phi_minus, status)
        call map%set_parameters(theta, status)
        if (.not. status_ok(status) .or. map%feature_count() /= 2 .or. &
            map%parameter_count() /= 3 .or. map%static_lowering_eligible() .or. &
            maxval(abs(phi_dot - (phi_plus - phi_minus)/(2.0_dp*h))) > 2.0e-10_dp) then
            write (error_unit, '(a)') "FAIL [callback] value/JVP or lowering boundary"
            failures = failures + 1
        end if

        u = reshape([0.4_dp, -0.2_dp, 0.3_dp, 0.5_dp, -0.7_dp, 0.1_dp], shape(u))
        call map%vjp(x, u, theta_bar, x_bar, status)
        lhs = sum(u*phi_dot)
        rhs = sum(theta_bar*theta_dot) + sum(x_bar*x_dot)
        if (.not. status_ok(status) .or. abs(lhs - rhs) > 1.0e-12_dp) then
            write (error_unit, '(a,es12.4)') "FAIL [callback] VJP identity=", &
                abs(lhs - rhs)
            failures = failures + 1
        end if
    end subroutine test_callback

    subroutine callback_value(x, theta, phi, status)
        real(dp), intent(in) :: x(:, :), theta(:)
        real(dp), intent(out) :: phi(:, :)
        type(fortnum_status_t), intent(out) :: status

        phi(:, 1) = theta(1) + theta(2)*x(:, 1)
        phi(:, 2) = theta(3)*x(:, 2)**2
        call status_set_ok(status)
    end subroutine callback_value

    subroutine callback_jvp(x, theta, x_dot, theta_dot, phi, phi_dot, status)
        real(dp), intent(in) :: x(:, :), theta(:), x_dot(:, :), theta_dot(:)
        real(dp), intent(out) :: phi(:, :), phi_dot(:, :)
        type(fortnum_status_t), intent(out) :: status

        call callback_value(x, theta, phi, status)
        phi_dot(:, 1) = theta_dot(1) + theta_dot(2)*x(:, 1) + &
            theta(2)*x_dot(:, 1)
        phi_dot(:, 2) = theta_dot(3)*x(:, 2)**2 + &
            2.0_dp*theta(3)*x(:, 2)*x_dot(:, 2)
    end subroutine callback_jvp

    subroutine callback_vjp(x, theta, u, theta_bar, x_bar, status)
        real(dp), intent(in) :: x(:, :), theta(:), u(:, :)
        real(dp), intent(out) :: theta_bar(:), x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status

        theta_bar = 0.0_dp
        x_bar = 0.0_dp
        theta_bar(1) = sum(u(:, 1))
        theta_bar(2) = sum(u(:, 1)*x(:, 1))
        theta_bar(3) = sum(u(:, 2)*x(:, 2)**2)
        x_bar(:, 1) = u(:, 1)*theta(2)
        x_bar(:, 2) = u(:, 2)*2.0_dp*theta(3)*x(:, 2)
        call status_set_ok(status)
    end subroutine callback_vjp

    subroutine status_set_ok(status)
        type(fortnum_status_t), intent(out) :: status

        status%code = 0
        status%msg = ""
    end subroutine status_set_ok

end program test_basis
