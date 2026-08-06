program test_basis
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_basis, only: basis_map_t
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer :: failures

    failures = 0
    call test_polynomial(failures)
    call test_fourier(failures)
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

end program test_basis
