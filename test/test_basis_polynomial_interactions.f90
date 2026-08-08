program test_basis_polynomial_interactions
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_basis, only: basis_map_t, make_polynomial_interaction_basis
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    type(basis_map_t) :: basis, basis3
    type(fortnum_status_t) :: status
    real(dp) :: x(3, 2), x_dot(3, 2), phi(3, 6), phi_dot(3, 6)
    real(dp) :: phi_plus(3, 6), phi_minus(3, 6), u(3, 6)
    real(dp) :: theta_dot(0), theta_hvp(0), x_bar(3, 2), x_hvp(3, 2)
    real(dp) :: x_bar_plus(3, 2), x_bar_minus(3, 2), h, lhs, rhs
    real(dp) :: x3(2, 3), phi3(2, 9)
    integer :: failures

    failures = 0
    x = reshape([0.2_dp, -0.4_dp, 0.7_dp, 0.1_dp, -0.5_dp, 0.8_dp], shape(x))
    x_dot = reshape([-0.3_dp, 0.2_dp, 0.1_dp, -0.4_dp, 0.5_dp, 0.6_dp], &
        shape(x_dot))
    u = reshape([0.4_dp, -0.2_dp, 0.3_dp, 0.5_dp, -0.7_dp, 0.1_dp, &
        0.2_dp, -0.6_dp, 0.8_dp, -0.4_dp, -0.3_dp, 0.9_dp, &
        -0.5_dp, 0.2_dp, 0.6_dp, 0.7_dp, -0.1_dp, 0.4_dp], shape(u))

    basis = make_polynomial_interaction_basis(2, 2, status, include_intercept=.true.)
    call basis%evaluate(x, phi, status)
    if (.not. status_ok(status)) then
        write (error_unit, '(a)') "FAIL [polynomial interactions] construction/evaluation"
        error stop 1
    end if
    if (maxval(abs(phi(:, 1) - 1.0_dp)) > 1.0e-14_dp .or. &
            maxval(abs(phi(:, 2) - x(:, 1))) > 1.0e-14_dp .or. &
            maxval(abs(phi(:, 3) - x(:, 2))) > 1.0e-14_dp .or. &
            maxval(abs(phi(:, 4) - x(:, 1)**2)) > 1.0e-14_dp .or. &
            maxval(abs(phi(:, 5) - x(:, 1)*x(:, 2))) > 1.0e-14_dp .or. &
            maxval(abs(phi(:, 6) - x(:, 2)**2)) > 1.0e-14_dp) then
        write (error_unit, '(a)') "FAIL [polynomial interactions] monomial ordering"
        failures = failures + 1
    end if

    x3 = reshape([0.2_dp, -0.4_dp, 0.7_dp, 0.1_dp, -0.5_dp, 0.8_dp], shape(x3))
    basis3 = make_polynomial_interaction_basis(3, 2, status)
    call basis3%evaluate(x3, phi3, status)
    if (.not. status_ok(status) .or. basis3%feature_count() /= 9 .or. &
            maxval(abs(phi3(:, 1) - x3(:, 1))) > 1.0e-14_dp .or. &
            maxval(abs(phi3(:, 2) - x3(:, 2))) > 1.0e-14_dp .or. &
            maxval(abs(phi3(:, 3) - x3(:, 3))) > 1.0e-14_dp .or. &
            maxval(abs(phi3(:, 5) - x3(:, 1)*x3(:, 2))) > 1.0e-14_dp .or. &
            maxval(abs(phi3(:, 6) - x3(:, 1)*x3(:, 3))) > 1.0e-14_dp .or. &
            maxval(abs(phi3(:, 8) - x3(:, 2)*x3(:, 3))) > 1.0e-14_dp) then
        write (error_unit, '(a)') "FAIL [polynomial interactions] 3D enumeration"
        failures = failures + 1
    end if

    call basis%jvp(x, theta_dot, x_dot, phi, phi_dot, status)
    h = 1.0e-6_dp
    call basis%evaluate(x + h*x_dot, phi_plus, status)
    call basis%evaluate(x - h*x_dot, phi_minus, status)
    if (.not. status_ok(status) .or. maxval(abs(phi_dot - &
            (phi_plus - phi_minus)/(2.0_dp*h))) > 2.0e-10_dp) then
        write (error_unit, '(a)') "FAIL [polynomial interactions] JVP oracle"
        failures = failures + 1
    end if

    call basis%vjp(x, u, theta_dot, x_bar, status)
    lhs = sum(u*phi_dot)
    rhs = sum(x_bar*x_dot)
    if (.not. status_ok(status) .or. abs(lhs - rhs) > 1.0e-12_dp) then
        write (error_unit, '(a,es12.4)') "FAIL [polynomial interactions] VJP identity=", &
            abs(lhs - rhs)
        failures = failures + 1
    end if

    call basis%hvp(x, u, theta_dot, x_dot, theta_hvp, x_hvp, status)
    call basis%vjp(x + h*x_dot, u, theta_dot, x_bar_plus, status)
    call basis%vjp(x - h*x_dot, u, theta_dot, x_bar_minus, status)
    if (.not. status_ok(status) .or. maxval(abs(x_hvp - &
            (x_bar_plus - x_bar_minus)/(2.0_dp*h))) > 5.0e-9_dp) then
        write (error_unit, '(a)') "FAIL [polynomial interactions] HVP oracle"
        failures = failures + 1
    end if

    if (failures /= 0) error stop 1
    write (*, '(a)') "PASS polynomial interaction basis independent oracle"
end program test_basis_polynomial_interactions
