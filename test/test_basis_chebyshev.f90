program test_basis_chebyshev
    use, intrinsic :: iso_fortran_env, only: dp => real64
    use fortml_basis, only: basis_map_t, make_chebyshev_basis
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    type(basis_map_t) :: basis
    type(fortnum_status_t) :: status
    real(dp) :: x(4, 2), x_dot(4, 2), phi(4, 9), phi_dot(4, 9)
    real(dp) :: phi_expected(4, 9), phi_plus(4, 9), phi_minus(4, 9)
    real(dp) :: u(4, 9), x_bar(4, 2), x_bar_plus(4, 2), x_bar_minus(4, 2)
    real(dp) :: x_hvp(4, 2), theta_hvp(0), theta_dot(0), theta_bar(0)
    real(dp) :: h, lhs, rhs
    integer :: i, j, column

    x = reshape([0.1_dp, -0.2_dp, 0.4_dp, 0.3_dp, &
        -0.5_dp, 0.8_dp, 0.7_dp, -0.6_dp], shape(x))
    x_dot = reshape([-0.3_dp, 0.2_dp, 0.1_dp, -0.4_dp, &
        0.5_dp, 0.6_dp, -0.2_dp, 0.3_dp], shape(x_dot))
    basis = make_chebyshev_basis(2, 4, status, include_intercept=.true.)
    if (.not. status_ok(status) .or. basis%feature_count() /= 9 .or. &
            basis%parameter_count() /= 0) error stop "invalid Chebyshev construction"

    phi_expected = 0.0_dp
    phi_expected(:, 1) = 1.0_dp
    column = 2
    do j = 1, 2
        do i = 1, 4
            phi_expected(i, column) = x(i, j)
            phi_expected(i, column + 1) = 2.0_dp*x(i, j)**2 - 1.0_dp
            phi_expected(i, column + 2) = 4.0_dp*x(i, j)**3 - 3.0_dp*x(i, j)
            phi_expected(i, column + 3) = 8.0_dp*x(i, j)**4 - &
                8.0_dp*x(i, j)**2 + 1.0_dp
        end do
        column = column + 4
    end do
    call basis%evaluate(x, phi, status)
    if (.not. status_ok(status) .or. maxval(abs(phi - phi_expected)) > 1.0e-13_dp) &
        error stop "Chebyshev recurrence does not match independent oracle"

    call basis%jvp(x, theta_dot, x_dot, phi, phi_dot, status)
    h = 1.0e-6_dp
    call basis%evaluate(x + h*x_dot, phi_plus, status)
    call basis%evaluate(x - h*x_dot, phi_minus, status)
    if (.not. status_ok(status) .or. &
            maxval(abs(phi_dot - (phi_plus - phi_minus)/(2.0_dp*h))) > 3.0e-10_dp) &
        error stop "Chebyshev JVP finite-difference check failed"

    u = reshape([ &
        0.4_dp, -0.2_dp, 0.3_dp, 0.5_dp, -0.7_dp, 0.1_dp, 0.2_dp, -0.6_dp, 0.8_dp, &
        -0.4_dp, -0.3_dp, 0.9_dp, -0.5_dp, 0.2_dp, 0.6_dp, 0.7_dp, -0.1_dp, 0.4_dp, &
        -0.8_dp, 0.3_dp, 0.2_dp, -0.9_dp, 0.5_dp, 0.1_dp, -0.6_dp, 0.8_dp, -0.2_dp, &
        0.3_dp, 0.4_dp, -0.7_dp, 0.9_dp, -0.1_dp, 0.5_dp, 0.6_dp, -0.3_dp, 0.2_dp], &
        shape(u))
    call basis%vjp(x, u, theta_bar, x_bar, status)
    lhs = sum(u*phi_dot)
    rhs = sum(x_bar*x_dot)
    if (.not. status_ok(status) .or. abs(lhs - rhs) > 1.0e-12_dp) &
        error stop "Chebyshev VJP adjoint check failed"

    call basis%hvp(x, u, theta_dot, x_dot, theta_hvp, x_hvp, status)
    call basis%vjp(x + h*x_dot, u, theta_bar, x_bar_plus, status)
    call basis%vjp(x - h*x_dot, u, theta_bar, x_bar_minus, status)
    if (.not. status_ok(status) .or. maxval(abs(x_hvp - &
            (x_bar_plus - x_bar_minus)/(2.0_dp*h))) > 3.0e-9_dp) &
        error stop "Chebyshev HVP finite-difference check failed"

    write (*, '(a)') "PASS"
end program test_basis_chebyshev
