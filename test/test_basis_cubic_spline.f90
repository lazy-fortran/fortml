program test_basis_cubic_spline
    !! Independent Cox--de Boor oracle for the public cubic spline shortcut.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_basis, only: basis_map_t, make_cubic_spline_basis
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    type(basis_map_t) :: basis
    type(fortnum_status_t) :: status
    real(dp) :: breakpoints(5, 2), x(4, 2), x_dot(4, 2)
    real(dp), allocatable :: phi(:, :), phi_dot(:, :), expected(:, :)
    real(dp), allocatable :: phi_plus(:, :), phi_minus(:, :)
    real(dp), allocatable :: u(:, :), theta(:), theta_bar(:), x_bar(:, :)
    real(dp), allocatable :: x_bar_plus(:, :), x_bar_minus(:, :)
    real(dp), allocatable :: x_bar_fd(:, :), x_probe(:, :)
    real(dp), allocatable :: theta_hvp(:), x_hvp(:, :)
    real(dp) :: h, err, err_jvp, err_vjp, err_hvp
    integer :: i, j, offset, ncoef, n_samples, n_features

    breakpoints(:, 1) = [0.0_dp, 0.4_dp, 0.9_dp, 1.5_dp, 2.0_dp]
    breakpoints(:, 2) = [-1.0_dp, -0.2_dp, 0.3_dp, 1.1_dp, 2.0_dp]
    x = reshape([0.17_dp, 0.62_dp, 1.23_dp, 1.83_dp, &
        -0.71_dp, -0.05_dp, 0.61_dp, 1.57_dp], shape(x))
    x_dot = reshape([0.11_dp, -0.07_dp, 0.09_dp, -0.13_dp, &
        -0.08_dp, 0.12_dp, -0.05_dp, 0.06_dp], shape(x_dot))

    basis = make_cubic_spline_basis(2, breakpoints, status, include_intercept=.true.)
    if (.not. status_ok(status)) error stop "cubic spline construction failed"
    ncoef = size(breakpoints, 1) + 4 - 2
    if (basis%feature_count() /= 2*ncoef + 1 .or. basis%parameter_count() /= 0) then
        error stop "cubic spline metadata mismatch"
    end if

    n_samples = size(x, 1)
    n_features = basis%feature_count()
    allocate(phi(n_samples, n_features), phi_dot(n_samples, n_features), &
        expected(n_samples, n_features), phi_plus(n_samples, n_features), &
        phi_minus(n_samples, n_features), u(n_samples, n_features), theta(0), &
        theta_bar(0), x_bar(n_samples, size(x, 2)), &
        x_bar_plus(n_samples, size(x, 2)), x_bar_minus(n_samples, size(x, 2)), &
        x_bar_fd(n_samples, size(x, 2)), x_probe(n_samples, size(x, 2)), &
        theta_hvp(0), x_hvp(n_samples, size(x, 2)))
    call basis%evaluate(x, phi, status)
    if (.not. status_ok(status)) error stop "cubic spline evaluate failed"
    expected = 0.0_dp
    expected(:, 1) = 1.0_dp
    offset = 2
    do j = 1, size(x, 2)
        do i = 1, size(x, 1)
            call oracle_values(x(i, j), breakpoints(:, j), expected(i, offset:offset+ncoef-1))
        end do
        offset = offset + ncoef
    end do
    err = maxval(abs(phi - expected))
    if (err > 2.0e-12_dp) then
        write (error_unit, '(a,es12.4)') "FAIL cubic spline Cox--de Boor error=", err
        error stop 1
    end if

    call basis%jvp(x, theta, x_dot, phi, phi_dot, status)
    h = 2.0e-6_dp
    call basis%evaluate(x + h*x_dot, phi_plus, status)
    call basis%evaluate(x - h*x_dot, phi_minus, status)
    err_jvp = maxval(abs(phi_dot - (phi_plus - phi_minus)/(2.0_dp*h)))

    do i = 1, size(u)
        u(i, :) = 0.17_dp*real(i, dp) - 0.29_dp
    end do
    call basis%vjp(x, u, theta_bar, x_bar, status)
    do j = 1, size(x, 2)
        do i = 1, size(x, 1)
            x_probe = x
            x_probe(i, j) = x_probe(i, j) + h
            call basis%evaluate(x_probe, phi_plus, status)
            x_probe(i, j) = x_probe(i, j) - 2.0_dp*h
            call basis%evaluate(x_probe, phi_minus, status)
            x_bar_fd(i, j) = sum(u(i, :)*(phi_plus(i, :) - phi_minus(i, :)))/(2.0_dp*h)
        end do
    end do
    err_vjp = maxval(abs(x_bar - x_bar_fd))
    call basis%vjp(x + h*x_dot, u, theta_bar, x_bar_plus, status)
    call basis%vjp(x - h*x_dot, u, theta_bar, x_bar_minus, status)
    call basis%hvp(x, u, theta, x_dot, theta_hvp, x_hvp, status)
    err_hvp = maxval(abs(x_hvp - (x_bar_plus - x_bar_minus)/(2.0_dp*h)))
    if (.not. status_ok(status) .or. err_jvp > 4.0e-8_dp .or. &
            err_vjp > 2.0e-8_dp .or. err_hvp > 2.0e-5_dp) then
        write (error_unit, '(a,3(es12.4,1x))') &
            "FAIL cubic spline products errors=", err_jvp, err_vjp, err_hvp
        error stop 1
    end if
    write (*, '(a,3(es12.4,1x))') &
        "PASS cubic spline independent oracle errors=", err, err_jvp, err_hvp

contains

    subroutine oracle_values(x0, breaks, values)
        real(dp), intent(in) :: x0, breaks(:)
        real(dp), intent(out) :: values(:)
        real(dp), allocatable :: knots(:)
        integer :: k, m, ncoef0, p, q

        k = 4
        m = size(breaks)
        ncoef0 = m + k - 2
        allocate(knots(m + 2*(k - 1)))
        knots(1:k) = breaks(1)
        p = k
        do q = 2, m - 1
            p = p + 1
            knots(p) = breaks(q)
        end do
        do q = 1, k
            p = p + 1
            knots(p) = breaks(m)
        end do
        do q = 1, ncoef0
            values(q) = cox_de_boor(q, k, x0, knots)
        end do
    end subroutine oracle_values

    recursive function cox_de_boor(i0, k, x0, knots) result(value)
        integer, intent(in) :: i0, k
        real(dp), intent(in) :: x0, knots(:)
        real(dp) :: value, left, right, denominator

        value = 0.0_dp
        if (k == 1) then
            if ((knots(i0) <= x0 .and. x0 < knots(i0 + 1)) .or. &
                    (x0 == knots(size(knots)) .and. i0 == size(knots) - 1)) value = 1.0_dp
            return
        end if
        denominator = knots(i0 + k - 1) - knots(i0)
        if (denominator > 0.0_dp) then
            left = (x0 - knots(i0))/denominator * cox_de_boor(i0, k - 1, x0, knots)
        else
            left = 0.0_dp
        end if
        denominator = knots(i0 + k) - knots(i0 + 1)
        if (denominator > 0.0_dp) then
            right = (knots(i0 + k) - x0)/denominator * &
                cox_de_boor(i0 + 1, k - 1, x0, knots)
        else
            right = 0.0_dp
        end if
        value = left + right
    end function cox_de_boor

end program test_basis_cubic_spline
