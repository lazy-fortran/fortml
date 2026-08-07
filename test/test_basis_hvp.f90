program test_basis_hvp
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_basis, only: basis_map_t
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    implicit none

    integer :: failures
    type(fortnum_status_t) :: status
    type(basis_map_t) :: map
    real(dp) :: x(4, 2), x_dot(4, 2), frequencies(2, 2), centers(2, 2), scales(2, 2)
    real(dp) :: breakpoints(4, 1)

    failures = 0
    x = reshape([0.17_dp, -0.31_dp, 0.52_dp, 0.91_dp, &
        -0.42_dp, 0.28_dp, 0.73_dp, -0.66_dp], shape(x))
    x_dot = reshape([-0.23_dp, 0.19_dp, 0.07_dp, -0.13_dp, &
        0.29_dp, -0.11_dp, 0.17_dp, 0.05_dp], shape(x_dot))

    call map%initialize_polynomial(2, 3, status, include_intercept=.true.)
    call check_map(map, x, x_dot, 2.0e-8_dp, failures, "polynomial")

    frequencies = reshape([0.8_dp, 1.3_dp, 1.1_dp, 0.6_dp], shape(frequencies))
    call map%initialize_fourier(2, frequencies, status, include_intercept=.true.)
    call check_map(map, x, x_dot, 3.0e-8_dp, failures, "fourier")

    centers = reshape([0.1_dp, -0.2_dp, 0.7_dp, 0.4_dp], shape(centers))
    scales = reshape([0.8_dp, 1.1_dp, 0.6_dp, 0.9_dp], shape(scales))
    call map%initialize_radial(2, centers, scales, status, include_intercept=.true.)
    call check_map(map, x, x_dot, 3.0e-8_dp, failures, "radial")

    breakpoints(:, 1) = [-1.0_dp, 0.0_dp, 1.0_dp, 2.0_dp]
    call map%initialize_spline(1, 3, breakpoints, status)
    call check_map(map, x(:, 1:1), x_dot(:, 1:1), 2.0e-7_dp, failures, "spline")

    call map%initialize_callback(2, 1, [0.7_dp], callback_value, callback_jvp, &
        callback_vjp, status)
    call check_callback_refusal(map, x, x_dot, failures)

    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, " basis HVP test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS"

contains

    subroutine check_map(map, x, x_dot, tolerance, failures, label)
        type(basis_map_t), intent(inout) :: map
        real(dp), intent(in) :: x(:, :), x_dot(:, :), tolerance
        integer, intent(inout) :: failures
        character(*), intent(in) :: label
        type(fortnum_status_t) :: local_status
        real(dp), allocatable :: theta(:), theta_dot(:), theta_bar(:)
        real(dp), allocatable :: theta_bar_plus(:), theta_bar_minus(:)
        real(dp), allocatable :: x_bar(:, :), x_bar_plus(:, :), x_bar_minus(:, :)
        real(dp), allocatable :: u(:, :), theta_hvp(:), x_hvp(:, :)
        real(dp) :: h, error_theta, error_x
        integer :: i

        h = 1.0e-5_dp
        allocate(u(size(x, 1), map%feature_count()))
        u = reshape([(0.13_dp*real(i, dp) - 0.41_dp, i=1, size(u))], shape(u))
        theta = map%parameters()
        allocate(theta_dot(size(theta)), theta_bar(size(theta)), &
            theta_bar_plus(size(theta)), theta_bar_minus(size(theta)), &
            theta_hvp(size(theta)))
        theta_dot = 0.0_dp
        if (size(theta) > 0) theta_dot = [(0.11_dp*real(i, dp) - 0.29_dp, &
            i=1, size(theta))]
        allocate(x_bar(size(x, 1), size(x, 2)), x_bar_plus(size(x, 1), size(x, 2)), &
            x_bar_minus(size(x, 1), size(x, 2)), x_hvp(size(x, 1), size(x, 2)))

        call map%vjp(x, u, theta_bar, x_bar, local_status)
        call map%hvp(x, u, theta_dot, x_dot, theta_hvp, x_hvp, local_status)
        if (.not. status_ok(local_status)) then
            write (error_unit, '(a,a)') "FAIL [", label//"] analytic HVP status"
            failures = failures + 1
            return
        end if
        call map%set_parameters(theta + h*theta_dot, local_status)
        call map%vjp(x + h*x_dot, u, theta_bar_plus, x_bar_plus, local_status)
        call map%set_parameters(theta - h*theta_dot, local_status)
        call map%vjp(x - h*x_dot, u, theta_bar_minus, x_bar_minus, local_status)
        call map%set_parameters(theta, local_status)
        error_theta = 0.0_dp
        if (size(theta) > 0) error_theta = maxval(abs(theta_hvp - &
            (theta_bar_plus - theta_bar_minus)/(2.0_dp*h)))
        error_x = maxval(abs(x_hvp - (x_bar_plus - x_bar_minus)/(2.0_dp*h)))
        if (error_theta > tolerance .or. error_x > tolerance) then
            write (error_unit, '(a,a,a,2(es12.4,1x))') "FAIL [", label, &
                "] HVP errors=", error_theta, error_x
            failures = failures + 1
        end if
    end subroutine check_map

    subroutine check_callback_refusal(map, x, x_dot, failures)
        type(basis_map_t), intent(inout) :: map
        real(dp), intent(in) :: x(:, :), x_dot(:, :)
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: local_status
        real(dp) :: u(size(x, 1), 1), theta_dot(1), theta_hvp(1), x_hvp(size(x, 1), 2)
        integer :: i

        do i = 1, size(u)
            u(i, 1) = 0.2_dp*real(i, dp)
        end do
        theta_dot = 0.1_dp
        call map%hvp(x, u, theta_dot, x_dot, theta_hvp, x_hvp, local_status)
        if (local_status%code /= FORTNUM_NOT_IMPLEMENTED) then
            write (error_unit, '(a)') "FAIL [callback] missing typed HVP refusal"
            failures = failures + 1
        end if
    end subroutine check_callback_refusal

    subroutine callback_value(x, theta, phi, status)
        real(dp), intent(in) :: x(:, :), theta(:)
        real(dp), intent(out) :: phi(:, :)
        type(fortnum_status_t), intent(out) :: status

        phi(:, 1) = theta(1)*x(:, 1)
        call status_ok_set(status)
    end subroutine callback_value

    subroutine callback_jvp(x, theta, x_dot, theta_dot, phi, phi_dot, status)
        real(dp), intent(in) :: x(:, :), theta(:), x_dot(:, :), theta_dot(:)
        real(dp), intent(out) :: phi(:, :), phi_dot(:, :)
        type(fortnum_status_t), intent(out) :: status

        call callback_value(x, theta, phi, status)
        phi_dot(:, 1) = theta_dot(1)*x(:, 1) + theta(1)*x_dot(:, 1)
    end subroutine callback_jvp

    subroutine callback_vjp(x, theta, u, theta_bar, x_bar, status)
        real(dp), intent(in) :: x(:, :), theta(:), u(:, :)
        real(dp), intent(out) :: theta_bar(:), x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status

        theta_bar(1) = sum(u(:, 1)*x(:, 1))
        x_bar(:, 1) = u(:, 1)*theta(1)
        x_bar(:, 2) = 0.0_dp
        call status_ok_set(status)
    end subroutine callback_vjp

    subroutine status_ok_set(status)
        use fortnum_status, only: status_set, FORTNUM_OK
        type(fortnum_status_t), intent(out) :: status
        call status_set(status, FORTNUM_OK, "")
    end subroutine status_ok_set

end program test_basis_hvp
