program test_basis_random_fourier
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_basis, only: basis_map_t, make_random_fourier_basis
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_DOMAIN_ERROR
    implicit none

    integer :: failures
    type(fortnum_status_t) :: status
    type(basis_map_t) :: map
    real(dp) :: frequencies(3, 2), phases(3)
    real(dp) :: x(4, 2), x_dot(4, 2), phi(4, 4), expected(4, 4)
    real(dp) :: phi_dot(4, 4), phi_plus(4, 4), phi_minus(4, 4)
    real(dp) :: u(4, 4), x_bar(4, 2), x_bar_plus(4, 2), x_bar_minus(4, 2)
    real(dp) :: x_hvp(4, 2), theta_bar(0), theta_hvp(0)
    real(dp) :: h, jvp_error, hvp_error, adjoint_error
    integer :: i, j, k

    failures = 0
    frequencies = reshape([0.7_dp, -1.1_dp, 0.3_dp, 0.4_dp, &
        1.2_dp, 0.8_dp], shape(frequencies))
    phases = [0.2_dp, -0.5_dp, 1.1_dp]
    x = reshape([0.17_dp, -0.31_dp, 0.52_dp, 0.91_dp, &
        -0.42_dp, 0.28_dp, 0.73_dp, -0.66_dp], shape(x))
    x_dot = reshape([-0.23_dp, 0.19_dp, 0.07_dp, -0.13_dp, &
        0.29_dp, -0.11_dp, 0.17_dp, 0.05_dp], shape(x_dot))

    map = make_random_fourier_basis(2, frequencies, phases, status, &
        include_intercept=.true.)
    if (.not. status_ok(status) .or. .not. map%valid() .or. &
            map%feature_count() /= 4 .or. map%parameter_count() /= 0) then
        write (error_unit, '(a)') "FAIL [random Fourier] construction contract"
        failures = failures + 1
    end if

    call map%evaluate(x, phi, status)
    expected(:, 1) = 1.0_dp
    do k = 1, 3
        do i = 1, size(x, 1)
            expected(i, k + 1) = sqrt(2.0_dp/3.0_dp)*cos(phases(k) + &
                sum(frequencies(k, :)*x(i, :)))
        end do
    end do
    if (.not. status_ok(status) .or. maxval(abs(phi - expected)) > 2.0e-14_dp) then
        write (error_unit, '(a,es12.4)') "FAIL [random Fourier] value error=", &
            maxval(abs(phi - expected))
        failures = failures + 1
    end if

    call map%jvp(x, theta_bar, x_dot, phi, phi_dot, status)
    h = 1.0e-6_dp
    call map%evaluate(x + h*x_dot, phi_plus, status)
    call map%evaluate(x - h*x_dot, phi_minus, status)
    jvp_error = maxval(abs(phi_dot - (phi_plus - phi_minus)/(2.0_dp*h)))
    if (.not. status_ok(status) .or. jvp_error > 2.0e-9_dp) then
        write (error_unit, '(a,es12.4)') "FAIL [random Fourier] JVP error=", &
            jvp_error
        failures = failures + 1
    end if

    u = reshape([(0.13_dp*real(i, dp) - 0.41_dp, i=1, size(u))], shape(u))
    call map%vjp(x, u, theta_bar, x_bar, status)
    adjoint_error = abs(sum(u*phi_dot) - sum(x_bar*x_dot))
    if (.not. status_ok(status) .or. adjoint_error > 2.0e-12_dp) then
        write (error_unit, '(a,es12.4)') &
            "FAIL [random Fourier] VJP identity error=", adjoint_error
        failures = failures + 1
    end if

    call map%hvp(x, u, theta_bar, x_dot, theta_hvp, x_hvp, status)
    call map%vjp(x + h*x_dot, u, theta_bar, x_bar_plus, status)
    call map%vjp(x - h*x_dot, u, theta_bar, x_bar_minus, status)
    hvp_error = maxval(abs(x_hvp - (x_bar_plus - x_bar_minus)/(2.0_dp*h)))
    if (.not. status_ok(status) .or. hvp_error > 3.0e-8_dp) then
        write (error_unit, '(a,es12.4)') "FAIL [random Fourier] HVP error=", &
            hvp_error
        failures = failures + 1
    end if

    call map%set_parameters([0.1_dp], status)
    if (status%code /= FORTNUM_DOMAIN_ERROR) then
        write (error_unit, '(a)') &
            "FAIL [random Fourier] fixed-state parameter refusal"
        failures = failures + 1
    end if

    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, &
            " random Fourier basis test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS random Fourier basis independent oracle"
end program test_basis_random_fourier
