program test_column_pipeline
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_basis, only: basis_map_t, make_fourier_basis, &
        make_polynomial_basis
    use fortml_column_pipeline, only: column_basis_pipeline_t, &
        make_column_basis_pipeline
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, &
        FORTML_DEVICE_CUDA
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_DOMAIN_ERROR, &
        FORTNUM_NOT_IMPLEMENTED
    implicit none

    integer :: failures

    failures = 0
    call test_selected_transform_products(failures)
    call test_device_dispatch(failures)
    call test_column_refusals(failures)
    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, &
            " column pipeline test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS"

contains

    subroutine test_selected_transform_products(failures)
        integer, intent(inout) :: failures
        type(basis_map_t) :: fourier, polynomial
        type(column_basis_pipeline_t) :: pipeline
        type(fortnum_status_t) :: status
        real(dp) :: x(5, 3), x_dot(5, 3), x_bar(5, 3)
        real(dp) :: phi_expected(5, 4), lhs, rhs, h
        real(dp) :: u(5, 4)
        real(dp), allocatable :: phi(:, :), phi_dot(:, :), phi_plus(:, :), &
            phi_minus(:, :), theta(:), theta_dot(:), theta_plus(:), &
            theta_minus(:), theta_bar(:)
        real(dp) :: frequency, argument(5), argument_dot(5)
        integer :: i

        x(:, 1) = [-0.8_dp, -0.2_dp, 0.3_dp, 0.9_dp, 1.4_dp]
        x(:, 2) = [1.2_dp, -0.7_dp, 0.4_dp, 0.1_dp, -1.1_dp]
        x(:, 3) = [0.5_dp, -0.4_dp, 1.3_dp, -0.9_dp, 0.2_dp]
        x_dot(:, 1) = [0.2_dp, -0.1_dp, 0.4_dp, -0.3_dp, 0.5_dp]
        x_dot(:, 2) = [-0.8_dp, 0.6_dp, 0.2_dp, 0.1_dp, -0.4_dp]
        x_dot(:, 3) = [0.3_dp, 0.7_dp, -0.2_dp, 0.5_dp, -0.6_dp]

        fourier = make_fourier_basis(1, reshape([0.8_dp], [1, 1]), status)
        polynomial = make_polynomial_basis(1, 2, status)
        pipeline = make_column_basis_pipeline(3, status)
        call pipeline%append(fourier, [3], status)
        call pipeline%append(polynomial, [1], status)
        call pipeline%fit(x, status)
        allocate(phi(5, pipeline%feature_count()), phi_dot(5, 4))
        allocate(phi_plus(5, 4), phi_minus(5, 4), theta(pipeline%parameter_count()))
        allocate(theta_dot(size(theta)), theta_plus(size(theta)), &
            theta_minus(size(theta)), theta_bar(size(theta)))
        theta = pipeline%parameters()
        theta_dot = [0.17_dp]
        call pipeline%transform(x, phi, status)
        frequency = exp(theta(1))
        argument = frequency*x(:, 3)
        phi_expected(:, 1) = sin(argument)
        phi_expected(:, 2) = cos(argument)
        phi_expected(:, 3) = x(:, 1)
        phi_expected(:, 4) = x(:, 1)**2
        if (.not. status_ok(status) .or. pipeline%stage_count() /= 2 .or. &
            pipeline%feature_count() /= 4 .or. pipeline%parameter_count() /= 1 .or. &
            .not. pipeline%is_fitted() .or. &
            maxval(abs(phi - phi_expected)) > 1.0e-13_dp) then
            write (error_unit, '(a)') &
                "FAIL [column pipeline] selected feature oracle"
            failures = failures + 1
        end if

        call pipeline%jvp(x, theta_dot, x_dot, phi, phi_dot, status)
        h = 1.0e-6_dp
        theta_plus = theta + h*theta_dot
        theta_minus = theta - h*theta_dot
        call pipeline%set_parameters(theta_plus, status)
        call pipeline%transform(x + h*x_dot, phi_plus, status)
        call pipeline%set_parameters(theta_minus, status)
        call pipeline%transform(x - h*x_dot, phi_minus, status)
        call pipeline%set_parameters(theta, status)
        if (.not. status_ok(status) .or. maxval(abs(phi_dot - &
            (phi_plus - phi_minus)/(2.0_dp*h))) > 5.0e-9_dp) then
            write (error_unit, '(a)') &
                "FAIL [column pipeline] JVP finite-difference oracle"
            failures = failures + 1
        end if

        do i = 1, 4
            u(:, i) = [0.13_dp, -0.22_dp, 0.31_dp, -0.17_dp, 0.29_dp] + &
                0.07_dp*real(i, dp)
        end do
        call pipeline%vjp(x, u, theta_bar, x_bar, status)
        lhs = sum(u*phi_dot)
        rhs = sum(theta_bar*theta_dot) + sum(x_bar*x_dot)
        if (.not. status_ok(status) .or. abs(lhs - rhs) > 5.0e-10_dp .or. &
            maxval(abs(x_bar(:, 2))) > 1.0e-14_dp) then
            write (error_unit, '(a,es12.4)') &
                "FAIL [column pipeline] VJP/scatter identity=", abs(lhs - rhs)
            failures = failures + 1
        end if
    end subroutine test_selected_transform_products

    subroutine test_device_dispatch(failures)
        integer, intent(inout) :: failures
        type(basis_map_t) :: polynomial
        type(column_basis_pipeline_t) :: pipeline
        type(fortml_device_t) :: cpu, cuda, unselected
        type(fortnum_status_t) :: status
        real(dp) :: x(4, 2), x_dot(4, 2), u(4, 2)
        real(dp), allocatable :: phi(:, :), phi_reference(:, :), phi_dot(:, :), &
            phi_dot_reference(:, :), theta_dot(:), theta_bar(:), &
            theta_bar_reference(:), x_bar(:, :), x_bar_reference(:, :), &
            theta_hvp(:), theta_hvp_reference(:), x_hvp(:, :), &
            x_hvp_reference(:, :)

        x = reshape([-0.8_dp, -0.2_dp, 0.4_dp, 1.1_dp, &
            1.2_dp, -0.7_dp, 0.3_dp, -1.0_dp], shape(x))
        x_dot = reshape([0.2_dp, -0.1_dp, 0.4_dp, -0.3_dp, &
            -0.8_dp, 0.6_dp, 0.2_dp, 0.1_dp], shape(x_dot))
        u = reshape([0.13_dp, -0.22_dp, 0.31_dp, -0.17_dp, &
            0.29_dp, 0.04_dp, -0.11_dp, 0.18_dp], shape(u))
        polynomial = make_fourier_basis(1, reshape([0.8_dp], [1, 1]), status)
        pipeline = make_column_basis_pipeline(2, status)
        call pipeline%append(polynomial, [2], status, name="polynomial")
        call pipeline%fit(x, status)
        call check(status_ok(status) .and. &
            pipeline%device_supported(FORTML_DEVICE_CPU) .and. &
            .not. pipeline%device_supported(FORTML_DEVICE_CUDA), &
            "device capability metadata", failures)

        allocate(phi(4, 2), phi_reference(4, 2), phi_dot(4, 2), &
            phi_dot_reference(4, 2), theta_dot(pipeline%parameter_count()), &
            theta_bar(pipeline%parameter_count()), &
            theta_bar_reference(pipeline%parameter_count()), x_bar(4, 2), &
            x_bar_reference(4, 2), theta_hvp(pipeline%parameter_count()), &
            theta_hvp_reference(pipeline%parameter_count()), x_hvp(4, 2), &
            x_hvp_reference(4, 2))
        theta_dot = 0.17_dp
        call cpu%select(FORTML_DEVICE_CPU, status)
        call pipeline%transform(x, phi_reference, status)
        call pipeline%transform_device(cpu, x, phi, status)
        call check(status_ok(status) .and. maxval(abs(phi - phi_reference)) < 2.0e-14_dp, &
            "CPU transform dispatch matches host oracle", failures)
        call pipeline%jvp(x, theta_dot, x_dot, phi_reference, phi_dot_reference, status)
        call pipeline%jvp_device(cpu, x, theta_dot, x_dot, phi, phi_dot, status)
        call check(status_ok(status) .and. maxval(abs(phi_dot - phi_dot_reference)) < 2.0e-13_dp, &
            "CPU JVP dispatch matches host oracle", failures)
        call pipeline%vjp(x, u, theta_bar_reference, x_bar_reference, status)
        call pipeline%vjp_device(cpu, x, u, theta_bar, x_bar, status)
        call check(status_ok(status) .and. maxval(abs(theta_bar - theta_bar_reference)) < 2.0e-13_dp &
            .and. maxval(abs(x_bar - x_bar_reference)) < 2.0e-13_dp, &
            "CPU VJP dispatch matches host oracle", failures)
        call pipeline%hvp(x, u, theta_dot, x_dot, theta_hvp_reference, &
            x_hvp_reference, status)
        call pipeline%hvp_device(cpu, x, u, theta_dot, x_dot, theta_hvp, x_hvp, status)
        call check(status_ok(status) .and. maxval(abs(theta_hvp - theta_hvp_reference)) < 2.0e-12_dp &
            .and. maxval(abs(x_hvp - x_hvp_reference)) < 2.0e-12_dp, &
            "CPU HVP dispatch matches host oracle", failures)

        cuda%kind = FORTML_DEVICE_CUDA
        cuda%selected = .true.
        cuda%available = .true.
        phi = 1234.0_dp
        phi_dot = 2345.0_dp
        theta_bar = 3456.0_dp
        x_bar = 4567.0_dp
        theta_hvp = 5678.0_dp
        x_hvp = 6789.0_dp
        call pipeline%transform_device(cuda, x, phi, status)
        call check(status%code == FORTNUM_NOT_IMPLEMENTED .and. all(phi == 1234.0_dp), &
            "CUDA transform refusal preserves output", failures)
        call pipeline%jvp_device(cuda, x, theta_dot, x_dot, phi, phi_dot, status)
        call check(status%code == FORTNUM_NOT_IMPLEMENTED .and. all(phi == 1234.0_dp) &
            .and. all(phi_dot == 2345.0_dp), "CUDA JVP refusal preserves outputs", failures)
        call pipeline%vjp_device(cuda, x, u, theta_bar, x_bar, status)
        call check(status%code == FORTNUM_NOT_IMPLEMENTED .and. all(theta_bar == 3456.0_dp) &
            .and. all(x_bar == 4567.0_dp), "CUDA VJP refusal preserves outputs", failures)
        call pipeline%hvp_device(cuda, x, u, theta_dot, x_dot, theta_hvp, x_hvp, status)
        call check(status%code == FORTNUM_NOT_IMPLEMENTED .and. all(theta_hvp == 5678.0_dp) &
            .and. all(x_hvp == 6789.0_dp), "CUDA HVP refusal preserves outputs", failures)

        call pipeline%transform_device(unselected, x, phi, status)
        call check(status%code == FORTNUM_DOMAIN_ERROR, &
            "unselected device is rejected", failures)
    end subroutine test_device_dispatch

    subroutine test_column_refusals(failures)
        integer, intent(inout) :: failures
        type(basis_map_t) :: polynomial
        type(column_basis_pipeline_t) :: pipeline
        type(fortnum_status_t) :: status
        real(dp) :: x(2, 3)

        x = reshape([0.0_dp, 1.0_dp, 2.0_dp, 3.0_dp, 4.0_dp, 5.0_dp], &
            shape(x))
        polynomial = make_polynomial_basis(1, 2, status)
        pipeline = make_column_basis_pipeline(3, status)
        call pipeline%append(polynomial, [1, 1], status)
        if (status%code /= FORTNUM_DOMAIN_ERROR .or. pipeline%stage_count() /= 0) then
            write (error_unit, '(a)') &
                "FAIL [column pipeline] duplicate-column refusal"
            failures = failures + 1
        end if
        call pipeline%append(polynomial, [4], status)
        if (status%code /= FORTNUM_DOMAIN_ERROR .or. pipeline%stage_count() /= 0) then
            write (error_unit, '(a)') &
                "FAIL [column pipeline] out-of-range refusal"
            failures = failures + 1
        end if
        call pipeline%append(polynomial, [1], status)
        call pipeline%fit(x(:, 1:2), status)
        if (status%code /= FORTNUM_DOMAIN_ERROR) then
            write (error_unit, '(a)') &
                "FAIL [column pipeline] fit-shape refusal"
            failures = failures + 1
        end if
    end subroutine test_column_refusals

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failures

        if (.not. condition) then
            write (error_unit, '(a)') "FAIL [column pipeline device] "//description
            failures = failures + 1
        end if
    end subroutine check

end program test_column_pipeline
