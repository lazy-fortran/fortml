program test_conditional_pipeline
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_basis, only: basis_map_t, make_fourier_basis
    use fortml_conditional_pipeline, only: conditional_basis_pipeline_t, &
        make_conditional_basis_pipeline
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_DOMAIN_ERROR, &
        FORTNUM_NOT_IMPLEMENTED
    implicit none

    integer :: failures

    failures = 0
    call test_products(failures)
    call test_metadata_and_transactionality(failures)
    call test_device_and_endpoint_refusals(failures)
    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, &
            " conditional pipeline test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS"

contains

    subroutine configure(pipeline, status)
        type(conditional_basis_pipeline_t), intent(out) :: pipeline
        type(fortnum_status_t), intent(out) :: status
        type(basis_map_t) :: left, right

        left = make_fourier_basis(1, reshape([0.8_dp], [1, 1]), status)
        if (.not. status_ok(status)) return
        right = make_fourier_basis(1, reshape([1.3_dp], [1, 1]), status)
        if (.not. status_ok(status)) return
        pipeline = make_conditional_basis_pipeline(2, status)
        if (.not. status_ok(status)) return
        call pipeline%append(left, [2], 1, -2.0_dp, 0.0_dp, status, name="left")
        if (.not. status_ok(status)) return
        call pipeline%append(right, [2], 1, 0.0_dp, 2.0_dp, status, name="right")
    end subroutine configure

    subroutine test_products(failures)
        integer, intent(inout) :: failures
        type(conditional_basis_pipeline_t) :: pipeline
        type(fortnum_status_t) :: status
        real(dp) :: x(6, 2), x_dot(6, 2), u(6, 4), x_bar(6, 2)
        real(dp), allocatable :: theta(:), theta_dot(:), theta_plus(:), &
            theta_minus(:), theta_bar(:), theta_bar_plus(:), theta_bar_minus(:)
        real(dp), allocatable :: phi(:, :), phi_dot(:, :), phi_plus(:, :), &
            phi_minus(:, :), theta_hvp(:), x_hvp(:, :), x_bar_plus(:, :), &
            x_bar_minus(:, :)
        real(dp) :: lhs, rhs, h
        integer :: i, n_parameters

        x(:, 1) = [-1.3_dp, -0.7_dp, -0.2_dp, 0.3_dp, 0.9_dp, 1.6_dp]
        x(:, 2) = [0.4_dp, -0.8_dp, 1.1_dp, -0.5_dp, 0.7_dp, -1.2_dp]
        x_dot(:, 1) = 0.0_dp
        x_dot(:, 2) = [0.2_dp, -0.3_dp, 0.4_dp, -0.1_dp, 0.5_dp, -0.6_dp]
        call configure(pipeline, status)
        call pipeline%fit(x, status)
        if (.not. status_ok(status) .or. pipeline%branch_count() /= 2 .or. &
                pipeline%feature_count() /= 4 .or. pipeline%parameter_count() /= 2) then
            write (error_unit, '(a)') "FAIL [conditional] configuration"
            failures = failures + 1
            return
        end if

        n_parameters = pipeline%parameter_count()
        allocate(theta(n_parameters), theta_dot(n_parameters), &
            theta_plus(n_parameters), theta_minus(n_parameters), &
            theta_bar(n_parameters), theta_bar_plus(n_parameters), &
            theta_bar_minus(n_parameters), phi(6, 4), phi_dot(6, 4), &
            phi_plus(6, 4), phi_minus(6, 4), theta_hvp(n_parameters), &
            x_hvp(6, 2), x_bar_plus(6, 2), x_bar_minus(6, 2))
        theta = pipeline%parameters()
        theta_dot = [0.17_dp, -0.23_dp]
        call pipeline%transform(x, phi, status)
        if (.not. status_ok(status) .or. maxval(abs(phi(1, 3:4))) > 1.0e-14_dp .or. &
                maxval(abs(phi(2, 3:4))) > 1.0e-14_dp .or. &
                maxval(abs(phi(4, 1:2))) > 1.0e-14_dp) then
            write (error_unit, '(a)') "FAIL [conditional] interval routing oracle"
            failures = failures + 1
        end if

        h = 1.0e-6_dp
        theta_plus = theta + h*theta_dot
        theta_minus = theta - h*theta_dot
        call pipeline%set_parameters(theta_plus, status)
        call pipeline%transform(x + h*x_dot, phi_plus, status)
        call pipeline%set_parameters(theta_minus, status)
        call pipeline%transform(x - h*x_dot, phi_minus, status)
        call pipeline%set_parameters(theta, status)
        call pipeline%jvp(x, theta_dot, x_dot, phi, phi_dot, status)
        if (.not. status_ok(status) .or. maxval(abs(phi_dot - &
                (phi_plus - phi_minus)/(2.0_dp*h))) > 7.0e-8_dp) then
            write (error_unit, '(a,es12.4)') "FAIL [conditional] JVP FD=", &
                maxval(abs(phi_dot - (phi_plus - phi_minus)/(2.0_dp*h)))
            failures = failures + 1
        end if

        do i = 1, 4
            u(:, i) = [0.13_dp, -0.22_dp, 0.31_dp, -0.17_dp, 0.29_dp, -0.11_dp] + &
                0.04_dp*real(i, dp)
        end do
        call pipeline%vjp(x, u, theta_bar, x_bar, status)
        lhs = sum(u*phi_dot)
        rhs = sum(theta_bar*theta_dot) + sum(x_bar*x_dot)
        if (.not. status_ok(status) .or. abs(lhs - rhs) > 2.0e-9_dp) then
            write (error_unit, '(a,es12.4)') "FAIL [conditional] VJP identity=", &
                abs(lhs-rhs)
            failures = failures + 1
        end if

        ! Independent HVP oracle: finite-difference the VJP along (theta_dot,x_dot).
        call pipeline%set_parameters(theta_plus, status)
        call pipeline%vjp(x + h*x_dot, u, theta_bar_plus, x_bar_plus, status)
        call pipeline%set_parameters(theta_minus, status)
        call pipeline%vjp(x - h*x_dot, u, theta_bar_minus, x_bar_minus, status)
        call pipeline%set_parameters(theta, status)
        call pipeline%hvp(x, u, theta_dot, x_dot, theta_hvp, x_hvp, status)
        if (.not. status_ok(status) .or. &
                maxval(abs(theta_hvp - (theta_bar_plus - theta_bar_minus)/(2.0_dp*h))) > &
                2.0e-6_dp .or. maxval(abs(x_hvp - (x_bar_plus - x_bar_minus)/(2.0_dp*h))) > &
                2.0e-6_dp) then
            write (error_unit, '(a,2es12.4)') "FAIL [conditional] HVP FD=", &
                maxval(abs(theta_hvp - (theta_bar_plus - theta_bar_minus)/(2.0_dp*h))), &
                maxval(abs(x_hvp - (x_bar_plus - x_bar_minus)/(2.0_dp*h)))
            failures = failures + 1
        end if
    end subroutine test_products

    subroutine test_metadata_and_transactionality(failures)
        integer, intent(inout) :: failures
        type(conditional_basis_pipeline_t) :: pipeline
        type(fortnum_status_t) :: status
        type(basis_map_t) :: map
        real(dp) :: x(4, 2), theta_before(2), theta_bad(2)
        character(len=16) :: names(2)

        x = reshape([-1.0_dp, -0.5_dp, 0.5_dp, 1.0_dp, &
            0.2_dp, 0.4_dp, -0.6_dp, -0.8_dp], shape(x))
        call configure(pipeline, status)
        theta_before = pipeline%parameters()
        map = make_fourier_basis(1, reshape([0.6_dp], [1, 1]), status)
        call pipeline%append(map, [2], 1, 1.0_dp, 0.0_dp, status, name="bad")
        if (status%code /= FORTNUM_DOMAIN_ERROR .or. pipeline%branch_count() /= 2) then
            write (error_unit, '(a)') "FAIL [conditional] transactional append"
            failures = failures + 1
        end if
        theta_bad = [ieee_nan(), theta_before(2)]
        call pipeline%set_parameters(theta_bad, status)
        if (status%code /= FORTNUM_DOMAIN_ERROR .or. &
                maxval(abs(pipeline%parameters() - theta_before)) > 1.0e-14_dp) then
            write (error_unit, '(a)') "FAIL [conditional] transactional parameters"
            failures = failures + 1
        end if
        names = [character(len=16) :: "time", "signal"]
        call pipeline%set_input_schema(names, status)
        if (.not. status_ok(status) .or. pipeline%input_schema_name(1) /= "time" .or. &
                pipeline%input_schema_name(2) /= "signal") then
            write (error_unit, '(a)') "FAIL [conditional] schema metadata"
            failures = failures + 1
        end if
        names(2) = "time"
        call pipeline%set_input_schema(names, status)
        if (status%code /= FORTNUM_DOMAIN_ERROR .or. &
                pipeline%input_schema_name(2) /= "signal") then
            write (error_unit, '(a)') "FAIL [conditional] transactional schema"
            failures = failures + 1
        end if
        if (pipeline%branch_name(1) /= "left" .or. pipeline%branch_name(2) /= "right" .or. &
                pipeline%branch_feature_offset(1) /= 1 .or. &
                pipeline%branch_feature_offset(2) /= 3 .or. &
                pipeline%branch_parameter_offset(2) /= 2 .or. &
                pipeline%branch_route_column(2) /= 1) then
            write (error_unit, '(a)') "FAIL [conditional] stable offsets"
            failures = failures + 1
        end if
    end subroutine test_metadata_and_transactionality

    subroutine test_device_and_endpoint_refusals(failures)
        integer, intent(inout) :: failures
        type(conditional_basis_pipeline_t) :: pipeline
        type(fortml_device_t) :: cpu, cuda
        type(fortnum_status_t) :: status
        real(dp) :: x(4, 2), x_dot(4, 2), phi(4, 4), phi_dot(4, 4)
        real(dp) :: u(4, 4), theta_dot(2), theta_bar(2), x_bar(4, 2)
        real(dp) :: theta_hvp(2), x_hvp(4, 2)

        x = reshape([-1.0_dp, -0.4_dp, 0.4_dp, 1.0_dp, &
            0.2_dp, -0.3_dp, 0.5_dp, -0.7_dp], shape(x))
        x_dot = 0.0_dp
        u = 0.2_dp
        theta_dot = 0.1_dp
        call configure(pipeline, status)
        call pipeline%fit(x, status)
        call cpu%select(FORTML_DEVICE_CPU, status)
        call pipeline%transform_device(cpu, x, phi, status)
        if (.not. status_ok(status)) then
            write (error_unit, '(a)') "FAIL [conditional] CPU dispatch"
            failures = failures + 1
        end if
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
        if (status%code /= FORTNUM_NOT_IMPLEMENTED .or. any(phi /= 1234.0_dp)) &
            failures = failures + 1
        call pipeline%jvp_device(cuda, x, theta_dot, x_dot, phi, phi_dot, status)
        if (status%code /= FORTNUM_NOT_IMPLEMENTED .or. any(phi /= 1234.0_dp) .or. &
                any(phi_dot /= 2345.0_dp)) failures = failures + 1
        call pipeline%vjp_device(cuda, x, u, theta_bar, x_bar, status)
        if (status%code /= FORTNUM_NOT_IMPLEMENTED .or. any(theta_bar /= 3456.0_dp) .or. &
                any(x_bar /= 4567.0_dp)) failures = failures + 1
        call pipeline%hvp_device(cuda, x, u, theta_dot, x_dot, theta_hvp, x_hvp, status)
        if (status%code /= FORTNUM_NOT_IMPLEMENTED .or. any(theta_hvp /= 5678.0_dp) .or. &
                any(x_hvp /= 6789.0_dp)) failures = failures + 1

        x(1, 1) = 0.0_dp
        call pipeline%jvp(x, theta_dot, x_dot, phi, phi_dot, status)
        if (status%code /= FORTNUM_DOMAIN_ERROR) then
            write (error_unit, '(a)') "FAIL [conditional] endpoint derivative refusal"
            failures = failures + 1
        end if
    end subroutine test_device_and_endpoint_refusals

    pure real(dp) function ieee_nan()
        use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_quiet_nan
        ieee_nan = ieee_value(0.0_dp, ieee_quiet_nan)
    end function ieee_nan

end program test_conditional_pipeline
