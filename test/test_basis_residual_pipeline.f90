program test_basis_residual_pipeline
    !! Independent behavioral oracle for the named residual-sum DAG.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_basis, only: basis_map_t, make_polynomial_basis, &
        make_fourier_basis
    use fortml_pipeline, only: basis_residual_pipeline_t, &
        sequential_basis_pipeline_t, make_basis_residual_pipeline
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, &
        FORTML_DEVICE_CUDA
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    implicit none

    integer :: failures

    failures = 0
    call check_value_and_metadata(failures)
    call check_derivative_products(failures)
    call check_device_refusal(failures)
    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, &
            " residual pipeline test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS basis residual pipeline independent behavioral oracle"

contains

    subroutine make_fixture(residual, main_branch, residual_branch, main_map, &
            residual_map, status)
        type(basis_residual_pipeline_t), intent(out) :: residual
        type(sequential_basis_pipeline_t), intent(out) :: main_branch
        type(sequential_basis_pipeline_t), intent(out) :: residual_branch
        type(basis_map_t), intent(out) :: main_map, residual_map
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: frequencies(1, 2)

        residual = make_basis_residual_pipeline(2, status)
        main_map = make_polynomial_basis(2, 2, status)
        call main_branch%initialize(2, status)
        call main_branch%append(main_map, status, "quadratic")
        frequencies = reshape([0.7_dp, 1.2_dp], shape(frequencies))
        residual_map = make_fourier_basis(2, frequencies, status)
        call residual_branch%initialize(2, status)
        call residual_branch%append(residual_map, status, "spectral")
        call residual%set_main(main_branch, status, "identity_features")
        call residual%set_residual(residual_branch, status, "fourier_residual")
    end subroutine make_fixture

    subroutine check_value_and_metadata(failures)
        integer, intent(inout) :: failures
        real(dp) :: x(5, 2), expected_main(5, 4), expected_residual(5, 4)
        real(dp) :: phi(5, 4)
        type(basis_residual_pipeline_t) :: residual
        type(sequential_basis_pipeline_t) :: main_branch, residual_branch
        type(basis_map_t) :: main_map, residual_map
        type(fortnum_status_t) :: status

        x = reshape([0.2_dp, -0.4_dp, 0.7_dp, 0.1_dp, -0.5_dp, &
            0.8_dp, 1.1_dp, -0.3_dp, 0.6_dp, -0.9_dp], shape(x))
        call make_fixture(residual, main_branch, residual_branch, main_map, &
            residual_map, status)
        call residual%fit(x, status)
        call residual%transform(x, phi, status)
        call main_map%evaluate(x, expected_main, status)
        call residual_map%evaluate(x, expected_residual, status)
        if (.not. status_ok(status) .or. .not. residual%valid() .or. &
            .not. residual%is_fitted() .or. residual%feature_count() /= 4 .or. &
            residual%parameter_count() /= 2 .or. &
            maxval(abs(phi - expected_main - expected_residual)) > 1.0e-13_dp) then
            write (error_unit, '(a)') "FAIL [residual] value/shape oracle"
            failures = failures + 1
        end if
        if (residual%branch_name(1) /= "identity_features" .or. &
            residual%branch_name(2) /= "fourier_residual" .or. &
            residual%feature_name(1) /= "residual_sum.feature_1" .or. &
            index(residual%parameter_name(1), "fourier_residual") /= 1 .or. &
            residual%main_feature_offset() /= 1 .or. &
            residual%residual_feature_offset() /= 1 .or. &
            residual%residual_parameter_offset() /= 1) then
            write (error_unit, '(a)') "FAIL [residual] metadata/offset oracle"
            failures = failures + 1
        end if
    end subroutine check_value_and_metadata

    subroutine check_derivative_products(failures)
        integer, intent(inout) :: failures
        integer, parameter :: n = 5
        real(dp) :: x(n, 2), x_dot(n, 2), u(n, 4), phi(n, 4), phi_dot(n, 4)
        real(dp), allocatable :: theta(:), theta_dot(:), theta_bar(:), theta_hvp(:)
        real(dp), allocatable :: x_bar(:, :), x_hvp(:, :), x_bar_plus(:, :), &
            x_bar_minus(:, :), theta_bar_plus(:), theta_bar_minus(:)
        real(dp) :: h, lhs, rhs, error_theta, error_x
        type(basis_residual_pipeline_t) :: residual
        type(sequential_basis_pipeline_t) :: main_branch, residual_branch
        type(basis_map_t) :: main_map, residual_map
        type(fortnum_status_t) :: status
        integer :: i

        x = reshape([0.2_dp, -0.4_dp, 0.7_dp, 0.1_dp, -0.5_dp, &
            0.8_dp, 1.1_dp, -0.3_dp, 0.6_dp, -0.9_dp], shape(x))
        x_dot = reshape([-0.3_dp, 0.2_dp, 0.1_dp, -0.4_dp, 0.5_dp, &
            0.6_dp, -0.2_dp, 0.3_dp, 0.4_dp, -0.1_dp], shape(x_dot))
        call make_fixture(residual, main_branch, residual_branch, main_map, &
            residual_map, status)
        call residual%fit(x, status)
        allocate(theta(residual%parameter_count()), &
            theta_dot(residual%parameter_count()), theta_bar(residual%parameter_count()), &
            theta_hvp(residual%parameter_count()), &
            theta_bar_plus(residual%parameter_count()), &
            theta_bar_minus(residual%parameter_count()))
        allocate(x_bar(n, 2), x_hvp(n, 2), x_bar_plus(n, 2), x_bar_minus(n, 2))
        theta = residual%parameters()
        theta_dot = [0.11_dp, -0.08_dp]
        u = reshape([(0.03_dp*real(i, dp) - 0.17_dp, i=1, size(u))], shape(u))
        call residual%jvp(x, theta_dot, x_dot, phi, phi_dot, status)
        call residual%vjp(x, u, theta_bar, x_bar, status)
        lhs = sum(u*phi_dot)
        rhs = dot_product(theta_bar, theta_dot) + sum(x_bar*x_dot)
        if (.not. status_ok(status) .or. abs(lhs-rhs) > 2.0e-11_dp) then
            write (error_unit, '(a,es12.4)') "FAIL [residual] JVP/VJP identity=", &
                abs(lhs-rhs)
            failures = failures + 1
        end if

        h = 2.0e-5_dp
        call residual%hvp(x, u, theta_dot, x_dot, theta_hvp, x_hvp, status)
        call residual%set_parameters(theta + h*theta_dot, status)
        call residual%vjp(x + h*x_dot, u, theta_bar_plus, x_bar_plus, status)
        call residual%set_parameters(theta - h*theta_dot, status)
        call residual%vjp(x - h*x_dot, u, theta_bar_minus, x_bar_minus, status)
        call residual%set_parameters(theta, status)
        error_theta = maxval(abs(theta_hvp - &
            (theta_bar_plus-theta_bar_minus)/(2.0_dp*h)))
        error_x = maxval(abs(x_hvp - (x_bar_plus-x_bar_minus)/(2.0_dp*h)))
        if (.not. status_ok(status) .or. error_theta > 2.0e-6_dp .or. &
            error_x > 2.0e-6_dp) then
            write (error_unit, '(a,2(es12.4,1x))') "FAIL [residual] HVP errors=", &
                error_theta, error_x
            failures = failures + 1
        end if
    end subroutine check_derivative_products

    subroutine check_device_refusal(failures)
        integer, intent(inout) :: failures
        real(dp) :: x(2, 2), x_dot(2, 2), u(2, 4)
        real(dp) :: phi(2, 4), phi_dot(2, 4), x_bar(2, 2), x_hvp(2, 2)
        real(dp) :: theta_dot(2), theta_bar(2), theta_hvp(2)
        type(basis_residual_pipeline_t) :: residual
        type(sequential_basis_pipeline_t) :: main_branch, residual_branch
        type(basis_map_t) :: main_map, residual_map
        type(fortml_device_t) :: device
        type(fortnum_status_t) :: status

        x = reshape([0.2_dp, -0.4_dp, 0.7_dp, 0.1_dp], shape(x))
        x_dot = 0.2_dp
        u = 0.3_dp
        call make_fixture(residual, main_branch, residual_branch, main_map, &
            residual_map, status)
        call residual%fit(x, status)
        device%kind = FORTML_DEVICE_CUDA
        device%selected = .true.
        device%available = .true.
        phi = -11.0_dp
        call residual%transform_device(device, x, phi, status)
        if (status%code /= FORTNUM_NOT_IMPLEMENTED .or. maxval(abs(phi+11.0_dp)) > 0.0_dp) &
            failures = failures + 1
        phi = -12.0_dp
        phi_dot = -13.0_dp
        theta_dot = 0.1_dp
        call residual%jvp_device(device, x, theta_dot, x_dot, phi, phi_dot, status)
        if (status%code /= FORTNUM_NOT_IMPLEMENTED .or. maxval(abs(phi+12.0_dp)) > 0.0_dp .or. &
            maxval(abs(phi_dot+13.0_dp)) > 0.0_dp) failures = failures + 1
        theta_bar = -14.0_dp
        x_bar = -15.0_dp
        call residual%vjp_device(device, x, u, theta_bar, x_bar, status)
        if (status%code /= FORTNUM_NOT_IMPLEMENTED .or. maxval(abs(theta_bar+14.0_dp)) > 0.0_dp .or. &
            maxval(abs(x_bar+15.0_dp)) > 0.0_dp) failures = failures + 1
        theta_hvp = -16.0_dp
        x_hvp = -17.0_dp
        call residual%hvp_device(device, x, u, theta_dot, x_dot, theta_hvp, x_hvp, status)
        if (status%code /= FORTNUM_NOT_IMPLEMENTED .or. maxval(abs(theta_hvp+16.0_dp)) > 0.0_dp .or. &
            maxval(abs(x_hvp+17.0_dp)) > 0.0_dp) failures = failures + 1
        device%kind = FORTML_DEVICE_CPU
        call residual%transform_device(device, x, phi, status)
        if (.not. status_ok(status)) failures = failures + 1
    end subroutine check_device_refusal

end program test_basis_residual_pipeline
