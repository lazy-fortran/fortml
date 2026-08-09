program test_basis_blend_pipeline
    !! Independent analytic and finite-difference oracle for learned basis fan-in.
    use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_quiet_nan
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_basis, only: basis_map_t, make_polynomial_basis, make_fourier_basis
    use fortml_pipeline, only: sequential_basis_pipeline_t
    use fortml_basis_blend_pipeline, only: basis_blend_pipeline_t, &
        make_basis_blend_pipeline
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, &
        FORTML_DEVICE_CUDA
    use fortml_estimator_capabilities, only: estimator_capability_t, &
        FORTML_CAPABILITY_DEVICE_CPU, FORTML_CAPABILITY_DEVICE_CUDA, &
        FORTML_CAPABILITY_DEVICE_OPENACC, FORTML_DERIVATIVE_INPUT_HVP, &
        FORTML_DERIVATIVE_PARAMETER_HVP
    use fortnum_status, only: fortnum_status_t, status_ok, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED
    implicit none

    integer :: failures

    failures = 0
    call check_value_metadata_and_transaction(failures)
    call check_derivative_products(failures)
    call check_capabilities_and_devices(failures)
    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, &
            " basis blend pipeline test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS basis blend pipeline independent behavioral oracle"

contains

    subroutine make_fixture(blend, polynomial_branch, fourier_branch, &
            polynomial_map, fourier_map, status)
        type(basis_blend_pipeline_t), intent(out) :: blend
        type(sequential_basis_pipeline_t), intent(out) :: polynomial_branch
        type(sequential_basis_pipeline_t), intent(out) :: fourier_branch
        type(basis_map_t), intent(out) :: polynomial_map, fourier_map
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: frequencies(1, 2)

        blend = make_basis_blend_pipeline(2, status)
        polynomial_map = make_polynomial_basis(2, 2, status)
        call polynomial_branch%initialize(2, status)
        call polynomial_branch%append(polynomial_map, status, "quadratic")
        frequencies = reshape([0.7_dp, 1.1_dp], shape(frequencies))
        fourier_map = make_fourier_basis(2, frequencies, status)
        call fourier_branch%initialize(2, status)
        call fourier_branch%append(fourier_map, status, "spectral")
        call blend%append(polynomial_branch, 1.25_dp, status, "trend")
        call blend%append(fourier_branch, -0.4_dp, status, "oscillation")
    end subroutine make_fixture

    subroutine check_value_metadata_and_transaction(failures)
        integer, intent(inout) :: failures
        real(dp) :: x(5, 2), y(5, 4), expected(5, 4)
        real(dp) :: polynomial_y(5, 4), fourier_y(5, 4), bad_theta(4)
        real(dp), allocatable :: before(:), after(:)
        type(basis_blend_pipeline_t) :: blend
        type(sequential_basis_pipeline_t) :: polynomial_branch, fourier_branch
        type(basis_map_t) :: polynomial_map, fourier_map, bad_map
        type(fortnum_status_t) :: status
        integer :: old_count

        x = reshape([0.2_dp, -0.4_dp, 0.7_dp, 0.1_dp, -0.5_dp, &
            0.8_dp, 1.1_dp, -0.3_dp, 0.6_dp, -0.9_dp], shape(x))
        call make_fixture(blend, polynomial_branch, fourier_branch, &
            polynomial_map, fourier_map, status)
        call blend%set_input_schema([character(len=8) :: "position", &
            "velocity"], status)
        call blend%fit(x, status)
        call blend%transform(x, y, status)
        call polynomial_map%evaluate(x, polynomial_y, status)
        call fourier_map%evaluate(x, fourier_y, status)
        expected = 1.25_dp*polynomial_y - 0.4_dp*fourier_y
        if (.not. status_ok(status) .or. .not. blend%valid() .or. &
            .not. blend%is_fitted() .or. blend%branch_count() /= 2 .or. &
            blend%feature_count() /= 4 .or. &
            maxval(abs(y - expected)) > 2.0e-13_dp) then
            write (error_unit, '(a)') "FAIL [blend] analytic value oracle"
            failures = failures + 1
        end if
        if (blend%branch_name(1) /= "trend" .or. &
            blend%branch_name(2) /= "oscillation" .or. &
            blend%feature_name(3) /= "blend.feature_3" .or. &
            blend%parameter_name(1) /= "trend.weight" .or. &
            blend%parameter_name(2) /= "oscillation.weight" .or. &
            blend%branch_parameter_offset(2) /= 2 .or. &
            blend%input_schema_name(1) /= "position") then
            write (error_unit, '(a)') "FAIL [blend] metadata oracle"
            failures = failures + 1
        end if

        before = blend%parameters()
        bad_theta = before
        bad_theta(1) = ieee_value(0.0_dp, ieee_quiet_nan)
        call blend%set_parameters(bad_theta, status)
        after = blend%parameters()
        if (status%code /= FORTNUM_DOMAIN_ERROR .or. &
            maxval(abs(after - before)) > 0.0_dp) then
            write (error_unit, '(a)') "FAIL [blend] parameter rollback oracle"
            failures = failures + 1
        end if
        old_count = blend%branch_count()
        call blend%append(fourier_branch, 0.2_dp, status, "trend")
        if (status%code /= FORTNUM_DOMAIN_ERROR .or. &
            blend%branch_count() /= old_count .or. &
            maxval(abs(blend%parameters() - before)) > 0.0_dp) then
            write (error_unit, '(a)') "FAIL [blend] append rollback oracle"
            failures = failures + 1
        end if
        bad_map = make_polynomial_basis(2, 1, status)
        call polynomial_branch%initialize(2, status)
        call polynomial_branch%append(bad_map, status, "wrong_shape")
        call blend%append(polynomial_branch, 0.2_dp, status, "linear")
        if (status%code /= FORTNUM_DOMAIN_ERROR .or. &
            blend%branch_count() /= old_count) then
            write (error_unit, '(a)') "FAIL [blend] shape rollback oracle"
            failures = failures + 1
        end if
    end subroutine check_value_metadata_and_transaction

    subroutine check_derivative_products(failures)
        integer, intent(inout) :: failures
        real(dp) :: x(5, 2), x_dot(5, 2), u(5, 4)
        real(dp) :: y(5, 4), y_dot(5, 4), polynomial_y(5, 4)
        real(dp), allocatable :: theta(:), theta_dot(:), theta_bar(:), theta_hvp(:)
        real(dp), allocatable :: theta_bar_plus(:), theta_bar_minus(:)
        real(dp), allocatable :: x_bar(:, :), x_hvp(:, :)
        real(dp), allocatable :: x_bar_plus(:, :), x_bar_minus(:, :)
        type(basis_blend_pipeline_t) :: blend
        type(sequential_basis_pipeline_t) :: polynomial_branch, fourier_branch
        type(basis_map_t) :: polynomial_map, fourier_map
        type(fortnum_status_t) :: status
        real(dp) :: lhs, rhs, h, error_theta, error_x
        integer :: i

        x = reshape([0.2_dp, -0.4_dp, 0.7_dp, 0.1_dp, -0.5_dp, &
            0.8_dp, 1.1_dp, -0.3_dp, 0.6_dp, -0.9_dp], shape(x))
        x_dot = reshape([-0.3_dp, 0.2_dp, 0.1_dp, -0.4_dp, 0.5_dp, &
            0.6_dp, -0.2_dp, 0.3_dp, 0.4_dp, -0.1_dp], shape(x_dot))
        u = reshape([(0.037_dp*real(i, dp) - 0.23_dp, i=1, size(u))], shape(u))
        call make_fixture(blend, polynomial_branch, fourier_branch, &
            polynomial_map, fourier_map, status)
        call blend%fit(x, status)
        theta = blend%parameters()
        allocate(theta_dot(size(theta)), theta_bar(size(theta)), &
            theta_hvp(size(theta)), theta_bar_plus(size(theta)), &
            theta_bar_minus(size(theta)))
        allocate(x_bar(5, 2), x_hvp(5, 2), x_bar_plus(5, 2), x_bar_minus(5, 2))
        theta_dot = [(0.09_dp*real(i, dp) - 0.14_dp, i=1, size(theta_dot))]
        call blend%jvp(x, theta_dot, x_dot, y, y_dot, status)
        call blend%vjp(x, u, theta_bar, x_bar, status)
        lhs = sum(u*y_dot)
        rhs = dot_product(theta_bar, theta_dot) + sum(x_bar*x_dot)
        call polynomial_map%evaluate(x, polynomial_y, status)
        if (.not. status_ok(status) .or. abs(lhs - rhs) > 3.0e-11_dp .or. &
            abs(theta_bar(1) - sum(u*polynomial_y)) > 2.0e-13_dp) then
            write (error_unit, '(a,2(es12.4,1x))') &
                "FAIL [blend] JVP/VJP oracle errors=", abs(lhs-rhs), &
                abs(theta_bar(1) - sum(u*polynomial_y))
            failures = failures + 1
        end if

        h = 2.0e-5_dp
        call blend%hvp(x, u, theta_dot, x_dot, theta_hvp, x_hvp, status)
        call blend%set_parameters(theta + h*theta_dot, status)
        call blend%vjp(x + h*x_dot, u, theta_bar_plus, x_bar_plus, status)
        call blend%set_parameters(theta - h*theta_dot, status)
        call blend%vjp(x - h*x_dot, u, theta_bar_minus, x_bar_minus, status)
        call blend%set_parameters(theta, status)
        error_theta = maxval(abs(theta_hvp - &
            (theta_bar_plus - theta_bar_minus)/(2.0_dp*h)))
        error_x = maxval(abs(x_hvp - (x_bar_plus - x_bar_minus)/(2.0_dp*h)))
        if (.not. status_ok(status) .or. error_theta > 3.0e-6_dp .or. &
            error_x > 3.0e-6_dp) then
            write (error_unit, '(a,2(es12.4,1x))') &
                "FAIL [blend] HVP finite-difference errors=", error_theta, error_x
            failures = failures + 1
        end if
    end subroutine check_derivative_products

    subroutine check_capabilities_and_devices(failures)
        integer, intent(inout) :: failures
        real(dp) :: x(2, 2), y(2, 4), x_dot(2, 2), y_dot(2, 4)
        real(dp), allocatable :: theta_dot(:), theta_bar(:), theta_hvp(:)
        real(dp) :: x_bar(2, 2), x_hvp(2, 2), u(2, 4)
        type(basis_blend_pipeline_t) :: blend
        type(sequential_basis_pipeline_t) :: polynomial_branch, fourier_branch
        type(basis_map_t) :: polynomial_map, fourier_map
        type(fortml_device_t) :: device
        type(estimator_capability_t) :: report
        type(fortnum_status_t) :: status

        x = reshape([0.2_dp, -0.4_dp, 0.7_dp, 0.1_dp], shape(x))
        x_dot = 0.2_dp
        u = 0.3_dp
        call make_fixture(blend, polynomial_branch, fourier_branch, &
            polynomial_map, fourier_map, status)
        call blend%fit(x, status)
        allocate(theta_dot(blend%parameter_count()), &
            theta_bar(blend%parameter_count()), theta_hvp(blend%parameter_count()))
        theta_dot = 0.1_dp
        call blend%capabilities(report, status)
        if (.not. status_ok(status) .or. &
            .not. report%supports_device(FORTML_CAPABILITY_DEVICE_CPU) .or. &
            report%supports_device(FORTML_CAPABILITY_DEVICE_CUDA) .or. &
            report%supports_device(FORTML_CAPABILITY_DEVICE_OPENACC) .or. &
            .not. report%supports_derivative(FORTML_DERIVATIVE_INPUT_HVP) .or. &
            .not. report%supports_derivative( &
            FORTML_DERIVATIVE_PARAMETER_HVP)) then
            write (error_unit, '(a)') "FAIL [blend] capability matrix oracle"
            failures = failures + 1
        end if

        device%kind = FORTML_DEVICE_CPU
        device%selected = .true.
        device%available = .true.
        call blend%transform_device(device, x, y, status)
        if (.not. status_ok(status)) failures = failures + 1
        device%kind = FORTML_DEVICE_CUDA
        y = -11.0_dp
        call blend%transform_device(device, x, y, status)
        if (status%code /= FORTNUM_NOT_IMPLEMENTED .or. &
            maxval(abs(y + 11.0_dp)) > 0.0_dp) failures = failures + 1
        y = -12.0_dp
        y_dot = -13.0_dp
        call blend%jvp_device(device, x, theta_dot, x_dot, y, y_dot, status)
        if (status%code /= FORTNUM_NOT_IMPLEMENTED .or. &
            maxval(abs(y + 12.0_dp)) > 0.0_dp .or. &
            maxval(abs(y_dot + 13.0_dp)) > 0.0_dp) failures = failures + 1
        theta_bar = -14.0_dp
        x_bar = -15.0_dp
        call blend%vjp_device(device, x, u, theta_bar, x_bar, status)
        if (status%code /= FORTNUM_NOT_IMPLEMENTED .or. &
            maxval(abs(theta_bar + 14.0_dp)) > 0.0_dp .or. &
            maxval(abs(x_bar + 15.0_dp)) > 0.0_dp) failures = failures + 1
        device%kind = FORTML_CAPABILITY_DEVICE_OPENACC
        theta_hvp = -16.0_dp
        x_hvp = -17.0_dp
        call blend%hvp_device(device, x, u, theta_dot, x_dot, theta_hvp, &
            x_hvp, status)
        if (status%code /= FORTNUM_NOT_IMPLEMENTED .or. &
            maxval(abs(theta_hvp + 16.0_dp)) > 0.0_dp .or. &
            maxval(abs(x_hvp + 17.0_dp)) > 0.0_dp) then
            write (error_unit, '(a)') "FAIL [blend] CUDA/OpenACC refusal oracle"
            failures = failures + 1
        end if
    end subroutine check_capabilities_and_devices

end program test_basis_blend_pipeline
