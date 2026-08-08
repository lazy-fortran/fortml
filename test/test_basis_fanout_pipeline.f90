program test_basis_fanout_pipeline
    !! Independent behavioral oracle for the fan-out/fan-in basis DAG.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_basis, only: basis_map_t, make_polynomial_interaction_basis, &
        make_polynomial_basis, make_fourier_basis
    use fortml_pipeline, only: basis_fanout_pipeline_t, &
        sequential_basis_pipeline_t, make_basis_fanout_pipeline
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
            " fanout pipeline test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS basis fanout pipeline independent behavioral oracle"

contains

    subroutine make_fixture(fanout, branch_one, branch_two, direct_one, &
            direct_two, status)
        type(basis_fanout_pipeline_t), intent(out) :: fanout
        type(sequential_basis_pipeline_t), intent(out) :: branch_one, branch_two
        type(basis_map_t), intent(out) :: direct_one, direct_two
        type(fortnum_status_t), intent(out) :: status
        type(basis_map_t) :: stage
        real(dp) :: frequencies(1, 2)

        fanout = make_basis_fanout_pipeline(2, status)
        direct_one = make_polynomial_interaction_basis(2, 2, status, &
            include_intercept=.true.)
        call branch_one%initialize(2, status)
        call branch_one%append(direct_one, status, "quadratic")

        call branch_two%initialize(2, status)
        direct_two = make_polynomial_basis(2, 1, status)
        call branch_two%append(direct_two, status, "linear")
        frequencies = reshape([0.7_dp, 1.2_dp], shape(frequencies))
        call stage%initialize_fourier(2, frequencies, status)
        call branch_two%append(stage, status, "fourier")

        call fanout%append(branch_one, status, "quadratic_branch")
        call fanout%append(branch_two, status, "spectral_branch")
    end subroutine make_fixture

    subroutine check_value_and_metadata(failures)
        integer, intent(inout) :: failures
        real(dp) :: x(5, 2), expected_one(5, 6), linear(5, 2)
        real(dp) :: expected_two(5, 4), phi(5, 10)
        type(basis_fanout_pipeline_t) :: fanout
        type(sequential_basis_pipeline_t) :: branch_one, branch_two
        type(basis_map_t) :: direct_one, direct_two
        type(basis_map_t) :: fourier
        type(fortnum_status_t) :: status
        real(dp) :: frequencies(1, 2)

        x = reshape([0.2_dp, -0.4_dp, 0.7_dp, 0.1_dp, -0.5_dp, &
            0.8_dp, 1.1_dp, -0.3_dp, 0.6_dp, -0.9_dp], shape(x))
        call make_fixture(fanout, branch_one, branch_two, direct_one, &
            direct_two, status)
        call fanout%fit(x, status)
        call fanout%transform(x, phi, status)
        call direct_one%evaluate(x, expected_one, status)
        call direct_two%evaluate(x, linear, status)
        frequencies = reshape([0.7_dp, 1.2_dp], shape(frequencies))
        fourier = make_fourier_basis(2, frequencies, status)
        call fourier%evaluate(linear, expected_two, status)
        if (.not. status_ok(status) .or. fanout%branch_count() /= 2 .or. &
            fanout%feature_count() /= 10 .or. &
            fanout%parameter_count() /= fourier%parameter_count() .or. &
            maxval(abs(phi(:, 1:6) - expected_one)) > 1.0e-13_dp .or. &
            maxval(abs(phi(:, 7:10) - expected_two)) > 1.0e-13_dp) then
            write (error_unit, '(a)') "FAIL [fanout] value/shape oracle"
            failures = failures + 1
        end if
        if (fanout%branch_name(1) /= "quadratic_branch" .or. &
            fanout%branch_name(2) /= "spectral_branch" .or. &
            index(fanout%feature_name(7), "spectral_branch") /= 1 .or. &
            index(fanout%parameter_name(1), "spectral_branch") /= 1 .or. &
            fanout%branch_feature_offset(2) /= 7 .or. &
            fanout%branch_parameter_offset(2) /= 1) then
            write (error_unit, '(a)') "FAIL [fanout] metadata/offset oracle"
            failures = failures + 1
        end if
    end subroutine check_value_and_metadata

    subroutine check_derivative_products(failures)
        integer, intent(inout) :: failures
        integer, parameter :: n = 5
        real(dp) :: x(n, 2), x_dot(n, 2), u(n, 10), phi(n, 10), phi_dot(n, 10)
        real(dp), allocatable :: theta(:), theta_dot(:), theta_bar(:), theta_hvp(:)
        real(dp), allocatable :: x_bar(:, :), x_hvp(:, :), x_bar_plus(:, :), &
            x_bar_minus(:, :)
        real(dp), allocatable :: theta_bar_plus(:), theta_bar_minus(:)
        real(dp) :: h, lhs, rhs
        real(dp) :: error_theta, error_x
        type(basis_fanout_pipeline_t) :: fanout
        type(sequential_basis_pipeline_t) :: branch_one, branch_two
        type(basis_map_t) :: direct_one, direct_two
        type(fortnum_status_t) :: status
        integer :: i

        x = reshape([0.2_dp, -0.4_dp, 0.7_dp, 0.1_dp, -0.5_dp, &
            0.8_dp, 1.1_dp, -0.3_dp, 0.6_dp, -0.9_dp], shape(x))
        x_dot = reshape([-0.3_dp, 0.2_dp, 0.1_dp, -0.4_dp, 0.5_dp, &
            0.6_dp, -0.2_dp, 0.3_dp, 0.4_dp, -0.1_dp], shape(x_dot))
        call make_fixture(fanout, branch_one, branch_two, direct_one, &
            direct_two, status)
        call fanout%fit(x, status)
        allocate(theta(fanout%parameter_count()), theta_dot(fanout%parameter_count()))
        allocate(theta_bar(fanout%parameter_count()), theta_hvp(fanout%parameter_count()))
        allocate(theta_bar_plus(fanout%parameter_count()), &
            theta_bar_minus(fanout%parameter_count()))
        allocate(x_bar(n, 2), x_hvp(n, 2), x_bar_plus(n, 2), x_bar_minus(n, 2))
        theta = fanout%parameters()
        theta_dot = [(0.13_dp*real(i, dp) - 0.07_dp, i=1, size(theta_dot))]
        u = reshape([(0.03_dp*real(i, dp) - 0.17_dp, i=1, size(u))], shape(u))
        call fanout%jvp(x, theta_dot, x_dot, phi, phi_dot, status)
        call fanout%vjp(x, u, theta_bar, x_bar, status)
        lhs = sum(u*phi_dot)
        rhs = dot_product(theta_bar, theta_dot) + sum(x_bar*x_dot)
        if (.not. status_ok(status) .or. abs(lhs-rhs) > 2.0e-11_dp) then
            write (error_unit, '(a,es12.4)') "FAIL [fanout] JVP/VJP identity=", &
                abs(lhs-rhs)
            failures = failures + 1
        end if

        h = 2.0e-5_dp
        call fanout%hvp(x, u, theta_dot, x_dot, theta_hvp, x_hvp, status)
        call fanout%set_parameters(theta + h*theta_dot, status)
        call fanout%vjp(x + h*x_dot, u, theta_bar_plus, x_bar_plus, status)
        call fanout%set_parameters(theta - h*theta_dot, status)
        call fanout%vjp(x - h*x_dot, u, theta_bar_minus, x_bar_minus, status)
        call fanout%set_parameters(theta, status)
        error_theta = maxval(abs(theta_hvp - &
            (theta_bar_plus-theta_bar_minus)/(2.0_dp*h)))
        error_x = maxval(abs(x_hvp - (x_bar_plus-x_bar_minus)/(2.0_dp*h)))
        if (.not. status_ok(status) .or. error_theta > 2.0e-6_dp .or. &
            error_x > 2.0e-6_dp) then
            write (error_unit, '(a,2(es12.4,1x))') "FAIL [fanout] HVP errors=", &
                error_theta, error_x
            failures = failures + 1
        end if
    end subroutine check_derivative_products

    subroutine check_device_refusal(failures)
        integer, intent(inout) :: failures
        real(dp) :: x(2, 2), phi(2, 10)
        type(basis_fanout_pipeline_t) :: fanout
        type(sequential_basis_pipeline_t) :: branch_one, branch_two
        type(basis_map_t) :: direct_one, direct_two
        type(fortml_device_t) :: device
        type(fortnum_status_t) :: status

        x = reshape([0.2_dp, -0.4_dp, 0.7_dp, 0.1_dp], shape(x))
        call make_fixture(fanout, branch_one, branch_two, direct_one, &
            direct_two, status)
        call fanout%fit(x, status)
        device%kind = FORTML_DEVICE_CUDA
        device%selected = .true.
        device%available = .true.
        call fanout%transform_device(device, x, phi, status)
        if (status%code /= FORTNUM_NOT_IMPLEMENTED .or. &
            fanout%device_supported(FORTML_DEVICE_CUDA)) then
            write (error_unit, '(a)') "FAIL [fanout] CUDA refusal contract"
            failures = failures + 1
        end if
        device%kind = FORTML_DEVICE_CPU
        call fanout%transform_device(device, x, phi, status)
        if (status%code == FORTNUM_NOT_IMPLEMENTED) then
            write (error_unit, '(a)') "FAIL [fanout] CPU dispatch contract"
            failures = failures + 1
        end if
    end subroutine check_device_refusal

end program test_basis_fanout_pipeline
