program test_basis_sequential_device
    !! Independent oracle for sequential basis device dispatch.
    !!
    !! The CPU device path is checked against a hand-written polynomial/Fourier
    !! composition and its analytic input/parameter products.  CUDA requests
    !! must refuse without modifying caller-owned output buffers.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_basis, only: basis_map_t, make_fourier_basis, &
        make_polynomial_basis
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, &
        FORTML_DEVICE_CUDA
    use fortml_pipeline, only: sequential_basis_pipeline_t, &
        make_sequential_basis_pipeline
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    implicit none

    integer :: failures

    failures = 0
    call check_cpu_products(failures)
    call check_cuda_refusal(failures)
    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, &
            " sequential device test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS sequential basis device independent behavioral oracle"

contains

    subroutine make_fixture(pipeline, status)
        type(sequential_basis_pipeline_t), intent(out) :: pipeline
        type(fortnum_status_t), intent(out) :: status
        type(basis_map_t) :: polynomial, fourier
        real(dp) :: frequencies(1, 2)

        pipeline = make_sequential_basis_pipeline(2, status)
        polynomial = make_polynomial_basis(2, 1, status)
        call pipeline%append(polynomial, status, "identity")
        frequencies = reshape([0.7_dp, 1.2_dp], shape(frequencies))
        fourier = make_fourier_basis(2, frequencies, status)
        call pipeline%append(fourier, status, "spectral")
    end subroutine make_fixture

    subroutine check_cpu_products(failures)
        integer, intent(inout) :: failures
        integer, parameter :: n = 5
        real(dp) :: x(n, 2), x_dot(n, 2), y(n, 4), y_dot(n, 4)
        real(dp) :: expected(n, 4), expected_dot(n, 4)
        real(dp) :: u(n, 4), theta_bar(2), x_bar(n, 2)
        real(dp) :: expected_theta_bar(2), expected_x_bar(n, 2)
        real(dp) :: theta_dot(2), theta(2), frequency(2), argument(n, 2)
        real(dp) :: z_bar(n, 2)
        type(sequential_basis_pipeline_t) :: pipeline
        type(fortml_device_t) :: device
        type(fortnum_status_t) :: status
        integer :: j

        x = reshape([0.2_dp, -0.4_dp, 0.7_dp, 0.1_dp, -0.5_dp, &
            0.8_dp, 1.1_dp, -0.3_dp, 0.6_dp, -0.9_dp], shape(x))
        x_dot = reshape([-0.3_dp, 0.2_dp, 0.1_dp, -0.4_dp, 0.5_dp, &
            0.6_dp, -0.2_dp, 0.3_dp, 0.4_dp, -0.1_dp], shape(x_dot))
        u = reshape([(0.03_dp*real(j, dp) - 0.17_dp, j=1, size(u))], shape(u))
        theta = [log(0.7_dp), log(1.2_dp)]
        theta_dot = [0.11_dp, -0.08_dp]
        frequency = exp(theta)
        do j = 1, 2
            argument(:, j) = frequency(j)*x(:, j)
            expected(:, 2*j - 1) = sin(argument(:, j))
            expected(:, 2*j) = cos(argument(:, j))
            expected_dot(:, 2*j - 1) = cos(argument(:, j))*frequency(j)* &
                (x_dot(:, j) + x(:, j)*theta_dot(j))
            expected_dot(:, 2*j) = -sin(argument(:, j))*frequency(j)* &
                (x_dot(:, j) + x(:, j)*theta_dot(j))
            z_bar(:, j) = u(:, 2*j - 1)*cos(argument(:, j)) - &
                u(:, 2*j)*sin(argument(:, j))
            expected_theta_bar(j) = sum(frequency(j)*x(:, j)*z_bar(:, j))
            expected_x_bar(:, j) = frequency(j)*z_bar(:, j)
        end do

        call make_fixture(pipeline, status)
        call pipeline%fit(x, status)
        device%kind = FORTML_DEVICE_CPU
        device%selected = .true.
        device%available = .true.
        y = -1.0_dp
        call pipeline%transform_device(device, x, y, status)
        if (.not. status_ok(status) .or. maxval(abs(y - expected)) > 2.0e-13_dp .or. &
            .not. pipeline%device_supported(FORTML_DEVICE_CPU)) then
            write (error_unit, '(a)') "FAIL [sequential] CPU value/device capability"
            failures = failures + 1
        end if

        y = -2.0_dp
        y_dot = -3.0_dp
        call pipeline%jvp_device(device, x, theta_dot, x_dot, y, y_dot, status)
        if (.not. status_ok(status) .or. maxval(abs(y - expected)) > 2.0e-13_dp .or. &
            maxval(abs(y_dot - expected_dot)) > 2.0e-12_dp) then
            write (error_unit, '(a)') "FAIL [sequential] CPU JVP oracle"
            failures = failures + 1
        end if

        theta_bar = -4.0_dp
        x_bar = -5.0_dp
        call pipeline%vjp_device(device, x, u, theta_bar, x_bar, status)
        if (.not. status_ok(status) .or. &
            maxval(abs(theta_bar - expected_theta_bar)) > 2.0e-12_dp .or. &
            maxval(abs(x_bar - expected_x_bar)) > 2.0e-12_dp) then
            write (error_unit, '(a)') "FAIL [sequential] CPU VJP oracle"
            failures = failures + 1
        end if
    end subroutine check_cpu_products

    subroutine check_cuda_refusal(failures)
        integer, intent(inout) :: failures
        real(dp) :: x(2, 2), x_dot(2, 2), y(2, 4), y_dot(2, 4)
        real(dp) :: u(2, 4), theta_dot(2), theta_bar(2), x_bar(2, 2)
        real(dp) :: theta_hvp(2), x_hvp(2, 2)
        type(sequential_basis_pipeline_t) :: pipeline
        type(fortml_device_t) :: device
        type(fortnum_status_t) :: status

        x = reshape([0.2_dp, -0.4_dp, 0.7_dp, 0.1_dp], shape(x))
        x_dot = 0.2_dp
        u = 0.3_dp
        theta_dot = [0.1_dp, -0.2_dp]
        call make_fixture(pipeline, status)
        call pipeline%fit(x, status)
        device%kind = FORTML_DEVICE_CUDA
        device%selected = .true.
        device%available = .true.

        y = -11.0_dp
        call pipeline%transform_device(device, x, y, status)
        if (status%code /= FORTNUM_NOT_IMPLEMENTED .or. maxval(abs(y + 11.0_dp)) > 0.0_dp) &
            failures = failures + 1
        y = -12.0_dp
        y_dot = -13.0_dp
        call pipeline%jvp_device(device, x, theta_dot, x_dot, y, y_dot, status)
        if (status%code /= FORTNUM_NOT_IMPLEMENTED .or. maxval(abs(y + 12.0_dp)) > 0.0_dp .or. &
            maxval(abs(y_dot + 13.0_dp)) > 0.0_dp) failures = failures + 1
        theta_bar = -14.0_dp
        x_bar = -15.0_dp
        call pipeline%vjp_device(device, x, u, theta_bar, x_bar, status)
        if (status%code /= FORTNUM_NOT_IMPLEMENTED .or. &
            maxval(abs(theta_bar + 14.0_dp)) > 0.0_dp .or. &
            maxval(abs(x_bar + 15.0_dp)) > 0.0_dp) failures = failures + 1
        theta_hvp = -16.0_dp
        x_hvp = -17.0_dp
        call pipeline%hvp_device(device, x, u, theta_dot, x_dot, theta_hvp, x_hvp, status)
        if (status%code /= FORTNUM_NOT_IMPLEMENTED .or. &
            maxval(abs(theta_hvp + 16.0_dp)) > 0.0_dp .or. &
            maxval(abs(x_hvp + 17.0_dp)) > 0.0_dp) failures = failures + 1
        if (pipeline%device_supported(FORTML_DEVICE_CUDA)) failures = failures + 1
    end subroutine check_cuda_refusal

end program test_basis_sequential_device
