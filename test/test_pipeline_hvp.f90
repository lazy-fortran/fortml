program test_pipeline_hvp
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_basis, only: basis_map_t
    use fortml_pipeline, only: basis_pipeline_t, sequential_basis_pipeline_t
    use fortml_column_pipeline, only: column_basis_pipeline_t
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer :: failures
    real(dp) :: x(4, 2), x_dot(4, 2)

    failures = 0
    x = reshape([0.17_dp, -0.31_dp, 0.52_dp, 0.91_dp, &
        -0.42_dp, 0.28_dp, 0.73_dp, -0.66_dp], shape(x))
    x_dot = reshape([-0.23_dp, 0.19_dp, 0.07_dp, -0.13_dp, &
        0.29_dp, -0.11_dp, 0.17_dp, 0.05_dp], shape(x_dot))

    call test_horizontal(x, x_dot, failures)
    call test_column_selecting(x, x_dot, failures)
    call test_sequential(x, x_dot, failures)
    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, " pipeline HVP test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS"

contains

    subroutine test_horizontal(x, x_dot, failures)
        real(dp), intent(in) :: x(:, :), x_dot(:, :)
        integer, intent(inout) :: failures
        type(basis_pipeline_t) :: pipeline
        type(basis_map_t) :: stage
        type(fortnum_status_t) :: status
        real(dp) :: frequencies(1, 2), h, error_theta, error_x
        real(dp), allocatable :: theta(:), dtheta(:), theta_bar(:), theta_plus(:), theta_minus(:)
        real(dp), allocatable :: x_bar(:, :), x_plus(:, :), x_minus(:, :), x_hvp(:, :)
        real(dp), allocatable :: u(:, :), theta_hvp(:)
        integer :: i

        call pipeline%initialize(2, status)
        call stage%initialize_polynomial(2, 2, status)
        call pipeline%append(stage, status, "poly")
        frequencies = reshape([0.8_dp, 1.3_dp], shape(frequencies))
        call stage%initialize_fourier(2, frequencies, status)
        call pipeline%append(stage, status, "fourier")
        call pipeline%fit(x, status)
        allocate(u(size(x, 1), pipeline%feature_count()))
        u = reshape([(0.07_dp*real(i, dp) - 0.23_dp, i=1, size(u))], shape(u))
        theta = pipeline%parameters()
        allocate(dtheta(size(theta)), theta_bar(size(theta)), theta_plus(size(theta)), &
            theta_minus(size(theta)), theta_hvp(size(theta)))
        dtheta = [(0.13_dp*real(i, dp) - 0.21_dp, i=1, size(theta))]
        allocate(x_bar(size(x, 1), size(x, 2)), x_plus(size(x, 1), size(x, 2)), &
            x_minus(size(x, 1), size(x, 2)), x_hvp(size(x, 1), size(x, 2)))
        h = 1.0e-5_dp
        call pipeline%vjp(x, u, theta_bar, x_bar, status)
        call pipeline%hvp(x, u, dtheta, x_dot, theta_hvp, x_hvp, status)
        call pipeline%set_parameters(theta + h*dtheta, status)
        call pipeline%vjp(x + h*x_dot, u, theta_plus, x_plus, status)
        call pipeline%set_parameters(theta - h*dtheta, status)
        call pipeline%vjp(x - h*x_dot, u, theta_minus, x_minus, status)
        call pipeline%set_parameters(theta, status)
        error_theta = maxval(abs(theta_hvp - (theta_plus - theta_minus)/(2.0_dp*h)))
        error_x = maxval(abs(x_hvp - (x_plus - x_minus)/(2.0_dp*h)))
        if (.not. status_ok(status) .or. error_theta > 3.0e-8_dp .or. &
                error_x > 3.0e-8_dp) then
            write (error_unit, '(a,2(es12.4,1x))') "FAIL [horizontal] errors=", &
                error_theta, error_x
            failures = failures + 1
        end if
    end subroutine test_horizontal

    subroutine test_column_selecting(x, x_dot, failures)
        real(dp), intent(in) :: x(:, :), x_dot(:, :)
        integer, intent(inout) :: failures
        type(column_basis_pipeline_t) :: pipeline
        type(basis_map_t) :: stage
        type(fortnum_status_t) :: status
        real(dp) :: frequencies(1, 1), centers(1, 1), scales(1, 1)
        real(dp) :: h, error_theta, error_x
        real(dp), allocatable :: theta(:), dtheta(:), theta_bar(:), theta_plus(:), theta_minus(:)
        real(dp), allocatable :: x_bar(:, :), x_plus(:, :), x_minus(:, :), x_hvp(:, :)
        real(dp), allocatable :: u(:, :), theta_hvp(:)
        integer :: i

        call pipeline%initialize(2, status)
        frequencies(1, 1) = 0.9_dp
        call stage%initialize_fourier(1, frequencies, status)
        call pipeline%append(stage, [1], status, "fourier-x1")
        centers(1, 1) = 0.2_dp
        scales(1, 1) = 0.75_dp
        call stage%initialize_radial(1, centers, scales, status)
        call pipeline%append(stage, [2], status, "radial-x2")
        call pipeline%fit(x, status)
        allocate(u(size(x, 1), pipeline%feature_count()))
        u = reshape([(0.11_dp*real(i, dp) - 0.37_dp, i=1, size(u))], shape(u))
        theta = pipeline%parameters()
        allocate(dtheta(size(theta)), theta_bar(size(theta)), theta_plus(size(theta)), &
            theta_minus(size(theta)), theta_hvp(size(theta)))
        dtheta = [(0.09_dp*real(i, dp) - 0.18_dp, i=1, size(theta))]
        allocate(x_bar(size(x, 1), size(x, 2)), x_plus(size(x, 1), size(x, 2)), &
            x_minus(size(x, 1), size(x, 2)), x_hvp(size(x, 1), size(x, 2)))
        h = 1.0e-5_dp
        call pipeline%vjp(x, u, theta_bar, x_bar, status)
        call pipeline%hvp(x, u, dtheta, x_dot, theta_hvp, x_hvp, status)
        call pipeline%set_parameters(theta + h*dtheta, status)
        call pipeline%vjp(x + h*x_dot, u, theta_plus, x_plus, status)
        call pipeline%set_parameters(theta - h*dtheta, status)
        call pipeline%vjp(x - h*x_dot, u, theta_minus, x_minus, status)
        call pipeline%set_parameters(theta, status)
        error_theta = maxval(abs(theta_hvp - (theta_plus - theta_minus)/(2.0_dp*h)))
        error_x = maxval(abs(x_hvp - (x_plus - x_minus)/(2.0_dp*h)))
        if (.not. status_ok(status) .or. error_theta > 3.0e-8_dp .or. &
                error_x > 3.0e-8_dp) then
            write (error_unit, '(a,2(es12.4,1x))') "FAIL [column] errors=", &
                error_theta, error_x
            failures = failures + 1
        end if
    end subroutine test_column_selecting

    subroutine test_sequential(x, x_dot, failures)
        real(dp), intent(in) :: x(:, :), x_dot(:, :)
        integer, intent(inout) :: failures
        type(sequential_basis_pipeline_t) :: pipeline
        type(basis_map_t) :: stage
        type(fortnum_status_t) :: status
        real(dp) :: frequencies(1, 4), h, error_theta, error_x
        real(dp), allocatable :: theta(:), dtheta(:), theta_bar(:), theta_plus(:), theta_minus(:)
        real(dp), allocatable :: x_bar(:, :), x_plus(:, :), x_minus(:, :), x_hvp(:, :)
        real(dp), allocatable :: u(:, :), theta_hvp(:)
        integer :: i

        call pipeline%initialize(2, status)
        call stage%initialize_polynomial(2, 2, status)
        call pipeline%append(stage, status, "poly")
        frequencies = reshape([0.6_dp, 0.8_dp, 1.1_dp, 1.4_dp], shape(frequencies))
        call stage%initialize_fourier(4, frequencies, status)
        call pipeline%append(stage, status, "fourier")
        call pipeline%fit(x, status)
        allocate(u(size(x, 1), pipeline%feature_count()))
        u = reshape([(0.05_dp*real(i, dp) - 0.14_dp, i=1, size(u))], shape(u))
        theta = pipeline%parameters()
        allocate(dtheta(size(theta)), theta_bar(size(theta)), theta_plus(size(theta)), &
            theta_minus(size(theta)), theta_hvp(size(theta)))
        dtheta = [(0.08_dp*real(i, dp) - 0.17_dp, i=1, size(theta))]
        allocate(x_bar(size(x, 1), size(x, 2)), x_plus(size(x, 1), size(x, 2)), &
            x_minus(size(x, 1), size(x, 2)), x_hvp(size(x, 1), size(x, 2)))
        h = 1.0e-5_dp
        call pipeline%vjp(x, u, theta_bar, x_bar, status)
        call pipeline%hvp(x, u, dtheta, x_dot, theta_hvp, x_hvp, status)
        call pipeline%set_parameters(theta + h*dtheta, status)
        call pipeline%vjp(x + h*x_dot, u, theta_plus, x_plus, status)
        call pipeline%set_parameters(theta - h*dtheta, status)
        call pipeline%vjp(x - h*x_dot, u, theta_minus, x_minus, status)
        call pipeline%set_parameters(theta, status)
        error_theta = maxval(abs(theta_hvp - (theta_plus - theta_minus)/(2.0_dp*h)))
        error_x = maxval(abs(x_hvp - (x_plus - x_minus)/(2.0_dp*h)))
        if (.not. status_ok(status) .or. error_theta > 5.0e-7_dp .or. &
                error_x > 5.0e-7_dp) then
            write (error_unit, '(a,2(es12.4,1x))') "FAIL [sequential] errors=", &
                error_theta, error_x
            failures = failures + 1
        end if
    end subroutine test_sequential

end program test_pipeline_hvp
