program test_basis_linear_regression_hvp
    !! Independent finite-difference oracle for basis/readout HVP products.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_basis, only: basis_map_t, make_fourier_basis
    use fortml_pipeline, only: basis_pipeline_t, make_basis_pipeline
    use fortml_basis_linear_regression, only: basis_linear_regression_t
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer, parameter :: n = 7
    real(dp) :: x(n, 1), y(n, 1), x_dot(n, 1), u(n, 1)
    real(dp), allocatable :: theta(:), theta_dot(:), theta_hvp(:)
    real(dp), allocatable :: theta_plus(:), theta_minus(:)
    real(dp), allocatable :: theta_bar_plus(:), theta_bar_minus(:)
    real(dp) :: x_plus(n, 1), x_minus(n, 1), x_hvp(n, 1)
    real(dp) :: x_bar_plus(n, 1), x_bar_minus(n, 1), x_hvp_fd(n, 1)
    real(dp) :: frequency, h, error_theta, error_x
    type(basis_map_t) :: fourier
    type(basis_pipeline_t) :: pipeline
    type(basis_linear_regression_t) :: model
    type(fortnum_status_t) :: status
    integer :: i, failures

    x(:, 1) = [-1.2_dp, -0.8_dp, -0.35_dp, 0.1_dp, 0.45_dp, 0.9_dp, 1.3_dp]
    x_dot(:, 1) = [0.03_dp, -0.02_dp, 0.04_dp, -0.01_dp, 0.02_dp, -0.03_dp, 0.01_dp]
    frequency = 0.73_dp
    do i = 1, n
        y(i, 1) = 0.4_dp + 1.2_dp*sin(frequency*x(i, 1)) - &
            0.8_dp*cos(frequency*x(i, 1))
        u(i, 1) = 0.2_dp + 0.1_dp*real(i, dp)
    end do
    fourier = make_fourier_basis(1, reshape([frequency], [1, 1]), status)
    pipeline = make_basis_pipeline(1, status)
    call pipeline%append(fourier, status, name="fourier")
    call model%fit(pipeline, x, y, status, ridge=0.03_dp)
    failures = 0
    if (.not. status_ok(status) .or. .not. model%is_fitted() .or. &
        model%parameter_count() /= 4) then
        write (error_unit, '(a)') "FAIL basis/readout HVP initialization"
        error stop 1
    end if
    theta = model%parameters()
    allocate(theta_dot(size(theta)), theta_hvp(size(theta)), theta_plus(size(theta)), &
        theta_minus(size(theta)), theta_bar_plus(size(theta)), theta_bar_minus(size(theta)))
    theta_dot = [0.17_dp, -0.11_dp, 0.07_dp, -0.05_dp]
    h = 2.0e-5_dp
    call model%predict_hvp(x, u, theta_dot, x_dot, theta_hvp, x_hvp, status)
    if (.not. status_ok(status)) then
        write (error_unit, '(a,i0)') "FAIL basis/readout HVP status ", status%code
        error stop 1
    end if
    theta_plus = theta + h*theta_dot; theta_minus = theta - h*theta_dot
    x_plus = x + h*x_dot; x_minus = x - h*x_dot
    call model%set_parameters(theta_plus, status)
    call model%predict_vjp(x_plus, u, theta_bar_plus, x_bar_plus, status)
    call model%set_parameters(theta_minus, status)
    call model%predict_vjp(x_minus, u, theta_bar_minus, x_bar_minus, status)
    call model%set_parameters(theta, status)
    x_hvp_fd = (x_bar_plus - x_bar_minus)/(2.0_dp*h)
    error_theta = maxval(abs(theta_hvp - (theta_bar_plus - theta_bar_minus)/(2.0_dp*h)))
    error_x = maxval(abs(x_hvp - x_hvp_fd))
    call check(error_theta < 3.0e-5_dp, "parameter HVP finite-difference oracle", failures)
    call check(error_x < 3.0e-5_dp, "input HVP finite-difference oracle", failures)
    if (failures > 0) then
        write (error_unit, '(a,i0)') "FAIL basis/readout HVP cases: ", failures
        error stop 1
    end if
    write (*, '(a,2(es12.4,1x))') &
        "PASS basis/readout HVP independent oracle ", error_theta, error_x

contains

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failures
        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "  FAIL [basis-readout-hvp] "//description
        end if
    end subroutine check

end program test_basis_linear_regression_hvp
