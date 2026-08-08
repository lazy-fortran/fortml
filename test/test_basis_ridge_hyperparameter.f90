program test_basis_ridge_hyperparameter
    !! Independent oracle for the joint basis/ridge hyperparameter products.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_basis, only: basis_map_t, make_fourier_basis
    use fortml_pipeline, only: basis_pipeline_t, make_basis_pipeline
    use fortml_basis_pipeline_training, only: basis_pipeline_training_objective_t
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer :: failures

    failures = 0
    call check_joint_products(failures)
    call check_fixed_ridge_layout(failures)
    if (failures /= 0) then
        write (error_unit, '(a,i0)') "FAIL basis ridge hyperparameter cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS basis ridge hyperparameter independent behavioral oracles"

contains

    subroutine check_joint_products(failures)
        integer, intent(inout) :: failures
        integer, parameter :: n = 7
        real(dp) :: x(n, 1), y(n, 1), theta(5), direction(5), gradient(5)
        real(dp) :: gradient_plus(5), gradient_minus(5), hvp(5)
        real(dp) :: theta_plus(5), theta_minus(5), value, value_plus, value_minus
        real(dp) :: tangent, h, frequency, direct_value, residual(n)
        type(basis_map_t) :: fourier
        type(basis_pipeline_t) :: pipeline
        type(basis_pipeline_training_objective_t) :: objective
        type(fortnum_status_t) :: status
        integer :: i, j

        x(:, 1) = [-1.2_dp, -0.8_dp, -0.35_dp, 0.1_dp, 0.45_dp, 0.9_dp, 1.3_dp]
        frequency = 0.73_dp
        do i = 1, n
            y(i, 1) = 0.4_dp + 1.2_dp*sin(frequency*x(i, 1)) - &
                0.8_dp*cos(frequency*x(i, 1))
        end do
        fourier = make_fourier_basis(1, reshape([frequency], [1, 1]), status)
        pipeline = make_basis_pipeline(1, status)
        call pipeline%append(fourier, status, name="fourier")
        call objective%initialize(pipeline, x, y, status, ridge=0.03_dp, &
            optimize_ridge=.true.)
        call check(status_ok(status) .and. objective%initialized() .and. &
            objective%parameter_count() == 5, "joint layout and initialization", failures)
        theta = [log(frequency), 0.4_dp, 1.2_dp, -0.8_dp, 0.03_dp]
        direction = [0.17_dp, -0.11_dp, 0.07_dp, -0.05_dp, 0.013_dp]
        call objective%value_gradient(theta, value, gradient, status)
        direct_value = 0.0_dp
        do i = 1, n
            residual(i) = theta(2) + theta(3)*sin(exp(theta(1))*x(i, 1)) + &
                theta(4)*cos(exp(theta(1))*x(i, 1)) - y(i, 1)
            direct_value = direct_value + 0.5_dp*residual(i)**2
        end do
        direct_value = direct_value/real(n, dp) + 0.5_dp*theta(5)* &
            (theta(3)**2 + theta(4)**2)
        call check(status_ok(status) .and. abs(value-direct_value) < 2.0e-13_dp .and. &
            abs(gradient(5) - 0.5_dp*(theta(3)**2 + theta(4)**2)) < 2.0e-13_dp, &
            "independent value and ridge derivative", failures)
        call objective%jvp(theta, direction, value_plus, tangent, status)
        call check(status_ok(status) .and. abs(tangent-dot_product(gradient, direction)) < &
            2.0e-12_dp, "joint JVP contraction", failures)
        call objective%hvp(theta, direction, hvp, status)
        h = 2.0e-5_dp
        theta_plus = theta + h*direction
        theta_minus = theta - h*direction
        call objective%value_gradient(theta_plus, value_plus, gradient_plus, status)
        call objective%value_gradient(theta_minus, value_minus, gradient_minus, status)
        call check(status_ok(status) .and. maxval(abs((gradient_plus-gradient_minus)/ &
            (2.0_dp*h)-hvp)) < 4.0e-6_dp, "joint HVP directional oracle", failures)
        call objective%value_gradient(theta, value_plus, gradient_plus, status)
        do j = 1, size(theta)
            theta_plus = theta
            theta_minus = theta
            theta_plus(j) = theta_plus(j) + h
            theta_minus(j) = theta_minus(j) - h
            call objective%value_gradient(theta_plus, value_plus, gradient_plus, status)
            call objective%value_gradient(theta_minus, value_minus, gradient_minus, status)
            call check(abs((value_plus-value_minus)/(2.0_dp*h)-gradient(j)) < 4.0e-6_dp, &
                "joint gradient coordinate", failures)
        end do
        theta(5) = -1.0e-3_dp
        call objective%value_gradient(theta, value, gradient, status)
        call check(.not. status_ok(status), "negative optimized ridge refusal", failures)
    end subroutine check_joint_products

    subroutine check_fixed_ridge_layout(failures)
        integer, intent(inout) :: failures
        real(dp) :: x(2, 1), y(2, 1)
        type(basis_map_t) :: fourier
        type(basis_pipeline_t) :: pipeline
        type(basis_pipeline_training_objective_t) :: objective
        type(fortnum_status_t) :: status

        x(:, 1) = [-1.0_dp, 1.0_dp]
        y(:, 1) = [0.0_dp, 1.0_dp]
        fourier = make_fourier_basis(1, reshape([1.0_dp], [1, 1]), status)
        pipeline = make_basis_pipeline(1, status)
        call pipeline%append(fourier, status)
        call objective%initialize(pipeline, x, y, status, ridge=0.2_dp)
        call check(status_ok(status) .and. objective%parameter_count() == 4, &
            "fixed-ridge layout remains explicit", failures)
    end subroutine check_fixed_ridge_layout

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "  FAIL [basis ridge] "//trim(description)
        end if
    end subroutine check

end program test_basis_ridge_hyperparameter
