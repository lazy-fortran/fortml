program test_physics_objective
    !! Independent affine-residual oracle for the PINN objective seam.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortnum_status, only: fortnum_status_t, status_ok, status_set, &
        FORTNUM_OK, FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortopt_objective, only: objective_t
    use fortml_physics_objective, only: physics_constraint_t, &
        physics_objective_t
    implicit none

    type :: affine_term_t
        real(dp) :: matrix(2, 2)
        real(dp) :: offset(2)
    end type affine_term_t

    type :: nonlinear_term_t
        real(dp) :: offset(2)
    end type nonlinear_term_t

    type(affine_term_t), target :: data_term, residual_term, boundary_term
    type(affine_term_t), target :: conservation_term
    type(nonlinear_term_t), target :: nonlinear_term
    type(physics_constraint_t) :: data, residual, boundary, conservation
    type(physics_constraint_t) :: nonlinear_constraint
    type(physics_objective_t) :: objective, bad_objective, partial_objective
    type(physics_objective_t) :: nonlinear_objective
    type(objective_t) :: fortopt_objective
    type(fortnum_status_t) :: status
    real(dp) :: theta(2), direction(2), gradient(2), adjoint(2), expected(2)
    real(dp) :: value, value_plus, value_minus, value_dot, h, scalar
    real(dp) :: term_values(4), expected_terms(4)
    real(dp) :: hessian_direction(2), hessian_fd(2), gradient_plus(2), gradient_minus(2)
    integer :: failures, i

    failures = 0
    theta = [0.37_dp, -0.42_dp]
    direction = [0.23_dp, -0.31_dp]
    data_term%matrix = reshape([1.0_dp, 0.2_dp, -0.4_dp, 0.7_dp], [2, 2])
    data_term%offset = [0.1_dp, -0.3_dp]
    residual_term%matrix = reshape([0.3_dp, -0.8_dp, 0.6_dp, 0.5_dp], [2, 2])
    residual_term%offset = [-0.5_dp, 0.2_dp]
    boundary_term%matrix = reshape([1.2_dp, 0.0_dp, 0.0_dp, -0.9_dp], [2, 2])
    boundary_term%offset = [0.2_dp, 0.1_dp]
    conservation_term%matrix = reshape([0.4_dp, 0.5_dp, -0.2_dp, 0.3_dp], [2, 2])
    conservation_term%offset = [-0.1_dp, 0.4_dp]

    call data%initialize(2, 2, 1.0_dp, data_term, affine_residual, &
        affine_jvp, affine_vjp, status)
    call check(status_ok(status), "data constraint initialization", failures)
    call residual%initialize(2, 2, 2.5_dp, residual_term, affine_residual, &
        affine_jvp, affine_vjp, status)
    call check(status_ok(status), "residual constraint initialization", failures)
    call boundary%initialize(2, 2, 0.75_dp, boundary_term, affine_residual, &
        affine_jvp, affine_vjp, status)
    call check(status_ok(status), "boundary constraint initialization", failures)
    call conservation%initialize(2, 2, 1.25_dp, conservation_term, affine_residual, &
        affine_jvp, affine_vjp, status)
    call check(status_ok(status), "conservation constraint initialization", failures)
    call objective%initialize(2, data, residual, boundary, conservation, status)
    call check(status_ok(status), "objective initialization", failures)
    call check(objective%initialized(), "objective initialized predicate", failures)

    call objective%value_gradient(theta, value, gradient, status)
    call check(status_ok(status), "objective value and gradient", failures)
    call oracle_value_gradient(theta, value_plus, expected)
    call check(abs(value - value_plus) < 2.0e-14_dp, &
        "independent weighted residual value", failures)
    call check(maxval(abs(gradient - expected)) < 2.0e-14_dp, &
        "independent weighted residual gradient", failures)
    call objective%term_values(theta, term_values, status)
    call oracle_term_values(theta, expected_terms)
    call check(status_ok(status) .and. maxval(abs(term_values - expected_terms)) < &
        2.0e-14_dp .and. abs(sum(term_values) - value) < 2.0e-14_dp, &
        "named residual contribution diagnostic", failures)
    call objective%value(theta, value_plus, status)
    call objective%gradient(theta, adjoint, status)
    call check(status_ok(status) .and. abs(value_plus - value) < 2.0e-14_dp .and. &
        maxval(abs(adjoint - gradient)) < 2.0e-14_dp, &
        "separate value and gradient methods", failures)

    h = 2.0e-6_dp
    do i = 1, 2
        theta(i) = theta(i) + h
        call objective%value_gradient(theta, value_plus, adjoint, status)
        call check(status_ok(status), "plus finite-difference value", failures)
        theta(i) = theta(i) - 2.0_dp*h
        call objective%value_gradient(theta, value_minus, adjoint, status)
        call check(status_ok(status), "minus finite-difference value", failures)
        theta(i) = theta(i) + h
        call check(abs(gradient(i) - (value_plus - value_minus)/(2.0_dp*h)) < &
            2.0e-8_dp, "central-difference gradient", failures)
    end do

    call objective%jvp(theta, direction, value, value_dot, status)
    call check(status_ok(status), "objective JVP", failures)
    call objective%value_gradient(theta + h*direction, value_plus, adjoint, status)
    call objective%value_gradient(theta - h*direction, value_minus, adjoint, status)
    call check(abs(value_dot - (value_plus - value_minus)/(2.0_dp*h)) < 2.0e-8_dp, &
        "central-difference JVP", failures)

    scalar = -1.7_dp
    call objective%vjp(theta, scalar, adjoint, status)
    call objective%value_gradient(theta, value, gradient, status)
    call check(status_ok(status), "objective VJP", failures)
    call check(maxval(abs(adjoint - scalar*gradient)) < 2.0e-14_dp, &
        "scalar adjoint identity", failures)

    call objective%as_objective(fortopt_objective, status)
    call check(status_ok(status), "FortOpt objective adapter", failures)
    call fortopt_objective%value_gradient(theta, value_plus, expected, status)
    call check(status_ok(status) .and. abs(value_plus - value) < 2.0e-14_dp .and. &
        maxval(abs(expected - gradient)) < 2.0e-14_dp, &
        "FortOpt callback equivalence", failures)

    call objective%hvp(theta, direction, hessian_direction, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED .and. &
        maxval(abs(hessian_direction)) == 0.0_dp, &
        "typed HVP refusal", failures)

    ! A nonlinear independent oracle exercises the optional exact residual HVP
    ! path used by PINN/symplectic adapters and FortOpt L-BFGS-B.
    nonlinear_term%offset = [0.17_dp, -0.23_dp]
    call nonlinear_constraint%initialize(2, 2, 1.75_dp, nonlinear_term, &
        nonlinear_residual, nonlinear_jvp, nonlinear_vjp, status, &
        nonlinear_hvp)
    call check(status_ok(status), "nonlinear HVP constraint initialization", failures)
    call nonlinear_objective%initialize(2, residual=nonlinear_constraint, status=status)
    call check(status_ok(status), "nonlinear HVP objective initialization", failures)
    call nonlinear_objective%value_gradient(theta, value, gradient, status)
    call check(status_ok(status), "nonlinear HVP objective gradient", failures)
    call nonlinear_objective%hvp(theta, direction, hessian_direction, status)
    call check(status_ok(status), "exact nonlinear objective HVP", failures)
    theta = theta + h*direction
    call nonlinear_objective%gradient(theta, gradient_plus, status)
    theta = theta - 2.0_dp*h*direction
    call nonlinear_objective%gradient(theta, gradient_minus, status)
    theta = theta + h*direction
    hessian_fd = (gradient_plus - gradient_minus)/(2.0_dp*h)
    call check(status_ok(status) .and. maxval(abs(hessian_direction - hessian_fd)) < &
        3.0e-8_dp, "nonlinear HVP central-difference oracle", failures)

    call data%initialize(2, 2, 0.0_dp, data_term, affine_residual, affine_jvp, &
        affine_vjp, status)
    call check(status%code == FORTNUM_DOMAIN_ERROR, "zero-weight refusal", failures)
    call bad_objective%initialize(2, status=status)
    call check(status%code == FORTNUM_DOMAIN_ERROR, &
        "empty objective refusal", failures)
    call partial_objective%initialize(2, residual=residual, status=status)
    call check(status_ok(status), "partial objective initialization", failures)
    call partial_objective%term_values(theta, term_values, status)
    call check(status_ok(status), "partial objective diagnostic", failures)
    call residual%value(theta, expected_terms(2), status)
    expected_terms = [0.0_dp, expected_terms(2), 0.0_dp, 0.0_dp]
    call check(status_ok(status) .and. abs(term_values(1)) == 0.0_dp .and. &
        abs(term_values(2) - expected_terms(2)) < 2.0e-14_dp .and. &
        abs(term_values(3)) == 0.0_dp .and. abs(term_values(4)) == 0.0_dp, &
        "inactive named contribution slots", failures)
    call objective%value_gradient([theta(1)], value, gradient, status)
    call check(status%code == FORTNUM_DOMAIN_ERROR, "shape refusal", failures)
    call objective%term_values([theta(1)], term_values, status)
    call check(status%code == FORTNUM_DOMAIN_ERROR, "diagnostic shape refusal", failures)

    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, " physics objective test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS physics objective independent behavioral oracles"

contains

    subroutine affine_residual(context, theta, residual, status)
        class(*), pointer, intent(in) :: context
        real(dp), intent(in) :: theta(:)
        real(dp), intent(out) :: residual(:)
        type(fortnum_status_t), intent(out) :: status

        select type (context)
            type is (affine_term_t)
            residual = matmul(context%matrix, theta) - context%offset
            call status_set_local(status, 0)
        class default
            residual = 0.0_dp
            call status_set_local(status, 1)
        end select
    end subroutine affine_residual

    subroutine affine_jvp(context, theta, theta_dot, residual, residual_dot, status)
        class(*), pointer, intent(in) :: context
        real(dp), intent(in) :: theta(:), theta_dot(:)
        real(dp), intent(out) :: residual(:), residual_dot(:)
        type(fortnum_status_t), intent(out) :: status

        select type (context)
            type is (affine_term_t)
            residual = matmul(context%matrix, theta) - context%offset
            residual_dot = matmul(context%matrix, theta_dot)
            call status_set_local(status, 0)
        class default
            residual = 0.0_dp
            residual_dot = 0.0_dp
            call status_set_local(status, 1)
        end select
    end subroutine affine_jvp

    subroutine affine_vjp(context, theta, residual_bar, theta_bar, status)
        class(*), pointer, intent(in) :: context
        real(dp), intent(in) :: theta(:), residual_bar(:)
        real(dp), intent(out) :: theta_bar(:)
        type(fortnum_status_t), intent(out) :: status

        associate (unused_theta => theta)
        end associate
        select type (context)
            type is (affine_term_t)
            theta_bar = matmul(transpose(context%matrix), residual_bar)
            call status_set_local(status, 0)
        class default
            theta_bar = 0.0_dp
            call status_set_local(status, 1)
        end select
    end subroutine affine_vjp

    subroutine nonlinear_residual(context, theta, residual, status)
        class(*), pointer, intent(in) :: context
        real(dp), intent(in) :: theta(:)
        real(dp), intent(out) :: residual(:)
        type(fortnum_status_t), intent(out) :: status

        select type (context)
            type is (nonlinear_term_t)
            residual(1) = theta(1)**2 + 0.5_dp*theta(2) - context%offset(1)
            residual(2) = theta(1)*theta(2) - context%offset(2)
            call status_set_local(status, 0)
        class default
            residual = 0.0_dp
            call status_set_local(status, 1)
        end select
    end subroutine nonlinear_residual

    subroutine nonlinear_jvp(context, theta, theta_dot, residual, residual_dot, status)
        class(*), pointer, intent(in) :: context
        real(dp), intent(in) :: theta(:), theta_dot(:)
        real(dp), intent(out) :: residual(:), residual_dot(:)
        type(fortnum_status_t), intent(out) :: status

        call nonlinear_residual(context, theta, residual, status)
        if (.not. status_ok(status)) then
            residual_dot = 0.0_dp
            return
        end if
        residual_dot(1) = 2.0_dp*theta(1)*theta_dot(1) + 0.5_dp*theta_dot(2)
        residual_dot(2) = theta_dot(1)*theta(2) + theta(1)*theta_dot(2)
        call status_set_local(status, 0)
    end subroutine nonlinear_jvp

    subroutine nonlinear_vjp(context, theta, residual_bar, theta_bar, status)
        class(*), pointer, intent(in) :: context
        real(dp), intent(in) :: theta(:), residual_bar(:)
        real(dp), intent(out) :: theta_bar(:)
        type(fortnum_status_t), intent(out) :: status

        associate (unused_context => context)
        end associate
        theta_bar(1) = 2.0_dp*theta(1)*residual_bar(1) + &
            theta(2)*residual_bar(2)
        theta_bar(2) = 0.5_dp*residual_bar(1) + &
            theta(1)*residual_bar(2)
        call status_set_local(status, 0)
    end subroutine nonlinear_vjp

    subroutine nonlinear_hvp(context, theta, theta_dot, residual_bar, &
            residual_bar_dot, theta_hvp, status)
        class(*), pointer, intent(in) :: context
        real(dp), intent(in) :: theta(:), theta_dot(:)
        real(dp), intent(in) :: residual_bar(:), residual_bar_dot(:)
        real(dp), intent(out) :: theta_hvp(:)
        type(fortnum_status_t), intent(out) :: status

        associate (unused_context => context)
        end associate
        theta_hvp(1) = 2.0_dp*theta_dot(1)*residual_bar(1) + &
            2.0_dp*theta(1)*residual_bar_dot(1) + &
            theta_dot(2)*residual_bar(2) + theta(2)*residual_bar_dot(2)
        theta_hvp(2) = 0.5_dp*residual_bar_dot(1) + &
            theta_dot(1)*residual_bar(2) + theta(1)*residual_bar_dot(2)
        call status_set_local(status, 0)
    end subroutine nonlinear_hvp

    subroutine oracle_value_gradient(theta, value, gradient)
        real(dp), intent(in) :: theta(:)
        real(dp), intent(out) :: value, gradient(:)

        value = 0.0_dp
        gradient = 0.0_dp
        call oracle_term(data_term, 1.0_dp, theta, value, gradient)
        call oracle_term(residual_term, 2.5_dp, theta, value, gradient)
        call oracle_term(boundary_term, 0.75_dp, theta, value, gradient)
        call oracle_term(conservation_term, 1.25_dp, theta, value, gradient)
    end subroutine oracle_value_gradient

    subroutine oracle_term_values(theta, values)
        real(dp), intent(in) :: theta(:)
        real(dp), intent(out) :: values(:)

        call oracle_term_value(data_term, 1.0_dp, theta, values(1))
        call oracle_term_value(residual_term, 2.5_dp, theta, values(2))
        call oracle_term_value(boundary_term, 0.75_dp, theta, values(3))
        call oracle_term_value(conservation_term, 1.25_dp, theta, values(4))
    end subroutine oracle_term_values

    subroutine oracle_term_value(term, weight, theta, value)
        type(affine_term_t), intent(in) :: term
        real(dp), intent(in) :: weight, theta(:)
        real(dp), intent(out) :: value
        real(dp) :: residual(2)

        residual = matmul(term%matrix, theta) - term%offset
        value = weight*dot_product(residual, residual)/4.0_dp
    end subroutine oracle_term_value

    subroutine oracle_term(term, weight, theta, value, gradient)
        type(affine_term_t), intent(in) :: term
        real(dp), intent(in) :: weight, theta(:)
        real(dp), intent(inout) :: value, gradient(:)
        real(dp) :: residual(2)

        residual = matmul(term%matrix, theta) - term%offset
        value = value + weight*dot_product(residual, residual)/4.0_dp
        gradient = gradient + weight*matmul(transpose(term%matrix), residual)/2.0_dp
    end subroutine oracle_term

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: description
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (*, '(a)') "FAIL "//trim(description)
        end if
    end subroutine check

    subroutine status_set_local(status, code)
        type(fortnum_status_t), intent(out) :: status
        integer, intent(in) :: code

        if (code == 0) then
            call status_set(status, FORTNUM_OK, "")
        else
            call status_set(status, FORTNUM_DOMAIN_ERROR, "bad test context")
        end if
    end subroutine status_set_local

end program test_physics_objective
