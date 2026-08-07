program test_pinn
    !! Manufactured PINN adapter oracle: four named terms, products, fit, device.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortnum_status, only: fortnum_status_t, status_ok, status_set, &
        FORTNUM_OK, FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    use fortml_physics_objective, only: physics_constraint_t, &
        physics_objective_t
    use fortml_pinn, only: pinn_training_adapter_t
    use fortopt_lbfgsb, only: lbfgsb_options_t, lbfgsb_result_t
    implicit none

    type :: affine_context_t
        real(dp) :: scale(2)
        real(dp) :: offset(2)
    end type affine_context_t

    type :: quadratic_context_t
        real(dp) :: offset = 0.25_dp
    end type quadratic_context_t

    type(affine_context_t), target :: data_context, residual_context
    type(affine_context_t), target :: boundary_context, conservation_context
    type(quadratic_context_t), target :: quadratic_context
    type(physics_constraint_t) :: data, residual, boundary, conservation
    type(physics_constraint_t) :: quadratic_constraint
    type(physics_objective_t) :: objective, quadratic_objective
    type(pinn_training_adapter_t) :: pinn, quadratic_pinn, cuda_pinn
    type(lbfgsb_options_t) :: options
    type(lbfgsb_result_t) :: result
    type(fortnum_status_t) :: status
    real(dp) :: theta(1), direction(1), gradient(1), gradient_plus(1)
    real(dp) :: gradient_minus(1), hessian_direction(1), term_values(4)
    real(dp) :: value, value_plus, value_minus, value_dot, h, scalar
    real(dp) :: fit_parameters(1), lower(1), upper(1)
    integer :: failures

    failures = 0
    h = 2.0e-6_dp
    theta = [0.25_dp]
    direction = [0.17_dp]
    scalar = -1.3_dp
    data_context%scale = [1.0_dp, 1.0_dp]
    residual_context%scale = [1.0_dp, 1.0_dp]
    boundary_context%scale = [1.0_dp, 1.0_dp]
    conservation_context%scale = [1.0_dp, 1.0_dp]
    data_context%offset = [1.0_dp, 1.0_dp]
    residual_context%offset = [1.0_dp, 1.0_dp]
    boundary_context%offset = [1.0_dp, 1.0_dp]
    conservation_context%offset = [1.0_dp, 1.0_dp]

    call data%initialize(1, 2, 1.0_dp, data_context, affine_residual, &
        affine_jvp, affine_vjp, status)
    call check(status_ok(status), "data constraint", failures)
    call residual%initialize(1, 2, 2.0_dp, residual_context, affine_residual, &
        affine_jvp, affine_vjp, status)
    call check(status_ok(status), "residual constraint", failures)
    call boundary%initialize(1, 2, 0.5_dp, boundary_context, affine_residual, &
        affine_jvp, affine_vjp, status)
    call check(status_ok(status), "boundary constraint", failures)
    call conservation%initialize(1, 2, 1.5_dp, conservation_context, &
        affine_residual, affine_jvp, affine_vjp, status)
    call check(status_ok(status), "conservation constraint", failures)
    call objective%initialize(1, data, residual, boundary, conservation, status)
    call check(status_ok(status), "four-term objective", failures)

    call pinn%initialize(objective, status)
    call check(status_ok(status) .and. pinn%initialized(), &
        "PINN CPU initialization", failures)
    call check(pinn%parameter_count() == 1 .and. &
        pinn%device_supported(FORTML_DEVICE_CPU) .and. &
        .not. pinn%device_supported(FORTML_DEVICE_CUDA), &
        "PINN capability contract", failures)
    call pinn%term_values(theta, term_values, status)
    call check(status_ok(status) .and. abs(sum(term_values) - &
        expected_affine_value(theta(1))) < 2.0e-14_dp, &
        "named PINN term values", failures)
    call pinn%value_gradient(theta, value, gradient, status)
    call check(status_ok(status), "PINN value and gradient", failures)
    call check(abs(value - expected_affine_value(theta(1))) < 2.0e-14_dp .and. &
        abs(gradient(1) - expected_affine_gradient(theta(1))) < 2.0e-14_dp, &
        "manufactured affine oracle", failures)
    call pinn%jvp(theta, direction, value, value_dot, status)
    call pinn%value(theta + h*direction, value_plus, status)
    call pinn%value(theta - h*direction, value_minus, status)
    call check(status_ok(status) .and. abs(value_dot - &
        (value_plus - value_minus)/(2.0_dp*h)) < 2.0e-8_dp, &
        "PINN directional JVP", failures)
    call pinn%vjp(theta, scalar, gradient_plus, status)
    call pinn%gradient(theta, gradient, status)
    call check(status_ok(status) .and. abs(gradient_plus(1) - scalar*gradient(1)) < &
        2.0e-14_dp, "PINN scalar VJP", failures)
    call pinn%hvp(theta, direction, hessian_direction, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED .and. &
        abs(hessian_direction(1)) == 0.0_dp, "typed affine HVP refusal", failures)

    fit_parameters = [0.0_dp]
    lower = [-2.0_dp]
    upper = [2.0_dp]
    options%max_iterations = 100
    options%gradient_tolerance = 1.0e-10_dp
    call pinn%fit_lbfgsb(fit_parameters, lower, upper, options, result, status)
    call check(status_ok(status) .and. abs(fit_parameters(1) - 1.0_dp) < 2.0e-7_dp, &
        "PINN FortOpt L-BFGS-B manufactured fit", failures)

    call cuda_pinn%initialize(objective, status, device_kind=FORTML_DEVICE_CUDA)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED .and. &
        .not. cuda_pinn%initialized(), "typed CUDA PINN refusal", failures)
    call pinn%select_device(FORTML_DEVICE_CUDA, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "typed CUDA selection refusal", failures)
    call pinn%select_device(FORTML_DEVICE_CPU, status)
    call check(status_ok(status), "CPU device selection", failures)

    quadratic_context%offset = 0.25_dp
    call quadratic_constraint%initialize(1, 1, 2.0_dp, quadratic_context, &
        quadratic_residual, quadratic_jvp, quadratic_vjp, status, quadratic_hvp)
    call check(status_ok(status), "quadratic HVP constraint", failures)
    call quadratic_objective%initialize(1, residual=quadratic_constraint, status=status)
    call check(status_ok(status), "quadratic HVP objective", failures)
    call quadratic_pinn%initialize(quadratic_objective, status)
    call check(status_ok(status), "quadratic PINN initialization", failures)
    theta = [0.7_dp]
    direction = [-0.23_dp]
    call quadratic_pinn%gradient(theta, gradient, status)
    call quadratic_pinn%hvp(theta, direction, hessian_direction, status)
    theta = theta + h*direction
    call quadratic_pinn%gradient(theta, gradient_plus, status)
    theta = theta - 2.0_dp*h*direction
    call quadratic_pinn%gradient(theta, gradient_minus, status)
    theta = theta + h*direction
    call check(status_ok(status) .and. abs(hessian_direction(1) - &
        (gradient_plus(1) - gradient_minus(1))/(2.0_dp*h)) < 3.0e-8_dp, &
        "exact nonlinear PINN HVP oracle", failures)

    call pinn%value([theta(1), theta(1)], value, status)
    call check(status%code == FORTNUM_DOMAIN_ERROR, &
        "PINN parameter shape refusal", failures)

    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, " PINN adapter test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS PINN adapter manufactured-solution oracles"

contains

    subroutine affine_residual(context, theta, residual, status)
        class(*), pointer, intent(in) :: context
        real(dp), intent(in) :: theta(:)
        real(dp), intent(out) :: residual(:)
        type(fortnum_status_t), intent(out) :: status

        select type (context)
            type is (affine_context_t)
            residual = context%scale*theta(1) - context%offset
            call status_set(status, FORTNUM_OK, "")
        class default
            residual = 0.0_dp
            call status_set(status, FORTNUM_DOMAIN_ERROR, "bad affine context")
        end select
    end subroutine affine_residual

    subroutine affine_jvp(context, theta, theta_dot, residual, residual_dot, status)
        class(*), pointer, intent(in) :: context
        real(dp), intent(in) :: theta(:), theta_dot(:)
        real(dp), intent(out) :: residual(:), residual_dot(:)
        type(fortnum_status_t), intent(out) :: status

        call affine_residual(context, theta, residual, status)
        if (.not. status_ok(status)) then
            residual_dot = 0.0_dp
            return
        end if
        select type (context)
            type is (affine_context_t)
            residual_dot = context%scale*theta_dot(1)
            call status_set(status, FORTNUM_OK, "")
        class default
            residual_dot = 0.0_dp
            call status_set(status, FORTNUM_DOMAIN_ERROR, "bad affine context")
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
            type is (affine_context_t)
            theta_bar(1) = dot_product(context%scale, residual_bar)
            call status_set(status, FORTNUM_OK, "")
        class default
            theta_bar = 0.0_dp
            call status_set(status, FORTNUM_DOMAIN_ERROR, "bad affine context")
        end select
    end subroutine affine_vjp

    subroutine quadratic_residual(context, theta, residual, status)
        class(*), pointer, intent(in) :: context
        real(dp), intent(in) :: theta(:)
        real(dp), intent(out) :: residual(:)
        type(fortnum_status_t), intent(out) :: status

        select type (context)
            type is (quadratic_context_t)
            residual(1) = theta(1)**2 - context%offset
            call status_set(status, FORTNUM_OK, "")
        class default
            residual = 0.0_dp
            call status_set(status, FORTNUM_DOMAIN_ERROR, "bad quadratic context")
        end select
    end subroutine quadratic_residual

    subroutine quadratic_jvp(context, theta, theta_dot, residual, residual_dot, status)
        class(*), pointer, intent(in) :: context
        real(dp), intent(in) :: theta(:), theta_dot(:)
        real(dp), intent(out) :: residual(:), residual_dot(:)
        type(fortnum_status_t), intent(out) :: status

        call quadratic_residual(context, theta, residual, status)
        if (.not. status_ok(status)) then
            residual_dot = 0.0_dp
            return
        end if
        residual_dot(1) = 2.0_dp*theta(1)*theta_dot(1)
        call status_set(status, FORTNUM_OK, "")
    end subroutine quadratic_jvp

    subroutine quadratic_vjp(context, theta, residual_bar, theta_bar, status)
        class(*), pointer, intent(in) :: context
        real(dp), intent(in) :: theta(:), residual_bar(:)
        real(dp), intent(out) :: theta_bar(:)
        type(fortnum_status_t), intent(out) :: status

        associate (unused_context => context)
        end associate
        theta_bar(1) = 2.0_dp*theta(1)*residual_bar(1)
        call status_set(status, FORTNUM_OK, "")
    end subroutine quadratic_vjp

    subroutine quadratic_hvp(context, theta, theta_dot, residual_bar, &
            residual_bar_dot, theta_hvp, status)
        class(*), pointer, intent(in) :: context
        real(dp), intent(in) :: theta(:), theta_dot(:)
        real(dp), intent(in) :: residual_bar(:), residual_bar_dot(:)
        real(dp), intent(out) :: theta_hvp(:)
        type(fortnum_status_t), intent(out) :: status

        associate (unused_context => context)
        end associate
        theta_hvp(1) = 2.0_dp*theta_dot(1)*residual_bar(1) + &
            2.0_dp*theta(1)*residual_bar_dot(1)
        call status_set(status, FORTNUM_OK, "")
    end subroutine quadratic_hvp

    real(dp) function expected_affine_value(parameter) result(value)
        real(dp), intent(in) :: parameter

        value = 0.5_dp*(1.0_dp + 2.0_dp + 0.5_dp + 1.5_dp)* &
            (parameter - 1.0_dp)**2
    end function expected_affine_value

    real(dp) function expected_affine_gradient(parameter) result(gradient)
        real(dp), intent(in) :: parameter

        gradient = (1.0_dp + 2.0_dp + 0.5_dp + 1.5_dp)* &
            (parameter - 1.0_dp)
    end function expected_affine_gradient

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: description
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (*, '(a)') "FAIL "//trim(description)
        end if
    end subroutine check

end program test_pinn
