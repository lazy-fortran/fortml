program fortml_bench_pinn_term_products
    !! Release executable for named PINN gradient/HVP diagnostics.
    use, intrinsic :: iso_fortran_env, only: output_unit
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, status_ok, &
        FORTNUM_OK, FORTNUM_DOMAIN_ERROR
    use fortml_physics_objective, only: physics_constraint_t, &
        physics_objective_t
    use fortml_pinn, only: pinn_training_adapter_t
    implicit none

    type :: context_t
        real(dp) :: offset = 0.25_dp
    end type context_t
    type(context_t), target :: context
    type(physics_constraint_t) :: constraint
    type(physics_objective_t) :: objective
    type(pinn_training_adapter_t) :: pinn
    type(fortnum_status_t) :: status
    real(dp) :: theta(1), direction(1), gradients(1,4), hvps(1,4)

    theta = [0.7_dp]
    direction = [-0.23_dp]
    call constraint%initialize(1, 1, 2.0_dp, context, residual, jvp, vjp, &
        status, hvp)
    if (.not. status_ok(status)) error stop 1
    call objective%initialize(1, residual=constraint, status=status)
    if (.not. status_ok(status)) error stop 1
    call pinn%initialize(objective, status)
    if (.not. status_ok(status)) error stop 1
    call pinn%term_gradients(theta, gradients, status)
    if (.not. status_ok(status)) error stop 1
    call pinn%term_hvps(theta, direction, hvps, status)
    if (.not. status_ok(status)) error stop 1
    write (output_unit, '(a)') &
        "workload,theta,direction,gradient_residual,hvp_residual,gradient_sum,hvp_sum,device,status"
    write (output_unit, '(a,",",es24.16,",",es24.16,4(",",es24.16),",",a)') &
        "quadratic_residual", theta(1), direction(1), gradients(1,2), &
        hvps(1,2), sum(gradients), sum(hvps), "cpu,pass"

contains

    subroutine residual(context_ptr, theta, residual_value, status)
        class(*), pointer, intent(in) :: context_ptr
        real(dp), intent(in) :: theta(:)
        real(dp), intent(out) :: residual_value(:)
        type(fortnum_status_t), intent(out) :: status

        select type (context_ptr)
            type is (context_t)
            residual_value(1) = theta(1)**2 - context_ptr%offset
            call status_set(status, FORTNUM_OK, "")
        class default
            residual_value = 0.0_dp
            call status_set(status, FORTNUM_DOMAIN_ERROR, "bad context")
        end select
    end subroutine residual

    subroutine jvp(context_ptr, theta, theta_dot, residual_value, residual_dot, status)
        class(*), pointer, intent(in) :: context_ptr
        real(dp), intent(in) :: theta(:), theta_dot(:)
        real(dp), intent(out) :: residual_value(:), residual_dot(:)
        type(fortnum_status_t), intent(out) :: status

        call residual(context_ptr, theta, residual_value, status)
        if (.not. status_ok(status)) then
            residual_dot = 0.0_dp
            return
        end if
        residual_dot(1) = 2.0_dp*theta(1)*theta_dot(1)
        call status_set(status, FORTNUM_OK, "")
    end subroutine jvp

    subroutine vjp(context_ptr, theta, residual_bar, theta_bar, status)
        class(*), pointer, intent(in) :: context_ptr
        real(dp), intent(in) :: theta(:), residual_bar(:)
        real(dp), intent(out) :: theta_bar(:)
        type(fortnum_status_t), intent(out) :: status

        associate (unused_context => context_ptr)
        end associate
        theta_bar(1) = 2.0_dp*theta(1)*residual_bar(1)
        call status_set(status, FORTNUM_OK, "")
    end subroutine vjp

    subroutine hvp(context_ptr, theta, theta_dot, residual_bar, residual_bar_dot, &
            theta_hvp, status)
        class(*), pointer, intent(in) :: context_ptr
        real(dp), intent(in) :: theta(:), theta_dot(:)
        real(dp), intent(in) :: residual_bar(:), residual_bar_dot(:)
        real(dp), intent(out) :: theta_hvp(:)
        type(fortnum_status_t), intent(out) :: status

        associate (unused_context => context_ptr)
        end associate
        theta_hvp(1) = 2.0_dp*theta_dot(1)*residual_bar(1) + &
            2.0_dp*theta(1)*residual_bar_dot(1)
        call status_set(status, FORTNUM_OK, "")
    end subroutine hvp

end program fortml_bench_pinn_term_products
