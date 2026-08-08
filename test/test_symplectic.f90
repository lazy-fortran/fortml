program test_symplectic
    !! Independent harmonic-oscillator/Verlet oracle for symplectic products.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortnum_status, only: fortnum_status_t, status_ok, status_set, &
        FORTNUM_NOT_IMPLEMENTED, FORTNUM_OK, FORTNUM_DOMAIN_ERROR
    use fortml_device, only: FORTML_DEVICE_CUDA
    use fortml_physics_objective, only: physics_constraint_t
    use fortml_symplectic, only: symplectic_form_diagnostic_t, &
        symplectic_constraint_t, symplectic_form_matrix, symplectic_form_residual, &
        symplectic_form_residual_jvp, symplectic_form_residual_vjp, &
        symplectic_form_value, symplectic_form_value_jvp, symplectic_form_value_vjp, &
        symplectic_form_is_symplectic
    implicit none

    type :: verlet_context_t
        real(dp) :: step = 0.0_dp
    end type verlet_context_t

    type(symplectic_form_diagnostic_t) :: diagnostic
    type(symplectic_constraint_t) :: symplectic_constraint
    type(physics_constraint_t) :: physics_constraint
    type(verlet_context_t), target :: map_context
    type(fortnum_status_t) :: status
    real(dp) :: jacobian(2, 2), jacobian_dot(2, 2), residual(4), residual_dot(4)
    real(dp) :: residual_bar(4)
    real(dp) :: jacobian_bar(2, 2), jacobian_bar_fd(2, 2), jacobian_plus(2, 2)
    real(dp) :: jacobian_minus(2, 2), value, value_dot, value_plus, value_minus
    real(dp) :: theta(1), theta_dot(1), theta_bar(1), omega(2, 2)
    real(dp) :: lhs, rhs, h, epsilon
    logical :: yes
    integer :: failures

    failures = 0
    map_context%step = 0.17_dp
    call verlet_jacobian(map_context%step, jacobian)
    call symplectic_form_matrix(1, omega, status)
    call check(status_ok(status) .and. maxval(abs(omega - reshape([0.0_dp, -1.0_dp, &
        1.0_dp, 0.0_dp], [2, 2]))) < 1.0e-15_dp, &
        "canonical form orientation", failures)

    call symplectic_form_residual(jacobian, residual, status)
    call check(status_ok(status) .and. maxval(abs(residual)) < 2.0e-15_dp, &
        "Verlet form residual", failures)
    call symplectic_form_is_symplectic(jacobian, 2.0e-14_dp, yes, status)
    call check(status_ok(status) .and. yes, "Verlet symplectic predicate", failures)

    theta = map_context%step
    theta_dot = [1.0_dp]
    call verlet_jacobian_dot(theta(1), jacobian_dot)
    call symplectic_form_residual_jvp(jacobian, jacobian_dot, residual, residual_dot, &
        status)
    call check(status_ok(status) .and. maxval(abs(residual_dot)) < 2.0e-14_dp, &
        "Verlet residual JVP", failures)
    epsilon = 1.0e-6_dp
    call verlet_jacobian(theta(1) + epsilon, jacobian_plus)
    call verlet_jacobian(theta(1) - epsilon, jacobian_minus)
    call check(maxval(abs(jacobian_dot - (jacobian_plus - jacobian_minus)/ &
        (2.0_dp*epsilon))) < 2.0e-10_dp, "Verlet Jacobian derivative oracle", failures)

    residual_bar = [0.4_dp, -0.7_dp, 0.2_dp, 0.9_dp]
    ! Reuse an arbitrary residual cotangent for the adjoint identity.
    call symplectic_form_residual_vjp(jacobian, residual_bar, jacobian_bar_fd, status)
    call symplectic_form_residual_jvp(jacobian, jacobian_dot, residual, residual_dot, status)
    lhs = sum(residual_bar*residual_dot)
    rhs = sum(jacobian_bar_fd*jacobian_dot)
    call check(status_ok(status) .and. abs(lhs - rhs) < 2.0e-14_dp, &
        "residual JVP/VJP dot identity", failures)

    call symplectic_form_value_jvp(jacobian, jacobian_dot, value, value_dot, status)
    call check(status_ok(status) .and. abs(value) < 2.0e-28_dp .and. &
        abs(value_dot) < 2.0e-14_dp, "Verlet value products", failures)
    call symplectic_form_value_vjp(jacobian, 1.0_dp, jacobian_bar_fd, status)
    call check(status_ok(status) .and. maxval(abs(jacobian_bar_fd)) < 2.0e-14_dp, &
        "Verlet value VJP", failures)

    jacobian(1, 1) = jacobian(1, 1) + 0.03_dp
    call symplectic_form_value(jacobian, value, status)
    call check(status_ok(status) .and. value > 0.0_dp, "non-symplectic defect value", failures)
    call symplectic_form_is_symplectic(jacobian, 1.0e-14_dp, yes, status)
    call check(status_ok(status) .and. .not. yes, "non-symplectic predicate", failures)

    call diagnostic%initialize(1, status)
    call check(status_ok(status) .and. diagnostic%residual_count() == 4, &
        "diagnostic initialization", failures)
    call diagnostic%residual(jacobian, residual, status)
    call check(status_ok(status), "diagnostic residual", failures)
    call diagnostic%value(jacobian, value, status)
    call check(status_ok(status) .and. value > 0.0_dp, "diagnostic value", failures)
    call diagnostic%select_device(FORTML_DEVICE_CUDA, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED .and. diagnostic%device_supported(), &
        "diagnostic CUDA refusal preserves CPU", failures)

    call symplectic_constraint%initialize(1, 1, map_context, verlet_map_jacobian, &
        verlet_map_jvp, verlet_map_vjp, status)
    call check(status_ok(status) .and. symplectic_constraint%initialized(), &
        "constraint initialization", failures)
    theta = [map_context%step]
    call symplectic_constraint%value_jvp(theta, theta_dot, value, value_dot, status)
    call check(status_ok(status) .and. abs(value) < 2.0e-28_dp .and. &
        abs(value_dot) < 2.0e-14_dp, "constraint value JVP", failures)
    call symplectic_constraint%value_vjp(theta, 1.0_dp, theta_bar, status)
    call check(status_ok(status) .and. abs(theta_bar(1)) < 2.0e-14_dp, &
        "constraint value VJP", failures)
    call symplectic_constraint%as_constraint(physics_constraint, status)
    call check(status_ok(status), "physics constraint adapter", failures)
    call physics_constraint%value(theta, value, status)
    call check(status_ok(status) .and. abs(value) < 2.0e-28_dp, &
        "physics constraint value", failures)

    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, " symplectic test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS symplectic residual independent Verlet oracle"

contains

    subroutine check(condition, label, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: label
        integer, intent(inout) :: failures

        if (.not. condition) then
            write (error_unit, '(a)') "FAIL ["//label//"]"
            failures = failures + 1
        end if
    end subroutine check

    subroutine verlet_jacobian(step, jacobian)
        real(dp), intent(in) :: step
        real(dp), intent(out) :: jacobian(2, 2)

        jacobian = reshape([1.0_dp - 0.5_dp*step**2, -step + 0.25_dp*step**3, &
            step, 1.0_dp - 0.5_dp*step**2], [2, 2])
    end subroutine verlet_jacobian

    subroutine verlet_jacobian_dot(step, jacobian_dot)
        real(dp), intent(in) :: step
        real(dp), intent(out) :: jacobian_dot(2, 2)

        jacobian_dot = reshape([-step, -1.0_dp + 0.75_dp*step**2, 1.0_dp, -step], [2, 2])
    end subroutine verlet_jacobian_dot

    subroutine verlet_map_jacobian(context, theta, jacobian, status)
        class(*), pointer, intent(in) :: context
        real(dp), intent(in) :: theta(:)
        real(dp), intent(out) :: jacobian(:, :)
        type(fortnum_status_t), intent(out) :: status

        select type (context)
            type is (verlet_context_t)
                call verlet_jacobian(theta(1) + 0.0_dp*context%step, jacobian)
                call status_set_ok(status)
            class default
                call status_set_bad(status)
        end select
    end subroutine verlet_map_jacobian

    subroutine verlet_map_jvp(context, theta, theta_dot, jacobian, jacobian_dot, status)
        class(*), pointer, intent(in) :: context
        real(dp), intent(in) :: theta(:), theta_dot(:)
        real(dp), intent(out) :: jacobian(:, :), jacobian_dot(:, :)
        type(fortnum_status_t), intent(out) :: status

        select type (context)
            type is (verlet_context_t)
                call verlet_jacobian(theta(1) + 0.0_dp*context%step, jacobian)
                call verlet_jacobian_dot(theta(1), jacobian_dot)
                jacobian_dot = theta_dot(1)*jacobian_dot
                call status_set_ok(status)
            class default
                jacobian = 0.0_dp
                jacobian_dot = 0.0_dp
                call status_set_bad(status)
        end select
    end subroutine verlet_map_jvp

    subroutine verlet_map_vjp(context, theta, jacobian_bar, theta_bar, status)
        class(*), pointer, intent(in) :: context
        real(dp), intent(in) :: theta(:), jacobian_bar(:, :)
        real(dp), intent(out) :: theta_bar(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: jacobian_dot(2, 2)

        select type (context)
            type is (verlet_context_t)
                call verlet_jacobian_dot(theta(1), jacobian_dot)
                theta_bar(1) = sum(jacobian_bar*jacobian_dot)
                call status_set_ok(status)
            class default
                theta_bar = 0.0_dp
                call status_set_bad(status)
        end select
    end subroutine verlet_map_vjp

    subroutine status_set_ok(status)
        type(fortnum_status_t), intent(out) :: status

        call status_set(status, FORTNUM_OK, "")
    end subroutine status_set_ok

    subroutine status_set_bad(status)
        type(fortnum_status_t), intent(out) :: status

        call status_set(status, FORTNUM_DOMAIN_ERROR, "invalid context")
    end subroutine status_set_bad

end program test_symplectic
