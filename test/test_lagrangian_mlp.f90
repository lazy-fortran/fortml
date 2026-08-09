program test_lagrangian_mlp
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_lagrangian_mlp, only: lagrangian_mlp_t
    use fortml_mlp, only: MLP_LINEAR
    use fortml_device, only: FORTML_DEVICE_CUDA
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer :: failures

    failures = 0
    call test_products_and_euler_residual(failures)
    call test_singular_and_device_refusals(failures)
    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, " Lagrangian MLP test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS"

contains

    subroutine test_products_and_euler_residual(failures)
        integer, intent(inout) :: failures
        type(lagrangian_mlp_t) :: model
        type(fortnum_status_t) :: status
        real(dp) :: q(1, 1), v(1, 1), acceleration(1, 1), residual(1, 1)
        real(dp) :: q_plus(1, 1), q_minus(1, 1), v_plus(1, 1), v_minus(1, 1)
        real(dp) :: lagrangian(1), gradient(1, 2), gradient_plus(1, 2), gradient_minus(1, 2)
        real(dp) :: mass(1, 1, 1), mass_fd, lagrangian_plus(1), lagrangian_minus(1)
        real(dp) :: q_direction, v_direction, directional, finite_difference
        real(dp) :: h, expected_residual, fd_gradient_v, fd_gradient_q

        call model%initialize(1, [4, 1], status, initialization_seed=31)
        call check(status_ok(status), "initialize nonlinear Lagrangian", failures)
        q(1, 1) = -0.35_dp
        v(1, 1) = 0.27_dp
        acceleration(1, 1) = 0.11_dp
        call model%lagrangian_gradient(q, v, lagrangian, gradient, status)
        call check(status_ok(status), "value/gradient", failures)
        q_direction = 0.23_dp
        v_direction = -0.19_dp
        directional = gradient(1, 1)*q_direction + gradient(1, 2)*v_direction
        h = 1.0e-6_dp
        q_plus = q
        q_minus = q
        v_plus = v
        v_minus = v
        q_plus(1, 1) = q_plus(1, 1) + h*q_direction
        q_minus(1, 1) = q_minus(1, 1) - h*q_direction
        v_plus(1, 1) = v_plus(1, 1) + h*v_direction
        v_minus(1, 1) = v_minus(1, 1) - h*v_direction
        call model%lagrangian(q_plus, v_plus, lagrangian_plus, status)
        call model%lagrangian(q_minus, v_minus, lagrangian_minus, status)
        finite_difference = (lagrangian_plus(1)-lagrangian_minus(1))/(2.0_dp*h)
        call check(abs(directional-finite_difference) < 3.0e-8_dp, &
            "state-gradient directional finite difference", failures)

        call model%mass_matrix(q, v, mass, status)
        call check(status_ok(status), "nonsingular mass matrix", failures)
        v_plus = v
        v_minus = v
        v_plus(1, 1) = v_plus(1, 1) + h
        v_minus(1, 1) = v_minus(1, 1) - h
        call model%lagrangian_gradient(q, v_plus, lagrangian_plus, gradient_plus, status)
        call model%lagrangian_gradient(q, v_minus, lagrangian_minus, gradient_minus, status)
        mass_fd = (gradient_plus(1, 2)-gradient_minus(1, 2))/(2.0_dp*h)
        call check(abs(mass(1, 1, 1)-mass_fd) < 2.0e-5_dp, &
            "velocity Hessian finite difference", failures)

        call model%euler_lagrange_residual(q, v, acceleration, residual, status)
        call check(status_ok(status), "Euler-Lagrange residual", failures)
        fd_gradient_v = (gradient_plus(1, 2)-gradient_minus(1, 2))/(2.0_dp*h)
        fd_gradient_q = gradient(1, 1)
        q_plus = q
        q_minus = q
        q_plus(1, 1) = q_plus(1, 1) + h
        q_minus(1, 1) = q_minus(1, 1) - h
        call model%lagrangian_gradient(q_plus, v, lagrangian_plus, gradient_plus, status)
        call model%lagrangian_gradient(q_minus, v, lagrangian_minus, gradient_minus, status)
        expected_residual = (gradient_plus(1, 2)-gradient_minus(1, 2))/(2.0_dp*h)*v(1, 1) + &
            fd_gradient_v*acceleration(1, 1) - fd_gradient_q
        ! The fixture is one-dimensional; q and v are evaluated pointwise.
        call check(abs(residual(1, 1)-expected_residual) < 2.0e-5_dp, &
            "Euler-Lagrange residual finite-difference oracle", failures)
        call check(abs(residual(1, 1)) < huge(1.0_dp), &
            "Euler-Lagrange residual is finite", failures)
    end subroutine test_products_and_euler_residual

    subroutine test_singular_and_device_refusals(failures)
        integer, intent(inout) :: failures
        type(lagrangian_mlp_t) :: model
        type(fortnum_status_t) :: status
        real(dp) :: q(1, 1), v(1, 1), mass(1, 1, 1)

        call model%initialize(1, [1], status, hidden_activation=MLP_LINEAR, initialization_seed=9)
        call check(status_ok(status), "initialize linear singular fixture", failures)
        call model%set_parameters([0.0_dp, 0.0_dp, 0.0_dp], status)
        call check(status_ok(status), "set singular fixture parameters", failures)
        q = 0.2_dp
        v = -0.1_dp
        call model%mass_matrix(q, v, mass, status)
        call check(.not. status_ok(status), "singular velocity Hessian refusal", failures)
        call model%select_device(FORTML_DEVICE_CUDA, status)
        call check(.not. status_ok(status), "CUDA refusal", failures)
    end subroutine test_singular_and_device_refusals

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failures
        if (.not. condition) then
            write (error_unit, '(a)') "FAIL ["//trim(description)//"]"
            failures = failures + 1
        end if
    end subroutine check

end program test_lagrangian_mlp
