program test_hamiltonian_mlp
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_quiet_nan
    use fortml_hamiltonian_mlp, only: hamiltonian_mlp_t
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    implicit none

    integer :: failures

    failures = 0
    call test_energy_products(failures)
    call test_vector_field_products(failures)
    call test_general_products(failures)
    call test_symplectic_leapfrog(failures)
    call test_refusal(failures)
    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, " Hamiltonian MLP test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS"

contains

    subroutine make_model(model, status)
        type(hamiltonian_mlp_t), intent(out) :: model
        type(fortnum_status_t), intent(out) :: status

        call model%initialize(1, [1, 2, 1], [1, 2, 1], status, initialization_seed=23)
    end subroutine make_model

    subroutine test_energy_products(failures)
        integer, intent(inout) :: failures
        type(hamiltonian_mlp_t) :: model
        type(fortnum_status_t) :: status
        real(dp) :: state(1, 2), dstate(1, 2), energy(1, 1), denergy(1, 1)
        real(dp) :: energy_plus(1, 1), energy_minus(1, 1), theta_plus(14), theta_minus(14)
        real(dp) :: theta(14), dtheta(14), parameter_bar(14), state_bar(1, 2)
        real(dp) :: lhs, rhs, h
        integer :: i

        call make_model(model, status)
        theta = model%parameters()
        dtheta = [(0.013_dp*real(i, dp), i=1, size(theta))]
        state = reshape([0.23_dp, -0.41_dp], shape(state))
        dstate = reshape([0.17_dp, -0.11_dp], shape(dstate))
        call model%energy_jvp(state, dtheta, dstate, energy, denergy, status)
        h = 1.0e-6_dp
        theta_plus = theta + h*dtheta
        theta_minus = theta - h*dtheta
        call model%set_parameters(theta_plus, status)
        call model%energy(state + h*dstate, energy_plus, status)
        call model%set_parameters(theta_minus, status)
        call model%energy(state - h*dstate, energy_minus, status)
        call model%set_parameters(theta, status)
        if (.not. status_ok(status) .or. &
            maxval(abs(denergy - (energy_plus - energy_minus)/(2.0_dp*h))) > 3.0e-8_dp) then
            write (error_unit, '(a,es12.4)') &
                "FAIL [hamiltonian-energy-jvp] finite difference=", &
                maxval(abs(denergy - (energy_plus - energy_minus)/(2.0_dp*h)))
            failures = failures + 1
        end if

        call model%energy_vjp(state, [0.7_dp], parameter_bar, state_bar, status)
        lhs = 0.7_dp*denergy(1, 1)
        rhs = sum(parameter_bar*dtheta) + sum(state_bar*dstate)
        if (.not. status_ok(status) .or. abs(lhs - rhs) > 5.0e-11_dp) then
            write (error_unit, '(a,es12.4)') &
                "FAIL [hamiltonian-energy-vjp] adjoint identity=", abs(lhs - rhs)
            failures = failures + 1
        end if
    end subroutine test_energy_products

    subroutine test_vector_field_products(failures)
        integer, intent(inout) :: failures
        type(hamiltonian_mlp_t) :: model
        type(fortnum_status_t) :: status
        real(dp) :: state(1, 2), dstate(1, 2), field(1, 2), dfield(1, 2)
        real(dp) :: field_plus(1, 2), field_minus(1, 2), theta(14), theta_plus(14)
        real(dp) :: theta_minus(14), dtheta(14), h
        integer :: i

        call make_model(model, status)
        theta = model%parameters()
        dtheta = [(0.009_dp*real(i, dp), i=1, size(theta))]
        state = reshape([0.31_dp, -0.27_dp], shape(state))
        dstate = reshape([-0.08_dp, 0.12_dp], shape(dstate))
        call model%vector_field_jvp(state, dtheta, dstate, field, dfield, status)
        h = 1.0e-6_dp
        theta_plus = theta + h*dtheta
        theta_minus = theta - h*dtheta
        call model%set_parameters(theta_plus, status)
        call model%vector_field(state + h*dstate, field_plus, status)
        call model%set_parameters(theta_minus, status)
        call model%vector_field(state - h*dstate, field_minus, status)
        call model%set_parameters(theta, status)
        if (.not. status_ok(status) .or. &
            maxval(abs(dfield - (field_plus - field_minus)/(2.0_dp*h))) > 2.0e-7_dp) then
            write (error_unit, '(a,es12.4)') &
                "FAIL [hamiltonian-field-jvp] finite difference=", &
                maxval(abs(dfield - (field_plus - field_minus)/(2.0_dp*h)))
            failures = failures + 1
        end if
    end subroutine test_vector_field_products

    subroutine test_general_products(failures)
        integer, intent(inout) :: failures
        type(hamiltonian_mlp_t) :: model
        type(fortnum_status_t) :: status
        real(dp), allocatable :: state(:, :), dstate(:, :), energy(:, :), denergy(:, :)
        real(dp), allocatable :: energy_plus(:, :), energy_minus(:, :)
        real(dp), allocatable :: field(:, :), dfield(:, :), field_plus(:, :), field_minus(:, :)
        real(dp), allocatable :: theta(:), theta_plus(:), theta_minus(:), dtheta(:)
        real(dp), allocatable :: parameter_bar(:), state_bar(:, :)
        real(dp) :: energy_bar(2), lhs, rhs, h, error
        integer :: i

        call model%initialize_general(1, [2, 4, 1], status, initialization_seed=41)
        if (.not. status_ok(status) .or. .not. model%is_general()) then
            write (error_unit, '(a)') "FAIL [hamiltonian-general] initialization"
            failures = failures + 1
            return
        end if
        theta = model%parameters()
        allocate(dtheta(size(theta)), theta_plus(size(theta)), theta_minus(size(theta)))
        dtheta = [(0.007_dp*real(i, dp), i=1, size(theta))]
        allocate(state(2, 2), dstate(2, 2), energy(2, 1), denergy(2, 1))
        allocate(energy_plus(2, 1), energy_minus(2, 1))
        state = reshape([0.23_dp, -0.41_dp, -0.17_dp, 0.29_dp], shape(state))
        dstate = reshape([0.13_dp, -0.08_dp, 0.04_dp, 0.11_dp], shape(dstate))
        call model%energy_jvp(state, dtheta, dstate, energy, denergy, status)
        h = 1.0e-6_dp
        theta_plus = theta + h*dtheta
        theta_minus = theta - h*dtheta
        call model%set_parameters(theta_plus, status)
        call model%energy(state + h*dstate, energy_plus, status)
        call model%set_parameters(theta_minus, status)
        call model%energy(state - h*dstate, energy_minus, status)
        call model%set_parameters(theta, status)
        error = maxval(abs(denergy - (energy_plus - energy_minus)/(2.0_dp*h)))
        if (.not. status_ok(status) .or. error > 3.0e-8_dp) then
            write (error_unit, '(a,es12.4)') &
                "FAIL [hamiltonian-general-energy-jvp] finite difference=", error
            failures = failures + 1
        end if

        allocate(parameter_bar(size(theta)), state_bar(2, 2))
        energy_bar = [0.7_dp, -0.4_dp]
        call model%energy_vjp(state, energy_bar, parameter_bar, state_bar, status)
        lhs = sum(energy_bar*denergy(:, 1))
        rhs = sum(parameter_bar*dtheta) + sum(state_bar*dstate)
        if (.not. status_ok(status) .or. abs(lhs - rhs) > 5.0e-10_dp) then
            write (error_unit, '(a,es12.4)') &
                "FAIL [hamiltonian-general-energy-vjp] adjoint identity=", abs(lhs-rhs)
            failures = failures + 1
        end if

        allocate(field(2, 2), dfield(2, 2), field_plus(2, 2), field_minus(2, 2))
        call model%vector_field_jvp(state, dtheta, dstate, field, dfield, status)
        call model%set_parameters(theta_plus, status)
        call model%vector_field(state + h*dstate, field_plus, status)
        call model%set_parameters(theta_minus, status)
        call model%vector_field(state - h*dstate, field_minus, status)
        call model%set_parameters(theta, status)
        error = maxval(abs(dfield - (field_plus - field_minus)/(2.0_dp*h)))
        if (.not. status_ok(status) .or. error > 3.0e-7_dp) then
            write (error_unit, '(a,es12.4)') &
                "FAIL [hamiltonian-general-field-jvp] finite difference=", error
            failures = failures + 1
        end if
    end subroutine test_general_products

    subroutine test_symplectic_leapfrog(failures)
        integer, intent(inout) :: failures
        type(hamiltonian_mlp_t) :: model
        type(fortnum_status_t) :: status
        real(dp) :: state(1, 2), state_plus(1, 2), state_minus(1, 2)
        real(dp) :: mapped(1, 2), forward(1, 2), recovered(1, 2)
        real(dp) :: jacobian(2, 2), omega(2, 2), defect(2, 2), h, step
        integer :: j

        call make_model(model, status)
        state = reshape([0.18_dp, -0.36_dp], shape(state))
        step = 0.07_dp
        call model%leapfrog(state, step, mapped, status)
        h = 1.0e-6_dp
        do j = 1, 2
            state_plus = state
            state_minus = state
            state_plus(1, j) = state_plus(1, j) + h
            state_minus(1, j) = state_minus(1, j) - h
            call model%leapfrog(state_plus, step, forward, status)
            call model%leapfrog(state_minus, step, recovered, status)
            jacobian(1, j) = (forward(1, 1) - recovered(1, 1))/(2.0_dp*h)
            jacobian(2, j) = (forward(1, 2) - recovered(1, 2))/(2.0_dp*h)
        end do
        omega = reshape([0.0_dp, -1.0_dp, 1.0_dp, 0.0_dp], shape(omega))
        defect = matmul(transpose(jacobian), matmul(omega, jacobian)) - omega
        if (.not. status_ok(status) .or. maxval(abs(defect)) > 3.0e-6_dp) then
            write (error_unit, '(a,es12.4)') &
                "FAIL [hamiltonian-leapfrog] symplectic defect=", maxval(abs(defect))
            failures = failures + 1
        end if

        call model%leapfrog(mapped, -step, recovered, status)
        if (.not. status_ok(status) .or. maxval(abs(recovered - state)) > 2.0e-10_dp) then
            write (error_unit, '(a,es12.4)') &
                "FAIL [hamiltonian-leapfrog] reversibility=", maxval(abs(recovered - state))
            failures = failures + 1
        end if
    end subroutine test_symplectic_leapfrog

    subroutine test_refusal(failures)
        integer, intent(inout) :: failures
        type(hamiltonian_mlp_t) :: model
        type(fortnum_status_t) :: status
        real(dp) :: state(1, 2), energy(1, 1)
        real(dp) :: next_state(1, 2)

        call model%initialize(1, [2, 1], [1, 1], status)
        if (status_ok(status)) then
            write (error_unit, '(a)') "FAIL [hamiltonian-refusal] bad layers accepted"
            failures = failures + 1
        end if
        call make_model(model, status)
        state = reshape([0.2_dp, 0.1_dp], shape(state))
        state(1, 2) = ieee_value(state(1, 2), ieee_quiet_nan)
        call model%energy(state, energy, status)
        if (status_ok(status)) then
            write (error_unit, '(a)') "FAIL [hamiltonian-refusal] nonfinite state accepted"
            failures = failures + 1
        end if
        call model%initialize_general(1, [2, 3, 1], status, initialization_seed=29)
        state = reshape([0.2_dp, 0.1_dp], shape(state))
        call model%leapfrog(state, 0.1_dp, next_state, status)
        if (status%code /= FORTNUM_NOT_IMPLEMENTED) then
            write (error_unit, '(a)') &
                "FAIL [hamiltonian-refusal] general leapfrog was not refused"
            failures = failures + 1
        end if
    end subroutine test_refusal

end program test_hamiltonian_mlp
