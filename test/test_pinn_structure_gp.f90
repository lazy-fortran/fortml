program test_pinn_structure_gp
    !! Manufactured-PDE oracle for finite-feature PINN GP initialization.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortnum_status, only: fortnum_status_t, status_ok, status_set, &
        FORTNUM_OK, FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortml_mlp, only: mlp_t, MLP_TANH, MLP_LINEAR
    use fortml_mlp_structure_gp, only: mlp_structure_gp_initializer_t
    use fortml_physics_objective, only: physics_constraint_t, physics_objective_t
    use fortml_pinn_structure_gp, only: pinn_structure_gp_initializer_t, &
        pinn_structure_gp_metadata_t
    implicit none

    type :: probe_context_t
        real(dp) :: probe(10) = 0.0_dp
        real(dp) :: target = 0.0_dp
    end type probe_context_t

    type(probe_context_t), target :: data_context, residual_context
    type(probe_context_t), target :: boundary_context, conservation_context
    type(physics_constraint_t) :: data, residual, boundary, conservation
    type(physics_objective_t) :: objective
    type(mlp_t) :: model
    type(pinn_structure_gp_initializer_t) :: initializer
    type(pinn_structure_gp_metadata_t) :: metadata
    type(fortnum_status_t) :: status
    real(dp) :: x(8, 1), pde_target(8, 1), lambda
    real(dp), allocatable :: hidden(:, :), design(:, :), coefficients(:, :)
    real(dp), allocatable :: before(:), after(:), hidden_before(:), hidden_after(:)
    real(dp), allocatable :: prediction(:, :), cuda_prediction(:, :)
    real(dp) :: expected_terms_before(4), expected_terms_after(4)
    integer :: failures, hidden_count, parameter_count, i

    failures = 0
    lambda = 0.07_dp
    do i = 1, size(x, 1)
        x(i, 1) = -1.0_dp + 2.0_dp*real(i - 1, dp)/real(size(x, 1) - 1, dp)
        pde_target(i, 1) = sin(acos(-1.0_dp)*x(i, 1)) + 0.2_dp*x(i, 1)**2
    end do
    call model%initialize([1, 3, 1], status, hidden_activation=MLP_TANH, &
        output_activation=MLP_LINEAR, initialization_seed=41)
    call check(status_ok(status), "manufactured PINN model initialization", failures)
    parameter_count = model%parameter_count()
    before = model%parameters()
    hidden_count = parameter_count - 3 - 1
    hidden_before = before(1:hidden_count)
    call model%feature_map(x, hidden, status)
    call check(status_ok(status), "manufactured feature map", failures)
    allocate(design(size(x, 1), size(hidden, 2) + 1))
    design(:, 1:size(hidden, 2)) = hidden
    design(:, size(hidden, 2) + 1) = 1.0_dp
    call solve_oracle(design, pde_target, lambda, coefficients)
    data_context%probe = 0.0_dp
    residual_context%probe = 0.0_dp
    boundary_context%probe = 0.0_dp
    conservation_context%probe = 0.0_dp
    data_context%probe(parameter_count) = 1.0_dp
    residual_context%probe(parameter_count) = 1.0_dp
    boundary_context%probe(parameter_count) = 1.0_dp
    conservation_context%probe(parameter_count) = 1.0_dp
    data_context%target = 0.25_dp
    residual_context%target = 0.15_dp
    boundary_context%target = 0.0_dp
    conservation_context%target = -0.10_dp
    call data%initialize(parameter_count, 1, 1.0_dp, data_context, probe_residual, &
        probe_jvp, probe_vjp, status)
    call check(status_ok(status), "data constraint", failures)
    call residual%initialize(parameter_count, 1, 2.0_dp, residual_context, probe_residual, &
        probe_jvp, probe_vjp, status)
    call check(status_ok(status), "PDE residual constraint", failures)
    call boundary%initialize(parameter_count, 1, 0.5_dp, boundary_context, probe_residual, &
        probe_jvp, probe_vjp, status)
    call check(status_ok(status), "boundary constraint", failures)
    call conservation%initialize(parameter_count, 1, 1.5_dp, conservation_context, &
        probe_residual, probe_jvp, probe_vjp, status)
    call check(status_ok(status), "conservation constraint", failures)
    call objective%initialize(parameter_count, data, residual, boundary, conservation, status)
    call check(status_ok(status), "named manufactured-PDE objective", failures)

    call manual_terms(before, expected_terms_before)
    call initializer%fit(model, x, pde_target, objective, status, lambda)
    call check(status_ok(status) .and. initializer%fitted(), &
        "PINN structure GP fit", failures)
    call check(maxval(abs(model%parameters() - before)) == 0.0_dp, &
        "fit is non-mutating", failures)
    metadata = initializer%metadata()
    call check(metadata%sample_count == size(x, 1) .and. &
        metadata%feature_dimension == 3 .and. metadata%output_dimension == 1 .and. &
        metadata%hidden_parameter_count == hidden_count .and. &
        metadata%total_parameter_count == parameter_count .and. &
        metadata%hidden_parameters_frozen .and. metadata%named_terms_preserved .and. &
        .not. metadata%exact_infinite_width .and. .not. metadata%cuda_supported, &
        "PINN structure GP metadata", failures)
    call check(maxval(abs(metadata%term_values_before - expected_terms_before)) < 2.0e-14_dp .and. &
        abs(metadata%residual_term_before - expected_terms_before(2)) < 2.0e-14_dp, &
        "pre-fit named residual diagnostics", failures)
    call check(maxval(abs(initializer%coefficients() - coefficients)) < 3.0e-12_dp, &
        "manufactured dense GP coefficient oracle", failures)

    call initializer%fit(model, x, pde_target, objective, status, lambda)
    call initializer%apply(model, status)
    call check(status_ok(status), "PINN structure GP fit_apply", failures)
    after = model%parameters()
    hidden_after = after(1:hidden_count)
    call check(maxval(abs(hidden_after - hidden_before)) < 1.0e-15_dp, &
        "hidden parameters preserved", failures)
    call model%predict(x, prediction, status)
    call check(status_ok(status) .and. maxval(abs(prediction - matmul(design, coefficients))) < &
        3.0e-12_dp, "applied manufactured posterior oracle", failures)
    call manual_terms(after, expected_terms_after)
    metadata = initializer%metadata()
    call check(maxval(abs(metadata%term_values_after - expected_terms_after)) < 2.0e-14_dp .and. &
        abs(metadata%residual_term_after - expected_terms_after(2)) < 2.0e-14_dp, &
        "post-fit named residual diagnostics", failures)
    allocate(cuda_prediction(size(x, 1), 1))
    cuda_prediction = 77.0_dp
    call initializer%predict_cuda(model, x, cuda_prediction, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED .and. &
        maxval(abs(cuda_prediction - 77.0_dp)) == 0.0_dp, &
        "typed CUDA prediction refusal", failures)
    call initializer%apply_cuda(model, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, "typed CUDA apply refusal", failures)

    after(1) = after(1) + 0.25_dp
    call model%set_parameters(after, status)
    call initializer%apply(model, status)
    call check(status%code == FORTNUM_DOMAIN_ERROR, &
        "hidden-state transaction refusal", failures)

    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, &
            " PINN structure GP test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS PINN structure GP manufactured-PDE oracle"

contains

    subroutine probe_residual(context, theta, residual_value, status)
        class(*), pointer, intent(in) :: context
        real(dp), intent(in) :: theta(:)
        real(dp), intent(out) :: residual_value(:)
        type(fortnum_status_t), intent(out) :: status

        select type (context)
            type is (probe_context_t)
            residual_value(1) = dot_product(context%probe, theta) - context%target
            call status_set(status, FORTNUM_OK, "")
        class default
            residual_value = 0.0_dp
            call status_set(status, FORTNUM_DOMAIN_ERROR, "bad manufactured context")
        end select
    end subroutine probe_residual

    subroutine probe_jvp(context, theta, theta_dot, residual_value, residual_dot, status)
        class(*), pointer, intent(in) :: context
        real(dp), intent(in) :: theta(:), theta_dot(:)
        real(dp), intent(out) :: residual_value(:), residual_dot(:)
        type(fortnum_status_t), intent(out) :: status

        call probe_residual(context, theta, residual_value, status)
        if (.not. status_ok(status)) then
            residual_dot = 0.0_dp
            return
        end if
        select type (context)
            type is (probe_context_t)
            residual_dot(1) = dot_product(context%probe, theta_dot)
            call status_set(status, FORTNUM_OK, "")
        class default
            residual_dot = 0.0_dp
            call status_set(status, FORTNUM_DOMAIN_ERROR, "bad manufactured context")
        end select
    end subroutine probe_jvp

    subroutine probe_vjp(context, theta, residual_bar, theta_bar, status)
        class(*), pointer, intent(in) :: context
        real(dp), intent(in) :: theta(:), residual_bar(:)
        real(dp), intent(out) :: theta_bar(:)
        type(fortnum_status_t), intent(out) :: status

        associate (unused_theta => theta)
        end associate
        select type (context)
            type is (probe_context_t)
            theta_bar = residual_bar(1)*context%probe
            call status_set(status, FORTNUM_OK, "")
        class default
            theta_bar = 0.0_dp
            call status_set(status, FORTNUM_DOMAIN_ERROR, "bad manufactured context")
        end select
    end subroutine probe_vjp

    subroutine manual_terms(theta, values)
        real(dp), intent(in) :: theta(:)
        real(dp), intent(out) :: values(4)

        values(1) = 0.5_dp*(theta(parameter_count) - data_context%target)**2
        values(2) = 2.0_dp*0.5_dp*(theta(parameter_count) - residual_context%target)**2
        values(3) = 0.5_dp*0.5_dp*(theta(parameter_count) - boundary_context%target)**2
        values(4) = 1.5_dp*0.5_dp*(theta(parameter_count) - conservation_context%target)**2
    end subroutine manual_terms

    subroutine solve_oracle(design, target, regularization, coefficients)
        real(dp), intent(in) :: design(:, :), target(:, :), regularization
        real(dp), allocatable, intent(out) :: coefficients(:, :)
        real(dp), allocatable :: matrix(:, :), rhs(:, :)
        real(dp) :: pivot, factor
        integer :: i, j, k, n

        n = size(design, 2)
        allocate(matrix(n, n), rhs(n, size(target, 2)))
        matrix = matmul(transpose(design), design)
        do i = 1, n
            matrix(i, i) = matrix(i, i) + regularization
        end do
        rhs = matmul(transpose(design), target)
        do k = 1, n
            pivot = matrix(k, k)
            do j = k, n
                matrix(k, j) = matrix(k, j)/pivot
            end do
            rhs(k, :) = rhs(k, :)/pivot
            do i = 1, n
                if (i == k) cycle
                factor = matrix(i, k)
                do j = k, n
                    matrix(i, j) = matrix(i, j) - factor*matrix(k, j)
                end do
                rhs(i, :) = rhs(i, :) - factor*rhs(k, :)
            end do
        end do
        allocate(coefficients, source=rhs)
    end subroutine solve_oracle

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: description
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "FAIL ["//trim(description)//"]"
        end if
    end subroutine check

end program test_pinn_structure_gp
