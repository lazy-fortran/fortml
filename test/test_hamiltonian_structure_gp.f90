program test_hamiltonian_structure_gp
    !! Independent finite-feature oracle for separable Hamiltonian GP starts.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_hamiltonian_mlp, only: hamiltonian_mlp_t
    use fortml_hamiltonian_structure_gp, only: &
        hamiltonian_structure_gp_initializer_t, hamiltonian_structure_gp_metadata_t
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_DOMAIN_ERROR, &
        FORTNUM_NOT_IMPLEMENTED
    use fortml_mlp, only: mlp_t
    implicit none

    integer :: failures

    failures = 0
    call test_separable_oracle(failures)
    call test_structure_and_refusals(failures)
    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, &
            " Hamiltonian structure GP test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS"

contains

    subroutine test_separable_oracle(failures)
        integer, intent(inout) :: failures
        type(hamiltonian_mlp_t) :: model
        type(hamiltonian_structure_gp_initializer_t) :: initializer
        type(hamiltonian_structure_gp_metadata_t) :: metadata
        type(fortnum_status_t) :: status
        real(dp) :: q(5, 2), p(4, 2), potential_target(5, 1), kinetic_target(4, 1)
        type(mlp_t) :: q_model, p_model
        real(dp), allocatable :: design_q(:, :), design_p(:, :)
        real(dp), allocatable :: q_model_features(:, :), p_model_features(:, :)
        real(dp), allocatable :: expected_q(:, :), expected_p(:, :), actual_q(:, :), actual_p(:, :)
        real(dp), allocatable :: coefficients_q(:, :), coefficients_p(:, :)
        real(dp), allocatable :: coefficients_q_actual(:, :), coefficients_p_actual(:, :)
        real(dp), allocatable :: actual_energy(:, :), expected_energy(:, :)
        real(dp), allocatable :: before(:), after(:)
        real(dp) :: lambda
        integer :: np, nk, hp, hk

        q = reshape([ -1.0_dp, 0.2_dp, 0.4_dp, -0.8_dp, 0.7_dp, 1.1_dp, &
            -0.3_dp, 0.9_dp, 1.4_dp, -1.2_dp ], shape(q))
        p = reshape([ 0.1_dp, -0.5_dp, 0.8_dp, 0.4_dp, -0.7_dp, 1.2_dp, &
            0.3_dp, -0.2_dp ], shape(p))
        potential_target(:, 1) = [0.3_dp, -0.4_dp, 1.0_dp, 0.2_dp, -0.7_dp]
        kinetic_target(:, 1) = [0.8_dp, 0.5_dp, -0.1_dp, 1.2_dp]
        lambda = 0.7_dp
        call model%initialize(2, [2, 3, 1], [2, 3, 1], status, initialization_seed=19)
        call check(status_ok(status), "separable model initialization", failures)
        before = model%parameters()
        call initializer%fit(model, q, potential_target, p, kinetic_target, status, lambda)
        call check(status_ok(status) .and. initializer%fitted(), "initializer fit", failures)
        after = model%parameters()
        call check(maxval(abs(after - before)) == 0.0_dp, &
            "fit is non-mutating", failures)

        q_model = model%potential_model()
        p_model = model%kinetic_model()
        call q_model%feature_map(q, q_model_features, status)
        call check(status_ok(status), "potential feature map", failures)
        call p_model%feature_map(p, p_model_features, status)
        call check(status_ok(status), "kinetic feature map", failures)
        allocate(design_q(size(q, 1), size(q_model_features, 2) + 1))
        allocate(design_p(size(p, 1), size(p_model_features, 2) + 1))
        design_q(:, 1:size(q_model_features, 2)) = q_model_features
        design_q(:, size(q_model_features, 2) + 1) = 1.0_dp
        design_p(:, 1:size(p_model_features, 2)) = p_model_features
        design_p(:, size(p_model_features, 2) + 1) = 1.0_dp
        call solve_oracle(design_q, potential_target, lambda, coefficients_q)
        call solve_oracle(design_p, kinetic_target, lambda, coefficients_p)
        coefficients_q_actual = initializer%potential_coefficients()
        call check(maxval(abs(coefficients_q_actual - coefficients_q)) < 2.0e-12_dp, &
            "potential coefficient oracle", failures)
        coefficients_p_actual = initializer%kinetic_coefficients()
        call check(maxval(abs(coefficients_p_actual - coefficients_p)) < 2.0e-12_dp, &
            "kinetic coefficient oracle", failures)

        call initializer%predict_components(model, q, p, actual_q, actual_p, status)
        expected_q = matmul(design_q, coefficients_q)
        expected_p = matmul(design_p, coefficients_p)
        call check(status_ok(status) .and. maxval(abs(actual_q - expected_q)) < 2.0e-12_dp .and. &
            maxval(abs(actual_p - expected_p)) < 2.0e-12_dp, &
            "component posterior oracle", failures)

        call initializer%fit_apply(model, q, potential_target, p, kinetic_target, status, lambda)
        call check(status_ok(status), "fit_apply", failures)
        after = model%parameters()
        np = model%potential_parameter_count()
        nk = model%kinetic_parameter_count()
        hp = np - size(q_model_features, 2) - 1
        hk = nk - size(p_model_features, 2) - 1
        call check(maxval(abs(after(1:hp) - before(1:hp))) == 0.0_dp .and. &
            maxval(abs(after(np + 1:np + hk) - before(np + 1:np + hk))) == 0.0_dp, &
            "hidden state frozen", failures)
        allocate(actual_energy(4, 1), expected_energy(4, 1))
        call model%energy(assemble_state(q(1:4, :), p), actual_energy, status)
        expected_energy = matmul(design_q(1:4, :), coefficients_q) + &
            matmul(design_p, coefficients_p)
        call check(status_ok(status) .and. maxval(abs(actual_energy - expected_energy)) < 2.0e-12_dp, &
            "applied Hamiltonian energy oracle", failures)
        metadata = initializer%metadata()
        call check(abs(metadata%structure_defect) == 0.0_dp .and. &
            metadata%separability_preserved .and. metadata%hidden_parameters_frozen .and. &
            .not. metadata%exact_infinite_width .and. .not. metadata%cuda_supported, &
            "structure metadata", failures)
    end subroutine test_separable_oracle

    subroutine test_structure_and_refusals(failures)
        integer, intent(inout) :: failures
        type(hamiltonian_mlp_t) :: model, general
        type(hamiltonian_structure_gp_initializer_t) :: initializer
        type(fortnum_status_t) :: status
        real(dp) :: q(2, 2), p(2, 2), target(2, 1), sentinel(2, 1)
        real(dp), allocatable :: before(:), changed(:), potential(:,:), kinetic(:,:)

        q = 0.2_dp
        p = -0.4_dp
        target = 0.5_dp
        call model%initialize(2, [2, 2, 1], [2, 2, 1], status)
        call initializer%fit(model, q, target, p, target, status, 0.3_dp)
        call check(status_ok(status), "refusal fixture fit", failures)
        before = model%parameters()
        changed = before
        changed(1) = changed(1) + 0.1_dp
        call model%set_parameters(changed, status)
        call initializer%apply(model, status)
        call check(status%code == FORTNUM_DOMAIN_ERROR, "hidden mutation refusal", failures)
        call model%set_parameters(before, status)
        sentinel = 77.0_dp
        call initializer%predict_cuda(model, q, p, potential, kinetic, status)
        call check(status%code == FORTNUM_NOT_IMPLEMENTED, "CUDA prediction refusal", failures)
        call initializer%apply_cuda(model, status)
        call check(status%code == FORTNUM_NOT_IMPLEMENTED, "CUDA apply refusal", failures)
        call general%initialize_general(2, [4, 2, 1], status)
        call initializer%fit(general, q, target, p, target, status)
        call check(status%code == FORTNUM_DOMAIN_ERROR, "general Hamiltonian refusal", failures)
    end subroutine test_structure_and_refusals

    function assemble_state(q, p) result(state)
        real(dp), intent(in) :: q(:, :), p(:, :)
        real(dp) :: state(size(q, 1), size(q, 2) + size(p, 2))

        state(:, 1:size(q, 2)) = q
        state(:, size(q, 2) + 1:) = p
    end function assemble_state

    subroutine solve_oracle(design, target, lambda, coefficients)
        real(dp), intent(in) :: design(:, :), target(:, :), lambda
        real(dp), allocatable, intent(out) :: coefficients(:, :)
        real(dp), allocatable :: matrix(:, :), rhs(:, :)
        real(dp) :: pivot, factor
        integer :: i, j, k, n

        n = size(design, 2)
        allocate(matrix(n, n), rhs(n, size(target, 2)))
        matrix = matmul(transpose(design), design)
        do i = 1, n
            matrix(i, i) = matrix(i, i) + lambda
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

    subroutine check(condition, label, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: label
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "FAIL [" // trim(label) // "]"
        end if
    end subroutine check

end program test_hamiltonian_structure_gp
