program test_mlp_structure_gp
    !! Independent finite-feature oracle for structure-aware MLP initialization.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_mlp, only: mlp_t, MLP_TANH, MLP_LINEAR
    use fortml_mlp_structure_gp, only: mlp_structure_gp_initializer_t, &
        mlp_structure_gp_metadata_t
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_DOMAIN_ERROR, &
        FORTNUM_NOT_IMPLEMENTED
    implicit none

    integer :: failures

    failures = 0
    call test_structure_contract(failures)
    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, &
            " structure-aware MLP GP test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS"

contains

    subroutine test_structure_contract(failures)
        integer, intent(inout) :: failures
        type(mlp_t) :: model
        type(mlp_structure_gp_initializer_t) :: initializer
        type(mlp_structure_gp_metadata_t) :: metadata
        type(fortnum_status_t) :: status
        real(dp) :: x(6, 2), target(6, 1)
        real(dp), allocatable :: hidden_before(:, :)
        real(dp), allocatable :: design(:, :), expected(:, :), wrapper_actual(:, :)
        real(dp) :: actual(6, 1)
        real(dp), allocatable :: parameters_before(:), parameters_after(:)
        real(dp), allocatable :: coefficients(:, :), snapshot(:)
        real(dp), allocatable :: cuda_prediction(:, :), cuda_variance(:)
        real(dp) :: lambda
        integer :: hidden_count

        x = reshape([ &
            -1.0_dp, 0.1_dp, 0.4_dp, -0.8_dp, 0.7_dp, 1.1_dp, &
             0.2_dp, -0.3_dp, 0.9_dp, 1.4_dp, -1.2_dp, 0.5_dp], shape(x))
        target(:, 1) = [0.3_dp, -0.4_dp, 1.0_dp, 0.2_dp, -0.7_dp, 0.8_dp]
        lambda = 0.07_dp
        call model%initialize([2, 3, 1], status, hidden_activation=MLP_TANH, &
            output_activation=MLP_LINEAR, initialization_seed=31)
        call check(status_ok(status), "model initialization", failures)
        call model%feature_map(x, hidden_before, status)
        call check(status_ok(status), "hidden feature map", failures)
        parameters_before = model%parameters()
        hidden_count = model%parameter_count() - 3*1 - 1

        call initializer%fit(model, x, target, status, lambda)
        call check(status_ok(status) .and. initializer%fitted(), &
            "structure GP fit", failures)
        metadata = initializer%metadata()
        call check(metadata%hidden_parameters_frozen .and. &
            metadata%hidden_parameter_count == hidden_count .and. &
            metadata%total_parameter_count == model%parameter_count() .and. &
            .not. metadata%exact_infinite_width .and. .not. metadata%cuda_supported, &
            "structure metadata", failures)
        snapshot = initializer%hidden_parameters()
        call check(size(snapshot) == hidden_count .and. &
            maxval(abs(snapshot - parameters_before(1:hidden_count))) < 1.0e-15_dp, &
            "hidden parameter snapshot", failures)

        allocate(design(size(x, 1), 4))
        design = 0.0_dp
        design(:, 1:3) = hidden_before
        design(:, 4) = 1.0_dp
        call solve_oracle(design, target, lambda, coefficients)
        call check(maxval(abs(initializer%coefficients() - coefficients)) < 3.0e-12_dp, &
            "finite-feature coefficient oracle", failures)
        call initializer%apply(model, status)
        call check(status_ok(status), "structure GP apply", failures)
        parameters_after = model%parameters()
        call check(maxval(abs(parameters_after(1:hidden_count) - &
            parameters_before(1:hidden_count))) < 1.0e-15_dp, &
            "hidden parameters preserved", failures)
        call model%predict(x, actual, status)
        expected = matmul(design, coefficients)
        call check(status_ok(status) .and. maxval(abs(actual - expected)) < 3.0e-12_dp, &
            "applied posterior oracle", failures)
        call initializer%predict(model, x, wrapper_actual, status)
        call check(status_ok(status) .and. maxval(abs(wrapper_actual - expected)) < 3.0e-12_dp, &
            "validated wrapper prediction", failures)

        parameters_after(1) = parameters_after(1) + 0.2_dp
        call model%set_parameters(parameters_after, status)
        call initializer%apply(model, status)
        call check(status%code == FORTNUM_DOMAIN_ERROR, &
            "hidden mutation refusal", failures)
        call initializer%apply_cuda(model, status)
        call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
            "CUDA structure refusal", failures)
        allocate(cuda_prediction(6, 1), cuda_variance(6))
        cuda_prediction = 77.0_dp
        cuda_variance = 77.0_dp
        call initializer%predict_cuda(model, x, cuda_prediction, status)
        call check(status%code == FORTNUM_NOT_IMPLEMENTED .and. &
            maxval(abs(cuda_prediction - 77.0_dp)) == 0.0_dp, &
            "CUDA prediction refusal and no mutation", failures)
        call initializer%predictive_variance_cuda(model, x, cuda_variance, status)
        call check(status%code == FORTNUM_NOT_IMPLEMENTED .and. &
            maxval(abs(cuda_variance - 77.0_dp)) == 0.0_dp, &
            "CUDA variance refusal and no mutation", failures)
    end subroutine test_structure_contract

    subroutine solve_oracle(design, target, lambda, coefficients)
        real(dp), intent(in) :: design(:, :), target(:, :), lambda
        real(dp), allocatable, intent(out) :: coefficients(:, :)
        real(dp), allocatable :: matrix(:, :), rhs(:, :)
        real(dp) :: pivot, factor
        integer :: n, i, j, k

        n = size(design, 2)
        allocate(matrix(n, n), rhs(n, size(target, 2)))
        matrix = matmul(transpose(design), design)
        do i = 1, n
            matrix(i, i) = matrix(i, i) + lambda
        end do
        rhs = matmul(transpose(design), target)
        do k = 1, n
            pivot = matrix(k, k)
            matrix(k, k:n) = matrix(k, k:n)/pivot
            rhs(k, :) = rhs(k, :)/pivot
            do i = 1, n
                if (i == k) cycle
                factor = matrix(i, k)
                matrix(i, k:n) = matrix(i, k:n) - factor*matrix(k, k:n)
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

end program test_mlp_structure_gp
