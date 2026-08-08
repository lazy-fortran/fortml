program test_mlp_last_layer_gp
    !! Independent finite-feature kernel-ridge oracle for the MLP initializer.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_mlp, only: mlp_t, MLP_TANH, MLP_LINEAR
    use fortml_mlp_last_layer_gp, only: mlp_last_layer_gp_initializer_t, &
        mlp_last_layer_gp_metadata_t, mlp_last_layer_parameter_t
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_DOMAIN_ERROR, &
        FORTNUM_NOT_IMPLEMENTED
    implicit none

    integer :: failures

    failures = 0
    call test_closed_form_and_products(failures)
    call test_refusals_and_cuda_boundary(failures)
    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, &
            " last-layer GP initializer test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS"

contains

    subroutine test_closed_form_and_products(failures)
        integer, intent(inout) :: failures
        type(mlp_t) :: model
        type(mlp_last_layer_gp_initializer_t) :: initializer, plus, minus
        type(fortnum_status_t) :: status
        real(dp) :: x(5, 2), target(5, 2), design(5, 3)
        real(dp), allocatable :: h(:, :)
        real(dp), allocatable :: expected(:, :), actual(:, :), derivative(:, :)
        real(dp), allocatable :: coefficients(:, :), parameters(:)
        real(dp), allocatable :: central(:, :)
        real(dp) :: theta(12), lambda, epsilon
        integer :: i
        type(mlp_last_layer_gp_initializer_t) :: metadata_owner
        type(mlp_last_layer_gp_metadata_t) :: metadata
        type(mlp_last_layer_parameter_t), allocatable :: parameter_metadata(:)

        x = reshape([ &
            -1.0_dp, 0.2_dp, 0.4_dp, -0.8_dp, 0.7_dp, &
             1.1_dp, -0.3_dp, 0.9_dp, 1.4_dp, -1.2_dp], shape(x))
        target = reshape([ &
            0.3_dp, -0.4_dp, 1.0_dp, 0.2_dp, -0.7_dp, &
            0.8_dp, 0.5_dp, -0.1_dp, 1.2_dp, -0.9_dp], shape(target))
        theta = [ &
            0.7_dp, -0.2_dp, 0.4_dp, 0.5_dp, &
            0.1_dp, -0.3_dp, &
            0.2_dp, -0.6_dp, 0.3_dp, 0.8_dp, &
            0.05_dp, -0.15_dp]
        call model%initialize([2, 2, 2], status, hidden_activation=MLP_TANH, &
            output_activation=MLP_LINEAR, initialization_seed=13)
        call check(status_ok(status), "model initialization", failures)
        call model%set_parameters(theta, status)
        call check(status_ok(status), "model parameter setup", failures)
        call model%feature_map(x, h, status)
        call check(status_ok(status), "feature map", failures)
        design(:, 1:2) = h
        design(:, 3) = 1.0_dp
        lambda = 0.7_dp

        call solve_oracle(design, target, lambda, coefficients)
        call initializer%fit(model, x, target, status, lambda)
        call check(status_ok(status), "initializer fit", failures)
        call check(initializer%fitted(), "fitted flag", failures)
        call check(initializer%sample_count() == 5 .and. &
            initializer%feature_dimension() == 2 .and. &
            initializer%output_dimension() == 2, "metadata dimensions", failures)
        call check(abs(initializer%regularization() - lambda) < 1.0e-15_dp, &
            "regularization metadata", failures)
        metadata = initializer%metadata()
        call check(.not. metadata%exact_infinite_width .and. &
            .not. metadata%cuda_supported, &
            "approximation metadata", failures)
        call check(maxval(abs(initializer%coefficients() - coefficients)) < 2.0e-13_dp, &
            "closed-form coefficient oracle", failures)

        call initializer%predict(model, x, actual, status)
        expected = matmul(design, coefficients)
        call check(status_ok(status) .and. maxval(abs(actual - expected)) < 2.0e-13_dp, &
            "posterior mean oracle", failures)
        call initializer%apply(model, status)
        call model%predict(x, actual, status)
        call check(status_ok(status) .and. maxval(abs(actual - expected)) < 2.0e-13_dp, &
            "applied final-layer oracle", failures)

        epsilon = 1.0e-5_dp
        call plus%fit(model, x, target, status, lambda + epsilon)
        call minus%fit(model, x, target, status, lambda - epsilon)
        call plus%predict(model, x, actual, status)
        call minus%predict(model, x, central, status)
        central = (actual - central)/(2.0_dp*epsilon)
        call initializer%jvp(model, x, 1.0_dp, actual, derivative, status)
        call check(status_ok(status) .and. maxval(abs(derivative - central)) < 2.0e-9_dp, &
            "regularization JVP oracle", failures)

        parameters = initializer%parameters()
        call check(size(parameters) == 1 .and. abs(parameters(1) - lambda) < 1.0e-15_dp, &
            "packed hyperparameter", failures)
        parameter_metadata = initializer%parameter_metadata()
        call check(trim(parameter_metadata(1)%name) == "regularization", &
            "hyperparameter metadata name", failures)
        metadata_owner = initializer
        i = metadata_owner%parameter_count()
        call check(i == 1, "hyperparameter count", failures)
    end subroutine test_closed_form_and_products

    subroutine test_refusals_and_cuda_boundary(failures)
        integer, intent(inout) :: failures
        type(mlp_t) :: model
        type(mlp_last_layer_gp_initializer_t) :: initializer
        type(fortnum_status_t) :: status
        real(dp) :: x(3, 2), target(3, 1)
        real(dp), allocatable :: before(:, :), sentinel(:, :)
        real(dp) :: parameters(1)

        x = 0.2_dp
        target = -0.4_dp
        call model%initialize([2, 2, 1], status, hidden_activation=MLP_TANH, &
            output_activation=MLP_LINEAR)
        call initializer%fit(model, x, target, status, 0.4_dp)
        call check(status_ok(status), "refusal fixture fit", failures)
        before = initializer%coefficients()
        parameters = [-1.0_dp]
        call initializer%set_parameters(parameters, status)
        call check(status%code == FORTNUM_DOMAIN_ERROR, &
            "invalid hyperparameter refusal", failures)
        call check(maxval(abs(initializer%coefficients() - before)) < 1.0e-15_dp, &
            "invalid hyperparameter transaction", failures)

        allocate(sentinel(3, 1))
        sentinel = 77.0_dp
        call initializer%predict_cuda(model, x, sentinel, status)
        call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
            "CUDA prediction refusal", failures)
        call check(maxval(abs(sentinel - 77.0_dp)) == 0.0_dp, &
            "CUDA prediction no mutation", failures)
        call initializer%apply_cuda(model, status)
        call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
            "CUDA apply refusal", failures)
    end subroutine test_refusals_and_cuda_boundary

    subroutine solve_oracle(design, target, lambda, coefficients)
        real(dp), intent(in) :: design(:, :), target(:, :), lambda
        real(dp), allocatable, intent(out) :: coefficients(:, :)
        real(dp), allocatable :: matrix(:, :), rhs(:, :)
        integer :: i, j, k, n
        real(dp) :: pivot, factor

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

end program test_mlp_last_layer_gp
