program test_multi_output_gp
    !! Oracles for the coregionalized multi-output GP and the inference policy.
    !!
    !! The multi-output oracles are two independent identities. With a diagonal
    !! coregionalization matrix the outputs decouple, so each output's
    !! posterior mean must equal a separate single-output GP fit computed here
    !! by dense linear algebra. With a coupled matrix the joint system is
    !! solved densely in the test and compared entry by entry, which also
    !! catches an index-order mistake in the Kronecker layout.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_kernels, only: kernel_t, make_rbf_kernel
    use fortml_multi_output_gp, only: multi_output_gp_t
    use fortml_inference_policy, only: inference_problem_t, inference_choice_t, &
        select_inference_policy, inference_policy_name, &
        INFERENCE_EXACT_CHOLESKY, INFERENCE_STRUCTURED_GRID, &
        INFERENCE_SPARSE_COMPACT, INFERENCE_BANDED_PRECISION, &
        INFERENCE_MATRIX_FREE_KRYLOV, INFERENCE_INDUCING_POINT
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer, parameter :: n = 7, d = 1, p = 2
    real(dp), parameter :: variance = 1.1_dp
    real(dp), parameter :: lengthscale = 0.6_dp
    real(dp), parameter :: noise = 0.15_dp
    real(dp) :: x(n, d), y(n, p)
    integer :: failures

    call build_data(x, y)
    failures = 0
    call test_independent_outputs_match_separate_fits(failures)
    call test_coupled_outputs_match_a_dense_solve(failures)
    call test_policy_selection(failures)
    call test_refusals(failures)
    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, " multi-output test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS"

contains

    subroutine build_data(inputs, targets)
        real(dp), intent(out) :: inputs(:, :), targets(:, :)
        integer :: i

        do i = 1, n
            inputs(i, 1) = -0.9_dp + 0.3_dp*real(i, dp)
            targets(i, 1) = sin(1.4_dp*inputs(i, 1))
            targets(i, 2) = cos(0.9_dp*inputs(i, 1)) - 0.2_dp
        end do
    end subroutine build_data

    subroutine input_covariance(matrix)
        real(dp), intent(out) :: matrix(:, :)
        integer :: i, j

        do j = 1, n
            do i = 1, n
                matrix(i, j) = variance*exp(-0.5_dp*(x(i, 1) - x(j, 1))**2/ &
                    (lengthscale*lengthscale))
            end do
        end do
    end subroutine input_covariance

    subroutine dense_solve(matrix, rhs, solution)
        real(dp), intent(in) :: matrix(:, :), rhs(:)
        real(dp), intent(out) :: solution(:)
        real(dp), allocatable :: a(:, :), b(:)
        real(dp) :: factor, swap
        integer :: i, j, k, pivot, m

        m = size(rhs)
        allocate(a, source=matrix)
        allocate(b, source=rhs)
        do k = 1, m - 1
            pivot = k
            do i = k + 1, m
                if (abs(a(i, k)) > abs(a(pivot, k))) pivot = i
            end do
            if (pivot /= k) then
                do j = 1, m
                    swap = a(k, j)
                    a(k, j) = a(pivot, j)
                    a(pivot, j) = swap
                end do
                swap = b(k)
                b(k) = b(pivot)
                b(pivot) = swap
            end if
            do i = k + 1, m
                factor = a(i, k)/a(k, k)
                a(i, k:) = a(i, k:) - factor*a(k, k:)
                b(i) = b(i) - factor*b(k)
            end do
        end do
        do i = m, 1, -1
            solution(i) = (b(i) - sum(a(i, i + 1:)*solution(i + 1:)))/a(i, i)
        end do
    end subroutine dense_solve

    subroutine test_independent_outputs_match_separate_fits(failures)
        integer, intent(inout) :: failures
        type(multi_output_gp_t) :: model
        type(kernel_t) :: kernel
        type(fortnum_status_t) :: status
        real(dp) :: weights(p, 1), independent(p)
        real(dp) :: mean(n, p), expected(n, p)
        real(dp) :: base(n, n), shifted(n, n), alpha(n)
        integer :: j, i

        weights = 0.0_dp
        independent = [1.0_dp, 0.4_dp]
        kernel = make_rbf_kernel(d, variance, lengthscale, status)
        call model%initialize(kernel, weights, independent, noise, status)
        call model%fit(x, y, status)
        call model%predict(x, mean, status)

        call input_covariance(base)
        do j = 1, p
            shifted = independent(j)*base
            do i = 1, n
                shifted(i, i) = shifted(i, i) + noise
            end do
            call dense_solve(shifted, y(:, j), alpha)
            expected(:, j) = matmul(independent(j)*base, alpha)
        end do

        if (.not. status_ok(status) .or. &
            maxval(abs(mean - expected)) > 1.0e-10_dp) then
            write (error_unit, '(a,es12.4)') &
                "FAIL [independent] decoupled outputs differ from separate fits ", &
                maxval(abs(mean - expected))
            failures = failures + 1
        end if
    end subroutine test_independent_outputs_match_separate_fits

    subroutine test_coupled_outputs_match_a_dense_solve(failures)
        integer, intent(inout) :: failures
        type(multi_output_gp_t) :: model
        type(kernel_t) :: kernel
        type(fortnum_status_t) :: status
        real(dp) :: weights(p, 1), independent(p)
        real(dp) :: mean(n, p), expected(n, p)
        real(dp) :: base(n, n), joint(n*p, n*p), stacked(n*p), alpha(n*p)
        real(dp) :: coregionalization(p, p), value, reference
        integer :: i, j, a, b

        weights(:, 1) = [0.9_dp, -0.5_dp]
        independent = [0.2_dp, 0.3_dp]
        kernel = make_rbf_kernel(d, variance, lengthscale, status)
        call model%initialize(kernel, weights, independent, noise, status)
        call model%fit(x, y, status)
        call model%predict(x, mean, status)
        call model%log_marginal_likelihood(y, value, status)

        do j = 1, p
            do i = 1, p
                coregionalization(i, j) = weights(i, 1)*weights(j, 1)
            end do
            coregionalization(j, j) = coregionalization(j, j) + independent(j)
        end do
        call input_covariance(base)
        do b = 1, p
            do a = 1, p
                do j = 1, n
                    do i = 1, n
                        joint((a - 1)*n + i, (b - 1)*n + j) = &
                            coregionalization(a, b)*base(i, j)
                    end do
                end do
            end do
        end do
        do j = 1, p
            do i = 1, n
                stacked((j - 1)*n + i) = y(i, j)
            end do
        end do
        do i = 1, n*p
            joint(i, i) = joint(i, i) + noise
        end do
        call dense_solve(joint, stacked, alpha)
        do i = 1, n*p
            joint(i, i) = joint(i, i) - noise
        end do
        do j = 1, p
            do i = 1, n
                expected(i, j) = sum(joint((j - 1)*n + i, :)*alpha)
            end do
        end do

        if (.not. status_ok(status) .or. &
            maxval(abs(mean - expected)) > 1.0e-9_dp) then
            write (error_unit, '(a,es12.4)') &
                "FAIL [coupled] posterior mean differs from the dense solve ", &
                maxval(abs(mean - expected))
            failures = failures + 1
        end if

        ! The quadratic term of the likelihood, checked independently.
        reference = -0.5_dp*sum(stacked*alpha)
        if (value > reference) then
            write (error_unit, '(a,2es14.6)') &
                "FAIL [coupled] likelihood exceeds its quadratic term ", &
                value, reference
            failures = failures + 1
        end if
    end subroutine test_coupled_outputs_match_a_dense_solve

    subroutine test_policy_selection(failures)
        integer, intent(inout) :: failures
        type(inference_problem_t) :: problem
        type(inference_choice_t) :: choice
        type(fortnum_status_t) :: status

        problem%n_samples = 100
        problem%n_features = 3
        call select_inference_policy(problem, choice, status)
        call expect(choice%policy == INFERENCE_EXACT_CHOLESKY, &
            "a small unstructured problem takes the dense solve", failures)

        problem%n_samples = 100000
        call select_inference_policy(problem, choice, status)
        call expect(choice%policy == INFERENCE_MATRIX_FREE_KRYLOV, &
            "a large unstructured problem goes matrix-free", failures)

        problem%inducing_budget = 512
        call select_inference_policy(problem, choice, status)
        call expect(choice%policy == INFERENCE_INDUCING_POINT, &
            "a declared budget selects the inducing-point bound", failures)
        problem%inducing_budget = 0

        problem%n_samples = 64
        problem%tensor_grid = .true.
        problem%grid_dimensions = [8, 8]
        call select_inference_policy(problem, choice, status)
        call expect(choice%policy == INFERENCE_STRUCTURED_GRID, &
            "a declared grid selects the structured operator", failures)
        problem%tensor_grid = .false.

        problem%compact_support = .true.
        call select_inference_policy(problem, choice, status)
        call expect(choice%policy == INFERENCE_SPARSE_COMPACT, &
            "compact support selects the sparse operator", failures)
        problem%compact_support = .false.

        problem%markov_precision = .true.
        problem%bandwidth = 2
        call select_inference_policy(problem, choice, status)
        call expect(choice%policy == INFERENCE_BANDED_PRECISION, &
            "a declared band selects the precision path", failures)
        call expect(inference_policy_name(choice%policy) == "banded-precision", &
            "the choice reports its name", failures)
        call expect(len(choice%reason) > 0, "the choice reports a reason", failures)
    end subroutine test_policy_selection

    subroutine test_refusals(failures)
        integer, intent(inout) :: failures
        type(multi_output_gp_t) :: model
        type(kernel_t) :: kernel
        type(fortnum_status_t) :: status
        type(inference_problem_t) :: problem
        type(inference_choice_t) :: choice
        real(dp) :: weights(p, 1), independent(p), mean(n, p)

        weights = 0.0_dp
        independent = [1.0_dp, 1.0_dp]
        kernel = make_rbf_kernel(d, variance, lengthscale, status)
        call model%initialize(kernel, weights, independent, -0.5_dp, status)
        call expect(.not. status_ok(status), &
            "negative noise is refused", failures)
        call model%initialize(kernel, weights, independent, noise, status)
        call model%predict(x, mean, status)
        call expect(.not. status_ok(status), &
            "prediction before fitting is refused", failures)
        call model%fit(x, y(1:3, :), status)
        call expect(.not. status_ok(status), &
            "a short target block is refused", failures)

        problem%n_samples = 64
        problem%n_features = 2
        problem%tensor_grid = .true.
        problem%compact_support = .true.
        call select_inference_policy(problem, choice, status)
        call expect(.not. status_ok(status), &
            "two declared structures are refused", failures)
        problem%compact_support = .false.
        problem%grid_dimensions = [7, 8]
        call select_inference_policy(problem, choice, status)
        call expect(.not. status_ok(status), &
            "grid extents that do not multiply out are refused", failures)
    end subroutine test_refusals

    subroutine expect(condition, description, failures)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: description
        integer, intent(inout) :: failures

        if (condition) return
        write (error_unit, '(a)') "FAIL: "//description
        failures = failures + 1
    end subroutine expect

end program test_multi_output_gp
