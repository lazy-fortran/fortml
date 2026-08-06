program test_kernel_operator
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_kernel_operator, only: kernel_operator_t, rbf_operator_t
    use fortml_kernels, only: kernel_add, kernel_t, make_constant_kernel, &
        make_rbf_kernel
    use fortnum_linalg, only: dense_solve, LINALG_OK
    use fortnum_krylov, only: KRYLOV_OK
    use fortnum_status, only: FORTNUM_OK, fortnum_status_t
    implicit none

    real(dp) :: sample_points(5, 2), input(5), output(5), expected(5)
    real(dp) :: matrix_input(5, 2), matrix_output(5, 2)
    real(dp) :: matrix_expected(5, 2), diagonal(5)
    real(dp) :: generic_matrix_output(5, 2), generic_matrix_expected(5, 2)
    real(dp) :: covariance(5, 5), right_hand_side(5), solution(5)
    real(dp) :: dense_solution(5), residual_norm
    real(dp) :: multi_solution(5, 2), dense_multi_solution(5, 2)
    real(dp) :: multi_residual_norm(2)
    real(dp) :: block_solution(5, 2), block_residual_norm(2)
    real(dp) :: nystrom_solution(5, 2), nystrom_residual_norm(2)
    real(dp) :: generic_covariance(5, 5)
    real(dp) :: generic_multi_solution(5, 2), generic_dense_multi_solution(5, 2)
    real(dp) :: generic_multi_residual_norm(2)
    real(dp) :: generic_block_solution(5, 2), generic_nystrom_solution(5, 2)
    real(dp) :: generic_block_residual_norm(2)
    real(dp) :: generic_nystrom_residual_norm(2)
    real(dp) :: block_reconstruction(2, 2)
    real(dp) :: small_reconstruction(3, 3), small_reference(3, 3)
    real(dp) :: generic_output(5), generic_expected(5)
    real(dp) :: sample_points_8(5, 8), input_8(5), output_8(5), expected_8(5)
    real(dp) :: matrix_input_8(5, 2), matrix_output_8(5, 2)
    real(dp) :: matrix_expected_8(5, 2)
    real(dp), parameter :: variance = 1.7_dp, lengthscale = 0.8_dp
    real(dp), parameter :: diagonal_shift = 0.03_dp
    type(rbf_operator_t) :: rbf_operator
    type(rbf_operator_t) :: rbf_operator_8
    type(kernel_operator_t) :: generic_operator
    type(kernel_t) :: rbf_kernel, constant_kernel, sum_kernel
    type(fortnum_status_t) :: status
    integer :: i, j, column, feature, info, iterations
    integer :: multi_info(2), multi_iterations(2)

    sample_points = reshape([ &
        -0.7_dp, 0.1_dp, 0.4_dp, 1.2_dp, 1.8_dp, &
        0.3_dp, -0.2_dp, 0.9_dp, 1.5_dp, 2.1_dp], [5, 2])
    input = [1.0_dp, -0.3_dp, 0.8_dp, 1.4_dp, -0.6_dp]
    call rbf_operator%initialize( &
        sample_points, variance, lengthscale, diagonal_shift, status, 2)
    call require(status%code == FORTNUM_OK, "RBF operator initializes")
    call require( &
        rbf_operator%sample_count() == 5, "RBF operator reports sample count")

    !$acc data copyin(rbf_operator%points, input) copyout(output)
    call rbf_operator%matvec(input, output)
    !$acc end data
    do i = 1, 5
        expected(i) = diagonal_shift*input(i)
        do j = 1, 5
            expected(i) = expected(i) + variance*exp( &
                -0.5_dp*sum((sample_points(i, :) - sample_points(j, :))**2)/ &
                (lengthscale*lengthscale))*input(j)
        end do
    end do
    call require(maxval(abs(output - expected)) < 2.0e-14_dp, &
        "RBF MVM matches the direct pairwise oracle")

    matrix_input(:, 1) = input
    matrix_input(:, 2) = [0.2_dp, -1.1_dp, 0.4_dp, 0.7_dp, 1.3_dp]
    call rbf_operator%matmat(matrix_input, matrix_output)
    do column = 1, 2
        do i = 1, 5
            matrix_expected(i, column) = diagonal_shift*matrix_input(i, column)
            do j = 1, 5
                matrix_expected(i, column) = matrix_expected(i, column) + &
                    variance*exp( &
                    -0.5_dp*sum((sample_points(i, :) - sample_points(j, :))**2)/ &
                    (lengthscale*lengthscale))*matrix_input(j, column)
            end do
        end do
    end do
    call require(maxval(abs(matrix_output - matrix_expected)) < 2.0e-14_dp, &
        "RBF batched MVM matches the direct pairwise oracle")

    diagonal = rbf_operator%diagonal()
    call require(maxval(abs(diagonal - (variance + diagonal_shift))) < 2.0e-14_dp, &
        "RBF diagonal is returned without materializing a matrix")

    do feature = 1, 8
        do i = 1, 5
            sample_points_8(i, feature) = &
                sin(0.11_dp*real(i + 2*feature, dp)) + &
                0.03_dp*cos(0.07_dp*real(i*feature, dp))
        end do
    end do
    input_8 = [0.4_dp, -0.9_dp, 1.2_dp, 0.1_dp, -0.5_dp]
    call rbf_operator_8%initialize( &
        sample_points_8, variance, lengthscale, diagonal_shift, status, 2)
    call require(status%code == FORTNUM_OK, "8-feature RBF operator initializes")
    !$acc data copyin(rbf_operator_8%points, input_8) copyout(output_8)
    call rbf_operator_8%matvec(input_8, output_8)
    !$acc end data
    do i = 1, 5
        expected_8(i) = diagonal_shift*input_8(i)
        do j = 1, 5
            expected_8(i) = expected_8(i) + variance*exp( &
                -0.5_dp*sum((sample_points_8(i, :) - sample_points_8(j, :))**2)/ &
                (lengthscale*lengthscale))*input_8(j)
        end do
    end do
    call require(maxval(abs(output_8 - expected_8)) < 2.0e-14_dp, &
        "8-feature RBF MVM matches the direct pairwise oracle")
    call rbf_operator_8%enter_data(status)
    call require(status%code == FORTNUM_OK, "RBF operator enters device data")
    call rbf_operator_8%matvec(input_8, output_8)
    call rbf_operator_8%exit_data(status)
    call require(status%code == FORTNUM_OK, "RBF operator exits device data")
    call require(maxval(abs(output_8 - expected_8)) < 2.0e-14_dp, &
        "operator-owned device residency preserves the MVM oracle")
    matrix_input_8(:, 1) = input_8
    matrix_input_8(:, 2) = [0.6_dp, -0.2_dp, 0.8_dp, -1.4_dp, 0.3_dp]
    call rbf_operator_8%matmat(matrix_input_8, matrix_output_8)
    do column = 1, 2
        do i = 1, 5
            matrix_expected_8(i, column) = &
                diagonal_shift*matrix_input_8(i, column)
            do j = 1, 5
                matrix_expected_8(i, column) = &
                    matrix_expected_8(i, column) + variance*exp( &
                    -0.5_dp*sum((sample_points_8(i, :) - &
                    sample_points_8(j, :))**2)/(lengthscale*lengthscale))* &
                    matrix_input_8(j, column)
            end do
        end do
    end do
    call require(maxval(abs(matrix_output_8 - matrix_expected_8)) < 2.0e-14_dp, &
        "8-feature RBF batched MVM matches the direct pairwise oracle")

    rbf_kernel = make_rbf_kernel(2, variance, lengthscale, status)
    call require(status%code == FORTNUM_OK, "generic RBF kernel initializes")
    constant_kernel = make_constant_kernel(2, 0.2_dp, status)
    call require(status%code == FORTNUM_OK, "constant kernel initializes")
    sum_kernel = kernel_add(rbf_kernel, constant_kernel, status)
    call require(status%code == FORTNUM_OK, "composite kernel initializes")
    call generic_operator%initialize( &
        sample_points, sum_kernel, diagonal_shift, status, 2)
    call require(status%code == FORTNUM_OK, "generic kernel operator initializes")
    call generic_operator%matvec(input, generic_output)
    do i = 1, 5
        generic_expected(i) = diagonal_shift*input(i)
        do j = 1, 5
            generic_expected(i) = generic_expected(i) + &
                (0.2_dp + variance*exp( &
                -0.5_dp*sum((sample_points(i, :) - sample_points(j, :))**2)/ &
                (lengthscale*lengthscale)))*input(j)
        end do
    end do
    call require(maxval(abs(generic_output - generic_expected)) < 2.0e-14_dp, &
        "generic composite operator matches the direct oracle")
    call generic_operator%matmat(matrix_input, generic_matrix_output)
    do column = 1, 2
        do i = 1, 5
            generic_matrix_expected(i, column) = diagonal_shift* &
                matrix_input(i, column)
            do j = 1, 5
                generic_matrix_expected(i, column) = &
                    generic_matrix_expected(i, column) + &
                    (0.2_dp + variance*exp( &
                    -0.5_dp*sum((sample_points(i, :) - &
                    sample_points(j, :))**2)/(lengthscale*lengthscale)))* &
                    matrix_input(j, column)
            end do
        end do
    end do
    call require(maxval(abs(generic_matrix_output - &
        generic_matrix_expected)) < 2.0e-14_dp, &
        "generic fused batched operator matches the direct oracle")
    do i = 1, 5
        do j = 1, 5
            generic_covariance(i, j) = diagonal_shift* &
                merge(1.0_dp, 0.0_dp, i == j) + 0.2_dp + variance*exp( &
                -0.5_dp*sum((sample_points(i, :) - sample_points(j, :))**2)/ &
                (lengthscale*lengthscale))
        end do
    end do
    call dense_solve( &
        generic_covariance, matrix_input, generic_dense_multi_solution, info)
    call require(info == LINALG_OK, &
        "generic dense multi-RHS solve provides the oracle")
    generic_multi_solution = 0.0_dp
    call generic_operator%solve_cg_multi( &
        matrix_input, generic_multi_solution, 1.0e-12_dp, 30, multi_info, &
        multi_iterations, generic_multi_residual_norm)
    do column = 1, 2
        call require(multi_info(column) == KRYLOV_OK, &
            "generic batched operator CG converges")
    end do
    call require(maxval(abs(generic_multi_solution - &
        generic_dense_multi_solution)) < 2.0e-11_dp, &
        "generic batched operator CG matches the dense oracle")
    call require(maxval(generic_multi_residual_norm) < 2.0e-11_dp, &
        "generic batched operator CG reports true residuals")
    generic_block_solution = 0.0_dp
    call generic_operator%solve_cg_multi_block( &
        matrix_input, generic_block_solution, 1.0e-12_dp, 30, 2, multi_info, &
        multi_iterations, generic_block_residual_norm)
    do column = 1, 2
        call require(multi_info(column) == KRYLOV_OK, &
            "block-preconditioned generic multi-RHS CG converges")
    end do
    call require(maxval(abs(generic_block_solution - &
        generic_dense_multi_solution)) < 2.0e-11_dp, &
        "block-preconditioned generic CG matches the dense oracle")
    call require(maxval(generic_block_residual_norm) < 2.0e-11_dp, &
        "block-preconditioned generic CG reports true residuals")
    generic_nystrom_solution = 0.0_dp
    call generic_operator%solve_cg_multi_nystrom( &
        matrix_input, generic_nystrom_solution, 1.0e-12_dp, 30, 3, multi_info, &
        multi_iterations, generic_nystrom_residual_norm)
    do column = 1, 2
        call require(multi_info(column) == KRYLOV_OK, &
            "Nystrom-preconditioned generic multi-RHS CG converges")
    end do
    call require(maxval(abs(generic_nystrom_solution - &
        generic_dense_multi_solution)) < 2.0e-11_dp, &
        "Nystrom-preconditioned generic CG matches the dense oracle")
    call require(maxval(generic_nystrom_residual_norm) < 2.0e-11_dp, &
        "Nystrom-preconditioned generic CG reports true residuals")
    ! The block factors must reproduce the shifted kernel block exactly, and
    ! the Nystrom apply must invert its own low-rank operator exactly. A CG
    ! answer alone cannot catch a wrong preconditioner, only a slower one.
    do i = 1, 2
        do j = 1, 2
            block_reconstruction(i, j) = sum( &
                generic_operator%block_preconditioner%factors(i, :, 1)* &
                generic_operator%block_preconditioner%factors(j, :, 1))
        end do
    end do
    call require(maxval(abs(block_reconstruction - &
        generic_covariance(1:2, 1:2))) < 2.0e-13_dp, &
        "generic block preconditioner factors the true kernel block")
    do i = 1, generic_operator%nystrom_preconditioner%rank
        do j = 1, generic_operator%nystrom_preconditioner%rank
            small_reconstruction(i, j) = sum( &
                generic_operator%nystrom_preconditioner%factor(i, :)* &
                generic_operator%nystrom_preconditioner%factor(j, :))
            small_reference(i, j) = sum( &
                generic_operator%nystrom_preconditioner%features(:, i)* &
                generic_operator%nystrom_preconditioner%features(:, j))
            if (i == j) small_reference(i, j) = small_reference(i, j) + &
                generic_operator%nystrom_preconditioner%regularization
        end do
    end do
    call require(maxval(abs(small_reconstruction - small_reference)) < 2.0e-11_dp, &
        "generic Nystrom factor is the Cholesky of its own normal matrix")
    diagonal = generic_operator%diagonal()
    call require(maxval(abs(diagonal - (variance + 0.2_dp + diagonal_shift))) < &
        2.0e-14_dp, "generic operator diagonal uses the kernel value API")

    do i = 1, 5
        do j = 1, 5
            covariance(i, j) = diagonal_shift*merge(1.0_dp, 0.0_dp, i == j) + &
                variance*exp( &
                -0.5_dp*sum((sample_points(i, :) - sample_points(j, :))**2)/ &
                (lengthscale*lengthscale))
        end do
    end do
    right_hand_side = [0.6_dp, -1.1_dp, 0.2_dp, 1.4_dp, -0.7_dp]
    call dense_solve(covariance, right_hand_side, dense_solution, info)
    call require(info == LINALG_OK, "dense solve provides the CG oracle")
    solution = 0.0_dp
    call rbf_operator%solve_cg( &
        right_hand_side, solution, 1.0e-12_dp, 30, info, iterations, &
        residual_norm)
    call require(info == KRYLOV_OK, "lazy RBF operator CG converges")
    call require(iterations > 0 .and. iterations <= 30, &
        "lazy RBF operator reports bounded CG iterations")
    call require(maxval(abs(solution - dense_solution)) < 2.0e-11_dp, &
        "lazy RBF CG matches the independent dense solve")
    call require(residual_norm < 2.0e-11_dp, &
        "lazy RBF CG reports the true residual")
    solution = 0.0_dp
    call rbf_operator%solve_cg( &
        right_hand_side, solution, 1.0e-12_dp, 30, info, iterations, &
        residual_norm, use_diagonal_preconditioner=.false.)
    call require(info == KRYLOV_OK, "unpreconditioned lazy RBF CG converges")
    call require(maxval(abs(solution - dense_solution)) < 2.0e-11_dp, &
        "unpreconditioned lazy RBF CG matches the dense solve")

    call dense_solve(covariance, matrix_input, dense_multi_solution, info)
    call require(info == LINALG_OK, "dense multi-RHS solve provides the oracle")
    call rbf_operator%enter_data(status, 2)
    call require(status%code == FORTNUM_OK, &
        "RBF operator enters reusable multi-RHS workspace")
    multi_solution = 0.0_dp
    call rbf_operator%solve_cg_multi( &
        matrix_input, multi_solution, 1.0e-12_dp, 30, multi_info, &
        multi_iterations, multi_residual_norm)
    do column = 1, 2
        call require(multi_info(column) == KRYLOV_OK, &
            "multi-RHS lazy RBF CG converges")
        call require(multi_iterations(column) > 0 .and. &
            multi_iterations(column) <= 30, &
            "multi-RHS lazy RBF CG reports bounded iterations")
    end do
    call require(maxval(abs(multi_solution - dense_multi_solution)) < 2.0e-11_dp, &
        "multi-RHS lazy RBF CG matches independent dense solves")
    call require(maxval(multi_residual_norm) < 2.0e-11_dp, &
        "multi-RHS lazy RBF CG reports true residuals")
    multi_solution = 0.0_dp
    call rbf_operator%solve_cg_multi( &
        matrix_input, multi_solution, 1.0e-12_dp, 30, multi_info, &
        multi_iterations, multi_residual_norm, &
        use_diagonal_preconditioner=.false.)
    do column = 1, 2
        call require(multi_info(column) == KRYLOV_OK, &
            "unpreconditioned multi-RHS lazy RBF CG converges")
    end do
    call require(maxval(abs(multi_solution - dense_multi_solution)) < 2.0e-11_dp, &
        "unpreconditioned multi-RHS CG matches independent dense solves")
    block_solution = 0.0_dp
    call rbf_operator%solve_cg_multi_block( &
        matrix_input, block_solution, 1.0e-12_dp, 30, 2, multi_info, &
        multi_iterations, block_residual_norm)
    do column = 1, 2
        call require(multi_info(column) == KRYLOV_OK, &
            "block-preconditioned multi-RHS lazy RBF CG converges")
    end do
    call require(maxval(abs(block_solution - dense_multi_solution)) < 2.0e-11_dp, &
        "block-preconditioned multi-RHS CG matches independent dense solves")
    call require(maxval(block_residual_norm) < 2.0e-11_dp, &
        "block-preconditioned multi-RHS CG reports true residuals")
    nystrom_solution = 0.0_dp
    call rbf_operator%solve_cg_multi_nystrom( &
        matrix_input, nystrom_solution, 1.0e-12_dp, 30, 2, multi_info, &
        multi_iterations, nystrom_residual_norm)
    do column = 1, 2
        call require(multi_info(column) == KRYLOV_OK, &
            "Nystrom-preconditioned multi-RHS lazy RBF CG converges")
    end do
    call require(maxval(abs(nystrom_solution - dense_multi_solution)) < 2.0e-11_dp, &
        "Nystrom-preconditioned multi-RHS CG matches independent dense solves")
    call require(maxval(nystrom_residual_norm) < 2.0e-11_dp, &
        "Nystrom-preconditioned multi-RHS CG reports true residuals")
    call rbf_operator%exit_data(status)
    call require(status%code == FORTNUM_OK, &
        "RBF operator exits reusable multi-RHS workspace")

contains

    subroutine require(condition, description)
        logical, intent(in) :: condition
        character(*), intent(in) :: description

        if (.not. condition) then
            write (error_unit, '(a)') "FAIL: "//description
            error stop 1
        end if
    end subroutine require

end program test_kernel_operator
