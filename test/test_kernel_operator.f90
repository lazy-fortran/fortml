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
    real(dp) :: covariance(5, 5), right_hand_side(5), solution(5)
    real(dp) :: dense_solution(5), residual_norm
    real(dp) :: generic_output(5), generic_expected(5)
    real(dp), parameter :: variance = 1.7_dp, lengthscale = 0.8_dp
    real(dp), parameter :: diagonal_shift = 0.03_dp
    type(rbf_operator_t) :: rbf_operator
    type(kernel_operator_t) :: generic_operator
    type(kernel_t) :: rbf_kernel, constant_kernel, sum_kernel
    type(fortnum_status_t) :: status
    integer :: i, j, column, info, iterations

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
