program test_toeplitz_operator
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_toeplitz_operator, only: toeplitz_gp_operator_t
    use fortnum_krylov, only: KRYLOV_OK
    use fortnum_linalg, only: dense_solve, LINALG_OK
    use fortnum_status, only: fortnum_status_t, FORTNUM_OK
    implicit none

    type(toeplitz_gp_operator_t) :: toeplitz_operator
    type(fortnum_status_t) :: status
    real(dp) :: column(5), dense(5, 5), input(5), output(5), expected(5)
    real(dp) :: inputs(5, 2), outputs(5, 2), expected_matrix(5, 2)
    real(dp) :: right_hand_side(5), solution(5), dense_solution(5)
    real(dp) :: diagonal(5), expected_diagonal(5), residual_norm
    integer :: dense_info, cg_info, iterations, column_index, row, nfail

    nfail = 0
    column = [2.4_dp, -0.25_dp, 0.08_dp, -0.03_dp, 0.01_dp]
    call toeplitz_operator%initialize(column, status)
    call require(status%code == FORTNUM_OK, &
        "Toeplitz GP operator initializes", nfail)
    call require(toeplitz_operator%sample_count() == 5, &
        "Toeplitz GP operator reports sample count", nfail)
    call assemble_dense(column, dense)

    input = [0.3_dp, -0.7_dp, 1.1_dp, 0.2_dp, -0.4_dp]
    expected = matmul(dense, input)
    call toeplitz_operator%matvec(input, output)
    call require(maxval(abs(output - expected)) < 2.0e-13_dp, &
        "Toeplitz GP vector product matches dense oracle", nfail)

    do column_index = 1, 2
        inputs(:, column_index) = input + 0.17_dp*real(column_index, dp)
    end do
    expected_matrix = matmul(dense, inputs)
    call toeplitz_operator%matmat(inputs, outputs)
    call require(maxval(abs(outputs - expected_matrix)) < 2.0e-13_dp, &
        "Toeplitz GP matrix product matches dense oracle", nfail)

    diagonal = toeplitz_operator%diagonal()
    expected_diagonal = [(column(1), row=1, 5)]
    call require(maxval(abs(diagonal - expected_diagonal)) < 2.0e-14_dp, &
        "Toeplitz GP diagonal matches dense oracle", nfail)

    right_hand_side = [1.0_dp, -0.4_dp, 0.7_dp, 1.2_dp, -0.8_dp]
    call dense_solve(dense, right_hand_side, dense_solution, dense_info)
    call require(dense_info == LINALG_OK, &
        "dense Toeplitz solve provides oracle", nfail)
    solution = 0.0_dp
    call toeplitz_operator%solve_cg( &
        right_hand_side, solution, 1.0e-12_dp, 50, cg_info, iterations, &
        residual_norm)
    call require(cg_info == KRYLOV_OK, "Toeplitz GP CG converges", nfail)
    call require(maxval(abs(solution - dense_solution)) < 2.0e-11_dp, &
        "Toeplitz GP CG matches dense solve", nfail)
    call require(residual_norm < 2.0e-11_dp, &
        "Toeplitz GP CG reports true residual", nfail)

    if (nfail > 0) then
        write (error_unit, '(a,i0)') "FAIL: Toeplitz GP checks: ", nfail
        error stop 1
    end if
    write (*, '(a)') "PASS: Toeplitz GP operator behavioral tests"

contains

    subroutine assemble_dense(first_column, output_matrix)
        real(dp), intent(in) :: first_column(:)
        real(dp), intent(out) :: output_matrix(:, :)
        integer :: i, j

        do i = 1, size(first_column)
            do j = 1, size(first_column)
                output_matrix(i, j) = first_column(abs(i - j) + 1)
            end do
        end do
    end subroutine assemble_dense

    subroutine require(condition, description, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failures

        if (.not. condition) then
            write (error_unit, '(a)') "FAIL: "//description
            failures = failures + 1
        end if
    end subroutine require

end program test_toeplitz_operator
