program test_sparse_operator
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_sparse_operator, only: sparse_gp_operator_t
    use fortnum_krylov, only: KRYLOV_OK
    use fortnum_linalg, only: dense_solve, LINALG_OK
    use fortsparse, only: FORTSPARSE_INVALID_MATRIX, FORTSPARSE_OK, &
        fortsparse_status_t
    implicit none

    integer, parameter :: n = 6, n_rhs = 2, n_triplets = 18
    integer :: rows(n_triplets), columns(n_triplets), nfail
    real(dp) :: values(n_triplets), dense(n, n), input(n), output(n)
    real(dp) :: expected_vector(n)
    real(dp) :: inputs(n, n_rhs), outputs(n, n_rhs), expected(n, n_rhs)
    real(dp) :: rhs(n), solution(n), dense_solution(n), diagonal(n)
    real(dp) :: expected_diagonal(n), residual_norm
    integer :: dense_info, cg_info, iterations, column, row
    type(sparse_gp_operator_t) :: sparse_operator
    type(fortsparse_status_t) :: status

    nfail = 0
    call make_triplets(rows, columns, values)
    call assemble_dense(rows, columns, values, dense)
    call sparse_operator%initialize(n, rows, columns, values, status)
    call require(status%code == FORTSPARSE_OK, &
        "sparse GP operator initializes through fortsparse", nfail)
    call require(sparse_operator%sample_count() == n, &
        "sparse GP operator reports sample count", nfail)
    call require(sparse_operator%nonzero_count() == 16, &
        "duplicate triplets are compressed", nfail)

    input = [0.3_dp, -0.7_dp, 1.1_dp, 0.2_dp, -0.4_dp, 0.8_dp]
    expected_vector = matmul(dense, input)
    call sparse_operator%matvec(input, output)
    call require(maxval(abs(output - expected_vector)) < 2.0e-13_dp, &
        "sparse vector product matches independent dense oracle", nfail)

    do column = 1, n_rhs
        inputs(:, column) = input + 0.17_dp*real(column, dp)
    end do
    expected = matmul(dense, inputs)
    call sparse_operator%matmat(inputs, outputs)
    call require(maxval(abs(outputs - expected)) < 2.0e-13_dp, &
        "sparse matrix product matches independent dense oracle", nfail)

    diagonal = sparse_operator%diagonal()
    expected_diagonal = [(dense(row, row), row=1, n)]
    call require(maxval(abs(diagonal - expected_diagonal)) < 2.0e-14_dp, &
        "sparse diagonal matches independent dense oracle", nfail)

    call sparse_operator%enter_data(status)
    call require(status%code == FORTSPARSE_OK, &
        "sparse data enters the accelerator", nfail)
    !$acc data copyin(input) copyout(output)
    call sparse_operator%matvec_device(input, output, status)
    !$acc end data
    call require(status%code == FORTSPARSE_OK, &
        "sparse device vector product returns success", nfail)
    call require(maxval(abs(output - expected_vector)) < 2.0e-13_dp, &
        "sparse device vector product matches dense oracle", nfail)
    !$acc data copyin(inputs) copyout(outputs)
    call sparse_operator%matmat_device(inputs, outputs, status)
    !$acc end data
    call require(status%code == FORTSPARSE_OK, &
        "sparse device matrix product returns success", nfail)
    call require(maxval(abs(outputs - expected)) < 2.0e-13_dp, &
        "sparse device matrix product matches dense oracle", nfail)
    call sparse_operator%exit_data(status)
    call require(status%code == FORTSPARSE_OK, &
        "sparse data exits the accelerator", nfail)

    rhs = [1.0_dp, -0.4_dp, 0.7_dp, 1.2_dp, -0.8_dp, 0.3_dp]
    call dense_solve(dense, rhs, dense_solution, dense_info)
    call require(dense_info == LINALG_OK, &
        "dense sparse solve provides independent oracle", nfail)
    solution = 0.0_dp
    call sparse_operator%solve_cg( &
        rhs, solution, 1.0e-12_dp, 50, cg_info, iterations, residual_norm)
    call require(cg_info == KRYLOV_OK, "sparse CG converges", nfail)
    call require(maxval(abs(solution - dense_solution)) < 2.0e-11_dp, &
        "sparse CG matches dense solve", nfail)
    call require(residual_norm < 2.0e-11_dp, &
        "sparse CG reports true residual", nfail)

    call sparse_operator%initialize(0, rows, columns, values, status)
    call require(status%code == FORTSPARSE_INVALID_MATRIX, &
        "sparse operator rejects an invalid sample count", nfail)

    if (nfail > 0) then
        write (error_unit, '(a,i0)') "FAIL: sparse operator checks: ", nfail
        error stop 1
    end if
    write (*, '(a)') "PASS: sparse GP operator behavioral tests"

contains

    subroutine make_triplets(output_rows, output_columns, output_values)
        integer, intent(out) :: output_rows(:), output_columns(:)
        real(dp), intent(out) :: output_values(:)

        output_rows = [ &
            1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 3, 4]
        output_columns = [ &
            1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 6, 5, 6, 3, 4]
        output_values = [ &
            2.5_dp, -0.3_dp, -0.3_dp, 0.4_dp, -0.25_dp, -0.25_dp, &
            2.6_dp, -0.2_dp, -0.2_dp, 2.7_dp, -0.18_dp, -0.18_dp, &
            2.8_dp, -0.22_dp, -0.22_dp, 2.9_dp, 0.1_dp, -0.1_dp]
    end subroutine make_triplets

    subroutine assemble_dense(input_rows, input_columns, input_values, matrix)
        integer, intent(in) :: input_rows(:), input_columns(:)
        real(dp), intent(in) :: input_values(:)
        real(dp), intent(out) :: matrix(:, :)
        integer :: entry

        matrix = 0.0_dp
        do entry = 1, size(input_values)
            matrix(input_rows(entry), input_columns(entry)) = &
                matrix(input_rows(entry), input_columns(entry)) + &
                input_values(entry)
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

end program test_sparse_operator
