program test_structured_operator
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_structured_operator, only: structured_gp_operator_t, &
        tensor_factor_t
    use fortnum_krylov, only: KRYLOV_OK
    use fortnum_linalg, only: dense_solve, LINALG_OK
    use fortnum_status, only: fortnum_status_t, FORTNUM_DOMAIN_ERROR, &
        FORTNUM_OK
    implicit none

    type(tensor_factor_t) :: factors(3), bad_factor(1)
    type(structured_gp_operator_t) :: structured_operator
    type(fortnum_status_t) :: status
    real(dp) :: dense(12, 12), input(12), output(12), expected(12)
    real(dp) :: inputs(12, 2), outputs(12, 2), expected_matrix(12, 2)
    real(dp) :: right_hand_side(12, 2), solution(12, 2), dense_solution(12, 2)
    real(dp) :: diagonal(12), expected_diagonal(12), residual_norm(2)
    integer :: dense_info, cg_info(2), iterations(2), column
    integer :: i1, i2, i3, j1, j2, j3
    integer :: row, row_index, column_index, nfail

    nfail = 0
    call make_factors(factors)
    call structured_operator%initialize(factors, status)
    call require(status%code == FORTNUM_OK, &
        "structured GP operator initializes", nfail)
    call require(structured_operator%sample_count() == 12, &
        "structured GP operator reports grid size", nfail)

    call assemble_dense(factors, dense)
    input = [(0.12_dp + 0.05_dp*real(row, dp), row=1, 12)]
    expected = matmul(dense, input)
    call structured_operator%matvec(input, output)
    call require(maxval(abs(output - expected)) < 2.0e-14_dp, &
        "structured GP matrix-vector product matches dense oracle", nfail)

    do column = 1, 2
        inputs(:, column) = input + 0.2_dp*real(column, dp)
    end do
    expected_matrix = matmul(dense, inputs)
    call structured_operator%matmat(inputs, outputs)
    call require(maxval(abs(outputs - expected_matrix)) < 2.0e-14_dp, &
        "structured GP multi-RHS product matches dense oracle", nfail)

    diagonal = structured_operator%diagonal()
    expected_diagonal = [(dense(row, row), row=1, 12)]
    call require(maxval(abs(diagonal - expected_diagonal)) < 2.0e-14_dp, &
        "structured GP diagonal matches dense oracle", nfail)

    right_hand_side(:, 1) = [ &
        1.0_dp, -0.4_dp, 0.7_dp, 1.2_dp, -0.8_dp, 0.3_dp, &
        0.6_dp, -1.1_dp, 0.2_dp, 0.9_dp, -0.5_dp, 0.4_dp]
    right_hand_side(:, 2) = [ &
        -0.2_dp, 0.8_dp, 1.1_dp, -0.6_dp, 0.5_dp, -0.9_dp, &
        0.3_dp, 0.4_dp, -1.2_dp, 0.7_dp, 0.1_dp, -0.3_dp]
    call dense_solve(dense, right_hand_side, dense_solution, dense_info)
    call require(dense_info == LINALG_OK, &
        "dense structured covariance solve provides oracle", nfail)
    solution = 0.0_dp
    call structured_operator%solve_cg_multi( &
        right_hand_side, solution, 1.0e-12_dp, 50, cg_info, iterations, &
        residual_norm)

    if (any(cg_info /= KRYLOV_OK)) then
        call require(.false., "structured GP CG converges", nfail)
    end if
    call require(maxval(abs(solution - dense_solution)) < 2.0e-11_dp, &
        "structured GP CG matches dense solve", nfail)
    call require(maxval(residual_norm) < 2.0e-11_dp, &
        "structured GP CG reports true residuals", nfail)

    allocate(bad_factor(1)%values(2, 3))
    call structured_operator%initialize(bad_factor, status)
    call require(status%code == FORTNUM_DOMAIN_ERROR, &
        "structured GP rejects nonsquare factor", nfail)

    if (nfail > 0) then
        write (error_unit, '(a,i0)') "FAIL: structured operator checks: ", nfail
        error stop 1
    end if
    write (*, '(a)') "PASS: structured GP operator behavioral tests"

contains

    subroutine make_factors(output_factors)
        type(tensor_factor_t), intent(out) :: output_factors(3)

        allocate( &
            output_factors(1)%values(2, 2), &
            output_factors(2)%values(3, 3), &
            output_factors(3)%values(2, 2))
        output_factors(1)%values = reshape([ &
            2.0_dp, -0.2_dp, -0.2_dp, 2.0_dp], [2, 2])
        output_factors(2)%values = reshape([ &
            2.2_dp, -0.3_dp, 0.0_dp, &
            -0.3_dp, 2.4_dp, -0.25_dp, &
            0.0_dp, -0.25_dp, 2.1_dp], [3, 3])
        output_factors(3)%values = reshape([ &
            1.8_dp, -0.15_dp, -0.15_dp, 1.9_dp], [2, 2])
    end subroutine make_factors

    subroutine assemble_dense(input_factors, output_matrix)
        type(tensor_factor_t), intent(in) :: input_factors(:)
        real(dp), intent(out) :: output_matrix(:, :)

        do i3 = 1, 2
            do i2 = 1, 3
                do i1 = 1, 2
                    row_index = flatten(i1, i2, i3)
                    do j3 = 1, 2
                        do j2 = 1, 3
                            do j1 = 1, 2
                                column_index = flatten(j1, j2, j3)
                                output_matrix(row_index, column_index) = &
                                    input_factors(3)%values(i3, j3)* &
                                    input_factors(2)%values(i2, j2)* &
                                    input_factors(1)%values(i1, j1)
                            end do
                        end do
                    end do
                end do
            end do
        end do
    end subroutine assemble_dense

    integer function flatten(i1, i2, i3) result(index)
        integer, intent(in) :: i1, i2, i3

        index = i1 + 2*(i2 - 1) + 6*(i3 - 1)
    end function flatten

    subroutine require(condition, description, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failures

        if (.not. condition) then
            write (error_unit, '(a)') "FAIL: "//description
            failures = failures + 1
        end if
    end subroutine require

end program test_structured_operator
