program test_sparse_operator_device
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_sparse_operator, only: sparse_gp_operator_t
    use fortsparse_status, only: FORTSPARSE_OK, fortsparse_status_t
    implicit none

    integer, parameter :: n = 8, n_rhs = 3, n_triplets = 22
    integer :: rows(n_triplets), columns(n_triplets), nfail
    real(dp) :: values(n_triplets), dense(n, n), input(n), output(n)
    real(dp) :: inputs(n, n_rhs), outputs(n, n_rhs), expected(n, n_rhs)
    real(dp) :: expected_vector(n)
    integer :: column
    type(sparse_gp_operator_t) :: sparse_operator
    type(fortsparse_status_t) :: status

    call make_triplets(rows, columns, values)
    call assemble_dense(rows, columns, values, dense)
    call sparse_operator%initialize(n, rows, columns, values, status)
    nfail = 0
    call require(status%code == FORTSPARSE_OK, &
        "sparse GPU operator initializes", nfail)
    input = [(sin(0.13_dp*real(column, dp)), column=1, n)]
    expected_vector = matmul(dense, input)
    do column = 1, n_rhs
        inputs(:, column) = input + 0.11_dp*real(column, dp)
    end do
    expected = matmul(dense, inputs)

    call sparse_operator%enter_data(status)
    call require(status%code == FORTSPARSE_OK, &
        "sparse GPU operator enters persistent data", nfail)
    !$acc data copyin(input) copyout(output)
    call sparse_operator%matvec_device(input, output, status)
    !$acc end data
    call require(status%code == FORTSPARSE_OK, &
        "sparse GPU vector product returns success", nfail)
    call require(maxval(abs(output - expected_vector)) < 2.0e-13_dp, &
        "sparse GPU vector product matches dense oracle", nfail)
    !$acc data copyin(inputs) copyout(outputs)
    call sparse_operator%matmat_device(inputs, outputs, status)
    !$acc end data
    call require(status%code == FORTSPARSE_OK, &
        "sparse GPU matrix product returns success", nfail)
    call require(maxval(abs(outputs - expected)) < 2.0e-13_dp, &
        "sparse GPU matrix product matches dense oracle", nfail)
    call sparse_operator%exit_data(status)
    call require(status%code == FORTSPARSE_OK, &
        "sparse GPU operator exits persistent data", nfail)

    if (nfail > 0) then
        write (error_unit, '(a,i0)') &
            "FAIL: sparse GPU operator checks: ", nfail
        error stop 1
    end if
    write (*, '(a)') "PASS: sparse GPU operator behavioral tests"

contains

    subroutine make_triplets(output_rows, output_columns, output_values)
        integer, intent(out) :: output_rows(:), output_columns(:)
        real(dp), intent(out) :: output_values(:)

        output_rows = [ &
            1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, &
            6, 7, 7, 8, 8, 4]
        output_columns = [ &
            1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 6, 5, 6, &
            7, 6, 7, 6, 8, 4]
        output_values = [ &
            2.5_dp, -0.3_dp, -0.3_dp, 2.6_dp, -0.25_dp, -0.25_dp, &
            2.7_dp, -0.2_dp, -0.2_dp, 2.8_dp, -0.18_dp, -0.18_dp, &
            2.9_dp, -0.22_dp, -0.22_dp, 3.0_dp, -0.2_dp, -0.2_dp, &
            3.1_dp, -0.2_dp, 3.2_dp, 0.1_dp]
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

end program test_sparse_operator_device
