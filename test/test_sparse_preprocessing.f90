program test_sparse_preprocessing
    use, intrinsic :: iso_fortran_env, only: real64, error_unit
    use fortml_sparse_preprocessing, only: sparse_standard_scaler_t
    use fortsparse, only: csc_from_triplet, csc_t, fortsparse_status_t, &
        FORTSPARSE_OK, FORTSPARSE_INVALID_MATRIX
    implicit none

    integer, parameter :: dp = real64
    integer, parameter :: nrow = 4, ncol = 3, nnz = 5
    integer :: rows(nnz), columns(nnz), failures, column, entry
    real(dp) :: values(nnz), dense(nrow, ncol), expected(nrow, ncol)
    real(dp) :: scales(ncol), mean_values(ncol)
    type(csc_t) :: input, transformed, restored, tangent, cotangent
    type(fortsparse_status_t) :: status
    type(sparse_standard_scaler_t) :: scaler

    failures = 0
    rows = [1, 3, 4, 2, 4]
    columns = [1, 1, 2, 2, 3]
    values = [2.0_dp, 4.0_dp, 3.0_dp, 1.0_dp, -2.0_dp]
    dense = 0.0_dp
    do entry = 1, nnz
        dense(rows(entry), columns(entry)) = &
            dense(rows(entry), columns(entry)) + values(entry)
    end do
    call csc_from_triplet(nrow, ncol, rows, columns, values, input, status)
    call require(status%code == FORTSPARSE_OK, &
        "CSC input construction", failures)

    call scaler%fit(input, status, with_mean=.false., with_std=.true.)
    call require(status%code == FORTSPARSE_OK .and. scaler%fitted(), &
        "sparse scaler fit", failures)
    mean_values = [1.5_dp, 1.0_dp, -0.5_dp]
    scales = [sqrt(11.0_dp/4.0_dp), sqrt(3.0_dp/2.0_dp), sqrt(3.0_dp/4.0_dp)]
    call require(maxval(abs(scaler%means() - mean_values)) < 1.0e-14_dp, &
        "means include implicit zeros", failures)
    call require(maxval(abs(scaler%scales() - scales)) < 1.0e-14_dp, &
        "scales include implicit zeros", failures)
    call scaler%transform(input, transformed, status)
    call require(status%code == FORTSPARSE_OK .and. transformed%nnz == nnz, &
        "sparse transform preserves structure", failures)
    expected = dense
    do column = 1, ncol
        expected(:, column) = expected(:, column)/scales(column)
    end do
    call require(maxval(abs(csc_to_dense(transformed) - expected)) < 1.0e-14_dp, &
        "sparse transform matches independent dense oracle", failures)

    call scaler%inverse_transform(transformed, restored, status)
    call require(status%code == FORTSPARSE_OK, "sparse inverse transform", failures)
    call require(maxval(abs(csc_to_dense(restored) - dense)) < 1.0e-14_dp, &
        "inverse transform restores sparse values", failures)

    tangent = input
    tangent%val = 2.0_dp*tangent%val
    call scaler%transform_jvp(tangent, cotangent, status)
    call require(status%code == FORTSPARSE_OK, "sparse transform JVP", failures)
    call require(maxval(abs(csc_to_dense(cotangent) - 2.0_dp*expected)) < 1.0e-14_dp, &
        "sparse JVP matches linear oracle", failures)
    call scaler%transform_vjp(tangent, cotangent, status)
    call require(status%code == FORTSPARSE_OK, "sparse transform VJP", failures)
    call require(maxval(abs(csc_to_dense(cotangent) - 2.0_dp*expected)) < 1.0e-14_dp, &
        "sparse VJP matches transpose scaling oracle", failures)

    call scaler%fit(input, status, with_mean=.true.)
    call require(status%code == FORTSPARSE_INVALID_MATRIX .and. .not. scaler%fitted(), &
        "sparse centering refusal", failures)
    if (failures > 0) then
        write (error_unit, '(a,i0)') "FAIL sparse preprocessing cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS sparse preprocessing behavioral oracle"

contains

    function csc_to_dense(matrix) result(dense_matrix)
        type(csc_t), intent(in) :: matrix
        real(dp) :: dense_matrix(matrix%nrow, matrix%ncol)
        integer :: j, p

        dense_matrix = 0.0_dp
        do j = 1, matrix%ncol
            do p = matrix%col_ptr(j), matrix%col_ptr(j + 1) - 1
                dense_matrix(matrix%row_idx(p), j) = matrix%val(p)
            end do
        end do
    end function csc_to_dense

    subroutine require(condition, description, count)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: count

        if (.not. condition) then
            write (error_unit, '(a)') "FAIL [sparse preprocessing] "//description
            count = count + 1
        end if
    end subroutine require

end program test_sparse_preprocessing
