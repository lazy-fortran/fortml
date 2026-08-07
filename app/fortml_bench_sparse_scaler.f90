program fortml_bench_sparse_scaler
    use, intrinsic :: iso_fortran_env, only: real64, int64, output_unit
    use fortml_sparse_preprocessing, only: sparse_standard_scaler_t
    use fortsparse, only: csc_from_triplet, csc_t, fortsparse_status_t, &
        FORTSPARSE_OK
    implicit none

    integer, parameter :: dp = real64
    integer, parameter :: nrow = 4, ncol = 3, nnz = 5
    integer, parameter :: repetitions = 20000
    integer :: rows(nnz), columns(nnz), i, ios, unit
    real(dp) :: values(nnz), tangent_values(nnz)
    real(dp) :: seconds, sink
    integer(int64) :: started, finished, rate
    character(512) :: oracle_path
    type(csc_t) :: input, transformed, restored, tangent, cotangent
    type(fortsparse_status_t) :: status
    type(sparse_standard_scaler_t) :: scaler

    rows = [1, 3, 4, 2, 4]
    columns = [1, 1, 2, 2, 3]
    values = [2.0_dp, 4.0_dp, 3.0_dp, 1.0_dp, -2.0_dp]
    tangent_values = 2.0_dp*values
    call csc_from_triplet(nrow, ncol, rows, columns, values, input, status)
    if (status%code /= FORTSPARSE_OK) error stop "sparse scaler benchmark input failed"
    call scaler%fit(input, status, with_mean=.false., with_std=.true.)
    if (status%code /= FORTSPARSE_OK) error stop "sparse scaler benchmark fit failed"
    tangent = input
    tangent%val = tangent_values

    call system_clock(count_rate=rate)
    call system_clock(started)
    do i = 1, repetitions
        call scaler%transform(input, transformed, status)
        if (status%code /= FORTSPARSE_OK) error stop "sparse scaler transform failed"
    end do
    call system_clock(finished)
    seconds = real(finished - started, dp)/real(rate, dp)/real(repetitions, dp)
    write (output_unit, '(a,",",es24.16)') "transform", seconds

    call system_clock(started)
    do i = 1, repetitions
        call scaler%inverse_transform(transformed, restored, status)
        if (status%code /= FORTSPARSE_OK) error stop "sparse scaler inverse failed"
    end do
    call system_clock(finished)
    seconds = real(finished - started, dp)/real(rate, dp)/real(repetitions, dp)
    write (output_unit, '(a,",",es24.16)') "inverse", seconds

    call system_clock(started)
    do i = 1, repetitions
        call scaler%transform_jvp(tangent, cotangent, status)
        if (status%code /= FORTSPARSE_OK) error stop "sparse scaler JVP failed"
    end do
    call system_clock(finished)
    seconds = real(finished - started, dp)/real(rate, dp)/real(repetitions, dp)
    write (output_unit, '(a,",",es24.16)') "jvp", seconds

    call system_clock(started)
    do i = 1, repetitions
        call scaler%transform_vjp(tangent, cotangent, status)
        if (status%code /= FORTSPARSE_OK) error stop "sparse scaler VJP failed"
    end do
    call system_clock(finished)
    seconds = real(finished - started, dp)/real(rate, dp)/real(repetitions, dp)
    write (output_unit, '(a,",",es24.16)') "vjp", seconds
    sink = sum(transformed%val) + sum(restored%val) + sum(cotangent%val)
    if (sink /= sink) error stop "sparse scaler benchmark produced NaN"

    call get_environment_variable("FORTML_BENCH_SPARSE_SCALER_ORACLE", &
        oracle_path, status=ios)
    if (ios == 0 .and. len_trim(oracle_path) > 0) then
        open (newunit=unit, file=trim(oracle_path), status="replace", &
            action="write", iostat=ios)
        if (ios /= 0) error stop "sparse scaler benchmark oracle open failed"
        write (unit, '(a)') "phase,row,column,value"
        call write_matrix(unit, "transform", transformed)
        call write_matrix(unit, "inverse", restored)
        call write_matrix(unit, "jvp", cotangent)
        call scaler%transform_vjp(tangent, cotangent, status)
        if (status%code /= FORTSPARSE_OK) error stop "sparse scaler VJP oracle failed"
        call write_matrix(unit, "vjp", cotangent)
        close (unit)
    end if

contains

    subroutine write_matrix(unit, phase, matrix)
        integer, intent(in) :: unit
        character(*), intent(in) :: phase
        type(csc_t), intent(in) :: matrix
        integer :: column, entry

        do column = 1, matrix%ncol
            do entry = matrix%col_ptr(column), matrix%col_ptr(column + 1) - 1
                write (unit, '(a,",",i0,",",i0,",",es24.16)') phase, &
                    matrix%row_idx(entry), column, matrix%val(entry)
            end do
        end do
    end subroutine write_matrix

end program fortml_bench_sparse_scaler
