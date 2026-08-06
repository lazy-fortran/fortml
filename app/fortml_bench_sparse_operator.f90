program fortml_bench_sparse_operator
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortml_sparse_operator, only: sparse_gp_operator_t
    use fortsparse_status, only: FORTSPARSE_OK, fortsparse_status_t
    implicit none

    integer, parameter :: default_n_samples = 4096
    integer, parameter :: default_radius = 8
    integer, parameter :: default_n_rhs = 4
    integer, parameter :: default_repetitions = 40
    real(dp), parameter :: diagonal_shift = 0.08_dp
    integer :: n_samples, radius, n_rhs, repetitions
    integer, allocatable :: rows(:), columns(:)
    real(dp), allocatable :: values(:), input(:, :), output(:, :), expected(:, :)
    real(dp) :: elapsed, scale, sink, relative_error
    integer(int64) :: clock_start, clock_end, clock_rate
    character(16) :: mode
    integer :: nonzeros, rhs, row, repetition, storage_bytes
    type(sparse_gp_operator_t) :: sparse_operator
    type(fortsparse_status_t) :: status

    call get_command_argument(1, mode)
    if (trim(mode) /= "host" .and. trim(mode) /= "transfer" .and. &
        trim(mode) /= "resident") then
        error stop "mode must be host, transfer, or resident"
    end if
    n_samples = default_n_samples
    radius = default_radius
    n_rhs = default_n_rhs
    repetitions = default_repetitions
    call read_optional_integer(2, n_samples)
    call read_optional_integer(3, radius)
    call read_optional_integer(4, n_rhs)
    call read_optional_integer(5, repetitions)
    if (radius >= n_samples) error stop "radius must be smaller than samples"

    allocate( &
        rows(n_samples*(2*radius + 1)), &
        columns(n_samples*(2*radius + 1)), &
        values(n_samples*(2*radius + 1)), &
        input(n_samples, n_rhs), output(n_samples, n_rhs), &
        expected(n_samples, n_rhs))
    call make_compact_triplets(n_samples, radius, rows, columns, values, nonzeros)
    call sparse_operator%initialize( &
        n_samples, rows(:nonzeros), columns(:nonzeros), values(:nonzeros), status)
    if (status%code /= FORTSPARSE_OK) error stop "sparse setup failed"
    storage_bytes = 8*(n_samples + 1) + 12*nonzeros

    do rhs = 1, n_rhs
        do row = 1, n_samples
            input(row, rhs) = sin(0.013_dp*real(row + 3*rhs, dp)) + &
                0.1_dp*cos(0.017_dp*real(2*row + rhs, dp))
        end do
    end do
    call direct_compact_product(input, expected, n_samples, radius)
    call sparse_operator%matmat(input, output)
    scale = max(1.0_dp, maxval(abs(expected)))
    relative_error = maxval(abs(output - expected))/scale
    if (relative_error > 2.0e-12_dp) then
        error stop "sparse benchmark correctness oracle failed"
    end if

    if (trim(mode) == "host") then
        call system_clock(clock_start, clock_rate)
        do repetition = 1, repetitions
            call sparse_operator%matmat(input, output)
        end do
        call system_clock(clock_end)
    else if (trim(mode) == "transfer") then
        !$acc wait
        call system_clock(clock_start, clock_rate)
        do repetition = 1, repetitions
            call sparse_operator%enter_data(status)
            if (status%code /= FORTSPARSE_OK) error stop "sparse data entry failed"
            !$acc data copyin(input) copyout(output)
            call sparse_operator%matmat_device(input, output, status)
            !$acc end data
            call sparse_operator%exit_data(status)
            if (status%code /= FORTSPARSE_OK) error stop "sparse data exit failed"
        end do
        call system_clock(clock_end)
    else
        call sparse_operator%enter_data(status)
        if (status%code /= FORTSPARSE_OK) error stop "sparse data entry failed"
        !$acc data copyin(input) create(output)
        call sparse_operator%matmat_device(input, output, status)
        !$acc wait
        call system_clock(clock_start, clock_rate)
        do repetition = 1, repetitions
            call sparse_operator%matmat_device(input, output, status)
            !$acc wait
            if (status%code /= FORTSPARSE_OK) error stop "sparse device timing failed"
        end do
        call system_clock(clock_end)
        !$acc update self(output)
        !$acc end data
        call sparse_operator%exit_data(status)
        if (status%code /= FORTSPARSE_OK) error stop "sparse data exit failed"
    end if
    elapsed = real(clock_end - clock_start, dp)/real(clock_rate, dp)
    sink = output(1, 1) + output(n_samples, n_rhs)
    if (sink /= sink) error stop "sparse benchmark produced NaN"
    write (*, '(a,i0,a,i0,a,i0,a,a,a,i0,a,i0,a,i0,a,es24.16,a,es24.16)') &
        "sparse_operator,", n_samples, ",", radius, ",", n_rhs, ",", &
        trim(mode), ",", repetitions, ",", nonzeros, ",", storage_bytes, ",", &
        elapsed/real(repetitions, dp), ",", relative_error

contains

    subroutine make_compact_triplets(n, support, output_rows, output_columns, &
            output_values, count)
        integer, intent(in) :: n, support
        integer, intent(out) :: output_rows(:), output_columns(:), count
        real(dp), intent(out) :: output_values(:)
        integer :: column, distance, row
        real(dp) :: normalized_distance

        count = 0
        do column = 1, n
            do distance = -support, support
                row = column + distance
                if (row < 1 .or. row > n) cycle
                count = count + 1
                output_rows(count) = row
                output_columns(count) = column
                normalized_distance = real(abs(distance), dp)/real(support, dp)
                output_values(count) = &
                    wendland_c2(normalized_distance) + merge( &
                    diagonal_shift, 0.0_dp, distance == 0)
            end do
        end do
    end subroutine make_compact_triplets

    subroutine direct_compact_product(input_values, output_values, n, support)
        real(dp), intent(in) :: input_values(:, :)
        real(dp), intent(out) :: output_values(:, :)
        integer, intent(in) :: n, support
        integer :: column, distance, row, rhs
        real(dp) :: normalized_distance

        output_values = 0.0_dp
        do rhs = 1, size(input_values, 2)
            do row = 1, n
                do distance = -support, support
                    column = row + distance
                    if (column < 1 .or. column > n) cycle
                    normalized_distance = &
                        real(abs(distance), dp)/real(support, dp)
                    output_values(row, rhs) = output_values(row, rhs) + &
                        (wendland_c2(normalized_distance) + merge( &
                        diagonal_shift, 0.0_dp, distance == 0))* &
                        input_values(column, rhs)
                end do
            end do
        end do
    end subroutine direct_compact_product

    pure real(dp) function wendland_c2(distance) result(value)
        real(dp), intent(in) :: distance
        real(dp) :: remainder

        if (distance >= 1.0_dp) then
            value = 0.0_dp
        else
            remainder = 1.0_dp - distance
            value = remainder**4*(4.0_dp*distance + 1.0_dp)
        end if
    end function wendland_c2

    subroutine read_optional_integer(number, value)
        integer, intent(in) :: number
        integer, intent(inout) :: value
        character(64) :: argument
        integer :: candidate, io_status

        call get_command_argument(number, argument)
        if (len_trim(argument) == 0) return
        read(argument, *, iostat=io_status) candidate
        if (io_status /= 0) error stop "invalid integer benchmark argument"
        if (candidate < 1) error stop "benchmark argument must be positive"
        value = candidate
    end subroutine read_optional_integer

end program fortml_bench_sparse_operator
