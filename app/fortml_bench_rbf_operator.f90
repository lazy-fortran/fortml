program fortml_bench_rbf_operator
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortml_kernel_operator, only: rbf_operator_t
    use fortnum_status, only: FORTNUM_OK, fortnum_status_t
    implicit none

    integer, parameter :: default_n_samples = 2048
    integer, parameter :: default_n_features = 8
    integer, parameter :: default_repetitions = 12
    integer, parameter :: tile_size = 128
    integer :: n_samples, n_features, repetitions
    real(dp), parameter :: variance = 1.4_dp
    real(dp), parameter :: lengthscale = 0.7_dp
    real(dp), parameter :: diagonal_shift = 0.08_dp
    real(dp), allocatable :: sample_points(:, :), input(:), output(:), expected(:)
    real(dp) :: elapsed, sink, scale
    integer(int64) :: clock_start, clock_end, clock_rate
    character(16) :: mode
    integer :: feature, i, j, repetition
    type(rbf_operator_t) :: rbf_operator
    type(fortnum_status_t) :: status

    call get_command_argument(1, mode)
    if (trim(mode) /= "cpu" .and. trim(mode) /= "transfer" .and. &
        trim(mode) /= "resident") then
        error stop "mode must be cpu, transfer, or resident"
    end if
    n_samples = default_n_samples
    n_features = default_n_features
    repetitions = default_repetitions
    call read_optional_integer(2, n_samples)
    call read_optional_integer(3, n_features)
    call read_optional_integer(4, repetitions)
    allocate(sample_points(n_samples, n_features), input(n_samples), output(n_samples), &
        expected(n_samples))

    do feature = 1, n_features
        do i = 1, n_samples
            sample_points(i, feature) = sin( &
                0.013_dp*real(i + 3*feature, dp)) + &
                0.1_dp*cos(0.017_dp*real(i*feature, dp))
        end do
    end do
    do i = 1, n_samples
        input(i) = sin(0.021_dp*real(i, dp)) + &
            0.3_dp*cos(0.007_dp*real(2*i + 1, dp))
    end do
    call rbf_operator%initialize( &
        sample_points, variance, lengthscale, diagonal_shift, status, tile_size)
    if (status%code /= FORTNUM_OK) error stop "RBF operator initialization failed"

    call rbf_operator%matvec(input, output)
    do i = 1, n_samples
        call direct_value(i, expected(i))
    end do
    scale = max(1.0_dp, maxval(abs(expected)))
    if (maxval(abs(output - expected)) > 2.0e-12_dp*scale) then
        error stop "RBF benchmark correctness oracle failed"
    end if

    call system_clock(clock_start, clock_rate)
    if (trim(mode) == "resident") then
        !$acc data copyin(rbf_operator%points, input) create(output)
        do repetition = 1, repetitions
            call rbf_operator%matvec(input, output)
        end do
        !$acc update self(output)
        !$acc end data
    else
        do repetition = 1, repetitions
            call rbf_operator%matvec(input, output)
        end do
    end if
    call system_clock(clock_end)
    elapsed = real(clock_end - clock_start, dp)/real(clock_rate, dp)
    sink = output(1) + output(n_samples)
    if (sink /= sink) error stop "RBF benchmark produced NaN"
    write (*, '(a,i0,a,i0,a,a,a,i0,a,es24.16)') &
        "rbf_operator,", n_samples, ",", n_features, ",", trim(mode), ",", &
        repetitions, ",", elapsed/real(repetitions, dp)

contains

    subroutine direct_value(row, value)
        integer, intent(in) :: row
        real(dp), intent(out) :: value
        real(dp) :: squared_distance, difference

        value = diagonal_shift*input(row)
        do j = 1, n_samples
            squared_distance = 0.0_dp
            do feature = 1, n_features
                difference = sample_points(row, feature) - &
                    sample_points(j, feature)
                squared_distance = squared_distance + difference*difference
            end do
            value = value + variance*exp( &
                -0.5_dp*squared_distance/(lengthscale*lengthscale))*input(j)
        end do
    end subroutine direct_value

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

end program fortml_bench_rbf_operator
