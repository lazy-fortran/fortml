program fortml_bench_rbf_operator
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortml_kernel_operator, only: rbf_operator_t
    use fortnum_status, only: FORTNUM_OK, fortnum_status_t
    implicit none

    integer, parameter :: n_samples = 2048
    integer, parameter :: n_features = 8
    integer, parameter :: repetitions = 12
    integer, parameter :: tile_size = 128
    real(dp), parameter :: variance = 1.4_dp
    real(dp), parameter :: lengthscale = 0.7_dp
    real(dp), parameter :: diagonal_shift = 0.08_dp
    real(dp) :: sample_points(n_samples, n_features), input(n_samples)
    real(dp) :: output(n_samples), expected_first, expected_last
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
    call direct_value(1, expected_first)
    call direct_value(n_samples, expected_last)
    scale = max(1.0_dp, abs(expected_first), abs(expected_last))
    if (max(abs(output(1) - expected_first), &
        abs(output(n_samples) - expected_last)) > 2.0e-12_dp*scale) then
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

end program fortml_bench_rbf_operator
