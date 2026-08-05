program fortml_bench_rbf_cg
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit, int64
    use fortml_kernel_operator, only: rbf_operator_t
    use fortnum_krylov, only: KRYLOV_OK
    use fortnum_status, only: FORTNUM_OK, fortnum_status_t
    implicit none

    integer, parameter :: default_n_samples = 2048
    integer, parameter :: default_n_features = 8
    integer, parameter :: default_repetitions = 8
    integer, parameter :: tile_size = 128
    real(dp), parameter :: variance = 1.4_dp
    real(dp), parameter :: lengthscale = 0.7_dp
    real(dp), parameter :: diagonal_shift = 0.08_dp
    real(dp), parameter :: tolerance = 1.0e-8_dp
    integer, parameter :: max_iterations = 500
    real(dp), allocatable :: sample_points(:, :), right_hand_side(:), solution(:)
    real(dp) :: elapsed, residual_norm, target
    integer(int64) :: clock_start, clock_end, clock_rate
    integer :: feature, i, n_samples, n_features, repetitions
    integer :: info, iterations, repetition
    type(rbf_operator_t) :: rbf_operator
    type(fortnum_status_t) :: status

    n_samples = default_n_samples
    n_features = default_n_features
    repetitions = default_repetitions
    call read_optional_integer(1, n_samples)
    call read_optional_integer(2, n_features)
    call read_optional_integer(3, repetitions)
    allocate(sample_points(n_samples, n_features), right_hand_side(n_samples), &
        solution(n_samples))

    do feature = 1, n_features
        do i = 1, n_samples
            sample_points(i, feature) = sin( &
                0.013_dp*real(i + 3*feature, dp)) + &
                0.1_dp*cos(0.017_dp*real(i*feature, dp))
        end do
    end do
    do i = 1, n_samples
        right_hand_side(i) = sin(0.021_dp*real(i, dp)) + &
            0.3_dp*cos(0.007_dp*real(2*i + 1, dp))
    end do
    target = tolerance*max(sqrt(sum(right_hand_side*right_hand_side)), 1.0_dp)
    call rbf_operator%initialize( &
        sample_points, variance, lengthscale, diagonal_shift, status, tile_size)
    if (status%code /= FORTNUM_OK) error stop "RBF operator initialization failed"

    solution = 0.0_dp
    call rbf_operator%solve_cg( &
        right_hand_side, solution, tolerance, max_iterations, info, &
        iterations, residual_norm, use_diagonal_preconditioner=.false.)
    if (info /= KRYLOV_OK .or. residual_norm > target) then
        write (error_unit, '(a,i0,a,i0,a,es24.16)') &
            "cg_check_info=", info, ",iterations=", iterations, &
            ",residual=", residual_norm
        error stop "matrix-free CG correctness check failed"
    end if

    call system_clock(clock_start, clock_rate)
    !$acc data copyin(rbf_operator%points, right_hand_side)
    do repetition = 1, repetitions
        solution = 0.0_dp
        call rbf_operator%solve_cg( &
            right_hand_side, solution, tolerance, max_iterations, info, &
            iterations, residual_norm, use_diagonal_preconditioner=.false.)
        if (info /= KRYLOV_OK) error stop "matrix-free CG timed solve failed"
    end do
    !$acc end data
    call system_clock(clock_end)
    elapsed = real(clock_end - clock_start, dp)/real(clock_rate, dp)
    write (*, '(a,i0,a,i0,a,i0,a,es24.16,a,i0,a,es24.16)') &
        "rbf_cg,", n_samples, ",", n_features, ",", repetitions, ",", &
        elapsed/real(repetitions, dp), ",", iterations, ",", residual_norm

contains

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

end program fortml_bench_rbf_cg
