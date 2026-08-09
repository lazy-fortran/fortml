program fortml_bench_relu_nngp
    !! Deterministic analytic ReLU NNGP covariance workload.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortml_relu_nngp, only: relu_nngp_t
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer, parameter :: n_left = 192, n_right = 160, n_features = 8
    integer, parameter :: depth = 3, repetitions = 8
    real(dp) :: x_left(n_left, n_features), x_right(n_right, n_features)
    real(dp), allocatable :: covariance(:, :)
    real(dp) :: seconds, checksum
    integer(int64) :: clock_start, clock_end, clock_rate
    integer :: i, j, repetition
    type(relu_nngp_t) :: kernel
    type(fortnum_status_t) :: status

    do j = 1, n_features
        do i = 1, n_left
            x_left(i, j) = sin(0.017_dp*real(i, dp) + 0.053_dp*real(j, dp)) + &
                cos(0.011_dp*real(i*j, dp))
        end do
        do i = 1, n_right
            x_right(i, j) = sin(0.019_dp*real(i, dp) + 0.041_dp*real(j, dp)) + &
                cos(0.007_dp*real(i*j, dp))
        end do
    end do
    call kernel%configure(n_features, depth, status, weight_variance=2.0_dp)
    if (.not. status_ok(status)) error stop "ReLU NNGP configuration failed"
    call system_clock(clock_start, clock_rate)
    do repetition = 1, repetitions
        call kernel%covariance(x_left, x_right, covariance, status)
        if (.not. status_ok(status)) error stop "ReLU NNGP covariance failed"
    end do
    call system_clock(clock_end)
    seconds = real(clock_end - clock_start, dp)/real(clock_rate, dp)/real(repetitions, dp)
    checksum = sum(covariance)
    write (*, '(a,i0,a,i0,a,i0,a,es24.16,a,es24.16)') "relu_nngp_covariance,", &
        n_left, ",", n_right, ",", depth, ",", seconds, ",", checksum
end program fortml_bench_relu_nngp
