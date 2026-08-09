program fortml_bench_multi_output_gp_hypergradients
    !! Release benchmark for exact ICM likelihood products.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortnum_status, only: fortnum_status_t, status_ok
    use fortml_kernels, only: kernel_t, make_rbf_kernel
    use fortml_multi_output_gp, only: multi_output_gp_t
    implicit none

    integer, parameter :: n = 40, d = 2, p = 3, rank = 2, repetitions = 5
    real(dp) :: x(n, d), y(n, p), weights(p, rank), independent(p)
    real(dp), allocatable :: direction(:), gradient(:), hvp(:), parameters(:)
    real(dp) :: value, value_dot, elapsed, checksum
    integer(int64) :: clock_start, clock_end, clock_rate
    integer :: i, k
    type(kernel_t) :: kernel
    type(multi_output_gp_t) :: model
    type(fortnum_status_t) :: status

    do i = 1, n
        x(i, 1) = -1.0_dp + 0.05_dp*real(i - 1, dp)
        x(i, 2) = sin(0.13_dp*real(i, dp))
        y(i, 1) = sin(0.8_dp*x(i, 1)) + 0.1_dp*x(i, 2)
        y(i, 2) = cos(0.6_dp*x(i, 1)) - 0.2_dp*x(i, 2)
        y(i, 3) = x(i, 1)*x(i, 2)
    end do
    weights = reshape([0.8_dp, -0.4_dp, 0.3_dp, 0.6_dp, -0.2_dp, 0.5_dp], [p, rank])
    independent = [0.25_dp, 0.35_dp, 0.18_dp]
    kernel = make_rbf_kernel(d, 1.2_dp, 0.65_dp, status)
    call model%initialize(kernel, weights, independent, 0.12_dp, status)
    call model%fit(x, y, status)
    if (.not. status_ok(status)) error stop "multi-output hypergradient benchmark fit failed"
    parameters = model%parameters()
    allocate(direction(size(parameters)), gradient(size(parameters)), hvp(size(parameters)))
    do k = 1, size(direction)
        direction(k) = 0.03_dp*sin(0.17_dp*real(k, dp))
    end do

    call system_clock(clock_start, clock_rate)
    do k = 1, repetitions
        call model%hyperparameter_gradient(gradient, status)
    end do
    call system_clock(clock_end)
    if (.not. status_ok(status)) error stop "multi-output hypergradient benchmark gradient failed"
    elapsed = real(clock_end - clock_start, dp)/real(clock_rate, dp)/real(repetitions, dp)
    checksum = sum(gradient)
    write (*, '(a,",",i0,",",i0,",",i0,",",es24.16,",",es24.16)') &
        "multi_output_gp,hyperparameter_gradient", n, p, size(parameters), elapsed, checksum

    call system_clock(clock_start, clock_rate)
    do k = 1, repetitions
        call model%hyperparameter_hvp(direction, hvp, status)
    end do
    call system_clock(clock_end)
    if (.not. status_ok(status)) error stop "multi-output hypergradient benchmark HVP failed"
    elapsed = real(clock_end - clock_start, dp)/real(clock_rate, dp)/real(repetitions, dp)
    checksum = sum(hvp)
    write (*, '(a,",",i0,",",i0,",",i0,",",es24.16,",",es24.16)') &
        "multi_output_gp,hyperparameter_hvp", n, p, size(parameters), elapsed, checksum

    call model%log_marginal_likelihood(y, value, status)
    call model%log_marginal_likelihood_jvp(direction, value_dot, status)
    if (.not. status_ok(status)) error stop "multi-output likelihood product benchmark failed"
    write (*, '(a,",",es24.16,",",es24.16)') &
        "multi_output_gp,likelihood_value_jvp", value, value_dot
end program fortml_bench_multi_output_gp_hypergradients
