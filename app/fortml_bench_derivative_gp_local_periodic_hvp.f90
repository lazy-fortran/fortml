program fortml_bench_derivative_gp_local_periodic_hvp
    !! CPU timing lane for the mixed-observation local-periodic GP HVP.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortml_derivative_gaussian_process, only: gp_derivative_regression_t
    use fortml_kernels, only: kernel_t, make_local_periodic_kernel
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer, parameter :: n = 5, d = 2, repetitions = 32
    real(dp) :: x_train(n, d), y_train(n, 1), direction(5), hvp(5)
    integer :: components(n), repetition
    type(kernel_t) :: kernel
    type(gp_derivative_regression_t) :: model
    type(fortnum_status_t) :: status
    integer(int64) :: begin_clock, end_clock, rate
    real(dp) :: seconds

    x_train = reshape([ &
        -0.80_dp, 0.10_dp, 0.35_dp, -0.55_dp, 0.90_dp, 0.72_dp, &
        -0.15_dp, 1.05_dp, 0.60_dp, -0.95_dp], shape(x_train))
    y_train(:, 1) = [0.70_dp, -0.20_dp, 0.95_dp, 0.30_dp, -0.65_dp]
    components = [0, 1, 2, 1, 0]
    direction = [0.11_dp, -0.08_dp, 0.14_dp, -0.06_dp, 0.17_dp]
    kernel = make_local_periodic_kernel(d, 1.30_dp, 0.85_dp, 0.62_dp, 1.70_dp, status)
    if (.not. status_ok(status)) error stop "local-periodic constructor failed"
    call model%fit(x_train, components, y_train, kernel, 0.045_dp, status, &
        jitter=1.0e-10_dp)
    if (.not. status_ok(status)) error stop "local-periodic derivative GP fit failed"
    call model%hyperparameter_hvp(direction, hvp, status)
    if (.not. status_ok(status)) error stop "local-periodic derivative GP HVP failed"

    call system_clock(begin_clock, rate)
    do repetition = 1, repetitions
        call model%hyperparameter_hvp(direction, hvp, status)
        if (.not. status_ok(status)) error stop "local-periodic derivative GP HVP failed"
    end do
    call system_clock(end_clock)
    seconds = real(end_clock - begin_clock, dp)/real(rate, dp)/real(repetitions, dp)
    write (*, '(a,a,a,es24.16,a,es24.16)') "derivative_gp_local_periodic,", &
        "local_periodic,", "hvp,", seconds, ",", sum(hvp)
end program fortml_bench_derivative_gp_local_periodic_hvp
