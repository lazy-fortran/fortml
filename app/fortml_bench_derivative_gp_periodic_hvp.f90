program fortml_bench_derivative_gp_periodic_hvp
    !! CPU timing lane for the mixed-observation periodic GP HVP.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortml_derivative_gaussian_process, only: gp_derivative_regression_t
    use fortml_kernels, only: kernel_t, make_periodic_kernel
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer, parameter :: n = 5, d = 2, repetitions = 32
    real(dp) :: x_train(n, d), y_train(n, 1), direction(4), hvp(4)
    integer :: components(n), repetition
    type(kernel_t) :: kernel
    type(gp_derivative_regression_t) :: model
    type(fortnum_status_t) :: status
    integer(int64) :: begin_clock, end_clock, rate
    real(dp) :: seconds

    x_train = reshape([ &
        -0.70_dp, 0.20_dp, 0.35_dp, -0.40_dp, 0.90_dp, 0.65_dp, &
        -0.10_dp, 1.10_dp, 0.55_dp, -0.85_dp], shape(x_train))
    y_train(:, 1) = [0.80_dp, -0.25_dp, 0.45_dp, 1.10_dp, -0.60_dp]
    components = [0, 1, 2, 0, 1]
    direction = [0.13_dp, -0.08_dp, 0.11_dp, -0.05_dp]
    kernel = make_periodic_kernel(d, 1.25_dp, 0.83_dp, 1.47_dp, status)
    if (.not. status_ok(status)) error stop "periodic constructor failed"
    call model%fit(x_train, components, y_train, kernel, 0.055_dp, status, &
        jitter=1.0e-10_dp)
    if (.not. status_ok(status)) error stop "periodic derivative GP fit failed"
    call model%hyperparameter_hvp(direction, hvp, status)
    if (.not. status_ok(status)) error stop "periodic derivative GP HVP failed"

    call system_clock(begin_clock, rate)
    do repetition = 1, repetitions
        call model%hyperparameter_hvp(direction, hvp, status)
        if (.not. status_ok(status)) error stop "periodic derivative GP HVP failed"
    end do
    call system_clock(end_clock)
    seconds = real(end_clock - begin_clock, dp)/real(rate, dp)/real(repetitions, dp)
    write (*, '(a,a,a,es24.16,a,es24.16)') "derivative_gp_periodic,", &
        "periodic,", "hvp,", seconds, ",", sum(hvp)
end program fortml_bench_derivative_gp_periodic_hvp
