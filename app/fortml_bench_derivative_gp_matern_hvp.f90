program fortml_bench_derivative_gp_matern_hvp
    !! CPU timing lane for mixed-observation Matérn likelihood HVPs.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortml_derivative_gaussian_process, only: gp_derivative_regression_t
    use fortml_kernels, only: kernel_t, make_matern32_kernel, make_matern52_kernel
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer, parameter :: n = 4, d = 1, repetitions = 32
    real(dp) :: x_train(n, d), y_train(n, 1), direction(3), hvp(3)
    integer :: components(n), repetition
    type(kernel_t) :: kernel
    type(gp_derivative_regression_t) :: model
    type(fortnum_status_t) :: status
    integer(int64) :: begin_clock, end_clock, rate
    real(dp) :: seconds

    x_train(:, 1) = [-0.20_dp, 0.35_dp, 0.90_dp, 1.40_dp]
    y_train(:, 1) = [0.70_dp, -0.10_dp, 0.55_dp, -0.35_dp]
    components = [0, 1, 0, 1]
    direction = [0.17_dp, -0.11_dp, 0.08_dp]

    kernel = make_matern32_kernel(d, 1.35_dp, 0.78_dp, status)
    if (.not. status_ok(status)) error stop "Matern32 constructor failed"
    call model%fit(x_train, components, y_train, kernel, 0.045_dp, status, &
        jitter=1.0e-10_dp)
    if (.not. status_ok(status)) error stop "Matern32 derivative GP fit failed"
    call model%hyperparameter_hvp(direction, hvp, status)
    if (.not. status_ok(status)) error stop "Matern32 derivative GP HVP failed"
    call system_clock(begin_clock, rate)
    do repetition = 1, repetitions
        call model%hyperparameter_hvp(direction, hvp, status)
        if (.not. status_ok(status)) error stop "Matern32 derivative GP HVP failed"
    end do
    call system_clock(end_clock)
    seconds = real(end_clock - begin_clock, dp)/real(rate, dp)/real(repetitions, dp)
    write (*, '(a,a,a,es24.16,a,es24.16)') "derivative_gp_matern,", &
        "matern32,", "hvp,", seconds, ",", sum(hvp)

    kernel = make_matern52_kernel(d, 1.35_dp, 0.78_dp, status)
    if (.not. status_ok(status)) error stop "Matern52 constructor failed"
    call model%fit(x_train, components, y_train, kernel, 0.045_dp, status, &
        jitter=1.0e-10_dp)
    if (.not. status_ok(status)) error stop "Matern52 derivative GP fit failed"
    call model%hyperparameter_hvp(direction, hvp, status)
    if (.not. status_ok(status)) error stop "Matern52 derivative GP HVP failed"
    call system_clock(begin_clock, rate)
    do repetition = 1, repetitions
        call model%hyperparameter_hvp(direction, hvp, status)
        if (.not. status_ok(status)) error stop "Matern52 derivative GP HVP failed"
    end do
    call system_clock(end_clock)
    seconds = real(end_clock - begin_clock, dp)/real(rate, dp)/real(repetitions, dp)
    write (*, '(a,a,a,es24.16,a,es24.16)') "derivative_gp_matern,", &
        "matern52,", "hvp,", seconds, ",", sum(hvp)
end program fortml_bench_derivative_gp_matern_hvp
