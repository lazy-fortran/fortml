program fortml_bench_derivative_gp_spectral_mixture_hvp
    !! CPU timing lane for the mixed-observation spectral-mixture GP HVP.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortml_derivative_gaussian_process, only: gp_derivative_regression_t
    use fortml_kernels, only: kernel_t, make_spectral_mixture_kernel
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer, parameter :: n = 5, d = 2, q = 2, repetitions = 32
    real(dp) :: x_train(n, d), y_train(n, 1), weights(q), means(q, d), scales(q, d)
    real(dp) :: direction(11), hvp(11)
    integer :: components(n), repetition
    type(kernel_t) :: kernel
    type(gp_derivative_regression_t) :: model
    type(fortnum_status_t) :: status
    integer(int64) :: begin_clock, end_clock, rate
    real(dp) :: seconds

    x_train = reshape([ &
        -0.7_dp, 0.2_dp, 0.35_dp, -0.4_dp, 0.9_dp, 0.65_dp, &
        -0.1_dp, 1.1_dp, 0.55_dp, -0.85_dp], shape(x_train))
    y_train(:, 1) = [0.8_dp, -0.25_dp, 0.45_dp, 1.1_dp, -0.6_dp]
    components = [0, 1, 2, 0, 1]
    weights = [1.15_dp, 0.63_dp]
    means = reshape([0.21_dp, -0.37_dp, 0.48_dp, 0.16_dp], shape(means))
    scales = reshape([0.31_dp, 0.57_dp, 0.22_dp, 0.44_dp], shape(scales))
    direction = [0.13_dp, -0.08_dp, 0.11_dp, -0.05_dp, 0.07_dp, -0.04_dp, &
        0.09_dp, -0.06_dp, 0.12_dp, -0.03_dp, 0.17_dp]
    kernel = make_spectral_mixture_kernel(d, q, weights, means, scales, status)
    if (.not. status_ok(status)) error stop "spectral-mixture constructor failed"
    call model%fit(x_train, components, y_train, kernel, 0.055_dp, status, &
        jitter=1.0e-10_dp)
    if (.not. status_ok(status)) error stop "spectral-mixture derivative GP fit failed"
    call model%hyperparameter_hvp(direction, hvp, status)
    if (.not. status_ok(status)) error stop "spectral-mixture derivative GP HVP failed"

    call system_clock(begin_clock, rate)
    do repetition = 1, repetitions
        call model%hyperparameter_hvp(direction, hvp, status)
        if (.not. status_ok(status)) error stop "spectral-mixture derivative GP HVP failed"
    end do
    call system_clock(end_clock)
    seconds = real(end_clock - begin_clock, dp)/real(rate, dp)/real(repetitions, dp)
    write (*, '(a,a,a,es24.16,a,es24.16)') "derivative_gp_spectral_mixture,", &
        "spectral_mixture,", "hvp,", seconds, ",", sum(hvp)
end program fortml_bench_derivative_gp_spectral_mixture_hvp
