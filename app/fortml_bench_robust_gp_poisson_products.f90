program fortml_bench_robust_gp_poisson_products
    !! Release timing for Poisson likelihood and fixed-latent GP HVP products.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortnum_status, only: fortnum_status_t, status_ok
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortml_kernels, only: kernel_t, make_rbf_kernel
    use fortml_robust_gp, only: robust_gp_t, FORTML_LIKELIHOOD_POISSON, &
        robust_poisson_log_likelihood_hvp
    implicit none

    integer, parameter :: n_samples = 8, repetitions = 64
    real(dp) :: counts(4), log_rate(4), direction(4), likelihood_hvp(4)
    real(dp) :: posterior_direction(n_samples)
    real(dp) :: x(n_samples, 1), targets(n_samples), posterior_hvp(n_samples)
    real(dp) :: response(n_samples), seconds
    integer(int64) :: clock_start, clock_end, clock_rate
    type(robust_gp_t) :: model
    type(kernel_t) :: kernel
    type(fortml_device_t) :: cuda
    type(fortnum_status_t) :: status
    integer :: i, code

    counts = [0.0_dp, 1.0_dp, 3.0_dp, 2.0_dp]
    log_rate = [-0.3_dp, 0.2_dp, 0.8_dp, -0.1_dp]
    direction = [0.4_dp, -0.2_dp, 0.1_dp, 0.3_dp]
    call robust_poisson_log_likelihood_hvp(counts, log_rate, direction, likelihood_hvp, status)
    if (.not. status_ok(status)) error stop "Poisson likelihood HVP warmup failed"
    call system_clock(clock_start, clock_rate)
    do i = 1, repetitions
        call robust_poisson_log_likelihood_hvp(counts, log_rate, direction, likelihood_hvp, status)
        if (.not. status_ok(status)) error stop "Poisson likelihood HVP failed"
    end do
    call system_clock(clock_end)
    seconds = real(clock_end - clock_start, dp)/real(clock_rate, dp)/real(repetitions, dp)
    write (*, '(a,es24.16,a,es24.16)') &
        "robust_gp_poisson_products,likelihood,cpu,seconds,", seconds, &
        ",hvp_checksum,", sum(likelihood_hvp)

    do i = 1, n_samples
        x(i, 1) = -1.4_dp + 0.4_dp*real(i - 1, dp)
        targets(i) = real(max(0, nint(3.0_dp*exp(0.6_dp*x(i, 1)))), dp)
    end do
    kernel = make_rbf_kernel(1, 1.0_dp, 0.9_dp, status)
    call model%fit(x, targets, kernel, FORTML_LIKELIHOOD_POISSON, status)
    if (.not. status_ok(status)) error stop "Poisson GP fit failed"
    posterior_direction = 0.0_dp
    posterior_direction(1) = 0.25_dp
    call model%log_posterior_hvp(posterior_direction, posterior_hvp, status)
    if (.not. status_ok(status)) error stop "Poisson posterior HVP warmup failed"
    call system_clock(clock_start, clock_rate)
    do i = 1, repetitions
        call model%log_posterior_hvp(posterior_direction, posterior_hvp, status)
        if (.not. status_ok(status)) error stop "Poisson posterior HVP failed"
    end do
    call system_clock(clock_end)
    seconds = real(clock_end - clock_start, dp)/real(clock_rate, dp)/real(repetitions, dp)
    write (*, '(a,es24.16,a,es24.16)') &
        "robust_gp_poisson_products,posterior,cpu,seconds,", seconds, &
        ",hvp_checksum,", sum(posterior_hvp)

    call model%predict_response(x, response, status)
    if (.not. status_ok(status)) error stop "Poisson response warmup failed"
    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call model%predict_response_device(cuda, x, response, status)
    code = status%code
    write (*, '(a,i0)') "robust_gp_poisson_products,device,cuda,refused,", code
end program fortml_bench_robust_gp_poisson_products
