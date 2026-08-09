program fortml_bench_poisson_likelihood
    !! Release probe for Poisson likelihood products, fitting, and dispatch.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortnum_status, only: fortnum_status_t, status_ok
    use fortml_device, only: FORTML_DEVICE_CUDA
    use fortml_poisson_likelihood, only: poisson_likelihood_t, &
        poisson_likelihood_lbfgsb_options_t, poisson_likelihood_lbfgsb_result_t, &
        poisson_log_likelihood_value_gradient, poisson_log_likelihood_hvp
    implicit none

    integer, parameter :: repeats = 20000
    real(dp) :: observations(6), latents(6), weights(6), latent_gradient(6)
    real(dp) :: latent_direction(6), latent_product(6), log_rate_offset, value
    real(dp) :: offset_gradient, offset_product, elapsed, rate
    real(dp), allocatable :: optimized(:)
    integer(int64) :: tick_start, tick_end, tick_rate
    integer :: i, cuda_code
    type(poisson_likelihood_t) :: model
    type(poisson_likelihood_lbfgsb_options_t) :: options
    type(poisson_likelihood_lbfgsb_result_t) :: result
    type(fortnum_status_t) :: status

    observations = [0.0_dp, 1.0_dp, 2.0_dp, 4.0_dp, 3.0_dp, 5.0_dp]
    latents = log([0.6_dp, 1.0_dp, 1.4_dp, 2.8_dp, 4.1_dp, 2.0_dp])
    weights = [1.0_dp, 0.5_dp, 2.0_dp, 1.5_dp, 0.0_dp, 0.75_dp]
    latent_direction = [0.3_dp, -0.2_dp, 0.1_dp, 0.45_dp, -0.4_dp, 0.25_dp]
    log_rate_offset = log(1.3_dp)

    call system_clock(tick_start, tick_rate)
    do i = 1, repeats
        call poisson_log_likelihood_value_gradient(observations, latents, log_rate_offset, &
            value, latent_gradient, offset_gradient, status, weights)
        if (.not. status_ok(status)) error stop "Poisson likelihood gradient benchmark failed"
        call poisson_log_likelihood_hvp(observations, latents, log_rate_offset, &
            latent_direction, -0.35_dp, latent_product, offset_product, status, weights)
        if (.not. status_ok(status)) error stop "Poisson likelihood HVP benchmark failed"
    end do
    call system_clock(tick_end)
    elapsed = real(tick_end - tick_start, dp)/real(tick_rate, dp)
    rate = real(repeats, dp)/max(elapsed, tiny(1.0_dp))

    call model%initialize(observations, latents, status, log_rate_offset=-0.4_dp, &
        sample_weight=weights)
    if (.not. status_ok(status)) error stop "Poisson likelihood benchmark init failed"
    options%gradient_tolerance = 1.0e-8_dp
    call system_clock(tick_start)
    call model%optimize_lbfgsb(options, result, status)
    call system_clock(tick_end)
    if (.not. status_ok(status)) error stop "Poisson likelihood benchmark fit failed"
    optimized = model%parameters()

    call model%initialize(observations, latents, status, device_kind=FORTML_DEVICE_CUDA)
    cuda_code = status%code

    write (*, '(a,i0)') "poisson_likelihood_repeats,", repeats
    write (*, '(a,es24.16)') "poisson_likelihood_product_seconds,", elapsed
    write (*, '(a,es24.16)') "poisson_likelihood_products_per_second,", rate
    write (*, '(a,es24.16)') "poisson_likelihood_value,", value
    write (*, '(a,es24.16)') "poisson_likelihood_offset_gradient,", offset_gradient
    write (*, '(a,es24.16)') "poisson_likelihood_offset_hvp,", offset_product
    elapsed = real(tick_end - tick_start, dp)/real(tick_rate, dp)
    write (*, '(a,es24.16)') "poisson_likelihood_fit_seconds,", elapsed
    write (*, '(a,es24.16)') "poisson_likelihood_fitted_log_rate_offset,", optimized(1)
    write (*, '(a,es24.16)') "poisson_likelihood_fit_gradient_norm,", result%gradient_norm
    write (*, '(a,i0)') "poisson_likelihood_fit_iterations,", result%iterations
    write (*, '(a,i0)') "poisson_likelihood_cuda_status,", cuda_code
end program fortml_bench_poisson_likelihood
