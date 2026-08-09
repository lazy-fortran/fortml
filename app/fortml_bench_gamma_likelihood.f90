program fortml_bench_gamma_likelihood
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortnum_status, only: fortnum_status_t, status_ok
    use fortml_device, only: FORTML_DEVICE_CUDA
    use fortml_gamma_likelihood, only: gamma_likelihood_t, &
        gamma_likelihood_lbfgsb_options_t, gamma_likelihood_lbfgsb_result_t, &
        gamma_log_likelihood_value_gradient, gamma_log_likelihood_hvp
    implicit none

    integer, parameter :: repeats = 20000
    real(dp) :: observations(6), latents(6), weights(6), latent_gradient(6)
    real(dp) :: latent_direction(6), latent_product(6), log_shape, value
    real(dp) :: shape_gradient, shape_product, elapsed, rate
    real(dp), allocatable :: optimized(:)
    integer(int64) :: tick_start, tick_end, tick_rate
    integer :: i, cuda_code
    type(gamma_likelihood_t) :: model
    type(gamma_likelihood_lbfgsb_options_t) :: options
    type(gamma_likelihood_lbfgsb_result_t) :: result
    type(fortnum_status_t) :: status

    observations = [0.35_dp, 0.8_dp, 1.7_dp, 3.2_dp, 5.5_dp, 2.4_dp]
    latents = log([0.5_dp, 1.0_dp, 1.4_dp, 2.8_dp, 4.1_dp, 2.0_dp])
    weights = [1.0_dp, 0.25_dp, 2.0_dp, 1.5_dp, 0.0_dp, 0.75_dp]
    latent_direction = [0.3_dp, -0.2_dp, 0.1_dp, 0.45_dp, -0.4_dp, 0.25_dp]
    log_shape = log(1.7_dp)

    call system_clock(tick_start, tick_rate)
    do i = 1, repeats
        call gamma_log_likelihood_value_gradient(observations, latents, log_shape, &
            value, latent_gradient, shape_gradient, status, weights)
        if (.not. status_ok(status)) &
            error stop "Gamma likelihood gradient benchmark failed"
        call gamma_log_likelihood_hvp(observations, latents, log_shape, &
            latent_direction, -0.35_dp, latent_product, shape_product, status, weights)
        if (.not. status_ok(status)) error stop "Gamma likelihood HVP benchmark failed"
    end do
    call system_clock(tick_end)
    elapsed = real(tick_end - tick_start, dp)/real(tick_rate, dp)
    rate = real(repeats, dp)/max(elapsed, tiny(1.0_dp))

    call model%initialize(observations, latents, status, log(0.2_dp), weights)
    if (.not. status_ok(status)) error stop "Gamma likelihood benchmark init failed"
    call system_clock(tick_start)
    call model%optimize_lbfgsb(options, result, status)
    call system_clock(tick_end)
    if (.not. status_ok(status)) error stop "Gamma likelihood benchmark fit failed"
    optimized = model%parameters()

    call model%initialize(observations, latents, status, log_shape, weights, &
        FORTML_DEVICE_CUDA)
    cuda_code = status%code

    write (*, '(a,i0)') "gamma_likelihood_repeats,", repeats
    write (*, '(a,es24.16)') "gamma_likelihood_product_seconds,", elapsed
    write (*, '(a,es24.16)') "gamma_likelihood_products_per_second,", rate
    write (*, '(a,es24.16)') "gamma_likelihood_value,", value
    do i = 1, size(latent_gradient)
        write (*, '(a,i0,a,es24.16)') "gamma_likelihood_latent_gradient_", i, ",", &
            latent_gradient(i)
    end do
    write (*, '(a,es24.16)') "gamma_likelihood_shape_gradient,", shape_gradient
    do i = 1, size(latent_product)
        write (*, '(a,i0,a,es24.16)') "gamma_likelihood_latent_hvp_", i, ",", &
            latent_product(i)
    end do
    write (*, '(a,es24.16)') "gamma_likelihood_shape_hvp,", shape_product
    elapsed = real(tick_end - tick_start, dp)/real(tick_rate, dp)
    write (*, '(a,es24.16)') "gamma_likelihood_fit_seconds,", elapsed
    write (*, '(a,es24.16)') "gamma_likelihood_fitted_log_shape,", optimized(1)
    write (*, '(a,es24.16)') "gamma_likelihood_fit_gradient_norm,", &
        result%gradient_norm
    write (*, '(a,i0)') "gamma_likelihood_fit_iterations,", result%iterations
    write (*, '(a,i0)') "gamma_likelihood_cuda_status,", cuda_code
end program fortml_bench_gamma_likelihood
