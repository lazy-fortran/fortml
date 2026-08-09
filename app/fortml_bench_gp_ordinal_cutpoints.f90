program fortml_bench_gp_ordinal_cutpoints
    !! Correctness-gated timing for fixed-latent ordinal-GP cut-point products.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortnum_status, only: fortnum_status_t, status_ok
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortml_kernels, only: kernel_t, make_rbf_kernel
    use fortml_gp_ordinal_classification, only: gp_ordinal_classification_t, &
        gp_ordinal_classification_options_t, GP_ORDINAL_LIKELIHOOD_PROBIT
    use fortml_gp_ordinal_cutpoint_training, only: gp_ordinal_cutpoint_options_t, &
        gp_ordinal_cutpoint_result_t, gp_ordinal_cutpoint_value_gradient, &
        gp_ordinal_cutpoint_hvp, gp_ordinal_cutpoint_value_gradient_device, &
        gp_ordinal_optimize_cutpoints
    implicit none

    integer, parameter :: n = 24, repetitions = 64
    type(gp_ordinal_classification_t) :: model
    type(gp_ordinal_classification_options_t) :: fit_options
    type(gp_ordinal_cutpoint_options_t) :: train_options
    type(gp_ordinal_cutpoint_result_t) :: result
    type(kernel_t) :: kernel
    type(fortml_device_t) :: cuda
    type(fortnum_status_t) :: status
    real(dp) :: x(n, 1), weight(n), thresholds(2), direction(2)
    real(dp) :: gradient(2), product(2), value, seconds
    integer :: labels(n), i, code
    integer(int64) :: clock_start, clock_end, clock_rate

    do i = 1, n
        x(i, 1) = -1.8_dp + 3.6_dp*real(i - 1, dp)/real(n - 1, dp)
        if (x(i, 1) < -0.62_dp) then
            labels(i) = -4
        else if (x(i, 1) < 0.47_dp) then
            labels(i) = 7
        else
            labels(i) = 19
        end if
        weight(i) = 0.75_dp + 0.09_dp*real(mod(5*i, 9), dp)
    end do
    thresholds = [1.18_dp, 2.78_dp]
    direction = [0.16_dp, -0.11_dp]
    kernel = make_rbf_kernel(1, 1.3_dp, 0.81_dp, status)
    fit_options%noise_variance = 0.08_dp
    fit_options%jitter = 1.0e-8_dp
    call model%fit(x, labels, kernel, status, fit_options)
    if (.not. status_ok(status)) error stop "ordinal cut-point benchmark fit failed"
    call model%set_thresholds(thresholds, status)
    if (.not. status_ok(status)) error stop "ordinal cut-point benchmark setter failed"

    call gp_ordinal_cutpoint_value_gradient(model, x, labels, thresholds, &
        GP_ORDINAL_LIKELIHOOD_PROBIT, value, gradient, status, weight)
    if (.not. status_ok(status)) error stop "ordinal cut-point gradient warmup failed"
    call system_clock(clock_start, clock_rate)
    do i = 1, repetitions
        call gp_ordinal_cutpoint_value_gradient(model, x, labels, thresholds, &
            GP_ORDINAL_LIKELIHOOD_PROBIT, value, gradient, status, weight)
        if (.not. status_ok(status)) error stop "ordinal cut-point gradient failed"
    end do
    call system_clock(clock_end)
    seconds = real(clock_end - clock_start, dp)/real(clock_rate, dp)/real(repetitions, dp)
    write (*, '(a,es24.16,a,es24.16,a,es24.16)') &
        "gp_ordinal_cutpoint_gradient,cpu,seconds,", seconds, ",nll,", value, &
        ",norm,", sqrt(sum(gradient*gradient))

    call gp_ordinal_cutpoint_hvp(model, x, labels, thresholds, &
        GP_ORDINAL_LIKELIHOOD_PROBIT, direction, value, gradient, product, status, weight)
    if (.not. status_ok(status)) error stop "ordinal cut-point HVP warmup failed"
    call system_clock(clock_start, clock_rate)
    do i = 1, repetitions
        call gp_ordinal_cutpoint_hvp(model, x, labels, thresholds, &
            GP_ORDINAL_LIKELIHOOD_PROBIT, direction, value, gradient, product, &
            status, weight)
        if (.not. status_ok(status)) error stop "ordinal cut-point HVP failed"
    end do
    call system_clock(clock_end)
    seconds = real(clock_end - clock_start, dp)/real(clock_rate, dp)/real(repetitions, dp)
    write (*, '(a,es24.16,a,es24.16)') &
        "gp_ordinal_cutpoint_hvp,cpu,seconds,", seconds, &
        ",norm,", sqrt(sum(product*product))

    train_options%likelihood = GP_ORDINAL_LIKELIHOOD_PROBIT
    train_options%max_iterations = 160
    train_options%max_line_search = 50
    train_options%gradient_tolerance = 2.0e-7_dp
    train_options%location_lower = -1.0_dp
    train_options%location_upper = 4.0_dp
    train_options%log_gap_lower = -4.0_dp
    train_options%log_gap_upper = 2.0_dp
    call system_clock(clock_start, clock_rate)
    call gp_ordinal_optimize_cutpoints(model, x, labels, train_options, result, &
        status, sample_weight=weight)
    call system_clock(clock_end)
    if (.not. status_ok(status) .or. .not. result%converged) then
        error stop "ordinal cut-point optimizer failed"
    end if
    seconds = real(clock_end - clock_start, dp)/real(clock_rate, dp)
    thresholds = model%thresholds()
    write (*, '(a,es24.16,a,i0,a,i0,a,es24.16,a,es24.16,a,es24.16,a,es24.16,a,es24.16)') &
        "gp_ordinal_cutpoint_training,cpu,seconds,", seconds, ",iterations,", &
        result%iterations, ",evaluations,", result%line_search_evaluations, &
        ",initial_nll,", result%initial_negative_log_likelihood, ",final_nll,", &
        result%negative_log_likelihood, ",gradient_norm,", result%gradient_norm, &
        ",cut1,", thresholds(1), ",cut2,", thresholds(2)

    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call gp_ordinal_cutpoint_value_gradient_device(cuda, model, x, labels, thresholds, &
        GP_ORDINAL_LIKELIHOOD_PROBIT, value, gradient, status, weight)
    code = status%code
    write (*, '(a,i0)') "gp_ordinal_cutpoint_gradient,device,cuda,refused,", code
    call gp_ordinal_optimize_cutpoints(model, x, labels, train_options, result, status, &
        cuda, weight)
    code = status%code
    write (*, '(a,i0)') "gp_ordinal_cutpoint_training,device,cuda,refused,", code
end program fortml_bench_gp_ordinal_cutpoints
