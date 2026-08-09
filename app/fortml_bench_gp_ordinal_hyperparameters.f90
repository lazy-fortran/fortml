program fortml_bench_gp_ordinal_hyperparameters
    !! Correctness-gated timing for ordinal-GP evidence products and training.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortml_gp_ordinal_classification, only: &
        gp_ordinal_classification_t, gp_ordinal_classification_options_t
    use fortml_gp_ordinal_classification_training, only: &
        gp_ordinal_hyperparameter_options_t, gp_ordinal_hyperparameter_result_t, &
        gp_ordinal_optimize_hyperparameters
    use fortml_kernels, only: kernel_t, make_rbf_kernel
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer, parameter :: n_samples = 18, repetitions = 32
    real(dp) :: x(n_samples, 1), direction(3), product(3), gradient(3), seconds
    integer :: labels(n_samples), i, code
    integer(int64) :: clock_start, clock_end, clock_rate
    type(gp_ordinal_classification_t) :: model, model_opt
    type(gp_ordinal_classification_options_t) :: fit_options
    type(gp_ordinal_hyperparameter_options_t) :: train_options
    type(gp_ordinal_hyperparameter_result_t) :: train_result
    type(kernel_t) :: kernel
    type(fortml_device_t) :: cuda
    type(fortnum_status_t) :: status

    do i = 1, n_samples
        x(i, 1) = -1.7_dp + 3.4_dp*real(i - 1, dp)/real(n_samples - 1, dp)
        if (x(i, 1) < -0.55_dp) then
            labels(i) = -4
        else if (x(i, 1) < 0.55_dp) then
            labels(i) = 7
        else
            labels(i) = 19
        end if
    end do
    direction = [0.07_dp, -0.04_dp, 0.03_dp]
    fit_options%noise_variance = 0.05_dp
    fit_options%jitter = 1.0e-8_dp
    kernel = make_rbf_kernel(1, 1.35_dp, 0.79_dp, status)
    call model%fit(x, labels, kernel, status, fit_options)
    if (.not. status_ok(status)) error stop "ordinal GP benchmark fit failed"

    call model%hyperparameter_gradient(gradient, status)
    if (.not. status_ok(status)) error stop "ordinal GP gradient warmup failed"
    call system_clock(clock_start, clock_rate)
    do i = 1, repetitions
        call model%hyperparameter_gradient(gradient, status)
        if (.not. status_ok(status)) error stop "ordinal GP gradient failed"
    end do
    call system_clock(clock_end)
    seconds = real(clock_end - clock_start, dp)/real(clock_rate, dp)/real(repetitions, dp)
    write (*, '(a,a,es24.16,a,es24.16)') &
        "gp_ordinal_hyperparameter_gradient,", "cpu,seconds,", seconds, &
        ",norm,", sqrt(sum(gradient*gradient))

    call model%hyperparameter_hvp(direction, product, status)
    if (.not. status_ok(status)) error stop "ordinal GP HVP warmup failed"
    call system_clock(clock_start, clock_rate)
    do i = 1, repetitions
        call model%hyperparameter_hvp(direction, product, status)
        if (.not. status_ok(status)) error stop "ordinal GP HVP failed"
    end do
    call system_clock(clock_end)
    seconds = real(clock_end - clock_start, dp)/real(clock_rate, dp)/real(repetitions, dp)
    write (*, '(a,a,es24.16,a,es24.16)') &
        "gp_ordinal_hyperparameter_hvp,", "cpu,seconds,", seconds, &
        ",norm,", sqrt(sum(product*product))

    call model_opt%fit(x, labels, kernel, status, fit_options)
    if (.not. status_ok(status)) error stop "ordinal GP optimizer fit failed"
    train_options%max_iterations = 120
    train_options%max_line_search = 40
    train_options%gradient_tolerance = 2.0e-5_dp
    train_options%lower_bound = -8.0_dp
    train_options%upper_bound = 8.0_dp
    call system_clock(clock_start, clock_rate)
    call gp_ordinal_optimize_hyperparameters(model_opt, train_options, train_result, status)
    call system_clock(clock_end)
    if (.not. status_ok(status) .or. .not. train_result%converged) then
        error stop "ordinal GP optimizer failed"
    end if
    seconds = real(clock_end - clock_start, dp)/real(clock_rate, dp)
    write (*, '(a,a,es24.16,a,i0,a,i0,a,es24.16,a,es24.16)') &
        "gp_ordinal_hyperparameter_training,", "seconds,", seconds, ",iterations,", &
        train_result%iterations, ",evaluations,", train_result%line_search_evaluations, &
        ",nll,", train_result%negative_log_marginal_likelihood, ",gradient_norm,", &
        train_result%gradient_norm

    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call model%hyperparameter_hvp_device(cuda, direction, product, status)
    code = status%code
    write (*, '(a,i0)') "gp_ordinal_hyperparameter_hvp,device,cuda,refused,", code
    call gp_ordinal_optimize_hyperparameters(model, train_options, train_result, status, cuda)
    code = status%code
    write (*, '(a,i0)') "gp_ordinal_hyperparameter_training,device,cuda,refused,", code
end program fortml_bench_gp_ordinal_hyperparameters
