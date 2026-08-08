program fortml_bench_gp_classification_hvp
    !! Correctness-gated timing for binary Laplace-GP hyperparameter HVPs.
    !!
    !! The fitted mode is held resident between calls.  The companion
    !! fortml-bench script independently refits perturbed kernels and checks
    !! the returned directional product before retaining these timings.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortml_gp_classification, only: gp_classification_t, &
        gp_classification_options_t, GP_LIKELIHOOD_LOGISTIC, GP_LIKELIHOOD_PROBIT
    use fortml_kernels, only: kernel_t, make_rbf_kernel
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer, parameter :: n_samples = 24, repetitions = 32
    real(dp), parameter :: jitter = 1.0e-7_dp
    real(dp) :: x(n_samples, 1), direction(2), product(2), seconds
    integer :: labels(n_samples)
    integer(int64) :: clock_start, clock_end, clock_rate
    type(gp_classification_options_t) :: options
    type(gp_classification_t) :: model
    type(kernel_t) :: kernel
    type(fortml_device_t) :: cuda
    type(fortnum_status_t) :: status
    integer :: i, likelihood, code

    do i = 1, n_samples
        x(i, 1) = -1.5_dp + 3.0_dp*real(i - 1, dp)/real(n_samples - 1, dp)
        labels(i) = merge(-7, 11, x(i, 1) >= 0.0_dp)
    end do
    direction = [0.07_dp, -0.04_dp]
    options%max_iterations = 100
    options%tolerance = 1.0e-10_dp
    options%jitter = jitter

    do likelihood = GP_LIKELIHOOD_LOGISTIC, GP_LIKELIHOOD_PROBIT
        options%likelihood = likelihood
        kernel = make_rbf_kernel(1, 1.35_dp, 0.79_dp, status)
        call model%fit(x, labels, kernel, status, options)
        if (.not. status_ok(status)) error stop "GP classification HVP fit failed"
        call model%hyperparameter_hvp(direction, product, status)
        if (.not. status_ok(status)) error stop "GP classification HVP warmup failed"
        call system_clock(clock_start, clock_rate)
        do i = 1, repetitions
            call model%hyperparameter_hvp(direction, product, status)
            if (.not. status_ok(status)) error stop "GP classification HVP failed"
        end do
        call system_clock(clock_end)
        seconds = real(clock_end - clock_start, dp)/real(clock_rate, dp)/ &
            real(repetitions, dp)
        write (*, '(a,a,a,a,es24.16,a,es24.16,a,es24.16)') &
            "gp_classification_hvp,", merge("logistic", "probit  ", &
            likelihood == GP_LIKELIHOOD_LOGISTIC), ",cpu,", "seconds,", seconds, &
            ",sum,", sum(product), ",norm,", sqrt(sum(product*product))
    end do

    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call model%hyperparameter_hvp_device(cuda, direction, product, status)
    code = status%code
    write (*, '(a,i0)') "gp_classification_hvp,device,cuda,refused,", code
end program fortml_bench_gp_classification_hvp
