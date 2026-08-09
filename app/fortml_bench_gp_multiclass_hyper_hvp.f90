program fortml_bench_gp_multiclass_hyper_hvp
    !! Correctness-gated timing for multiclass Laplace-GP hyperparameter HVPs.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortml_gp_multiclass_classification, only: &
        gp_multiclass_classification_t, gp_multiclass_classification_options_t
    use fortml_kernels, only: kernel_t, make_rbf_kernel
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer, parameter :: n_samples = 9, n_classes = 3, repetitions = 16
    real(dp) :: x(n_samples, 2), direction(n_classes*2), product(n_classes*2)
    real(dp) :: seconds
    integer :: labels(n_samples), i, code
    integer(int64) :: clock_start, clock_end, clock_rate
    type(gp_multiclass_classification_options_t) :: options
    type(gp_multiclass_classification_t) :: model
    type(kernel_t) :: kernel
    type(fortml_device_t) :: cuda
    type(fortnum_status_t) :: status

    x(1, :) = [-0.1_dp, 1.9_dp]
    x(2, :) = [0.1_dp, 2.1_dp]
    x(3, :) = [0.2_dp, 1.8_dp]
    x(4, :) = [-0.1_dp, -0.1_dp]
    x(5, :) = [0.1_dp, 0.2_dp]
    x(6, :) = [0.3_dp, 0.0_dp]
    x(7, :) = [1.9_dp, 0.0_dp]
    x(8, :) = [2.1_dp, 0.2_dp]
    x(9, :) = [1.8_dp, 0.3_dp]
    labels = [42, 42, 42, -7, -7, -7, 10, 10, 10]
    direction = [0.021_dp, -0.014_dp, 0.017_dp, -0.011_dp, &
        0.013_dp, -0.009_dp]
    options%max_iterations = 100
    options%tolerance = 1.0e-9_dp
    options%jitter = 1.0e-7_dp
    kernel = make_rbf_kernel(2, 1.5_dp, 0.55_dp, status)
    call model%fit(x, labels, kernel, status, options)
    if (.not. status_ok(status)) error stop "GP multiclass HVP fit failed"
    call model%hyperparameter_hvp(direction, product, status)
    if (.not. status_ok(status)) error stop "GP multiclass HVP warmup failed"
    call system_clock(clock_start, clock_rate)
    do i = 1, repetitions
        call model%hyperparameter_hvp(direction, product, status)
        if (.not. status_ok(status)) error stop "GP multiclass HVP failed"
    end do
    call system_clock(clock_end)
    seconds = real(clock_end - clock_start, dp)/real(clock_rate, dp)/real(repetitions, dp)
    write (*, '(a,es24.16,a,es24.16,a,es24.16)') &
        "gp_multiclass_hyper_hvp,cpu,seconds,", seconds, ",sum,", &
        sum(product), ",norm,", sqrt(sum(product*product))

    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call model%hyperparameter_hvp_device(cuda, direction, product, status)
    code = status%code
    write (*, '(a,i0)') "gp_multiclass_hyper_hvp,device,cuda,refused,", code
end program fortml_bench_gp_multiclass_hyper_hvp
