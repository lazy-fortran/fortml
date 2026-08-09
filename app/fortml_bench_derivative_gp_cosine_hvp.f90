program fortml_bench_derivative_gp_cosine_hvp
    !! CPU timing lane for cosine mixed-observation GP HVPs.
    !! CUDA is reported as an explicit typed capability refusal.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortml_derivative_gaussian_process, only: gp_derivative_regression_t
    use fortml_kernels, only: kernel_t, make_cosine_kernel
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer, parameter :: n = 3, d = 1, repetitions = 32
    real(dp) :: x_train(n, d), y_train(n, 1), direction(3), hvp(3), seconds
    real(dp) :: x_query(1, d), query_mean(1, 1), query_variance(1)
    integer :: components(n), query_components(1), repetition, cuda_code
    integer(int64) :: begin_clock, end_clock, rate
    type(kernel_t) :: kernel
    type(gp_derivative_regression_t) :: model
    type(fortml_device_t) :: cuda
    type(fortnum_status_t) :: status

    x_train(:, 1) = [0.0_dp, 0.42_dp, 1.03_dp]
    y_train(:, 1) = [0.8_dp, -0.2_dp, 0.6_dp]
    components = [0, 1, 0]
    direction = [0.17_dp, -0.11_dp, 0.08_dp]
    x_query(1, 1) = 0.31_dp
    query_components = [0]

    kernel = make_cosine_kernel(d, 1.2_dp, 0.75_dp, status)
    if (.not. status_ok(status)) error stop "cosine constructor failed"
    call model%fit(x_train, components, y_train, kernel, 0.08_dp, status, &
        jitter=1.0e-10_dp)
    if (.not. status_ok(status)) error stop "cosine derivative GP fit failed"
    call model%hyperparameter_hvp(direction, hvp, status)
    if (.not. status_ok(status)) error stop "cosine derivative GP HVP failed"
    call system_clock(begin_clock, rate)
    do repetition = 1, repetitions
        call model%hyperparameter_hvp(direction, hvp, status)
        if (.not. status_ok(status)) error stop "cosine derivative GP HVP failed"
    end do
    call system_clock(end_clock)
    seconds = real(end_clock - begin_clock, dp)/real(rate, dp)/real(repetitions, dp)
    write (*, '(a,a,a,es24.16,a,es24.16)') &
        "derivative_gp_cosine,", "cosine,", "hvp,cpu,", seconds, ",", sum(hvp)

    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call model%predict_device(cuda, x_query, query_components, query_mean, &
        query_variance, status)
    cuda_code = status%code
    write (*, '(a,i0)') "derivative_gp_cosine,hvp,cuda_refused,", cuda_code
end program fortml_bench_derivative_gp_cosine_hvp
