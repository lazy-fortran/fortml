program fortml_bench_gp_posterior_covariance
    !! Release benchmark for the exact GP posterior covariance contract.
    use, intrinsic :: iso_fortran_env, only: real64
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    use fortml_kernels, only: kernel_t, make_rbf_kernel
    use fortml_gaussian_process, only: gp_regression_t
    implicit none

    integer, parameter :: n = 32, m = 16, d = 1, p = 2, repetitions = 20
    real(dp) :: x(n, d), y(n, p), query(m, d), covariance(m, m)
    real(dp) :: covariance_dot(m, m), covariance_bar(m, m)
    real(dp) :: parameter_bar(3), direction(3)
    real(dp) :: mean(m, p), variance(m)
    type(gp_regression_t) :: model
    type(kernel_t) :: kernel
    type(fortnum_status_t) :: status
    type(fortml_device_t) :: cpu, cuda
    real(real64) :: started, finished
    real(dp) :: checksum, variance_checksum
    real(dp) :: covariance_dot_checksum, parameter_bar_checksum
    integer :: i, repetition, cuda_code

    do i = 1, n
        x(i, 1) = -1.2_dp + 0.08_dp*real(i - 1, dp)
        y(i, 1) = sin(1.3_dp*x(i, 1))
        y(i, 2) = cos(0.7_dp*x(i, 1)) - 0.2_dp
    end do
    do i = 1, m
        query(i, 1) = -1.0_dp + 0.13_dp*real(i - 1, dp)
    end do
    kernel = make_rbf_kernel(d, 1.2_dp, 0.55_dp, status)
    call model%fit(x, y, kernel, 0.08_dp, status)
    if (.not. status_ok(status)) error stop "GP covariance benchmark fit failed"

    call cpu_time(started)
    do repetition = 1, repetitions
    call model%predict_covariance(query, covariance, status)
    if (.not. status_ok(status)) error stop "GP covariance benchmark prediction failed"
    end do
    call cpu_time(finished)
    checksum = sum(covariance)
    call model%predict(query, mean, variance, status)
    if (.not. status_ok(status)) error stop "GP covariance benchmark marginal failed"
    variance_checksum = sum(variance)
    write (*, '(a,",seconds,",es24.16,",checksum,",es24.16)') &
        "gp_posterior_covariance", (finished - started)/real(repetitions, real64), checksum
    write (*, '(a,",variance_checksum,",es24.16)') &
        "gp_posterior_covariance", variance_checksum

    direction = [0.13_dp, -0.19_dp, 0.23_dp]
    covariance_bar = reshape([0.4_dp, -0.2_dp, 0.3_dp, 0.5_dp, -0.7_dp, 0.1_dp, &
        0.6_dp, -0.4_dp, 0.2_dp], shape(covariance_bar))
    call cpu_time(started)
    do repetition = 1, repetitions
        call model%predict_covariance_jvp(query, direction, covariance, covariance_dot, status)
        if (.not. status_ok(status)) error stop "GP covariance JVP benchmark failed"
    end do
    call cpu_time(finished)
    covariance_dot_checksum = sum(covariance_dot)
    write (*, '(a,",seconds,",es24.16,",checksum,",es24.16)') &
        "gp_posterior_covariance_jvp", (finished - started)/real(repetitions, real64), &
        covariance_dot_checksum
    call model%predict_covariance_vjp(query, covariance_bar, parameter_bar, status)
    if (.not. status_ok(status)) error stop "GP covariance VJP benchmark failed"
    parameter_bar_checksum = sum(parameter_bar)
    write (*, '(a,",parameter_bar_checksum,",es24.16)') &
        "gp_posterior_covariance_vjp", parameter_bar_checksum

    cpu%kind = FORTML_DEVICE_CPU
    cpu%selected = .true.
    cpu%available = .true.
    call model%predict_covariance_device(cpu, query, covariance, status)
    write (*, '(a,",cpu,supported,",a1)') "gp_posterior_covariance_device", &
        merge("T", "F", status_ok(status))
    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call model%predict_covariance_device(cuda, query, covariance, status)
    cuda_code = status%code
    write (*, '(a,",cuda,refused,",i0)') "gp_posterior_covariance_device", cuda_code
    if (cuda_code /= FORTNUM_NOT_IMPLEMENTED) error stop "CUDA boundary changed"
    call model%predict_covariance_jvp_device(cpu, query, direction, covariance, &
        covariance_dot, status)
    if (.not. status_ok(status)) error stop "CPU covariance JVP dispatch failed"
    write (*, '(a,",cpu,supported,",a1)') "gp_posterior_covariance_jvp_device", &
        merge("T", "F", status_ok(status))
    call model%predict_covariance_jvp_device(cuda, query, direction, covariance, &
        covariance_dot, status)
    cuda_code = status%code
    write (*, '(a,",cuda,refused,",i0)') "gp_posterior_covariance_jvp_device", cuda_code
    if (cuda_code /= FORTNUM_NOT_IMPLEMENTED) error stop "CUDA JVP boundary changed"
    call model%predict_covariance_vjp_device(cpu, query, covariance_bar, parameter_bar, status)
    if (.not. status_ok(status)) error stop "CPU covariance VJP dispatch failed"
    write (*, '(a,",cpu,supported,",a1)') "gp_posterior_covariance_vjp_device", &
        merge("T", "F", status_ok(status))
    call model%predict_covariance_vjp_device(cuda, query, covariance_bar, parameter_bar, status)
    cuda_code = status%code
    write (*, '(a,",cuda,refused,",i0)') "gp_posterior_covariance_vjp_device", cuda_code
    if (cuda_code /= FORTNUM_NOT_IMPLEMENTED) error stop "CUDA VJP boundary changed"
end program fortml_bench_gp_posterior_covariance
