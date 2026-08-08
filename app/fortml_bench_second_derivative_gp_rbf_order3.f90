program fortml_bench_second_derivative_gp_rbf_order3
    !! Release CPU timing app for scalar RBF order-three derivative GPs.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortnum_status, only: fortnum_status_t, status_ok
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortml_kernels, only: kernel_t, make_rbf_kernel
    use fortml_second_derivative_gaussian_process, only: second_derivative_gp_t
    implicit none

    integer, parameter :: n = 24, q = 16, repetitions = 8
    real(dp) :: x(n, 1), y(n), query(q, 1), direction(q)
    real(dp) :: mean(q), variance(q), mean_dot(q), variance_dot(q)
    real(dp) :: mean_bar(q), variance_bar(q), query_bar(q)
    real(dp) :: hvp(3), parameter_direction(3), seconds, value
    integer :: orders(n), query_orders(q), i, repetition
    integer(int64) :: begin_clock, end_clock, rate
    type(kernel_t) :: kernel
    type(second_derivative_gp_t) :: model
    type(fortml_device_t) :: cuda
    type(fortnum_status_t) :: status

    do i = 1, n
        x(i, 1) = -1.45_dp + 2.9_dp*real(i - 1, dp)/real(n - 1, dp)
        y(i) = 0.4_dp*sin(0.7_dp*x(i, 1)) + 0.08_dp*x(i, 1)**2
        orders(i) = mod(i - 1, 4)
    end do
    do i = 1, q
        query(i, 1) = -1.31_dp + 2.62_dp*real(i - 1, dp)/real(q - 1, dp)
        direction(i) = 0.04_dp*cos(0.31_dp*real(i, dp))
        mean_bar(i) = 0.1_dp - 0.003_dp*real(i, dp)
        variance_bar(i) = -0.06_dp + 0.002_dp*real(i, dp)
        query_orders(i) = mod(i, 4)
    end do
    parameter_direction = [0.07_dp, -0.04_dp, 0.09_dp]

    kernel = make_rbf_kernel(1, 1.35_dp, 0.79_dp, status)
    if (.not. status_ok(status)) error stop "RBF constructor failed"
    call model%fit(x, orders, y, kernel, 0.041_dp, status, 1.0e-10_dp)
    if (.not. status_ok(status)) error stop "RBF order-three fit failed"

    call model%predict(query, query_orders, mean, variance, status)
    if (.not. status_ok(status)) error stop "RBF order-three prediction failed"
    call system_clock(begin_clock, rate)
    do repetition = 1, repetitions
        call model%predict(query, query_orders, mean, variance, status)
        if (.not. status_ok(status)) error stop "RBF order-three prediction failed"
    end do
    call system_clock(end_clock)
    seconds = real(end_clock - begin_clock, dp)/real(rate, dp)/real(repetitions, dp)
    value = sum(mean) + sum(variance)
    write (*, '(a,",",a,",",a,",",es24.16,",",es24.16)') &
        "second_derivative_gp_rbf_order3", "prediction", "cpu", seconds, value

    call model%predict_input_jvp(query, query_orders, direction, mean, mean_dot, variance, &
        variance_dot, status)
    if (.not. status_ok(status)) error stop "RBF order-three input JVP failed"
    call system_clock(begin_clock, rate)
    do repetition = 1, repetitions
        call model%predict_input_jvp(query, query_orders, direction, mean, mean_dot, variance, &
            variance_dot, status)
        if (.not. status_ok(status)) error stop "RBF order-three input JVP failed"
    end do
    call system_clock(end_clock)
    seconds = real(end_clock - begin_clock, dp)/real(rate, dp)/real(repetitions, dp)
    value = sum(mean_dot) + sum(variance_dot)
    write (*, '(a,",",a,",",a,",",es24.16,",",es24.16)') &
        "second_derivative_gp_rbf_order3", "input_jvp", "cpu", seconds, value

    call model%predict_input_vjp(query, query_orders, mean_bar, variance_bar, query_bar, status)
    if (.not. status_ok(status)) error stop "RBF order-three input VJP failed"
    call system_clock(begin_clock, rate)
    do repetition = 1, repetitions
        call model%predict_input_vjp(query, query_orders, mean_bar, variance_bar, query_bar, status)
        if (.not. status_ok(status)) error stop "RBF order-three input VJP failed"
    end do
    call system_clock(end_clock)
    seconds = real(end_clock - begin_clock, dp)/real(rate, dp)/real(repetitions, dp)
    value = sum(query_bar)
    write (*, '(a,",",a,",",a,",",es24.16,",",es24.16)') &
        "second_derivative_gp_rbf_order3", "input_vjp", "cpu", seconds, value

    call model%hyperparameter_hvp(parameter_direction, hvp, status)
    if (.not. status_ok(status)) error stop "RBF order-three hyperparameter HVP failed"
    call system_clock(begin_clock, rate)
    do repetition = 1, repetitions
        call model%hyperparameter_hvp(parameter_direction, hvp, status)
        if (.not. status_ok(status)) error stop "RBF order-three hyperparameter HVP failed"
    end do
    call system_clock(end_clock)
    seconds = real(end_clock - begin_clock, dp)/real(rate, dp)/real(repetitions, dp)
    value = sum(hvp)
    write (*, '(a,",",a,",",a,",",es24.16,",",es24.16)') &
        "second_derivative_gp_rbf_order3", "hyperparameter_hvp", "cpu", seconds, value

    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call model%predict_device(cuda, query, query_orders, mean, variance, status)
    if (status_ok(status)) error stop "CUDA order-three request was unexpectedly accepted"
    write (*, '(a,",",a,",",a,",",a,",",i0)') &
        "second_derivative_gp_rbf_order3", "device", "cuda", "refused", status%code
end program fortml_bench_second_derivative_gp_rbf_order3
