program fortml_bench_gp_linear_operator
    !! Timing/provenance app for the registered linear-operator GP lane.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortnum_status, only: fortnum_status_t, status_ok
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortml_kernels, only: kernel_t, make_rbf_kernel
    use fortml_operator_gaussian_process, only: &
        linear_differential_operator_registry_t, gp_linear_operator_regression_t
    implicit none

    integer, parameter :: n = 4, q = 4, d = 1, repetitions = 64
    real(dp) :: x(n, d), query(q, d), y(n, 1)
    real(dp) :: mean(q, 1), mean_dot(q, 1), variance(q), variance_dot(q)
    real(dp) :: mean_bar(q, 1), variance_bar(q), operator_bar(d + 1, q)
    real(dp) :: direction(d + 1, q), seconds, lhs, rhs
    type(linear_differential_operator_registry_t) :: train_ops, query_ops
    type(gp_linear_operator_regression_t) :: model
    type(kernel_t) :: kernel
    type(fortml_device_t) :: cuda
    type(fortnum_status_t) :: status
    integer(int64) :: begin_clock, end_clock, rate
    integer :: i, repetition

    x(:, 1) = [-0.7_dp, -0.1_dp, 0.45_dp, 1.0_dp]
    y(:, 1) = [0.6_dp, -0.2_dp, 0.75_dp, 0.35_dp]
    query(:, 1) = [-0.5_dp, 0.0_dp, 0.6_dp, 1.2_dp]
    call train_ops%initialize(d, n, status)
    if (.not. status_ok(status)) error stop "training operator registry failed"
    call train_ops%set_operator(1, "value", [1.0_dp, 0.0_dp], status)
    call train_ops%set_operator(2, "gradient", [0.0_dp, 1.0_dp], status)
    call train_ops%set_operator(3, "robin_left", [0.7_dp, -0.4_dp], status)
    call train_ops%set_operator(4, "robin_right", [0.5_dp, 0.25_dp], status)
    call query_ops%initialize(d, q, status)
    if (.not. status_ok(status)) error stop "query operator registry failed"
    call query_ops%set_operator(1, "value", [1.0_dp, 0.0_dp], status)
    call query_ops%set_operator(2, "gradient", [0.0_dp, 1.0_dp], status)
    call query_ops%set_operator(3, "robin_left", [0.7_dp, -0.4_dp], status)
    call query_ops%set_operator(4, "robin_right", [0.5_dp, 0.25_dp], status)
    direction = reshape([0.05_dp, -0.04_dp, -0.07_dp, 0.06_dp, &
        0.03_dp, -0.02_dp, 0.02_dp, 0.08_dp], shape(direction))
    do i = 1, q
        mean_bar(i, 1) = 0.2_dp - 0.03_dp*real(i, dp)
        variance_bar(i) = -0.08_dp + 0.02_dp*real(i, dp)
    end do
    kernel = make_rbf_kernel(d, 1.25_dp, 0.82_dp, status)
    if (.not. status_ok(status)) error stop "RBF constructor failed"
    call model%fit(x, train_ops, y, kernel, 0.06_dp, status, jitter=1.0e-11_dp)
    if (.not. status_ok(status)) error stop "operator GP fit failed"

    call system_clock(begin_clock, rate)
    do repetition = 1, repetitions
        call model%predict(query, query_ops, mean, variance, status)
        if (.not. status_ok(status)) error stop "operator GP prediction failed"
    end do
    call system_clock(end_clock)
    seconds = real(end_clock - begin_clock, dp)/real(rate, dp)/real(repetitions, dp)
    write (*, '(a,es24.16)') "operator_gp,predict_seconds,", seconds
    do i = 1, q
        write (*, '(a,i0,a,es24.16)') "operator_gp,mean,", i, ",", mean(i, 1)
        write (*, '(a,i0,a,es24.16)') "operator_gp,variance,", i, ",", variance(i)
    end do

    call model%predict_operator_jvp(query, query_ops, direction, mean, mean_dot, &
        variance, variance_dot, status)
    if (.not. status_ok(status)) error stop "operator GP JVP failed"
    do i = 1, q
        write (*, '(a,i0,a,es24.16)') "operator_gp,jvp_mean,", i, ",", mean_dot(i, 1)
        write (*, '(a,i0,a,es24.16)') "operator_gp,jvp_variance,", i, ",", variance_dot(i)
    end do
    call model%predict_operator_vjp(query, query_ops, mean_bar, variance_bar, operator_bar, status)
    if (.not. status_ok(status)) error stop "operator GP VJP failed"
    lhs = sum(mean_bar*mean_dot) + dot_product(variance_bar, variance_dot)
    rhs = sum(operator_bar*direction)
    write (*, '(a,es24.16)') "operator_gp,adjoint_error,", abs(lhs - rhs)

    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call model%predict_device(cuda, query, query_ops, mean, variance, status)
    write (*, '(a,i0)') "operator_gp,cuda_status,", status%code
end program fortml_bench_gp_linear_operator
