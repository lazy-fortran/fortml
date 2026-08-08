program fortml_bench_multi_output_gp_products
    use, intrinsic :: iso_fortran_env, only: dp => real64
    use fortnum_status, only: fortnum_status_t, status_ok
    use fortml_kernels, only: kernel_t, make_rbf_kernel
    use fortml_multi_output_gp, only: multi_output_gp_t
    implicit none
    integer, parameter :: n = 48, m = 24, d = 2, p = 3, rank = 2, repetitions = 8
    real(dp) :: x(n, d), y(n, p), query(m, d), query_direction(m, d)
    real(dp) :: weights(p, rank), independent(p), mean(m, p), mean_dot(m, p)
    real(dp) :: mean_bar(m, p), query_bar(m, d), noise
    real(dp) :: covariance(n*p, n*p), covariance_dot(n*p, n*p)
    real(dp) :: covariance_bar(n*p, n*p)
    real(dp), allocatable :: direction(:), parameter_bar(:)
    type(kernel_t) :: kernel
    type(multi_output_gp_t) :: model
    type(fortnum_status_t) :: status
    integer :: i, j, k, tick, rate
    real(dp) :: started, elapsed, adjoint_error

    do i = 1, n
        x(i, 1) = -1.0_dp + 0.041_dp*real(i - 1, dp)
        x(i, 2) = sin(0.17_dp*real(i, dp))
        y(i, 1) = sin(0.7_dp*x(i, 1)) + 0.1_dp*x(i, 2)
        y(i, 2) = cos(0.9_dp*x(i, 1)) - 0.2_dp*x(i, 2)
        y(i, 3) = x(i, 1)*x(i, 2)
    end do
    do i = 1, m
        query(i, 1) = -0.8_dp + 0.061_dp*real(i - 1, dp)
        query(i, 2) = cos(0.13_dp*real(i, dp))
        query_direction(i, 1) = 0.05_dp*cos(0.11_dp*real(i, dp))
        query_direction(i, 2) = -0.03_dp*sin(0.19_dp*real(i, dp))
        do j = 1, p
            mean_bar(i, j) = 0.2_dp*sin(0.07_dp*real(i + 3*j, dp))
        end do
    end do
    do i = 1, n*p
        do j = 1, n*p
            covariance_bar(i, j) = 0.11_dp*sin(0.023_dp*real(i + 2*j, dp))
        end do
    end do
    weights = reshape([0.8_dp, -0.45_dp, 0.3_dp, 0.6_dp, -0.2_dp, 0.55_dp], [p, rank])
    independent = [0.25_dp, 0.35_dp, 0.18_dp]
    noise = 0.12_dp
    kernel = make_rbf_kernel(d, 1.2_dp, 0.65_dp, status)
    call model%initialize(kernel, weights, independent, noise, status)
    call model%fit(x, y, status)
    if (.not. status_ok(status)) error stop "multi-output benchmark fit failed"
    allocate(direction(model%parameter_count()), parameter_bar(model%parameter_count()))
    do k = 1, size(direction)
        direction(k) = 0.04_dp*sin(0.17_dp*real(k, dp))
    end do

    call system_clock(tick, rate)
    started = real(tick, dp)
    do k = 1, repetitions
        call model%predict(query, mean, status)
    end do
    call system_clock(tick)
    elapsed = (real(tick, dp) - started)/real(rate, dp)/real(repetitions, dp)
    if (.not. status_ok(status)) error stop "multi-output benchmark predict failed"
    write (*, '(a,",",i0,",",i0,",",i0,",",es24.16,",",es24.16)') &
        "multi_output_gp,predict", n, m, p, elapsed, sum(mean)

    call system_clock(tick, rate)
    started = real(tick, dp)
    do k = 1, repetitions
        call model%predict_input_jvp(query, query_direction, mean, mean_dot, status)
    end do
    call system_clock(tick)
    elapsed = (real(tick, dp) - started)/real(rate, dp)/real(repetitions, dp)
    if (.not. status_ok(status)) error stop "multi-output benchmark input JVP failed"
    write (*, '(a,",",i0,",",i0,",",i0,",",es24.16,",",es24.16)') &
        "multi_output_gp,input_jvp", n, m, p, elapsed, sqrt(sum(mean_dot**2))

    call system_clock(tick, rate)
    started = real(tick, dp)
    do k = 1, repetitions
        call model%predict_parameter_jvp(query, direction, mean, mean_dot, status)
    end do
    call system_clock(tick)
    elapsed = (real(tick, dp) - started)/real(rate, dp)/real(repetitions, dp)
    if (.not. status_ok(status)) error stop "multi-output benchmark parameter JVP failed"
    write (*, '(a,",",i0,",",i0,",",i0,",",es24.16,",",es24.16)') &
        "multi_output_gp,parameter_jvp", n, m, p, elapsed, sqrt(sum(mean_dot**2))

    call system_clock(tick, rate)
    started = real(tick, dp)
    do k = 1, repetitions
        call model%predict_parameter_vjp(query, mean_bar, parameter_bar, status)
    end do
    call system_clock(tick)
    elapsed = (real(tick, dp) - started)/real(rate, dp)/real(repetitions, dp)
    if (.not. status_ok(status)) error stop "multi-output benchmark parameter VJP failed"
    call model%predict_parameter_jvp(query, direction, mean, mean_dot, status)
    adjoint_error = abs(sum(mean_bar*mean_dot) - dot_product(parameter_bar, direction))
    write (*, '(a,",",i0,",",i0,",",i0,",",es24.16,",",es24.16)') &
        "multi_output_gp,parameter_vjp", n, m, p, elapsed, adjoint_error

    call system_clock(tick, rate)
    started = real(tick, dp)
    do k = 1, repetitions
        call model%joint_covariance_parameter_jvp(x, direction, covariance, &
            covariance_dot, status)
    end do
    call system_clock(tick)
    elapsed = (real(tick, dp) - started)/real(rate, dp)/real(repetitions, dp)
    if (.not. status_ok(status)) error stop "multi-output benchmark covariance JVP failed"
    write (*, '(a,",",i0,",",i0,",",i0,",",es24.16,",",es24.16)') &
        "multi_output_gp,covariance_parameter_jvp", n, m, p, elapsed, &
        sqrt(sum(covariance_dot**2))

    call system_clock(tick, rate)
    started = real(tick, dp)
    do k = 1, repetitions
        call model%joint_covariance_parameter_vjp(x, covariance_bar, parameter_bar, status)
    end do
    call system_clock(tick)
    elapsed = (real(tick, dp) - started)/real(rate, dp)/real(repetitions, dp)
    if (.not. status_ok(status)) error stop "multi-output benchmark covariance VJP failed"
    call model%joint_covariance_parameter_jvp(x, direction, covariance, covariance_dot, status)
    adjoint_error = abs(sum(covariance_bar*covariance_dot) - dot_product(parameter_bar, direction))
    write (*, '(a,",",i0,",",i0,",",i0,",",es24.16,",",es24.16)') &
        "multi_output_gp,covariance_parameter_vjp", n, m, p, elapsed, adjoint_error
end program fortml_bench_multi_output_gp_products
