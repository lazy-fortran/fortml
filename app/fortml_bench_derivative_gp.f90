program fortml_bench_derivative_gp
    !! Correctness-gated host timing app for exact derivative-GP query products.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortml_derivative_gaussian_process, only: gp_derivative_regression_t
    use fortml_kernels, only: kernel_t, make_periodic_kernel, &
        make_rational_quadratic_kernel
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer, parameter :: n = 8, d = 2, q = 4, repetitions = 16
    real(dp) :: x(n, d), y(n, 1), query(q, d), direction(q, d)
    real(dp) :: mean(q, 1), mean_dot(q, 1), variance(q), variance_dot(q)
    real(dp) :: mean_bar(q, 1), variance_bar(q), x_bar(q, d)
    integer :: components(n), query_components(q), i, j
    type(kernel_t) :: periodic, rational_quadratic
    type(fortnum_status_t) :: status
    integer(int64) :: begin_clock, end_clock, rate

    do i = 1, n
        x(i, 1) = -0.8_dp + 0.21_dp*real(i - 1, dp)
        x(i, 2) = 0.3_dp*sin(0.31_dp*real(i, dp))
        y(i, 1) = 0.6_dp*cos(x(i, 1)) + 0.1_dp*x(i, 2)
        components(i) = mod(i - 1, 2)
    end do
    do i = 1, q
        query(i, 1) = -0.53_dp + 0.27_dp*real(i - 1, dp)
        query(i, 2) = 0.2_dp*cos(0.37_dp*real(i, dp))
        direction(i, 1) = 0.07_dp - 0.01_dp*real(i, dp)
        direction(i, 2) = -0.04_dp + 0.013_dp*real(i, dp)
        query_components(i) = mod(i, 2)
        mean_bar(i, 1) = 0.23_dp - 0.04_dp*real(i, dp)
        variance_bar(i) = -0.09_dp + 0.02_dp*real(i, dp)
    end do

    periodic = make_periodic_kernel(d, 1.3_dp, 0.8_dp, 2.1_dp, status)
    if (.not. status_ok(status)) error stop "periodic constructor failed"
    rational_quadratic = make_rational_quadratic_kernel(d, 1.3_dp, 0.8_dp, 1.7_dp, status)
    if (.not. status_ok(status)) error stop "rational quadratic constructor failed"
    call benchmark(periodic, "periodic", status)
    if (.not. status_ok(status)) error stop "periodic derivative GP benchmark failed"
    call benchmark(rational_quadratic, "rational_quadratic", status)
    if (.not. status_ok(status)) error stop "rational quadratic derivative GP benchmark failed"

contains

    subroutine benchmark(kernel, name, final_status)
        type(kernel_t), intent(in) :: kernel
        character(len=*), intent(in) :: name
        type(fortnum_status_t), intent(out) :: final_status
        type(gp_derivative_regression_t) :: model
        real(dp) :: seconds
        integer :: repetition

        call model%fit(x, components, y, kernel, 0.07_dp, final_status, jitter=1.0e-10_dp)
        if (.not. status_ok(final_status)) return
        call model%predict_input_jvp(query, query_components, direction, mean, mean_dot, &
            variance, variance_dot, final_status)
        if (.not. status_ok(final_status)) return
        call system_clock(begin_clock, rate)
        do repetition = 1, repetitions
            call model%predict_input_jvp(query, query_components, direction, mean, mean_dot, &
                variance, variance_dot, final_status)
            if (.not. status_ok(final_status)) return
        end do
        call system_clock(end_clock)
        seconds = real(end_clock - begin_clock, dp)/real(rate, dp)/real(repetitions, dp)
        write (*, '(a,a,a,es24.16,a,es24.16)') "derivative_gp,", trim(name), &
            ",input_jvp,", seconds, ",", sum(mean_dot) + sum(variance_dot)

        call model%predict_input_vjp(query, query_components, mean_bar, variance_bar, &
            x_bar, final_status)
        if (.not. status_ok(final_status)) return
        call system_clock(begin_clock, rate)
        do repetition = 1, repetitions
            call model%predict_input_vjp(query, query_components, mean_bar, variance_bar, &
                x_bar, final_status)
            if (.not. status_ok(final_status)) return
        end do
        call system_clock(end_clock)
        seconds = real(end_clock - begin_clock, dp)/real(rate, dp)/real(repetitions, dp)
        write (*, '(a,a,a,es24.16,a,es24.16)') "derivative_gp,", trim(name), &
            ",input_vjp,", seconds, ",", sum(x_bar)
    end subroutine benchmark

end program fortml_bench_derivative_gp
