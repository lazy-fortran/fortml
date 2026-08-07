program fortml_bench_spectral_mixture
    !! Correctness-gated timing protocol for the spectral-mixture kernel.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortml_kernels, only: kernel_t, make_spectral_mixture_kernel
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer, parameter :: n = 256, d = 3, q = 2, repetitions = 24
    real(dp) :: x(n, d), matrix(n, n), matrix_dot(n, n), matrix_bar(n, n)
    real(dp) :: weights(q), means(q, d), scales(q, d)
    type(kernel_t) :: kernel
    type(fortnum_status_t) :: status
    integer(int64) :: begin_clock, end_clock, rate
    real(dp), allocatable :: direction(:), parameter_bar(:), parameter_bar_dot(:)
    real(dp) :: value, gradient_x1(d), gradient_x2(d), mixed_hessian(d, d)
    integer :: repetition, n_parameters
    real(dp) :: seconds

    call fill_fixture(x, matrix_bar)
    weights = [1.3_dp, 0.8_dp]
    means = reshape([0.18_dp, -0.27_dp, 0.41_dp, 0.12_dp, 0.33_dp, -0.21_dp], shape(means))
    scales = reshape([0.24_dp, 0.31_dp, 0.18_dp, 0.37_dp, 0.29_dp, 0.22_dp], shape(scales))
    kernel = make_spectral_mixture_kernel(d, q, weights, means, scales, status)
    if (.not. status_ok(status)) error stop "spectral-mixture constructor failed"
    n_parameters = kernel%parameter_count()
    allocate(direction(n_parameters), parameter_bar(n_parameters), parameter_bar_dot(n_parameters))
    direction = 0.08_dp - 0.013_dp*[ (real(repetition, dp), repetition=1,n_parameters) ]

    call kernel%matrix(x, x, matrix, status)
    if (.not. status_ok(status)) error stop "spectral-mixture matrix failed"
    call system_clock(begin_clock, rate)
    do repetition = 1, repetitions
        call kernel%matrix(x, x, matrix, status)
        if (.not. status_ok(status)) error stop "spectral-mixture matrix failed"
    end do
    call system_clock(end_clock)
    seconds = real(end_clock - begin_clock, dp)/real(rate, dp)/real(repetitions, dp)
    write (*, '(a,a,a,es24.16,a,es24.16)') "spectral_mixture,", "spectral_mixture", ",matrix,", &
        seconds, ",", sum(matrix)

    call kernel%matrix_jvp(x, x, direction, matrix, matrix_dot, status)
    if (.not. status_ok(status)) error stop "spectral-mixture matrix JVP failed"
    call system_clock(begin_clock, rate)
    do repetition = 1, repetitions
        call kernel%matrix_jvp(x, x, direction, matrix, matrix_dot, status)
        if (.not. status_ok(status)) error stop "spectral-mixture matrix JVP failed"
    end do
    call system_clock(end_clock)
    seconds = real(end_clock - begin_clock, dp)/real(rate, dp)/real(repetitions, dp)
    write (*, '(a,a,a,es24.16,a,es24.16)') "spectral_mixture,", "spectral_mixture", ",matrix_jvp,", &
        seconds, ",", sum(matrix_dot)

    call kernel%parameter_vjp(x, x, matrix_bar, parameter_bar, status)
    if (.not. status_ok(status)) error stop "spectral-mixture parameter VJP failed"
    call system_clock(begin_clock, rate)
    do repetition = 1, repetitions
        call kernel%parameter_vjp(x, x, matrix_bar, parameter_bar, status)
        if (.not. status_ok(status)) error stop "spectral-mixture parameter VJP failed"
    end do
    call system_clock(end_clock)
    seconds = real(end_clock - begin_clock, dp)/real(rate, dp)/real(repetitions, dp)
    write (*, '(a,a,a,es24.16,a,es24.16)') "spectral_mixture,", "spectral_mixture", ",parameter_vjp,", &
        seconds, ",", sum(parameter_bar)

    call kernel%parameter_hvp(x, x, matrix_bar, direction, parameter_bar, parameter_bar_dot, status)
    if (.not. status_ok(status)) error stop "spectral-mixture parameter HVP failed"
    call system_clock(begin_clock, rate)
    do repetition = 1, repetitions
        call kernel%parameter_hvp(x, x, matrix_bar, direction, parameter_bar, parameter_bar_dot, status)
        if (.not. status_ok(status)) error stop "spectral-mixture parameter HVP failed"
    end do
    call system_clock(end_clock)
    seconds = real(end_clock - begin_clock, dp)/real(rate, dp)/real(repetitions, dp)
    write (*, '(a,a,a,es24.16,a,es24.16)') "spectral_mixture,", "spectral_mixture", ",parameter_hvp,", &
        seconds, ",", sum(parameter_bar_dot)

    call kernel%input_derivatives(x(1, :), x(2, :), value, gradient_x1, gradient_x2, mixed_hessian, status)
    if (.not. status_ok(status)) error stop "spectral-mixture input derivatives failed"
    write (*, '(a,a,a,es24.16,a,es24.16)') "spectral_mixture,", "spectral_mixture", ",input_derivatives,", &
        0.0_dp, ",", value + sum(gradient_x1) + sum(gradient_x2) + sum(mixed_hessian)

contains

    subroutine fill_fixture(points, cotangent)
        real(dp), intent(out) :: points(:, :), cotangent(:, :)
        integer :: i, j

        do j = 1, size(points, 2)
            do i = 1, size(points, 1)
                points(i, j) = sin(0.013_dp*real(i, dp) + 0.17_dp*real(j, dp)) + &
                    0.2_dp*cos(0.007_dp*real(i*j, dp))
            end do
        end do
        do j = 1, size(cotangent, 2)
            do i = 1, size(cotangent, 1)
                cotangent(i, j) = sin(0.003_dp*real(i + 2*j, dp))
            end do
        end do
    end subroutine fill_fixture

end program fortml_bench_spectral_mixture
