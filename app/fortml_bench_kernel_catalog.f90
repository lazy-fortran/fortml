program fortml_bench_kernel_catalog
    !! Correctness-gated timing protocol for smooth kernel leaves.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortml_kernels, only: kernel_t, make_periodic_kernel, &
        make_rational_quadratic_kernel, make_cosine_kernel, make_polynomial_kernel
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer, parameter :: n = 256, d = 3, repetitions = 24
    real(dp) :: x(n, d), matrix(n, n), matrix_dot(n, n), matrix_bar(n, n)
    real(dp) :: value, value_dot, gradient_x1(d), gradient_x2(d), mixed_hessian(d, d)
    type(kernel_t) :: periodic, rational_quadratic, cosine, polynomial
    type(fortnum_status_t) :: status
    integer(int64) :: begin_clock, end_clock, rate

    call fill_fixture(x, matrix_bar)
    periodic = make_periodic_kernel(d, 1.3_dp, 0.7_dp, 1.1_dp, status)
    if (.not. status_ok(status)) error stop "periodic constructor failed"
    rational_quadratic = make_rational_quadratic_kernel(d, 1.2_dp, 0.9_dp, 1.4_dp, status)
    if (.not. status_ok(status)) error stop "rational quadratic constructor failed"
    call benchmark_kernel(periodic, "periodic", x, matrix_bar, status)
    if (.not. status_ok(status)) error stop "periodic benchmark failed"
    call benchmark_kernel(rational_quadratic, "rational_quadratic", x, matrix_bar, status)
    if (.not. status_ok(status)) error stop "rational quadratic benchmark failed"
    cosine = make_cosine_kernel(d, 1.3_dp, 0.7_dp, status)
    if (.not. status_ok(status)) error stop "cosine constructor failed"
    call benchmark_kernel(cosine, "cosine", x, matrix_bar, status)
    if (.not. status_ok(status)) error stop "cosine benchmark failed"
    polynomial = make_polynomial_kernel(d, 1.1_dp, 0.1_dp, 5.0_dp, 2.3_dp, status)
    if (.not. status_ok(status)) error stop "polynomial constructor failed"
    call benchmark_kernel(polynomial, "polynomial", x, matrix_bar, status)
    if (.not. status_ok(status)) error stop "polynomial benchmark failed"

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

    subroutine benchmark_kernel(kernel, name, points, cotangent, final_status)
        type(kernel_t), intent(inout) :: kernel
        character(len=*), intent(in) :: name
        real(dp), intent(in) :: points(:, :), cotangent(:, :)
        type(fortnum_status_t), intent(out) :: final_status
        real(dp), allocatable :: direction(:), parameter_bar(:), parameter_bar_dot(:)
        integer :: n_parameters, parameter
        integer :: repetition
        real(dp) :: seconds, checksum

        n_parameters = kernel%parameter_count()
        allocate(direction(n_parameters), parameter_bar(n_parameters), &
            parameter_bar_dot(n_parameters))
        do parameter = 1, n_parameters
            direction(parameter) = 0.03_dp*real(parameter, dp) - 0.08_dp
        end do
        call kernel%matrix(points, points, matrix, final_status)
        if (.not. status_ok(final_status)) return
        call system_clock(begin_clock, rate)
        do repetition = 1, repetitions
            call kernel%matrix(points, points, matrix, final_status)
            if (.not. status_ok(final_status)) return
        end do
        call system_clock(end_clock)
        seconds = real(end_clock - begin_clock, dp)/real(rate, dp)/real(repetitions, dp)
        checksum = sum(matrix)
        write (*, '(a,a,a,es24.16,a,es24.16)') &
            "kernel_catalog,", trim(name), ",matrix,", seconds, ",", checksum

        call kernel%matrix_jvp(points, points, direction, matrix, matrix_dot, final_status)
        if (.not. status_ok(final_status)) return
        call system_clock(begin_clock, rate)
        do repetition = 1, repetitions
            call kernel%matrix_jvp(points, points, direction, matrix, matrix_dot, final_status)
            if (.not. status_ok(final_status)) return
        end do
        call system_clock(end_clock)
        seconds = real(end_clock - begin_clock, dp)/real(rate, dp)/real(repetitions, dp)
        write (*, '(a,a,a,es24.16,a,es24.16)') &
            "kernel_catalog,", trim(name), ",matrix_jvp,", seconds, ",", sum(matrix_dot)

        call kernel%parameter_vjp(points, points, cotangent, parameter_bar, final_status)
        if (.not. status_ok(final_status)) return
        call system_clock(begin_clock, rate)
        do repetition = 1, repetitions
            call kernel%parameter_vjp(points, points, cotangent, parameter_bar, final_status)
            if (.not. status_ok(final_status)) return
        end do
        call system_clock(end_clock)
        seconds = real(end_clock - begin_clock, dp)/real(rate, dp)/real(repetitions, dp)
        write (*, '(a,a,a,es24.16,a,es24.16)') &
            "kernel_catalog,", trim(name), ",parameter_vjp,", seconds, ",", sum(parameter_bar)

        call kernel%parameter_hvp(points, points, cotangent, direction, parameter_bar, &
            parameter_bar_dot, final_status)
        if (.not. status_ok(final_status)) return
        call system_clock(begin_clock, rate)
        do repetition = 1, repetitions
            call kernel%parameter_hvp(points, points, cotangent, direction, parameter_bar, &
                parameter_bar_dot, final_status)
            if (.not. status_ok(final_status)) return
        end do
        call system_clock(end_clock)
        seconds = real(end_clock - begin_clock, dp)/real(rate, dp)/real(repetitions, dp)
        write (*, '(a,a,a,es24.16,a,es24.16)') &
            "kernel_catalog,", trim(name), ",parameter_hvp,", seconds, ",", &
            sum(parameter_bar_dot)

        call kernel%input_derivatives(points(1, :), points(2, :), value, gradient_x1, &
            gradient_x2, mixed_hessian, final_status)
        if (.not. status_ok(final_status)) return
        value_dot = value + sum(gradient_x1) + sum(gradient_x2) + sum(mixed_hessian)
        write (*, '(a,a,a,es24.16,a,es24.16)') &
            "kernel_catalog,", trim(name), ",input_derivatives,0.0,", value_dot
    end subroutine benchmark_kernel

end program fortml_bench_kernel_catalog
