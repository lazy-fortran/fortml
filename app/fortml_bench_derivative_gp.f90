program fortml_bench_derivative_gp
    !! Correctness-gated host timing app for exact derivative-GP query products.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortml_derivative_gaussian_process, only: gp_derivative_regression_t
    use fortml_generated_matern52_products, only: fortml_matern52_hvp
    use fortml_kernels, only: kernel_t, make_periodic_kernel, &
        make_rational_quadratic_kernel, make_cosine_kernel, make_polynomial_kernel, &
        make_spectral_mixture_kernel, make_rbf_ard_kernel
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer, parameter :: n = 8, d = 2, q = 4, repetitions = 16
    real(dp) :: x(n, d), y(n, 1), query(q, d), direction(q, d)
    real(dp) :: mean(q, 1), mean_dot(q, 1), variance(q), variance_dot(q)
    real(dp) :: mean_bar(q, 1), variance_bar(q), x_bar(q, d)
    real(dp) :: covariance(q, q), covariance_dot(q, q), covariance_bar(q, q)
    real(dp), allocatable :: parameter_direction(:), parameter_bar(:)
    integer :: components(n), query_components(q), i, j
    type(kernel_t) :: periodic, rational_quadratic, cosine, polynomial, spectral_mixture, ard_rbf
    real(dp) :: spectral_weights(2), spectral_means(2, 2), spectral_scales(2, 2)
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
        do j = 1, q
            covariance_bar(i, j) = 0.15_dp + 0.02_dp*real(i, dp) - &
                0.03_dp*real(j, dp)
        end do
    end do

    periodic = make_periodic_kernel(d, 1.3_dp, 0.8_dp, 2.1_dp, status)
    if (.not. status_ok(status)) error stop "periodic constructor failed"
    rational_quadratic = make_rational_quadratic_kernel(d, 1.3_dp, 0.8_dp, 1.7_dp, status)
    if (.not. status_ok(status)) error stop "rational quadratic constructor failed"
    cosine = make_cosine_kernel(d, 1.3_dp, 0.8_dp, status)
    if (.not. status_ok(status)) error stop "cosine constructor failed"
    polynomial = make_polynomial_kernel(d, 1.3_dp, 0.4_dp, 1.5_dp, 2.2_dp, status)
    if (.not. status_ok(status)) error stop "polynomial constructor failed"
    spectral_weights = [1.15_dp, 0.63_dp]
    spectral_means = reshape([0.21_dp, -0.37_dp, 0.48_dp, 0.16_dp], shape(spectral_means))
    spectral_scales = reshape([0.31_dp, 0.57_dp, 0.22_dp, 0.44_dp], shape(spectral_scales))
    spectral_mixture = make_spectral_mixture_kernel(d, 2, spectral_weights, &
        spectral_means, spectral_scales, status)
    if (.not. status_ok(status)) error stop "spectral mixture constructor failed"
    ard_rbf = make_rbf_ard_kernel(d, 1.3_dp, [0.75_dp, 1.25_dp], status)
    if (.not. status_ok(status)) error stop "ARD RBF constructor failed"
    call benchmark(periodic, "periodic", status)
    if (.not. status_ok(status)) error stop "periodic derivative GP benchmark failed"
    call benchmark(rational_quadratic, "rational_quadratic", status)
    if (.not. status_ok(status)) error stop "rational quadratic derivative GP benchmark failed"
    call benchmark(cosine, "cosine", status)
    if (.not. status_ok(status)) error stop "cosine derivative GP benchmark failed"
    call benchmark(polynomial, "polynomial", status)
    if (.not. status_ok(status)) error stop "polynomial derivative GP benchmark failed"
    call benchmark(spectral_mixture, "spectral_mixture", status)
    if (.not. status_ok(status)) error stop "spectral mixture derivative GP benchmark failed"
    call benchmark(ard_rbf, "ard_rbf", status)
    if (.not. status_ok(status)) error stop "ARD RBF derivative GP benchmark failed"
    call benchmark_matern52_leaf()

contains

    subroutine benchmark_matern52_leaf()
        real(dp), parameter :: distance = 0.73_dp, distance_d = 0.21_dp
        real(dp), parameter :: lv = log(1.4_dp), lv_d = -0.17_dp
        real(dp), parameter :: ll = log(0.82_dp), ll_d = 0.11_dp, y_bar = 0.83_dp
        real(dp) :: y, y_d, distance_bar, distance_bar_d, lv_bar, lv_bar_d
        real(dp) :: ll_bar, ll_bar_d, seconds
        integer :: repetition

        call fortml_matern52_hvp(distance, distance_d, lv, lv_d, ll, ll_d, y, y_d, &
            y_bar, distance_bar, distance_bar_d, lv_bar, lv_bar_d, ll_bar, ll_bar_d)
        call system_clock(begin_clock, rate)
        do repetition = 1, repetitions
            call fortml_matern52_hvp(distance, distance_d, lv, lv_d, ll, ll_d, y, y_d, &
                y_bar, distance_bar, distance_bar_d, lv_bar, lv_bar_d, ll_bar, ll_bar_d)
        end do
        call system_clock(end_clock)
        seconds = real(end_clock - begin_clock, dp)/real(rate, dp)/real(repetitions, dp)
        write (*, '(a,es24.16,a,es24.16)') "derivative_gp,matern52,leaf_hvp,", &
            seconds, ",", distance_bar_d + lv_bar_d + ll_bar_d
    end subroutine benchmark_matern52_leaf

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

        call model%joint_covariance(query, query_components, covariance, final_status)
        if (.not. status_ok(final_status)) return
        call system_clock(begin_clock, rate)
        do repetition = 1, repetitions
            call model%joint_covariance(query, query_components, covariance, final_status)
            if (.not. status_ok(final_status)) return
        end do
        call system_clock(end_clock)
        seconds = real(end_clock - begin_clock, dp)/real(rate, dp)/real(repetitions, dp)
        write (*, '(a,a,a,es24.16,a,es24.16)') "derivative_gp,", trim(name), &
            ",joint_covariance,", seconds, ",", sum(covariance)

        if (allocated(parameter_direction)) deallocate(parameter_direction)
        if (allocated(parameter_bar)) deallocate(parameter_bar)
        allocate(parameter_direction(model%parameter_count()), &
            parameter_bar(model%parameter_count()))
        do i = 1, model%parameter_count()
            parameter_direction(i) = 0.08_dp - 0.017_dp*real(i, dp)
        end do
        call model%joint_covariance_jvp(query, query_components, parameter_direction, &
            covariance, covariance_dot, final_status)
        if (.not. status_ok(final_status)) return
        call system_clock(begin_clock, rate)
        do repetition = 1, repetitions
            call model%joint_covariance_jvp(query, query_components, parameter_direction, &
                covariance, covariance_dot, final_status)
            if (.not. status_ok(final_status)) return
        end do
        call system_clock(end_clock)
        seconds = real(end_clock - begin_clock, dp)/real(rate, dp)/real(repetitions, dp)
        write (*, '(a,a,a,es24.16,a,es24.16)') "derivative_gp,", trim(name), &
            ",joint_covariance_jvp,", seconds, ",", sum(covariance_dot)

        call model%joint_covariance_vjp(query, query_components, covariance_bar, &
            parameter_bar, final_status)
        if (.not. status_ok(final_status)) return
        call system_clock(begin_clock, rate)
        do repetition = 1, repetitions
            call model%joint_covariance_vjp(query, query_components, covariance_bar, &
                parameter_bar, final_status)
            if (.not. status_ok(final_status)) return
        end do
        call system_clock(end_clock)
        seconds = real(end_clock - begin_clock, dp)/real(rate, dp)/real(repetitions, dp)
        write (*, '(a,a,a,es24.16,a,es24.16)') "derivative_gp,", trim(name), &
            ",joint_covariance_vjp,", seconds, ",", sum(parameter_bar)

        if (trim(name) == "polynomial" .or. trim(name) == "ard_rbf") then
            call model%hyperparameter_hvp(parameter_direction, parameter_bar, final_status)
            if (.not. status_ok(final_status)) return
            call system_clock(begin_clock, rate)
            do repetition = 1, repetitions
                call model%hyperparameter_hvp(parameter_direction, parameter_bar, final_status)
                if (.not. status_ok(final_status)) return
            end do
            call system_clock(end_clock)
            seconds = real(end_clock - begin_clock, dp)/real(rate, dp)/real(repetitions, dp)
            write (*, '(a,a,a,es24.16,a,es24.16)') "derivative_gp,", trim(name), &
                ",hyperparameter_hvp,", seconds, ",", sum(parameter_bar)
        end if
    end subroutine benchmark

end program fortml_bench_derivative_gp
