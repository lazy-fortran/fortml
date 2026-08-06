program fortml_bench_gp_features
    !! Correctness-gated timings for GP features that have different execution
    !! paths from dense exact regression.
    !!
    !! Usage: fortml_bench_gp_features <workload> <repetitions>
    !! Workloads: logdet, predictive_variance, derivative, multi_output,
    !! variational.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64, output_unit
    use fortml_derivative_gaussian_process, only: gp_derivative_regression_t
    use fortml_kernel_operator, only: kernel_operator_t
    use fortml_kernels, only: kernel_t, make_rbf_kernel
    use fortml_lanczos, only: lanczos_log_determinant, &
        lanczos_predictive_variance
    use fortml_multi_output_gp, only: multi_output_gp_t
    use fortml_sparse_gp, only: sparse_gp_t
    use fortnum_linalg, only: dense_solve
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    character(len=32) :: workload, argument
    integer :: repetitions, io_status
    integer(int64) :: clock_begin = 0_int64, clock_rate = 1_int64

    call get_command_argument(1, workload)
    call get_command_argument(2, argument)
    repetitions = 3
    if (len_trim(argument) > 0) then
        read (argument, *, iostat=io_status) repetitions
        if (io_status /= 0 .or. repetitions < 1) repetitions = 3
    end if

    select case (trim(workload))
    case ("logdet")
        call benchmark_logdet(repetitions)
    case ("predictive_variance")
        call benchmark_predictive_variance(repetitions)
    case ("derivative")
        call benchmark_derivative_gp(repetitions)
    case ("multi_output")
        call benchmark_multi_output_gp(repetitions)
    case ("variational")
        call benchmark_variational_gp(repetitions)
    case default
        write (output_unit, '(a)') &
            "usage: fortml_bench_gp_features <workload> <repetitions>"
        stop 1
    end select

contains

    subroutine benchmark_logdet(reps)
        integer, intent(in) :: reps
        integer, parameter :: n = 64, d = 2, probes = 64, steps = 48
        real(dp), parameter :: signal = 1.4_dp, scale = 0.9_dp
        real(dp), parameter :: shift = 0.35_dp
        real(dp) :: x(n, d), matrix(n, n), estimate, exact, error, seconds
        integer :: repetition
        type(kernel_t) :: kernel
        type(kernel_operator_t) :: operator
        type(fortnum_status_t) :: status

        call deterministic_points(x)
        call rbf_matrix(x, x, signal, scale, matrix)
        call add_diagonal(matrix, shift)
        exact = dense_log_determinant(matrix)
        kernel = make_rbf_kernel(d, signal, scale, status)
        call operator%initialize(x, kernel, shift, status)
        call lanczos_log_determinant(operator, probes, steps, 20260806, &
            estimate, status)
        error = abs(estimate - exact)/max(abs(exact), 1.0_dp)
        if (.not. status_ok(status) .or. error > 5.0e-2_dp) then
            write (output_unit, '(a,3es24.16)') &
                "logdet estimate, exact, relative error: ", estimate, exact, error
            write (output_unit, '(a,i0,2a)') "status ", status%code, ": ", &
                trim(status%msg)
            error stop "logdet benchmark oracle failed"
        end if
        call write_oracle([estimate])
        if (oracle_only_requested()) return

        call timer_start()
        do repetition = 1, reps
            call lanczos_log_determinant(operator, probes, steps, 20260806, &
                estimate, status)
        end do
        call timer_stop(reps, seconds)
        call write_row("logdet", n, d, 1, 0, 0, reps, seconds, error)
    end subroutine benchmark_logdet

    subroutine benchmark_predictive_variance(reps)
        integer, intent(in) :: reps
        integer, parameter :: n = 64, d = 2, n_test = 16, steps = 48
        real(dp), parameter :: signal = 1.4_dp, scale = 0.9_dp
        real(dp), parameter :: shift = 0.35_dp
        real(dp) :: x(n, d), query(n_test, d), matrix(n, n)
        real(dp) :: cross(n), solution(n), expected(n_test), actual(n_test)
        real(dp) :: error, seconds
        integer :: i, repetition, info
        type(kernel_t) :: kernel
        type(kernel_operator_t) :: operator
        type(fortnum_status_t) :: status

        call deterministic_points(x)
        call deterministic_queries(query)
        call rbf_matrix(x, x, signal, scale, matrix)
        call add_diagonal(matrix, shift)
        kernel = make_rbf_kernel(d, signal, scale, status)
        call operator%initialize(x, kernel, shift, status)
        do i = 1, n_test
            call rbf_cross(x, query(i, :), signal, scale, cross)
            call dense_solve(matrix, cross, solution, info)
            if (info /= 0) error stop "predictive variance oracle solve failed"
            expected(i) = signal - dot_product(cross, solution)
            call lanczos_predictive_variance(operator, cross, signal, steps, &
                actual(i), status)
        end do
        error = maxval(abs(actual - expected))/max(maxval(abs(expected)), 1.0_dp)
        if (.not. status_ok(status) .or. error > 2.0e-5_dp) then
            write (output_unit, '(a,3es24.16)') &
                "variance actual, expected, error: ", actual(1), expected(1), error
            write (output_unit, '(a,i0,2a)') "status ", status%code, ": ", &
                trim(status%msg)
            error stop "predictive variance benchmark oracle failed"
        end if
        call write_oracle(actual)
        if (oracle_only_requested()) return

        call timer_start()
        do repetition = 1, reps
            do i = 1, n_test
                call rbf_cross(x, query(i, :), signal, scale, cross)
                call lanczos_predictive_variance(operator, cross, signal, &
                    steps, actual(i), status)
            end do
        end do
        call timer_stop(reps, seconds)
        call write_row("predictive_variance", n, d, 1, n_test, 0, reps, &
            seconds, error)
    end subroutine benchmark_predictive_variance

    subroutine benchmark_derivative_gp(reps)
        integer, intent(in) :: reps
        integer, parameter :: n = 48, d = 1, p = 2, n_test = 24
        real(dp), parameter :: signal = 1.3_dp, scale = 0.7_dp
        real(dp), parameter :: noise = 0.08_dp, jitter = 1.0e-10_dp
        real(dp) :: x(n, d), y(n, p), query(n_test, d)
        real(dp) :: mean(n_test, p), variance(n_test)
        real(dp) :: expected_mean(n_test, p), expected_variance(n_test)
        real(dp) :: error, seconds
        integer :: components(n), query_components(n_test), i, repetition
        type(kernel_t) :: kernel
        type(gp_derivative_regression_t) :: model
        type(fortnum_status_t) :: status

        do i = 1, n
            x(i, 1) = -2.4_dp + 4.8_dp*real(i - 1, dp)/real(n - 1, dp)
            components(i) = mod(i, 2)
            if (components(i) == 0) then
                y(i, 1) = sin(1.3_dp*x(i, 1))
                y(i, 2) = cos(0.7_dp*x(i, 1))
            else
                y(i, 1) = 1.3_dp*cos(1.3_dp*x(i, 1))
                y(i, 2) = -0.7_dp*sin(0.7_dp*x(i, 1))
            end if
        end do
        do i = 1, n_test
            query(i, 1) = -2.2_dp + 4.4_dp*real(i - 1, dp)/ &
                real(n_test - 1, dp)
            query_components(i) = mod(i + 1, 2)
        end do

        kernel = make_rbf_kernel(d, signal, scale, status)
        call model%fit(x, components, y, kernel, noise, status, jitter=jitter)
        call model%predict(query, query_components, mean, variance, status)
        call derivative_reference(x, components, y, query, query_components, &
            signal, scale, noise + jitter, expected_mean, expected_variance)
        error = max(maxval(abs(mean - expected_mean)), &
            maxval(abs(variance - expected_variance)))
        if (.not. status_ok(status) .or. error > 2.0e-9_dp) then
            error stop "derivative GP benchmark oracle failed"
        end if
        call write_oracle([reshape(transpose(mean), [size(mean)]), variance])
        if (oracle_only_requested()) return

        call timer_start()
        do repetition = 1, reps
            call model%fit(x, components, y, kernel, noise, status, jitter=jitter)
            call model%predict(query, query_components, mean, variance, status)
        end do
        call timer_stop(reps, seconds)
        call write_row("derivative", n, d, p, n_test, 0, reps, seconds, error)
    end subroutine benchmark_derivative_gp

    subroutine benchmark_multi_output_gp(reps)
        integer, intent(in) :: reps
        integer, parameter :: n = 24, d = 1, p = 3, rank = 2, n_test = 12
        real(dp), parameter :: signal = 1.2_dp, scale = 0.75_dp
        real(dp), parameter :: noise = 0.12_dp
        real(dp) :: x(n, d), y(n, p), query(n_test, d), mean(n_test, p)
        real(dp) :: expected(n_test, p), weights(p, rank), independent(p)
        real(dp) :: error, seconds
        integer :: i, j, repetition
        type(kernel_t) :: kernel
        type(multi_output_gp_t) :: model
        type(fortnum_status_t) :: status

        do i = 1, n
            x(i, 1) = -1.8_dp + 3.6_dp*real(i - 1, dp)/real(n - 1, dp)
            do j = 1, p
                y(i, j) = sin(real(j, dp)*x(i, 1)) + &
                    0.1_dp*real(j, dp)*cos(0.4_dp*x(i, 1))
            end do
        end do
        do i = 1, n_test
            query(i, 1) = -1.7_dp + 3.4_dp*real(i - 1, dp)/ &
                real(n_test - 1, dp)
        end do
        weights(:, 1) = [0.9_dp, -0.4_dp, 0.6_dp]
        weights(:, 2) = [0.2_dp, 0.7_dp, -0.3_dp]
        independent = [0.25_dp, 0.35_dp, 0.45_dp]

        kernel = make_rbf_kernel(d, signal, scale, status)
        call model%initialize(kernel, weights, independent, noise, status)
        call model%fit(x, y, status)
        call model%predict(query, mean, status)
        call multi_output_reference(x, y, query, weights, independent, signal, &
            scale, noise, expected)
        error = maxval(abs(mean - expected))
        if (.not. status_ok(status) .or. error > 2.0e-9_dp) then
            error stop "multi-output GP benchmark oracle failed"
        end if
        call write_oracle(reshape(transpose(mean), [size(mean)]))
        if (oracle_only_requested()) return

        call timer_start()
        do repetition = 1, reps
            call model%initialize(kernel, weights, independent, noise, status)
            call model%fit(x, y, status)
            call model%predict(query, mean, status)
        end do
        call timer_stop(reps, seconds)
        call write_row("multi_output", n, d, p, n_test, 0, reps, seconds, error)
    end subroutine benchmark_multi_output_gp

    subroutine benchmark_variational_gp(reps)
        integer, intent(in) :: reps
        integer, parameter :: n = 64, d = 1, m = 12, n_test = 20
        real(dp), parameter :: signal = 1.3_dp, scale = 0.7_dp
        real(dp), parameter :: noise = 0.2_dp
        real(dp) :: x(n, d), y(n), inducing(m, d), query(n_test, d)
        real(dp) :: variational_mean(m), factor(m, m), mean(n_test)
        real(dp) :: variance(n_test), expected_mean(n_test)
        real(dp) :: expected_variance(n_test), value, expected_value
        real(dp) :: error, seconds
        integer :: i, repetition
        type(kernel_t) :: kernel
        type(sparse_gp_t) :: model
        type(fortnum_status_t) :: status

        do i = 1, n
            x(i, 1) = -2.0_dp + 4.0_dp*real(i - 1, dp)/real(n - 1, dp)
            y(i) = sin(1.7_dp*x(i, 1)) + 0.05_dp*cos(real(i, dp))
        end do
        do i = 1, m
            inducing(i, 1) = -2.0_dp + 4.0_dp*real(i - 1, dp)/real(m - 1, dp)
            variational_mean(i) = 0.25_dp*sin(0.4_dp*real(i, dp))
        end do
        do i = 1, n_test
            query(i, 1) = -1.9_dp + 3.8_dp*real(i - 1, dp)/ &
                real(n_test - 1, dp)
        end do
        factor = 0.0_dp
        do i = 1, m
            factor(i, i) = 0.55_dp + 0.01_dp*real(i, dp)
            if (i > 1) factor(i, i - 1) = 0.015_dp
        end do

        kernel = make_rbf_kernel(d, signal, scale, status)
        call model%initialize(inducing, kernel, noise, status)
        call model%set_variational(variational_mean, factor, status)
        call model%elbo(x, y, value, status)
        call model%predict(query, mean, variance, status)
        call variational_reference(x, y, inducing, query, variational_mean, &
            factor, signal, scale, noise, expected_value, expected_mean, &
            expected_variance)
        error = max(abs(value - expected_value)/max(abs(expected_value), 1.0_dp), &
            maxval(abs(mean - expected_mean)), &
            maxval(abs(variance - expected_variance)))
        if (.not. status_ok(status) .or. error > 3.0e-9_dp) then
            write (output_unit, '(a,3es24.16)') &
                "variational value, expected, error: ", value, expected_value, error
            error stop "variational GP benchmark oracle failed"
        end if
        call write_oracle([value, mean, variance])
        if (oracle_only_requested()) return

        call timer_start()
        do repetition = 1, reps
            call model%elbo(x, y, value, status)
            call model%predict(query, mean, variance, status)
        end do
        call timer_stop(reps, seconds)
        call write_row("variational", n, d, 1, n_test, m, reps, seconds, error)
    end subroutine benchmark_variational_gp

    subroutine deterministic_points(x)
        real(dp), intent(out) :: x(:, :)
        integer :: i, j

        do j = 1, size(x, 2)
            do i = 1, size(x, 1)
                x(i, j) = 0.4_dp*sin(real(i + 3*j, dp)) + &
                    0.2_dp*cos(real(2*i - j, dp))
            end do
        end do
    end subroutine deterministic_points

    subroutine deterministic_queries(x)
        real(dp), intent(out) :: x(:, :)
        integer :: i, j

        do j = 1, size(x, 2)
            do i = 1, size(x, 1)
                x(i, j) = 0.31_dp*sin(real(2*i + j, dp)) - &
                    0.17_dp*cos(real(i - 2*j, dp))
            end do
        end do
    end subroutine deterministic_queries

    subroutine rbf_matrix(left, right, signal, scale, matrix)
        real(dp), intent(in) :: left(:, :), right(:, :), signal, scale
        real(dp), intent(out) :: matrix(:, :)
        integer :: i, j

        do j = 1, size(right, 1)
            do i = 1, size(left, 1)
                matrix(i, j) = signal*exp(-0.5_dp* &
                    sum((left(i, :) - right(j, :))**2)/(scale*scale))
            end do
        end do
    end subroutine rbf_matrix

    subroutine rbf_cross(points, query, signal, scale, cross)
        real(dp), intent(in) :: points(:, :), query(:), signal, scale
        real(dp), intent(out) :: cross(:)
        integer :: i

        do i = 1, size(points, 1)
            cross(i) = signal*exp(-0.5_dp*sum((points(i, :) - query)**2)/ &
                (scale*scale))
        end do
    end subroutine rbf_cross

    subroutine add_diagonal(matrix, value)
        real(dp), intent(inout) :: matrix(:, :)
        real(dp), intent(in) :: value
        integer :: i

        do i = 1, min(size(matrix, 1), size(matrix, 2))
            matrix(i, i) = matrix(i, i) + value
        end do
    end subroutine add_diagonal

    real(dp) function dense_log_determinant(matrix) result(value)
        real(dp), intent(in) :: matrix(:, :)
        real(dp) :: factor(size(matrix, 1), size(matrix, 2)), total
        integer :: i, j, k

        factor = 0.0_dp
        do i = 1, size(matrix, 1)
            do j = 1, i
                total = matrix(i, j)
                do k = 1, j - 1
                    total = total - factor(i, k)*factor(j, k)
                end do
                if (i == j) then
                    factor(i, j) = sqrt(total)
                else
                    factor(i, j) = total/factor(j, j)
                end if
            end do
        end do
        value = 0.0_dp
        do i = 1, size(matrix, 1)
            value = value + 2.0_dp*log(factor(i, i))
        end do
    end function dense_log_determinant

    subroutine derivative_reference(x, components, y, query, query_components, &
            signal, scale, diagonal_shift, mean, variance)
        real(dp), intent(in) :: x(:, :), y(:, :), query(:, :)
        integer, intent(in) :: components(:), query_components(:)
        real(dp), intent(in) :: signal, scale, diagonal_shift
        real(dp), intent(out) :: mean(:, :), variance(:)
        real(dp) :: matrix(size(x, 1), size(x, 1))
        real(dp) :: cross(size(x, 1), size(query, 1))
        real(dp) :: alpha(size(x, 1), size(y, 2))
        real(dp) :: solved(size(x, 1), size(query, 1)), prior
        integer :: i, j, info

        do j = 1, size(x, 1)
            do i = 1, size(x, 1)
                matrix(i, j) = derivative_rbf(x(i, 1), components(i), &
                    x(j, 1), components(j), signal, scale)
            end do
        end do
        call add_diagonal(matrix, diagonal_shift)
        do j = 1, size(query, 1)
            do i = 1, size(x, 1)
                cross(i, j) = derivative_rbf(x(i, 1), components(i), &
                    query(j, 1), query_components(j), signal, scale)
            end do
        end do
        call dense_solve(matrix, y, alpha, info)
        if (info /= 0) error stop "derivative reference alpha solve failed"
        call dense_solve(matrix, cross, solved, info)
        if (info /= 0) error stop "derivative reference variance solve failed"
        mean = matmul(transpose(cross), alpha)
        do j = 1, size(query, 1)
            prior = derivative_rbf(query(j, 1), query_components(j), &
                query(j, 1), query_components(j), signal, scale)
            variance(j) = prior - dot_product(cross(:, j), solved(:, j))
        end do
    end subroutine derivative_reference

    real(dp) function derivative_rbf(x1, component1, x2, component2, &
            signal, scale) result(value)
        real(dp), intent(in) :: x1, x2, signal, scale
        integer, intent(in) :: component1, component2
        real(dp) :: difference, kernel, inverse_scale_squared

        difference = x1 - x2
        inverse_scale_squared = 1.0_dp/(scale*scale)
        kernel = signal*exp(-0.5_dp*difference*difference*inverse_scale_squared)
        if (component1 == 0 .and. component2 == 0) then
            value = kernel
        else if (component1 == 1 .and. component2 == 0) then
            value = -difference*inverse_scale_squared*kernel
        else if (component1 == 0 .and. component2 == 1) then
            value = difference*inverse_scale_squared*kernel
        else
            value = (inverse_scale_squared - difference*difference* &
                inverse_scale_squared*inverse_scale_squared)*kernel
        end if
    end function derivative_rbf

    subroutine multi_output_reference(x, y, query, weights, independent, &
            signal, scale, noise, mean)
        real(dp), intent(in) :: x(:, :), y(:, :), query(:, :)
        real(dp), intent(in) :: weights(:, :), independent(:)
        real(dp), intent(in) :: signal, scale, noise
        real(dp), intent(out) :: mean(:, :)
        integer :: n, p, m, i, j, a, b, info
        real(dp) :: base(size(x, 1), size(x, 1))
        real(dp) :: cross_base(size(query, 1), size(x, 1))
        real(dp) :: coreg(size(y, 2), size(y, 2))
        real(dp) :: joint(size(x, 1)*size(y, 2), size(x, 1)*size(y, 2))
        real(dp) :: cross(size(query, 1)*size(y, 2), &
            size(x, 1)*size(y, 2))
        real(dp) :: stacked(size(x, 1)*size(y, 2))
        real(dp) :: alpha(size(x, 1)*size(y, 2)), product(size(query, 1)*size(y, 2))

        n = size(x, 1)
        p = size(y, 2)
        m = size(query, 1)
        call rbf_matrix(x, x, signal, scale, base)
        call rbf_matrix(query, x, signal, scale, cross_base)
        coreg = matmul(weights, transpose(weights))
        do i = 1, p
            coreg(i, i) = coreg(i, i) + independent(i)
        end do
        do b = 1, p
            do a = 1, p
                do j = 1, n
                    do i = 1, n
                        joint((a - 1)*n + i, (b - 1)*n + j) = &
                            coreg(a, b)*base(i, j)
                    end do
                    do i = 1, m
                        cross((a - 1)*m + i, (b - 1)*n + j) = &
                            coreg(a, b)*cross_base(i, j)
                    end do
                end do
            end do
        end do
        call add_diagonal(joint, noise)
        do j = 1, p
            stacked((j - 1)*n + 1:j*n) = y(:, j)
        end do
        call dense_solve(joint, stacked, alpha, info)
        if (info /= 0) error stop "multi-output reference solve failed"
        product = matmul(cross, alpha)
        do j = 1, p
            mean(:, j) = product((j - 1)*m + 1:j*m)
        end do
    end subroutine multi_output_reference

    subroutine variational_reference(x, y, inducing, query, variational_mean, &
            factor, signal, scale, noise, value, mean, variance)
        real(dp), intent(in) :: x(:, :), y(:), inducing(:, :), query(:, :)
        real(dp), intent(in) :: variational_mean(:), factor(:, :)
        real(dp), intent(in) :: signal, scale, noise
        real(dp), intent(out) :: value, mean(:), variance(:)
        integer :: m, n, i, info
        real(dp) :: k_uu(size(inducing, 1), size(inducing, 1))
        real(dp) :: k_uf(size(inducing, 1), size(x, 1))
        real(dp) :: k_us(size(inducing, 1), size(query, 1))
        real(dp) :: a(size(inducing, 1), size(x, 1))
        real(dp) :: a_star(size(inducing, 1), size(query, 1))
        real(dp) :: covariance(size(inducing, 1), size(inducing, 1))
        real(dp) :: prior_inverse_covariance(size(inducing, 1), &
            size(inducing, 1))
        real(dp) :: mean_solve(size(inducing, 1)), train_mean(size(x, 1))
        real(dp) :: marginal, residual, likelihood, divergence, trace_term
        real(dp) :: quadratic, log_det_prior, log_det_posterior, pi

        m = size(inducing, 1)
        n = size(x, 1)
        pi = 4.0_dp*atan(1.0_dp)
        call rbf_matrix(inducing, inducing, signal, scale, k_uu)
        call add_diagonal(k_uu, 1.0e-10_dp*max(maxval(abs(k_uu)), 1.0_dp))
        call rbf_matrix(inducing, x, signal, scale, k_uf)
        call rbf_matrix(inducing, query, signal, scale, k_us)
        call dense_solve(k_uu, k_uf, a, info)
        if (info /= 0) error stop "variational reference training solve failed"
        call dense_solve(k_uu, k_us, a_star, info)
        if (info /= 0) error stop "variational reference prediction solve failed"
        covariance = matmul(factor, transpose(factor))
        train_mean = matmul(transpose(a), variational_mean)
        likelihood = 0.0_dp
        do i = 1, n
            marginal = signal - dot_product(a(:, i), k_uf(:, i)) + &
                dot_product(a(:, i), matmul(covariance, a(:, i)))
            residual = y(i) - train_mean(i)
            likelihood = likelihood - 0.5_dp*log(2.0_dp*pi*noise) - &
                0.5_dp*(residual*residual + marginal)/noise
        end do
        prior_inverse_covariance = covariance
        call dense_solve(k_uu, covariance, prior_inverse_covariance, info)
        if (info /= 0) error stop "variational reference covariance solve failed"
        trace_term = 0.0_dp
        do i = 1, m
            trace_term = trace_term + prior_inverse_covariance(i, i)
        end do
        call dense_solve(k_uu, variational_mean, mean_solve, info)
        if (info /= 0) error stop "variational reference mean solve failed"
        quadratic = dot_product(variational_mean, mean_solve)
        log_det_prior = dense_log_determinant(k_uu)
        log_det_posterior = 0.0_dp
        do i = 1, m
            log_det_posterior = log_det_posterior + 2.0_dp*log(factor(i, i))
        end do
        divergence = 0.5_dp*(trace_term + quadratic - real(m, dp) + &
            log_det_prior - log_det_posterior)
        value = likelihood - divergence

        mean = matmul(transpose(a_star), variational_mean)
        do i = 1, size(query, 1)
            variance(i) = signal - dot_product(a_star(:, i), k_us(:, i)) + &
                dot_product(a_star(:, i), matmul(covariance, a_star(:, i)))
        end do
    end subroutine variational_reference

    subroutine timer_start()
        call system_clock(clock_begin, clock_rate)
    end subroutine timer_start

    subroutine timer_stop(repetitions, seconds)
        integer, intent(in) :: repetitions
        real(dp), intent(out) :: seconds
        integer(int64) :: clock_end

        call system_clock(clock_end)
        seconds = real(clock_end - clock_begin, dp)/ &
            (real(clock_rate, dp)*real(repetitions, dp))
    end subroutine timer_stop

    subroutine write_row(name, n, d, p, n_test, m, repetitions, seconds, error)
        character(len=*), intent(in) :: name
        integer, intent(in) :: n, d, p, n_test, m, repetitions
        real(dp), intent(in) :: seconds, error

        write (output_unit, '(a,6(",",i0),2(",",es24.16))') trim(name), n, d, &
            p, n_test, m, repetitions, seconds, error
    end subroutine write_row

    subroutine write_oracle(values)
        real(dp), intent(in) :: values(:)
        character(len=1024) :: path
        integer :: environment_status, i, unit

        call get_environment_variable("FORTML_BENCH_ORACLE", path, &
            status=environment_status)
        if (environment_status /= 0 .or. len_trim(path) == 0) return

        open (newunit=unit, file=trim(path), status="replace", action="write")
        write (unit, '(a)') "index,value"
        do i = 1, size(values)
            write (unit, '(i0,a,es26.17e3)') i, ",", values(i)
        end do
        close (unit)
    end subroutine write_oracle

    logical function oracle_only_requested()
        character(len=16) :: value
        integer :: environment_status

        call get_environment_variable("FORTML_BENCH_ORACLE_ONLY", value, &
            status=environment_status)
        oracle_only_requested = environment_status == 0 .and. &
            trim(value) == "1"
    end function oracle_only_requested

end program fortml_bench_gp_features
