program test_multi_output_gp_training
    !! Independent dense oracle for the multi-output GP FortOpt adapter.
    !!
    !! The reference LML below assembles the ICM covariance and solves it with
    !! a local Gaussian-elimination implementation.  It therefore does not
    !! call the production covariance contractions or factorization.  The test
    !! also checks that bounded L-BFGS-B improves (or preserves) the exact
    !! objective and that CUDA requests are refused explicitly.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    use fortml_kernels, only: kernel_t, make_rbf_kernel
    use fortml_multi_output_gp, only: multi_output_gp_t
    use fortml_multi_output_gp_training, only: &
        multi_output_gp_hyperparameter_options_t, &
        multi_output_gp_hyperparameter_result_t, &
        multi_output_gp_optimize_hyperparameters
    implicit none

    integer, parameter :: n = 4, d = 1, p = 2, total = n*p
    real(dp), parameter :: signal = 1.2_dp, lengthscale = 0.7_dp
    real(dp), parameter :: noise = 0.12_dp
    real(dp) :: x(n, d), y(n, p), weights(p, 1), independent(p)
    real(dp) :: reference, production, optimized, initial_negative
    real(dp) :: matrix(total, total), stacked(total)
    type(kernel_t) :: kernel
    type(multi_output_gp_t) :: model
    type(multi_output_gp_hyperparameter_options_t) :: options
    type(multi_output_gp_hyperparameter_result_t) :: result
    type(fortml_device_t) :: cpu, cuda
    type(fortnum_status_t) :: status
    integer :: failures, i

    do i = 1, n
        x(i, 1) = -0.75_dp + 0.45_dp*real(i - 1, dp)
        y(i, 1) = sin(1.1_dp*x(i, 1))
        y(i, 2) = cos(0.8_dp*x(i, 1)) - 0.15_dp
    end do
    weights(:, 1) = [0.8_dp, -0.4_dp]
    independent = [0.25_dp, 0.3_dp]
    failures = 0

    kernel = make_rbf_kernel(d, signal, lengthscale, status)
    call check(status_ok(status), "kernel initialization", failures)
    call model%initialize(kernel, weights, independent, noise, status)
    call check(status_ok(status), "model initialization", failures)
    call model%fit(x, y, status)
    call check(status_ok(status), "model fit", failures)

    call model%log_marginal_likelihood(y, production, status)
    call check(status_ok(status), "production LML", failures)
    call assemble_reference(x, y, weights, independent, signal, lengthscale, noise, &
        matrix, stacked)
    reference = dense_lml(matrix, stacked)
    call check(abs(production - reference) < 2.0e-10_dp, &
        "independent dense LML oracle", failures)

    initial_negative = -production
    options%max_iterations = 150
    options%max_line_search = 80
    options%gradient_tolerance = 5.0e-3_dp
    options%kernel_lower_bound = -8.0_dp
    options%kernel_upper_bound = 8.0_dp
    options%noise_lower_bound = -8.0_dp
    options%noise_upper_bound = 8.0_dp
    options%weight_lower_bound = -5.0_dp
    options%weight_upper_bound = 5.0_dp
    options%independent_upper_bound = 5.0_dp
    options%starts = 4
    options%seed = 1234
    cpu%kind = FORTML_DEVICE_CPU
    cpu%selected = .true.
    cpu%available = .true.
    call multi_output_gp_optimize_hyperparameters(model, options, result, status, cpu)
    call check(status_ok(status) .and. result%converged, &
        "FortOpt multi-output optimization", failures)
    call model%log_marginal_likelihood(y, optimized, status)
    call check(status_ok(status) .and. abs(result%negative_log_marginal_likelihood + optimized) &
        < 2.0e-8_dp, "optimizer result matches fitted LML", failures)
    call check(result%negative_log_marginal_likelihood <= initial_negative + 2.0e-8_dp, &
        "bounded optimizer does not worsen objective", failures)

    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call multi_output_gp_optimize_hyperparameters(model, options, result, status, cuda)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "CUDA optimizer capability refusal", failures)

    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, " multi-output GP training test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS multi-output GP FortOpt independent oracle"

contains

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failures

        if (condition) return
        write (error_unit, '(a)') "FAIL: "//description
        failures = failures + 1
    end subroutine check

    subroutine assemble_reference(inputs, targets, local_weights, local_independent, &
            variance, local_lengthscale, local_noise, covariance, response)
        real(dp), intent(in) :: inputs(:, :), targets(:, :), local_weights(:, :)
        real(dp), intent(in) :: local_independent(:), variance, local_lengthscale, local_noise
        real(dp), intent(out) :: covariance(:, :), response(:)
        real(dp) :: input_covariance(size(inputs, 1), size(inputs, 1))
        real(dp) :: coregionalization(size(local_independent), size(local_independent))
        integer :: i, j, a, b, sample_count, output_count

        sample_count = size(inputs, 1)
        output_count = size(local_independent)
        do j = 1, sample_count
            do i = 1, sample_count
                input_covariance(i, j) = variance*exp(-0.5_dp*sum((inputs(i, :) - &
                    inputs(j, :))**2)/(local_lengthscale*local_lengthscale))
            end do
        end do
        coregionalization = matmul(local_weights, transpose(local_weights))
        do i = 1, output_count
            coregionalization(i, i) = coregionalization(i, i) + local_independent(i)
        end do
        do b = 1, output_count
            do a = 1, output_count
                do j = 1, sample_count
                    do i = 1, sample_count
                        covariance((a - 1)*sample_count + i, (b - 1)*sample_count + j) = &
                            coregionalization(a, b)*input_covariance(i, j)
                    end do
                end do
            end do
        end do
        do i = 1, sample_count*output_count
            covariance(i, i) = covariance(i, i) + local_noise
        end do
        do j = 1, output_count
            do i = 1, sample_count
                response((j - 1)*sample_count + i) = targets(i, j)
            end do
        end do
    end subroutine assemble_reference

    real(dp) function dense_lml(matrix, response) result(value)
        real(dp), intent(in) :: matrix(:, :), response(:)
        real(dp) :: a(size(response), size(response)), b(size(response))
        real(dp) :: factor, logdet, quadratic
        integer :: i, j, k, m

        m = size(response)
        a = matrix
        b = response
        logdet = 0.0_dp
        do k = 1, m
            logdet = logdet + log(a(k, k))
            do i = k + 1, m
                factor = a(i, k)/a(k, k)
                a(i, k:) = a(i, k:) - factor*a(k, k:)
                b(i) = b(i) - factor*b(k)
            end do
        end do
        do i = m, 1, -1
            b(i) = (b(i) - sum(a(i, i + 1:)*b(i + 1:)))/a(i, i)
        end do
        quadratic = dot_product(response, b)
        value = -0.5_dp*quadratic - 0.5_dp*logdet - &
            0.5_dp*real(m, dp)*log(2.0_dp*acos(-1.0_dp))
    end function dense_lml

end program test_multi_output_gp_training
