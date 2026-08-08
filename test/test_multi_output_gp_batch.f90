program test_multi_output_gp_batch
    !! Independent shape, derivative, and device contracts for batched queries.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_quiet_nan
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    use fortml_kernels, only: kernel_t, make_rbf_kernel
    use fortml_multi_output_gp, only: multi_output_gp_t
    implicit none

    integer, parameter :: n = 5, d = 1, p = 2, batch = 2, m = 3
    real(dp), parameter :: variance = 1.3_dp, lengthscale = 0.7_dp
    real(dp), parameter :: noise = 0.14_dp, eps = 2.0e-6_dp
    real(dp) :: x(n, d), y(n, p), query(batch, m, d), direction(batch, m, d)
    real(dp) :: mean(batch, m, p), expected(batch, m, p)
    real(dp) :: mean_dot(batch, m, p), mean_plus(batch, m, p), mean_minus(batch, m, p)
    real(dp) :: mean_bar(batch, m, p), query_bar(batch, m, d)
    real(dp) :: qplus(batch, m, d), qminus(batch, m, d)
    real(dp) :: weights(p, 1), independent(p)
    type(kernel_t) :: kernel
    type(multi_output_gp_t) :: model
    type(fortnum_status_t) :: status
    type(fortml_device_t) :: cpu, cuda
    real(dp) :: max_error, product_left, product_right
    real(dp) :: coreg(p, p), kval, oracle
    integer :: failures, i, j, a, b, t
    real(dp), allocatable :: bad_mean(:, :, :), bad_query(:, :, :)

    do i = 1, n
        x(i, 1) = -0.8_dp + 0.35_dp*real(i - 1, dp)
        y(i, 1) = sin(1.2_dp*x(i, 1))
        y(i, 2) = cos(0.9_dp*x(i, 1)) - 0.2_dp
    end do
    do t = 1, batch
        do i = 1, m
            query(t, i, 1) = -0.55_dp + 0.27_dp*real(i - 1, dp) + &
                0.18_dp*real(t - 1, dp)
            direction(t, i, 1) = 0.11_dp - 0.035_dp*real(i, dp) + &
                0.04_dp*real(t - 1, dp)
        end do
    end do
    mean_bar = reshape([0.3_dp, -0.2_dp, 0.7_dp, -0.4_dp, 0.1_dp, 0.6_dp, &
        -0.5_dp, 0.8_dp, -0.25_dp, 0.35_dp, -0.15_dp, 0.45_dp], shape(mean_bar))
    weights(:, 1) = [0.75_dp, -0.4_dp]
    independent = [0.22_dp, 0.31_dp]
    kernel = make_rbf_kernel(d, variance, lengthscale, status)
    call model%initialize(kernel, weights, independent, noise, status)
    call model%fit(x, y, status)
    failures = 0
    call check(status_ok(status), "fit", failures)

    call model%predict_batch(query, mean, status)
    call check(status_ok(status), "batch prediction status", failures)
    do j = 1, p
        do i = 1, p
            coreg(i, j) = weights(i, 1)*weights(j, 1)
        end do
        coreg(j, j) = coreg(j, j) + independent(j)
    end do
    expected = 0.0_dp
    do t = 1, batch
        do i = 1, m
            do a = 1, p
                do b = 1, p
                    do j = 1, n
                        kval = variance*exp(-0.5_dp*(query(t, i, 1) - x(j, 1))**2/ &
                            (lengthscale*lengthscale))
                        oracle = coreg(a, b)*kval*model%alpha((b - 1)*n + j)
                        expected(t, i, a) = expected(t, i, a) + oracle
                    end do
                end do
            end do
        end do
    end do
    max_error = maxval(abs(mean - expected))
    call check(max_error < 2.0e-11_dp, "batch prediction dense oracle", failures)

    call model%predict_batch_input_jvp(query, direction, mean, mean_dot, status)
    call check(status_ok(status), "batch input JVP status", failures)
    qplus = query + eps*direction
    qminus = query - eps*direction
    call model%predict_batch(qplus, mean_plus, status)
    call model%predict_batch(qminus, mean_minus, status)
    max_error = maxval(abs(mean_dot - (mean_plus - mean_minus)/(2.0_dp*eps)))
    call check(max_error < 4.0e-5_dp, "batch input JVP finite difference", failures)

    call model%predict_batch_input_vjp(query, mean_bar, query_bar, status)
    call check(status_ok(status), "batch input VJP status", failures)
    product_left = sum(mean_bar*mean_dot)
    product_right = sum(query_bar*direction)
    call check(abs(product_left - product_right) < 4.0e-10_dp, &
        "batch input VJP adjoint", failures)

    cpu%kind = FORTML_DEVICE_CPU
    call model%predict_batch_device(cpu, query, mean_plus, status)
    call check(status_ok(status), "CPU batch device dispatch", failures)
    call check(maxval(abs(mean_plus - mean)) < 2.0e-11_dp, &
        "CPU batch device value", failures)
    cuda%kind = FORTML_DEVICE_CUDA
    call model%predict_batch_device(cuda, query, mean_plus, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "CUDA batch device refusal", failures)
    call model%predict_batch_input_jvp_device(cuda, query, direction, mean, mean_dot, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "CUDA batch JVP refusal", failures)
    call model%predict_batch_input_vjp_device(cuda, query, mean_bar, query_bar, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "CUDA batch VJP refusal", failures)

    allocate(bad_mean(batch, m, p + 1), bad_query(batch, m, d + 1))
    call model%predict_batch(query, bad_mean, status)
    call check(.not. status_ok(status), "batch output shape refusal", failures)
    call model%predict_batch(bad_query, mean, status)
    call check(.not. status_ok(status), "batch input shape refusal", failures)
    direction(1, 1, 1) = ieee_value(0.0_dp, ieee_quiet_nan)
    call model%predict_batch_input_jvp(query, direction, mean, mean_dot, status)
    call check(.not. status_ok(status), "batch nonfinite direction refusal", failures)

    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, " multi-output batch test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS"

contains

    subroutine check(condition, label, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: label
        integer, intent(inout) :: failures

        if (.not. condition) then
            write (error_unit, '(a)') "FAIL ["//label//"]"
            failures = failures + 1
        end if
    end subroutine check

end program test_multi_output_gp_batch
