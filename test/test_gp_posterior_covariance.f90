program test_gp_posterior_covariance
    !! Independent dense oracle for exact GP posterior covariance.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    use fortml_kernels, only: kernel_t, make_rbf_kernel
    use fortml_gaussian_process, only: gp_regression_t
    implicit none

    integer, parameter :: n = 5, m = 3, d = 1, p = 2
    real(dp), parameter :: signal = 1.4_dp, lengthscale = 0.65_dp
    real(dp), parameter :: noise = 0.09_dp, jitter = 1.0e-10_dp
    real(dp) :: x(n, d), y(n, p), query(m, d)
    real(dp) :: covariance(m, m), expected(m, m), prior(m, m)
    real(dp) :: cross(n, m), train(n, n), solved(n, m)
    real(dp) :: mean(m, p), variance(m), expected_variance(m)
    type(gp_regression_t) :: model
    type(kernel_t) :: kernel
    type(fortnum_status_t) :: status
    type(fortml_device_t) :: cpu, cuda
    integer :: failures, i, j

    do i = 1, n
        x(i, 1) = -0.8_dp + 0.37_dp*real(i - 1, dp)
        y(i, 1) = sin(1.1_dp*x(i, 1))
        y(i, 2) = cos(0.8_dp*x(i, 1)) - 0.25_dp
    end do
    do i = 1, m
        query(i, 1) = -0.63_dp + 0.51_dp*real(i - 1, dp)
    end do

    failures = 0
    kernel = make_rbf_kernel(d, signal, lengthscale, status)
    call model%fit(x, y, kernel, noise, status, jitter)
    call check(status_ok(status), "fit", failures)
    call model%predict(query, mean, variance, status)
    call check(status_ok(status), "marginal prediction", failures)
    call model%predict_covariance(query, covariance, status)
    call check(status_ok(status), "posterior covariance", failures)

    do j = 1, n
        do i = 1, n
            train(i, j) = signal*exp(-0.5_dp*(x(i, 1) - x(j, 1))**2/lengthscale**2)
        end do
    end do
    do i = 1, n
        train(i, i) = train(i, i) + noise + jitter
    end do
    do j = 1, m
        do i = 1, n
            cross(i, j) = signal*exp(-0.5_dp*(x(i, 1) - query(j, 1))**2/lengthscale**2)
        end do
    end do
    do j = 1, m
        do i = 1, m
            prior(i, j) = signal*exp(-0.5_dp*(query(i, 1) - query(j, 1))**2/lengthscale**2)
        end do
    end do
    solved = cross
    call dense_solve_matrix(train, solved)
    expected = prior - matmul(transpose(cross), solved)
    expected = 0.5_dp*(expected + transpose(expected))
    expected_variance = [(expected(i, i), i = 1, m)]
    call check(maxval(abs(covariance - expected)) < 3.0e-11_dp, &
        "dense posterior covariance oracle", failures)
    call check(maxval(abs(covariance - transpose(covariance))) < 2.0e-13_dp, &
        "covariance symmetry", failures)
    call check(maxval(abs(variance - expected_variance)) < 3.0e-11_dp, &
        "covariance diagonal agrees with marginal variance", failures)

    cpu%kind = FORTML_DEVICE_CPU
    cpu%selected = .true.
    cpu%available = .true.
    call model%predict_covariance_device(cpu, query, covariance, status)
    call check(status_ok(status), "CPU covariance dispatch", failures)
    call check(maxval(abs(covariance - expected)) < 3.0e-11_dp, &
        "CPU covariance value", failures)
    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    covariance = 7.0_dp
    call model%predict_covariance_device(cuda, query, covariance, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, "CUDA typed refusal", failures)
    call check(maxval(abs(covariance)) == 0.0_dp, "CUDA refusal clears output", failures)

    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, " GP posterior covariance test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS"

contains

    subroutine dense_solve_matrix(matrix, rhs)
        real(dp), intent(in) :: matrix(:, :)
        real(dp), intent(inout) :: rhs(:, :)
        real(dp), allocatable :: a(:, :)
        real(dp) :: factor, swap
        integer :: i, j, k, pivot, size_matrix

        size_matrix = size(matrix, 1)
        allocate(a, source=matrix)
        do k = 1, size_matrix - 1
            pivot = k
            do i = k + 1, size_matrix
                if (abs(a(i, k)) > abs(a(pivot, k))) pivot = i
            end do
            if (pivot /= k) then
                do j = 1, size_matrix
                    swap = a(k, j)
                    a(k, j) = a(pivot, j)
                    a(pivot, j) = swap
                end do
                do j = 1, size(rhs, 2)
                    swap = rhs(k, j)
                    rhs(k, j) = rhs(pivot, j)
                    rhs(pivot, j) = swap
                end do
            end if
            do i = k + 1, size_matrix
                factor = a(i, k)/a(k, k)
                a(i, k:) = a(i, k:) - factor*a(k, k:)
                rhs(i, :) = rhs(i, :) - factor*rhs(k, :)
            end do
        end do
        do k = size_matrix, 1, -1
            rhs(k, :) = rhs(k, :)/a(k, k)
            do i = 1, k - 1
                rhs(i, :) = rhs(i, :) - a(i, k)*rhs(k, :)
            end do
        end do
    end subroutine dense_solve_matrix

    subroutine check(condition, label, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: label
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "FAIL ["//label//"]"
        end if
    end subroutine check

end program test_gp_posterior_covariance
