program test_lanczos
    !! Oracles for the Lanczos log determinant and the LOVE-style predictive
    !! variance.
    !!
    !! Both are checked against dense linear algebra on the same operator: the
    !! log determinant against the sum of logs of the Cholesky diagonal, and
    !! the predictive variance against an LU solve of the full system. With as
    !! many Lanczos steps as samples the variance estimate is exact up to
    !! round-off, so that case is a hard equality rather than a tolerance on a
    !! sampling error.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_kernels, only: kernel_t, make_rbf_kernel
    use fortml_kernel_operator, only: kernel_operator_t
    use fortml_lanczos, only: lanczos_log_determinant, lanczos_predictive_variance
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer, parameter :: n = 24, d = 2
    real(dp), parameter :: variance = 1.4_dp
    real(dp), parameter :: lengthscale = 0.9_dp
    real(dp), parameter :: shift = 0.35_dp
    real(dp) :: points(n, d), covariance(n, n)
    integer :: failures

    call build_points(points)
    call build_covariance(points, covariance)
    failures = 0
    call test_log_determinant(failures)
    call test_predictive_variance(failures)
    call test_refusals(failures)
    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, " Lanczos test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS"

contains

    subroutine build_points(x)
        real(dp), intent(out) :: x(:, :)
        integer :: i, j

        do j = 1, size(x, 2)
            do i = 1, size(x, 1)
                x(i, j) = 0.4_dp*sin(real(i + 3*j, dp)) &
                    + 0.2_dp*cos(real(2*i - j, dp))
            end do
        end do
    end subroutine build_points

    subroutine build_covariance(x, matrix)
        !! The dense matrix the operator represents, formed independently here.
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: matrix(:, :)
        integer :: i, j

        do j = 1, size(x, 1)
            do i = 1, size(x, 1)
                matrix(i, j) = variance*exp(-0.5_dp*sum((x(i, :) - x(j, :))**2)/ &
                    (lengthscale*lengthscale))
                if (i == j) matrix(i, j) = matrix(i, j) + shift
            end do
        end do
    end subroutine build_covariance

    subroutine build_operator(operator, status)
        type(kernel_operator_t), intent(out) :: operator
        type(fortnum_status_t), intent(out) :: status
        type(kernel_t) :: kernel

        kernel = make_rbf_kernel(d, variance, lengthscale, status)
        call operator%initialize(points, kernel, shift, status)
    end subroutine build_operator

    real(dp) function dense_log_determinant(matrix) result(value)
        !! Cholesky by hand, so the oracle shares no code with the estimator.
        real(dp), intent(in) :: matrix(:, :)
        real(dp) :: factor(size(matrix, 1), size(matrix, 1)), total
        integer :: i, j, k

        factor = 0.0_dp
        do i = 1, size(matrix, 1)
            do j = 1, i
                total = matrix(i, j)
                do k = 1, j - 1
                    total = total - factor(i, k)*factor(j, k)
                end do
                if (i == j) then
                    factor(i, i) = sqrt(total)
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

    subroutine dense_solve(matrix, rhs, solution)
        !! Gaussian elimination with partial pivoting.
        real(dp), intent(in) :: matrix(:, :), rhs(:)
        real(dp), intent(out) :: solution(:)
        real(dp) :: a(size(rhs), size(rhs)), b(size(rhs)), factor, swap
        integer :: i, j, k, pivot

        a = matrix
        b = rhs
        do k = 1, size(b) - 1
            pivot = k
            do i = k + 1, size(b)
                if (abs(a(i, k)) > abs(a(pivot, k))) pivot = i
            end do
            if (pivot /= k) then
                do j = 1, size(b)
                    swap = a(k, j)
                    a(k, j) = a(pivot, j)
                    a(pivot, j) = swap
                end do
                swap = b(k)
                b(k) = b(pivot)
                b(pivot) = swap
            end if
            do i = k + 1, size(b)
                factor = a(i, k)/a(k, k)
                a(i, k:) = a(i, k:) - factor*a(k, k:)
                b(i) = b(i) - factor*b(k)
            end do
        end do
        do i = size(b), 1, -1
            solution(i) = (b(i) - sum(a(i, i + 1:)*solution(i + 1:)))/a(i, i)
        end do
    end subroutine dense_solve

    subroutine test_log_determinant(failures)
        integer, intent(inout) :: failures
        type(kernel_operator_t) :: operator
        type(fortnum_status_t) :: status
        real(dp) :: estimate, repeat, other_seed, exact, relative

        call build_operator(operator, status)
        exact = dense_log_determinant(covariance)
        call lanczos_log_determinant(operator, 64, n, 20260806, estimate, status)
        relative = abs(estimate - exact)/abs(exact)
        if (.not. status_ok(status) .or. relative > 0.05_dp) then
            write (error_unit, '(a,3es12.4)') &
                "FAIL [logdet] estimate, exact, relative error ", estimate, &
                exact, relative
            failures = failures + 1
        end if

        call lanczos_log_determinant(operator, 64, n, 20260806, repeat, status)
        call lanczos_log_determinant(operator, 64, n, 991, other_seed, status)
        if (abs(estimate - repeat) > 0.0_dp) then
            write (error_unit, '(a)') "FAIL [logdet] the same seed differs"
            failures = failures + 1
        end if
        if (abs(estimate - other_seed) < 1.0e-12_dp) then
            write (error_unit, '(a)') "FAIL [logdet] different seeds agree"
            failures = failures + 1
        end if
        if (abs(other_seed - exact)/abs(exact) > 0.05_dp) then
            write (error_unit, '(a)') &
                "FAIL [logdet] a second seed misses the exact value"
            failures = failures + 1
        end if
    end subroutine test_log_determinant

    subroutine test_predictive_variance(failures)
        integer, intent(inout) :: failures
        type(kernel_operator_t) :: operator
        type(fortnum_status_t) :: status
        real(dp) :: cross(n), solution(n), estimate, exact, truncated
        real(dp) :: query(d)
        integer :: i

        call build_operator(operator, status)
        query = [0.21_dp, -0.34_dp]
        do i = 1, n
            cross(i) = variance*exp(-0.5_dp*sum((points(i, :) - query)**2)/ &
                (lengthscale*lengthscale))
        end do
        call dense_solve(covariance, cross, solution)
        exact = variance - sum(cross*solution)

        call lanczos_predictive_variance(operator, cross, variance, n, &
            estimate, status)
        if (.not. status_ok(status) .or. abs(estimate - exact) > 1.0e-9_dp) then
            write (error_unit, '(a,2es12.4)') &
                "FAIL [love] full-rank variance ", estimate, exact
            failures = failures + 1
        end if

        ! A truncated run is an approximation, but it must stay a variance:
        ! non-negative and no larger than the prior.
        call lanczos_predictive_variance(operator, cross, variance, 4, &
            truncated, status)
        if (.not. status_ok(status) .or. truncated < 0.0_dp .or. &
            truncated > variance) then
            write (error_unit, '(a,es12.4)') &
                "FAIL [love] truncated variance left its bounds ", truncated
            failures = failures + 1
        end if

        ! A zero cross-covariance leaves the prior untouched.
        cross = 0.0_dp
        call lanczos_predictive_variance(operator, cross, variance, n, &
            estimate, status)
        if (.not. status_ok(status) .or. abs(estimate - variance) > 0.0_dp) then
            write (error_unit, '(a)') "FAIL [love] zero cross-covariance"
            failures = failures + 1
        end if
    end subroutine test_predictive_variance

    subroutine test_refusals(failures)
        integer, intent(inout) :: failures
        type(kernel_operator_t) :: operator
        type(fortnum_status_t) :: status
        real(dp) :: value, cross(n + 1)

        call build_operator(operator, status)
        call lanczos_log_determinant(operator, 0, n, 7, value, status)
        if (status_ok(status)) then
            write (error_unit, '(a)') "FAIL [guard] zero probes accepted"
            failures = failures + 1
        end if
        call lanczos_log_determinant(operator, 4, 0, 7, value, status)
        if (status_ok(status)) then
            write (error_unit, '(a)') "FAIL [guard] zero Lanczos steps accepted"
            failures = failures + 1
        end if
        cross = 1.0_dp
        call lanczos_predictive_variance(operator, cross, variance, n, value, &
            status)
        if (status_ok(status)) then
            write (error_unit, '(a)') &
                "FAIL [guard] mismatched cross-covariance accepted"
            failures = failures + 1
        end if
    end subroutine test_refusals

end program test_lanczos
