program test_sparse_gp
    !! Oracles for the inducing-point variational GP.
    !!
    !! The strong case is an identity, not a tolerance: put the inducing inputs
    !! on the data, set `q(u)` to the exact GP posterior over `f`, and the ELBO
    !! must equal the exact log marginal likelihood computed independently from
    !! a dense Cholesky. Away from that setting the ELBO must stay a strict
    !! lower bound, and the predictive marginals must match the exact posterior
    !! in the same collapsed case.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_kernels, only: kernel_t, make_rbf_kernel
    use fortml_sparse_gp, only: sparse_gp_t
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer, parameter :: n = 9, d = 1
    real(dp), parameter :: variance = 1.3_dp
    real(dp), parameter :: lengthscale = 0.7_dp
    real(dp), parameter :: noise = 0.2_dp
    real(dp) :: x(n, d), y(n)
    integer :: failures

    call build_data(x, y)
    failures = 0
    call test_collapsed_bound_is_tight(failures)
    call test_bound_is_a_lower_bound(failures)
    call test_predictions_match_the_exact_posterior(failures)
    call test_refusals(failures)
    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, " sparse GP test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS"

contains

    subroutine build_data(inputs, targets)
        real(dp), intent(out) :: inputs(:, :), targets(:)
        integer :: i

        do i = 1, n
            inputs(i, 1) = -1.0_dp + 0.25_dp*real(i, dp)
            targets(i) = sin(1.7_dp*inputs(i, 1)) + 0.1_dp*cos(real(i, dp))
        end do
    end subroutine build_data

    subroutine dense_covariance(matrix, with_noise)
        real(dp), intent(out) :: matrix(:, :)
        logical, intent(in) :: with_noise
        integer :: i, j

        do j = 1, n
            do i = 1, n
                matrix(i, j) = variance*exp(-0.5_dp*(x(i, 1) - x(j, 1))**2/ &
                    (lengthscale*lengthscale))
            end do
        end do
        if (.not. with_noise) return
        do i = 1, n
            matrix(i, i) = matrix(i, i) + noise
        end do
    end subroutine dense_covariance

    subroutine cholesky(matrix, factor)
        real(dp), intent(in) :: matrix(:, :)
        real(dp), intent(out) :: factor(:, :)
        real(dp) :: total
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
    end subroutine cholesky

    subroutine cholesky_solve(factor, rhs, solution)
        real(dp), intent(in) :: factor(:, :), rhs(:)
        real(dp), intent(out) :: solution(:)
        integer :: i, k
        real(dp) :: total

        do i = 1, size(rhs)
            total = rhs(i)
            do k = 1, i - 1
                total = total - factor(i, k)*solution(k)
            end do
            solution(i) = total/factor(i, i)
        end do
        do i = size(rhs), 1, -1
            total = solution(i)
            do k = i + 1, size(rhs)
                total = total - factor(k, i)*solution(k)
            end do
            solution(i) = total/factor(i, i)
        end do
    end subroutine cholesky_solve

    real(dp) function exact_log_marginal_likelihood() result(value)
        real(dp) :: matrix(n, n), factor(n, n), alpha(n)
        integer :: i

        call dense_covariance(matrix, .true.)
        call cholesky(matrix, factor)
        call cholesky_solve(factor, y, alpha)
        value = -0.5_dp*sum(y*alpha)
        do i = 1, n
            value = value - log(factor(i, i))
        end do
        value = value - 0.5_dp*real(n, dp)*log(8.0_dp*atan(1.0_dp))
    end function exact_log_marginal_likelihood

    subroutine exact_posterior(mean, covariance)
        !! Posterior over the latent values at the data: mean `K(K+sI)^{-1}y`
        !! and covariance `K - K(K+sI)^{-1}K`.
        real(dp), intent(out) :: mean(:), covariance(:, :)
        real(dp) :: prior(n, n), shifted(n, n), factor(n, n), column(n)
        integer :: j

        call dense_covariance(prior, .false.)
        call dense_covariance(shifted, .true.)
        call cholesky(shifted, factor)
        call cholesky_solve(factor, y, column)
        mean = matmul(prior, column)
        covariance = prior
        do j = 1, n
            call cholesky_solve(factor, prior(:, j), column)
            covariance(:, j) = covariance(:, j) - matmul(prior, column)
        end do
    end subroutine exact_posterior

    subroutine test_collapsed_bound_is_tight(failures)
        integer, intent(inout) :: failures
        type(sparse_gp_t) :: model
        type(kernel_t) :: kernel
        type(fortnum_status_t) :: status
        real(dp) :: mean(n), covariance(n, n), factor(n, n)
        real(dp) :: value, exact

        kernel = make_rbf_kernel(d, variance, lengthscale, status)
        call model%initialize(x, kernel, noise, status)
        call exact_posterior(mean, covariance)
        call cholesky(covariance, factor)
        call model%set_variational(mean, factor, status)
        call model%elbo(x, y, value, status)
        exact = exact_log_marginal_likelihood()

        ! The identity is exact only for a jitter-free K_uu; the operator adds
        ! a relative 1e-10 jitter for safety, and that is what this tolerance
        ! covers.
        if (.not. status_ok(status) .or. abs(value - exact) > 1.0e-6_dp) then
            write (error_unit, '(a,2es16.8)') &
                "FAIL [collapsed] ELBO does not reach the exact evidence ", &
                value, exact
            failures = failures + 1
        end if
    end subroutine test_collapsed_bound_is_tight

    subroutine test_bound_is_a_lower_bound(failures)
        integer, intent(inout) :: failures
        type(sparse_gp_t) :: model
        type(kernel_t) :: kernel
        type(fortnum_status_t) :: status
        real(dp) :: inducing(4, d), mean(4), factor(4, 4)
        real(dp) :: value, likelihood, kl_value, exact
        integer :: i

        do i = 1, 4
            inducing(i, 1) = -0.8_dp + 0.5_dp*real(i, dp)
        end do
        kernel = make_rbf_kernel(d, variance, lengthscale, status)
        call model%initialize(inducing, kernel, noise, status)
        factor = 0.0_dp
        do i = 1, 4
            mean(i) = 0.3_dp*sin(real(i, dp))
            factor(i, i) = 0.6_dp + 0.1_dp*real(i, dp)
        end do
        factor(3, 1) = 0.05_dp
        call model%set_variational(mean, factor, status)
        call model%elbo(x, y, value, status, &
            expected_log_likelihood=likelihood, kl_value=kl_value)
        exact = exact_log_marginal_likelihood()

        if (.not. status_ok(status) .or. value > exact) then
            write (error_unit, '(a,2es16.8)') &
                "FAIL [bound] ELBO exceeds the exact evidence ", value, exact
            failures = failures + 1
        end if
        if (abs(value - (likelihood - kl_value)) > 1.0e-12_dp) then
            write (error_unit, '(a)') "FAIL [bound] decomposition is inconsistent"
            failures = failures + 1
        end if
        if (kl_value < 0.0_dp) then
            write (error_unit, '(a,es12.4)') "FAIL [bound] negative KL ", kl_value
            failures = failures + 1
        end if
    end subroutine test_bound_is_a_lower_bound

    subroutine test_predictions_match_the_exact_posterior(failures)
        integer, intent(inout) :: failures
        type(sparse_gp_t) :: model
        type(kernel_t) :: kernel
        type(fortnum_status_t) :: status
        real(dp) :: mean(n), covariance(n, n), factor(n, n)
        real(dp) :: predicted_mean(n), predicted_variance(n)
        real(dp) :: expected_variance(n)
        integer :: i

        kernel = make_rbf_kernel(d, variance, lengthscale, status)
        call model%initialize(x, kernel, noise, status)
        call exact_posterior(mean, covariance)
        call cholesky(covariance, factor)
        call model%set_variational(mean, factor, status)
        call model%predict(x, predicted_mean, predicted_variance, status)
        do i = 1, n
            expected_variance(i) = covariance(i, i)
        end do

        if (.not. status_ok(status) .or. &
            maxval(abs(predicted_mean - mean)) > 1.0e-6_dp .or. &
            maxval(abs(predicted_variance - expected_variance)) > 1.0e-6_dp) then
            write (error_unit, '(a,2es12.4)') &
                "FAIL [predict] collapsed marginals ", &
                maxval(abs(predicted_mean - mean)), &
                maxval(abs(predicted_variance - expected_variance))
            failures = failures + 1
        end if
    end subroutine test_predictions_match_the_exact_posterior

    subroutine test_refusals(failures)
        integer, intent(inout) :: failures
        type(sparse_gp_t) :: model
        type(kernel_t) :: kernel
        type(fortnum_status_t) :: status
        real(dp) :: mean(n), factor(n, n), value
        real(dp) :: short_mean(3)

        kernel = make_rbf_kernel(d, variance, lengthscale, status)
        call model%initialize(x, kernel, -1.0_dp, status)
        if (status_ok(status)) then
            write (error_unit, '(a)') "FAIL [guard] negative noise accepted"
            failures = failures + 1
        end if
        call model%initialize(x, kernel, noise, status)
        call model%set_variational(short_mean, factor, status)
        if (status_ok(status)) then
            write (error_unit, '(a)') "FAIL [guard] a short mean accepted"
            failures = failures + 1
        end if
        mean = 0.0_dp
        factor = 0.0_dp
        call model%set_variational(mean, factor, status)
        if (status_ok(status)) then
            write (error_unit, '(a)') &
                "FAIL [guard] a singular covariance factor accepted"
            failures = failures + 1
        end if
        call model%elbo(x, y(1:3), value, status)
        if (status_ok(status)) then
            write (error_unit, '(a)') "FAIL [guard] mismatched targets accepted"
            failures = failures + 1
        end if
    end subroutine test_refusals

end program test_sparse_gp
