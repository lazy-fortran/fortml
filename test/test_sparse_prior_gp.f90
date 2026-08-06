program test_sparse_prior_gp
    !! Oracles for the SoR, DTC, FITC and PITC prior approximations.
    !!
    !! Two kinds of check. First, exact identities: with the inducing set equal
    !! to the training set, `Q_nn = K_nn`, so DTC, FITC and PITC all collapse
    !! to the full GP, and their predictive mean, variance and log marginal
    !! likelihood must equal a dense reference computed here. Second, the
    !! qualitative behaviours the review reports in its Fig. 4: SoR shares the
    !! DTC mean but has a strictly smaller predictive variance, and that
    !! variance collapses far from the inducing set instead of returning to the
    !! prior, while DTC returns to the prior there.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_kernels, only: kernel_t, make_rbf_kernel
    use fortml_sparse_prior_gp, only: sparse_prior_gp_t, sparse_prior_method_name, &
        SPARSE_SOR, SPARSE_DTC, SPARSE_FITC, SPARSE_PITC
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer, parameter :: n = 8, d = 1
    real(dp), parameter :: variance = 1.2_dp
    real(dp), parameter :: lengthscale = 0.8_dp
    real(dp), parameter :: noise = 0.25_dp
    real(dp) :: x(n, d), y(n)
    integer :: failures

    call build_data(x, y)
    failures = 0
    call test_collapse_to_full_gp(failures)
    call test_sor_is_overconfident(failures)
    call test_dtc_returns_to_the_prior(failures)
    call test_method_names_and_refusals(failures)
    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, " sparse prior test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS"

contains

    subroutine build_data(inputs, targets)
        real(dp), intent(out) :: inputs(:, :), targets(:)
        integer :: i

        do i = 1, n
            inputs(i, 1) = -1.4_dp + 0.4_dp*real(i, dp)
            targets(i) = sin(2.0_dp*inputs(i, 1)) + 0.05_dp*cos(real(i, dp))
        end do
    end subroutine build_data

    real(dp) function rbf(a, b) result(value)
        real(dp), intent(in) :: a, b

        value = variance*exp(-0.5_dp*(a - b)**2/(lengthscale*lengthscale))
    end function rbf

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

    subroutine exact_reference(query, mean, variance_out, log_marginal)
        !! Dense exact GP: the oracle every collapsed case must reproduce.
        real(dp), intent(in) :: query(:, :)
        real(dp), intent(out) :: mean(:), variance_out(:), log_marginal
        real(dp) :: matrix(n, n), factor(n, n), alpha(n), cross(n), column(n)
        integer :: i, j

        do j = 1, n
            do i = 1, n
                matrix(i, j) = rbf(x(i, 1), x(j, 1))
            end do
            matrix(j, j) = matrix(j, j) + noise
        end do
        call cholesky(matrix, factor)
        call cholesky_solve(factor, y, alpha)
        log_marginal = -0.5_dp*sum(y*alpha) - &
            0.5_dp*real(n, dp)*log(8.0_dp*atan(1.0_dp))
        do i = 1, n
            log_marginal = log_marginal - log(factor(i, i))
        end do
        do i = 1, size(query, 1)
            do j = 1, n
                cross(j) = rbf(query(i, 1), x(j, 1))
            end do
            mean(i) = sum(cross*alpha)
            call cholesky_solve(factor, cross, column)
            variance_out(i) = rbf(query(i, 1), query(i, 1)) - sum(cross*column)
        end do
    end subroutine exact_reference

    subroutine test_collapse_to_full_gp(failures)
        integer, intent(inout) :: failures
        type(sparse_prior_gp_t) :: model
        type(kernel_t) :: kernel
        type(fortnum_status_t) :: status
        real(dp) :: query(5, d), mean(5), variance_out(5)
        real(dp) :: reference_mean(5), reference_variance(5), reference_lml
        real(dp) :: value
        integer :: methods(3), index, i

        do i = 1, 5
            query(i, 1) = -1.1_dp + 0.55_dp*real(i, dp)
        end do
        call exact_reference(query, reference_mean, reference_variance, &
            reference_lml)

        methods = [SPARSE_DTC, SPARSE_FITC, SPARSE_PITC]
        kernel = make_rbf_kernel(d, variance, lengthscale, status)
        do index = 1, 3
            if (methods(index) == SPARSE_PITC) then
                call model%initialize(x, kernel, noise, methods(index), status, &
                    block_size=3)
            else
                call model%initialize(x, kernel, noise, methods(index), status)
            end if
            call model%fit(x, y, status)
            call model%predict(query, mean, variance_out, status)
            call model%log_marginal_likelihood(value, status)
            if (.not. status_ok(status) .or. &
                maxval(abs(mean - reference_mean)) > 1.0e-7_dp .or. &
                maxval(abs(variance_out - reference_variance)) > 1.0e-7_dp .or. &
                abs(value - reference_lml) > 1.0e-6_dp) then
                write (error_unit, '(a,a,3es12.4)') "FAIL [collapse] ", &
                    sparse_prior_method_name(methods(index)), &
                    maxval(abs(mean - reference_mean)), &
                    maxval(abs(variance_out - reference_variance)), &
                    abs(value - reference_lml)
                failures = failures + 1
            end if
        end do
    end subroutine test_collapse_to_full_gp

    subroutine test_sor_is_overconfident(failures)
        !! Paper Fig. 4: SoR shares the DTC mean but drops the prior term, so
        !! its variance is strictly smaller everywhere and collapses far away.
        integer, intent(inout) :: failures
        type(sparse_prior_gp_t) :: sor, dtc
        type(kernel_t) :: kernel
        type(fortnum_status_t) :: status
        real(dp) :: inducing(3, d), query(4, d)
        real(dp) :: sor_mean(4), sor_variance(4), dtc_mean(4), dtc_variance(4)
        integer :: i

        do i = 1, 3
            inducing(i, 1) = -0.8_dp + 0.8_dp*real(i - 1, dp)
        end do
        query(1, 1) = -0.5_dp
        query(2, 1) = 0.3_dp
        query(3, 1) = 6.0_dp
        query(4, 1) = 12.0_dp

        kernel = make_rbf_kernel(d, variance, lengthscale, status)
        call sor%initialize(inducing, kernel, noise, SPARSE_SOR, status)
        call sor%fit(x, y, status)
        call sor%predict(query, sor_mean, sor_variance, status)
        call dtc%initialize(inducing, kernel, noise, SPARSE_DTC, status)
        call dtc%fit(x, y, status)
        call dtc%predict(query, dtc_mean, dtc_variance, status)

        if (.not. status_ok(status) .or. &
            maxval(abs(sor_mean - dtc_mean)) > 1.0e-12_dp) then
            write (error_unit, '(a)') "FAIL [sor] SoR and DTC means differ"
            failures = failures + 1
        end if
        if (any(sor_variance >= dtc_variance)) then
            write (error_unit, '(a)') &
                "FAIL [sor] SoR variance is not below DTC everywhere"
            failures = failures + 1
        end if
        ! Far from the inducing set the degenerate variance goes to zero.
        if (sor_variance(4) > 1.0e-6_dp .or. &
            sor_variance(4) >= sor_variance(1)) then
            write (error_unit, '(a,es12.4)') &
                "FAIL [sor] the degenerate variance does not collapse ", &
                sor_variance(4)
            failures = failures + 1
        end if
    end subroutine test_sor_is_overconfident

    subroutine test_dtc_returns_to_the_prior(failures)
        !! Paper Fig. 4: away from the inducing points the DTC variance grows
        !! back to the prior variance.
        integer, intent(inout) :: failures
        type(sparse_prior_gp_t) :: model
        type(kernel_t) :: kernel
        type(fortnum_status_t) :: status
        real(dp) :: inducing(3, d), query(2, d), mean(2), variance_out(2)
        integer :: i

        do i = 1, 3
            inducing(i, 1) = -0.8_dp + 0.8_dp*real(i - 1, dp)
        end do
        query(1, 1) = 0.0_dp
        query(2, 1) = 14.0_dp
        kernel = make_rbf_kernel(d, variance, lengthscale, status)
        call model%initialize(inducing, kernel, noise, SPARSE_DTC, status)
        call model%fit(x, y, status)
        call model%predict(query, mean, variance_out, status)

        if (.not. status_ok(status) .or. &
            abs(variance_out(2) - variance) > 1.0e-8_dp .or. &
            variance_out(1) >= variance) then
            write (error_unit, '(a,2es12.4)') &
                "FAIL [dtc] variance does not return to the prior ", &
                variance_out(1), variance_out(2)
            failures = failures + 1
        end if
    end subroutine test_dtc_returns_to_the_prior

    subroutine test_method_names_and_refusals(failures)
        integer, intent(inout) :: failures
        type(sparse_prior_gp_t) :: model
        type(kernel_t) :: kernel
        type(fortnum_status_t) :: status
        real(dp) :: mean(2), variance_out(2), query(2, d)

        kernel = make_rbf_kernel(d, variance, lengthscale, status)
        if (sparse_prior_method_name(SPARSE_FITC) /= "FITC" .or. &
            sparse_prior_method_name(99) /= "unknown") then
            write (error_unit, '(a)') "FAIL [name] method names"
            failures = failures + 1
        end if
        call model%initialize(x, kernel, noise, SPARSE_PITC, status)
        if (status_ok(status)) then
            write (error_unit, '(a)') "FAIL [guard] PITC without a block size"
            failures = failures + 1
        end if
        call model%initialize(x, kernel, -1.0_dp, SPARSE_DTC, status)
        if (status_ok(status)) then
            write (error_unit, '(a)') "FAIL [guard] negative noise accepted"
            failures = failures + 1
        end if
        call model%initialize(x, kernel, noise, 42, status)
        if (status_ok(status)) then
            write (error_unit, '(a)') "FAIL [guard] an unknown method accepted"
            failures = failures + 1
        end if
        call model%initialize(x, kernel, noise, SPARSE_DTC, status)
        query = 0.0_dp
        call model%predict(query, mean, variance_out, status)
        if (status_ok(status)) then
            write (error_unit, '(a)') "FAIL [guard] predict before fit accepted"
            failures = failures + 1
        end if
    end subroutine test_method_names_and_refusals

end program test_sparse_prior_gp
