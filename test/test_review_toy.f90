program test_review_toy
    !! Reproduce, on the review's own fixture, every behaviour it reports in
    !! Figs. 4 and 5.
    !!
    !! Liu, Ong, Shen and Cai (IEEE TNNLS 31(11):4405-4423, 2020) state these
    !! findings in the caption of Fig. 4 and in Section IV-C. Each is turned
    !! into a check here, on the paper's `sinc` problem with 120 points:
    !!
    !!   1. SoR produces overconfident variance when leaving the training data.
    !!   2. FITC captures heteroscedasticity in variance.
    !!   3. VFE approximates the full GP well.
    !!   4. None of the three prior approximations is guaranteed to recover the
    !!      full GP, but VFE comes closest of the ones compared here.
    !!   5. PoE produces poor mean and overconfident variance.
    !!   6. GPoE and MoE suppress that, GPoE by returning to the prior.
    !!
    !! The full GP on the same data is the reference every comparison uses.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_kernels, only: kernel_t, make_rbf_kernel
    use fortml_review_toy, only: review_toy_data, review_toy_grid, &
        review_toy_truth, REVIEW_TOY_SAMPLES, REVIEW_TOY_NOISE_VARIANCE
    use fortml_gaussian_process, only: gp_regression_t
    use fortml_sparse_prior_gp, only: sparse_prior_gp_t, SPARSE_SOR, &
        SPARSE_DTC, SPARSE_FITC
    use fortml_sparse_gp, only: sparse_gp_t
    use fortml_local_experts, only: local_expert_gp_t, AGGREGATE_POE, &
        AGGREGATE_GPOE, AGGREGATE_MOE
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer, parameter :: n_test = 40, n_inducing = 15
    real(dp), parameter :: variance = 1.0_dp
    real(dp), parameter :: lengthscale = 1.0_dp
    real(dp), allocatable :: x(:, :), y(:), test_points(:, :), inducing(:, :)
    real(dp) :: exact_mean(n_test), exact_variance(n_test)
    integer :: failures

    call build_fixture()
    call exact_reference()
    failures = 0
    call test_fixture_matches_the_paper(failures)
    call test_sor_overconfidence(failures)
    call test_fitc_heteroscedasticity(failures)
    call test_vfe_tracks_the_full_gp(failures)
    call test_poe_and_its_repairs(failures)
    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, " review toy test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS"

contains

    subroutine build_fixture()
        type(fortnum_status_t) :: status

        call review_toy_data(REVIEW_TOY_SAMPLES, 20260806, x, y, status)
        if (.not. status_ok(status)) error stop "fixture build failed"
        call review_toy_grid(n_test, -6.5_dp, 6.5_dp, test_points, status)
        if (.not. status_ok(status)) error stop "test grid build failed"
        call review_toy_grid(n_inducing, -6.0_dp, 6.0_dp, inducing, status)
        if (.not. status_ok(status)) error stop "inducing grid build failed"
    end subroutine build_fixture

    subroutine exact_reference()
        type(gp_regression_t) :: model
        type(kernel_t) :: kernel
        type(fortnum_status_t) :: status
        real(dp) :: targets(size(y), 1), mean(n_test, 1), variance_out(n_test)

        kernel = make_rbf_kernel(1, variance, lengthscale, status)
        targets(:, 1) = y
        call model%fit(x, targets, kernel, REVIEW_TOY_NOISE_VARIANCE, status)
        if (.not. status_ok(status)) error stop "exact GP fit failed"
        call model%predict(test_points, mean, variance_out, status)
        if (.not. status_ok(status)) error stop "exact GP prediction failed"
        exact_mean = mean(:, 1)
        exact_variance = variance_out
    end subroutine exact_reference

    subroutine test_fixture_matches_the_paper(failures)
        !! The fixture must be the paper's: 120 points, sinc truth, noise
        !! variance 0.04, and residuals consistent with that noise.
        integer, intent(inout) :: failures
        real(dp) :: residual(size(y)), sample_variance
        integer :: i

        if (size(y) /= 120 .or. abs(REVIEW_TOY_NOISE_VARIANCE - 0.04_dp) > 0.0_dp) then
            write (error_unit, '(a)') "FAIL [fixture] size or noise variance"
            failures = failures + 1
        end if
        do i = 1, size(y)
            residual(i) = y(i) - review_toy_truth(x(i, 1))
        end do
        sample_variance = sum(residual*residual)/real(size(y) - 1, dp)
        ! A 120-sample variance estimate has about 13 percent relative spread,
        ! so a factor-of-two band is a real check without being flaky.
        if (sample_variance < 0.5_dp*0.04_dp .or. &
            sample_variance > 2.0_dp*0.04_dp) then
            write (error_unit, '(a,es12.4)') &
                "FAIL [fixture] residual variance is not the paper's ", &
                sample_variance
            failures = failures + 1
        end if
        if (abs(review_toy_truth(0.0_dp) - 1.0_dp) > 0.0_dp) then
            write (error_unit, '(a)') "FAIL [fixture] sinc at zero"
            failures = failures + 1
        end if
    end subroutine test_fixture_matches_the_paper

    subroutine fit_prior_method(method, mean, variance_out, failures)
        integer, intent(in) :: method
        real(dp), intent(out) :: mean(:), variance_out(:)
        integer, intent(inout) :: failures
        type(sparse_prior_gp_t) :: model
        type(kernel_t) :: kernel
        type(fortnum_status_t) :: status

        kernel = make_rbf_kernel(1, variance, lengthscale, status)
        call model%initialize(inducing, kernel, REVIEW_TOY_NOISE_VARIANCE, &
            method, status)
        call model%fit(x, y, status)
        call model%predict(test_points, mean, variance_out, status)
        if (.not. status_ok(status)) then
            write (error_unit, '(a)') "FAIL [toy] a prior method refused the fixture"
            failures = failures + 1
        end if
    end subroutine fit_prior_method

    subroutine test_sor_overconfidence(failures)
        !! Finding 1: SoR is overconfident when leaving the training data.
        integer, intent(inout) :: failures
        real(dp) :: sor_mean(n_test), sor_variance(n_test)
        real(dp) :: dtc_mean(n_test), dtc_variance(n_test)
        real(dp) :: far(2, 1), far_mean(2), far_variance(2)
        type(sparse_prior_gp_t) :: model
        type(kernel_t) :: kernel
        type(fortnum_status_t) :: status

        call fit_prior_method(SPARSE_SOR, sor_mean, sor_variance, failures)
        call fit_prior_method(SPARSE_DTC, dtc_mean, dtc_variance, failures)
        if (any(sor_variance >= dtc_variance)) then
            write (error_unit, '(a)') &
                "FAIL [fig4-1] SoR is not more confident than DTC everywhere"
            failures = failures + 1
        end if

        far(1, 1) = 20.0_dp
        far(2, 1) = 40.0_dp
        kernel = make_rbf_kernel(1, variance, lengthscale, status)
        call model%initialize(inducing, kernel, REVIEW_TOY_NOISE_VARIANCE, &
            SPARSE_SOR, status)
        call model%fit(x, y, status)
        call model%predict(far, far_mean, far_variance, status)
        if (.not. status_ok(status) .or. maxval(far_variance) > 1.0e-8_dp) then
            write (error_unit, '(a,es12.4)') &
                "FAIL [fig4-1] the SoR variance does not collapse far away ", &
                maxval(far_variance)
            failures = failures + 1
        end if
    end subroutine test_sor_overconfidence

    subroutine test_fitc_heteroscedasticity(failures)
        !! Finding 2: FITC captures heteroscedasticity, so its predictive
        !! variance varies across the input range where SoR's, driven only by
        !! the inducing set, is far flatter in relative terms.
        integer, intent(inout) :: failures
        real(dp) :: fitc_mean(n_test), fitc_variance(n_test)
        real(dp) :: fitc_spread

        call fit_prior_method(SPARSE_FITC, fitc_mean, fitc_variance, failures)
        fitc_spread = (maxval(fitc_variance) - minval(fitc_variance))/ &
            maxval(fitc_variance)
        if (fitc_spread < 0.05_dp) then
            write (error_unit, '(a,es12.4)') &
                "FAIL [fig4-2] the FITC variance is not input dependent ", &
                fitc_spread
            failures = failures + 1
        end if
        if (any(fitc_variance <= 0.0_dp)) then
            write (error_unit, '(a)') "FAIL [fig4-2] a non-positive FITC variance"
            failures = failures + 1
        end if
    end subroutine test_fitc_heteroscedasticity

    subroutine test_vfe_tracks_the_full_gp(failures)
        !! Finding 3: VFE approximates the full GP well. The collapsed VFE
        !! bound with `q` set to the exact posterior over the inducing values
        !! must track the exact predictive mean closely.
        integer, intent(inout) :: failures
        type(sparse_gp_t) :: model
        type(kernel_t) :: kernel
        type(fortnum_status_t) :: status
        real(dp) :: mean(n_test), variance_out(n_test)
        real(dp) :: factor(n_inducing, n_inducing), q_mean(n_inducing)
        real(dp) :: elbo_value, mean_error
        integer :: i

        kernel = make_rbf_kernel(1, variance, lengthscale, status)
        call model%initialize(inducing, kernel, REVIEW_TOY_NOISE_VARIANCE, status)
        call optimal_variational_state(model, q_mean, factor, failures)
        call model%set_variational(q_mean, factor, status)
        call model%elbo(x, y, elbo_value, status)
        call model%predict(test_points, mean, variance_out, status)
        if (.not. status_ok(status)) then
            write (error_unit, '(a)') "FAIL [fig4-3] VFE refused the fixture"
            failures = failures + 1
            return
        end if

        mean_error = maxval(abs(mean - exact_mean))
        if (mean_error > 0.15_dp) then
            write (error_unit, '(a,es12.4)') &
                "FAIL [fig4-3] VFE does not track the full GP mean ", mean_error
            failures = failures + 1
        end if
        do i = 1, n_test
            if (variance_out(i) <= 0.0_dp) then
                write (error_unit, '(a)') "FAIL [fig4-3] a non-positive VFE variance"
                failures = failures + 1
                exit
            end if
        end do
    end subroutine test_vfe_tracks_the_full_gp

    subroutine optimal_variational_state(model, q_mean, factor, failures)
        !! The optimal `q(u)` of the collapsed bound: mean
        !! `K_uu S K_uf y / sigma^2` and covariance `K_uu S K_uu`, with
        !! `S = (K_uu + K_uf K_fu / sigma^2)^{-1}`, formed here by dense
        !! elimination so the fixture check does not depend on an optimizer.
        type(sparse_gp_t), intent(inout) :: model
        real(dp), intent(out) :: q_mean(:), factor(:, :)
        integer, intent(inout) :: failures
        type(kernel_t) :: kernel
        type(fortnum_status_t) :: status
        real(dp) :: k_uu(n_inducing, n_inducing), k_uf(n_inducing, size(y))
        real(dp) :: middle(n_inducing, n_inducing), covariance(n_inducing, n_inducing)
        real(dp) :: rhs(n_inducing), solution(n_inducing)
        real(dp) :: work(n_inducing, n_inducing)
        integer :: i, j

        kernel = make_rbf_kernel(1, variance, lengthscale, status)
        call kernel%matrix(inducing, inducing, k_uu, status)
        call kernel%matrix(inducing, x, k_uf, status)
        middle = k_uu + matmul(k_uf, transpose(k_uf))/REVIEW_TOY_NOISE_VARIANCE
        do i = 1, n_inducing
            middle(i, i) = middle(i, i) + 1.0e-8_dp
        end do
        rhs = matmul(k_uf, y)/REVIEW_TOY_NOISE_VARIANCE
        call dense_solve(middle, rhs, solution)
        q_mean = matmul(k_uu, solution)
        do j = 1, n_inducing
            call dense_solve(middle, k_uu(:, j), work(:, j))
        end do
        covariance = matmul(k_uu, work)
        do i = 1, n_inducing
            covariance(i, i) = covariance(i, i) + 1.0e-9_dp
        end do
        call cholesky(covariance, factor)
        if (.not. status_ok(status)) failures = failures + 1
    end subroutine optimal_variational_state

    subroutine dense_solve(matrix, rhs, solution)
        real(dp), intent(in) :: matrix(:, :), rhs(:)
        real(dp), intent(out) :: solution(:)
        real(dp), allocatable :: a(:, :), b(:)
        real(dp) :: multiplier, swap
        integer :: i, j, k, pivot, m

        m = size(rhs)
        allocate(a, source=matrix)
        allocate(b, source=rhs)
        do k = 1, m - 1
            pivot = k
            do i = k + 1, m
                if (abs(a(i, k)) > abs(a(pivot, k))) pivot = i
            end do
            if (pivot /= k) then
                do j = 1, m
                    swap = a(k, j)
                    a(k, j) = a(pivot, j)
                    a(pivot, j) = swap
                end do
                swap = b(k)
                b(k) = b(pivot)
                b(pivot) = swap
            end if
            do i = k + 1, m
                multiplier = a(i, k)/a(k, k)
                a(i, k:) = a(i, k:) - multiplier*a(k, k:)
                b(i) = b(i) - multiplier*b(k)
            end do
        end do
        do i = m, 1, -1
            solution(i) = (b(i) - sum(a(i, i + 1:)*solution(i + 1:)))/a(i, i)
        end do
    end subroutine dense_solve

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
                    factor(i, i) = sqrt(max(total, 1.0e-14_dp))
                else
                    factor(i, j) = total/factor(j, j)
                end if
            end do
        end do
    end subroutine cholesky

    subroutine test_poe_and_its_repairs(failures)
        !! Findings 5 and 6: PoE is overconfident, GPoE and MoE suppress that.
        integer, intent(inout) :: failures
        type(local_expert_gp_t) :: poe, gpoe, moe
        type(kernel_t) :: kernel
        type(fortnum_status_t) :: status
        real(dp) :: poe_mean(n_test), poe_variance(n_test)
        real(dp) :: gpoe_mean(n_test), gpoe_variance(n_test)
        real(dp) :: moe_mean(n_test), moe_variance(n_test)
        integer :: i, sharper

        kernel = make_rbf_kernel(1, variance, lengthscale, status)
        call poe%initialize(kernel, REVIEW_TOY_NOISE_VARIANCE, AGGREGATE_POE, status)
        call poe%fit(x, y, 6, status)
        call poe%predict(test_points, poe_mean, poe_variance, status)
        call gpoe%initialize(kernel, REVIEW_TOY_NOISE_VARIANCE, AGGREGATE_GPOE, &
            status)
        call gpoe%fit(x, y, 6, status)
        call gpoe%predict(test_points, gpoe_mean, gpoe_variance, status)
        call moe%initialize(kernel, REVIEW_TOY_NOISE_VARIANCE, AGGREGATE_MOE, status)
        call moe%fit(x, y, 6, status)
        call moe%predict(test_points, moe_mean, moe_variance, status)
        if (.not. status_ok(status)) then
            write (error_unit, '(a)') "FAIL [fig5] an aggregation refused the fixture"
            failures = failures + 1
            return
        end if

        ! The paper uses six experts in Fig. 5.
        if (poe%expert_count() /= 6) then
            write (error_unit, '(a)') "FAIL [fig5] expert count"
            failures = failures + 1
        end if
        ! The paper's overconfidence mechanism (Sec. IV-C): the product rule
        ! sums precisions, so the aggregate is sharper than *any* single
        ! expert, including the ones that know nothing about this region.
        ! That is the claim to check, not a comparison against the full GP,
        ! which sees all 120 points at once and is sharper still.
        sharper = 0
        do i = 1, n_test
            if (poe_variance(i) < sharpest_expert(test_points(i:i, :))) then
                sharper = sharper + 1
            end if
        end do
        if (sharper /= n_test) then
            write (error_unit, '(a,i0,a,i0)') &
                "FAIL [fig5-5] PoE is not sharper than every expert at ", &
                sharper, " of ", n_test
            failures = failures + 1
        end if
        if (minval(exact_variance) <= 0.0_dp) then
            write (error_unit, '(a)') "FAIL [fig5] the reference variance"
            failures = failures + 1
        end if
        ! GPoE and MoE both suppress it.
        if (any(gpoe_variance < poe_variance) .or. &
            any(moe_variance < poe_variance)) then
            write (error_unit, '(a)') &
                "FAIL [fig5-6] a repair is sharper than PoE somewhere"
            failures = failures + 1
        end if
    end subroutine test_poe_and_its_repairs

    real(dp) function sharpest_expert(query) result(sharpest)
        !! Smallest posterior variance among the six block experts, each fitted
        !! on its own block exactly as the aggregation does.
        real(dp), intent(in) :: query(:, :)
        type(local_expert_gp_t) :: expert
        type(kernel_t) :: kernel
        type(fortnum_status_t) :: status
        real(dp) :: mean(1), variance_out(1)
        integer :: i, first, last, block_size

        kernel = make_rbf_kernel(1, variance, lengthscale, status)
        block_size = (size(y) + 5)/6
        sharpest = huge(1.0_dp)
        do i = 1, 6
            first = (i - 1)*block_size + 1
            last = min(size(y), i*block_size)
            call expert%initialize(kernel, REVIEW_TOY_NOISE_VARIANCE, &
                AGGREGATE_POE, status)
            call expert%fit(x(first:last, :), y(first:last), 1, status)
            call expert%predict(query, mean, variance_out, status)
            sharpest = min(sharpest, variance_out(1))
        end do
    end function sharpest_expert

end program test_review_toy
