program test_robust_gp
    !! GP regression under Poisson and Student-t observation models.
    !!
    !! Oracles:
    !!
    !!   * **the Poisson mode satisfies its own stationarity condition.** At the
    !!     optimum `K^-1 f = grad log p(y|f)`, and both sides are computable
    !!     from the fitted state, so the check does not depend on the iteration
    !!     that produced it;
    !!   * **Poisson rates stay positive and track counts.** The log-rate
    !!     construction guarantees the first by shape, so the test measures the
    !!     second;
    !!   * **the Student-t fit resists an outlier.** One badly corrupted point
    !!     is added to a clean dataset, and the robust fit must move far less
    !!     than a Gaussian one. That contrast is the entire reason the
    !!     likelihood exists, and it is measured rather than asserted.

    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_DOMAIN_ERROR
    use fortml_kernels, only: kernel_t, make_rbf_kernel
    use fortml_gaussian_process, only: gp_regression_t
    use fortml_robust_gp, only: robust_gp_t, FORTML_LIKELIHOOD_POISSON, &
        FORTML_LIKELIHOOD_STUDENT_T
    implicit none

    integer :: failures

    failures = 0
    call test_poisson_stationarity(failures)
    call test_poisson_rates_track_counts(failures)
    call test_student_t_resists_an_outlier(failures)
    call test_refusals(failures)

    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, " robust GP test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS: robust GP"

contains

    !! At the Laplace mode, `alpha = K^-1 f` must equal `grad log p(y|f)`.
    !! Checking the condition the iteration was solving, rather than the
    !! iteration's own output, makes this independent of how it got there.
    subroutine test_poisson_stationarity(failures)
        integer, intent(inout) :: failures
        type(robust_gp_t) :: model
        type(kernel_t) :: kernel
        type(fortnum_status_t) :: status
        real(dp) :: x(8, 1), y(8), gradient(8)
        integer :: k
        logical :: stationary

        do k = 1, 8
            x(k, 1) = -1.4_dp + 0.4_dp*real(k - 1, dp)
            ! Counts rising with x, so the log rate is genuinely non-constant.
            y(k) = real(max(0, nint(3.0_dp*exp(0.6_dp*x(k, 1)))), dp)
        end do

        kernel = make_rbf_kernel(1, 1.0_dp, 0.9_dp, status)
        call model%fit(x, y, kernel, FORTML_LIKELIHOOD_POISSON, status)
        call check(status_ok(status), "the Poisson model fits", failures)
        call check(model%converged, "the Poisson iteration converges", failures)

        ! grad log p = y - exp(f), computed here from the reported mode.
        stationary = .true.
        do k = 1, 8
            gradient(k) = y(k) - exp(model%mode(k))
            if (abs(gradient(k) - model%alpha(k)) > 1.0e-6_dp) stationary = .false.
        end do
        call check(stationary, &
            "the fitted alpha equals the log-likelihood gradient at the mode", &
            failures)
    end subroutine test_poisson_stationarity

    subroutine test_poisson_rates_track_counts(failures)
        integer, intent(inout) :: failures
        type(robust_gp_t) :: model
        type(kernel_t) :: kernel
        type(fortnum_status_t) :: status
        real(dp) :: x(8, 1), y(8), probe(8, 1), response(8)
        integer :: k
        logical :: tracks

        do k = 1, 8
            x(k, 1) = -1.4_dp + 0.4_dp*real(k - 1, dp)
            y(k) = real(max(0, nint(3.0_dp*exp(0.6_dp*x(k, 1)))), dp)
            probe(k, 1) = x(k, 1)
        end do

        kernel = make_rbf_kernel(1, 1.0_dp, 0.9_dp, status)
        call model%fit(x, y, kernel, FORTML_LIKELIHOOD_POISSON, status)
        call model%predict_response(probe, response, status)
        call check(status_ok(status), "the Poisson response evaluates", failures)

        ! Positive by construction: the latent is a log rate, so no clipping is
        ! involved and this cannot fail for a finite mode.
        call check(all(response > 0.0_dp), &
            "predicted rates are positive without clipping", failures)
        ! And rising, following the counts.
        tracks = .true.
        do k = 2, 8
            if (response(k) < response(k - 1)) tracks = .false.
        end do
        call check(tracks, "predicted rates rise with the observed counts", &
            failures)
        call check(response(8) > response(1), &
            "the rate spans the range the counts do", failures)
    end subroutine test_poisson_rates_track_counts

    !! The contrast that justifies the likelihood. One corrupted observation,
    !! and a Gaussian fit chases it while a Student-t fit does not.
    subroutine test_student_t_resists_an_outlier(failures)
        integer, intent(inout) :: failures
        type(robust_gp_t) :: robust
        type(gp_regression_t) :: gaussian
        type(kernel_t) :: kernel
        type(fortnum_status_t) :: status
        real(dp) :: x(11, 1), clean(11), spoiled(11), probe(1, 1)
        real(dp) :: robust_mean(1), robust_variance(1)
        real(dp) :: gaussian_mean(1, 1), gaussian_variance(1)
        real(dp) :: truth, robust_error, gaussian_error
        integer :: k, spoiled_index

        do k = 1, 11
            x(k, 1) = -2.0_dp + 0.4_dp*real(k - 1, dp)
            clean(k) = 0.5_dp*x(k, 1)
            spoiled(k) = clean(k)
        end do
        ! One observation corrupted by a large amount, at the middle of the
        ! domain where it does the most damage.
        spoiled_index = 6
        spoiled(spoiled_index) = clean(spoiled_index) + 25.0_dp
        probe(1, 1) = x(spoiled_index, 1)
        truth = clean(spoiled_index)

        kernel = make_rbf_kernel(1, 4.0_dp, 0.8_dp, status)
        call robust%fit(x, spoiled, kernel, FORTML_LIKELIHOOD_STUDENT_T, status, &
            nu=3.0_dp, scale=0.2_dp)
        call check(status_ok(status), "the Student-t model fits", failures)
        call robust%predict_latent(probe, robust_mean, robust_variance, status)

        kernel = make_rbf_kernel(1, 4.0_dp, 0.8_dp, status)
        call gaussian%fit(x, reshape(spoiled, [11, 1]), kernel, 0.04_dp, status)
        call gaussian%predict(probe, gaussian_mean, gaussian_variance, status)

        robust_error = abs(robust_mean(1) - truth)
        gaussian_error = abs(gaussian_mean(1, 1) - truth)
        call check(robust_error < gaussian_error, &
            "the Student-t fit is pulled less by the outlier than a Gaussian one", &
            failures)
        ! And by a wide margin, not a hair: the whole point is saturation.
        call check(robust_error < 0.5_dp*gaussian_error, &
            "the outlier's influence is genuinely saturated", failures)
    end subroutine test_student_t_resists_an_outlier

    subroutine test_refusals(failures)
        integer, intent(inout) :: failures
        type(robust_gp_t) :: model, unfitted
        type(kernel_t) :: kernel
        type(fortnum_status_t) :: status
        real(dp) :: x(4, 1), y(4), probe(2, 1), mean(2), variance(2)
        integer :: k

        do k = 1, 4
            x(k, 1) = real(k, dp)
            y(k) = real(k, dp)
        end do
        kernel = make_rbf_kernel(1, 1.0_dp, 1.0_dp, status)

        call model%fit(x, [-1.0_dp, 1.0_dp, 2.0_dp, 3.0_dp], kernel, &
            FORTML_LIKELIHOOD_POISSON, status)
        call check(status%code == FORTNUM_DOMAIN_ERROR, &
            "a negative Poisson count is refused", failures)

        call model%fit(x, y, kernel, 99, status)
        call check(status%code == FORTNUM_DOMAIN_ERROR, &
            "an unknown likelihood is refused", failures)

        call model%fit(x, y, kernel, FORTML_LIKELIHOOD_STUDENT_T, status, &
            nu=-1.0_dp)
        call check(status%code == FORTNUM_DOMAIN_ERROR, &
            "a non-positive nu is refused", failures)

        probe = 0.0_dp
        call unfitted%predict_latent(probe, mean, variance, status)
        call check(status%code == FORTNUM_DOMAIN_ERROR, &
            "predicting before fitting is refused", failures)
    end subroutine test_refusals

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: description
        integer, intent(inout) :: failures

        if (.not. condition) then
            write (error_unit, '(a)') "FAIL ["//description//"]"
            failures = failures + 1
        end if
    end subroutine check

end program test_robust_gp
