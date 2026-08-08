program test_heteroskedastic_gp
    !! Heteroskedastic GP regression.
    !!
    !! Oracles:
    !!
    !!   * **constant noise reduces to an ordinary GP exactly.** With every
    !!     observation carrying the same variance the model must agree with
    !!     `gp_regression_t` to rounding, in both mean and variance. That is the
    !!     anchor: a heteroskedastic model that did not contain the homoskedastic
    !!     one as a special case would be a different model, not a generalization;
    !!   * **the posterior follows the noise.** Where measurements are precise
    !!     the posterior must track them closely and be confident; where they are
    !!     noisy it must not. Measured against a constructed dataset whose two
    !!     halves differ only in noise;
    !!   * **the log-noise interpolation is positive and reverts to the mean.**
    !!     A process on the variance directly would put mass on negatives; the
    !!     log construction cannot, and far from data it must return the mean
    !!     noise level rather than one.

    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_DOMAIN_ERROR
    use fortml_kernels, only: kernel_t, make_rbf_kernel
    use fortml_gaussian_process, only: gp_regression_t
    use fortml_heteroskedastic_gp, only: heteroskedastic_gp_t
    implicit none

    integer :: failures

    failures = 0
    call test_constant_noise_matches_a_plain_gp(failures)
    call test_posterior_follows_the_noise(failures)
    call test_noise_interpolation(failures)
    call test_refusals(failures)

    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, " heteroskedastic GP test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS: heteroskedastic GP"

contains

    !! The anchor. A generalization must contain the special case exactly.
    subroutine test_constant_noise_matches_a_plain_gp(failures)
        integer, intent(inout) :: failures
        type(heteroskedastic_gp_t) :: model
        type(gp_regression_t) :: reference
        type(kernel_t) :: signal, noise
        type(fortnum_status_t) :: status
        real(dp) :: x(8, 1), y(8), variances(8), query(5, 1)
        real(dp) :: h_mean(5), h_variance(5)
        real(dp) :: g_mean(5, 1), g_variance(5)
        integer :: k

        do k = 1, 8
            x(k, 1) = -1.4_dp + 0.4_dp*real(k - 1, dp)
            y(k) = sin(1.9_dp*x(k, 1))
            variances(k) = 0.05_dp
        end do
        do k = 1, 5
            query(k, 1) = -1.1_dp + 0.55_dp*real(k - 1, dp)
        end do

        signal = make_rbf_kernel(1, 1.0_dp, 0.7_dp, status)
        noise = make_rbf_kernel(1, 1.0_dp, 1.5_dp, status)
        call model%fit(x, y, variances, signal, noise, status)
        call check(status_ok(status), "the heteroskedastic model fits", failures)
        call model%predict(query, h_mean, h_variance, status)
        call check(status_ok(status), "it predicts", failures)

        signal = make_rbf_kernel(1, 1.0_dp, 0.7_dp, status)
        call reference%fit(x, reshape(y, [8, 1]), signal, 0.05_dp, status)
        call reference%predict(query, g_mean, g_variance, status)

        call check(maxval(abs(h_mean - g_mean(:, 1))) < 1.0e-12_dp, &
            "constant noise reproduces the plain GP's mean exactly", failures)
        call check(maxval(abs(h_variance - g_variance)) < 1.0e-12_dp, &
            "constant noise reproduces the plain GP's variance exactly", failures)
    end subroutine test_constant_noise_matches_a_plain_gp

    !! Two halves of a dataset differing only in measurement noise. The model
    !! must be confident where the data are good and not where they are not —
    !! which a single-noise GP cannot do, since its posterior variance depends
    !! only on the inputs.
    subroutine test_posterior_follows_the_noise(failures)
        integer, intent(inout) :: failures
        type(heteroskedastic_gp_t) :: model
        type(kernel_t) :: signal, noise
        type(fortnum_status_t) :: status
        real(dp) :: x(10, 1), y(10), variances(10)
        real(dp) :: quiet_query(1, 1), loud_query(1, 1)
        real(dp) :: quiet_mean(1), quiet_variance(1)
        real(dp) :: loud_mean(1), loud_variance(1)
        integer :: k

        ! Left half measured precisely, right half poorly.
        do k = 1, 10
            x(k, 1) = -2.0_dp + 0.4_dp*real(k - 1, dp)
            y(k) = 0.5_dp*x(k, 1)
            if (x(k, 1) < 0.0_dp) then
                variances(k) = 1.0e-4_dp
            else
                variances(k) = 1.0_dp
            end if
        end do

        signal = make_rbf_kernel(1, 1.0_dp, 0.6_dp, status)
        noise = make_rbf_kernel(1, 1.0_dp, 1.2_dp, status)
        call model%fit(x, y, variances, signal, noise, status)
        call check(status_ok(status), "the split-noise model fits", failures)

        quiet_query(1, 1) = -1.4_dp
        loud_query(1, 1) = 1.4_dp
        call model%predict(quiet_query, quiet_mean, quiet_variance, status)
        call model%predict(loud_query, loud_mean, loud_variance, status)

        call check(quiet_variance(1) < loud_variance(1), &
            "the posterior is tighter where the measurements are precise", &
            failures)
        ! And it must actually track the precise data, not merely be confident.
        call check(abs(quiet_mean(1) - 0.5_dp*quiet_query(1, 1)) < 0.05_dp, &
            "the posterior tracks the precisely measured half", failures)
    end subroutine test_posterior_follows_the_noise

    subroutine test_noise_interpolation(failures)
        integer, intent(inout) :: failures
        type(heteroskedastic_gp_t) :: model
        type(kernel_t) :: signal, noise
        type(fortnum_status_t) :: status
        real(dp) :: x(10, 1), y(10), variances(10)
        real(dp) :: probe(21, 1), estimated(21)
        real(dp) :: distant(1, 1), far_noise(1)
        real(dp) :: geometric_mean
        integer :: k

        do k = 1, 10
            x(k, 1) = -2.0_dp + 0.4_dp*real(k - 1, dp)
            y(k) = 0.5_dp*x(k, 1)
            if (x(k, 1) < 0.0_dp) then
                variances(k) = 1.0e-4_dp
            else
                variances(k) = 1.0_dp
            end if
        end do
        signal = make_rbf_kernel(1, 1.0_dp, 0.6_dp, status)
        noise = make_rbf_kernel(1, 1.0_dp, 1.2_dp, status)
        call model%fit(x, y, variances, signal, noise, status)

        do k = 1, 21
            probe(k, 1) = -2.0_dp + 0.2_dp*real(k - 1, dp)
        end do
        call model%noise_at(probe, estimated, status)
        call check(status_ok(status), "the noise interpolation evaluates", failures)

        ! Positive by construction, not by clipping: this is why the latent
        ! process lives on the log scale.
        call check(all(estimated > 0.0_dp), &
            "interpolated noise is positive everywhere", failures)
        call check(estimated(1) < estimated(21), &
            "interpolated noise rises from the precise half to the poor one", &
            failures)

        ! Far from the data the log-noise process reverts to the *mean* log
        ! noise, so the variance returns to the geometric mean of the supplied
        ! ones — not to unit variance, which nobody claimed.
        distant(1, 1) = 60.0_dp
        call model%noise_at(distant, far_noise, status)
        geometric_mean = exp(sum(log(variances))/real(size(variances), dp))
        call check(abs(far_noise(1) - geometric_mean) &
            < 1.0e-6_dp*geometric_mean, &
            "far from data the noise reverts to the mean level, not to one", &
            failures)
    end subroutine test_noise_interpolation

    subroutine test_refusals(failures)
        integer, intent(inout) :: failures
        type(heteroskedastic_gp_t) :: model, unfitted
        type(kernel_t) :: signal, noise
        type(fortnum_status_t) :: status
        real(dp) :: x(4, 1), y(4), variances(4), query(2, 1)
        real(dp) :: mean(2), variance(2)
        integer :: k

        do k = 1, 4
            x(k, 1) = real(k, dp)
            y(k) = real(k, dp)
            variances(k) = 0.1_dp
        end do
        signal = make_rbf_kernel(1, 1.0_dp, 1.0_dp, status)
        noise = make_rbf_kernel(1, 1.0_dp, 1.0_dp, status)

        ! A zero variance asserts a noiseless measurement, whose log does not
        ! exist. Refused rather than floored, since flooring would invent a
        ! precision the caller did not claim.
        variances(2) = 0.0_dp
        call model%fit(x, y, variances, signal, noise, status)
        call check(status%code == FORTNUM_DOMAIN_ERROR, &
            "a zero observation variance is refused", failures)

        variances(2) = 0.1_dp
        call model%fit(x, y(1:3), variances, signal, noise, status)
        call check(status%code == FORTNUM_DOMAIN_ERROR, &
            "mismatched targets are refused", failures)

        query = 0.0_dp
        call unfitted%predict(query, mean, variance, status)
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

end program test_heteroskedastic_gp
