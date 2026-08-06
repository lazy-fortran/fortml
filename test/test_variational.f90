program test_variational
    !! Oracles for the reusable variational-inference contract.
    !!
    !! The convergence case is a conjugate Bayesian linear model, whose exact
    !! posterior is available in closed form: with prior `N(0, s_p^2 I)`,
    !! Gaussian noise of variance `s_n^2` and design `X`, the posterior is
    !! `N(S X^T y / s_n^2, S)` with `S = (X^T X / s_n^2 + I / s_p^2)^{-1}`.
    !! Maximizing the ELBO over a full-covariance Gaussian family must recover
    !! exactly that distribution, so the optimized variational mean and
    !! covariance are checked against the analytic ones.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_variational, only: gaussian_family_t, vi_elbo, vi_elbo_gradient
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_OK
    use fortopt_adam, only: adam_t
    implicit none

    integer, parameter :: n_features = 2
    integer, parameter :: n_samples = 6
    real(dp), parameter :: prior_variance = 4.0_dp
    real(dp), parameter :: noise_variance = 0.25_dp
    real(dp) :: design(n_samples, n_features), observations(n_samples)
    integer :: failures

    call build_data(design, observations)
    failures = 0
    call test_kl_positivity_and_zero(failures)
    call test_seeded_whitened_draws(failures)
    call test_elbo_decomposition_and_minibatch_scaling(failures)
    call test_gradient_against_finite_difference(failures)
    call test_convergence_to_analytic_posterior(failures)
    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, " variational test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS"

contains

    subroutine build_data(x, y)
        real(dp), intent(out) :: x(:, :), y(:)
        integer :: i

        do i = 1, size(x, 1)
            x(i, 1) = 0.6_dp*sin(real(i, dp)) + 0.2_dp
            x(i, 2) = 0.4_dp*cos(real(2*i, dp)) - 0.3_dp
            y(i) = 0.9_dp*x(i, 1) - 1.4_dp*x(i, 2) + 0.05_dp*sin(real(3*i, dp))
        end do
    end subroutine build_data

    subroutine model_log_likelihood(weights, extra, value, weight_gradient, &
            extra_gradient, status)
        !! Gaussian log likelihood of the linear model at one weight draw.
        real(dp), intent(in) :: weights(:), extra(:)
        real(dp), intent(out) :: value
        real(dp), intent(out) :: weight_gradient(:), extra_gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: residual(n_samples)
        integer :: i

        residual = matmul(design, weights) - observations
        value = -0.5_dp*sum(residual**2)/noise_variance &
            - 0.5_dp*real(n_samples, dp)*log(8.0_dp*atan(1.0_dp)*noise_variance)
        do i = 1, size(weights)
            weight_gradient(i) = -sum(residual*design(:, i))/noise_variance
        end do
        if (size(extra_gradient) > 0) extra_gradient = 0.0_dp
        status%code = FORTNUM_OK
        if (size(extra) > 0) value = value
    end subroutine model_log_likelihood

    subroutine test_kl_positivity_and_zero(failures)
        integer, intent(inout) :: failures
        type(gaussian_family_t) :: family
        type(fortnum_status_t) :: status
        real(dp), allocatable :: lambda(:)
        real(dp) :: value, worst
        integer :: i, n

        call family%initialize(n_features, 16, 20260806, status, &
            prior_variance=prior_variance)
        n = family%parameter_count()
        allocate(lambda(n))
        lambda = family%parameters()
        call family%kl(value, status)
        if (.not. status_ok(status) .or. abs(value) > 1.0e-14_dp) then
            write (error_unit, '(a)') &
                "FAIL [kl] KL is not zero at the prior"
            failures = failures + 1
        end if

        worst = 0.0_dp
        do i = 1, 40
            lambda(1:n_features) = 0.3_dp*sin(real(i, dp))
            lambda(n_features + 1:) = 0.5_dp*cos(real(i, dp)) - 0.2_dp
            call family%set_parameters(lambda, status)
            call family%kl(value, status)
            worst = min(worst, value)
        end do
        if (worst < -1.0e-14_dp) then
            write (error_unit, '(a,es12.4)') "FAIL [kl] negative KL ", worst
            failures = failures + 1
        end if
    end subroutine test_kl_positivity_and_zero

    subroutine test_seeded_whitened_draws(failures)
        integer, intent(inout) :: failures
        type(gaussian_family_t) :: first, second, other
        type(fortnum_status_t) :: status
        real(dp) :: moment(n_features, n_features), mean(n_features)
        integer :: i, n_mc

        n_mc = 64
        call first%initialize(n_features, n_mc, 4242, status)
        call second%initialize(n_features, n_mc, 4242, status)
        call other%initialize(n_features, n_mc, 99, status)
        if (maxval(abs(first%noise - second%noise)) > 0.0_dp) then
            write (error_unit, '(a)') "FAIL [mc] the same seed differs"
            failures = failures + 1
        end if
        if (maxval(abs(first%noise - other%noise)) < 1.0e-8_dp) then
            write (error_unit, '(a)') "FAIL [mc] different seeds agree"
            failures = failures + 1
        end if

        do i = 1, n_features
            mean(i) = sum(first%noise(i, :))/real(n_mc, dp)
        end do
        moment = matmul(first%noise, transpose(first%noise))/real(n_mc, dp)
        do i = 1, n_features
            moment(i, i) = moment(i, i) - 1.0_dp
        end do
        if (maxval(abs(mean)) > 1.0e-12_dp .or. &
            maxval(abs(moment)) > 1.0e-12_dp) then
            write (error_unit, '(a)') &
                "FAIL [mc] the draw table is not centred and whitened"
            failures = failures + 1
        end if
    end subroutine test_seeded_whitened_draws

    subroutine test_elbo_decomposition_and_minibatch_scaling(failures)
        integer, intent(inout) :: failures
        type(gaussian_family_t) :: family
        type(fortnum_status_t) :: status
        real(dp), allocatable :: lambda(:), extra(:)
        real(dp) :: value, likelihood, kl_value, scaled, scaled_likelihood
        real(dp) :: scaled_kl, direct_kl

        call family%initialize(n_features, 32, 7, status, &
            prior_variance=prior_variance)
        allocate(lambda(family%parameter_count()), extra(0))
        lambda = family%parameters()
        lambda(1:n_features) = [0.4_dp, -0.7_dp]
        call family%set_parameters(lambda, status)

        call vi_elbo(family, extra, model_log_likelihood, 1.0_dp, value, &
            likelihood, kl_value, status)
        call family%kl(direct_kl, status)
        if (.not. status_ok(status) .or. &
            abs(value - (likelihood - kl_value)) > 1.0e-13_dp .or. &
            abs(kl_value - direct_kl) > 1.0e-14_dp) then
            write (error_unit, '(a)') "FAIL [elbo] decomposition is inconsistent"
            failures = failures + 1
        end if

        call vi_elbo(family, extra, model_log_likelihood, 3.0_dp, scaled, &
            scaled_likelihood, scaled_kl, status)
        if (abs(scaled - (3.0_dp*likelihood - kl_value)) > 1.0e-12_dp .or. &
            abs(scaled_likelihood - likelihood) > 1.0e-14_dp .or. &
            abs(scaled_kl - kl_value) > 1.0e-14_dp) then
            write (error_unit, '(a)') &
                "FAIL [elbo] minibatch scaling touches the KL term"
            failures = failures + 1
        end if
    end subroutine test_elbo_decomposition_and_minibatch_scaling

    subroutine test_gradient_against_finite_difference(failures)
        integer, intent(inout) :: failures
        type(gaussian_family_t) :: family
        type(fortnum_status_t) :: status
        real(dp), allocatable :: lambda(:), gradient(:), reference(:), extra(:)
        real(dp), allocatable :: shifted(:)
        real(dp) :: value, plus, minus, likelihood, kl_value, h
        integer :: i, n

        call family%initialize(n_features, 32, 11, status, &
            prior_variance=prior_variance)
        n = family%parameter_count()
        allocate(lambda(n), gradient(n), reference(n), shifted(n), extra(0))
        do i = 1, n
            lambda(i) = 0.25_dp*sin(real(2*i, dp)) - 0.1_dp
        end do
        call family%set_parameters(lambda, status)
        call vi_elbo_gradient(family, extra, model_log_likelihood, 2.5_dp, &
            value, gradient, status)

        h = 1.0e-6_dp
        do i = 1, n
            shifted = lambda
            shifted(i) = shifted(i) + h
            call family%set_parameters(shifted, status)
            call vi_elbo(family, extra, model_log_likelihood, 2.5_dp, plus, &
                likelihood, kl_value, status)
            shifted(i) = lambda(i) - h
            call family%set_parameters(shifted, status)
            call vi_elbo(family, extra, model_log_likelihood, 2.5_dp, minus, &
                likelihood, kl_value, status)
            reference(i) = (plus - minus)/(2.0_dp*h)
        end do
        call family%set_parameters(lambda, status)
        if (.not. status_ok(status) .or. &
            maxval(abs(gradient - reference)) > 1.0e-6_dp) then
            write (error_unit, '(a,es12.4)') &
                "FAIL [grad] ELBO gradient finite difference ", &
                maxval(abs(gradient - reference))
            failures = failures + 1
        end if
    end subroutine test_gradient_against_finite_difference

    subroutine test_convergence_to_analytic_posterior(failures)
        integer, intent(inout) :: failures
        type(gaussian_family_t) :: family
        type(adam_t) :: optimizer
        type(fortnum_status_t) :: status
        real(dp), allocatable :: lambda(:), gradient(:), extra(:)
        real(dp) :: posterior_mean(n_features)
        real(dp) :: posterior_covariance(n_features, n_features)
        real(dp) :: covariance(n_features, n_features)
        real(dp) :: value, mean_error, covariance_error
        integer :: iteration, n

        call analytic_posterior(posterior_mean, posterior_covariance)
        call family%initialize(n_features, 64, 2026, status, &
            prior_variance=prior_variance)
        n = family%parameter_count()
        allocate(lambda(n), gradient(n), extra(0))
        lambda = family%parameters()
        call optimizer%initialize(n, status, learning_rate=0.02_dp)
        do iteration = 1, 4000
            call vi_elbo_gradient(family, extra, model_log_likelihood, 1.0_dp, &
                value, gradient, status)
            if (status%code /= FORTNUM_OK) exit
            ! fortopt minimizes, and the ELBO is maximized.
            gradient = -gradient
            call optimizer%step(lambda, gradient, status)
            call family%set_parameters(lambda, status)
        end do

        call family%covariance(covariance, status)
        mean_error = maxval(abs(family%mean - posterior_mean))
        covariance_error = maxval(abs(covariance - posterior_covariance))
        if (.not. status_ok(status) .or. mean_error > 1.0e-5_dp .or. &
            covariance_error > 1.0e-5_dp) then
            write (error_unit, '(a,2es12.4)') &
                "FAIL [vi] posterior mean/covariance error ", mean_error, &
                covariance_error
            failures = failures + 1
        end if
    end subroutine test_convergence_to_analytic_posterior

    subroutine analytic_posterior(mean, covariance)
        !! Exact conjugate posterior, formed by inverting the 2x2 precision
        !! directly so it does not share code with the model under test.
        real(dp), intent(out) :: mean(:), covariance(:, :)
        real(dp) :: precision(n_features, n_features), determinant
        integer :: i, j

        do j = 1, n_features
            do i = 1, n_features
                precision(i, j) = sum(design(:, i)*design(:, j))/noise_variance
            end do
            precision(j, j) = precision(j, j) + 1.0_dp/prior_variance
        end do
        determinant = precision(1, 1)*precision(2, 2) &
            - precision(1, 2)*precision(2, 1)
        covariance(1, 1) = precision(2, 2)/determinant
        covariance(2, 2) = precision(1, 1)/determinant
        covariance(1, 2) = -precision(1, 2)/determinant
        covariance(2, 1) = -precision(2, 1)/determinant
        do i = 1, n_features
            mean(i) = sum(covariance(i, :)* &
                [(sum(design(:, j)*observations)/noise_variance, &
                j=1, n_features)])
        end do
    end subroutine analytic_posterior

end program test_variational
