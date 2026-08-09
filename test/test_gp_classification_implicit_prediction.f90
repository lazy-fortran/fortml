program test_gp_classification_implicit_prediction
    !! Exact prediction JVP through a converged Laplace fit, checked by
    !! independent central differences that refit both parameter probes.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortnum_status, only: fortnum_status_t, status_ok, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    use fortml_kernels, only: kernel_t, make_rbf_kernel, clone_kernel
    use fortml_gp_classification, only: gp_classification_t, &
        gp_classification_options_t, GP_LIKELIHOOD_LOGISTIC, GP_LIKELIHOOD_PROBIT
    implicit none

    integer :: failures, likelihood

    failures = 0
    do likelihood = GP_LIKELIHOOD_LOGISTIC, GP_LIKELIHOOD_PROBIT
        call test_refit_product(likelihood, failures)
    end do
    call test_contracts(failures)
    if (failures /= 0) then
        write (error_unit, '(a,i0)') &
            "FAIL GP classification implicit prediction cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS GP classification implicit prediction JVP oracles"

contains

    subroutine fixture(x, labels, weights, query, options, kernel, status)
        real(dp), intent(out) :: x(8, 1), weights(8), query(3, 1)
        integer, intent(out) :: labels(8)
        type(gp_classification_options_t), intent(out) :: options
        type(kernel_t), intent(out) :: kernel
        type(fortnum_status_t), intent(out) :: status

        x(:, 1) = [-1.7_dp, -1.15_dp, -0.62_dp, -0.18_dp, &
            0.14_dp, 0.55_dp, 1.08_dp, 1.63_dp]
        labels = [-3, -3, -3, -3, 9, 9, 9, 9]
        weights = [0.45_dp, 1.4_dp, 0.0_dp, 0.8_dp, &
            1.7_dp, 0.6_dp, 1.25_dp, 0.9_dp]
        query(:, 1) = [-0.85_dp, 0.05_dp, 0.92_dp]
        options%max_iterations = 120
        options%tolerance = 1.0e-11_dp
        options%jitter = 1.0e-7_dp
        kernel = make_rbf_kernel(1, 1.35_dp, 0.72_dp, status)
    end subroutine fixture

    subroutine test_refit_product(likelihood, failures)
        integer, intent(in) :: likelihood
        integer, intent(inout) :: failures
        type(gp_classification_t) :: model, model_plus, model_minus
        type(gp_classification_options_t) :: options
        type(kernel_t) :: kernel, kernel_plus, kernel_minus
        type(fortnum_status_t) :: status
        real(dp) :: x(8, 1), weights(8), query(3, 1), direction(2), theta(2)
        real(dp) :: mean(3), mean_dot(3), variance(3), variance_dot(3)
        real(dp) :: mean_plus(3), mean_minus(3), variance_plus(3), variance_minus(3)
        real(dp) :: probabilities(3, 2), probabilities_dot(3, 2)
        real(dp) :: probabilities_plus(3, 2), probabilities_minus(3, 2)
        real(dp) :: h
        integer :: labels(8)

        call fixture(x, labels, weights, query, options, kernel, status)
        call check(status_ok(status), "implicit prediction kernel fixture", failures)
        options%likelihood = likelihood
        call model%fit(x, labels, kernel, status, options, sample_weight=weights)
        call check(status_ok(status), "implicit prediction base fit", failures)
        direction = [0.19_dp, -0.14_dp]
        call model%predict_latent_hyperparameter_jvp(query, direction, mean, &
            mean_dot, variance, variance_dot, status)
        call check(status_ok(status), "implicit latent JVP status", failures)
        call model%predict_proba_hyperparameter_jvp(query, direction, probabilities, &
            probabilities_dot, status)
        call check(status_ok(status), "implicit probability JVP status", failures)

        theta = kernel%parameters()
        h = 2.0e-5_dp
        kernel_plus = clone_kernel(kernel)
        kernel_minus = clone_kernel(kernel)
        call kernel_plus%set_parameters(theta + h*direction, status)
        call check(status_ok(status), "positive implicit probe kernel", failures)
        call kernel_minus%set_parameters(theta - h*direction, status)
        call check(status_ok(status), "negative implicit probe kernel", failures)
        call model_plus%fit(x, labels, kernel_plus, status, options, &
            sample_weight=weights)
        call check(status_ok(status), "positive implicit probe refit", failures)
        call model_minus%fit(x, labels, kernel_minus, status, options, &
            sample_weight=weights)
        call check(status_ok(status), "negative implicit probe refit", failures)
        call model_plus%predict_latent(query, mean_plus, variance_plus, status)
        call check(status_ok(status), "positive latent probe", failures)
        call model_minus%predict_latent(query, mean_minus, variance_minus, status)
        call check(status_ok(status), "negative latent probe", failures)
        call model_plus%predict_proba(query, probabilities_plus, status)
        call check(status_ok(status), "positive probability probe", failures)
        call model_minus%predict_proba(query, probabilities_minus, status)
        call check(status_ok(status), "negative probability probe", failures)

        call check(maxval(abs(mean_dot - (mean_plus - mean_minus)/(2.0_dp*h))) &
            < 3.0e-6_dp, "implicit latent mean finite difference", failures)
        call check(maxval(abs(variance_dot - (variance_plus - variance_minus)/ &
            (2.0_dp*h))) < 3.0e-6_dp, &
            "implicit latent variance finite difference", failures)
        call check(maxval(abs(probabilities_dot - (probabilities_plus - &
            probabilities_minus)/(2.0_dp*h))) < 3.0e-6_dp, &
            "implicit probability finite difference", failures)
        call check(maxval(abs(sum(probabilities_dot, dim=2))) < 1.0e-13_dp, &
            "implicit probability tangent conserves row mass", failures)
    end subroutine test_refit_product

    subroutine test_contracts(failures)
        integer, intent(inout) :: failures
        type(gp_classification_t) :: model, unfitted
        type(gp_classification_options_t) :: options
        type(kernel_t) :: kernel
        type(fortml_device_t) :: device
        type(fortnum_status_t) :: status
        real(dp) :: x(8, 1), weights(8), query(3, 1), probabilities(3, 2)
        real(dp) :: probabilities_dot(3, 2), expected(3, 2)
        integer :: labels(8)

        call fixture(x, labels, weights, query, options, kernel, status)
        call model%fit(x, labels, kernel, status, options, sample_weight=weights)
        call check(status_ok(status), "implicit contract fit", failures)
        call unfitted%predict_proba_hyperparameter_jvp(query, [0.1_dp, -0.2_dp], &
            probabilities, probabilities_dot, status)
        call check(status%code == FORTNUM_DOMAIN_ERROR, &
            "unfitted implicit prediction refusal", failures)
        call model%predict_proba_hyperparameter_jvp(query, [0.1_dp], &
            probabilities, probabilities_dot, status)
        call check(status%code == FORTNUM_DOMAIN_ERROR, &
            "implicit direction-shape refusal", failures)

        device%kind = FORTML_DEVICE_CPU
        device%selected = .true.
        device%available = .true.
        call model%predict_proba_hyperparameter_jvp_device(device, query, &
            [0.1_dp, -0.2_dp], probabilities, probabilities_dot, status)
        call check(status_ok(status), "implicit CPU device dispatch", failures)
        expected = probabilities_dot

        device%kind = FORTML_DEVICE_CUDA
        probabilities = -71.0_dp
        probabilities_dot = 83.0_dp
        call model%predict_proba_hyperparameter_jvp_device(device, query, &
            [0.1_dp, -0.2_dp], probabilities, probabilities_dot, status)
        call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
            "implicit CUDA device refusal", failures)
        call check(all(probabilities == -71.0_dp) .and. &
            all(probabilities_dot == 83.0_dp), &
            "implicit CUDA refusal preserves outputs", failures)
        call check(all(expected == expected), "implicit CPU result is finite", failures)
    end subroutine test_contracts

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "  FAIL [gp implicit prediction] "//description
        end if
    end subroutine check

end program test_gp_classification_implicit_prediction
