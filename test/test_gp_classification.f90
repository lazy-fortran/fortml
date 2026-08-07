program test_gp_classification
    !! Independent behavioral and finite-difference checks for GP classification.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_gp_classification, only: gp_classification_t, &
        gp_classification_options_t, gp_classification_state_t, &
        GP_LIKELIHOOD_PROBIT
    use fortml_kernels, only: kernel_t, make_rbf_kernel
    use fortml_kernels, only: clone_kernel
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    type(gp_classification_t) :: model, probit_model, unfitted
    type(gp_classification_t) :: model_plus, model_minus
    type(gp_classification_options_t) :: options
    type(gp_classification_state_t) :: state, state_plus, state_minus
    type(fortnum_status_t) :: status
    type(kernel_t) :: kernel, kernel_plus, kernel_minus
    real(dp) :: x(8, 1), x_test(5, 1), x_dot(5, 1), x_plus(5, 1), x_minus(5, 1)
    real(dp) :: mean(5), mean_dot(5), variance(5), variance_dot(5)
    real(dp) :: mean_plus(5), mean_minus(5), variance_plus(5), variance_minus(5)
    real(dp) :: probabilities(5, 2), probabilities_dot(5, 2)
    real(dp) :: probabilities_plus(5, 2), probabilities_minus(5, 2)
    real(dp) :: mean_bar(5), variance_bar(5), x_bar(5, 1)
    real(dp) :: probabilities_bar(5, 2), probabilities_x_bar(5, 1)
    real(dp), allocatable :: kernel_parameters(:), model_parameters(:), gradient(:)
    real(dp), allocatable :: gradient_fd(:), theta_plus(:), theta_minus(:)
    integer :: labels(8), predicted(8), classes(2), failures, k
    real(dp) :: h

    x(:, 1) = [-1.5_dp, -1.0_dp, -0.5_dp, -0.1_dp, 0.1_dp, 0.5_dp, 1.0_dp, 1.5_dp]
    labels = [-7, -7, -7, -7, 11, 11, 11, 11]
    x_test(:, 1) = [-1.25_dp, -0.4_dp, 0.0_dp, 0.4_dp, 1.25_dp]
    x_dot(:, 1) = [0.3_dp, -0.4_dp, 0.2_dp, 0.8_dp, -0.1_dp]
    failures = 0

    kernel = make_rbf_kernel(1, 1.4_dp, 0.7_dp, status)
    options%max_iterations = 100
    options%tolerance = 1.0e-9_dp
    options%jitter = 1.0e-7_dp
    call model%fit(x, labels, kernel, status, options, state)
    call check(status_ok(status) .and. state%converged .and. model%fitted(), &
        "logistic Laplace fit", failures)
    classes = model%classes()
    call check(all(classes == [-7, 11]) .and. model%feature_count() == 1, &
        "class metadata", failures)
    kernel_parameters = kernel%parameters()
    model_parameters = model%parameters()
    allocate(gradient(model%parameter_count()), gradient_fd(model%parameter_count()))
    call check(model%parameter_count() == size(kernel_parameters) .and. &
        maxval(abs(model_parameters - kernel_parameters)) < 1.0e-14_dp, &
        "kernel parameter metadata", failures)
    call model%hyperparameter_gradient(gradient, status)
    call check(status_ok(status), "Laplace hyperparameter gradient status", failures)
    ! The independent oracle refits at perturbed log-kernel parameters and
    ! differentiates the converged mode log posterior returned in state.
    h = 1.0e-5_dp
    allocate(theta_plus(size(kernel_parameters)), theta_minus(size(kernel_parameters)))
    do k = 1, size(kernel_parameters)
        theta_plus = kernel_parameters
        theta_minus = kernel_parameters
        theta_plus(k) = theta_plus(k) + h
        theta_minus(k) = theta_minus(k) - h
        kernel_plus = clone_kernel(kernel)
        kernel_minus = clone_kernel(kernel)
        call kernel_plus%set_parameters(theta_plus, status)
        call check(status_ok(status), "positive kernel probe setup", failures)
        call kernel_minus%set_parameters(theta_minus, status)
        call check(status_ok(status), "negative kernel probe setup", failures)
        call model_plus%fit(x, labels, kernel_plus, status, options, state_plus)
        call check(status_ok(status) .and. state_plus%converged, &
            "positive kernel probe fit", failures)
        call model_minus%fit(x, labels, kernel_minus, status, options, state_minus)
        call check(status_ok(status) .and. state_minus%converged, &
            "negative kernel probe fit", failures)
        gradient_fd(k) = (state_plus%log_posterior - state_minus%log_posterior)/ &
            (2.0_dp*h)
    end do
    call check(maxval(abs(gradient - gradient_fd)) < 2.0e-5_dp, &
        "Laplace hyperparameter gradient finite difference", failures)
    call model%predict(x, predicted, status)
    call check(status_ok(status) .and. count(predicted == labels) >= 7, &
        "training classification", failures)

    call model%predict_latent(x_test, mean, variance, status)
    call model%predict_proba(x_test, probabilities, status)
    call check(status_ok(status) .and. all(variance >= 0.0_dp) .and. &
        maxval(abs(sum(probabilities, dim=2) - 1.0_dp)) < 2.0e-14_dp .and. &
        all(probabilities >= 0.0_dp), "latent and probability outputs", failures)
    call check(probabilities(1, 1) > probabilities(1, 2) .and. &
        probabilities(5, 2) > probabilities(5, 1), &
        "probability orientation", failures)

    ! Input JVPs are checked against an independently evaluated central difference.
    call model%predict_latent_jvp(x_test, x_dot, mean, mean_dot, variance, &
        variance_dot, status)
    h = 1.0e-5_dp
    x_plus = x_test + h*x_dot
    x_minus = x_test - h*x_dot
    call model%predict_latent(x_plus, mean_plus, variance_plus, status)
    call model%predict_latent(x_minus, mean_minus, variance_minus, status)
    call check(status_ok(status) .and. maxval(abs(mean_dot - &
        (mean_plus - mean_minus)/(2.0_dp*h))) < 2.0e-6_dp .and. &
        maxval(abs(variance_dot - (variance_plus - variance_minus)/(2.0_dp*h))) &
        < 2.0e-6_dp, "latent input JVP finite difference", failures)

    call model%predict_proba_jvp(x_test, x_dot, probabilities, probabilities_dot, status)
    call model%predict_proba(x_plus, probabilities_plus, status)
    call model%predict_proba(x_minus, probabilities_minus, status)
    call check(status_ok(status) .and. maxval(abs(probabilities_dot - &
        (probabilities_plus - probabilities_minus)/(2.0_dp*h))) < 2.0e-6_dp, &
        "probability input JVP finite difference", failures)

    ! Reverse products satisfy the dot-product identity against the JVPs.
    mean_bar = [0.4_dp, -0.2_dp, 0.7_dp, -0.1_dp, 0.3_dp]
    variance_bar = [-0.3_dp, 0.6_dp, -0.4_dp, 0.2_dp, 0.5_dp]
    call model%predict_latent_vjp(x_test, mean_bar, variance_bar, x_bar, status)
    call check(status_ok(status) .and. abs(sum(x_bar(:, 1)*x_dot(:, 1)) - &
        (sum(mean_bar*mean_dot) + sum(variance_bar*variance_dot))) < 3.0e-6_dp, &
        "latent input VJP dot-product identity", failures)
    probabilities_bar(:, 1) = [-0.2_dp, 0.4_dp, 0.1_dp, -0.5_dp, 0.3_dp]
    probabilities_bar(:, 2) = [0.6_dp, -0.1_dp, 0.8_dp, 0.2_dp, -0.4_dp]
    call model%predict_proba_vjp(x_test, probabilities_bar, probabilities_x_bar, &
        status)
    call check(status_ok(status) .and. abs(sum(probabilities_x_bar(:, 1)* &
        x_dot(:, 1)) - sum(probabilities_bar*probabilities_dot)) < 3.0e-6_dp, &
        "probability input VJP dot-product identity", failures)

    options%likelihood = GP_LIKELIHOOD_PROBIT
    call probit_model%fit(x, labels, kernel, status, options, state)
    call probit_model%predict_proba(x_test, probabilities, status)
    call check(status_ok(status) .and. state%converged .and. &
        maxval(abs(sum(probabilities, dim=2) - 1.0_dp)) < 2.0e-14_dp .and. &
        probabilities(1, 1) > probabilities(1, 2) .and. &
        probabilities(5, 2) > probabilities(5, 1), "probit Laplace fit", failures)

    call unfitted%predict_proba(x_test, probabilities, status)
    call check(.not. status_ok(status), "unfitted prediction refusal", failures)
    call model%fit(x, [1, 1, 1, 1, 1, 1, 1, 1], kernel, status, options)
    call check(.not. status_ok(status), "one-class refusal", failures)
    options%likelihood = 99
    call model%fit(x, labels, kernel, status, options)
    call check(.not. status_ok(status), "unsupported likelihood refusal", failures)

    if (failures > 0) then
        write (error_unit, '(a,i0)') "FAIL GP classification cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS GP classification independent behavioral oracles"

contains

    subroutine check(condition, name, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: name
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "  FAIL [gp-classification] "//name
        end if
    end subroutine check

end program test_gp_classification
