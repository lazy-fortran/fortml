program test_gp_variational_categorical_likelihood
    !! Independent finite-difference oracle for categorical-GP likelihood products.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortml_kernels, only: kernel_t, make_rbf_kernel
    use fortml_gp_variational_categorical_classification, only: &
        gp_variational_categorical_classification_t, &
        gp_variational_likelihood_options_t, gp_variational_likelihood_state_t
    implicit none

    type(gp_variational_categorical_classification_t) :: model, fit_model
    type(gp_variational_likelihood_options_t) :: options
    type(gp_variational_likelihood_state_t) :: fit_state
    type(kernel_t) :: kernel
    type(fortnum_status_t) :: status
    type(fortml_device_t) :: cuda
    real(dp) :: x(6, 1), inducing(3, 1), probabilities(6, 3), probabilities_dot(6, 3)
    real(dp) :: probabilities_plus(6, 3), probabilities_minus(6, 3), probabilities_bar(6, 3)
    real(dp) :: parameter_bar(1), parameter_bar_fd(1), labels_value, value_plus, value_minus
    real(dp) :: value, tangent, h, log_scale, expected_scale, expected(6, 3), logits(3), total
    real(dp), allocatable :: likelihood_parameters(:)
    integer :: labels(6), classes(3), failures, i, j

    x(:, 1) = [-1.2_dp, -0.7_dp, -0.15_dp, 0.25_dp, 0.8_dp, 1.3_dp]
    inducing(:, 1) = [-0.9_dp, 0.0_dp, 0.85_dp]
    labels = [30, 10, 20, 10, 30, 20]
    classes = [30, 10, 20]
    probabilities_bar = reshape([ &
        0.17_dp, -0.09_dp, 0.04_dp, 0.12_dp, -0.06_dp, 0.08_dp, &
        -0.05_dp, 0.11_dp, -0.08_dp, 0.07_dp, 0.03_dp, -0.02_dp, &
        0.02_dp, 0.06_dp, 0.09_dp, -0.04_dp, 0.05_dp, 0.01_dp], [6, 3])
    failures = 0
    h = 2.0e-6_dp

    kernel = make_rbf_kernel(1, 1.2_dp, 0.8_dp, status)
    call check(status_ok(status), "kernel constructor", failures)
    call model%initialize(inducing, classes, kernel, 8, 20260808, status)
    call check(status_ok(status), "categorical initialization", failures)
    call check(model%likelihood_parameter_count() == 1, "one likelihood coordinate", failures)
    likelihood_parameters = model%likelihood_parameters()
    call check(abs(likelihood_parameters(1)) < 1.0e-14_dp, &
        "unit log-scale initialization", failures)

    log_scale = log(1.7_dp)
    call model%set_likelihood_parameters([log_scale], status)
    call check(status_ok(status), "set likelihood coordinate", failures)
    expected_scale = exp(log_scale)
    call model%predict_latent(x, expected, probabilities, status)
    call check(status_ok(status), "latent reference", failures)
    do i = 1, size(x, 1)
        do j = 1, 3
            logits(j) = expected_scale*expected(i, j)/sqrt( &
                1.0_dp + 3.141592653589793_dp*probabilities(i, j)/8.0_dp)
        end do
        total = sum(exp(logits - maxval(logits)))
        expected(i, :) = exp(logits - maxval(logits))/total
    end do
    call model%predict_proba(x, probabilities_plus, status)
    call check(status_ok(status), "scaled probability prediction", failures)
    call check(maxval(abs(probabilities_plus - expected)) < 2.0e-13_dp, &
        "temperature softmax oracle", failures)

    call model%predict_proba_likelihood_parameter_jvp(x, [1.0_dp], probabilities, &
        probabilities_dot, status)
    call check(status_ok(status), "likelihood probability JVP", failures)
    call model%set_likelihood_parameters([log_scale + h], status)
    call model%predict_proba(x, probabilities_plus, status)
    call model%set_likelihood_parameters([log_scale - h], status)
    call model%predict_proba(x, probabilities_minus, status)
    call model%set_likelihood_parameters([log_scale], status)
    call check(maxval(abs(probabilities_dot - (probabilities_plus - probabilities_minus)/(2.0_dp*h))) < 2.0e-4_dp, &
        "likelihood probability JVP finite difference", failures)

    call model%predict_proba_likelihood_parameter_vjp(x, probabilities_bar, parameter_bar, status)
    call check(status_ok(status), "likelihood probability VJP", failures)
    parameter_bar_fd(1) = 0.0_dp
    call model%set_likelihood_parameters([log_scale + h], status)
    call model%predict_proba(x, probabilities_plus, status)
    call model%set_likelihood_parameters([log_scale - h], status)
    call model%predict_proba(x, probabilities_minus, status)
    parameter_bar_fd(1) = sum(probabilities_bar*(probabilities_plus - probabilities_minus))/(2.0_dp*h)
    call model%set_likelihood_parameters([log_scale], status)
    call check(abs(parameter_bar(1) - parameter_bar_fd(1)) < 2.0e-4_dp, &
        "likelihood probability VJP finite difference", failures)
    call check(abs(parameter_bar(1) - sum(probabilities_dot*probabilities_bar)) < 2.0e-4_dp, &
        "likelihood probability adjoint identity", failures)

    call model%elbo_likelihood_parameter_gradient(x, labels, value, parameter_bar, status)
    call check(status_ok(status), "likelihood ELBO gradient", failures)
    call model%set_likelihood_parameters([log_scale + h], status)
    call model%elbo(x, labels, value_plus, status)
    call model%set_likelihood_parameters([log_scale - h], status)
    call model%elbo(x, labels, value_minus, status)
    call model%set_likelihood_parameters([log_scale], status)
    call check(abs(parameter_bar(1) - (value_plus - value_minus)/(2.0_dp*h)) < 2.0e-4_dp, &
        "likelihood ELBO gradient finite difference", failures)
    call model%elbo_likelihood_parameter_jvp(x, labels, [1.0_dp], value, tangent, status)
    call check(status_ok(status), "likelihood ELBO JVP", failures)
    call check(abs(tangent - parameter_bar(1)) < 2.0e-12_dp, &
        "likelihood ELBO JVP equals gradient", failures)

    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call model%predict_proba_likelihood_parameter_jvp_device(cuda, x, [1.0_dp], &
        probabilities, probabilities_dot, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, "CUDA likelihood JVP refusal", failures)
    call model%predict_proba_likelihood_parameter_vjp_device(cuda, x, probabilities_bar, &
        parameter_bar, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, "CUDA likelihood VJP refusal", failures)

    call fit_model%initialize(inducing, classes, kernel, 8, 20260808, status)
    options%max_iterations = 80
    options%gradient_tolerance = 1.0e-6_dp
    call fit_model%fit_likelihood(x, labels, status, options, fit_state)
    call check(status_ok(status), "FortOpt likelihood fit", failures)
    call check(fit_state%converged, "FortOpt likelihood convergence", failures)
    call check(fit_model%likelihood_scale() > 0.0_dp, "positive fitted likelihood scale", failures)

    if (failures > 0) then
        write (error_unit, '(a,i0)') "FAIL categorical likelihood cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS categorical-GP likelihood independent oracles"

contains

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "  FAIL [categorical-likelihood] "//description
        end if
    end subroutine check

end program test_gp_variational_categorical_likelihood
