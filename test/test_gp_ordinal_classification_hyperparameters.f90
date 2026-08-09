program test_gp_ordinal_classification_hyperparameters
    !! Independent evidence-gradient/HVP and optimizer oracle for ordinal GPs.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortml_kernels, only: kernel_t, make_rbf_kernel
    use fortml_gp_ordinal_classification, only: &
        gp_ordinal_classification_t, gp_ordinal_classification_options_t
    use fortml_gp_ordinal_classification_training, only: &
        gp_ordinal_hyperparameter_options_t, gp_ordinal_hyperparameter_result_t, &
        gp_ordinal_optimize_hyperparameters
    implicit none

    type(gp_ordinal_classification_t) :: model, model_plus, model_minus, model_opt
    type(gp_ordinal_classification_options_t) :: fit_options
    type(gp_ordinal_hyperparameter_options_t) :: train_options
    type(gp_ordinal_hyperparameter_result_t) :: train_result
    type(kernel_t) :: kernel
    type(fortml_device_t) :: cuda
    type(fortnum_status_t) :: status
    real(dp) :: x(9, 1), direction(3), hvp(3), hvp_fd(3)
    real(dp) :: lml, lml_plus, lml_minus, lml_dot, lml_fd, h
    real(dp), allocatable :: theta(:), theta_plus(:), theta_minus(:)
    real(dp), allocatable :: gradient(:), gradient_plus(:), gradient_minus(:)
    integer :: labels(9), failures, i, j, n_parameters

    x(:, 1) = [-1.5_dp, -1.1_dp, -0.8_dp, -0.2_dp, 0.0_dp, 0.3_dp, &
        0.8_dp, 1.1_dp, 1.5_dp]
    labels = [-4, -4, -4, 7, 7, 7, 19, 19, 19]
    fit_options%noise_variance = 0.04_dp
    fit_options%jitter = 1.0e-8_dp
    h = 2.0e-5_dp
    failures = 0
    kernel = make_rbf_kernel(1, 1.2_dp, 0.7_dp, status)
    call check(status_ok(status), "RBF setup", failures)
    call model%fit(x, labels, kernel, status, fit_options)
    call check(status_ok(status) .and. model%fitted(), "ordinal GP fit", failures)
    call check(model%hyperparameter_count() == model%parameter_count(), &
        "hyperparameter metadata", failures)

    n_parameters = model%hyperparameter_count()
    allocate(theta(n_parameters), theta_plus(n_parameters), theta_minus(n_parameters), &
        gradient(n_parameters), gradient_plus(n_parameters), gradient_minus(n_parameters))
    theta = model%hyperparameters()
    call model%log_marginal_likelihood(lml, status)
    call check(status_ok(status), "latent evidence", failures)
    call model%hyperparameter_gradient(gradient, status)
    call check(status_ok(status) .and. all(abs(gradient) < huge(1.0_dp)), &
        "analytic evidence gradient", failures)

    ! Every packed coordinate is compared against an independently refit
    ! central difference of the evidence, including log noise.
    do j = 1, size(theta)
        direction = 0.0_dp
        direction(j) = 1.0_dp
        call model%log_marginal_likelihood_jvp(direction, lml_dot, status)
        call check(status_ok(status), "evidence JVP status", failures)
        theta_plus = theta + h*direction
        theta_minus = theta - h*direction
        call model_plus%fit(x, labels, kernel, status, fit_options)
        call check(status_ok(status), "positive evidence probe fit", failures)
        call model_minus%fit(x, labels, kernel, status, fit_options)
        call check(status_ok(status), "negative evidence probe fit", failures)
        call model_plus%set_hyperparameters(theta_plus, status)
        call check(status_ok(status), "positive evidence probe setter", failures)
        call model_minus%set_hyperparameters(theta_minus, status)
        call check(status_ok(status), "negative evidence probe setter", failures)
        call model_plus%log_marginal_likelihood(lml_plus, status)
        call model_minus%log_marginal_likelihood(lml_minus, status)
        lml_fd = (lml_plus - lml_minus)/(2.0_dp*h)
        call check(status_ok(status) .and. abs(lml_dot - lml_fd) < 3.0e-5_dp, &
            "evidence gradient finite difference", failures)
    end do

    direction = [0.13_dp, -0.21_dp, 0.17_dp]
    call model%hyperparameter_hvp(direction, hvp, status)
    call check(status_ok(status), "analytic evidence HVP", failures)
    theta_plus = theta + h*direction
    theta_minus = theta - h*direction
    call model_plus%set_hyperparameters(theta_plus, status)
    call model_minus%set_hyperparameters(theta_minus, status)
    call model_plus%hyperparameter_gradient(gradient_plus, status)
    call model_minus%hyperparameter_gradient(gradient_minus, status)
    hvp_fd = (gradient_plus - gradient_minus)/(2.0_dp*h)
    call check(status_ok(status) .and. maxval(abs(hvp - hvp_fd)) < 3.0e-4_dp, &
        "evidence HVP finite difference", failures)

    ! A failed setter is transactional, including the factorization state.
    call model%set_hyperparameters(theta(:2), status)
    call check(.not. status_ok(status) .and. maxval(abs(model%hyperparameters() - theta)) < &
        1.0e-14_dp, "transactional hyperparameter refusal", failures)

    ! The FortOpt adapter must consume the same analytic objective and retain
    ! a finite converged state on this deterministic fixture.
    call model_opt%fit(x, labels, kernel, status, fit_options)
    train_options%max_iterations = 120
    train_options%max_line_search = 40
    train_options%gradient_tolerance = 2.0e-5_dp
    train_options%lower_bound = -8.0_dp
    train_options%upper_bound = 8.0_dp
    call gp_ordinal_optimize_hyperparameters(model_opt, train_options, train_result, status)
    call check(status_ok(status) .and. train_result%converged .and. &
        train_result%negative_log_marginal_likelihood < huge(1.0_dp) .and. &
        train_result%gradient_norm < 1.0e-3_dp .and. &
        train_result%negative_log_marginal_likelihood < -lml, &
        "FortOpt evidence training improves the fitted evidence", failures)

    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call model%hyperparameter_gradient_device(cuda, gradient, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "typed CUDA evidence gradient refusal", failures)
    call model%hyperparameter_hvp_device(cuda, direction, hvp, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "typed CUDA evidence HVP refusal", failures)
    call gp_ordinal_optimize_hyperparameters(model, train_options, train_result, status, cuda)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "typed CUDA optimizer refusal", failures)

    if (failures > 0) then
        write (error_unit, '(a,i0)') "FAIL ordinal GP hyperparameter cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS ordinal GP evidence gradient/HVP and FortOpt oracle"

contains

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "  FAIL [gp-ordinal-hyperparameters] "//description
        end if
    end subroutine check

end program test_gp_ordinal_classification_hyperparameters
