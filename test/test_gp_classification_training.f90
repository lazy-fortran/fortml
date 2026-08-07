program test_gp_classification_training
    !! Independent finite-difference and refusal checks for GP optimizers.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_gp_classification, only: gp_classification_t, &
        gp_classification_options_t, gp_classification_state_t
    use fortml_gp_multiclass_classification, only: &
        gp_multiclass_classification_t
    use fortml_gp_classification_training, only: &
        gp_classification_hyperparameter_options_t, &
        gp_classification_hyperparameter_result_t, &
        gp_multiclass_hyperparameter_options_t, &
        gp_multiclass_hyperparameter_result_t, &
        gp_classification_optimize_hyperparameters, &
        gp_multiclass_optimize_hyperparameters
    use fortml_kernels, only: kernel_t, make_rbf_kernel, clone_kernel
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    type(gp_classification_t) :: model, model_plus, model_minus
    type(gp_multiclass_classification_t) :: multiclass_model
    type(gp_classification_hyperparameter_options_t) :: options
    type(gp_classification_hyperparameter_result_t) :: result
    type(gp_multiclass_hyperparameter_options_t) :: multiclass_options
    type(gp_multiclass_hyperparameter_result_t) :: multiclass_result
    type(gp_classification_options_t) :: fit_options
    type(gp_classification_state_t) :: state_plus, state_minus
    type(fortnum_status_t) :: status
    type(kernel_t) :: kernel, kernel_plus, kernel_minus, multiclass_kernel
    real(dp) :: x(8, 1), x_fd(8, 1), theta(2), theta_plus(2), theta_minus(2)
    real(dp), allocatable :: gradient(:), gradient_fd(:)
    integer :: labels(8), multiclass_labels(8), failures, k
    real(dp) :: h

    x(:, 1) = [-1.5_dp, -1.0_dp, -0.5_dp, -0.1_dp, 0.1_dp, 0.5_dp, 1.0_dp, 1.5_dp]
    labels = [-7, -7, -7, -7, 11, 11, 11, 11]
    x_fd = x
    failures = 0
    fit_options%max_iterations = 100
    fit_options%tolerance = 1.0e-9_dp
    fit_options%jitter = 1.0e-7_dp

    kernel = make_rbf_kernel(1, 1.2_dp, 0.8_dp, status)
    options%fit = fit_options
    options%max_iterations = 100
    options%gradient_tolerance = 2.0e-3_dp
    options%lower_bound = -5.0_dp
    options%upper_bound = 5.0_dp
    call gp_classification_optimize_hyperparameters(model, x, labels, kernel, &
        options, result, status)
    call check(status_ok(status) .and. result%converged .and. model%fitted(), &
        "binary L-BFGS-B fit", failures)
    call check(result%negative_log_posterior < huge(1.0_dp) .and. &
        result%gradient_norm < huge(1.0_dp), &
        "binary finite objective and gradient", failures)

    theta = kernel%parameters()
    allocate(gradient(size(theta)), gradient_fd(size(theta)))
    call model%hyperparameter_gradient(gradient, status)
    h = 2.0e-5_dp
    do k = 1, size(theta)
        theta_plus = theta
        theta_minus = theta
        theta_plus(k) = theta_plus(k) + h
        theta_minus(k) = theta_minus(k) - h
        kernel_plus = clone_kernel(kernel)
        kernel_minus = clone_kernel(kernel)
        call kernel_plus%set_parameters(theta_plus, status)
        call model_plus%fit(x_fd, labels, kernel_plus, status, fit_options, state_plus)
        call kernel_minus%set_parameters(theta_minus, status)
        call model_minus%fit(x_fd, labels, kernel_minus, status, fit_options, state_minus)
        call check(status_ok(status) .and. state_plus%converged .and. &
            state_minus%converged, "binary finite-difference refits", failures)
        gradient_fd(k) = (state_plus%log_posterior - state_minus%log_posterior) / (2.0_dp*h)
    end do
    call check(maxval(abs(gradient - gradient_fd)) < 3.0e-5_dp, &
        "binary envelope gradient oracle", failures)

    options%lower_bound = 2.0_dp
    options%upper_bound = -2.0_dp
    call gp_classification_optimize_hyperparameters(model, x, labels, kernel, &
        options, result, status)
    call check(.not. status_ok(status), "binary invalid-bound refusal", failures)

    multiclass_labels = [42, 42, 42, -7, -7, 11, 11, 11]
    multiclass_kernel = make_rbf_kernel(1, 1.2_dp, 0.8_dp, status)
    multiclass_options%fit%max_iterations = 100
    multiclass_options%fit%tolerance = 1.0e-9_dp
    multiclass_options%fit%jitter = 1.0e-7_dp
    multiclass_options%max_iterations = 100
    multiclass_options%gradient_tolerance = 5.0e-3_dp
    multiclass_options%lower_bound = -5.0_dp
    multiclass_options%upper_bound = 5.0_dp
    call gp_multiclass_optimize_hyperparameters(multiclass_model, x_fd, &
        multiclass_labels, multiclass_kernel, multiclass_options, multiclass_result, status)
    call check(status_ok(status) .and. multiclass_result%converged .and. &
        multiclass_model%fitted() .and. multiclass_result%negative_log_posterior < huge(1.0_dp), &
        "shared-kernel multiclass L-BFGS-B fit", failures)

    multiclass_options%fit%likelihood = 99
    call gp_multiclass_optimize_hyperparameters(multiclass_model, x_fd, multiclass_labels, &
        multiclass_kernel, multiclass_options, multiclass_result, status)
    call check(.not. status_ok(status), "multiclass invalid-likelihood refusal", failures)

    if (failures > 0) then
        write (error_unit, '(a,i0)') "FAIL GP classification training cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS GP classification training independent oracles"

contains

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "  FAIL [gp-classification-training] "//description
        end if
    end subroutine check

end program test_gp_classification_training
