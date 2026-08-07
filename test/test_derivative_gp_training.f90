program test_derivative_gp_training
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_derivative_gaussian_process, only: gp_derivative_regression_t
    use fortml_derivative_gp_training, only: &
        gp_optimize_derivative_hyperparameters
    use fortml_gp_training, only: gp_hyperparameter_options_t, &
        gp_hyperparameter_result_t
    use fortml_kernels, only: kernel_t, make_rbf_kernel
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    type(gp_derivative_regression_t) :: model
    type(gp_hyperparameter_options_t) :: options
    type(gp_hyperparameter_result_t) :: result
    type(kernel_t) :: kernel
    type(fortnum_status_t) :: status
    real(dp) :: x(4, 1), y(4, 1), initial_lml, final_lml, gradient(3)
    integer :: i

    do i = 1, 4
        x(i, 1) = real(i - 2, dp)*0.45_dp
        y(i, 1) = exp(-0.5_dp*x(i, 1)**2) + 0.04_dp*x(i, 1)
    end do
    kernel = make_rbf_kernel(1, 0.5_dp, 1.5_dp, status)
    call model%fit(x, [0, 1, 0, 1], y, kernel, 0.25_dp, status)
    call model%log_marginal_likelihood(initial_lml, status)

    options%max_iterations = 80
    options%gradient_tolerance = 2.0e-2_dp
    options%lower_bound = -6.0_dp
    options%upper_bound = 6.0_dp
    call gp_optimize_derivative_hyperparameters(model, options, result, status)
    call model%log_marginal_likelihood(final_lml, status)
    call model%hyperparameter_gradient(gradient, status)

    if (.not. status_ok(status) .or. .not. result%converged .or. &
        final_lml < initial_lml .or. maxval(abs(gradient)) > 2.0e-2_dp .or. &
        abs(result%negative_log_marginal_likelihood + final_lml) > 2.0e-10_dp) then
        write (error_unit, '(a,4es14.5)') &
            "FAIL [derivative GP training] lml/gradient=", &
            initial_lml, final_lml, maxval(abs(gradient)), &
            result%negative_log_marginal_likelihood
        error stop 1
    end if

    options%lower_bound = 2.0_dp
    options%upper_bound = -2.0_dp
    call gp_optimize_derivative_hyperparameters(model, options, result, status)
    if (status_ok(status)) then
        write (error_unit, '(a)') &
            "FAIL [derivative GP training] invalid bounds accepted"
        error stop 1
    end if
    write (*, '(a)') "PASS"
end program test_derivative_gp_training
