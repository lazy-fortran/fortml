program test_gp_hyperparameter_training
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_gp_training, only: gp_hyperparameter_options_t, &
        gp_hyperparameter_result_t, gp_optimize_hyperparameters
    use fortml_gaussian_process, only: gp_regression_t
    use fortml_kernels, only: kernel_t, make_rbf_kernel
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    implicit none

    type(gp_regression_t) :: model
    type(gp_hyperparameter_options_t) :: options
    type(gp_hyperparameter_result_t) :: result, single_result, repeat_result
    type(kernel_t) :: kernel
    type(fortnum_status_t) :: status
    type(fortml_device_t) :: cuda
    real(dp) :: x(5, 1), y(5, 1), initial_lml, final_lml, gradient(3)
    real(dp) :: finite_difference(3), plus_lml, minus_lml, epsilon
    real(dp), allocatable :: best_parameters(:), repeat_parameters(:)
    integer :: i, j

    do i = 1, 5
        x(i, 1) = real(i - 3, dp)
        y(i, 1) = exp(-0.5_dp*x(i, 1)**2) + 0.05_dp*x(i, 1)
    end do
    kernel = make_rbf_kernel(1, 0.25_dp, 2.0_dp, status)
    call model%fit(x, y, kernel, 0.5_dp, status)
    call model%log_marginal_likelihood(initial_lml, status)

    options%max_iterations = 150
    options%gradient_tolerance = 5.0e-3_dp
    options%lower_bound = -8.0_dp
    options%upper_bound = 8.0_dp
    options%starts = 4
    options%seed = 1234
    call gp_optimize_hyperparameters(model, options, result, status)
    call model%log_marginal_likelihood(final_lml, status)
    call model%hyperparameter_gradient(gradient, status)

    if (.not. status_ok(status) .or. .not. result%converged .or. &
        final_lml < initial_lml .or. maxval(abs(gradient)) > 5.0e-3_dp .or. &
        abs(result%negative_log_marginal_likelihood + final_lml) > 2.0e-10_dp .or. &
        result%start_count /= 4 .or. result%successful_starts < 1 .or. &
        result%best_start < 1 .or. result%best_start > 4) then
        write (error_unit, '(a,4es14.5,3i6)') &
            "FAIL [gp hyperparameter training] lml/gradient=", &
            initial_lml, final_lml, maxval(abs(gradient)), &
            result%negative_log_marginal_likelihood, result%start_count, &
            result%successful_starts, result%best_start
        error stop 1
    end if

    ! The analytic likelihood product is checked against an independent dense
    ! central difference at the retained best state.
    best_parameters = model%parameters()
    epsilon = 1.0e-5_dp
    do j = 1, size(best_parameters)
        call model%set_parameters(best_parameters, status)
        best_parameters(j) = best_parameters(j) + epsilon
        call model%set_parameters(best_parameters, status)
        call model%log_marginal_likelihood(plus_lml, status)
        best_parameters(j) = best_parameters(j) - 2.0_dp*epsilon
        call model%set_parameters(best_parameters, status)
        call model%log_marginal_likelihood(minus_lml, status)
        finite_difference(j) = (plus_lml - minus_lml)/(2.0_dp*epsilon)
        best_parameters(j) = best_parameters(j) + epsilon
    end do
    call model%set_parameters(best_parameters, status)
    call model%hyperparameter_gradient(gradient, status)
    if (.not. status_ok(status) .or. maxval(abs(finite_difference - gradient)) > 2.0e-4_dp) then
        write (error_unit, '(a,es14.5)') "FAIL [gp hyperparameter gradient FD] error=", &
            maxval(abs(finite_difference - gradient))
        error stop 1
    end if

    ! Seeded starts and best-state retention must be reproducible.
    call gp_optimize_hyperparameters(model, options, repeat_result, status)
    repeat_parameters = model%parameters()
    if (.not. status_ok(status) .or. repeat_result%best_start /= result%best_start .or. &
        maxval(abs(repeat_parameters - best_parameters)) > 2.0e-10_dp .or. &
        abs(repeat_result%negative_log_marginal_likelihood - &
        result%negative_log_marginal_likelihood) > 2.0e-10_dp) then
        write (error_unit, '(a)') "FAIL [gp hyperparameter multistart reproducibility]"
        error stop 1
    end if

    options%starts = 1
    call gp_optimize_hyperparameters(model, options, single_result, status)
    if (.not. status_ok(status) .or. single_result%negative_log_marginal_likelihood < &
        result%negative_log_marginal_likelihood - 2.0e-7_dp) then
        write (error_unit, '(a)') "FAIL [gp hyperparameter best-state retention]"
        error stop 1
    end if

    cuda%selected = .true.
    cuda%available = .true.
    cuda%kind = FORTML_DEVICE_CUDA
    call gp_optimize_hyperparameters(model, options, repeat_result, status, cuda)
    if (status%code /= FORTNUM_NOT_IMPLEMENTED) then
        write (error_unit, '(a)') "FAIL [gp hyperparameter CUDA refusal]"
        error stop 1
    end if

    options%lower_bound = 2.0_dp
    options%upper_bound = -2.0_dp
    call gp_optimize_hyperparameters(model, options, result, status)
    if (status_ok(status)) then
        write (error_unit, '(a)') &
            "FAIL [gp hyperparameter training] invalid bounds accepted"
        error stop 1
    end if
    write (*, '(a)') "PASS"
end program test_gp_hyperparameter_training
