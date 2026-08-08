program test_gp_variational_classification_training
    !! Independent convergence and finite-difference oracle for the bounded
    !! variational-GP FortOpt adapter.
    use, intrinsic :: iso_fortran_env, only: real64, error_unit
    use fortml_gp_variational_classification, only: gp_variational_classification_t
    use fortml_gp_variational_classification_training, only: &
        gp_variational_classification_lbfgsb_options_t, &
        gp_variational_classification_lbfgsb_result_t, &
        gp_variational_classification_optimize
    use fortml_kernels, only: kernel_t, make_rbf_kernel
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    implicit none

    type(gp_variational_classification_t) :: model, cuda_model
    type(gp_variational_classification_lbfgsb_options_t) :: options
    type(gp_variational_classification_lbfgsb_result_t) :: result
    type(kernel_t) :: kernel
    type(fortnum_status_t) :: status
    type(fortml_device_t) :: device
    real(real64) :: inducing(2, 1), x(4, 1), initial_parameters(5)
    real(real64), allocatable :: final_parameters(:), gradient(:), parameters_plus(:)
    real(real64) :: value, value_plus, value_minus, finite_difference, h
    real(real64) :: initial_elbo, max_error
    integer :: failures, i

    inducing(:, 1) = [-0.7_real64, 0.55_real64]
    x(:, 1) = [-0.9_real64, -0.2_real64, 0.35_real64, 1.0_real64]
    initial_parameters = [0.0_real64, 0.0_real64, -0.15_real64, 0.08_real64, &
        0.05_real64]
    failures = 0
    h = 2.0e-6_real64

    kernel = make_rbf_kernel(1, 1.4_real64, 0.8_real64, status)
    call check(status_ok(status), "RBF kernel constructor", failures)
    call model%initialize(inducing, kernel, 32, 20260808, status)
    call check(status_ok(status), "variational model initialization", failures)
    call model%set_parameters(initial_parameters, status)
    call check(status_ok(status), "initial packed state", failures)
    call model%elbo(x, [0, 0, 1, 1], initial_elbo, status)
    call check(status_ok(status), "initial ELBO", failures)

    options%max_iterations = 200
    options%max_line_search = 80
    options%gradient_tolerance = 2.0e-5_real64
    options%lower_bound = -8.0_real64
    options%upper_bound = 8.0_real64
    call gp_variational_classification_optimize(model, x, [0, 0, 1, 1], options, &
        result, status)
    call check(status_ok(status), "CPU L-BFGS-B status", failures)
    call check(result%converged, "CPU L-BFGS-B convergence", failures)
    call check(result%elbo > initial_elbo, "ELBO improves from initial state", failures)
    call check(result%gradient_norm < 3.0e-4_real64, &
        "final ELBO gradient norm", failures)
    final_parameters = model%parameters()
    call check(all(final_parameters >= options%lower_bound) .and. &
        all(final_parameters <= options%upper_bound), "packed bounds", failures)

    allocate(gradient(size(final_parameters)), parameters_plus(size(final_parameters)))
    call model%elbo_gradient(x, [0, 0, 1, 1], value, gradient, status)
    call check(status_ok(status), "final ELBO gradient", failures)
    max_error = 0.0_real64
    do i = 1, size(final_parameters)
        parameters_plus = final_parameters
        parameters_plus(i) = parameters_plus(i) + h
        call model%set_parameters(parameters_plus, status)
        call model%elbo(x, [0, 0, 1, 1], value_plus, status)
        parameters_plus(i) = final_parameters(i) - h
        call model%set_parameters(parameters_plus, status)
        call model%elbo(x, [0, 0, 1, 1], value_minus, status)
        finite_difference = (value_plus - value_minus)/(2.0_real64*h)
        max_error = max(max_error, abs(gradient(i) - finite_difference))
    end do
    call model%set_parameters(final_parameters, status)
    call check(max_error < 2.0e-5_real64, &
        "final ELBO gradient finite-difference oracle", failures)

    call cuda_model%initialize(inducing, kernel, 32, 20260808, status)
    call check(status_ok(status), "CUDA refusal model initialization", failures)
    device%kind = FORTML_DEVICE_CUDA
    device%selected = .true.
    device%available = .true.
    call gp_variational_classification_optimize(cuda_model, x, [0, 0, 1, 1], options, &
        result, status, device)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "CUDA L-BFGS-B typed refusal", failures)

    if (failures > 0) then
        write (error_unit, '(a,i0,a,es12.4)') &
            "FAIL variational GP training cases: ", failures, " max FD error=", max_error
        error stop 1
    end if
    write (*, '(a,es12.4,a,es12.4)') &
        "PASS variational GP L-BFGS-B oracle (ELBO=", result%elbo, &
        ", max FD error=", max_error

contains

    subroutine check(condition, description, failure_count)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failure_count
        if (.not. condition) then
            failure_count = failure_count + 1
            write (error_unit, '(a)') "  FAIL [variational-gp-training] "//description
        end if
    end subroutine check

end program test_gp_variational_classification_training
