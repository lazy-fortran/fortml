program test_gp_classification_hvp
    !! Independent implicit-mode hyperparameter HVP and device-boundary checks.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortml_kernels, only: kernel_t, make_rbf_kernel, clone_kernel
    use fortml_gp_classification, only: gp_classification_t, &
        gp_classification_options_t, gp_classification_state_t, &
        GP_LIKELIHOOD_LOGISTIC, GP_LIKELIHOOD_PROBIT
    implicit none

    type(gp_classification_t) :: model, model_plus, model_minus
    type(gp_classification_options_t) :: options
    type(gp_classification_state_t) :: state_plus, state_minus
    type(kernel_t) :: kernel, kernel_plus, kernel_minus
    type(fortml_device_t) :: cuda
    type(fortnum_status_t) :: status
    real(dp) :: x(8, 1), direction(2), hvp(2), hvp_fd(2)
    real(dp) :: gradient_plus(2), gradient_minus(2)
    real(dp), allocatable :: theta(:), theta_plus(:), theta_minus(:)
    real(dp) :: h
    integer :: labels(8), failures, likelihood

    x(:, 1) = [-1.5_dp, -1.0_dp, -0.5_dp, -0.1_dp, 0.1_dp, 0.5_dp, 1.0_dp, 1.5_dp]
    labels = [-7, -7, -7, -7, 11, 11, 11, 11]
    direction = [0.13_dp, -0.21_dp]
    failures = 0
    h = 2.0e-4_dp
    options%max_iterations = 100
    options%tolerance = 1.0e-10_dp
    options%jitter = 1.0e-7_dp
    kernel = make_rbf_kernel(1, 1.4_dp, 0.7_dp, status)
    call check(status_ok(status), "RBF setup", failures)
    theta = kernel%parameters()
    allocate(theta_plus(size(theta)), theta_minus(size(theta)))

    do likelihood = GP_LIKELIHOOD_LOGISTIC, GP_LIKELIHOOD_PROBIT
        options%likelihood = likelihood
        call model%fit(x, labels, kernel, status, options)
        call check(status_ok(status) .and. model%fitted(), "binary Laplace fit", failures)
        call model%hyperparameter_hvp(direction, hvp, status)
        call check(status_ok(status), "hyperparameter HVP status", failures)
        theta_plus = theta + h*direction
        theta_minus = theta - h*direction
        kernel_plus = clone_kernel(kernel)
        kernel_minus = clone_kernel(kernel)
        call kernel_plus%set_parameters(theta_plus, status)
        call check(status_ok(status), "positive probe setup", failures)
        call model_plus%fit(x, labels, kernel_plus, status, options, state_plus)
        call check(status_ok(status) .and. state_plus%converged, &
            "positive probe fit", failures)
        call kernel_minus%set_parameters(theta_minus, status)
        call check(status_ok(status), "negative probe setup", failures)
        call model_minus%fit(x, labels, kernel_minus, status, options, state_minus)
        call check(status_ok(status) .and. state_minus%converged, &
            "negative probe fit", failures)
        call model_plus%hyperparameter_gradient(gradient_plus, status)
        call check(status_ok(status), "positive gradient", failures)
        call model_minus%hyperparameter_gradient(gradient_minus, status)
        call check(status_ok(status), "negative gradient", failures)
        hvp_fd = (gradient_plus - gradient_minus)/(2.0_dp*h)
        call check(status_ok(status) .and. maxval(abs(hvp-hvp_fd)) < 3.0e-4_dp, &
            "implicit-mode HVP finite difference", failures)
    end do

    ! A failed transactional update must not alter the fitted kernel state.
    theta_plus = theta
    theta_plus(1) = theta_plus(1) + 0.1_dp
    call model%set_parameters([theta_plus(1)], status)
    call check(.not. status_ok(status) .and. maxval(abs(model%parameters()-theta)) < 1.0e-14_dp, &
        "transactional parameter refusal", failures)

    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call model%hyperparameter_hvp_device(cuda, direction, hvp, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, "CUDA HVP refusal", failures)

    if (failures > 0) then
        write (error_unit, '(a,i0)') "FAIL GP classification HVP cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS GP classification implicit-mode HVP independent oracle"

contains

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "  FAIL [gp-classification-hvp] "//description
        end if
    end subroutine check

end program test_gp_classification_hvp
