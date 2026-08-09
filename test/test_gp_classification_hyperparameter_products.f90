program test_gp_classification_hyperparameter_products
    !! Independent objective-product and device-boundary oracle for Laplace GP
    !! classification hyperparameters.
    !!
    !! The finite-difference oracle refits two independent classifiers and
    !! compares their converged mode log-posteriors with the analytic
    !! envelope JVP.  It therefore does not call the production gradient or
    !! JVP implementation while constructing the reference value.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    use fortml_kernels, only: kernel_t, make_rbf_kernel, clone_kernel
    use fortml_gp_classification, only: gp_classification_t, &
        gp_classification_options_t, gp_classification_state_t, &
        GP_LIKELIHOOD_LOGISTIC, GP_LIKELIHOOD_PROBIT
    implicit none

    type(gp_classification_t) :: model, model_plus, model_minus
    type(gp_classification_options_t) :: options
    type(gp_classification_state_t) :: state, state_plus, state_minus
    type(kernel_t) :: kernel, kernel_plus, kernel_minus
    type(fortml_device_t) :: cpu, cuda
    type(fortnum_status_t) :: status
    real(dp) :: x(8, 1), direction(2), gradient(2), vjp(2)
    real(dp) :: value, tangent, tangent_fd, value_plus, value_minus
    real(dp) :: tangent_device, value_device, h, cotangent
    real(dp), allocatable :: theta(:), theta_plus(:), theta_minus(:)
    integer :: labels(8), failures, likelihood

    x(:, 1) = [-1.5_dp, -1.0_dp, -0.5_dp, -0.1_dp, 0.1_dp, 0.5_dp, 1.0_dp, 1.5_dp]
    labels = [-7, -7, -7, -7, 11, 11, 11, 11]
    direction = [0.13_dp, -0.21_dp]
    cotangent = -1.7_dp
    h = 2.0e-4_dp
    failures = 0
    options%max_iterations = 100
    options%tolerance = 1.0e-10_dp
    options%jitter = 1.0e-7_dp
    kernel = make_rbf_kernel(1, 1.4_dp, 0.7_dp, status)
    call check(status_ok(status), "RBF setup", failures)
    theta = kernel%parameters()
    allocate(theta_plus(size(theta)), theta_minus(size(theta)))

    do likelihood = GP_LIKELIHOOD_LOGISTIC, GP_LIKELIHOOD_PROBIT
        options%likelihood = likelihood
        call model%fit(x, labels, kernel, status, options, state)
        call check(status_ok(status), "binary Laplace fit", failures)
        call check(state%converged, "binary Laplace convergence", failures)
        call model%hyperparameter_gradient(gradient, status)
        call check(status_ok(status), "hyperparameter gradient", failures)
        call model%hyperparameter_jvp(direction, value, tangent, status)
        call check(status_ok(status), "hyperparameter JVP", failures)
        call check(abs(tangent - dot_product(gradient, direction)) < 1.0e-12_dp, &
            "JVP/gradient contraction", failures)
        call model%hyperparameter_vjp(cotangent, vjp, status)
        call check(status_ok(status), "hyperparameter VJP", failures)
        call check(maxval(abs(vjp - cotangent*gradient)) < 1.0e-12_dp, &
            "VJP/gradient contraction", failures)
        call check(abs(value - state%log_posterior) < 1.0e-10_dp, &
            "JVP value is the converged mode objective", failures)

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
        value_plus = state_plus%log_posterior
        value_minus = state_minus%log_posterior
        tangent_fd = (value_plus - value_minus)/(2.0_dp*h)
        call check(abs(tangent - tangent_fd) < 3.0e-4_dp, &
            "JVP independent refit finite difference", failures)
    end do

    cpu%kind = FORTML_DEVICE_CPU
    cpu%selected = .true.
    cpu%available = .true.
    call model%hyperparameter_jvp_device(cpu, direction, value_device, tangent_device, status)
    call check(status_ok(status), "CPU JVP dispatch", failures)
    call check(abs(value_device - value) < 1.0e-12_dp .and. &
        abs(tangent_device - tangent) < 1.0e-12_dp, &
        "CPU JVP dispatch result", failures)
    call model%hyperparameter_vjp_device(cpu, cotangent, vjp, status)
    call check(status_ok(status), "CPU VJP dispatch", failures)
    call check(maxval(abs(vjp - cotangent*gradient)) < 1.0e-12_dp, &
        "CPU VJP dispatch result", failures)

    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call model%hyperparameter_jvp_device(cuda, direction, value_device, tangent_device, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, "CUDA JVP refusal", failures)
    call model%hyperparameter_vjp_device(cuda, cotangent, vjp, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, "CUDA VJP refusal", failures)

    if (failures > 0) then
        write (error_unit, '(a,i0)') "FAIL GP classification hyperparameter products: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS GP classification hyperparameter JVP/VJP independent oracle"

contains

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "  FAIL [gp-classification-products] "//description
        end if
    end subroutine check

end program test_gp_classification_hyperparameter_products
