program test_gp_classification_sample_weights
    !! Independent weighted Laplace-GP fit, envelope-gradient, and device gates.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_quiet_nan
    use fortml_gp_classification, only: gp_classification_t, &
        gp_classification_options_t, gp_classification_state_t, &
        GP_LIKELIHOOD_PROBIT
    use fortml_gp_multiclass_classification, only: &
        gp_multiclass_classification_t, gp_multiclass_classification_options_t, &
        gp_multiclass_classification_state_t
    use fortml_kernels, only: kernel_t, make_rbf_kernel, clone_kernel
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    implicit none

    type(gp_classification_t) :: model, unit_model, weighted_model, probe_plus, probe_minus
    type(gp_classification_t) :: binary
    type(gp_multiclass_classification_t) :: multiclass
    type(gp_classification_options_t) :: options
    type(gp_multiclass_classification_options_t) :: multiclass_options
    type(gp_classification_state_t) :: state, unit_state, weighted_state
    type(gp_classification_state_t) :: plus_state, minus_state, binary_state
    type(gp_multiclass_classification_state_t) :: multiclass_state
    type(kernel_t) :: kernel, kernel_plus, kernel_minus
    type(fortnum_status_t) :: status
    type(fortml_device_t) :: device
    real(dp) :: x(8, 1), x_query(5, 1), weights(8), unit_weights(8)
    real(dp) :: probabilities(5, 2), probabilities_cpu(5, 2)
    real(dp), allocatable :: parameters(:), gradient(:), gradient_fd(:), shifted(:)
    integer :: labels(8), multiclass_labels(8), classes(3), failures, i, class_index
    real(dp) :: h, multiclass_reference, value
    real(dp) :: nonfinite

    x(:, 1) = [-1.5_dp, -1.0_dp, -0.5_dp, -0.1_dp, &
        0.1_dp, 0.5_dp, 1.0_dp, 1.5_dp]
    x_query(:, 1) = [-1.25_dp, -0.4_dp, 0.0_dp, 0.4_dp, 1.25_dp]
    labels = [-7, -7, -7, -7, 11, 11, 11, 11]
    weights = [0.5_dp, 0.0_dp, 1.5_dp, 2.0_dp, 0.7_dp, 1.2_dp, 2.3_dp, 0.8_dp]
    unit_weights = 1.0_dp
    failures = 0
    nonfinite = ieee_value(0.0_dp, ieee_quiet_nan)
    kernel = make_rbf_kernel(1, 1.4_dp, 0.7_dp, status)
    call check(status_ok(status), "kernel constructor", failures)
    options%max_iterations = 100
    options%tolerance = 1.0e-9_dp
    options%jitter = 1.0e-7_dp

    call model%fit(x, labels, kernel, status, options, state, sample_weight=weights)
    call check(status_ok(status) .and. state%converged .and. model%fitted(), &
        "weighted binary Laplace fit", failures)
    call model%predict_proba(x_query, probabilities, status)
    call check(status_ok(status) .and. all(probabilities >= 0.0_dp) .and. &
        maxval(abs(sum(probabilities, dim=2) - 1.0_dp)) < 2.0e-14_dp, &
        "weighted binary probability simplex", failures)

    call unit_model%fit(x, labels, kernel, status, options, unit_state, &
        sample_weight=unit_weights)
    call check(status_ok(status) .and. unit_state%converged, &
        "unit-weight binary fit", failures)
    call weighted_model%fit(x, labels, kernel, status, options, weighted_state)
    call check(status_ok(status) .and. weighted_state%converged, &
        "implicit-unit binary fit", failures)
    call unit_model%predict_proba(x_query, probabilities_cpu, status)
    call weighted_model%predict_proba(x_query, probabilities, status)
    call check(maxval(abs(probabilities - probabilities_cpu)) < 2.0e-12_dp .and. &
        abs(unit_state%log_posterior - weighted_state%log_posterior) < 2.0e-12_dp, &
        "explicit unit weights preserve binary fit", failures)

    ! The envelope hyperparameter product must include the weighted likelihood
    ! through the fitted mode. Refit perturbed kernels independently.
    parameters = model%parameters()
    allocate(gradient(size(parameters)), gradient_fd(size(parameters)), shifted(size(parameters)))
    call model%hyperparameter_gradient(gradient, status)
    call check(status_ok(status), "weighted hyperparameter gradient status", failures)
    h = 1.0e-5_dp
    do i = 1, size(parameters)
        shifted = parameters
        shifted(i) = shifted(i) + h
        kernel_plus = clone_kernel(kernel)
        call kernel_plus%set_parameters(shifted, status)
        call probe_plus%fit(x, labels, kernel_plus, status, options, plus_state, &
            sample_weight=weights)
        shifted(i) = parameters(i) - h
        kernel_minus = clone_kernel(kernel)
        call kernel_minus%set_parameters(shifted, status)
        call probe_minus%fit(x, labels, kernel_minus, status, options, minus_state, &
            sample_weight=weights)
        gradient_fd(i) = (plus_state%log_posterior - minus_state%log_posterior)/(2.0_dp*h)
    end do
    call check(maxval(abs(gradient - gradient_fd)) < 3.0e-5_dp, &
        "weighted hyperparameter envelope finite difference", failures)

    options%likelihood = GP_LIKELIHOOD_PROBIT
    call weighted_model%fit(x, labels, kernel, status, options, weighted_state, &
        sample_weight=weights)
    call check(status_ok(status) .and. weighted_state%converged, &
        "weighted probit Laplace fit", failures)
    options%likelihood = 1

    call model%fit(x, labels, kernel, status, options, sample_weight=weights(:7))
    call check(.not. status_ok(status), "sample-weight shape refusal", failures)
    call model%fit(x, labels, kernel, status, options, &
        sample_weight=[1.0_dp, -1.0_dp, 1.0_dp, 1.0_dp, 1.0_dp, 1.0_dp, 1.0_dp, 1.0_dp])
    call check(.not. status_ok(status), "negative sample-weight refusal", failures)
    call model%fit(x, labels, kernel, status, options, &
        sample_weight=[0.0_dp, 0.0_dp, 0.0_dp, 0.0_dp, 0.0_dp, 0.0_dp, 0.0_dp, 0.0_dp])
    call check(.not. status_ok(status), "zero sample-weight mass refusal", failures)
    call model%fit(x, labels, kernel, status, options, &
        sample_weight=[1.0_dp, 1.0_dp, 1.0_dp, 1.0_dp, 1.0_dp, 1.0_dp, &
        1.0_dp, nonfinite])
    call check(.not. status_ok(status), "nonfinite sample-weight refusal", failures)

    device%kind = FORTML_DEVICE_CPU
    device%selected = .true.
    device%available = .true.
    call model%fit(x, labels, kernel, status, options, sample_weight=weights)
    call check(status_ok(status), "weighted fit for device gate", failures)
    call model%predict_proba_device(device, x_query, probabilities_cpu, status)
    call check(status_ok(status), "weighted CPU prediction dispatch", failures)
    call model%predict_proba(x_query, probabilities, status)
    call check(maxval(abs(probabilities_cpu - probabilities)) < 2.0e-12_dp, &
        "weighted CPU prediction value", failures)
    device%kind = FORTML_DEVICE_CUDA
    call model%predict_proba_device(device, x_query, probabilities_cpu, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "weighted CUDA prediction refusal", failures)

    ! Multiclass OVR fitting must use exactly the same weighted binary objective
    ! in sorted class order.
    classes = [10, 20, 30]
    multiclass_labels = [10, 10, 20, 20, 30, 30, 10, 30]
    multiclass_options%max_iterations = options%max_iterations
    multiclass_options%tolerance = options%tolerance
    multiclass_options%jitter = options%jitter
    call multiclass%fit(x, multiclass_labels, kernel, status, multiclass_options, &
        multiclass_state, sample_weight=weights)
    call check(status_ok(status) .and. multiclass_state%converged, &
        "weighted multiclass Laplace fit", failures)
    multiclass_reference = 0.0_dp
    do class_index = 1, size(classes)
        where (multiclass_labels == classes(class_index))
            labels = 1
        elsewhere
            labels = 0
        end where
        call binary%fit(x, labels, kernel, status, options, binary_state, &
            sample_weight=weights)
        call check(status_ok(status), "weighted binary OVR reference fit", failures)
        multiclass_reference = multiclass_reference + binary_state%log_posterior
    end do
    call check(abs(multiclass_state%log_posterior - multiclass_reference) < 3.0e-10_dp, &
        "weighted multiclass OVR composition", failures)

    if (failures > 0) then
        write (error_unit, '(a,i0)') "FAIL weighted GP classification cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS weighted GP classification independent oracles"

contains

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "  FAIL [weighted-gp-classification] "//description
        end if
    end subroutine check

end program test_gp_classification_sample_weights
