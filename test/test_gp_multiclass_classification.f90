program test_gp_multiclass_classification
    !! Independent behavior checks for the one-vs-rest GP classifier.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_gp_multiclass_classification, only: &
        gp_multiclass_classification_t, &
        gp_multiclass_classification_options_t, &
        gp_multiclass_classification_state_t
    use fortml_kernels, only: kernel_t, make_rbf_kernel
    use fortml_kernels, only: clone_kernel
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    type(gp_multiclass_classification_t) :: model, repeat_model, probit_model, unfitted
    type(gp_multiclass_classification_t) :: model_plus, model_minus
    type(gp_multiclass_classification_options_t) :: options, probit_options
    type(gp_multiclass_classification_state_t) :: state, repeat_state
    type(gp_multiclass_classification_state_t) :: state_plus, state_minus
    type(fortnum_status_t) :: status
    type(kernel_t) :: kernel, kernel_plus, kernel_minus
    real(dp) :: x(9, 2), query(6, 2), query_dot(6, 2), query_plus(6, 2)
    real(dp) :: query_minus(6, 2), probabilities(6, 3), probabilities_dot(6, 3)
    real(dp) :: probabilities_plus(6, 3), probabilities_minus(6, 3)
    real(dp) :: margins(6, 3), margins_dot(6, 3), margins_plus(6, 3), &
        margins_minus(6, 3)
    real(dp) :: margins_bar(6, 3), margins_x_bar(6, 2), probabilities_bar(6, 3), &
        probabilities_x_bar(6, 2)
    real(dp) :: repeat_probabilities(6, 3), probit_probabilities(6, 3)
    real(dp), allocatable :: kernel_parameters(:), model_parameters(:), gradient(:)
    real(dp), allocatable :: gradient_fd(:), theta_plus(:), theta_minus(:)
    integer :: labels(9), predicted(9), query_predicted(6), classes(3)
    integer :: failures, k, class_index

    x(1, :) = [-0.1_dp, 1.9_dp]
    x(2, :) = [0.1_dp, 2.1_dp]
    x(3, :) = [0.2_dp, 1.8_dp]
    x(4, :) = [-0.1_dp, -0.1_dp]
    x(5, :) = [0.1_dp, 0.2_dp]
    x(6, :) = [0.3_dp, 0.0_dp]
    x(7, :) = [1.9_dp, 0.0_dp]
    x(8, :) = [2.1_dp, 0.2_dp]
    x(9, :) = [1.8_dp, 0.3_dp]
    labels = [42, 42, 42, -7, -7, -7, 10, 10, 10]
    query(1, :) = [0.0_dp, 2.0_dp]
    query(2, :) = [0.0_dp, 0.0_dp]
    query(3, :) = [2.0_dp, 0.0_dp]
    query(4, :) = [0.2_dp, 1.0_dp]
    query(5, :) = [1.0_dp, 0.2_dp]
    query(6, :) = [0.5_dp, 0.5_dp]
    query_dot = 0.0_dp
    query_dot(:, 1) = [0.2_dp, -0.1_dp, 0.3_dp, 0.0_dp, -0.2_dp, 0.1_dp]
    query_dot(:, 2) = [-0.1_dp, 0.2_dp, 0.1_dp, -0.3_dp, 0.2_dp, 0.0_dp]
    failures = 0

    kernel = make_rbf_kernel(2, 1.5_dp, 0.55_dp, status)
    options%max_iterations = 100
    options%tolerance = 1.0e-9_dp
    options%jitter = 1.0e-7_dp
    call model%fit(x, labels, kernel, status, options, state)
    call check(status_ok(status) .and. state%converged .and. model%fitted(), &
        "one-vs-rest fit", failures)
    classes = model%classes()
    call check(all(classes == [-7, 10, 42]) .and. model%class_count() == 3 .and. &
        model%feature_count() == 2, "sorted class metadata", failures)
    kernel_parameters = kernel%parameters()
    model_parameters = model%parameters()
    allocate(gradient(model%parameter_count()))
    call check(model%parameter_count() == 3*size(kernel_parameters) .and. &
        size(model_parameters) == model%parameter_count(), &
        "multiclass packed kernel metadata", failures)
    call model%hyperparameter_gradient(gradient, status)
    call check(status_ok(status), "multiclass hyperparameter gradient status", failures)
    ! The fitted wrapper stores one independent binary parameter block per
    ! sorted class.  Perturbing the single constructor kernel probes all
    ! blocks together; the oracle therefore compares against the sum of the
    ! corresponding packed gradient blocks.
    allocate(gradient_fd(size(kernel_parameters)))
    allocate(theta_plus(size(kernel_parameters)), theta_minus(size(kernel_parameters)))
    do k = 1, size(kernel_parameters)
        theta_plus = kernel_parameters
        theta_minus = kernel_parameters
        theta_plus(k) = theta_plus(k) + 1.0e-5_dp
        theta_minus(k) = theta_minus(k) - 1.0e-5_dp
        kernel_plus = clone_kernel(kernel)
        kernel_minus = clone_kernel(kernel)
        call kernel_plus%set_parameters(theta_plus, status)
        call check(status_ok(status), "multiclass positive probe setup", failures)
        call kernel_minus%set_parameters(theta_minus, status)
        call check(status_ok(status), "multiclass negative probe setup", failures)
        call model_plus%fit(x, labels, kernel_plus, status, options, state_plus)
        call check(status_ok(status) .and. state_plus%converged, &
            "multiclass positive probe fit", failures)
        call model_minus%fit(x, labels, kernel_minus, status, options, state_minus)
        call check(status_ok(status) .and. state_minus%converged, &
            "multiclass negative probe fit", failures)
        gradient_fd(k) = 0.0_dp
        do class_index = 1, model%class_count()
            gradient_fd(k) = gradient_fd(k) + gradient( &
                (class_index - 1)*size(kernel_parameters) + k)
        end do
        call check(abs(gradient_fd(k) - (state_plus%log_posterior - &
            state_minus%log_posterior)/(2.0e-5_dp)) < 2.0e-5_dp, &
            "multiclass hyperparameter gradient finite difference", failures)
    end do
    call model%predict_proba(query, probabilities, status)
    call model%predict(query, query_predicted, status)
    call check(status_ok(status) .and. &
        maxval(abs(sum(probabilities, dim=2) - 1.0_dp)) < 2.0e-14_dp .and. &
        all(probabilities >= 0.0_dp), "normalized probability simplex", failures)
    call check(query_predicted(1) == 42 .and. query_predicted(2) == -7 .and. &
        query_predicted(3) == 10, "class ranking at separated centers", failures)

    call model%predict(x, predicted, status)
    call check(status_ok(status) .and. count(predicted == labels) >= 8, &
        "one-vs-rest training behavior", failures)

    call model%predict_proba_jvp(query, query_dot, probabilities, &
        probabilities_dot, status)
    query_plus = query + 1.0e-5_dp*query_dot
    query_minus = query - 1.0e-5_dp*query_dot
    call model%predict_proba(query_plus, probabilities_plus, status)
    call model%predict_proba(query_minus, probabilities_minus, status)
    call check(status_ok(status) .and. maxval(abs(probabilities_dot - &
        (probabilities_plus - probabilities_minus)/(2.0e-5_dp))) < 3.0e-6_dp, &
        "normalized probability JVP finite difference", failures)

    call model%decision_function(query, margins, status)
    call model%decision_function_jvp(query, query_dot, margins, margins_dot, status)
    query_plus = query + 1.0e-5_dp*query_dot
    query_minus = query - 1.0e-5_dp*query_dot
    call model%decision_function(query_plus, margins_plus, status)
    call model%decision_function(query_minus, margins_minus, status)
    call check(status_ok(status) .and. maxval(abs(margins_dot - &
        (margins_plus - margins_minus)/(2.0e-5_dp))) < 3.0e-6_dp, &
        "latent decision-function JVP finite difference", failures)
    margins_bar = 0.0_dp
    margins_bar(:, 1) = [0.2_dp, -0.4_dp, 0.7_dp, -0.1_dp, 0.5_dp, -0.3_dp]
    margins_bar(:, 2) = [-0.6_dp, 0.1_dp, 0.2_dp, 0.8_dp, -0.2_dp, 0.4_dp]
    margins_bar(:, 3) = [0.3_dp, 0.5_dp, -0.3_dp, 0.2_dp, 0.6_dp, -0.7_dp]
    call model%decision_function_vjp(query, margins_bar, margins_x_bar, status)
    call check(status_ok(status) .and. abs(sum(margins_x_bar*query_dot) - &
        sum(margins_bar*margins_dot)) < 4.0e-6_dp, &
        "latent decision-function VJP dot-product identity", failures)
    probabilities_bar(:, 1) = [0.2_dp, -0.4_dp, 0.7_dp, -0.1_dp, 0.5_dp, -0.3_dp]
    probabilities_bar(:, 2) = [-0.6_dp, 0.1_dp, 0.2_dp, 0.8_dp, -0.2_dp, 0.4_dp]
    probabilities_bar(:, 3) = [0.3_dp, 0.5_dp, -0.3_dp, 0.2_dp, 0.6_dp, -0.7_dp]
    call model%predict_proba_vjp(query, probabilities_bar, probabilities_x_bar, &
        status)
    call check(status_ok(status) .and. abs(sum(probabilities_x_bar*query_dot) - &
        sum(probabilities_bar*probabilities_dot)) < 4.0e-6_dp, &
        "multiclass probability VJP dot-product identity", failures)

    call repeat_model%fit(x, labels, kernel, status, options, repeat_state)
    call repeat_model%predict_proba(query, repeat_probabilities, status)
    call check(status_ok(status) .and. repeat_state%converged .and. &
        maxval(abs(repeat_probabilities - probabilities)) < 2.0e-14_dp, &
        "deterministic one-vs-rest fit", failures)

    probit_options = options
    probit_options%likelihood = 2
    call probit_model%fit(x, labels, kernel, status, probit_options)
    call probit_model%predict_proba(query, probit_probabilities, status)
    call check(status_ok(status) .and. maxval(abs(sum(probit_probabilities, dim=2) - &
        1.0_dp)) < 2.0e-14_dp, "probit one-vs-rest fit", failures)

    call unfitted%predict_proba(query, probabilities, status)
    call check(.not. status_ok(status), "unfitted prediction refusal", failures)
    call unfitted%decision_function(query, margins, status)
    call check(.not. status_ok(status), "unfitted decision-function refusal", failures)
    call model%predict(query, predicted(:5), status)
    call check(.not. status_ok(status), "prediction label shape refusal", failures)
    call model%fit(x, [1, 1, 1, 1, 1, 1, 1, 1, 1], kernel, status, options)
    call check(.not. status_ok(status), "one-class fit refusal", failures)
    options%likelihood = 99
    call model%fit(x, labels, kernel, status, options)
    call check(.not. status_ok(status), "invalid likelihood refusal", failures)

    if (failures > 0) then
        write (error_unit, '(a,i0)') &
            "FAIL GP multiclass classification cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS GP multiclass classification independent oracles"

contains

    subroutine check(condition, name, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: name
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "  FAIL [gp-multiclass] "//name
        end if
    end subroutine check

end program test_gp_multiclass_classification
