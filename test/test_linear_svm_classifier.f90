program test_linear_svm_classifier
    !! Independent behavior and derivative checks for the linear SVM slice.
    use, intrinsic :: iso_fortran_env, only: real64, error_unit
    use fortml_linear_svm_classifier, only: linear_svm_classifier_t, &
        SVM_LOSS_HINGE, SVM_LOSS_SQUARED_HINGE
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    implicit none

    type(linear_svm_classifier_t) :: model, weighted_model, hinge_model
    type(fortnum_status_t) :: status
    type(fortml_device_t) :: cuda
    real(real64) :: x(6, 2), query(4, 2), x_dot(4, 2)
    real(real64) :: scores(4), scores_dot(4), scores_plus(4), scores_minus(4)
    real(real64) :: scores_bar(4), theta_dot(3), theta_bar(3), x_bar(4, 2)
    real(real64) :: value, value_plus, value_minus, l2_gradient
    real(real64) :: gradient(3), gradient_fd(3), parameters(3), parameters_plus(3)
    real(real64) :: parameters_minus(3), prediction_scores(6)
    integer :: labels(6), predicted(6), device_labels(4), classes(2), failures, i
    real(real64), parameter :: step = 1.0e-6_real64

    failures = 0
    x(1, :) = [-2.0_real64, -1.0_real64]
    x(2, :) = [-1.5_real64, -0.5_real64]
    x(3, :) = [-1.0_real64, -1.0_real64]
    x(4, :) = [1.0_real64, 1.0_real64]
    x(5, :) = [1.5_real64, 0.5_real64]
    x(6, :) = [2.0_real64, 1.0_real64]
    labels = [41, 41, 41, -8, -8, -8]
    call model%fit(x, labels, status, l2=0.2_real64, &
        loss=SVM_LOSS_SQUARED_HINGE, max_iterations=1000, tolerance=1.0e-8_real64)
    call check(status_ok(status) .and. model%fitted(), &
        "squared-hinge fit", failures)
    if (.not. status_ok(status)) error stop 1

    classes = model%classes()
    call model%decision_function(x, prediction_scores, status)
    call model%predict(x, predicted, status)
    call check(status_ok(status) .and. all(classes == [-8, 41]), &
        "arbitrary sorted integer classes", failures)
    call check(status_ok(status) .and. all(predicted(:3) == 41) .and. &
        all(predicted(4:) == -8), "separable training predictions", failures)
    call check(status_ok(status) .and. all(prediction_scores(:3) > 0.0_real64) .and. &
        all(prediction_scores(4:) < 0.0_real64), "signed decision margins", failures)
    call check(model%parameter_count() == 3 .and. size(model%parameters()) == 3, &
        "packed coefficient/intercept state", failures)

    query(1, :) = [-1.25_real64, -0.75_real64]
    query(2, :) = [-0.25_real64, -0.25_real64]
    query(3, :) = [0.25_real64, 0.25_real64]
    query(4, :) = [1.25_real64, 0.75_real64]
    x_dot = reshape([ &
        0.10_real64, -0.20_real64, &
        -0.30_real64,  0.40_real64, &
        0.20_real64,  0.30_real64, &
        -0.15_real64,  0.05_real64], shape(x_dot))
    theta_dot = [0.07_real64, -0.11_real64, 0.13_real64]
    call model%decision_function_jvp(query, theta_dot, x_dot, scores, scores_dot, status)
    parameters = model%parameters()
    parameters_plus = parameters + step*theta_dot
    call model%set_parameters(parameters_plus, status)
    call model%decision_function(query + step*x_dot, scores_plus, status)
    parameters_minus = parameters - step*theta_dot
    call model%set_parameters(parameters_minus, status)
    call model%decision_function(query - step*x_dot, scores_minus, status)
    call model%set_parameters(parameters, status)
    call check(status_ok(status) .and. maxval(abs(scores_dot - &
        (scores_plus - scores_minus)/(2.0_real64*step))) < 2.0e-8_real64, &
        "affine input/parameter JVP", failures)

    scores_bar = [0.2_real64, -0.1_real64, 0.4_real64, -0.3_real64]
    call model%decision_function_vjp(query, scores_bar, theta_bar, x_bar, status)
    call check(status_ok(status) .and. abs(sum(scores_bar*scores_dot) - &
        (sum(theta_bar*theta_dot) + sum(x_bar*x_dot))) < 2.0e-10_real64, &
        "affine input/parameter VJP adjoint", failures)

    parameters = model%parameters()
    call model%objective_value_gradient(x, labels, parameters, value, gradient, status, &
        l2=0.2_real64, loss=SVM_LOSS_SQUARED_HINGE, l2_gradient=l2_gradient)
    do i = 1, size(parameters)
        parameters_plus = parameters
        parameters_plus(i) = parameters_plus(i) + step
        call model%objective_value_gradient(x, labels, parameters_plus, value_plus, &
            gradient_fd, status, l2=0.2_real64, loss=SVM_LOSS_SQUARED_HINGE)
        parameters_minus = parameters
        parameters_minus(i) = parameters_minus(i) - step
        call model%objective_value_gradient(x, labels, parameters_minus, value_minus, &
            gradient_fd, status, l2=0.2_real64, loss=SVM_LOSS_SQUARED_HINGE)
        gradient_fd(i) = (value_plus - value_minus)/(2.0_real64*step)
    end do
    call check(status_ok(status) .and. maxval(abs(gradient-gradient_fd)) < 2.0e-6_real64, &
        "squared-hinge objective gradient oracle", failures)
    call check(abs(l2_gradient - 0.5_real64*sum(parameters(:2)**2)) < 2.0e-12_real64, &
        "L2 hyperparameter derivative", failures)

    block
        real(real64) :: split_x(2, 1), split_theta(2), split_gradient(2), split_value
        integer :: split_labels(2)
        split_x(:, 1) = [1.0_real64, 0.0_real64]
        split_labels = [42, -8]
        split_theta = [1.0_real64, 0.0_real64]
        call model%objective_value_gradient(split_x, split_labels, split_theta, &
            split_value, split_gradient, status, l2=0.0_real64, &
            fit_intercept=.true., loss=SVM_LOSS_HINGE)
        call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
            "ordinary-hinge exact split refusal", failures)
    end block

    call weighted_model%fit(x, labels, status, l2=0.1_real64, &
        sample_weight=[1.0_real64, 2.0_real64, 1.0_real64, 1.0_real64, &
            2.0_real64, 1.0_real64], max_iterations=1000)
    call check(status_ok(status), "weighted fit", failures)
    call hinge_model%fit(x, labels, status, l2=0.1_real64, loss=SVM_LOSS_HINGE, &
        max_iterations=1500, tolerance=1.0e-7_real64)
    call check(status_ok(status), "ordinary-hinge fit", failures)

    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call model%decision_function_device(cuda, query, scores, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "CUDA decision refusal", failures)
    call model%predict_device(cuda, query, device_labels, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "CUDA label refusal", failures)
    call check(model%device_supported(FORTML_DEVICE_CUDA) .eqv. .false., &
        "CUDA capability refusal", failures)

    if (failures > 0) then
        write (error_unit, '(a,i0)') "FAIL linear SVM cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS linear SVM independent behavioral oracles"

contains

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failures
        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "  FAIL [linear-svm] "//description
        end if
    end subroutine check

end program test_linear_svm_classifier
