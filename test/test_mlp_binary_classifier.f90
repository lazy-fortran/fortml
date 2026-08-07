program test_mlp_binary_classifier
    !! Independent behavioral oracle for the sigmoid MLP binary head.
    use, intrinsic :: iso_fortran_env, only: real64
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    use fortml_mlp_binary_classifier, only: mlp_binary_classifier_t, &
        mlp_binary_classifier_options_t, mlp_binary_classifier_state_t
    implicit none

    integer, parameter :: dp = real64
    type(mlp_binary_classifier_t) :: model, repeat_model, unfitted
    type(mlp_binary_classifier_options_t) :: options
    type(mlp_binary_classifier_state_t) :: state, repeat_state
    type(fortml_device_t) :: cpu, cuda
    type(fortnum_status_t) :: status
    real(dp) :: x(8, 2), scores(8), scores_dot(8), scores_plus(8), scores_minus(8)
    real(dp) :: probabilities(8, 2), probabilities_dot(8, 2)
    real(dp) :: probabilities_plus(8, 2), probabilities_minus(8, 2)
    real(dp), allocatable :: theta(:), theta_dot(:), theta_plus(:), theta_minus(:)
    real(dp), allocatable :: theta_bar(:), x_dot(:, :), x_bar(:, :), gradient(:)
    real(dp), allocatable :: hvp(:), gradient_plus(:), gradient_minus(:)
    real(dp) :: value, value_plus, value_minus, tangent_left, tangent_right
    real(dp) :: h, hvp_error
    integer :: labels(8), predicted(8), classes(2), failures, i

    x = reshape([ &
        -2.0_dp, -1.0_dp, -1.0_dp, -2.0_dp, &
         2.0_dp,  1.0_dp,  1.0_dp,  2.0_dp, &
        -1.0_dp,  2.0_dp, -2.0_dp,  1.0_dp, &
         1.0_dp, -2.0_dp,  2.0_dp, -1.0_dp], shape(x))
    labels = [-4, -4, -4, -4, 9, 9, 9, 9]
    failures = 0

    call unfitted%predict(x, predicted, status)
    call check(.not. status_ok(status), "unfitted prediction refusal", failures)

    options%max_epochs = 800
    options%learning_rate = 0.04_dp
    options%l2 = 1.0e-4_dp
    options%tolerance = 1.0e-8_dp
    options%initialization_seed = 13
    options%hidden_activation = 2
    options%restore_best = .false.
    call model%fit(x, labels, status, hidden_layer_sizes=[4], options=options, &
        state=state)
    call check(status_ok(status), "binary MLP fit", failures)
    call check(model%fitted() .and. state%final_loss < state%initial_loss, &
        "binary loss decreases", failures)
    classes = model%classes()
    call check(all(classes == [-4, 9]), "sorted binary classes", failures)

    call model%decision_function(x, scores, status)
    call model%predict_proba(x, probabilities, status)
    call model%predict(x, predicted, status)
    call check(status_ok(status), "binary prediction APIs", failures)
    call check(maxval(abs(sum(probabilities, dim=2) - 1.0_dp)) < 1.0e-13_dp, &
        "binary probability normalization", failures)
    call check(count(predicted == labels) >= 7, "binary classification behavior", failures)
    call check(any(abs(scores) > 1.0e-12_dp), "nonconstant binary logits", failures)

    theta = model%parameters()
    allocate(theta_dot(size(theta)), theta_plus(size(theta)), theta_minus(size(theta)))
    allocate(theta_bar(size(theta)), x_dot(size(x, 1), size(x, 2)), &
        x_bar(size(x, 1), size(x, 2)), gradient(size(theta)), hvp(size(theta)))
    allocate(gradient_plus(size(theta)), gradient_minus(size(theta)))
    theta_dot = [(0.013_dp*real(i, dp), i=1, size(theta))]
    x_dot = reshape([(0.007_dp*real(i, dp), i=1, size(x))], shape(x))
    h = 1.0e-6_dp

    call model%decision_function_jvp(x, theta_dot, x_dot, scores, scores_dot, status)
    theta_plus = theta + h*theta_dot
    theta_minus = theta - h*theta_dot
    call model%set_parameters(theta_plus, status)
    call model%decision_function(x + h*x_dot, scores_plus, status)
    call model%set_parameters(theta_minus, status)
    call model%decision_function(x - h*x_dot, scores_minus, status)
    call model%set_parameters(theta, status)
    call check(status_ok(status) .and. maxval(abs(scores_dot - &
        (scores_plus - scores_minus)/(2.0_dp*h))) < 5.0e-5_dp, &
        "binary decision JVP finite-difference oracle", failures)

    probabilities_dot = 0.0_dp
    call model%predict_proba_jvp(x, theta_dot, x_dot, probabilities, &
        probabilities_dot, status)
    call model%set_parameters(theta_plus, status)
    call model%predict_proba(x + h*x_dot, probabilities_plus, status)
    call model%set_parameters(theta_minus, status)
    call model%predict_proba(x - h*x_dot, probabilities_minus, status)
    call model%set_parameters(theta, status)
    call check(status_ok(status) .and. maxval(abs(probabilities_dot - &
        (probabilities_plus - probabilities_minus)/(2.0_dp*h))) < 5.0e-5_dp, &
        "binary probability JVP finite-difference oracle", failures)

    probabilities_plus = reshape([(0.011_dp*real(i, dp), i=1, size(probabilities))], &
        shape(probabilities))
    call model%decision_function_vjp(x, probabilities_plus(:, 2), theta_bar, &
        x_bar, status)
    tangent_left = sum(probabilities_plus(:, 2)*scores_dot)
    tangent_right = dot_product(theta_bar, theta_dot) + sum(x_bar*x_dot)
    call check(status_ok(status) .and. abs(tangent_left - tangent_right) < 2.0e-10_dp, &
        "binary decision VJP duality", failures)

    call model%predict_proba_vjp(x, probabilities_plus, theta_bar, x_bar, status)
    tangent_left = sum(probabilities_plus*probabilities_dot)
    tangent_right = dot_product(theta_bar, theta_dot) + sum(x_bar*x_dot)
    call check(status_ok(status) .and. abs(tangent_left - tangent_right) < 2.0e-10_dp, &
        "binary probability VJP duality", failures)

    call model%loss_gradient(x, labels, options%l2, value, gradient, status)
    theta_plus = theta
    theta_plus(1) = theta_plus(1) + h
    call model%set_parameters(theta_plus, status)
    call model%loss_gradient(x, labels, options%l2, value_plus, gradient_plus, status)
    theta_minus = theta
    theta_minus(1) = theta_minus(1) - h
    call model%set_parameters(theta_minus, status)
    call model%loss_gradient(x, labels, options%l2, value_minus, gradient_minus, status)
    call model%set_parameters(theta, status)
    call check(status_ok(status) .and. abs(gradient(1) - &
        (value_plus - value_minus)/(2.0_dp*h)) < 5.0e-6_dp, &
        "binary BCE parameter gradient oracle", failures)

    call model%loss_hvp(x, labels, options%l2, theta_dot, hvp, status)
    theta_plus = theta + h*theta_dot
    theta_minus = theta - h*theta_dot
    call model%set_parameters(theta_plus, status)
    call model%loss_gradient(x, labels, options%l2, value_plus, gradient_plus, status)
    call model%set_parameters(theta_minus, status)
    call model%loss_gradient(x, labels, options%l2, value_minus, gradient_minus, status)
    call model%set_parameters(theta, status)
    hvp_error = maxval(abs(hvp - (gradient_plus - gradient_minus)/(2.0_dp*h)))
    call check(status_ok(status) .and. hvp_error < 2.0e-4_dp, &
        "binary BCE HVP finite-difference oracle", failures)

    call cpu%select(FORTML_DEVICE_CPU, status)
    call model%predict_proba_device(cpu, x, probabilities, status)
    call check(status_ok(status), "binary CPU device probability", failures)
    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call model%predict_proba_device(cuda, x, probabilities, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "binary CUDA probability refusal", failures)
    call check(model%device_supported(FORTML_DEVICE_CPU) .and. &
        .not. model%device_supported(FORTML_DEVICE_CUDA), &
        "binary device capability contract", failures)

    call repeat_model%fit(x, labels, status, hidden_layer_sizes=[4], options=options, &
        state=repeat_state)
    call check(status_ok(status) .and. maxval(abs(model%parameters() - &
        repeat_model%parameters())) < 1.0e-14_dp, "deterministic binary Adam", failures)

    if (failures > 0) then
        write (*, '(a,i0)') "FAIL MLP binary-classifier cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS MLP binary-classifier behavioral oracle"

contains

    subroutine check(condition, label, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: label
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (*, '(a)') "FAIL: " // trim(label)
        end if
    end subroutine check

end program test_mlp_binary_classifier
