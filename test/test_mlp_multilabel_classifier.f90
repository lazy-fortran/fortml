program test_mlp_multilabel_classifier
    !! Independent behavioral oracle for the shared multilabel MLP head.
    use, intrinsic :: iso_fortran_env, only: dp => real64
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    use fortml_mlp_multilabel_classifier, only: &
        mlp_multilabel_classifier_t, mlp_multilabel_classifier_options_t, &
        mlp_multilabel_classifier_state_t
    implicit none

    type(mlp_multilabel_classifier_t) :: model, repeat_model, unfitted
    type(mlp_multilabel_classifier_options_t) :: options
    type(mlp_multilabel_classifier_state_t) :: state, repeat_state
    type(fortml_device_t) :: cpu, cuda
    type(fortnum_status_t) :: status
    real(dp) :: x(8, 2), x_dot(8, 2), scores(8, 2), scores_dot(8, 2)
    real(dp) :: scores_plus(8, 2), scores_minus(8, 2)
    real(dp) :: probabilities(8, 2), probabilities_dot(8, 2)
    real(dp) :: probabilities_plus(8, 2), probabilities_minus(8, 2)
    real(dp) :: probabilities_bar(8, 2), x_bar(8, 2)
    real(dp), allocatable :: theta(:), theta_dot(:), theta_plus(:), theta_minus(:)
    real(dp), allocatable :: theta_bar(:), gradient(:), gradient_plus(:), gradient_minus(:), hvp(:)
    real(dp) :: value, value_plus, value_minus, lhs, rhs, h, hvp_error
    real(dp) :: thresholds(2)
    integer :: indicators(8, 2), predicted(8, 2), failures, i

    x = reshape([ &
        -2.0_dp, -1.0_dp, -1.0_dp, -2.0_dp, &
         2.0_dp,  1.0_dp,  1.0_dp,  2.0_dp, &
        -1.0_dp,  2.0_dp, -2.0_dp,  1.0_dp, &
         1.0_dp, -2.0_dp,  2.0_dp, -1.0_dp], shape(x))
    indicators = reshape([ &
        0, 0, 0, 0, 1, 1, 1, 1, &
        1, 1, 0, 0, 0, 0, 1, 1], shape(indicators))
    x_dot = reshape([(0.007_dp*real(i, dp), i=1, size(x))], shape(x))
    probabilities_bar = reshape([(0.011_dp*real(i, dp), &
        i=1, size(probabilities))], shape(probabilities))
    failures = 0
    h = 1.0e-6_dp

    call unfitted%predict(x, predicted, status)
    call check(.not. status_ok(status), "unfitted prediction refusal", failures)

    options%max_epochs = 500
    options%learning_rate = 0.04_dp
    options%l2 = 1.0e-4_dp
    options%tolerance = 1.0e-8_dp
    options%initialization_seed = 13
    options%restore_best = .false.
    call model%fit(x, indicators, status, hidden_layer_sizes=[4], options=options, &
        state=state)
    call check(status_ok(status) .and. model%fitted(), "multilabel MLP fit", failures)
    call check(state%final_loss < state%initial_loss, "multilabel loss decreases", failures)
    call model%decision_function(x, scores, status)
    call model%predict_proba(x, probabilities, status)
    call model%predict(x, predicted, status)
    call check(status_ok(status) .and. all(probabilities > 0.0_dp) .and. &
        all(probabilities < 1.0_dp), "multilabel probability range", failures)
    call check(count(predicted == indicators) >= 12, &
        "multilabel threshold behavior", failures)

    theta = model%parameters()
    allocate(theta_dot(size(theta)), theta_plus(size(theta)), theta_minus(size(theta)))
    allocate(theta_bar(size(theta)), gradient(size(theta)), hvp(size(theta)))
    allocate(gradient_plus(size(theta)), gradient_minus(size(theta)))
    theta_dot = [(0.013_dp*real(i, dp), i=1, size(theta))]
    call model%decision_function_jvp(x, theta_dot, x_dot, scores, scores_dot, status)
    theta_plus = theta + h*theta_dot
    theta_minus = theta - h*theta_dot
    call model%set_parameters(theta_plus, status)
    call model%decision_function(x+h*x_dot, scores_plus, status)
    call model%set_parameters(theta_minus, status)
    call model%decision_function(x-h*x_dot, scores_minus, status)
    call model%set_parameters(theta, status)
    call check(status_ok(status) .and. maxval(abs(scores_dot - &
        (scores_plus-scores_minus)/(2.0_dp*h))) < 2.0e-5_dp, &
        "multilabel decision JVP finite difference", failures)

    call model%predict_proba_jvp(x, theta_dot, x_dot, probabilities, &
        probabilities_dot, status)
    call model%set_parameters(theta_plus, status)
    call model%predict_proba(x+h*x_dot, probabilities_plus, status)
    call model%set_parameters(theta_minus, status)
    call model%predict_proba(x-h*x_dot, probabilities_minus, status)
    call model%set_parameters(theta, status)
    call check(status_ok(status) .and. maxval(abs(probabilities_dot - &
        (probabilities_plus-probabilities_minus)/(2.0_dp*h))) < 2.0e-5_dp, &
        "multilabel probability JVP finite difference", failures)

    call model%predict_proba_vjp(x, probabilities_bar, theta_bar, x_bar, status)
    lhs = sum(probabilities_bar*probabilities_dot)
    rhs = dot_product(theta_bar, theta_dot) + sum(x_bar*x_dot)
    call check(status_ok(status) .and. abs(lhs-rhs) < 3.0e-9_dp, &
        "multilabel probability VJP duality", failures)

    call model%loss_gradient(x, indicators, options%l2, value, gradient, status)
    theta_plus = theta
    theta_plus(1) = theta_plus(1) + h
    call model%set_parameters(theta_plus, status)
    call model%loss_gradient(x, indicators, options%l2, value_plus, gradient_plus, status)
    theta_minus = theta
    theta_minus(1) = theta_minus(1) - h
    call model%set_parameters(theta_minus, status)
    call model%loss_gradient(x, indicators, options%l2, value_minus, gradient_minus, status)
    call model%set_parameters(theta, status)
    call check(status_ok(status) .and. abs(gradient(1) - &
        (value_plus-value_minus)/(2.0_dp*h)) < 5.0e-6_dp, &
        "multilabel BCE gradient finite difference", failures)

    call model%loss_hvp(x, indicators, options%l2, theta_dot, hvp, status)
    theta_plus = theta + h*theta_dot
    theta_minus = theta - h*theta_dot
    call model%set_parameters(theta_plus, status)
    call model%loss_gradient(x, indicators, options%l2, value_plus, gradient_plus, status)
    call model%set_parameters(theta_minus, status)
    call model%loss_gradient(x, indicators, options%l2, value_minus, gradient_minus, status)
    call model%set_parameters(theta, status)
    hvp_error = maxval(abs(hvp-(gradient_plus-gradient_minus)/(2.0_dp*h)))
    call check(status_ok(status) .and. hvp_error < 2.0e-4_dp, &
        "multilabel BCE HVP finite difference", failures)

    thresholds = [0.6_dp, 0.4_dp]
    call model%set_thresholds(thresholds, status)
    call model%predict(x, predicted, status)
    call check(status_ok(status) .and. maxval(abs(model%thresholds()-thresholds)) < 1.0e-14_dp, &
        "per-label thresholds", failures)

    call cpu%select(FORTML_DEVICE_CPU, status)
    call model%predict_proba_device(cpu, x, probabilities_plus, status)
    call check(status_ok(status), "multilabel CPU device probability", failures)
    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call model%predict_proba_device(cuda, x, probabilities, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED .and. &
        .not. model%device_supported(FORTML_DEVICE_CUDA), &
        "multilabel CUDA probability refusal", failures)

    call repeat_model%fit(x, indicators, status, hidden_layer_sizes=[4], options=options, &
        state=repeat_state)
    call check(status_ok(status) .and. maxval(abs(model%parameters() - &
        repeat_model%parameters())) < 1.0e-14_dp, "deterministic full-batch Adam", failures)

    if (failures > 0) then
        write (*, '(a,i0)') "FAIL MLP multilabel cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS MLP multilabel behavioral oracle"

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

end program test_mlp_multilabel_classifier
