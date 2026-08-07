program test_mlp_classifier
    !! Independent behavioral checks for the multiclass MLP classifier.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_mlp_classifier, only: mlp_classifier_t, &
        mlp_classifier_options_t, mlp_classifier_state_t
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    type(mlp_classifier_t) :: first_model, second_model, weighted_model, &
        sample_weighted_model, combined_model, minibatch_model, unfitted
    type(mlp_classifier_options_t) :: options
    type(mlp_classifier_state_t) :: first_state, second_state
    type(fortnum_status_t) :: status
    real(dp) :: x(6, 3), probabilities(6, 3), scores(6, 3), wrong_scores(6, 2)
    real(dp) :: weighted_probabilities(6, 3), weighted_x(6, 3)
    real(dp) :: sample_weighted_probabilities(6, 3)
    real(dp) :: combined_probabilities(6, 3)
    real(dp) :: sample_weights(6), combined_weights(6)
    real(dp), allocatable :: theta(:), gradient(:), plus_gradient(:)
    real(dp) :: value, plus, minus, h
    integer :: labels(6), predicted(6), predicted_second(6), classes(3), failures

    x = reshape([ &
        2.0_dp, 0.0_dp, 0.0_dp, &
        1.0_dp, 0.0_dp, 0.0_dp, &
        0.0_dp, 2.0_dp, 0.0_dp, &
        0.0_dp, 1.0_dp, 0.0_dp, &
        0.0_dp, 0.0_dp, 2.0_dp, &
        0.0_dp, 0.0_dp, 1.0_dp], shape(x))
    labels = [10, 10, -7, -7, 42, 42]
    failures = 0

    call unfitted%predict(x, predicted, status)
    call check(.not. status_ok(status), "unfitted prediction refusal", failures)

    options%max_epochs = 500
    options%learning_rate = 0.05_dp
    options%l2 = 1.0e-3_dp
    options%tolerance = 1.0e-8_dp
    options%initialization_seed = 23
    options%shuffle = .true.
    options%shuffle_seed = 91
    options%restore_best = .false.
    call first_model%fit(x, labels, status, hidden_layer_sizes=[4], &
        options=options, state=first_state)
    call check(status_ok(status), "classifier fit", failures)
    call check(first_model%fitted() .and. first_state%final_loss < &
        first_state%initial_loss, "loss decreases", failures)

    call first_model%decision_function(x, scores, status)
    call first_model%predict_proba(x, probabilities, status)
    call first_model%predict(x, predicted, status)
    classes = first_model%classes()
    call check(status_ok(status), "classifier prediction APIs", failures)
    call check(all(classes == [-7, 10, 42]), "sorted arbitrary classes", failures)
    call check(maxval(abs(sum(probabilities, dim=2) - 1.0_dp)) < 1.0e-13_dp, &
        "probability normalization", failures)
    call check(count(predicted == labels) >= 5, "classification behavior", failures)
    call check(any(abs(scores) > 1.0e-12_dp), "nonconstant logits", failures)
    call first_model%decision_function(x, wrong_scores, status)
    call check(.not. status_ok(status), "decision shape refusal", failures)

    theta = first_model%parameters()
    allocate(gradient(size(theta)), plus_gradient(size(theta)))
    call first_model%loss_gradient(x, labels, 0.0_dp, value, gradient, status)
    ! The public model product is checked by finite differences through the
    ! parameter setter.  A linearized MLP output is not assumed here.
    h = 1.0e-6_dp
    theta(1) = theta(1) + h
    call first_model%set_parameters(theta, status)
    call first_model%loss_gradient(x, labels, 0.0_dp, plus, plus_gradient, status)
    theta(1) = theta(1) - 2.0_dp*h
    call first_model%set_parameters(theta, status)
    call first_model%loss_gradient(x, labels, 0.0_dp, minus, plus_gradient, status)
    theta(1) = theta(1) + h
    call first_model%set_parameters(theta, status)
    call check(status_ok(status) .and. abs(gradient(1) - (plus - minus)/(2.0_dp*h)) &
        < 3.0e-6_dp, "MLP cross-entropy parameter oracle", failures)

    call second_model%fit(x, labels, status, hidden_layer_sizes=[4], &
        options=options, state=second_state)
    call first_model%predict(x, predicted, status)
    call second_model%predict(x, predicted_second, status)
    call check(status_ok(status), "repeat classifier fit", failures)
    call check(maxval(abs(first_model%parameters() - second_model%parameters())) &
        < 1.0e-14_dp .and. all(predicted == predicted_second), &
        "deterministic Adam", failures)

    weighted_x = 0.0_dp
    options%max_epochs = 2000
    options%learning_rate = 0.05_dp
    options%l2 = 0.0_dp
    options%tolerance = 1.0e-8_dp
    options%restore_best = .false.
    call weighted_model%fit(weighted_x, labels, status, options=options, &
        class_weight=[1.0_dp, 2.0_dp, 4.0_dp])
    call weighted_model%predict_proba(weighted_x, weighted_probabilities, status)
    call check(status_ok(status) .and. maxval(abs(weighted_probabilities(:, 1) - &
        1.0_dp/7.0_dp)) < 2.0e-4_dp .and. &
        maxval(abs(weighted_probabilities(:, 2) - 2.0_dp/7.0_dp)) < 2.0e-4_dp .and. &
        maxval(abs(weighted_probabilities(:, 3) - 4.0_dp/7.0_dp)) < 2.0e-4_dp, &
        "class-weighted probability oracle", failures)

    sample_weights = [0.0_dp, 1.0_dp, 0.0_dp, 2.0_dp, 0.0_dp, 3.0_dp]
    call sample_weighted_model%fit(weighted_x, labels, status, options=options, &
        sample_weight=sample_weights)
    call sample_weighted_model%predict_proba(weighted_x, &
        sample_weighted_probabilities, status)
    call check(status_ok(status) .and. maxval(abs(sample_weighted_probabilities(:, 1) - &
        2.0_dp/6.0_dp)) < 2.0e-4_dp .and. &
        maxval(abs(sample_weighted_probabilities(:, 2) - 1.0_dp/6.0_dp)) < &
        2.0e-4_dp .and. maxval(abs(sample_weighted_probabilities(:, 3) - &
        3.0_dp/6.0_dp)) < 2.0e-4_dp, &
        "sample-weighted probability oracle", failures)

    combined_weights = sample_weights
    call combined_model%fit(weighted_x, labels, status, options=options, &
        sample_weight=combined_weights, class_weight=[1.0_dp, 2.0_dp, 4.0_dp])
    call combined_model%predict_proba(weighted_x, combined_probabilities, status)
    call check(status_ok(status) .and. maxval(abs(combined_probabilities(:, 1) - &
        1.0_dp/8.0_dp)) < 2.0e-4_dp .and. &
        maxval(abs(combined_probabilities(:, 2) - 1.0_dp/8.0_dp)) < &
        2.0e-4_dp .and. maxval(abs(combined_probabilities(:, 3) - &
        3.0_dp/4.0_dp)) < 2.0e-4_dp, &
        "sample/class-weight composition oracle", failures)

    call first_model%loss_gradient(x, labels, 0.0_dp, value, gradient, status, &
        sample_weight=sample_weights)
    theta = first_model%parameters()
    theta(1) = theta(1) + h
    call first_model%set_parameters(theta, status)
    call first_model%loss_gradient(x, labels, 0.0_dp, plus, plus_gradient, status, &
        sample_weight=sample_weights)
    theta(1) = theta(1) - 2.0_dp*h
    call first_model%set_parameters(theta, status)
    call first_model%loss_gradient(x, labels, 0.0_dp, minus, plus_gradient, status, &
        sample_weight=sample_weights)
    theta(1) = theta(1) + h
    call first_model%set_parameters(theta, status)
    call check(status_ok(status) .and. abs(gradient(1) - (plus - minus)/(2.0_dp*h)) &
        < 3.0e-6_dp, "sample-weighted cross-entropy parameter oracle", failures)

    options%batch_size = 2
    call minibatch_model%fit(weighted_x, labels, status, options=options, &
        sample_weight=[0.0_dp, 0.0_dp, 1.0_dp, 1.0_dp, 1.0_dp, 1.0_dp])
    call check(status_ok(status), "all-zero minibatch is skipped", failures)
    options%batch_size = 0

    call weighted_model%fit(weighted_x, labels, status, options=options, &
        class_weight=[1.0_dp, 2.0_dp])
    call check(.not. status_ok(status), "class-weight shape refusal", failures)
    call weighted_model%fit(weighted_x, labels, status, options=options, &
        sample_weight=[1.0_dp, 2.0_dp])
    call check(.not. status_ok(status), "sample-weight shape refusal", failures)
    call weighted_model%fit(weighted_x, labels, status, options=options, &
        sample_weight=[1.0_dp, 1.0_dp, -1.0_dp, 1.0_dp, 1.0_dp, 1.0_dp])
    call check(.not. status_ok(status), "negative sample-weight refusal", failures)
    call weighted_model%fit(weighted_x, labels, status, options=options, &
        sample_weight=0.0_dp*sample_weights)
    call check(.not. status_ok(status), "zero effective sample-weight refusal", failures)

    call first_model%fit(x, [1, 1, 1, 1, 1, 1], status, options=options)
    call check(.not. status_ok(status), "one-class refusal", failures)
    call first_model%fit(x, labels(:5), status, options=options)
    call check(.not. status_ok(status), "label/sample shape refusal", failures)
    call first_model%fit(x, labels, status, hidden_layer_sizes=[0], options=options)
    call check(.not. status_ok(status), "invalid hidden width refusal", failures)

    if (failures > 0) then
        write (error_unit, '(a,i0)') "FAIL MLP classifier cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS MLP classifier independent behavioral oracles"

contains

    subroutine check(condition, name, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: name
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "  FAIL [mlp-classifier] "//name
        end if
    end subroutine check

end program test_mlp_classifier
