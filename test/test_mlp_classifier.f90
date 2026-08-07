program test_mlp_classifier
    !! Independent behavioral checks for the multiclass MLP classifier.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_mlp_classifier, only: mlp_classifier_t, &
        mlp_classifier_options_t, mlp_classifier_state_t
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    type(mlp_classifier_t) :: first_model, second_model, unfitted
    type(mlp_classifier_options_t) :: options
    type(mlp_classifier_state_t) :: first_state, second_state
    type(fortnum_status_t) :: status
    real(dp) :: x(6, 3), probabilities(6, 3), scores(6, 3)
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

    call first_model%fit(x, [1, 1, 1, 1, 1, 1], status, options=options)
    call check(.not. status_ok(status), "one-class refusal", failures)
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
