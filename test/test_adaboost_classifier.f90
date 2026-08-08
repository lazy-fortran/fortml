program test_adaboost_classifier
    !! Independent one-stump AdaBoost oracle and device-boundary checks.
    use, intrinsic :: iso_fortran_env, only: real64, error_unit
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortml_adaboost_classifier, only: adaboost_classifier_t
    implicit none

    integer, parameter :: dp = real64
    type(adaboost_classifier_t) :: model
    type(fortnum_status_t) :: status
    type(fortml_device_t) :: cuda
    real(dp) :: x(6, 1), query(4, 1), probabilities(4, 2), expected(4, 2)
    real(dp) :: score(4), expected_score(4), x_dot(4, 1), probabilities_dot(4, 2)
    integer :: labels(6), predicted(4), classes(2), failures
    real(dp) :: alpha, error

    failures = 0
    x(:, 1) = [0.0_dp, 1.0_dp, 2.0_dp, 3.0_dp, 4.0_dp, 5.0_dp]
    labels = [-3, -3, 8, -3, 8, 8]
    query(:, 1) = [0.5_dp, 1.5_dp, 2.5_dp, 4.5_dp]
    call model%fit(x, labels, status, n_estimators=1, max_depth=1, min_samples_leaf=1)
    call check(status_ok(status), "one-stump fit", failures)
    call check(model%fitted() .and. model%feature_count() == 1 .and. &
        model%estimator_count() == 1, "fitted metadata", failures)
    classes = model%classes()
    call check(all(classes == [-3, 8]), "sorted arbitrary classes", failures)

    ! The independent Gini oracle chooses threshold 1.5.  Its only error is
    ! the fourth row, so AdaBoost's weight is 1/6 and alpha = log(sqrt(5)).
    error = 1.0_dp/6.0_dp
    alpha = 0.5_dp*log((1.0_dp-error)/error)
    expected_score = [-alpha, alpha, alpha, alpha]
    expected = 0.0_dp
    expected(:, 2) = 1.0_dp/(1.0_dp + exp(-2.0_dp*expected_score))
    expected(:, 1) = 1.0_dp - expected(:, 2)
    call model%decision_function(query, score, status)
    call check(status_ok(status) .and. maxval(abs(score-expected_score)) < 2.0e-12_dp, &
        "independent stump margin oracle", failures)
    call model%predict_proba(query, probabilities, status)
    call check(status_ok(status) .and. maxval(abs(probabilities-expected)) < 2.0e-12_dp, &
        "independent logistic probability oracle", failures)
    call model%predict(query, predicted, status)
    call check(status_ok(status) .and. all(predicted == [-3, 8, 8, 8]), &
        "predicted labels", failures)
    call check(maxval(abs(sum(probabilities, dim=2)-1.0_dp)) < 2.0e-14_dp, &
        "probability simplex", failures)

    x_dot = 0.0_dp
    probabilities_dot = 0.0_dp
    call model%predict_proba_jvp(query, x_dot, probabilities, probabilities_dot, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "split-routing JVP refusal", failures)
    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call model%predict_proba_device(cuda, query, probabilities, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, "CUDA refusal", failures)

    if (failures /= 0) then
        write (error_unit, '(a,i0)') "FAIL AdaBoost cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS AdaBoost independent oracle"

contains

    subroutine check(condition, description, failure_count)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failure_count
        if (.not. condition) then
            failure_count = failure_count + 1
            write (error_unit, '(a)') "  FAIL [AdaBoost] "//description
        end if
    end subroutine check

end program test_adaboost_classifier
