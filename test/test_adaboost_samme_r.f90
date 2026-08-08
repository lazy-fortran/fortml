program test_adaboost_samme_r
    !! Independent probability-update oracle and typed-boundary checks.
    use, intrinsic :: iso_fortran_env, only: real64, error_unit
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortml_adaboost_classifier, only: adaboost_classifier_t, &
        ADABOOST_ALGORITHM_SAMME_R
    implicit none

    integer, parameter :: dp = real64
    type(adaboost_classifier_t) :: model
    type(fortnum_status_t) :: status
    type(fortml_device_t) :: cuda
    real(dp) :: x(9, 1), query(5, 1), probabilities(5, 3), expected(5, 3)
    real(dp) :: margins(5, 3), x_dot(5, 1), probability_dot(5, 3)
    real(dp) :: epsilon
    integer :: labels(9), predicted(5), classes(3), failures, i

    failures = 0
    epsilon = 1.0e-12_dp
    x(:, 1) = real([(i, i=0, 8)], dp)
    labels = [-7, -7, -7, 4, 4, 4, 99, 99, 99]
    query(:, 1) = [0.5_dp, 2.5_dp, 3.5_dp, 5.5_dp, 7.5_dp]

    call model%fit(x, labels, status, n_estimators=1, max_depth=1, &
        min_samples_leaf=1, seed=7, algorithm=ADABOOST_ALGORITHM_SAMME_R)
    call check(status_ok(status), "SAMME.R fit", failures)
    call check(model%algorithm() == ADABOOST_ALGORITHM_SAMME_R, &
        "SAMME.R algorithm metadata", failures)
    call check(model%estimator_count() == 1, "SAMME.R stage count", failures)
    classes = model%classes()
    call check(all(classes == [-7, 4, 99]), "SAMME.R sorted labels", failures)

    ! The deterministic stump has probabilities [1,0,0] on the left leaf and
    ! [0,1/2,1/2] on the right.  SAMME.R clips zero probabilities at epsilon,
    ! accumulates centred log probabilities, then applies a softmax scaled by
    ! 1/(K-1), which is exactly the clipped tree probability for one stage.
    expected = 0.0_dp
    expected(1, :) = [1.0_dp, epsilon, epsilon]
    do i = 2, 5
        expected(i, :) = [epsilon, 0.5_dp, 0.5_dp]
    end do
    do i = 1, size(expected, 1)
        expected(i, :) = expected(i, :)/sum(expected(i, :))
    end do
    call model%predict_proba(query, probabilities, status)
    call check(status_ok(status) .and. maxval(abs(probabilities-expected)) < 2.0e-12_dp, &
        "independent SAMME.R probability oracle", failures)
    call model%decision_function(query, margins, status)
    call check(status_ok(status), "SAMME.R decision function", failures)
    call model%predict(query, predicted, status)
    call check(status_ok(status) .and. all(predicted == [-7, 4, 4, 4, 4]), &
        "SAMME.R predicted labels", failures)
    call check(abs(sum(probabilities(1, :))-1.0_dp) < 2.0e-14_dp .and. &
        maxval(abs(sum(probabilities, dim=2)-1.0_dp)) < 2.0e-14_dp, &
        "SAMME.R probability simplex", failures)

    x_dot = 0.0_dp
    probability_dot = 0.0_dp
    call model%predict_proba_jvp(query, x_dot, probabilities, probability_dot, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "SAMME.R split-routing JVP refusal", failures)
    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    probabilities = 17.0_dp
    call model%predict_proba_device(cuda, query, probabilities, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED .and. all(probabilities == 17.0_dp), &
        "SAMME.R output-preserving CUDA refusal", failures)

    ! A malformed refit must not overwrite a valid SAMME.R model.
    call model%fit(x, labels, status, n_estimators=1, algorithm=ADABOOST_ALGORITHM_SAMME_R, &
        seed=-1)
    call check(status%code /= 0 .and. model%fitted(), "transactional invalid seed", failures)
    call model%predict(query, predicted, status)
    call check(status_ok(status) .and. all(predicted == [-7, 4, 4, 4, 4]), &
        "transactional model preservation", failures)

    if (failures /= 0) then
        write (error_unit, '(a,i0)') "FAIL AdaBoost SAMME.R cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS AdaBoost SAMME.R independent oracle"

contains

    subroutine check(condition, description, failure_count)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failure_count
        if (.not. condition) then
            failure_count = failure_count + 1
            write (error_unit, '(a)') "  FAIL [AdaBoost SAMME.R] "//description
        end if
    end subroutine check

end program test_adaboost_samme_r
