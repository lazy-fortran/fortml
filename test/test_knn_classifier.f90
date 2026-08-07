program test_knn_classifier
    use, intrinsic :: iso_fortran_env, only: error_unit
    use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_quiet_nan
    use fortml_knn_classifier, only: knn_classifier_t, KNN_WEIGHTS_UNIFORM, &
        KNN_WEIGHTS_DISTANCE
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_DOMAIN_ERROR, &
        FORTNUM_NOT_IMPLEMENTED
    implicit none

    integer :: failures

    failures = 0
    call test_uniform_oracle(failures)
    call test_distance_oracle(failures)
    call test_tie_and_weight_oracles(failures)
    call test_derivative_refusals(failures)
    call test_fit_refusals(failures)
    if (failures /= 0) error stop "KNN classifier tests failed"
    write (*, '(a)') "KNN classifier tests passed"

contains

    subroutine test_uniform_oracle(failures)
        integer, intent(inout) :: failures
        type(knn_classifier_t) :: model
        type(fortnum_status_t) :: status
        real(dp) :: x(5, 1), query(2, 1), probabilities(2, 3)
        integer :: labels(5), prediction(2), classes(3)
        real(dp) :: expected(2, 3)

        x(:, 1) = [-2.0_dp, 0.0_dp, 2.0_dp, 4.0_dp, 6.0_dp]
        labels = [30, -10, 30, 50, -10]
        query(:, 1) = [1.0_dp, 3.0_dp]
        call model%fit(x, labels, status, n_neighbors=3, &
            weights=KNN_WEIGHTS_UNIFORM)
        call check(status_ok(status), "uniform fit", failures)
        call check(model%fitted() .and. model%n_neighbors_value() == 3 .and. &
            model%feature_count() == 1 .and. model%sample_count() == 5, &
            "uniform metadata", failures)
        classes = model%classes()
        call check(all(classes == [-10, 30, 50]), "sorted arbitrary integer classes", &
            failures)
        call model%predict_proba(query, probabilities, status)
        expected(1, :) = [1.0_dp/3.0_dp, 2.0_dp/3.0_dp, 0.0_dp]
        expected(2, :) = [1.0_dp/3.0_dp, 1.0_dp/3.0_dp, 1.0_dp/3.0_dp]
        call check(status_ok(status), "uniform prediction status", failures)
        call check(maxval(abs(probabilities - expected)) < 2.0e-14_dp, &
            "independent uniform-vote oracle", failures)
        call model%predict(query, prediction, status)
        call check(status_ok(status) .and. all(prediction == [30, -10]), &
            "uniform predictions and sorted-label tie", failures)
    end subroutine test_uniform_oracle

    subroutine test_distance_oracle(failures)
        integer, intent(inout) :: failures
        type(knn_classifier_t) :: model
        type(fortnum_status_t) :: status
        real(dp) :: x(4, 1), query(2, 1), probabilities(2, 2)
        real(dp) :: expected(2, 2)
        integer :: labels(4)

        x(:, 1) = [0.0_dp, 2.0_dp, 5.0_dp, 7.0_dp]
        labels = [10, 20, 10, 20]
        query(:, 1) = [0.5_dp, 2.0_dp]
        call model%fit(x, labels, status, n_neighbors=2, &
            weights=KNN_WEIGHTS_DISTANCE)
        call model%predict_proba(query, probabilities, status)
        expected(1, :) = [0.75_dp, 0.25_dp]
        expected(2, :) = [0.0_dp, 1.0_dp]
        call check(status_ok(status) .and. model%weighting() == KNN_WEIGHTS_DISTANCE, &
            "distance prediction status and metadata", failures)
        call check(maxval(abs(probabilities - expected)) < 2.0e-14_dp, &
            "independent inverse-distance oracle", failures)
    end subroutine test_distance_oracle

    subroutine test_tie_and_weight_oracles(failures)
        integer, intent(inout) :: failures
        type(knn_classifier_t) :: model
        type(fortnum_status_t) :: status
        real(dp) :: x(3, 1), query(1, 1), probabilities(1, 2), sample_weight(3)
        integer :: labels(3), prediction(1)

        ! At query zero, rows one and two are exactly equidistant.  k=1 must
        ! choose row one, proving that a distance tie is resolved by row order.
        x(:, 1) = [-1.0_dp, 1.0_dp, 4.0_dp]
        labels = [20, 10, 20]
        query(1, 1) = 0.0_dp
        call model%fit(x, labels, status, n_neighbors=1)
        call model%predict(query, prediction, status)
        call check(status_ok(status) .and. prediction(1) == 20, &
            "stable original-index tie order", failures)

        x(:, 1) = [0.0_dp, 2.0_dp, 5.0_dp]
        labels = [10, 20, 10]
        sample_weight = [2.0_dp, 1.0_dp, 1.0_dp]
        query(1, 1) = 1.0_dp
        call model%fit(x, labels, status, n_neighbors=2, &
            sample_weight=sample_weight)
        call model%predict_proba(query, probabilities, status)
        call check(status_ok(status), "sample-weighted prediction status", failures)
        call check(maxval(abs(probabilities(1, :) - [2.0_dp/3.0_dp, 1.0_dp/3.0_dp])) &
            < 2.0e-14_dp, "independent weighted-vote oracle", failures)
    end subroutine test_tie_and_weight_oracles

    subroutine test_derivative_refusals(failures)
        integer, intent(inout) :: failures
        type(knn_classifier_t) :: model
        type(fortnum_status_t) :: status
        real(dp) :: x(2, 1), x_dot(1, 1), query(1, 1)
        real(dp) :: probabilities(1, 2), probabilities_dot(1, 2)
        real(dp) :: probabilities_bar(1, 2), x_bar(1, 1)
        integer :: labels(2)

        x(:, 1) = [0.0_dp, 1.0_dp]
        labels = [3, 8]
        query(1, 1) = 0.25_dp
        x_dot = 1.0_dp
        probabilities_bar = 1.0_dp
        call model%fit(x, labels, status, n_neighbors=1)
        call model%predict_proba_jvp(query, x_dot, probabilities, probabilities_dot, &
            status)
        call check(status%code == FORTNUM_NOT_IMPLEMENTED .and. &
            maxval(abs(probabilities_dot)) == 0.0_dp, &
            "discrete input JVP refusal", failures)
        call model%predict_proba_vjp(query, probabilities_bar, x_bar, status)
        call check(status%code == FORTNUM_NOT_IMPLEMENTED .and. &
            maxval(abs(x_bar)) == 0.0_dp, "discrete input VJP refusal", failures)
    end subroutine test_derivative_refusals

    subroutine test_fit_refusals(failures)
        integer, intent(inout) :: failures
        type(knn_classifier_t) :: model
        type(fortnum_status_t) :: status
        real(dp) :: x(2, 1), nan_value, bad_weight(2)
        integer :: labels(2)

        x(:, 1) = [0.0_dp, 1.0_dp]
        labels = [1, 2]
        nan_value = ieee_value(0.0_dp, ieee_quiet_nan)
        x(2, 1) = nan_value
        call model%fit(x, labels, status)
        call check(status%code == FORTNUM_DOMAIN_ERROR, "nonfinite fit refusal", failures)
        x(2, 1) = 1.0_dp
        call model%fit(x, labels, status, n_neighbors=0)
        call check(status%code == FORTNUM_DOMAIN_ERROR, "invalid k refusal", failures)
        call model%fit(x, labels, status, weights=99)
        call check(status%code == FORTNUM_DOMAIN_ERROR, "invalid weighting refusal", failures)
        bad_weight = [-1.0_dp, 1.0_dp]
        call model%fit(x, labels, status, sample_weight=bad_weight)
        call check(status%code == FORTNUM_DOMAIN_ERROR, "invalid sample weight refusal", &
            failures)
    end subroutine test_fit_refusals

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: description
        integer, intent(inout) :: failures

        if (.not. condition) then
            write (error_unit, '(a)') "FAIL [KNN classifier] "//description
            failures = failures + 1
        end if
    end subroutine check

end program test_knn_classifier
