program test_xgboost_multiclass
    !! Independent behavior and input-JVP checks for OVR XGBoost classification.
    use, intrinsic :: iso_fortran_env, only: error_unit
    use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_quiet_nan
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_ok
    use fortml_xgboost, only: xgboost_options_t
    use fortml_xgboost_multiclass, only: xgboost_multiclass_t
    implicit none

    integer :: failures

    failures = 0
    call test_probability_and_labels(failures)
    call test_probability_jvp(failures)
    call test_missing_value_routing(failures)
    call test_refusals(failures)
    if (failures > 0) then
        write (error_unit, '(i0,a)') failures, &
            " XGBoost multiclass test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS XGBoost multiclass independent behavioral oracles"

contains

    subroutine test_missing_value_routing(failures)
        integer, intent(inout) :: failures
        type(xgboost_multiclass_t) :: model, rejected
        type(xgboost_options_t) :: options
        type(fortnum_status_t) :: status
        real(dp) :: x(6, 1), query(2, 1), probabilities(2, 3)
        integer :: labels(6)

        x(:, 1) = [-2.0_dp, -1.0_dp, 0.0_dp, 1.0_dp, 2.0_dp, &
            ieee_value(0.0_dp, ieee_quiet_nan)]
        labels = [-8, -8, 2, 2, 11, 11]
        options%n_estimators = 1
        options%max_depth = 1
        options%learning_rate = 1.0_dp
        options%l2 = 1.0_dp
        options%min_child_weight = 0.0_dp
        call rejected%fit(x, labels, status, options)
        call check(.not. status_ok(status), "default missing policy refusal", failures)
        options%missing_policy = "learn"
        call model%fit(x, labels, status, options)
        query(:, 1) = [ieee_value(0.0_dp, ieee_quiet_nan), 0.5_dp]
        call model%predict_proba(query, probabilities, status)
        call check(status_ok(status), "missing-value probability prediction", failures)
        call check(maxval(abs(sum(probabilities, dim=2) - 1.0_dp)) < 3.0e-14_dp, &
            "missing-value probability normalization", failures)
    end subroutine test_missing_value_routing

    subroutine test_probability_and_labels(failures)
        integer, intent(inout) :: failures
        type(xgboost_multiclass_t) :: model
        type(xgboost_options_t) :: options
        type(fortnum_status_t) :: status
        real(dp) :: x(9, 1), query(3, 1), probabilities(3, 3), margins(3, 3)
        real(dp) :: expected(3, 3), total
        integer :: labels(9), predicted(3), classes(3), i, j

        x(:, 1) = [-4.0_dp, -3.0_dp, -2.0_dp, -1.0_dp, 0.0_dp, 1.0_dp, &
            2.0_dp, 3.0_dp, 4.0_dp]
        labels = [-8, -8, -8, 2, 2, 2, 11, 11, 11]
        query(:, 1) = [-2.3_dp, 0.1_dp, 2.4_dp]
        options%n_estimators = 4
        options%learning_rate = 0.4_dp
        options%l2 = 1.0_dp
        options%min_child_weight = 0.0_dp
        call model%fit(x, labels, status, options)
        call model%predict_proba(query, probabilities, status)
        call model%decision_function(query, margins, status)
        call model%predict(query, predicted, status)
        classes = model%classes()
        do i = 1, size(query, 1)
            total = 0.0_dp
            do j = 1, size(classes)
                expected(i, j) = stable_sigmoid(margins(i, j))
                total = total + expected(i, j)
            end do
            expected(i, :) = expected(i, :)/total
        end do
        call check(status_ok(status), "fit and prediction status", failures)
        call check(model%fitted() .and. model%feature_count() == 1 .and. &
            model%class_count() == 3 .and. model%estimator_count() == 4, &
            "model metadata", failures)
        call check(all(classes == [-8, 2, 11]), "sorted arbitrary classes", failures)
        call check(maxval(abs(probabilities - expected)) < 3.0e-14_dp, &
            "OVR probability normalization", failures)
        call check(maxval(abs(sum(probabilities, dim=2) - 1.0_dp)) < 3.0e-14_dp, &
            "probability rows sum to one", failures)
        call check(all(predicted == [-8, 2, 11]), &
            "multiclass argmax prediction", failures)
    end subroutine test_probability_and_labels

    subroutine test_probability_jvp(failures)
        integer, intent(inout) :: failures
        type(xgboost_multiclass_t) :: model
        type(xgboost_options_t) :: options
        type(fortnum_status_t) :: status
        real(dp) :: x(9, 1), query(3, 1), query_dot(3, 1)
        integer :: labels(9)
        real(dp) :: probabilities(3, 3), probabilities_dot(3, 3)
        real(dp) :: plus(3, 3), minus(3, 3), h

        x(:, 1) = [-4.0_dp, -3.0_dp, -2.0_dp, -1.0_dp, 0.0_dp, 1.0_dp, &
            2.0_dp, 3.0_dp, 4.0_dp]
        labels = [-8, -8, -8, 2, 2, 2, 11, 11, 11]
        query(:, 1) = [-2.3_dp, 0.1_dp, 2.4_dp]
        query_dot(:, 1) = [0.17_dp, -0.13_dp, 0.23_dp]
        options%n_estimators = 4
        options%learning_rate = 0.4_dp
        options%l2 = 1.0_dp
        options%min_child_weight = 0.0_dp
        call model%fit(x, labels, status, options)
        call model%predict_proba_jvp(query, query_dot, probabilities, &
            probabilities_dot, status)
        h = 1.0e-6_dp
        call model%predict_proba(query + h*query_dot, plus, status)
        call model%predict_proba(query - h*query_dot, minus, status)
        call check(status_ok(status), "probability JVP status", failures)
        call check(maxval(abs(probabilities_dot - (plus - minus)/(2.0_dp*h))) < &
            3.0e-8_dp, "probability JVP finite difference", failures)
        call check(maxval(abs(sum(probabilities_dot, dim=2))) < 3.0e-10_dp, &
            "probability JVP preserves normalization", failures)
    end subroutine test_probability_jvp

    subroutine test_refusals(failures)
        integer, intent(inout) :: failures
        type(xgboost_multiclass_t) :: model
        type(fortnum_status_t) :: status
        real(dp) :: x(3, 1), probabilities(3, 3)
        integer :: labels(3)

        x(:, 1) = [0.0_dp, 1.0_dp, 2.0_dp]
        labels = [1, 1, 1]
        call model%fit(x, labels, status)
        call check(.not. status_ok(status), "single-class refusal", failures)
        labels = [1, 2, 1]
        call model%fit(x, labels, status)
        call model%predict_proba(x, probabilities, status)
        call check(.not. status_ok(status), "probability output-shape refusal", &
            failures)
    end subroutine test_refusals

    real(dp) function stable_sigmoid(value) result(probability)
        real(dp), intent(in) :: value

        if (value >= 0.0_dp) then
            probability = 1.0_dp/(1.0_dp + exp(-value))
        else
            probability = exp(value)/(1.0_dp + exp(value))
        end if
    end function stable_sigmoid

    subroutine check(condition, label, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: label
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "  FAIL [xgb multiclass] "//label
        end if
    end subroutine check

end program test_xgboost_multiclass
