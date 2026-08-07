program test_xgboost_workflow
    !! Independent staged-prediction and feature-diagnostic oracles for XGBoost.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_quiet_nan
    use fortml_xgboost, only: xgboost_t, xgboost_options_t
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_DOMAIN_ERROR
    implicit none

    integer :: failures

    failures = 0
    call test_staged_regression(failures)
    call test_staged_logistic(failures)
    call test_feature_importance(failures)
    call test_staged_missing_policy(failures)
    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, &
            " XGBoost workflow test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS XGBoost workflow independent behavioral oracles"

contains

    subroutine test_staged_regression(failures)
        integer, intent(inout) :: failures
        type(xgboost_t) :: model
        type(xgboost_options_t) :: options
        type(fortnum_status_t) :: status
        real(dp) :: x(4, 1), y(4), staged(4, 3), margins(4, 3), prediction(4)
        real(dp) :: expected(4, 3)

        x(:, 1) = [0.0_dp, 1.0_dp, 2.0_dp, 3.0_dp]
        y = [0.0_dp, 0.0_dp, 4.0_dp, 4.0_dp]
        options%n_estimators = 3
        options%max_depth = 1
        options%learning_rate = 0.5_dp
        options%l2 = 1.0_dp
        options%min_child_weight = 0.0_dp
        call model%fit_regression(x, y, status, options)
        call model%predict_staged_margin(x, margins, status)
        call model%predict_staged(x, staged, status)
        call model%predict(x, prediction, status)

        ! A closed-form Newton recurrence for the fixed split at 1.5:
        ! the three cumulative margins are [4/3,8/9,16/27] and
        ! [8/3,28/9,92/27] on the left and right leaves respectively.
        expected = reshape([ &
            4.0_dp/3.0_dp, 4.0_dp/3.0_dp, 8.0_dp/3.0_dp, 8.0_dp/3.0_dp, &
            8.0_dp/9.0_dp, 8.0_dp/9.0_dp, 28.0_dp/9.0_dp, 28.0_dp/9.0_dp, &
            16.0_dp/27.0_dp, 16.0_dp/27.0_dp, 92.0_dp/27.0_dp, &
            92.0_dp/27.0_dp], shape(expected))
        call check(status_ok(status), "regression staged status", failures)
        call check(maxval(abs(margins - expected)) < 2.0e-13_dp, &
            "regression staged margin recurrence", failures)
        call check(maxval(abs(staged - expected)) < 2.0e-13_dp, &
            "regression staged prediction recurrence", failures)
        call check(maxval(abs(prediction - margins(:, 3))) < 2.0e-13_dp, &
            "regression final stage equals predict", failures)

        call model%predict_staged_margin(x, margins(:2, :), status)
        call check(status%code == FORTNUM_DOMAIN_ERROR, &
            "staged margin shape refusal", failures)
    end subroutine test_staged_regression

    subroutine test_staged_logistic(failures)
        integer, intent(inout) :: failures
        type(xgboost_t) :: model, regression
        type(xgboost_options_t) :: options
        type(fortnum_status_t) :: status
        real(dp) :: x(4, 1), labels(4), staged(4, 2), margins(4, 2)
        real(dp) :: probabilities(4, 2, 2), prediction(4), expected(4, 2)
        real(dp) :: left_margin, right_margin, left_probability
        real(dp) :: left_gradient, left_hessian, left_weight
        real(dp) :: right_gradient, right_hessian, right_weight

        x(:, 1) = [0.0_dp, 1.0_dp, 2.0_dp, 3.0_dp]
        labels = [0.0_dp, 0.0_dp, 1.0_dp, 1.0_dp]
        options%n_estimators = 2
        options%max_depth = 1
        options%learning_rate = 0.5_dp
        options%l2 = 1.0_dp
        options%min_child_weight = 0.0_dp
        call model%fit_binary(x, labels, status, options)
        call model%predict_staged(x, staged, status)
        call model%predict_staged_margin(x, margins, status)
        call model%predict_proba_staged(x, probabilities, status)
        call model%predict(x, prediction, status)
        ! The independent logistic Newton recurrence uses p=1/2 and h=1/4
        ! initially, then evaluates the exact sigmoid/Hessian at stage one.
        left_margin = -options%learning_rate*(1.0_dp/1.5_dp)
        right_margin = -left_margin
        left_probability = 1.0_dp/(1.0_dp + exp(-left_margin))
        left_gradient = 2.0_dp*left_probability
        left_hessian = 2.0_dp*left_probability*(1.0_dp - left_probability)
        left_weight = -left_gradient/(left_hessian + options%l2)
        right_gradient = -left_gradient
        right_hessian = left_hessian
        right_weight = -right_gradient/(right_hessian + options%l2)
        expected(:, 1) = [left_probability, left_probability, &
            1.0_dp - left_probability, 1.0_dp - left_probability]
        expected(:, 2) = [ &
            1.0_dp/(1.0_dp + exp(-(left_margin + &
            options%learning_rate*left_weight))), &
            1.0_dp/(1.0_dp + exp(-(left_margin + &
            options%learning_rate*left_weight))), &
            1.0_dp/(1.0_dp + exp(-(right_margin + &
            options%learning_rate*right_weight))), &
            1.0_dp/(1.0_dp + exp(-(right_margin + &
            options%learning_rate*right_weight)))]
        call check(status_ok(status), "logistic staged status", failures)
        call check(maxval(abs(staged - expected)) < 2.0e-13_dp, &
            "logistic staged probability oracle", failures)
        call check(maxval(abs(prediction - staged(:, 2))) < 2.0e-13_dp, &
            "logistic final stage equals predict", failures)
        call check(maxval(abs(probabilities(:, 2, :) - staged)) < 2.0e-13_dp .and. &
            maxval(abs(probabilities(:, 1, :) + staged - 1.0_dp)) < 2.0e-13_dp, &
            "logistic staged probability matrix", failures)
        call model%predict_proba_staged(x, probabilities(:2, :, :), status)
        call check(status%code == FORTNUM_DOMAIN_ERROR, &
            "staged probability shape refusal", failures)

        call regression%fit_regression(x, labels, status, options)
        call regression%predict_proba_staged(x, probabilities, status)
        call check(status%code == FORTNUM_DOMAIN_ERROR, &
            "regression staged probability refusal", failures)
    end subroutine test_staged_logistic

    subroutine test_feature_importance(failures)
        integer, intent(inout) :: failures
        type(xgboost_t) :: model, unfitted
        type(xgboost_options_t) :: options
        type(fortnum_status_t) :: status
        real(dp) :: x(4, 2), y(4), gain(2), weight(2), cover(2), normalized(2)

        x = reshape([ &
            0.0_dp, 1.0_dp, 2.0_dp, 3.0_dp, &
            0.0_dp, 0.0_dp, 1.0_dp, 1.0_dp], shape(x))
        y = [0.0_dp, 0.0_dp, 4.0_dp, 4.0_dp]
        options%n_estimators = 1
        options%max_depth = 1
        options%learning_rate = 1.0_dp
        options%l2 = 1.0_dp
        options%min_child_weight = 0.0_dp
        call model%fit_regression(x, y, status, options)
        call model%feature_importance(gain, status, "gain")
        call check(status_ok(status), "gain importance status", failures)
        call check(maxval(abs(gain - [16.0_dp/3.0_dp, 0.0_dp])) < 2.0e-13_dp, &
            "gain importance oracle", failures)
        call model%feature_importance(weight, status, "weight")
        call model%feature_importance(cover, status, "cover")
        call model%feature_importance(normalized, status, "gain", .true.)
        call check(maxval(abs(weight - [1.0_dp, 0.0_dp])) < 2.0e-13_dp .and. &
            maxval(abs(cover - [4.0_dp, 0.0_dp])) < 2.0e-13_dp .and. &
            maxval(abs(normalized - [1.0_dp, 0.0_dp])) < 2.0e-13_dp, &
            "weight, cover, and normalized importance oracles", failures)
        call model%feature_importance(gain, status, "unsupported")
        call check(status%code == FORTNUM_DOMAIN_ERROR, &
            "importance kind refusal", failures)
        call unfitted%feature_importance(gain, status)
        call check(status%code == FORTNUM_DOMAIN_ERROR, &
            "unfitted importance refusal", failures)
    end subroutine test_feature_importance

    subroutine test_staged_missing_policy(failures)
        integer, intent(inout) :: failures
        type(xgboost_t) :: model, rejected
        type(xgboost_options_t) :: options
        type(fortnum_status_t) :: status
        real(dp) :: x(5, 1), y(5), staged(5, 2)

        x(:, 1) = [0.0_dp, 1.0_dp, 2.0_dp, 3.0_dp, &
            ieee_value(0.0_dp, ieee_quiet_nan)]
        y = [0.0_dp, 0.0_dp, 4.0_dp, 4.0_dp, 0.0_dp]
        options%n_estimators = 2
        options%max_depth = 1
        options%learning_rate = 0.5_dp
        options%l2 = 1.0_dp
        options%min_child_weight = 0.0_dp
        options%missing_policy = "learn"
        call model%fit_regression(x, y, status, options)
        call model%predict_staged(x, staged, status)
        call check(status_ok(status) .and. maxval(abs(staged(5, :))) < huge(1.0_dp), &
            "staged prediction preserves learned NaN policy", failures)
        options%missing_policy = "error"
        call rejected%fit_regression(x, y, status, options)
        call check(status%code == FORTNUM_DOMAIN_ERROR, &
            "staged workflow finite-only policy refusal", failures)
    end subroutine test_staged_missing_policy

    subroutine check(condition, label, failures)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: label
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "  FAIL "//label
        end if
    end subroutine check

end program test_xgboost_workflow
