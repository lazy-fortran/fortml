program test_xgboost
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_xgboost, only: xgboost_t, xgboost_options_t
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR
    implicit none

    integer :: failures

    failures = 0
    call test_squared_second_order_oracle(failures)
    call test_logistic_second_order_oracle(failures)
    call test_regularisation_and_determinism(failures)
    call test_refusals(failures)
    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, " xgboost test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS"

contains

    subroutine test_squared_second_order_oracle(failures)
        integer, intent(inout) :: failures
        type(xgboost_t) :: model
        type(xgboost_options_t) :: options
        type(fortnum_status_t) :: status
        real(dp) :: x(4, 1), y(4), prediction(4), weights(2)
        real(dp) :: expected(4), expected_gain

        x(:, 1) = [0.0_dp, 1.0_dp, 2.0_dp, 3.0_dp]
        y = [0.0_dp, 0.0_dp, 4.0_dp, 4.0_dp]
        options%n_estimators = 1
        options%learning_rate = 1.0_dp
        options%l2 = 1.0_dp
        options%min_child_weight = 0.0_dp
        call model%fit_regression(x, y, status, options)
        call model%predict(x, prediction, status)
        call model%leaf_weights(1, weights, status)

        ! Base margin is 2.  At the exact split x=1.5, G_L=4, H_L=2 and
        ! G_R=-4, H_R=2, hence the regularised leaf weights are +/-4/3.
        expected = [2.0_dp - 4.0_dp/3.0_dp, 2.0_dp - 4.0_dp/3.0_dp, &
            2.0_dp + 4.0_dp/3.0_dp, 2.0_dp + 4.0_dp/3.0_dp]
        expected_gain = 0.5_dp*((4.0_dp**2)/3.0_dp + &
            ((-4.0_dp)**2)/3.0_dp)
        if (.not. status_ok(status) .or. .not. model%fitted() .or. &
            model%estimator_count() /= 1 .or. model%feature_count() /= 1 .or. &
            maxval(abs(prediction - expected)) > 2.0e-13_dp .or. &
            maxval(abs(weights - [-4.0_dp/3.0_dp, 4.0_dp/3.0_dp])) > 2.0e-13_dp .or. &
            abs(model%split_gain(1) - expected_gain) > 2.0e-13_dp) then
            write (error_unit, '(a,es12.4)') &
                "FAIL [xgb squared] second-order split oracle ", &
                maxval(abs(prediction - expected))
            failures = failures + 1
        end if
    end subroutine test_squared_second_order_oracle

    subroutine test_logistic_second_order_oracle(failures)
        integer, intent(inout) :: failures
        type(xgboost_t) :: model
        type(xgboost_options_t) :: options
        type(fortnum_status_t) :: status
        real(dp) :: x(4, 1), labels(4), probabilities(4), probability_matrix(4, 2)
        real(dp) :: margin(4), weights(2), initial_loss, final_loss

        x(:, 1) = [0.0_dp, 1.0_dp, 2.0_dp, 3.0_dp]
        labels = [0.0_dp, 0.0_dp, 1.0_dp, 1.0_dp]
        options%n_estimators = 1
        options%learning_rate = 0.5_dp
        options%l2 = 0.0_dp
        options%min_child_weight = 0.0_dp
        call model%fit_binary(x, labels, status, options)
        call model%predict(x, probabilities, status)
        call model%predict_margin(x, margin, status)
        call model%predict_proba(x, probability_matrix, status)
        call model%leaf_weights(1, weights, status)
        initial_loss = log(2.0_dp)
        final_loss = -0.5_dp*sum(labels*log(probabilities) + &
            (1.0_dp - labels)*log(1.0_dp - probabilities))/real(size(labels), dp)

        ! At margin zero, p=.5, G_L=1, H_L=.5 and G_R=-1, H_R=.5;
        ! the Newton leaf updates are -2 and +2, then shrinkage is one half.
        if (.not. status_ok(status) .or. trim(model%objective_name()) /= "logistic" .or. &
            maxval(abs(weights - [-2.0_dp, 2.0_dp])) > 2.0e-12_dp .or. &
            maxval(abs(margin - [-1.0_dp, -1.0_dp, 1.0_dp, 1.0_dp])) > 2.0e-12_dp .or. &
            maxval(abs(probabilities - [0.2689414213699951_dp, &
                0.2689414213699951_dp, 0.7310585786300049_dp, &
                0.7310585786300049_dp])) > 2.0e-12_dp .or. &
            maxval(abs(probability_matrix(:, 2) - probabilities)) > 2.0e-14_dp .or. &
            final_loss >= initial_loss) then
            write (error_unit, '(a,es12.4)') &
                "FAIL [xgb logistic] Newton objective oracle ", final_loss
            failures = failures + 1
        end if
    end subroutine test_logistic_second_order_oracle

    subroutine test_regularisation_and_determinism(failures)
        integer, intent(inout) :: failures
        type(xgboost_t) :: unregularised, regularised, repeat
        type(xgboost_options_t) :: options, strong_options
        type(fortnum_status_t) :: status
        real(dp) :: x(6, 1), y(6), first(6), second(6), repeated(6)

        x(:, 1) = [0.0_dp, 0.2_dp, 0.4_dp, 1.6_dp, 1.8_dp, 2.0_dp]
        y = [0.0_dp, 0.0_dp, 0.0_dp, 3.0_dp, 3.0_dp, 3.0_dp]
        options%n_estimators = 3
        options%learning_rate = 0.4_dp
        options%l2 = 0.0_dp
        options%min_child_weight = 0.0_dp
        strong_options = options
        strong_options%l2 = 100.0_dp
        call unregularised%fit_regression(x, y, status, options)
        call regularised%fit_regression(x, y, status, strong_options)
        call repeat%fit_regression(x, y, status, options)
        call unregularised%predict(x, first, status)
        call regularised%predict(x, second, status)
        call repeat%predict(x, repeated, status)
        if (.not. status_ok(status) .or. maxval(abs(first - repeated)) > 0.0_dp .or. &
            maxval(abs(second - sum(y)/real(size(y), dp))) >= &
                maxval(abs(first - sum(y)/real(size(y), dp)))) then
            write (error_unit, '(a)') &
                "FAIL [xgb policy] deterministic fit or L2 shrinkage"
            failures = failures + 1
        end if
    end subroutine test_regularisation_and_determinism

    subroutine test_refusals(failures)
        integer, intent(inout) :: failures
        type(xgboost_t) :: model
        type(xgboost_options_t) :: options
        type(fortnum_status_t) :: status
        real(dp) :: x(3, 1), y(3), prediction(3), probabilities(3, 2)

        x(:, 1) = [0.0_dp, 1.0_dp, 2.0_dp]
        y = [0.0_dp, 1.0_dp, 2.0_dp]
        options%objective = "unsupported"
        call model%fit(x, y, status, options)
        if (status%code /= FORTNUM_DOMAIN_ERROR) then
            write (error_unit, '(a)') "FAIL [xgb refusal] unsupported objective"
            failures = failures + 1
        end if
        options = xgboost_options_t()
        options%max_depth = 2
        call model%fit(x, y, status, options)
        if (status%code /= FORTNUM_DOMAIN_ERROR) then
            write (error_unit, '(a)') "FAIL [xgb refusal] unsupported depth"
            failures = failures + 1
        end if
        call model%fit_regression(x, y, status)
        call model%predict_proba(x, probabilities, status)
        if (status%code /= FORTNUM_DOMAIN_ERROR) then
            write (error_unit, '(a)') "FAIL [xgb refusal] regression probabilities"
            failures = failures + 1
        end if
        call model%predict(x, prediction, status)
        if (status%code /= FORTNUM_OK) then
            write (error_unit, '(a)') "FAIL [xgb refusal] valid model poisoned"
            failures = failures + 1
        end if
    end subroutine test_refusals

end program test_xgboost
