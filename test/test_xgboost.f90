program test_xgboost
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_quiet_nan
    use fortml_xgboost, only: xgboost_t, xgboost_options_t
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR
    implicit none

    integer :: failures

    failures = 0
    call test_squared_second_order_oracle(failures)
    call test_deeper_tree_oracle(failures)
    call test_logistic_second_order_oracle(failures)
    call test_regularisation_and_determinism(failures)
    call test_missing_value_routing(failures)
    call test_missing_logistic_routing(failures)
    call test_weighted_histogram_oracle(failures)
    call test_histogram_logistic_and_missing(failures)
    call test_refusals(failures)
    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, " xgboost test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS"

contains

    subroutine test_weighted_histogram_oracle(failures)
        integer, intent(inout) :: failures
        type(xgboost_t) :: model
        type(xgboost_options_t) :: options
        type(fortnum_status_t) :: status
        real(dp) :: x(6, 1), y(6), weights(6), prediction(6), expected(6)

        ! The weighted median is between x=4 and x=5: with max_bin=2 the
        ! histogram path has exactly one admissible cut at 4.5.  The expected
        ! leaf value on the left is the weighted mean 60/9, and the right
        ! leaf is 100/10; these values are computed independently of fitting.
        x(:, 1) = [0.0_dp, 1.0_dp, 2.0_dp, 3.0_dp, 4.0_dp, 5.0_dp]
        y = [0.0_dp, 0.0_dp, 0.0_dp, 10.0_dp, 10.0_dp, 10.0_dp]
        weights = [1.0_dp, 1.0_dp, 1.0_dp, 1.0_dp, 5.0_dp, 5.0_dp]
        options%n_estimators = 1
        options%max_depth = 1
        options%learning_rate = 1.0_dp
        options%l2 = 0.0_dp
        options%min_child_weight = 0.0_dp
        options%tree_method = "hist"
        options%max_bin = 2
        call model%fit_regression(x, y, status, options, weights)
        call model%predict(x, prediction, status)
        expected = [20.0_dp/3.0_dp, 20.0_dp/3.0_dp, 20.0_dp/3.0_dp, &
            20.0_dp/3.0_dp, 20.0_dp/3.0_dp, 10.0_dp]
        if (status%code /= FORTNUM_OK .or. trim(model%tree_method()) /= "hist" .or. &
            model%max_bin_count() /= 2 .or. model%tree_node_count(1) /= 3 .or. &
            maxval(abs(prediction - expected)) > 2.0e-12_dp) then
            write (error_unit, '(a,es12.4)') &
                "FAIL [xgb histogram] weighted-quantile oracle ", &
                maxval(abs(prediction - expected))
            failures = failures + 1
        end if
    end subroutine test_weighted_histogram_oracle

    subroutine test_histogram_logistic_and_missing(failures)
        integer, intent(inout) :: failures
        type(xgboost_t) :: model
        type(xgboost_options_t) :: options
        type(fortnum_status_t) :: status
        real(dp) :: x(6, 1), labels(6), probability(6), expected(6), nan
        real(dp) :: negative, positive

        nan = ieee_value(0.0_dp, ieee_quiet_nan)
        x(:, 1) = [0.0_dp, 1.0_dp, 2.0_dp, 3.0_dp, nan, nan]
        labels = [0.0_dp, 0.0_dp, 1.0_dp, 1.0_dp, 0.0_dp, 1.0_dp]
        options%n_estimators = 1
        options%max_depth = 1
        options%learning_rate = 0.5_dp
        options%l2 = 0.0_dp
        options%min_child_weight = 0.0_dp
        options%tree_method = "hist"
        options%max_bin = 2
        options%missing_policy = "left"
        call model%fit_binary(x, labels, status, options)
        call model%predict(x, probability, status)
        negative = 1.0_dp/(1.0_dp + exp(0.5_dp))
        positive = 1.0_dp/(1.0_dp + exp(-1.0_dp))
        expected = [negative, negative, positive, positive, negative, negative]
        if (status%code /= FORTNUM_OK .or. maxval(abs(probability - expected)) > &
            2.0e-12_dp .or. trim(model%tree_method()) /= "hist") then
            write (error_unit, '(a,es12.4)') &
                "FAIL [xgb histogram] logistic/missing oracle ", &
                maxval(abs(probability - expected))
            failures = failures + 1
        end if
    end subroutine test_histogram_logistic_and_missing

    subroutine test_missing_value_routing(failures)
        integer, intent(inout) :: failures
        type(xgboost_t) :: learned, forced_right, forced_left, rejected
        type(xgboost_options_t) :: options
        type(fortnum_status_t) :: status
        real(dp) :: x(6, 1), y(6), prediction(6), x_dot(6, 1), prediction_dot(6)
        real(dp) :: expected(6), boundary(1, 1), boundary_dot(1, 1), boundary_value(1)

        x(:, 1) = [0.0_dp, 1.0_dp, 2.0_dp, 3.0_dp, 4.0_dp, &
            ieee_value(0.0_dp, ieee_quiet_nan)]
        y = [0.0_dp, 0.0_dp, 10.0_dp, 10.0_dp, 10.0_dp, 0.0_dp]
        options%n_estimators = 1
        options%max_depth = 1
        options%min_samples_leaf = 1
        options%learning_rate = 1.0_dp
        options%l2 = 0.0_dp
        options%min_child_weight = 0.0_dp

        call rejected%fit_regression(x, y, status, options)
        if (status%code /= FORTNUM_DOMAIN_ERROR) then
            write (error_unit, '(a)') &
                "FAIL [xgb missing] default policy must reject NaN fit input"
            failures = failures + 1
        end if

        options%missing_policy = "learn"
        call learned%fit_regression(x, y, status, options)
        call learned%predict(x, prediction, status)
        expected = [0.0_dp, 0.0_dp, 10.0_dp, 10.0_dp, 10.0_dp, 0.0_dp]
        x_dot = 0.0_dp
        call learned%predict_jvp(x, x_dot, prediction, prediction_dot, status)
        if (status%code /= FORTNUM_OK .or. trim(learned%missing_policy()) /= "learn" .or. &
            .not. learned%accepts_missing() .or. maxval(abs(prediction - expected)) > &
            2.0e-13_dp .or. maxval(abs(prediction_dot)) > 2.0e-14_dp) then
            write (error_unit, '(a,es12.4)') &
                "FAIL [xgb missing] learned default-direction oracle ", &
                maxval(abs(prediction - expected))
            failures = failures + 1
        end if

        options%missing_policy = "right"
        call forced_right%fit_regression(x, y, status, options)
        call forced_right%predict(x, prediction, status)
        expected = [0.0_dp, 0.0_dp, 7.5_dp, 7.5_dp, 7.5_dp, 7.5_dp]
        if (status%code /= FORTNUM_OK .or. maxval(abs(prediction - expected)) > &
            2.0e-13_dp) then
            write (error_unit, '(a,es12.4)') &
                "FAIL [xgb missing] forced-right oracle ", &
                maxval(abs(prediction - expected))
            failures = failures + 1
        end if

        options%missing_policy = "left"
        call forced_left%fit_regression(x, y, status, options)
        call forced_left%predict(x, prediction, status)
        expected = [0.0_dp, 0.0_dp, 10.0_dp, 10.0_dp, 10.0_dp, 0.0_dp]
        if (status%code /= FORTNUM_OK .or. maxval(abs(prediction - expected)) > &
            2.0e-13_dp) then
            write (error_unit, '(a,es12.4)') &
                "FAIL [xgb missing] forced-left oracle ", &
                maxval(abs(prediction - expected))
            failures = failures + 1
        end if

        boundary(1, 1) = 1.5_dp
        boundary_dot(1, 1) = 1.0_dp
        call learned%predict_jvp(boundary, boundary_dot, boundary_value, &
            prediction_dot(:1), status)
        if (status%code /= FORTNUM_DOMAIN_ERROR) then
            write (error_unit, '(a)') &
                "FAIL [xgb missing] split boundary derivative refusal"
            failures = failures + 1
        end if
    end subroutine test_missing_value_routing

    subroutine test_missing_logistic_routing(failures)
        integer, intent(inout) :: failures
        type(xgboost_t) :: model
        type(xgboost_options_t) :: options
        type(fortnum_status_t) :: status
        real(dp) :: x(6, 1), labels(6), probabilities(6), expected(6)
        real(dp) :: negative, positive

        x(:, 1) = [0.0_dp, 1.0_dp, 2.0_dp, 3.0_dp, 4.0_dp, &
            ieee_value(0.0_dp, ieee_quiet_nan)]
        labels = [0.0_dp, 0.0_dp, 1.0_dp, 1.0_dp, 1.0_dp, 0.0_dp]
        options%n_estimators = 1
        options%max_depth = 1
        options%min_samples_leaf = 1
        options%learning_rate = 1.0_dp
        options%l2 = 0.0_dp
        options%min_child_weight = 0.0_dp
        options%missing_policy = "learn"
        call model%fit_binary(x, labels, status, options)
        call model%predict(x, probabilities, status)
        negative = 1.0_dp/(1.0_dp + exp(2.0_dp))
        positive = 1.0_dp - negative
        expected = [negative, negative, positive, positive, positive, negative]
        if (status%code /= FORTNUM_OK .or. maxval(abs(probabilities - expected)) > &
            2.0e-13_dp) then
            write (error_unit, '(a,es12.4)') &
                "FAIL [xgb missing] logistic default-direction oracle ", &
                maxval(abs(probabilities - expected))
            failures = failures + 1
        end if
    end subroutine test_missing_logistic_routing

    subroutine test_deeper_tree_oracle(failures)
        integer, intent(inout) :: failures
        type(xgboost_t) :: model
        type(xgboost_options_t) :: options
        type(fortnum_status_t) :: status
        real(dp) :: x(8, 2), y(8), prediction(8), x_dot(8, 2), prediction_dot(8)
        real(dp) :: boundary(1, 2), boundary_dot(1, 2), boundary_value(1)
        real(dp) :: boundary_value_dot(1), expected(8)

        ! Feature one separates the two groups.  Feature two then separates
        ! each group, so a depth-two exact tree has a seven-node independent
        ! oracle and reproduces the four constant leaves exactly.
        x(:, 1) = [0.0_dp, 0.0_dp, 0.0_dp, 0.0_dp, 1.0_dp, 1.0_dp, 1.0_dp, 1.0_dp]
        x(:, 2) = [0.0_dp, 1.0_dp, 2.0_dp, 3.0_dp, 0.0_dp, 1.0_dp, 2.0_dp, 3.0_dp]
        y = [0.0_dp, 0.0_dp, 4.0_dp, 4.0_dp, 10.0_dp, 10.0_dp, 14.0_dp, 14.0_dp]
        options%n_estimators = 1
        options%max_depth = 2
        options%learning_rate = 1.0_dp
        options%l2 = 0.0_dp
        options%min_child_weight = 0.0_dp
        call model%fit_regression(x, y, status, options)
        call model%predict(x, prediction, status)
        x_dot = 0.0_dp
        x_dot(:, 1) = 0.25_dp
        call model%predict_jvp(x, x_dot, prediction, prediction_dot, status)
        expected = y
        if (.not. status_ok(status) .or. .not. model%fitted() .or. &
            model%tree_node_count(1) /= 7 .or. model%tree_depth(1) /= 2 .or. &
            maxval(abs(prediction - expected)) > 2.0e-13_dp .or. &
            maxval(abs(prediction_dot)) > 2.0e-14_dp) then
            write (error_unit, '(a,es12.4)') &
                "FAIL [xgb depth two] recursive tree oracle ", &
                maxval(abs(prediction - expected))
            failures = failures + 1
        end if
        boundary = reshape([0.0_dp, 1.5_dp], shape(boundary))
        boundary_dot = reshape([0.0_dp, 1.0_dp], shape(boundary_dot))
        call model%predict_jvp(boundary, boundary_dot, boundary_value, &
            boundary_value_dot, status)
        if (status%code /= FORTNUM_DOMAIN_ERROR) then
            write (error_unit, '(a)') &
                "FAIL [xgb depth two] internal split derivative refusal"
            failures = failures + 1
        end if
    end subroutine test_deeper_tree_oracle

    subroutine test_squared_second_order_oracle(failures)
        integer, intent(inout) :: failures
        type(xgboost_t) :: model
        type(xgboost_options_t) :: options
        type(fortnum_status_t) :: status
        real(dp) :: x(4, 1), y(4), prediction(4), weights(2)
        real(dp) :: x_dot(4, 1), prediction_dot(4)
        real(dp) :: boundary(1, 1), boundary_dot(1, 1), boundary_value(1)
        real(dp) :: expected(4), expected_gain

        x(:, 1) = [0.0_dp, 1.0_dp, 2.0_dp, 3.0_dp]
        y = [0.0_dp, 0.0_dp, 4.0_dp, 4.0_dp]
        options%n_estimators = 1
        options%learning_rate = 1.0_dp
        options%l2 = 1.0_dp
        options%min_child_weight = 0.0_dp
        call model%fit_regression(x, y, status, options)
        call model%predict(x, prediction, status)
        x_dot(:, 1) = [0.2_dp, -0.1_dp, 0.3_dp, -0.2_dp]
        call model%predict_jvp(x, x_dot, prediction, prediction_dot, status)
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
            maxval(abs(prediction_dot)) > 2.0e-14_dp .or. &
            maxval(abs(weights - [-4.0_dp/3.0_dp, 4.0_dp/3.0_dp])) > 2.0e-13_dp .or. &
            abs(model%split_gain(1) - expected_gain) > 2.0e-13_dp) then
            write (error_unit, '(a,es12.4)') &
                "FAIL [xgb squared] second-order split oracle ", &
                maxval(abs(prediction - expected))
            failures = failures + 1
        end if
        boundary(1, 1) = 1.5_dp
        boundary_dot(1, 1) = 1.0_dp
        call model%predict_jvp(boundary, boundary_dot, boundary_value, &
            prediction_dot(:1), status)
        if (status%code /= FORTNUM_DOMAIN_ERROR) then
            write (error_unit, '(a)') &
                "FAIL [xgb squared] split-boundary derivative refusal"
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
        real(dp) :: output_bar(3), x_bar(3, 1), x_dot(3, 1), prediction_dot(3)
        real(dp) :: boundary(1, 1), boundary_bar(1), boundary_x_bar(1, 1)
        real(dp) :: invalid_weight(3)

        x(:, 1) = [0.0_dp, 1.0_dp, 2.0_dp]
        y = [0.0_dp, 1.0_dp, 2.0_dp]
        options%objective = "unsupported"
        call model%fit(x, y, status, options)
        if (status%code /= FORTNUM_DOMAIN_ERROR) then
            write (error_unit, '(a)') "FAIL [xgb refusal] unsupported objective"
            failures = failures + 1
        end if
        options = xgboost_options_t()
        options%max_depth = 0
        call model%fit(x, y, status, options)
        if (status%code /= FORTNUM_DOMAIN_ERROR) then
            write (error_unit, '(a)') "FAIL [xgb refusal] invalid depth"
            failures = failures + 1
        end if
        options = xgboost_options_t()
        options%tree_method = "hist"
        options%max_bin = 1
        call model%fit(x, y, status, options)
        if (status%code /= FORTNUM_DOMAIN_ERROR) then
            write (error_unit, '(a)') "FAIL [xgb refusal] invalid histogram bins"
            failures = failures + 1
        end if
        options = xgboost_options_t()
        options%tree_method = "unsupported"
        call model%fit(x, y, status, options)
        if (status%code /= FORTNUM_DOMAIN_ERROR) then
            write (error_unit, '(a)') "FAIL [xgb refusal] unsupported tree method"
            failures = failures + 1
        end if
        invalid_weight = [1.0_dp, 0.0_dp, 1.0_dp]
        call model%fit_regression(x, y, status, sample_weight=invalid_weight)
        if (status%code /= FORTNUM_DOMAIN_ERROR) then
            write (error_unit, '(a)') "FAIL [xgb refusal] nonpositive sample weight"
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
        x_dot(:, 1) = [0.1_dp, -0.2_dp, 0.3_dp]
        output_bar = [0.7_dp, -0.4_dp, 0.2_dp]
        call model%predict_jvp(x, x_dot, prediction, prediction_dot, status)
        call model%predict_vjp(x, output_bar, x_bar, status)
        if (status%code /= FORTNUM_OK .or. maxval(abs(prediction_dot)) > 1.0e-14_dp .or. &
            maxval(abs(x_bar)) > 1.0e-14_dp .or. &
            abs(dot_product(output_bar, prediction_dot) - sum(x_bar*x_dot)) > 1.0e-14_dp) then
            write (error_unit, '(a)') "FAIL [xgb derivative] piecewise VJP oracle"
            failures = failures + 1
        end if
        boundary(1, 1) = 0.5_dp
        boundary_bar(1) = 1.0_dp
        call model%predict_vjp(boundary, boundary_bar, boundary_x_bar, status)
        if (status%code /= FORTNUM_DOMAIN_ERROR) then
            write (error_unit, '(a)') "FAIL [xgb derivative] split-boundary VJP refusal"
            failures = failures + 1
        end if
    end subroutine test_refusals

end program test_xgboost
