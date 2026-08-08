program test_xgboost_classifier
    !! Independent behavioral oracle for the binary XGBoost classifier facade.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite, ieee_value, &
        ieee_quiet_nan
    use fortml_xgboost, only: xgboost_options_t
    use fortml_xgboost_classifier, only: xgboost_classifier_t
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, &
        FORTML_DEVICE_CUDA
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_DOMAIN_ERROR, &
        FORTNUM_NOT_IMPLEMENTED
    implicit none

    integer :: failures

    failures = 0
    call test_classifier_contract(failures)
    call test_weighted_validation_and_missing(failures)
    call test_log_probability_products(failures)
    call test_derivative_and_device_contract(failures)
    call test_refusals(failures)
    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, &
            " XGBoost binary classifier test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS XGBoost binary classifier independent behavioral oracles"

contains

    subroutine test_classifier_contract(failures)
        integer, intent(inout) :: failures
        type(xgboost_classifier_t) :: model
        type(xgboost_options_t) :: options
        type(fortnum_status_t) :: status
        real(dp) :: x(8, 2), query(4, 2), probabilities(4, 2)
        real(dp) :: staged(4, 2, 3), margins(4, 3), importance(2)
        real(dp) :: expected_positive(4), normalized(2)
        integer :: labels(8), predicted(4), classes(2), stage

        x(:, 1) = [0.0_dp, 1.0_dp, 2.0_dp, 3.0_dp, 4.0_dp, 5.0_dp, &
            6.0_dp, 7.0_dp]
        x(:, 2) = [0.0_dp, 0.1_dp, -0.1_dp, 0.2_dp, 0.0_dp, -0.2_dp, &
            0.1_dp, -0.1_dp]
        labels = [-7, -7, -7, -7, 11, 11, 11, 11]
        query(:, 1) = [0.2_dp, 1.8_dp, 4.2_dp, 6.8_dp]
        query(:, 2) = 0.0_dp
        options%n_estimators = 3
        options%max_depth = 1
        options%learning_rate = 0.5_dp
        options%l2 = 1.0_dp
        options%min_child_weight = 0.0_dp
        options%monotone_constraints = [1, 0]
        call model%fit(x, labels, status, options)
        call model%predict_proba(query, probabilities, status)
        call model%predict(query, predicted, status)
        call model%decision_function_staged(query, margins, status)
        call model%predict_proba_staged(query, staged, status)
        call model%feature_importance(importance, status, "gain")
        call model%feature_importance(normalized, status, "gain", .true.)
        classes = model%classes()

        do stage = 1, 3
            expected_positive = 1.0_dp/(1.0_dp + exp(-margins(:, stage)))
            call check(maxval(abs(staged(:, 2, stage) - expected_positive)) < &
                3.0e-14_dp, "staged logit/probability link", failures)
            call check(maxval(abs(sum(staged(:, :, stage), dim=2) - 1.0_dp)) < &
                3.0e-14_dp, "staged simplex", failures)
        end do
        call check(status_ok(status) .and. model%fitted(), "classifier fit", failures)
        call check(all(classes == [-7, 11]), "arbitrary sorted class labels", failures)
        call check(model%feature_count() == 2 .and. model%estimator_count() == 3, &
            "classifier metadata", failures)
        call check(maxval(abs(staged(:, :, 3) - probabilities)) < 3.0e-14_dp, &
            "final stage equals probabilities", failures)
        call check(all(predicted == merge(11, -7, probabilities(:, 2) > &
            probabilities(:, 1))), "argmax label prediction", failures)
        call check(importance(1) > 0.0_dp .and. normalized(1) > 0.0_dp .and. &
            abs(sum(normalized) - 1.0_dp) < 3.0e-14_dp, &
            "feature importance diagnostics", failures)
        call check(model%monotone_constraint(1) == 1 .and. &
            model%monotone_constraint(2) == 0, "monotone metadata", failures)
    end subroutine test_classifier_contract

    subroutine test_weighted_validation_and_missing(failures)
        integer, intent(inout) :: failures
        type(xgboost_classifier_t) :: weighted, missing, rejected
        type(xgboost_options_t) :: options
        type(fortnum_status_t) :: status
        real(dp) :: x(8, 1), validation_x(4, 1), weights(8), validation_weight(4)
        real(dp) :: probabilities(4, 2), missing_probabilities(2, 2)
        integer :: labels(8), validation_labels(4)

        x(:, 1) = [0.0_dp, 1.0_dp, 2.0_dp, 3.0_dp, 4.0_dp, 5.0_dp, &
            6.0_dp, 7.0_dp]
        labels = [-3, -3, -3, -3, 9, 9, 9, 9]
        validation_x(:, 1) = [0.5_dp, 2.5_dp, 4.5_dp, 6.5_dp]
        validation_labels = [-3, -3, 9, 9]
        weights = [1.0_dp, 2.0_dp, 1.0_dp, 2.0_dp, 1.0_dp, 2.0_dp, 1.0_dp, 2.0_dp]
        validation_weight = 1.0_dp
        options%n_estimators = 4
        options%max_depth = 1
        options%learning_rate = 0.4_dp
        options%l2 = 1.0_dp
        options%min_child_weight = 0.0_dp
        options%early_stopping_rounds = 2
        call weighted%fit(x, labels, status, options, weights, validation_x, &
            validation_labels, validation_weight)
        if (.not. status_ok(status)) write (error_unit, '(a,i0,1x,a)') &
            "weighted fit status ", status%code, trim(status%msg)
        call weighted%predict_proba(validation_x, probabilities, status)
        call check(status_ok(status) .and. all(ieee_is_finite(probabilities)), &
            "weighted validation fit", failures)
        call check(weighted%best_iteration() >= 1 .and. &
            weighted%best_iteration() <= options%n_estimators, &
            "validation best iteration metadata", failures)

        x(8, 1) = ieee_value(0.0_dp, ieee_quiet_nan)
        options = xgboost_options_t()
        options%n_estimators = 2
        options%max_depth = 1
        options%learning_rate = 0.5_dp
        options%l2 = 1.0_dp
        options%min_child_weight = 0.0_dp
        call rejected%fit(x, labels, status, options)
        call check(status%code == FORTNUM_DOMAIN_ERROR, &
            "default missing policy fit refusal", failures)
        options%missing_policy = "learn"
        call missing%fit(x, labels, status, options)
        call missing%predict_proba(x(7:8, :), missing_probabilities, status)
        call check(status_ok(status) .and. missing%accepts_missing() .and. &
            maxval(abs(sum(missing_probabilities, dim=2) - 1.0_dp)) < 3.0e-14_dp, &
            "learned missing routing", failures)
    end subroutine test_weighted_validation_and_missing

    subroutine test_log_probability_products(failures)
        integer, intent(inout) :: failures
        type(xgboost_classifier_t) :: model, categorical
        type(xgboost_options_t) :: options
        type(fortnum_status_t) :: status
        real(dp) :: x(8, 2), query(3, 2), tangent(3, 2)
        real(dp) :: log_probabilities(3, 2), probabilities(3, 2)
        real(dp) :: log_probabilities_dot(3, 2), x_bar(3, 2), cotangent(3, 2)
        real(dp) :: finite_difference(3, 2), plus(3, 2), minus(3, 2)
        integer :: labels(8), i

        x(:, 1) = [0.0_dp, 1.0_dp, 2.0_dp, 3.0_dp, 4.0_dp, 5.0_dp, &
            6.0_dp, 7.0_dp]
        x(:, 2) = [0.0_dp, 1.0_dp, 0.0_dp, 1.0_dp, 0.0_dp, 1.0_dp, &
            0.0_dp, 1.0_dp]
        labels = [-9, -9, -9, -9, 13, 13, 13, 13]
        query(:, 1) = [0.25_dp, 3.25_dp, 6.75_dp]
        query(:, 2) = [0.0_dp, 1.0_dp, 0.0_dp]
        options%n_estimators = 3
        options%max_depth = 1
        options%learning_rate = 0.4_dp
        options%l2 = 1.0_dp
        options%min_child_weight = 0.0_dp
        call model%fit(x, labels, status, options)
        call model%predict_log_proba(query, log_probabilities, status)
        call model%predict_proba(query, probabilities, status)
        call check(status_ok(status) .and. all(ieee_is_finite(log_probabilities)), &
            "finite log probabilities", failures)
        call check(maxval(abs(exp(log_probabilities) - probabilities)) < 3.0e-14_dp, &
            "log probability exponential link", failures)
        call check(maxval(abs(exp(log_probabilities(:, 1)) + &
            exp(log_probabilities(:, 2)) - 1.0_dp)) < 3.0e-14_dp, &
            "log probability simplex", failures)

        tangent = 1.0_dp
        call model%predict_log_proba_jvp(query, tangent, log_probabilities, &
            log_probabilities_dot, status)
        call check(status_ok(status) .and. maxval(abs(log_probabilities_dot)) < &
            3.0e-14_dp, "piecewise log probability JVP", failures)
        cotangent = reshape([1.0_dp, -2.0_dp, 0.5_dp, 3.0_dp, -1.0_dp, 2.0_dp], &
            shape(cotangent))
        call model%predict_log_proba_vjp(query, cotangent, x_bar, status)
        call check(status_ok(status) .and. maxval(abs(x_bar)) < 3.0e-14_dp, &
            "piecewise log probability VJP", failures)

        finite_difference = 0.0_dp
        do i = 1, 2
            plus = query
            minus = query
            plus(:, i) = plus(:, i) + 1.0e-6_dp
            minus(:, i) = minus(:, i) - 1.0e-6_dp
            call model%predict_log_proba(plus, probabilities, status)
            call model%predict_log_proba(minus, log_probabilities, status)
            finite_difference(:, i) = maxval(abs((probabilities - log_probabilities)/2.0e-6_dp))
        end do
        call check(maxval(finite_difference) < 3.0e-8_dp, &
            "independent finite-difference log probability oracle", failures)

        categorical = model
        options%categorical_policy = "ordered"
        options%categorical_max_categories = 2
        options%categorical_features = [2]
        call categorical%fit(x, labels, status, options)
        call check(status_ok(status) .and. categorical%categorical_policy() == "ordered" .and. &
            categorical%categorical_max_categories() == 2 .and. &
            categorical%categorical_feature(2) .and. &
            categorical%interaction_group(1) == 0, &
            "classifier categorical metadata", failures)
        call categorical%predict_log_proba_jvp(query, tangent, log_probabilities, &
            log_probabilities_dot, status)
        call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
            "categorical log-probability JVP refusal", failures)
        call categorical%predict_log_proba_vjp(query, cotangent, x_bar, status)
        call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
            "categorical log-probability VJP refusal", failures)
    end subroutine test_log_probability_products

    subroutine test_derivative_and_device_contract(failures)
        integer, intent(inout) :: failures
        type(xgboost_classifier_t) :: model
        type(xgboost_options_t) :: options
        type(fortml_device_t) :: cpu, cuda
        type(fortnum_status_t) :: status
        real(dp) :: x(8, 1), query(2, 1), tangent(2, 1)
        real(dp) :: probabilities(2, 2), probabilities_dot(2, 2), cotangent(2, 2)
        real(dp) :: x_bar(2, 1), cpu_probabilities(2, 2)
        integer :: labels(8), predicted(2), cpu_predicted(2)

        x(:, 1) = [0.0_dp, 1.0_dp, 2.0_dp, 3.0_dp, 4.0_dp, 5.0_dp, &
            6.0_dp, 7.0_dp]
        labels = [-4, -4, -4, -4, 8, 8, 8, 8]
        options%n_estimators = 2
        options%max_depth = 1
        options%learning_rate = 0.5_dp
        options%l2 = 1.0_dp
        options%min_child_weight = 0.0_dp
        call model%fit(x, labels, status, options)
        query(:, 1) = [0.25_dp, 6.75_dp]
        tangent(:, 1) = 1.0_dp
        call model%predict_proba_jvp(query, tangent, probabilities, &
            probabilities_dot, status)
        cotangent = reshape([1.0_dp, -2.0_dp, 0.5_dp, 3.0_dp], shape(cotangent))
        call model%predict_proba_vjp(query, cotangent, x_bar, status)
        call check(status_ok(status) .and. maxval(abs(probabilities_dot)) < &
            3.0e-14_dp .and. maxval(abs(x_bar)) < 3.0e-14_dp, &
            "piecewise classifier input products", failures)

        cpu%kind = FORTML_DEVICE_CPU
        cpu%selected = .true.
        cpu%available = .true.
        cuda%kind = FORTML_DEVICE_CUDA
        cuda%selected = .true.
        cuda%available = .true.
        call model%predict_proba_device(cpu, query, cpu_probabilities, status)
        call model%predict_device(cpu, query, cpu_predicted, status)
        call model%predict(query, predicted, status)
        call check(status_ok(status) .and. maxval(abs(cpu_probabilities - &
            probabilities)) < 3.0e-14_dp .and. all(cpu_predicted == predicted), &
            "CPU device dispatch parity", failures)
        call model%predict_proba_device(cuda, query, cpu_probabilities, status)
        call check(status%code == FORTNUM_NOT_IMPLEMENTED .and. &
            .not. model%device_supported(FORTML_DEVICE_CUDA), &
            "typed CUDA probability refusal", failures)
        call model%predict_device(cuda, query, cpu_predicted, status)
        call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
            "typed CUDA label refusal", failures)
    end subroutine test_derivative_and_device_contract

    subroutine test_refusals(failures)
        integer, intent(inout) :: failures
        type(xgboost_classifier_t) :: model, rejected
        type(xgboost_options_t) :: options
        type(fortnum_status_t) :: status
        real(dp) :: x(6, 1), validation_x(2, 1)
        integer :: labels(6), validation_labels(2), ternary(6)

        x(:, 1) = [0.0_dp, 1.0_dp, 2.0_dp, 3.0_dp, 4.0_dp, 5.0_dp]
        labels = [-1, -1, -1, 7, 7, 7]
        validation_x(:, 1) = [0.5_dp, 3.5_dp]
        validation_labels = [-1, 9]
        ternary = [-1, 0, 7, 7, 0, -1]
        options%n_estimators = 1
        options%max_depth = 1
        options%learning_rate = 0.5_dp
        options%l2 = 1.0_dp
        options%min_child_weight = 0.0_dp
        call rejected%fit(x, ternary, status, options)
        call check(status%code == FORTNUM_DOMAIN_ERROR, &
            "more-than-two class refusal", failures)
        call rejected%fit(x, labels, status, options, &
            validation_x=validation_x, validation_labels=validation_labels)
        call check(status%code == FORTNUM_DOMAIN_ERROR, &
            "unknown validation class refusal", failures)
        call model%fit(x, labels, status, options)
        call check(status_ok(status), "refusal fixture fit", failures)
        call model%predict(x, labels(:2), status)
        call check(status%code == FORTNUM_DOMAIN_ERROR, &
            "prediction output shape refusal", failures)
    end subroutine test_refusals

    subroutine check(condition, name, failures)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: name
        integer, intent(inout) :: failures

        if (.not. condition) then
            write (error_unit, '(a)') "FAIL ["//trim(name)//"]"
            failures = failures + 1
        end if
    end subroutine check

end program test_xgboost_classifier
