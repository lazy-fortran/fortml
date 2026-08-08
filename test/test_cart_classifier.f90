program test_cart_classifier
    use, intrinsic :: ieee_arithmetic, only: ieee_quiet_nan, ieee_value
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_cart_classifier, only: cart_classifier_t, &
        CART_CRITERION_GINI, CART_CRITERION_ENTROPY
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_DOMAIN_ERROR
    implicit none

    integer :: failures

    failures = 0
    call test_weighted_leaf_oracle(failures)
    call test_gini_split_oracle(failures)
    call test_entropy_split_oracle(failures)
    call test_tie_order_oracle(failures)
    call test_missing_routing(failures)
    call test_refusals(failures)
    if (failures /= 0) then
        write (error_unit, '(a,i0)') "FAIL CART classifier cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS CART classifier independent behavioral oracles"

contains

    subroutine test_weighted_leaf_oracle(failures)
        integer, intent(inout) :: failures
        type(cart_classifier_t) :: model
        type(fortnum_status_t) :: status
        real(dp) :: x(4, 1), probabilities(1, 2), weights(4)
        integer, allocatable :: classes(:)
        integer :: labels(4)

        x(:, 1) = [0.0_dp, 1.0_dp, 2.0_dp, 3.0_dp]
        labels = [10, 20, 20, 10]
        weights = [1.0_dp, 3.0_dp, 1.0_dp, 1.0_dp]
        call model%fit(x, labels, status, max_depth=0, sample_weight=weights)
        call model%predict_proba(reshape([1.5_dp], [1, 1]), probabilities, status)
        classes = model%classes()
        if (.not. status_ok(status) .or. model%class_count() /= 2 .or. &
            classes(1) /= 10 .or. classes(2) /= 20 .or. &
            maxval(abs(probabilities(1, :) - [2.0_dp/6.0_dp, 4.0_dp/6.0_dp])) &
            > 1.0e-13_dp .or. model%node_count() /= 1) then
            write (error_unit, '(a)') &
                "FAIL [CART classifier weighted leaf] class-frequency oracle"
            failures = failures + 1
        end if
    end subroutine test_weighted_leaf_oracle

    subroutine test_gini_split_oracle(failures)
        integer, intent(inout) :: failures
        type(cart_classifier_t) :: model
        type(fortnum_status_t) :: status
        real(dp) :: x(6, 1), query(4, 1), probabilities(4, 2)
        integer :: labels(6), prediction(4), expected(4)

        x(:, 1) = [0.0_dp, 1.0_dp, 2.0_dp, 3.0_dp, 4.0_dp, 5.0_dp]
        labels = [10, 10, 10, 20, 20, 20]
        query(:, 1) = [0.25_dp, 1.75_dp, 3.25_dp, 4.75_dp]
        expected = [10, 10, 20, 20]
        call model%fit(x, labels, status, max_depth=1, &
            criterion=CART_CRITERION_GINI)
        call model%predict(query, prediction, status)
        call model%predict_proba(query, probabilities, status)
        if (.not. status_ok(status) .or. model%node_count() /= 3 .or. &
            model%criterion() /= CART_CRITERION_GINI .or. &
            any(prediction /= expected) .or. &
            maxval(abs(probabilities - reshape([1.0_dp, 1.0_dp, 0.0_dp, 0.0_dp, &
            0.0_dp, 0.0_dp, 1.0_dp, 1.0_dp], shape(probabilities)))) > &
            1.0e-13_dp) then
            write (error_unit, '(a)') &
                "FAIL [CART classifier Gini] exhaustive split oracle"
            failures = failures + 1
        end if
    end subroutine test_gini_split_oracle

    subroutine test_entropy_split_oracle(failures)
        integer, intent(inout) :: failures
        type(cart_classifier_t) :: model
        type(fortnum_status_t) :: status
        real(dp) :: x(6, 1), probabilities(2, 2)
        integer :: labels(6), prediction(2)

        x(:, 1) = [0.0_dp, 1.0_dp, 2.0_dp, 3.0_dp, 4.0_dp, 5.0_dp]
        labels = [4, 4, 4, 9, 9, 9]
        call model%fit(x, labels, status, max_depth=1, &
            criterion=CART_CRITERION_ENTROPY)
        call model%predict(reshape([0.25_dp, 4.75_dp], [2, 1]), prediction, status)
        call model%predict_proba(reshape([0.25_dp, 4.75_dp], [2, 1]), probabilities, &
            status)
        if (.not. status_ok(status) .or. model%criterion() /= CART_CRITERION_ENTROPY &
            .or. any(prediction /= [4, 9]) .or. &
            maxval(abs(probabilities - reshape([1.0_dp, 0.0_dp, 0.0_dp, 1.0_dp], &
            shape(probabilities)))) > 1.0e-13_dp) then
            write (error_unit, '(a)') &
                "FAIL [CART classifier entropy] pure-leaf oracle"
            failures = failures + 1
        end if
    end subroutine test_entropy_split_oracle

    subroutine test_tie_order_oracle(failures)
        integer, intent(inout) :: failures
        type(cart_classifier_t) :: model
        type(fortnum_status_t) :: status
        real(dp) :: x(6, 2), query(2, 2)
        integer :: labels(6), prediction(2)

        x(:, 1) = [0.0_dp, 1.0_dp, 2.0_dp, 3.0_dp, 4.0_dp, 5.0_dp]
        x(:, 2) = [0.0_dp, 10.0_dp, 20.0_dp, 30.0_dp, 40.0_dp, 50.0_dp]
        labels = [3, 3, 3, 8, 8, 8]
        query(:, 1) = [0.0_dp, 4.0_dp]
        query(:, 2) = [50.0_dp, 50.0_dp]
        call model%fit(x, labels, status, max_depth=1)
        call model%predict(query, prediction, status)
        if (.not. status_ok(status) .or. any(prediction /= [3, 8])) then
            write (error_unit, '(a)') &
                "FAIL [CART classifier tie] ascending feature order"
            failures = failures + 1
        end if
    end subroutine test_tie_order_oracle

    subroutine test_missing_routing(failures)
        integer, intent(inout) :: failures
        type(cart_classifier_t) :: learned, finite_error, finite_learned
        type(fortnum_status_t) :: status
        real(dp) :: x(6, 1), finite_x(4, 1), query(4, 1)
        real(dp) :: probabilities(4, 2), finite_error_probabilities(4, 2)
        real(dp) :: finite_learned_probabilities(4, 2), nan_value
        integer :: labels(6), finite_labels(4), prediction(4)
        real(dp) :: wrong_shape(1, 1)

        nan_value = ieee_value(0.0_dp, ieee_quiet_nan)
        x(:, 1) = [0.0_dp, 1.0_dp, 2.0_dp, 3.0_dp, nan_value, nan_value]
        labels = [0, 0, 1, 1, 0, 1]
        query(:, 1) = [0.25_dp, 2.75_dp, nan_value, nan_value]
        call learned%fit(x, labels, status, max_depth=1, missing_policy="learn")
        call learned%predict_proba(query, probabilities, status)
        call learned%predict(query, prediction, status)
        call check(status_ok(status) .and. learned%missing_policy() == "learn" .and. &
            learned%accepts_missing() .and. all(prediction == [0, 1, 0, 0]) .and. &
            maxval(abs(probabilities(1, :) - [0.75_dp, 0.25_dp])) < 1.0e-13_dp .and. &
            maxval(abs(probabilities(2, :) - [0.0_dp, 1.0_dp])) < 1.0e-13_dp .and. &
            maxval(abs(probabilities(3, :) - [0.75_dp, 0.25_dp])) < 1.0e-13_dp .and. &
            maxval(abs(probabilities(4, :) - [0.75_dp, 0.25_dp])) < 1.0e-13_dp, &
            "learned NaN branch fit and prediction", failures)

        finite_x(:, 1) = x(:4, 1)
        finite_labels = labels(:4)
        query(:2, :) = reshape([0.25_dp, 2.75_dp], [2, 1])
        call finite_error%fit(finite_x, finite_labels, status, max_depth=1)
        call finite_error%predict_proba(query(:2, :), finite_error_probabilities(:2, :), status)
        call finite_learned%fit(finite_x, finite_labels, status, max_depth=1, &
            missing_policy="learn")
        call finite_learned%predict_proba(query(:2, :), finite_learned_probabilities(:2, :), &
            status)
        call check(status_ok(status) .and. maxval(abs(finite_error_probabilities(:2, :) - &
            finite_learned_probabilities(:2, :))) < 1.0e-13_dp, &
            "finite behavior is unchanged by learn policy", failures)

        call learned%predict_proba(reshape([0.5_dp], [1, 1]), wrong_shape, status)
        call check(status%code == FORTNUM_DOMAIN_ERROR, "missing model shape refusal", failures)
    end subroutine test_missing_routing

    subroutine test_refusals(failures)
        integer, intent(inout) :: failures
        type(cart_classifier_t) :: model
        type(fortnum_status_t) :: status
        real(dp) :: x(4, 1), query(1, 1), probabilities(1, 2), weights(4)
        real(dp) :: nan_value
        integer :: labels(4)

        nan_value = ieee_value(0.0_dp, ieee_quiet_nan)
        x(:, 1) = [0.0_dp, 1.0_dp, 2.0_dp, 3.0_dp]
        labels = [0, 0, 1, 1]
        weights = 1.0_dp
        x(2, 1) = nan_value
        call model%fit(x, labels, status)
        if (status%code /= FORTNUM_DOMAIN_ERROR) then
            write (error_unit, '(a)') "FAIL [CART classifier refusal] NaN fit"
            failures = failures + 1
        end if
        x(2, 1) = 1.0_dp
        weights(3) = 0.0_dp
        call model%fit(x, labels, status, sample_weight=weights)
        if (status%code /= FORTNUM_DOMAIN_ERROR) then
            write (error_unit, '(a)') &
                "FAIL [CART classifier refusal] nonpositive weight"
            failures = failures + 1
        end if
        weights = 1.0_dp
        call model%fit(x, labels, status, criterion=99)
        if (status%code /= FORTNUM_DOMAIN_ERROR) then
            write (error_unit, '(a)') &
                "FAIL [CART classifier refusal] unknown criterion"
            failures = failures + 1
        end if
        call model%fit(x, labels, status, missing_policy="unsupported")
        if (status%code /= FORTNUM_DOMAIN_ERROR) then
            write (error_unit, '(a)') &
                "FAIL [CART classifier refusal] unknown missing policy"
            failures = failures + 1
        end if
        call model%fit(x, labels, status)
        query(1, 1) = nan_value
        call model%predict_proba(query, probabilities, status)
        if (status%code /= FORTNUM_DOMAIN_ERROR) then
            write (error_unit, '(a)') &
                "FAIL [CART classifier refusal] NaN prediction"
            failures = failures + 1
        end if
    end subroutine test_refusals

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: description
        integer, intent(inout) :: failures

        if (.not. condition) then
            write (error_unit, '(a)') "FAIL [CART classifier missing] "//description
            failures = failures + 1
        end if
    end subroutine check

end program test_cart_classifier
