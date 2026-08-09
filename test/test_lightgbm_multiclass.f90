program test_lightgbm_multiclass
    !! Independent behavioral oracles for the LightGBM-style OVR classifier.
    use, intrinsic :: iso_fortran_env, only: error_unit
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortml_lightgbm, only: lightgbm_options_t
    use fortml_lightgbm_multiclass, only: lightgbm_multiclass_t
    implicit none

    integer :: failures

    failures = 0
    call test_probability_contract(failures)
    call test_derivative_contract(failures)
    call test_validation_and_transaction(failures)
    call test_device_refusal(failures)
    if (failures > 0) then
        write (error_unit, '(i0,a)') failures, &
            " LightGBM multiclass test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS LightGBM multiclass independent behavioral oracles"

contains

    subroutine make_fixture(x, labels)
        real(dp), intent(out) :: x(:, :)
        integer, intent(out) :: labels(:)

        x(:, 1) = [-4.0_dp, -3.0_dp, -2.0_dp, -1.0_dp, 0.0_dp, 1.0_dp, &
            2.0_dp, 3.0_dp, 4.0_dp, 5.0_dp, 6.0_dp, 7.0_dp]
        x(:, 2) = [0.0_dp, 0.2_dp, -0.1_dp, 0.1_dp, 0.0_dp, -0.2_dp, &
            0.2_dp, -0.1_dp, 0.1_dp, 0.0_dp, -0.2_dp, 0.2_dp]
        labels = [7, -3, -3, -3, 7, 7, 11, 11, 11, 7, -3, 11]
    end subroutine make_fixture

    subroutine default_options(options)
        type(lightgbm_options_t), intent(out) :: options

        options%n_estimators = 2
        options%num_leaves = 3
        options%max_depth = 2
        options%min_data_in_leaf = 2
        options%max_bin = 16
        options%learning_rate = 0.4_dp
        options%l2 = 1.0_dp
        options%seed = 19
    end subroutine default_options

    subroutine test_probability_contract(failures)
        integer, intent(inout) :: failures
        type(lightgbm_multiclass_t) :: model
        type(lightgbm_options_t) :: options
        type(fortnum_status_t) :: status
        real(dp) :: x(12, 2), query(3, 2), probabilities(3, 3)
        real(dp) :: staged(3, 3, 2), margins(3, 3, 2), expected(3, 3)
        integer :: labels(12), predicted(3), classes(3), i, j, k

        call make_fixture(x, labels)
        call default_options(options)
        query(:, 1) = [-3.31_dp, 0.37_dp, 4.29_dp]
        query(:, 2) = [0.13_dp, -0.07_dp, 0.19_dp]
        call model%fit(x, labels, status, options)
        call check(status_ok(status), "multiclass fit status", failures)
        classes = model%classes()
        call check(all(classes == [-3, 7, 11]), "sorted integer class metadata", failures)
        call check(model%class_count() == 3 .and. model%feature_count() == 2, &
            "class and feature counts", failures)
        call model%predict_proba(query, probabilities, status)
        call check(status_ok(status), "multiclass probability status", failures)
        call check(all(ieee_is_finite(probabilities)), "finite probabilities", failures)
        call check(maxval(abs(sum(probabilities, dim=2) - 1.0_dp)) < 2.0e-14_dp, &
            "probabilities normalize", failures)
        call model%predict(query, predicted, status)
        call check(status_ok(status), "multiclass label status", failures)
        do i = 1, size(predicted)
            call check(any(predicted(i) == classes), "predicted label uses sorted class set", failures)
        end do
        call model%predict_proba_staged(query, staged, status)
        call model%decision_function_staged(query, margins, status)
        call check(status_ok(status), "staged products status", failures)
        call check(maxval(abs(staged(:, :, 2) - probabilities)) < 2.0e-14_dp, &
            "final staged probability equals prediction", failures)
        do k = 1, 2
            do i = 1, size(query, 1)
                do j = 1, 3
                    expected(i, j) = stable_sigmoid(margins(i, j, k))
                end do
                expected(i, :) = expected(i, :)/sum(expected(i, :))
            end do
            call check(maxval(abs(staged(:, :, k) - expected)) < 2.0e-14_dp, &
                "staged margin normalization oracle", failures)
        end do
    end subroutine test_probability_contract

    subroutine test_derivative_contract(failures)
        integer, intent(inout) :: failures
        type(lightgbm_multiclass_t) :: model
        type(lightgbm_options_t) :: options
        type(fortnum_status_t) :: status
        real(dp) :: x(12, 2), query(3, 2), query_dot(3, 2)
        real(dp) :: probabilities(3, 3), probabilities_dot(3, 3)
        real(dp) :: probabilities_bar(3, 3), query_bar(3, 2)
        integer :: labels(12)

        call make_fixture(x, labels)
        call default_options(options)
        query(:, 1) = [-3.31_dp, 0.37_dp, 4.29_dp]
        query(:, 2) = [0.13_dp, -0.07_dp, 0.19_dp]
        query_dot = reshape([0.07_dp, -0.03_dp, 0.02_dp, 0.01_dp, &
            -0.04_dp, 0.06_dp], shape(query))
        probabilities_bar = reshape([0.4_dp, -0.2_dp, 0.3_dp, -0.1_dp, &
            0.6_dp, -0.5_dp, 0.2_dp, 0.1_dp, -0.3_dp], shape(probabilities_bar))
        call model%fit(x, labels, status, options)
        call model%predict_proba_jvp(query, query_dot, probabilities, &
            probabilities_dot, status)
        call check(status_ok(status), "fixed-structure probability JVP status", failures)
        call check(maxval(abs(probabilities_dot)) < 2.0e-14_dp, &
            "piecewise-constant input JVP oracle", failures)
        call check(maxval(abs(sum(probabilities_dot, dim=2))) < 2.0e-14_dp, &
            "probability JVP preserves simplex", failures)
        call model%predict_proba_vjp(query, probabilities_bar, query_bar, status)
        call check(status_ok(status), "fixed-structure probability VJP status", failures)
        call check(maxval(abs(query_bar)) < 2.0e-14_dp, &
            "piecewise-constant input VJP oracle", failures)
    end subroutine test_derivative_contract

    subroutine test_validation_and_transaction(failures)
        integer, intent(inout) :: failures
        type(lightgbm_multiclass_t) :: model
        type(lightgbm_options_t) :: options
        type(fortnum_status_t) :: status
        real(dp) :: x(12, 2), validation_x(6, 2), before(6, 3), after(6, 3)
        real(dp) :: validation_weight(6)
        integer :: labels(12), validation_labels(6), invalid_labels(6)

        call make_fixture(x, labels)
        validation_x = x(1:6, :)
        validation_labels = labels(1:6)
        validation_weight = [1.0_dp, 2.0_dp, 1.0_dp, 1.0_dp, 2.0_dp, 3.0_dp]
        call default_options(options)
        options%n_estimators = 4
        options%early_stopping_rounds = 1
        options%early_stopping_min_delta = 1.0e6_dp
        options%restore_best = .true.
        call model%fit(x, labels, status, options, validation_x=validation_x, &
            validation_labels=validation_labels, validation_weight=validation_weight)
        call check(status_ok(status), "validation fit status", failures)
        call check(model%requested_estimator_count() == 4 .and. model%best_iteration() == 1 .and. &
            model%estimator_count() == 1 .and. model%early_stopped(), &
            "validation best-prefix metadata", failures)
        call model%predict_proba(validation_x, before, status)
        invalid_labels = validation_labels
        invalid_labels(1) = 999
        call model%fit(x, labels, status, options, validation_x=validation_x, &
            validation_labels=invalid_labels, validation_weight=validation_weight)
        call check(.not. status_ok(status), "unknown validation label refusal", failures)
        call model%predict_proba(validation_x, after, status)
        call check(status_ok(status), "transactional prediction after refused fit", failures)
        call check(maxval(abs(before - after)) < 2.0e-14_dp, &
            "refused fit leaves fitted model unchanged", failures)
    end subroutine test_validation_and_transaction

    subroutine test_device_refusal(failures)
        integer, intent(inout) :: failures
        type(lightgbm_multiclass_t) :: model
        type(lightgbm_options_t) :: options
        type(fortnum_status_t) :: status
        type(fortml_device_t) :: cuda
        real(dp) :: x(12, 2), probabilities(2, 3), query(2, 2)
        integer :: labels(12), predicted(2)

        call make_fixture(x, labels)
        call default_options(options)
        query = x(1:2, :)
        call model%fit(x, labels, status, options)
        cuda%kind = FORTML_DEVICE_CUDA
        cuda%selected = .true.
        cuda%available = .true.
        call model%predict_proba_device(cuda, query, probabilities, status)
        call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
            "typed multiclass CUDA probability refusal", failures)
        call model%predict_device(cuda, query, predicted, status)
        call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
            "typed multiclass CUDA label refusal", failures)
    end subroutine test_device_refusal

    real(dp) function stable_sigmoid(value) result(probability)
        real(dp), intent(in) :: value

        if (value >= 0.0_dp) then
            probability = 1.0_dp/(1.0_dp + exp(-value))
        else
            probability = exp(value)/(1.0_dp + exp(value))
        end if
    end function stable_sigmoid

    subroutine check(condition, message, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: message
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "FAIL "//trim(message)
        end if
    end subroutine check

end program test_lightgbm_multiclass
