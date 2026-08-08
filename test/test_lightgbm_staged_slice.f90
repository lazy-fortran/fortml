program test_lightgbm_staged_slice
    !! Independent tree-walk oracle for LightGBM staged products and slicing.
    !! The expected margins below are computed from the two-child split,
    !! Newton leaf formula, and shrinkage recurrence; no model diagnostics or
    !! private tree fields are consulted.
    use, intrinsic :: iso_fortran_env, only: real64, error_unit
    use fortml_lightgbm, only: lightgbm_t, lightgbm_options_t
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_DOMAIN_ERROR
    implicit none

    type(lightgbm_t) :: model, prefix, destination, binary
    type(lightgbm_options_t) :: options, binary_options
    type(fortnum_status_t) :: status
    real(real64) :: x(4, 1), target(4), staged(4, 3), staged_margin(4, 3)
    real(real64) :: expected(4, 3), contributions(4, 4), prediction(4)
    real(real64) :: prefix_prediction(4), destination_before(4), destination_after(4)
    real(real64) :: labels(4), probabilities(4, 2), binary_staged(4, 2)
    real(real64) :: margin(4), gradient(4), left_weight, right_weight
    integer :: failures, round

    failures = 0
    x(:, 1) = [0.0_real64, 1.0_real64, 2.0_real64, 3.0_real64]
    target = [0.0_real64, 0.0_real64, 10.0_real64, 10.0_real64]
    options = lightgbm_options_t()
    options%n_estimators = 3
    options%num_leaves = 2
    options%min_data_in_leaf = 1
    options%max_bin = 16
    options%learning_rate = 0.5_real64
    options%l2 = 1.0_real64
    call model%fit_regression(x, target, status, options)
    call check(status_ok(status), "regression fit", failures)

    ! Independent split/tree walk: each round has the optimal split x=1.5.
    ! For four unit-weight rows, each child Hessian is two and the leaf
    ! correction is -sum(gradient)/(2+l2).
    margin = sum(target)/real(size(target), real64)
    do round = 1, 3
        gradient = margin-target
        left_weight = -sum(gradient(:2))/(2.0_real64+options%l2)
        right_weight = -sum(gradient(3:))/(2.0_real64+options%l2)
        margin(:2) = margin(:2) + options%learning_rate*left_weight
        margin(3:) = margin(3:) + options%learning_rate*right_weight
        expected(:2, round) = margin(:2)
        expected(3:, round) = margin(3:)
    end do

    call model%predict_staged_margin(x, staged_margin, status)
    call check(status_ok(status), "staged margin status", failures)
    call check(maxval(abs(staged_margin-expected)) < 2.0e-13_real64, &
        "independent split/tree-walk margin oracle", failures)
    call model%predict_staged(x, staged, status)
    call check(status_ok(status), "staged prediction status", failures)
    call check(maxval(abs(staged-expected)) < 2.0e-13_real64, &
        "regression staged prediction is margin", failures)

    call model%predict_contributions(x, contributions, status)
    call check(status_ok(status), "contribution status", failures)
    call model%predict_margin(x, prediction, status)
    call check(status_ok(status), "margin status", failures)
    call check(maxval(abs(sum(contributions, dim=2)-prediction)) < 2.0e-13_real64, &
        "contributions sum to margin", failures)
    call check(maxval(abs(sum(contributions(:, 1:3), dim=2)-staged_margin(:, 2))) < &
        2.0e-13_real64, "contribution prefix recurrence", failures)

    options%n_estimators = 2
    call model%slice(2, prefix, status)
    call check(status_ok(status) .and. prefix%estimator_count() == 2, &
        "transactional fitted prefix slice", failures)
    call prefix%predict(x, prefix_prediction, status)
    call check(status_ok(status), "prefix prediction status", failures)
    call check(maxval(abs(prefix_prediction-expected(:, 2))) < 2.0e-13_real64, &
        "slice equals independently walked prefix", failures)
    call check(maxval(abs(prefix_prediction-prediction)) > 1.0e-8_real64, &
        "slice differs from complete ensemble", failures)

    call destination%fit_regression(x, target, status, options)
    call destination%predict(x, destination_before, status)
    call destination%slice(0, prefix, status)
    call check(status%code == FORTNUM_DOMAIN_ERROR, "invalid slice refusal", failures)
    call destination%predict(x, destination_after, status)
    call check(status_ok(status) .and. maxval(abs(destination_before-destination_after)) < &
        2.0e-13_real64, "invalid slice leaves destination unchanged", failures)

    labels = [0.0_real64, 0.0_real64, 1.0_real64, 1.0_real64]
    binary_options = options
    binary_options%n_estimators = 2
    call binary%fit_binary(x, labels, status, binary_options)
    call binary%predict_staged(x, binary_staged, status)
    call check(status_ok(status), "binary staged status", failures)
    call binary%predict_proba(x, probabilities, status)
    call check(status_ok(status), "binary probability status", failures)
    call check(maxval(abs(binary_staged(:, 2)-probabilities(:, 2))) < 2.0e-13_real64 .and. &
        all(binary_staged >= 0.0_real64) .and. all(binary_staged <= 1.0_real64), &
        "binary staged probabilities and final parity", failures)

    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, " LightGBM staged/slice test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS LightGBM staged/slice independent tree-walk oracle"

contains

    subroutine check(condition, label, failures)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: label
        integer, intent(inout) :: failures

        if (.not. condition) then
            write (error_unit, '(a)') "FAIL ["//trim(label)//"]"
            failures = failures + 1
        end if
    end subroutine check

end program test_lightgbm_staged_slice
