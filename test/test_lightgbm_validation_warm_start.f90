program test_lightgbm_validation_warm_start
    !! Independent weighted validation and patience oracle for transactional
    !! LightGBM warm-start continuation.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortml_lightgbm, only: lightgbm_t, lightgbm_options_t
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_DOMAIN_ERROR, &
        FORTNUM_NOT_IMPLEMENTED
    implicit none

    integer :: failures

    failures = 0
    call test_weighted_warm_oracle(failures)
    call test_transactional_refusals(failures)
    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, &
            " LightGBM validation warm-start test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS LightGBM validation warm-start independent oracle"

contains

    subroutine test_weighted_warm_oracle(failures)
        integer, intent(inout) :: failures
        type(lightgbm_t) :: prefix, restored, retained, reference
        type(lightgbm_options_t) :: options, fit_options
        type(fortnum_status_t) :: status
        real(dp) :: x(8, 1), y(8), validation_y(8), validation_weight(8)
        real(dp) :: prediction(8), expected_prediction(8), losses(4)
        real(dp) :: best_loss
        integer :: i, best_iteration, expected_stop

        x(:, 1) = real([(i, i = 0, 7)], dp)
        y = [0.0_dp, 0.0_dp, 0.0_dp, 0.0_dp, 10.0_dp, 10.0_dp, 10.0_dp, 10.0_dp]
        validation_y = 10.0_dp - y
        validation_weight = [1.0_dp, 2.0_dp, 1.0_dp, 2.0_dp, 1.0_dp, 2.0_dp, 1.0_dp, 2.0_dp]
        options = lightgbm_options_t()
        options%num_leaves = 2
        options%min_data_in_leaf = 1
        options%max_bin = 16
        options%learning_rate = 1.0_dp
        options%l2 = 1.0_dp

        ! Fit unrestricted prefixes and select the best round using only the
        ! independent weighted validation loss below.
        do i = 1, size(losses)
            fit_options = options
            fit_options%n_estimators = i
            call reference%fit_regression(x, y, status, fit_options)
            call reference%predict(x, prediction, status)
            losses(i) = weighted_loss(prediction, validation_y, validation_weight)
            if (.not. status_ok(status)) then
                failures = failures + 1
                return
            end if
        end do
        best_iteration = 1
        best_loss = losses(1)
        do i = 2, size(losses)
            if (losses(i) < best_loss) then
                best_loss = losses(i)
                best_iteration = i
            end if
        end do
        expected_stop = size(losses)
        do i = best_iteration + 2, size(losses)
            if (all(losses(i-1:i) >= best_loss)) then
                expected_stop = i
                exit
            end if
        end do

        options%n_estimators = 1
        call prefix%fit_regression(x, y, status, options)
        call check(status_ok(status), "warm prefix fit", failures)
        options%n_estimators = size(losses)
        options%early_stopping_rounds = 2
        options%restore_best = .true.
        call prefix%fit_warm_start(x, y, status, options, validation_x=x, &
            validation_y=validation_y, validation_weight=validation_weight)
        call check(status_ok(status) .and. prefix%early_stopped(), &
            "weighted warm early stop", failures)
        call check(prefix%best_iteration() == best_iteration .and. &
            prefix%estimator_count() == best_iteration .and. &
            abs(prefix%best_validation_loss()-best_loss) < 2.0e-12_dp, &
            "weighted warm restore-best metadata", failures)

        fit_options = options
        fit_options%n_estimators = best_iteration
        fit_options%early_stopping_rounds = 0
        fit_options%restore_best = .true.
        call reference%fit_regression(x, y, status, fit_options)
        call reference%predict(x, expected_prediction, status)
        call prefix%predict(x, prediction, status)
        call check(status_ok(status) .and. maxval(abs(prediction-expected_prediction)) < &
            2.0e-12_dp, "weighted warm restored prediction", failures)

        fit_options = options
        fit_options%n_estimators = 1
        fit_options%early_stopping_rounds = 0
        call retained%fit_regression(x, y, status, fit_options)
        options%n_estimators = size(losses)
        options%restore_best = .false.
        call retained%fit_warm_start(x, y, status, options, validation_x=x, &
            validation_y=validation_y, validation_weight=validation_weight)
        call check(status_ok(status) .and. retained%early_stopped() .and. &
            retained%best_iteration() == best_iteration .and. &
            retained%estimator_count() == expected_stop .and. &
            retained%estimator_count() > retained%best_iteration(), &
            "weighted warm retain-all metadata", failures)
    end subroutine test_weighted_warm_oracle

    subroutine test_transactional_refusals(failures)
        integer, intent(inout) :: failures
        type(lightgbm_t) :: model
        type(lightgbm_options_t) :: options
        type(fortnum_status_t) :: status
        type(fortml_device_t) :: cuda
        real(dp) :: x(8, 1), y(8), bad_validation(3), before(8), after(8)
        integer :: i

        x(:, 1) = real([(i, i = 0, 7)], dp)
        y = [0.0_dp, 0.0_dp, 0.0_dp, 0.0_dp, 10.0_dp, 10.0_dp, 10.0_dp, 10.0_dp]
        options = lightgbm_options_t()
        options%n_estimators = 1
        options%num_leaves = 2
        options%min_data_in_leaf = 1
        options%max_bin = 16
        options%learning_rate = 1.0_dp
        options%l2 = 1.0_dp
        call model%fit_regression(x, y, status, options)
        call model%predict(x, before, status)
        options%n_estimators = 4
        options%early_stopping_rounds = 2
        bad_validation = 1.0_dp
        call model%fit_warm_start(x, y, status, options, validation_x=x, &
            validation_y=bad_validation)
        call check(status%code == FORTNUM_DOMAIN_ERROR .and. model%estimator_count() == 1, &
            "malformed validation transactional refusal", failures)
        call model%predict(x, after, status)
        call check(status_ok(status) .and. maxval(abs(after-before)) < 2.0e-12_dp, &
            "malformed validation preserves prefix", failures)

        cuda%kind = FORTML_DEVICE_CUDA
        cuda%selected = .true.
        cuda%available = .true.
        call model%predict_device(cuda, x, after, status)
        call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
            "validation warm CUDA typed refusal", failures)
    end subroutine test_transactional_refusals

    real(dp) function weighted_loss(prediction, target, weight) result(value)
        real(dp), intent(in) :: prediction(:), target(:), weight(:)
        value = 0.5_dp*sum(weight*(prediction-target)**2)/sum(weight)
    end function weighted_loss

    subroutine check(condition, label, failures)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: label
        integer, intent(inout) :: failures

        if (.not. condition) then
            write (error_unit, '(a)') "FAIL ["//trim(label)//"]"
            failures = failures + 1
        end if
    end subroutine check

end program test_lightgbm_validation_warm_start
