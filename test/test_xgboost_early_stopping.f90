program test_xgboost_early_stopping
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_xgboost, only: xgboost_t, xgboost_options_t
    use fortnum_status, only: fortnum_status_t, FORTNUM_OK, FORTNUM_DOMAIN_ERROR
    implicit none

    integer :: failures

    failures = 0
    call test_validation_oracle(failures)
    call test_configuration_refusals(failures)
    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, " xgboost early-stopping test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS"

contains

    subroutine test_validation_oracle(failures)
        integer, intent(inout) :: failures
        type(xgboost_t) :: full, restored, retained
        type(xgboost_options_t) :: options, early_options
        type(fortnum_status_t) :: status
        real(dp) :: x(8, 1), y(8), validation_y(8), staged(8, 8)
        real(dp) :: prediction(8), expected(8), losses(8), best_loss
        integer :: i, best_iteration, expected_stop

        x(:, 1) = real([(i, i = 0, 7)], dp)
        y = [0.0_dp, 0.0_dp, 0.0_dp, 0.0_dp, 10.0_dp, 10.0_dp, 10.0_dp, 10.0_dp]
        validation_y = 10.0_dp - y
        options = xgboost_options_t()
        options%n_estimators = 8
        options%max_depth = 1
        options%learning_rate = 1.0_dp
        options%l2 = 1.0_dp
        options%min_child_weight = 0.0_dp

        ! Independent oracle: score every unrestricted stage with the
        ! definition of the squared validation objective, then derive the
        ! expected best round and patience stop from those scalar scores.
        call full%fit_regression(x, y, status, options)
        call full%predict_staged(x, staged, status)
        losses = 0.5_dp*sum((staged - spread(validation_y, 2, 8))**2, dim=1)/8.0_dp
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
            if (all(losses(i - 1:i) >= best_loss)) then
                expected_stop = i
                exit
            end if
        end do

        early_options = options
        early_options%early_stopping_rounds = 2
        early_options%restore_best = .true.
        call restored%fit_regression(x, y, status, early_options, &
            validation_x=x, validation_y=validation_y)
        call restored%predict(x, prediction, status)
        expected = staged(:, best_iteration)
        if (status%code /= FORTNUM_OK .or. .not. restored%early_stopped() .or. &
            restored%best_iteration() /= best_iteration .or. &
            restored%estimator_count() /= best_iteration .or. &
            abs(restored%best_validation_loss() - best_loss) > 2.0e-12_dp .or. &
            maxval(abs(prediction - expected)) > 2.0e-12_dp) then
            write (error_unit, '(a,2(i0,1x),2(es12.4,1x))') &
                "FAIL [xgb early] restore oracle", best_iteration, &
                restored%best_iteration(), best_loss, restored%best_validation_loss()
            failures = failures + 1
        end if

        early_options%restore_best = .false.
        call retained%fit_regression(x, y, status, early_options, &
            validation_x=x, validation_y=validation_y)
        if (status%code /= FORTNUM_OK .or. .not. retained%early_stopped() .or. &
            retained%best_iteration() /= best_iteration .or. &
            retained%estimator_count() /= expected_stop .or. &
            retained%estimator_count() <= retained%best_iteration()) then
            write (error_unit, '(a,3(i0,1x))') "FAIL [xgb early] retain oracle", &
                best_iteration, expected_stop, retained%estimator_count()
            failures = failures + 1
        end if
    end subroutine test_validation_oracle

    subroutine test_configuration_refusals(failures)
        integer, intent(inout) :: failures
        type(xgboost_t) :: model
        type(xgboost_options_t) :: options
        type(fortnum_status_t) :: status
        real(dp) :: x(4, 1), y(4), validation_x(4, 1), validation_y(3)

        x(:, 1) = [0.0_dp, 1.0_dp, 2.0_dp, 3.0_dp]
        y = [0.0_dp, 0.0_dp, 1.0_dp, 1.0_dp]
        validation_x = x
        validation_y = 1.0_dp
        options = xgboost_options_t()
        options%early_stopping_rounds = 1
        call model%fit_binary(x, y, status, options)
        if (status%code /= FORTNUM_DOMAIN_ERROR) then
            write (error_unit, '(a)') &
                "FAIL [xgb early] missing validation data was accepted"
            failures = failures + 1
        end if
        call model%fit_binary(x, y, status, options, validation_x=validation_x, &
            validation_y=validation_y)
        if (status%code /= FORTNUM_DOMAIN_ERROR) then
            write (error_unit, '(a)') &
                "FAIL [xgb early] mismatched validation target shape was accepted"
            failures = failures + 1
        end if
    end subroutine test_configuration_refusals

end program test_xgboost_early_stopping
