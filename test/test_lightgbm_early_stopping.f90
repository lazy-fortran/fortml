program test_lightgbm_early_stopping
    !! Independent validation-loss and patience oracle for the LightGBM path.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_lightgbm, only: lightgbm_t, lightgbm_options_t
    use fortnum_status, only: fortnum_status_t, FORTNUM_OK, FORTNUM_DOMAIN_ERROR
    implicit none

    integer :: failures

    failures = 0
    call test_validation_oracle(failures)
    call test_binary_validation(failures)
    call test_configuration_refusals(failures)
    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, &
            " LightGBM early-stopping test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS LightGBM early-stopping independent behavioral oracle"

contains

    subroutine test_validation_oracle(failures)
        integer, intent(inout) :: failures
        type(lightgbm_t) :: full, one, early, retained
        type(lightgbm_options_t) :: options, early_options
        type(fortnum_status_t) :: status
        real(dp) :: x(8, 1), y(8), validation_y(8), prediction(8)
        real(dp) :: losses(4), expected(8), best_loss
        integer :: i, best_iteration, expected_stop

        x(:, 1) = real([(i, i = 0, 7)], dp)
        y = [0.0_dp, 0.0_dp, 0.0_dp, 0.0_dp, 10.0_dp, 10.0_dp, 10.0_dp, 10.0_dp]
        validation_y = 10.0_dp - y
        options = lightgbm_options_t()
        options%n_estimators = 4
        options%num_leaves = 2
        options%min_data_in_leaf = 1
        options%max_bin = 16
        options%learning_rate = 1.0_dp
        options%l2 = 1.0_dp

        ! Fit each unrestricted prefix independently.  The scalar validation
        ! losses are the oracle; no model best-round metadata is consulted.
        do i = 1, size(losses)
            options%n_estimators = i
            call one%fit_regression(x, y, status, options)
            call one%predict(x, prediction, status)
            losses(i) = 0.5_dp*sum((prediction-validation_y)**2)/size(y)
            if (status%code /= FORTNUM_OK) then
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

        options%n_estimators = size(losses)
        early_options = options
        early_options%early_stopping_rounds = 2
        early_options%restore_best = .true.
        call early%fit_regression(x, y, status, early_options, &
            validation_x=x, validation_y=validation_y)
        call early%predict(x, prediction, status)
        call one%fit_regression(x, y, status, options)
        call one%predict(x, expected, status)
        ! Refit the independently selected prefix so the restored prediction
        ! is compared with a model that never saw validation data.
        options%n_estimators = best_iteration
        call full%fit_regression(x, y, status, options)
        call full%predict(x, expected, status)
        if (status%code /= FORTNUM_OK .or. .not. early%early_stopped() .or. &
            early%best_iteration() /= best_iteration .or. &
            early%estimator_count() /= best_iteration .or. &
            abs(early%best_validation_loss()-best_loss) > 2.0e-12_dp) then
            write (error_unit, '(a,2(i0,1x),2(es12.4,1x))') &
                "FAIL [lgbm early] restore metadata", best_iteration, &
                early%best_iteration(), best_loss, early%best_validation_loss()
            failures = failures + 1
        end if
        call early%predict(x, prediction, status)
        if (status%code /= FORTNUM_OK .or. maxval(abs(prediction-expected)) > 2.0e-12_dp) then
            write (error_unit, '(a)') "FAIL [lgbm early] restored prediction"
            failures = failures + 1
        end if

        early_options%restore_best = .false.
        call retained%fit_regression(x, y, status, early_options, &
            validation_x=x, validation_y=validation_y)
        if (status%code /= FORTNUM_OK .or. .not. retained%early_stopped() .or. &
            retained%best_iteration() /= best_iteration .or. &
            retained%estimator_count() /= expected_stop .or. &
            retained%estimator_count() <= retained%best_iteration()) then
            write (error_unit, '(a,3(i0,1x))') "FAIL [lgbm early] retain metadata", &
                best_iteration, expected_stop, retained%estimator_count()
            failures = failures + 1
        end if
    end subroutine test_validation_oracle

    subroutine test_binary_validation(failures)
        integer, intent(inout) :: failures
        type(lightgbm_t) :: prefix, early
        type(lightgbm_options_t) :: options
        type(fortnum_status_t) :: status
        real(dp) :: x(8, 1), labels(8), validation_labels(8), probabilities(8, 2)
        real(dp) :: losses(4), probability, best_loss
        integer :: i, j, best_iteration

        x(:, 1) = real([(i, i = 0, 7)], dp)
        labels = [0.0_dp, 0.0_dp, 0.0_dp, 0.0_dp, 1.0_dp, 1.0_dp, 1.0_dp, 1.0_dp]
        validation_labels = 1.0_dp - labels
        options = lightgbm_options_t()
        options%num_leaves = 2
        options%min_data_in_leaf = 1
        options%max_bin = 16
        options%learning_rate = 1.0_dp
        options%l2 = 1.0_dp
        do i = 1, size(losses)
            options%n_estimators = i
            call prefix%fit_binary(x, labels, status, options)
            call prefix%predict_proba(x, probabilities, status)
            losses(i) = 0.0_dp
            do j = 1, size(labels)
                probability = min(max(probabilities(j, 2), 1.0e-15_dp), &
                    1.0_dp-1.0e-15_dp)
                losses(i) = losses(i) - (validation_labels(j)*log(probability) + &
                    (1.0_dp-validation_labels(j))*log(1.0_dp-probability))
            end do
            losses(i) = losses(i)/size(labels)
        end do
        best_iteration = 1
        best_loss = losses(1)
        do i = 2, size(losses)
            if (losses(i) < best_loss) then
                best_loss = losses(i)
                best_iteration = i
            end if
        end do
        options%n_estimators = size(losses)
        options%early_stopping_rounds = 2
        call early%fit_binary(x, labels, status, options, &
            validation_x=x, validation_y=validation_labels)
        if (status%code /= FORTNUM_OK .or. .not. early%early_stopped() .or. &
            early%best_iteration() /= best_iteration .or. &
            abs(early%best_validation_loss()-best_loss) > 2.0e-12_dp) then
            write (error_unit, '(a,2(i0,1x),2(es12.4,1x))') &
                "FAIL [lgbm binary early] validation oracle", best_iteration, &
                early%best_iteration(), best_loss, early%best_validation_loss()
            failures = failures + 1
        end if
    end subroutine test_binary_validation

    subroutine test_configuration_refusals(failures)
        integer, intent(inout) :: failures
        type(lightgbm_t) :: model
        type(lightgbm_options_t) :: options
        type(fortnum_status_t) :: status
        real(dp) :: x(4, 1), y(4), validation_x(4, 1), validation_y(3)

        x(:, 1) = [0.0_dp, 1.0_dp, 2.0_dp, 3.0_dp]
        y = [0.0_dp, 0.0_dp, 1.0_dp, 1.0_dp]
        validation_x = x
        validation_y = 1.0_dp
        options = lightgbm_options_t()
        options%early_stopping_rounds = 1
        call model%fit_binary(x, y, status, options)
        if (status%code /= FORTNUM_DOMAIN_ERROR) then
            write (error_unit, '(a)') &
                "FAIL [lgbm early] missing validation data was accepted"
            failures = failures + 1
        end if
        call model%fit_binary(x, y, status, options, validation_x=validation_x)
        if (status%code /= FORTNUM_DOMAIN_ERROR) then
            write (error_unit, '(a)') &
                "FAIL [lgbm early] validation_x without validation_y was accepted"
            failures = failures + 1
        end if
        call model%fit_binary(x, y, status, options, validation_x=validation_x, &
            validation_y=validation_y)
        if (status%code /= FORTNUM_DOMAIN_ERROR) then
            write (error_unit, '(a)') &
                "FAIL [lgbm early] mismatched validation target shape was accepted"
            failures = failures + 1
        end if
    end subroutine test_configuration_refusals

end program test_lightgbm_early_stopping
