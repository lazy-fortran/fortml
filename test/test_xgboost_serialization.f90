program test_xgboost_serialization
    !! Independent behavioral oracle for portable XGBoost tree persistence.
    use, intrinsic :: iso_fortran_env, only: real64, error_unit
    use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_quiet_nan
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_DOMAIN_ERROR
    use fortml_xgboost, only: xgboost_t, xgboost_options_t
    implicit none

    integer :: failures
    character(*), parameter :: path = "test_xgboost_serialization.txt"

    failures = 0
    call test_round_trip(failures)
    call test_refusals(failures)
    call execute_command_line("rm -f "//path)
    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, &
            " xgboost serialization test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS"

contains

    subroutine test_round_trip(failures)
        integer, intent(inout) :: failures
        type(xgboost_t) :: original, restored
        type(xgboost_options_t) :: options
        type(fortnum_status_t) :: status
        real(real64) :: x(10, 2), y(10), validation_y(10), query(7, 2)
        real(real64) :: before(7), after(7), margins_before(7), margins_after(7)
        real(real64) :: staged_before(7, 4), staged_after(7, 4)
        integer :: i

        do i = 1, 10
            x(i, 1) = real(i - 1, real64)
            x(i, 2) = real(mod(3*i + 1, 7), real64) - 2.0_real64
            y(i) = merge(4.0_real64, -1.0_real64, i > 5)
            validation_y(i) = 0.5_real64*y(i)
        end do
        x(3, 2) = ieee_value(x(3, 2), ieee_quiet_nan)
        query(:, 1) = [-1.0_real64, 0.0_real64, 1.5_real64, 3.5_real64, &
            5.0_real64, 7.0_real64, 9.0_real64]
        query(:, 2) = [-2.0_real64, -1.0_real64, 0.0_real64, 1.0_real64, &
            2.0_real64, -2.0_real64, 1.0_real64]
        options = xgboost_options_t()
        options%n_estimators = 4
        options%max_depth = 2
        options%learning_rate = 0.6_real64
        options%missing_policy = "learn"
        options%subsample = 0.8_real64
        options%colsample_bytree = 0.5_real64
        options%seed = 71
        options%early_stopping_rounds = 2
        options%restore_best = .false.
        options%monotone_constraints = [1, 0]
        call original%fit_regression(x, y, status, options, &
            validation_x=x, validation_y=validation_y)
        call check(status_ok(status), "fit before save", failures)
        call original%predict(query, before, status)
        call check(status_ok(status), "prediction before save", failures)
        call original%predict_margin(query, margins_before, status)
        call original%predict_staged(query, staged_before, status)
        call original%save_text(path, status)
        call check(status_ok(status), "save text", failures)
        call restored%load_text(path, status)
        call check(status_ok(status) .and. restored%fitted(), "load text", failures)
        call restored%predict(query, after, status)
        call check(status_ok(status) .and. maxval(abs(before - after)) < 2.0e-13_real64, &
            "prediction equivalence", failures)
        call restored%predict_margin(query, margins_after, status)
        call check(status_ok(status) .and. maxval(abs(margins_before - margins_after)) < &
            2.0e-13_real64, "margin equivalence", failures)
        call restored%predict_staged(query, staged_after, status)
        call check(status_ok(status) .and. maxval(abs(staged_before - staged_after)) < &
            2.0e-13_real64, "staged prediction equivalence", failures)
        call check(restored%estimator_count() == original%estimator_count() .and. &
            restored%best_iteration() == original%best_iteration() .and. &
            abs(restored%best_validation_loss() - original%best_validation_loss()) < &
            2.0e-13_real64 .and. restored%monotone_constraint(1) == 1 .and. &
            restored%monotone_constraint(2) == 0 .and. &
            restored%missing_policy() == "learn", "metadata equivalence", failures)
    end subroutine test_round_trip

    subroutine test_refusals(failures)
        integer, intent(inout) :: failures
        type(xgboost_t) :: model, destination
        type(xgboost_options_t) :: options
        type(fortnum_status_t) :: status
        real(real64) :: x(4, 1), y(4), prediction(4)
        integer :: unit, output_unit, ios, output_ios
        character(len=256) :: line

        x(:, 1) = [0.0_real64, 1.0_real64, 2.0_real64, 3.0_real64]
        y = [0.0_real64, 0.0_real64, 1.0_real64, 1.0_real64]
        options = xgboost_options_t()
        options%n_estimators = 2
        call model%fit_binary(x, y, status, options)
        call model%save_text(path, status)
        call check(status_ok(status), "refusal fixture save", failures)
        call destination%fit_binary(x, y, status, options)
        call destination%predict(x, prediction, status)

        ! A short file must be rejected before destination replacement.
        open(newunit=unit, file=path//".truncated", status="replace", action="write", &
            form="formatted")
        write(unit, '(A)') "FORTML_XGBOOST_TEXT"
        close(unit)
        call destination%load_text(path//".truncated", status)
        call check(status%code == FORTNUM_DOMAIN_ERROR .and. destination%fitted(), &
            "truncated snapshot refusal", failures)
        open(newunit=unit, file=path, status="old", action="read", form="formatted")
        open(newunit=output_unit, file=path//".unknown", status="replace", &
            action="write", form="formatted")
        do
            read(unit, '(A)', iostat=ios) line
            if (ios /= 0) exit
            write(output_unit, '(A)', iostat=output_ios) trim(line)
            if (output_ios /= 0) exit
        end do
        write(output_unit, '(A)') "unknown_record 1"
        close(unit)
        close(output_unit)
        call destination%load_text(path//".unknown", status)
        call check(status%code == FORTNUM_DOMAIN_ERROR .and. destination%fitted(), &
            "unknown record refusal", failures)
        call execute_command_line("rm -f "//path//".truncated "//path//".unknown")
    end subroutine test_refusals

    subroutine check(condition, label, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: label
        integer, intent(inout) :: failures

        if (.not. condition) then
            write (error_unit, '(a)') "FAIL [xgb serialization] "//label
            failures = failures + 1
        end if
    end subroutine check

end program test_xgboost_serialization
