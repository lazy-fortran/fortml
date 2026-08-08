program test_lightgbm_persistence
    !! Independent round-trip and malformed-record oracle for LightGBM text
    !! snapshots.  Predictions, staged margins, and additive terms are saved
    !! before writing and compared after loading; metadata is checked through
    !! the public accessors rather than private storage.
    use, intrinsic :: iso_fortran_env, only: real64, error_unit
    use fortml_lightgbm, only: lightgbm_t, lightgbm_options_t, &
        LIGHTGBM_MODEL_TEXT_MAGIC
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_DOMAIN_ERROR
    implicit none

    character(*), parameter :: path = "test_lightgbm_persistence.txt"
    type(lightgbm_t) :: source, restored, destination
    type(lightgbm_options_t) :: options
    type(fortnum_status_t) :: status
    real(real64) :: x(8, 2), target(8), before(8), after(8), before_margin(8), after_margin(8)
    real(real64) :: before_staged(8, 4), after_staged(8, 4)
    real(real64) :: before_contrib(8, 5), after_contrib(8, 5), destination_before(8), destination_after(8)
    integer :: failures, i, unit

    failures = 0
    do i = 1, size(x, 1)
        x(i, 1) = real(i-1, real64)
        x(i, 2) = real(mod(i, 3), real64) - 1.0_real64
        target(i) = merge(2.0_real64 + 0.5_real64*x(i, 2), &
            -1.0_real64 + 0.25_real64*x(i, 2), x(i, 1) >= 3.5_real64)
    end do
    options = lightgbm_options_t()
    options%n_estimators = 4
    options%num_leaves = 3
    options%min_data_in_leaf = 1
    options%max_bin = 16
    options%learning_rate = 0.25_real64
    options%l2 = 0.75_real64
    call source%fit_regression(x, target, status, options)
    call check(status_ok(status), "source fit", failures)
    call source%predict(x, before, status)
    call check(status_ok(status), "source prediction", failures)
    call source%predict_margin(x, before_margin, status)
    call check(status_ok(status), "source margin", failures)
    call source%predict_staged_margin(x, before_staged, status)
    call check(status_ok(status), "source staged margins", failures)
    call source%predict_contributions(x, before_contrib, status)
    call check(status_ok(status), "source contributions", failures)
    call source%save_text(path, status)
    call check(status_ok(status), "save text", failures)

    call restored%load_text(path, status)
    call check(status_ok(status) .and. restored%fitted(), "load text", failures)
    call restored%predict(x, after, status)
    call check(status_ok(status), "restored prediction", failures)
    call restored%predict_margin(x, after_margin, status)
    call check(status_ok(status), "restored margin", failures)
    call restored%predict_staged_margin(x, after_staged, status)
    call check(status_ok(status), "restored staged margins", failures)
    call restored%predict_contributions(x, after_contrib, status)
    call check(status_ok(status), "restored contributions", failures)
    call check(maxval(abs(after-before)) < 2.0e-13_real64 .and. &
        maxval(abs(after_margin-before_margin)) < 2.0e-13_real64 .and. &
        maxval(abs(after_staged-before_staged)) < 2.0e-13_real64 .and. &
        maxval(abs(after_contrib-before_contrib)) < 2.0e-13_real64, &
        "prediction/staged/contribution round trip", failures)
    call check(restored%estimator_count() == source%estimator_count() .and. &
        restored%feature_count() == source%feature_count() .and. &
        restored%num_leaves() == source%num_leaves() .and. &
        restored%best_iteration() == source%best_iteration(), &
        "metadata round trip", failures)

    call destination%fit_regression(x, target, status, options)
    call destination%predict(x, destination_before, status)
    call write_truncated(path//".truncated")
    call destination%load_text(path//".truncated", status)
    call check(status%code == FORTNUM_DOMAIN_ERROR, "truncated refusal", failures)
    call destination%predict(x, destination_after, status)
    call check(status_ok(status) .and. maxval(abs(destination_after-destination_before)) < &
        2.0e-13_real64, "truncated load leaves destination unchanged", failures)

    open(newunit=unit, file=path, status="old", position="append", action="write")
    write(unit, '(a)') "unexpected_record 1"
    close(unit)
    call destination%load_text(path, status)
    call check(status%code == FORTNUM_DOMAIN_ERROR, "trailing record refusal", failures)
    call destination%predict(x, destination_after, status)
    call check(status_ok(status) .and. maxval(abs(destination_after-destination_before)) < &
        2.0e-13_real64, "trailing load leaves destination unchanged", failures)

    call write_bad_schema(path//".schema")
    call destination%load_text(path//".schema", status)
    call check(status%code == FORTNUM_DOMAIN_ERROR, "schema refusal", failures)

    call delete_file(path)
    call delete_file(path//".truncated")
    call delete_file(path//".schema")
    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, " LightGBM persistence test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS LightGBM text persistence independent round-trip oracle"

contains

    subroutine write_truncated(filename)
        character(len=*), intent(in) :: filename
        integer :: local_unit

        open(newunit=local_unit, file=filename, status="replace", action="write")
        write(local_unit, '(a)') LIGHTGBM_MODEL_TEXT_MAGIC
        write(local_unit, '(a)') "schema_version 1"
        close(local_unit)
    end subroutine write_truncated

    subroutine write_bad_schema(filename)
        character(len=*), intent(in) :: filename
        integer :: local_unit

        open(newunit=local_unit, file=filename, status="replace", action="write")
        write(local_unit, '(a)') LIGHTGBM_MODEL_TEXT_MAGIC
        write(local_unit, '(a)') "schema_version 999"
        close(local_unit)
    end subroutine write_bad_schema

    subroutine delete_file(filename)
        character(len=*), intent(in) :: filename
        integer :: local_unit, local_iostat

        open(newunit=local_unit, file=filename, status="old", action="read", iostat=local_iostat)
        if (local_iostat == 0) close(local_unit, status="delete")
    end subroutine delete_file

    subroutine check(condition, label, failures)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: label
        integer, intent(inout) :: failures

        if (.not. condition) then
            write (error_unit, '(a)') "FAIL ["//trim(label)//"]"
            failures = failures + 1
        end if
    end subroutine check

end program test_lightgbm_persistence
