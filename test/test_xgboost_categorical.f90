program test_xgboost_categorical
    !! Independent behavioral oracle for the bounded ordered-gradient
    !! categorical XGBoost policy.  The two-category grouping is known in
    !! advance, so the expected one-tree predictions do not depend on model
    !! internals or serialized node layout.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_xgboost, only: xgboost_t, xgboost_options_t
    use fortnum_status, only: fortnum_status_t, FORTNUM_OK, FORTNUM_DOMAIN_ERROR, &
        FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    implicit none

    type(xgboost_t) :: model, restored
    type(xgboost_options_t) :: options, too_many, changed_metadata
    type(fortml_device_t) :: cpu, cuda
    type(fortnum_status_t) :: status
    real(dp) :: x(8, 2), y(8), prediction(8), restored_prediction(8)
    real(dp) :: tangent(8, 2), y_dot(8), output_bar(8), x_bar(8, 2)
    character(*), parameter :: path = "test_xgboost_categorical.txt"
    integer :: failures

    failures = 0
    x(:, 1) = [0.0_dp, 0.0_dp, 1.0_dp, 1.0_dp, 2.0_dp, 2.0_dp, 3.0_dp, 3.0_dp]
    x(:, 2) = 0.0_dp
    y = [0.0_dp, 0.0_dp, 0.0_dp, 0.0_dp, 4.0_dp, 4.0_dp, 4.0_dp, 4.0_dp]
    options = xgboost_options_t()
    options%n_estimators = 1
    options%max_depth = 1
    options%learning_rate = 1.0_dp
    options%l2 = 0.0_dp
    options%min_child_weight = 0.0_dp
    options%categorical_policy = "ordered"
    options%categorical_max_categories = 4
    options%categorical_features = [1]

    call model%fit_regression(x, y, status, options)
    call check(status%code == FORTNUM_OK, "ordered categorical fit", failures)
    call model%predict(x, prediction, status)
    call check(status%code == FORTNUM_OK .and. maxval(abs(prediction - y)) < 2.0e-13_dp, &
        "ordered-gradient partition oracle", failures)
    call check(model%categorical_policy() == "ordered" .and. &
        model%categorical_max_categories() == 4 .and. model%categorical_feature(1), &
        "categorical metadata accessors", failures)

    call model%save_text(path, status)
    call check(status%code == FORTNUM_OK, "categorical metadata save", failures)
    call restored%load_text(path, status)
    call restored%predict(x, restored_prediction, status)
    call check(status%code == FORTNUM_OK .and. restored%categorical_feature(1) .and. &
        maxval(abs(restored_prediction - prediction)) < 2.0e-13_dp, &
        "categorical metadata round trip", failures)
    call delete_file(path)

    changed_metadata = options
    changed_metadata%n_estimators = 2
    changed_metadata%categorical_features = [2]
    call model%fit_warm_start(x, y, status, changed_metadata)
    call check(status%code == FORTNUM_DOMAIN_ERROR, &
        "warm-start categorical metadata refusal", failures)

    x(1, 1) = 0.25_dp
    call model%predict(x, prediction, status)
    call check(status%code == FORTNUM_DOMAIN_ERROR, &
        "non-integer categorical query refusal", failures)
    x(1, 1) = 0.0_dp

    tangent = 1.0_dp
    call model%predict_jvp(x, tangent, prediction, y_dot, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "categorical JVP discrete refusal", failures)
    output_bar = 1.0_dp
    call model%predict_vjp(x, output_bar, x_bar, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "categorical VJP discrete refusal", failures)

    cpu%kind = FORTML_DEVICE_CPU
    cpu%selected = .true.
    cpu%available = .true.
    call model%predict_device(cpu, x, restored_prediction, status)
    call check(status%code == FORTNUM_OK .and. maxval(abs(restored_prediction - prediction)) < &
        2.0e-13_dp, "CPU categorical dispatch", failures)
    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call model%predict_device(cuda, x, restored_prediction, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED .and. &
        .not. model%device_supported(FORTML_DEVICE_CUDA), "CUDA typed refusal", failures)

    too_many = options
    too_many%categorical_max_categories = 3
    call model%fit_regression(x, y, status, too_many)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "categorical cardinality refusal", failures)

    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, " XGBoost categorical test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS XGBoost categorical oracle"

contains

    subroutine delete_file(name)
        character(*), intent(in) :: name
        integer :: unit, ios

        open(newunit=unit, file=name, status="old", action="read", iostat=ios)
        if (ios == 0) close(unit, status="delete")
    end subroutine delete_file

    subroutine check(condition, name, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: name
        integer, intent(inout) :: failures

        if (.not. condition) then
            write (error_unit, '(a)') "FAIL ["//trim(name)//"]"
            failures = failures + 1
        end if
    end subroutine check

end program test_xgboost_categorical
