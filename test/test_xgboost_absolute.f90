program test_xgboost_absolute
    !! Independent behavioral oracle for XGBoost-style absolute regression.
    use, intrinsic :: iso_fortran_env, only: real64, error_unit
    use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_quiet_nan
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    use fortml_xgboost, only: xgboost_t, xgboost_options_t
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_DOMAIN_ERROR, &
        FORTNUM_NOT_IMPLEMENTED
    implicit none

    integer :: failures
    type(xgboost_t) :: model, hist_model
    type(xgboost_options_t) :: options, hist_options
    type(fortnum_status_t) :: status
    type(fortml_device_t) :: cpu, cuda
    real(real64) :: x(4, 1), target(4), prediction(4), hist_prediction(4)
    real(real64) :: x_dot(4, 1), prediction_dot(4), expected(4)

    failures = 0
    x(:, 1) = [0.0_real64, 1.0_real64, 2.0_real64, 3.0_real64]
    target = [0.0_real64, 10.0_real64, 20.0_real64, 30.0_real64]
    options%n_estimators = 1
    options%max_depth = 1
    options%learning_rate = 1.0_real64
    options%l2 = 1.0_real64
    options%min_child_weight = 0.0_real64
    call model%fit_absolute(x, target, status, options)
    call model%predict(x, prediction, status)
    expected = [9.0_real64, 9.0_real64, 12.0_real64, 12.0_real64]
    call check(status_ok(status), "absolute fit/predict status", failures)
    call check(trim(model%objective_name()) == "absolute", &
        "absolute objective name", failures)
    call check(abs(model%base_margin() - 10.0_real64) < 1.0e-14_real64, &
        "weighted-median absolute base margin", failures)
    call check(maxval(abs(prediction - expected)) < 1.0e-10_real64, &
        "absolute one-tree Newton oracle", failures)

    hist_options = options
    hist_options%tree_method = "hist"
    hist_options%max_bin = 2
    call hist_model%fit_absolute(x, target, status, hist_options)
    call hist_model%predict(x, hist_prediction, status)
    call check(status_ok(status), "absolute histogram status", failures)
    call check(maxval(abs(hist_prediction - prediction)) < 1.0e-10_real64, &
        "absolute exact/hist parity", failures)

    x_dot = 0.0_real64
    call model%predict_jvp(x, x_dot, prediction, prediction_dot, status)
    call check(status_ok(status) .and. maxval(abs(prediction_dot)) < 1.0e-14_real64, &
        "absolute piecewise-constant input JVP", failures)

    call cpu%select(FORTML_DEVICE_CPU, status)
    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call check(model%device_supported(FORTML_DEVICE_CPU), &
        "absolute CPU capability", failures)
    call check(.not. model%device_supported(FORTML_DEVICE_CUDA), &
        "absolute CUDA capability refusal", failures)
    call model%predict_device(cuda, x, prediction, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "absolute CUDA prediction refusal", failures)
    call model%predict_device(cpu, x, prediction, status)
    call check(status_ok(status), "absolute CPU device dispatch", failures)

    options = xgboost_options_t()
    options%n_estimators = 1
    options%max_depth = 1
    call model%fit_absolute(x, [0.0_real64, 1.0_real64, 2.0_real64, &
        ieee_value(0.0_real64, ieee_quiet_nan)], status, options)
    call check(status%code == FORTNUM_DOMAIN_ERROR, &
        "absolute nonfinite target refusal", failures)

    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, " XGBoost absolute test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS XGBoost absolute independent behavioral oracle"

contains

    subroutine check(condition, label, failures)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: label
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "  FAIL "//trim(label)
        end if
    end subroutine check

end program test_xgboost_absolute
