program test_xgboost_robust
    !! Independent behavioral oracles for Huber and quantile objectives.
    use, intrinsic :: iso_fortran_env, only: real64, error_unit
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    use fortml_xgboost, only: xgboost_t, xgboost_options_t
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_DOMAIN_ERROR, &
        FORTNUM_NOT_IMPLEMENTED
    implicit none

    integer :: failures

    failures = 0
    call test_huber_oracle(failures)
    call test_quantile_oracle(failures)
    call test_refusals_and_device(failures)
    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, &
            " XGBoost robust objective test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS XGBoost Huber/quantile independent behavioral oracles"

contains

    subroutine fixture(x, target)
        real(real64), intent(out) :: x(4, 1), target(4)

        x(:, 1) = [0.0_real64, 1.0_real64, 2.0_real64, 3.0_real64]
        target = [0.0_real64, 0.0_real64, 10.0_real64, 10.0_real64]
    end subroutine fixture

    subroutine test_huber_oracle(failures)
        integer, intent(inout) :: failures
        type(xgboost_t) :: model, hist_model
        type(xgboost_options_t) :: options, hist_options
        type(fortnum_status_t) :: status
        real(real64) :: x(4, 1), target(4), prediction(4), margin(4)
        real(real64) :: hist_prediction(4), expected(4)

        call fixture(x, target)
        options%n_estimators = 1
        options%max_depth = 1
        options%learning_rate = 1.0_real64
        options%l2 = 1.0_real64
        options%min_child_weight = 0.0_real64
        options%huber_delta = 1.0_real64
        call model%fit_huber(x, target, status, options)
        call model%predict(x, prediction, status)
        call model%predict_margin(x, margin, status)
        expected = [3.0_real64, 3.0_real64, 7.0_real64, 7.0_real64]
        call check(status_ok(status), "Huber fit/predict status", failures)
        call check(trim(model%objective_name()) == "huber", &
            "Huber objective name", failures)
        call check(abs(model%objective_parameter_value() - 1.0_real64) < 1.0e-14_real64, &
            "Huber delta metadata", failures)
        call check(abs(model%base_margin() - 5.0_real64) < 1.0e-14_real64, &
            "Huber mean base margin", failures)
        call check(maxval(abs(prediction - expected)) < 1.0e-10_real64, &
            "Huber one-tree Newton oracle", failures)
        call check(maxval(abs(margin - prediction)) < 2.0e-14_real64, &
            "Huber margin/prediction identity", failures)

        hist_options = options
        hist_options%tree_method = "hist"
        hist_options%max_bin = 2
        call hist_model%fit_huber(x, target, status, hist_options)
        call hist_model%predict(x, hist_prediction, status)
        call check(status_ok(status), "Huber histogram status", failures)
        call check(maxval(abs(hist_prediction - prediction)) < 1.0e-10_real64, &
            "Huber exact/hist parity", failures)
    end subroutine test_huber_oracle

    subroutine test_quantile_oracle(failures)
        integer, intent(inout) :: failures
        type(xgboost_t) :: model
        type(xgboost_options_t) :: options
        type(fortnum_status_t) :: status
        real(real64) :: x(4, 1), target(4), prediction(4), expected(4)

        call fixture(x, target)
        options%n_estimators = 1
        options%max_depth = 1
        options%learning_rate = 1.0_real64
        options%l2 = 1.0_real64
        options%min_child_weight = 0.0_real64
        options%quantile_alpha = 0.5_real64
        call model%fit_quantile(x, target, status, options)
        call model%predict(x, prediction, status)
        ! The weighted lower median is zero.  The alpha/alpha-1 subgradient
        ! gives left/right leaf corrections -1/+1 with L2=1.
        expected = [-1.0_real64, -1.0_real64, 1.0_real64, 1.0_real64]
        call check(status_ok(status), "quantile fit/predict status", failures)
        call check(trim(model%objective_name()) == "quantile", &
            "quantile objective name", failures)
        call check(abs(model%objective_parameter_value() - 0.5_real64) < 1.0e-14_real64, &
            "quantile alpha metadata", failures)
        call check(abs(model%base_margin()) < 1.0e-14_real64, &
            "quantile weighted median base margin", failures)
        call check(maxval(abs(prediction - expected)) < 1.0e-10_real64, &
            "quantile one-tree Newton oracle", failures)
    end subroutine test_quantile_oracle

    subroutine test_refusals_and_device(failures)
        integer, intent(inout) :: failures
        type(xgboost_t) :: model
        type(xgboost_options_t) :: options
        type(fortnum_status_t) :: status
        type(fortml_device_t) :: cpu, cuda
        real(real64) :: x(4, 1), target(4), prediction(4)

        call fixture(x, target)
        options%huber_delta = 0.0_real64
        call model%fit_huber(x, target, status, options)
        call check(status%code == FORTNUM_DOMAIN_ERROR, &
            "nonpositive Huber delta refusal", failures)
        options = xgboost_options_t()
        options%quantile_alpha = 1.0_real64
        call model%fit_quantile(x, target, status, options)
        call check(status%code == FORTNUM_DOMAIN_ERROR, &
            "unit quantile alpha refusal", failures)

        options = xgboost_options_t()
        options%n_estimators = 1
        options%max_depth = 1
        options%min_child_weight = 0.0_real64
        call model%fit_huber(x, target, status, options)
        call cpu%select(FORTML_DEVICE_CPU, status)
        cuda%kind = FORTML_DEVICE_CUDA
        cuda%selected = .true.
        cuda%available = .true.
        call model%predict_device(cuda, x, prediction, status)
        call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
            "Huber CUDA prediction refusal", failures)
        call model%predict_device(cpu, x, prediction, status)
        call check(status_ok(status) .and. model%device_supported(FORTML_DEVICE_CPU), &
            "Huber CPU device dispatch", failures)
        call check(.not. model%device_supported(FORTML_DEVICE_CUDA), &
            "Huber CUDA capability refusal", failures)
    end subroutine test_refusals_and_device

    subroutine check(condition, label, failures)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: label
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "  FAIL "//trim(label)
        end if
    end subroutine check

end program test_xgboost_robust
