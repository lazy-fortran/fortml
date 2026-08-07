program test_xgboost_squared_log
    !! Independent behavioral oracle for XGBoost squared-log/RMSLE fitting.
    use, intrinsic :: iso_fortran_env, only: real64, error_unit
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    use fortml_xgboost, only: xgboost_t, xgboost_options_t
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_DOMAIN_ERROR, &
        FORTNUM_NOT_IMPLEMENTED
    implicit none

    integer :: failures

    failures = 0
    call test_newton_oracle(failures)
    call test_weighted_and_alias(failures)
    call test_domain_and_device_contract(failures)
    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, &
            " XGBoost squared-log objective test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS XGBoost squared-log independent behavioral oracles"

contains

    subroutine fixture(x, target)
        real(real64), intent(out) :: x(4, 1), target(4)

        x(:, 1) = [0.0_real64, 1.0_real64, 2.0_real64, 3.0_real64]
        target = [0.0_real64, 0.0_real64, 3.0_real64, 3.0_real64]
    end subroutine fixture

    subroutine test_newton_oracle(failures)
        integer, intent(inout) :: failures
        type(xgboost_t) :: model, hist_model
        type(xgboost_options_t) :: options, hist_options
        type(fortnum_status_t) :: status
        real(real64) :: x(4, 1), target(4), prediction(4), margin(4)
        real(real64) :: staged(4, 1), expected_margin(4), expected(4)
        real(real64) :: hist_prediction(4)
        real(real64) :: transformed(4), residual(4), gradient(4), hessian(4)
        real(real64) :: base, left_correction, right_correction, l2

        call fixture(x, target)
        options%n_estimators = 1
        options%max_depth = 1
        options%learning_rate = 1.0_real64
        options%l2 = 0.7_real64
        options%min_child_weight = 0.0_real64
        call model%fit_squared_log(x, target, status, options)
        call model%predict(x, prediction, status)
        call model%predict_margin(x, margin, status)
        call model%predict_staged(x, staged, status)

        ! Independent one-split Newton calculation in the log1p coordinate.
        transformed = log(1.0_real64 + target)
        base = sum(transformed)/4.0_real64
        residual = base - transformed
        gradient = residual/exp(base)
        hessian = max((1.0_real64 - residual)/exp(base), 1.0e-12_real64)
        l2 = options%l2
        left_correction = -sum(gradient(1:2))/(sum(hessian(1:2)) + l2)
        right_correction = -sum(gradient(3:4))/(sum(hessian(3:4)) + l2)
        expected_margin = [base + left_correction, base + left_correction, &
            base + right_correction, base + right_correction]
        expected = exp(expected_margin) - 1.0_real64

        call check(status_ok(status), "squared-log fit/predict status", failures)
        call check(trim(model%objective_name()) == "squaredlog", &
            "squared-log objective name", failures)
        call check(abs(model%base_margin() - base) < 2.0e-14_real64, &
            "squared-log geometric base margin", failures)
        call check(maxval(abs(margin - expected_margin)) < 2.0e-12_real64, &
            "squared-log Newton margin oracle", failures)
        call check(maxval(abs(prediction - expected)) < 2.0e-12_real64, &
            "squared-log inverse-link oracle", failures)
        call check(maxval(abs(staged(:, 1) - prediction)) < 2.0e-13_real64, &
            "squared-log staged prediction parity", failures)
        call check(all(prediction >= -1.0_real64), &
            "squared-log inverse link lower bound", failures)

        hist_options = options
        hist_options%tree_method = "hist"
        hist_options%max_bin = 2
        call hist_model%fit_squared_log(x, target, status, hist_options)
        call hist_model%predict(x, hist_prediction, status)
        call check(status_ok(status), "histogram squared-log status", failures)
        call check(maxval(abs(hist_prediction - prediction)) < 2.0e-12_real64, &
            "squared-log exact/hist parity", failures)
    end subroutine test_newton_oracle

    subroutine test_weighted_and_alias(failures)
        integer, intent(inout) :: failures
        type(xgboost_t) :: model, alias_model
        type(xgboost_options_t) :: options, alias_options
        type(fortnum_status_t) :: status
        real(real64) :: x(4, 1), target(4), weights(4), prediction(4)
        real(real64) :: transformed(4), weighted_base

        call fixture(x, target)
        weights = [1.0_real64, 2.0_real64, 3.0_real64, 4.0_real64]
        options%n_estimators = 2
        options%max_depth = 1
        options%learning_rate = 0.25_real64
        options%l2 = 1.0_real64
        options%min_child_weight = 0.0_real64
        call model%fit_squared_log(x, target, status, options, weights)
        call model%predict(x, prediction, status)
        transformed = log(1.0_real64 + target)
        weighted_base = sum(weights*transformed)/sum(weights)
        call check(status_ok(status), "weighted squared-log fit", failures)
        call check(abs(model%base_margin() - weighted_base) < 2.0e-14_real64, &
            "weighted transformed base margin", failures)
        call check(all(prediction >= -1.0_real64), &
            "weighted squared-log finite predictions", failures)

        alias_options = options
        alias_options%objective = "reg:squaredlogerror"
        call alias_model%fit(x, target, status, alias_options, weights)
        call alias_model%predict(x, prediction, status)
        call check(status_ok(status), "squared-log objective alias fit", failures)
        call check(trim(alias_model%objective_name()) == "squaredlog", &
            "squared-log objective alias metadata", failures)
        call check(abs(alias_model%base_margin() - model%base_margin()) < &
            2.0e-14_real64, "squared-log alias base parity", failures)
    end subroutine test_weighted_and_alias

    subroutine test_domain_and_device_contract(failures)
        integer, intent(inout) :: failures
        type(xgboost_t) :: model
        type(xgboost_options_t) :: options
        type(fortnum_status_t) :: status
        type(fortml_device_t) :: cpu, cuda
        real(real64) :: x(2, 1), target(2), bad_target(2), prediction(2)

        x(:, 1) = [0.0_real64, 1.0_real64]
        target = [0.0_real64, 1.0_real64]
        bad_target = [0.0_real64, -1.0_real64]
        options%n_estimators = 1
        options%max_depth = 1
        options%min_child_weight = 0.0_real64
        call model%fit_squared_log(x, bad_target, status, options)
        call check(status%code == FORTNUM_DOMAIN_ERROR, &
            "negative squared-log target refusal", failures)
        call model%fit_squared_log(x, target, status, options)
        call cpu%select(FORTML_DEVICE_CPU, status)
        cuda%kind = FORTML_DEVICE_CUDA
        cuda%selected = .true.
        cuda%available = .true.
        call model%predict_device(cuda, x, prediction, status)
        call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
            "squared-log CUDA prediction refusal", failures)
        call model%predict_device(cpu, x, prediction, status)
        call check(status_ok(status) .and. model%device_supported(FORTML_DEVICE_CPU), &
            "squared-log CPU device dispatch", failures)
        call check(.not. model%device_supported(FORTML_DEVICE_CUDA), &
            "squared-log CUDA capability refusal", failures)
    end subroutine test_domain_and_device_contract

    subroutine check(condition, label, failures)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: label
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "  FAIL "//trim(label)
        end if
    end subroutine check

end program test_xgboost_squared_log
