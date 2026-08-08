program test_xgboost_gamma
    !! Independent value/gradient/Hessian and fit oracles for reg:gamma.
    use, intrinsic :: iso_fortran_env, only: real64, error_unit
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    use fortml_xgboost, only: xgboost_t, xgboost_options_t, &
        xgb_gamma_loss, xgb_gamma_derivatives
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_DOMAIN_ERROR, &
        FORTNUM_NOT_IMPLEMENTED
    implicit none

    integer :: failures

    failures = 0
    call test_derivative_oracle(failures)
    call test_fit_and_prediction_semantics(failures)
    call test_domain_and_device_contract(failures)
    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, " XGBoost Gamma test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS XGBoost Gamma independent behavioral oracles"

contains

    subroutine test_derivative_oracle(failures)
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status
        real(real64) :: margin(3), target(3), direction(3), gradient(3), hessian(3)
        real(real64) :: value_plus, value_minus, value, shape, h
        real(real64) :: expected_gradient(3), expected_hessian(3), inverse_mean
        integer :: i

        margin = [-0.4_real64, 0.0_real64, 0.7_real64]
        target = [0.5_real64, 1.5_real64, 4.0_real64]
        direction = [0.2_real64, -0.3_real64, 0.5_real64]
        shape = 2.3_real64
        call xgb_gamma_derivatives(margin, target, shape, gradient, hessian, status)
        call check(status_ok(status), "Gamma derivative status", failures)
        expected_gradient = 0.0_real64
        expected_hessian = 0.0_real64
        do i = 1, 3
            inverse_mean = exp(-margin(i))
            expected_gradient(i) = shape*(1.0_real64 - target(i)*inverse_mean)
            expected_hessian(i) = shape*target(i)*inverse_mean
        end do
        call check(maxval(abs(gradient - expected_gradient)) < 2.0e-14_real64, &
            "Gamma gradient formula", failures)
        call check(maxval(abs(hessian - expected_hessian)) < 2.0e-14_real64, &
            "Gamma Hessian formula", failures)
        call xgb_gamma_loss(margin, target, shape, value, status)
        h = 1.0e-6_real64
        call xgb_gamma_loss(margin + h*direction, target, shape, value_plus, status)
        call xgb_gamma_loss(margin - h*direction, target, shape, value_minus, status)
        call check(abs((value_plus - value_minus)/(2.0_real64*h) - &
            sum(expected_gradient*direction)/3.0_real64) < 1.0e-9_real64, &
            "Gamma value/gradient finite-difference oracle", failures)
    end subroutine test_derivative_oracle

    subroutine test_fit_and_prediction_semantics(failures)
        integer, intent(inout) :: failures
        type(xgboost_t) :: model, alias_model
        type(xgboost_options_t) :: options, alias_options
        type(fortnum_status_t) :: status
        real(real64) :: x(4, 1), target(4), prediction(4), margin(4), staged(4, 1)
        real(real64) :: expected_base

        x(:, 1) = [0.0_real64, 1.0_real64, 2.0_real64, 3.0_real64]
        target = [1.0_real64, 1.0_real64, 9.0_real64, 9.0_real64]
        options%n_estimators = 1
        options%max_depth = 1
        options%learning_rate = 1.0_real64
        options%l2 = 0.0_real64
        options%min_child_weight = 0.0_real64
        options%gamma_shape = 2.0_real64
        call model%fit_gamma(x, target, status, options)
        call model%predict_margin(x, margin, status)
        call model%predict(x, prediction, status)
        call model%predict_staged(x, staged, status)
        expected_base = log(5.0_real64)
        call check(status_ok(status), "Gamma fit/predict status", failures)
        call check(trim(model%objective_name()) == "gamma", &
            "Gamma objective name", failures)
        call check(abs(model%objective_parameter_value() - 2.0_real64) < &
            2.0e-14_real64, "Gamma shape metadata", failures)
        call check(abs(model%base_margin() - expected_base) < 2.0e-14_real64, &
            "Gamma log-mean base margin", failures)
        call check(all(prediction > 0.0_real64), &
            "Gamma inverse-link predictions are positive", failures)
        call check(maxval(abs(prediction - exp(margin))) < 2.0e-13_real64, &
            "Gamma inverse-link prediction semantics", failures)
        call check(maxval(abs(staged(:, 1) - prediction)) < 2.0e-13_real64, &
            "Gamma staged prediction parity", failures)

        alias_options = options
        alias_options%objective = "reg:gamma"
        call alias_model%fit(x, target, status, alias_options)
        call alias_model%predict(x, prediction, status)
        call check(status_ok(status) .and. trim(alias_model%objective_name()) == "gamma", &
            "Gamma objective alias", failures)
    end subroutine test_fit_and_prediction_semantics

    subroutine test_domain_and_device_contract(failures)
        integer, intent(inout) :: failures
        type(xgboost_t) :: model
        type(xgboost_options_t) :: options
        type(fortnum_status_t) :: status
        type(fortml_device_t) :: cpu, cuda
        real(real64) :: x(2, 1), target(2), bad_target(2), prediction(2)

        x(:, 1) = [0.0_real64, 1.0_real64]
        target = [0.5_real64, 1.0_real64]
        bad_target = [0.0_real64, -1.0_real64]
        options%n_estimators = 1
        options%max_depth = 1
        options%min_child_weight = 0.0_real64
        options%gamma_shape = 0.0_real64
        call model%fit_gamma(x, target, status, options)
        call check(status%code == FORTNUM_DOMAIN_ERROR, &
            "nonpositive Gamma shape refusal", failures)
        options%gamma_shape = 2.0_real64
        call model%fit_gamma(x, bad_target, status, options)
        call check(status%code == FORTNUM_DOMAIN_ERROR, &
            "nonpositive Gamma target refusal", failures)
        call model%fit_gamma(x, target, status, options)
        call check(status_ok(status), "Gamma valid fit", failures)

        call cpu%select(FORTML_DEVICE_CPU, status)
        cuda%kind = FORTML_DEVICE_CUDA
        cuda%selected = .true.
        cuda%available = .true.
        call check(model%device_supported(FORTML_DEVICE_CPU), &
            "Gamma CPU capability", failures)
        call check(.not. model%device_supported(FORTML_DEVICE_CUDA), &
            "Gamma CUDA capability refusal", failures)
        call model%predict_device(cuda, x, prediction, status)
        call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
            "Gamma CUDA prediction refusal", failures)
        call model%predict_device(cpu, x, prediction, status)
        call check(status_ok(status), "Gamma CPU device dispatch", failures)
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

end program test_xgboost_gamma
