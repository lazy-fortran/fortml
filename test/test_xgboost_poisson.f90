program test_xgboost_poisson
    !! Independent behavioral oracles for the Poisson count objective.
    use, intrinsic :: iso_fortran_env, only: real64, error_unit
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    use fortml_xgboost, only: xgboost_t, xgboost_options_t
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_DOMAIN_ERROR, &
        FORTNUM_NOT_IMPLEMENTED
    implicit none

    integer :: failures

    failures = 0
    call test_exact_poisson_oracle(failures)
    call test_histogram_and_products(failures)
    call test_domain_and_device_contract(failures)
    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, &
            " XGBoost Poisson test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS XGBoost Poisson independent behavioral oracles"

contains

    subroutine test_exact_poisson_oracle(failures)
        integer, intent(inout) :: failures
        type(xgboost_t) :: model
        type(xgboost_options_t) :: options
        type(fortnum_status_t) :: status
        real(real64) :: x(4, 1), target(4), prediction(4), margin(4), staged(4, 1)
        real(real64) :: expected_margin(4), expected_mean(4), base

        x(:, 1) = [0.0_real64, 1.0_real64, 2.0_real64, 3.0_real64]
        target = [1.0_real64, 1.0_real64, 9.0_real64, 9.0_real64]
        options%n_estimators = 1
        options%max_depth = 1
        options%learning_rate = 1.0_real64
        options%l2 = 0.0_real64
        options%min_child_weight = 0.0_real64
        call model%fit_poisson(x, target, status, options)
        call model%predict_margin(x, margin, status)
        call model%predict(x, prediction, status)
        call model%predict_staged(x, staged, status)

        ! Independent Newton oracle: base mean=5, gradient=[4,4,-4,-4],
        ! Hessian=5, and the best split is x=1.5 with leaf corrections +/-0.8.
        base = log(5.0_real64)
        expected_margin = [base - 0.8_real64, base - 0.8_real64, &
            base + 0.8_real64, base + 0.8_real64]
        expected_mean = exp(expected_margin)
        call check(status_ok(status), "exact Poisson status", failures)
        call check(trim(model%objective_name()) == "poisson", &
            "Poisson objective name", failures)
        call check(abs(model%base_margin() - base) < 2.0e-14_real64, &
            "Poisson log-mean base margin", failures)
        call check(maxval(abs(margin - expected_margin)) < 2.0e-13_real64, &
            "Poisson Newton margin oracle", failures)
        call check(maxval(abs(prediction - expected_mean)) < 2.0e-13_real64, &
            "Poisson inverse-link prediction oracle", failures)
        call check(maxval(abs(staged(:, 1) - prediction)) < 2.0e-13_real64, &
            "Poisson staged prediction parity", failures)
        call check(all(prediction > 0.0_real64), &
            "Poisson predictions are positive", failures)
    end subroutine test_exact_poisson_oracle

    subroutine test_histogram_and_products(failures)
        integer, intent(inout) :: failures
        type(xgboost_t) :: exact_model, hist_model
        type(xgboost_options_t) :: exact_options, hist_options
        type(fortnum_status_t) :: status
        real(real64) :: x(6, 1), target(6), exact_prediction(6), hist_prediction(6)
        real(real64) :: x_dot(6, 1), prediction_dot(6)

        x(:, 1) = [0.0_real64, 1.0_real64, 2.0_real64, 3.0_real64, &
            4.0_real64, 5.0_real64]
        target = [1.0_real64, 1.0_real64, 1.0_real64, 9.0_real64, &
            9.0_real64, 9.0_real64]
        exact_options%n_estimators = 2
        exact_options%max_depth = 1
        exact_options%learning_rate = 0.4_real64
        exact_options%l2 = 0.7_real64
        exact_options%min_child_weight = 0.0_real64
        hist_options = exact_options
        hist_options%tree_method = "hist"
        hist_options%max_bin = 2
        call exact_model%fit_poisson(x, target, status, exact_options)
        call exact_model%predict(x, exact_prediction, status)
        call hist_model%fit_poisson(x, target, status, hist_options)
        call hist_model%predict(x, hist_prediction, status)
        call check(status_ok(status), "histogram Poisson status", failures)
        call check(trim(hist_model%tree_method()) == "hist", &
            "histogram Poisson method accessor", failures)
        call check(maxval(abs(exact_prediction - hist_prediction)) < 2.0e-12_real64, &
            "exact/hist Poisson parity on two-bin fixture", failures)

        x_dot = 0.0_real64
        call exact_model%predict_jvp(x, x_dot, exact_prediction, prediction_dot, status)
        call check(status_ok(status) .and. maxval(abs(prediction_dot)) < 2.0e-14_real64, &
            "Poisson input JVP piecewise-constant product", failures)
    end subroutine test_histogram_and_products

    subroutine test_domain_and_device_contract(failures)
        integer, intent(inout) :: failures
        type(xgboost_t) :: model
        type(xgboost_options_t) :: options
        type(fortnum_status_t) :: status
        type(fortml_device_t) :: cpu, cuda
        real(real64) :: x(2, 1), bad_target(2), target(2), prediction(2)

        x(:, 1) = [0.0_real64, 1.0_real64]
        bad_target = [1.0_real64, -1.0_real64]
        target = [0.0_real64, 0.0_real64]
        options%n_estimators = 1
        options%max_depth = 1
        options%min_child_weight = 0.0_real64
        call model%fit_poisson(x, bad_target, status, options)
        call check(status%code == FORTNUM_DOMAIN_ERROR, &
            "negative Poisson target refusal", failures)
        call model%fit_poisson(x, target, status, options)
        call check(status_ok(status), "zero-count Poisson fit", failures)
        call model%predict(x, prediction, status)
        call check(status_ok(status) .and. all(prediction > 0.0_real64) .and. &
            all(prediction < 1.0e-9_real64), "finite zero-count prediction", failures)

        call cpu%select(FORTML_DEVICE_CPU, status)
        cuda%kind = FORTML_DEVICE_CUDA
        cuda%selected = .true.
        cuda%available = .true.
        call check(model%device_supported(FORTML_DEVICE_CPU), &
            "Poisson CPU capability", failures)
        call check(.not. model%device_supported(FORTML_DEVICE_CUDA), &
            "Poisson CUDA capability refusal", failures)
        call model%predict_device(cuda, x, prediction, status)
        call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
            "Poisson CUDA prediction refusal", failures)
        call model%predict_device(cpu, x, prediction, status)
        call check(status_ok(status), "Poisson CPU device dispatch", failures)
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

end program test_xgboost_poisson
