program test_xgboost_contributions
    !! Independent oracle for additive XGBoost raw-margin contributions.
    use, intrinsic :: iso_fortran_env, only: real64, error_unit
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, &
        FORTML_DEVICE_CUDA
    use fortml_xgboost, only: xgboost_t, xgboost_options_t
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    implicit none

    integer, parameter :: dp = real64
    type(xgboost_t) :: regression, logistic
    type(xgboost_options_t) :: options
    type(fortml_device_t) :: cpu, cuda
    type(fortnum_status_t) :: status
    real(dp) :: x(8, 2), y(8), labels(8), margin(8), prediction(8)
    real(dp) :: contributions(8, 4), staged(8, 3)
    real(dp) :: logistic_margin(8), logistic_contributions(8, 4)
    integer :: failures

    x = reshape([ &
        0.0_dp, 0.0_dp, 0.0_dp, 1.0_dp, 1.0_dp, 0.0_dp, 1.0_dp, 1.0_dp, &
        0.0_dp, 1.0_dp, 2.0_dp, 3.0_dp, 0.0_dp, 1.0_dp, 2.0_dp, 3.0_dp], &
        shape(x))
    y = [0.0_dp, 1.0_dp, 2.0_dp, 3.0_dp, 10.0_dp, 11.0_dp, 12.0_dp, 13.0_dp]
    labels = [0.0_dp, 0.0_dp, 0.0_dp, 0.0_dp, 1.0_dp, 1.0_dp, 1.0_dp, 1.0_dp]
    failures = 0

    options%n_estimators = 3
    options%max_depth = 1
    options%learning_rate = 0.7_dp
    options%l2 = 0.0_dp
    options%min_child_weight = 0.0_dp
    call regression%fit_regression(x, y, status, options)
    call check(status_ok(status), "regression fit", failures)
    call regression%predict_margin(x, margin, status)
    call regression%predict_contributions(x, contributions, status)
    call regression%predict_staged_margin(x, staged, status)
    call check(status_ok(status), "regression contribution APIs", failures)
    call check(maxval(abs(sum(contributions, dim=2) - margin)) < 2.0e-13_dp, &
        "contributions sum to raw margin", failures)
    call check(maxval(abs(contributions(:, 1) - regression%base_margin())) < &
        2.0e-13_dp, "base margin contribution", failures)
    call check(maxval(abs(contributions(:, 2) - &
        (staged(:, 1) - regression%base_margin()))) < 2.0e-13_dp, &
        "first tree contribution", failures)
    call check(maxval(abs(contributions(:, 3) - &
        (staged(:, 2) - staged(:, 1)))) < 2.0e-13_dp, &
        "second tree contribution", failures)
    call check(maxval(abs(contributions(:, 4) - &
        (staged(:, 3) - staged(:, 2)))) < 2.0e-13_dp, &
        "third tree contribution", failures)

    call regression%predict(x, prediction, status)
    call check(status_ok(status) .and. maxval(abs(prediction - margin)) < &
        2.0e-13_dp, "regression margin matches prediction", failures)

    call logistic%fit_binary(x, labels, status, options)
    call logistic%predict_margin(x, logistic_margin, status)
    call logistic%predict_contributions(x, logistic_contributions, status)
    call check(status_ok(status) .and. maxval(abs(sum(logistic_contributions, &
        dim=2) - logistic_margin)) < 2.0e-13_dp, &
        "logistic raw-link contribution sum", failures)

    call cpu%select(FORTML_DEVICE_CPU, status)
    call regression%predict_contributions_device(cpu, x, contributions, status)
    call check(status_ok(status), "CPU contribution dispatch", failures)
    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call regression%predict_contributions_device(cuda, x, contributions, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "CUDA contribution refusal", failures)

    call regression%predict_contributions(x, contributions(:, 1:3), status)
    call check(.not. status_ok(status), "invalid contribution shape refusal", failures)

    if (failures > 0) then
        write (error_unit, '(a,i0)') "FAIL XGBoost contribution cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS XGBoost contribution behavioral oracle"

contains

    subroutine check(condition, label, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: label
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "FAIL: " // trim(label)
        end if
    end subroutine check

end program test_xgboost_contributions
