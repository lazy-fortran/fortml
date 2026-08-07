program fortml_bench_xgboost_early_stopping
    !! Release protocol for deterministic XGBoost validation monitoring.
    !!
    !! Each row is machine-readable: objective, model-retention policy,
    !! one-based best validation round, retained estimator count, early-stop
    !! flag, and best weighted objective.  The Python lane independently
    !! recomputes the squared/logistic/squared-log losses from staged outputs.
    use, intrinsic :: iso_fortran_env, only: dp => real64
    use fortml_xgboost, only: xgboost_t, xgboost_options_t
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_DOMAIN_ERROR
    implicit none

    integer, parameter :: n_samples = 8
    real(dp) :: x(n_samples, 1), squared_target(n_samples)
    real(dp) :: logistic_target(n_samples), squared_log_target(n_samples)
    real(dp) :: validation_squared(n_samples), validation_logistic(n_samples)
    real(dp) :: validation_squared_log(n_samples)
    type(xgboost_options_t) :: options
    type(fortnum_status_t) :: status
    type(xgboost_t) :: model
    integer :: i

    do i = 1, n_samples
        x(i, 1) = real(i - 1, dp)
    end do
    squared_target = [0.0_dp, 0.0_dp, 0.0_dp, 0.0_dp, &
        10.0_dp, 10.0_dp, 10.0_dp, 10.0_dp]
    validation_squared = 10.0_dp - squared_target
    logistic_target = [0.0_dp, 0.0_dp, 0.0_dp, 0.0_dp, &
        1.0_dp, 1.0_dp, 1.0_dp, 1.0_dp]
    validation_logistic = 1.0_dp - logistic_target
    squared_log_target = squared_target
    validation_squared_log = validation_squared

    options = xgboost_options_t()
    options%n_estimators = 8
    options%max_depth = 1
    options%learning_rate = 1.0_dp
    options%l2 = 1.0_dp
    options%min_child_weight = 0.0_dp
    options%early_stopping_rounds = 2

    options%restore_best = .true.
    call model%fit_regression(x, squared_target, status, options, &
        validation_x=x, validation_y=validation_squared)
    if (.not. status_ok(status)) error stop "squared early-stopping fit failed"
    call write_row("xgb_early_squared", "restore_best", model)

    options%restore_best = .false.
    call model%fit_regression(x, squared_target, status, options, &
        validation_x=x, validation_y=validation_squared)
    if (.not. status_ok(status)) error stop "squared retain-all fit failed"
    call write_row("xgb_early_squared", "retain_all", model)

    options%restore_best = .true.
    call model%fit_binary(x, logistic_target, status, options, &
        validation_x=x, validation_y=validation_logistic)
    if (.not. status_ok(status)) error stop "logistic early-stopping fit failed"
    call write_row("xgb_early_logistic", "restore_best", model)

    options%restore_best = .false.
    call model%fit_binary(x, logistic_target, status, options, &
        validation_x=x, validation_y=validation_logistic)
    if (.not. status_ok(status)) error stop "logistic retain-all fit failed"
    call write_row("xgb_early_logistic", "retain_all", model)

    options%restore_best = .true.
    call model%fit_squared_log(x, squared_log_target, status, options, &
        validation_x=x, validation_y=validation_squared_log)
    if (.not. status_ok(status)) error stop "squared-log early-stopping fit failed"
    call write_row("xgb_early_squared_log", "restore_best", model)

    options%restore_best = .false.
    call model%fit_squared_log(x, squared_log_target, status, options, &
        validation_x=x, validation_y=validation_squared_log)
    if (.not. status_ok(status)) error stop "squared-log retain-all fit failed"
    call write_row("xgb_early_squared_log", "retain_all", model)

    ! A validation feature matrix without its target vector is a typed
    ! domain refusal, never an implicit training-set fallback.
    options%early_stopping_rounds = 1
    call model%fit_binary(x, logistic_target, status, options, validation_x=x)
    if (status%code /= FORTNUM_DOMAIN_ERROR) then
        error stop "malformed validation contract changed unexpectedly"
    end if
    write (*, '(a,i0)') "xgb_early_invalid_validation,", status%code

contains

    subroutine write_row(name, policy, model)
        character(len=*), intent(in) :: name, policy
        type(xgboost_t), intent(in) :: model

        write (*, '(a,a,a,a,i0,a,i0,a,i0,a,es24.16)') trim(name), ",", &
            trim(policy), ",", model%best_iteration(), ",", &
            model%estimator_count(), ",", logical_integer(model%early_stopped()), &
            ",", model%best_validation_loss()
    end subroutine write_row

    integer function logical_integer(value) result(encoded)
        logical, intent(in) :: value

        if (value) then
            encoded = 1
        else
            encoded = 0
        end if
    end function logical_integer

end program fortml_bench_xgboost_early_stopping
