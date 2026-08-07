program test_regression_metrics
    use fortml_regression_metrics, only: regression_mean_squared_error, &
        regression_root_mean_squared_error, regression_mean_absolute_error, &
        regression_median_absolute_error, regression_max_error, &
        regression_r2_score, regression_explained_variance, &
        regression_mean_squared_log_error, &
        regression_mean_absolute_percentage_error, regression_mean_pinball_loss
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer :: failures

    failures = 0
    call test_metric_oracles(failures)
    call test_weighted_and_multoutput(failures)
    call test_refusals(failures)
    if (failures > 0) then
        write (*, '(a,i0)') "FAIL regression metric cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS regression metric independent behavioral oracles"

contains

    subroutine check(condition, label, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: label
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (*, '(a)') "  FAIL "//label
        end if
    end subroutine check

    subroutine test_metric_oracles(failures)
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status
        real(dp), parameter :: target(3, 1) = reshape([1.0_dp, 2.0_dp, 4.0_dp], [3, 1])
        real(dp), parameter :: prediction(3, 1) = reshape([2.0_dp, 1.0_dp, 2.0_dp], [3, 1])
        real(dp) :: value

        call regression_mean_squared_error(target, prediction, value, status)
        call check(status_ok(status), "MSE status", failures)
        call check(abs(value - 2.0_dp) < 1.0e-14_dp, "MSE oracle", failures)
        call regression_root_mean_squared_error(target, prediction, value, status)
        call check(status_ok(status), "RMSE status", failures)
        call check(abs(value - sqrt(2.0_dp)) < 1.0e-14_dp, "RMSE oracle", failures)
        call regression_mean_absolute_error(target, prediction, value, status)
        call check(status_ok(status), "MAE status", failures)
        call check(abs(value - 4.0_dp/3.0_dp) < 1.0e-14_dp, "MAE oracle", failures)
        call regression_median_absolute_error(target, prediction, value, status)
        call check(status_ok(status), "median absolute error status", failures)
        call check(abs(value - 1.0_dp) < 1.0e-14_dp, "median absolute error oracle", failures)
        call regression_max_error(target, prediction, value, status)
        call check(status_ok(status), "max error status", failures)
        call check(abs(value - 2.0_dp) < 1.0e-14_dp, "max error oracle", failures)
        call regression_r2_score(target, prediction, value, status)
        call check(status_ok(status), "R2 status", failures)
        call check(abs(value - (-2.0_dp/7.0_dp)) < 1.0e-14_dp, "R2 oracle", failures)
        call regression_explained_variance(target, prediction, value, status)
        call check(status_ok(status), "explained variance status", failures)
        call check(abs(value - 0.0_dp) < 1.0e-14_dp, &
            "explained variance oracle", failures)
        call regression_mean_squared_log_error(target, prediction, value, status)
        call check(status_ok(status), "MSLE status", failures)
        call check(abs(value - ((log(2.0_dp) - log(3.0_dp))**2 + &
            (log(3.0_dp) - log(2.0_dp))**2 + &
            (log(5.0_dp) - log(3.0_dp))**2)/3.0_dp) < 1.0e-14_dp, &
            "MSLE oracle", failures)
        call regression_mean_absolute_percentage_error(target, prediction, value, status)
        call check(status_ok(status), "MAPE status", failures)
        call check(abs(value - (1.0_dp + 0.5_dp + 0.5_dp)/3.0_dp) < &
            1.0e-14_dp, "MAPE oracle", failures)
        call regression_mean_pinball_loss(target, prediction, 0.25_dp, value, status)
        call check(status_ok(status), "pinball status", failures)
        call check(abs(value - (0.75_dp + 0.25_dp + 0.5_dp)/3.0_dp) < &
            1.0e-14_dp, "pinball oracle", failures)
    end subroutine test_metric_oracles

    subroutine test_weighted_and_multoutput(failures)
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status
        real(dp), parameter :: target(2, 2) = reshape([1.0_dp, 3.0_dp, 2.0_dp, 4.0_dp], [2, 2])
        real(dp), parameter :: prediction(2, 2) = reshape([2.0_dp, 1.0_dp, 4.0_dp, 2.0_dp], [2, 2])
        real(dp), parameter :: weights(2) = [1.0_dp, 3.0_dp]
        real(dp) :: value

        call regression_mean_squared_error(target, prediction, value, status, weights)
        call check(status_ok(status), "weighted MSE status", failures)
        call check(abs(value - (1.0_dp + 4.0_dp + 12.0_dp + 12.0_dp)/8.0_dp) < &
            1.0e-14_dp, "weighted MSE oracle", failures)
        call regression_median_absolute_error(target, prediction, value, status, weights)
        call check(status_ok(status), "weighted median status", failures)
        call check(abs(value - 2.0_dp) < 1.0e-14_dp, "weighted median oracle", failures)
    end subroutine test_weighted_and_multoutput

    subroutine test_refusals(failures)
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status
        real(dp) :: value

        call regression_mean_squared_error(reshape([1.0_dp, 2.0_dp], [2, 1]), &
            reshape([1.0_dp], [1, 1]), value, status)
        call check(.not. status_ok(status), "shape refusal", failures)
        call regression_mean_squared_error(reshape([1.0_dp, 2.0_dp], [2, 1]), &
            reshape([1.0_dp, 2.0_dp], [2, 1]), value, status, [1.0_dp, -1.0_dp])
        call check(.not. status_ok(status), "negative weight refusal", failures)
        call regression_r2_score(reshape([1.0_dp, 1.0_dp], [2, 1]), &
            reshape([1.0_dp, 2.0_dp], [2, 1]), value, status)
        call check(.not. status_ok(status), "constant-target R2 refusal", failures)
        call regression_mean_squared_log_error(reshape([1.0_dp, -1.0_dp], [2, 1]), &
            reshape([1.0_dp, 2.0_dp], [2, 1]), value, status)
        call check(.not. status_ok(status), "negative MSLE refusal", failures)
        call regression_mean_absolute_percentage_error(reshape([1.0_dp, 0.0_dp], [2, 1]), &
            reshape([1.0_dp, 2.0_dp], [2, 1]), value, status)
        call check(.not. status_ok(status), "zero-target MAPE refusal", failures)
        call regression_mean_pinball_loss(reshape([1.0_dp], [1, 1]), &
            reshape([1.0_dp], [1, 1]), 1.5_dp, value, status)
        call check(.not. status_ok(status), "invalid quantile refusal", failures)
    end subroutine test_refusals

end program test_regression_metrics
