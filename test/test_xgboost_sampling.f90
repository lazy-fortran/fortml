program test_xgboost_sampling
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit, int64
    use fortml_xgboost, only: xgboost_t, xgboost_options_t
    use fortnum_status, only: fortnum_status_t, FORTNUM_OK, FORTNUM_DOMAIN_ERROR
    implicit none

    integer :: failures

    failures = 0
    call test_full_sampling_oracle(failures)
    call test_seeded_sampling(failures)
    call test_invalid_fractions(failures)
    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, " xgboost sampling test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS xgboost sampling independent behavioral oracles"

contains

    subroutine make_data(x, y)
        real(dp), intent(out) :: x(:, :), y(:)
        integer :: i

        do i = 1, size(x, 1)
            x(i, 1) = real(i - 1, dp)
            x(i, 2) = real(mod(3*i + 1, 11), dp)
            x(i, 3) = real(mod(5*i + 2, 13), dp)
            x(i, 4) = real(mod(7*i + 3, 17), dp)
            y(i) = 0.7_dp*x(i, 1) - 0.25_dp*x(i, 2) + &
                0.13_dp*x(i, 3) + 0.05_dp*x(i, 4)
        end do
    end subroutine make_data

    subroutine test_full_sampling_oracle(failures)
        integer, intent(inout) :: failures
        type(xgboost_t) :: default_model, explicit_model
        type(xgboost_options_t) :: defaults, explicit_options
        type(fortnum_status_t) :: status
        real(dp) :: x(16, 4), y(16), default_prediction(16), explicit_prediction(16)
        real(dp) :: default_importance(4), explicit_importance(4)
        integer :: i

        call make_data(x, y)
        defaults%n_estimators = 4
        defaults%max_depth = 2
        defaults%learning_rate = 0.6_dp
        defaults%min_child_weight = 0.0_dp
        explicit_options = defaults
        explicit_options%subsample = 1.0_dp
        explicit_options%colsample_bytree = 1.0_dp
        explicit_options%seed = 773_int64
        call default_model%fit_regression(x, y, status, defaults)
        call explicit_model%fit_regression(x, y, status, explicit_options)
        call default_model%predict(x, default_prediction, status)
        call explicit_model%predict(x, explicit_prediction, status)
        call default_model%feature_importance(default_importance, status, kind="weight")
        call explicit_model%feature_importance(explicit_importance, status, kind="weight")
        if (status%code /= FORTNUM_OK .or. maxval(abs(default_prediction - explicit_prediction)) > &
            2.0e-14_dp .or. maxval(abs(default_importance - explicit_importance)) > 2.0e-14_dp) then
            write (error_unit, '(a,es12.4)') &
                "FAIL [xgb sampling] full fractions changed the baseline ", &
                maxval(abs(default_prediction - explicit_prediction))
            failures = failures + 1
        end if
        do i = 1, defaults%n_estimators
            if (default_model%tree_node_count(i) /= explicit_model%tree_node_count(i)) then
                write (error_unit, '(a,i0)') "FAIL [xgb sampling] baseline tree shape ", i
                failures = failures + 1
            end if
        end do
    end subroutine test_full_sampling_oracle

    subroutine test_seeded_sampling(failures)
        integer, intent(inout) :: failures
        type(xgboost_t) :: first_model, repeat_model, other_model
        type(xgboost_options_t) :: options, other_options
        type(fortnum_status_t) :: status
        real(dp) :: x(20, 4), y(20), first_prediction(20), repeat_prediction(20), &
            other_prediction(20), first_importance(4), repeat_importance(4), &
            other_importance(4)

        call make_data(x, y)
        options%n_estimators = 6
        options%max_depth = 2
        options%learning_rate = 0.6_dp
        options%min_child_weight = 0.0_dp
        options%subsample = 0.55_dp
        options%colsample_bytree = 0.5_dp
        options%seed = 12345_int64
        other_options = options
        other_options%seed = 12346_int64
        call first_model%fit_regression(x, y, status, options)
        call repeat_model%fit_regression(x, y, status, options)
        call other_model%fit_regression(x, y, status, other_options)
        call first_model%predict(x, first_prediction, status)
        call repeat_model%predict(x, repeat_prediction, status)
        call other_model%predict(x, other_prediction, status)
        call first_model%feature_importance(first_importance, status, kind="weight")
        call repeat_model%feature_importance(repeat_importance, status, kind="weight")
        call other_model%feature_importance(other_importance, status, kind="weight")
        if (status%code /= FORTNUM_OK .or. maxval(abs(first_prediction - repeat_prediction)) > &
            2.0e-14_dp .or. maxval(abs(first_importance - repeat_importance)) > 2.0e-14_dp) then
            write (error_unit, '(a)') "FAIL [xgb sampling] equal seeds are not deterministic"
            failures = failures + 1
        end if
        if (maxval(abs(first_prediction - other_prediction)) < 1.0e-12_dp .and. &
            maxval(abs(first_importance - other_importance)) < 1.0e-12_dp) then
            write (error_unit, '(a)') "FAIL [xgb sampling] changed seed did not change structure"
            failures = failures + 1
        end if
    end subroutine test_seeded_sampling

    subroutine test_invalid_fractions(failures)
        integer, intent(inout) :: failures
        type(xgboost_t) :: model
        type(xgboost_options_t) :: options
        type(fortnum_status_t) :: status
        real(dp) :: x(8, 2), y(8)
        integer :: i

        do i = 1, 8
            x(i, :) = [real(i, dp), real(9 - i, dp)]
            y(i) = real(i, dp)
        end do
        options%n_estimators = 1
        options%max_depth = 1
        options%min_child_weight = 0.0_dp
        options%subsample = 0.0_dp
        call model%fit_regression(x, y, status, options)
        if (status%code /= FORTNUM_DOMAIN_ERROR) failures = failures + 1
        options%subsample = 1.01_dp
        call model%fit_regression(x, y, status, options)
        if (status%code /= FORTNUM_DOMAIN_ERROR) failures = failures + 1
        options%subsample = 1.0_dp
        options%colsample_bytree = -0.1_dp
        call model%fit_regression(x, y, status, options)
        if (status%code /= FORTNUM_DOMAIN_ERROR) failures = failures + 1
        options%colsample_bytree = 1.1_dp
        call model%fit_regression(x, y, status, options)
        if (status%code /= FORTNUM_DOMAIN_ERROR) failures = failures + 1
        options%colsample_bytree = 1.0_dp
        options%seed = 0_int64
        call model%fit_regression(x, y, status, options)
        if (status%code /= FORTNUM_DOMAIN_ERROR) failures = failures + 1
    end subroutine test_invalid_fractions

end program test_xgboost_sampling
