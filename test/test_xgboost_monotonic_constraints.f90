program test_xgboost_monotonic_constraints
    !! Independent behavioral oracle for XGBoost monotonic constraints.
    !! The oracle checks the public prediction contract directly: a +1 feature
    !! must be nondecreasing and a -1 feature nonincreasing on an independent
    !! query grid, for both exact and weighted-histogram tree growth.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite, ieee_value, &
        ieee_quiet_nan
    use fortml_xgboost, only: xgboost_t, xgboost_options_t
    use fortnum_status, only: fortnum_status_t, FORTNUM_OK, FORTNUM_DOMAIN_ERROR, &
        FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    implicit none

    integer :: failures

    failures = 0
    call test_exact_monotone(failures)
    call test_histogram_monotone(failures)
    call test_zero_constraint_identity(failures)
    call test_constraint_validation(failures)
    call test_cuda_refusal(failures)
    call test_large_finite_midpoints(failures)
    call test_missing_monotone_policies(failures)
    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, &
            " XGBoost monotonic-constraint test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS XGBoost monotonic-constraint independent oracle"

contains

    subroutine make_fixture(x, y)
        real(dp), intent(out) :: x(:, :), y(:)
        integer :: i

        do i = 1, size(x, 1)
            x(i, 1) = real(i - 1, dp)
            x(i, 2) = mod(real(i - 1, dp), 3.0_dp)
            ! Deliberate local dips make unconstrained boosting non-monotone.
            y(i) = real(i - 1, dp) + 2.0_dp*merge(1.0_dp, -1.0_dp, mod(i, 3) == 0)
        end do
    end subroutine make_fixture

    subroutine query_grid(x)
        real(dp), intent(out) :: x(:, :)
        integer :: i

        do i = 1, size(x, 1)
            x(i, 1) = 7.0_dp*real(i - 1, dp)/real(size(x, 1) - 1, dp)
            x(i, 2) = 1.25_dp
        end do
    end subroutine query_grid

    subroutine test_exact_monotone(failures)
        integer, intent(inout) :: failures
        type(xgboost_t) :: model
        type(xgboost_options_t) :: options
        type(fortnum_status_t) :: status
        real(dp) :: x(12, 2), y(12), query(57, 2), prediction(57)
        integer :: i, j

        call make_fixture(x, y)
        call query_grid(query)
        options%n_estimators = 5
        options%max_depth = 3
        options%learning_rate = 0.5_dp
        options%l2 = 0.25_dp
        options%min_child_weight = 0.0_dp
        options%monotone_constraints = [1, 0]
        call model%fit_regression(x, y, status, options)
        call model%predict(query, prediction, status)
        call check(status%code == FORTNUM_OK, &
            "exact monotone fit/predict status", failures)
        call check(model%monotone_constraint(1) == 1 .and. &
            model%monotone_constraint(2) == 0, &
            "exact monotone metadata", failures)
        do j = 0, 2
            query(:, 2) = -0.75_dp + 0.75_dp*real(j, dp)
            call model%predict(query, prediction, status)
            do i = 1, size(prediction) - 1
                call check(prediction(i + 1) >= prediction(i) - 2.0e-13_dp, &
                    "exact +1 prediction monotonicity", failures)
            end do
        end do
    end subroutine test_exact_monotone

    subroutine test_histogram_monotone(failures)
        integer, intent(inout) :: failures
        type(xgboost_t) :: model
        type(xgboost_options_t) :: options
        type(fortnum_status_t) :: status
        real(dp) :: x(12, 2), y(12), query(57, 2), prediction(57)
        real(dp) :: weights(12)
        integer :: i, j

        call make_fixture(x, y)
        call query_grid(query)
        weights = 1.0_dp
        weights(10:12) = 2.0_dp
        options%n_estimators = 5
        options%max_depth = 3
        options%learning_rate = 0.5_dp
        options%l2 = 0.25_dp
        options%min_child_weight = 0.0_dp
        options%tree_method = "hist"
        options%max_bin = 4
        options%monotone_constraints = [-1, 0]
        call model%fit_regression(x, y, status, options, weights)
        call model%predict(query, prediction, status)
        call check(status%code == FORTNUM_OK, &
            "histogram monotone fit/predict status", failures)
        call check(trim(model%tree_method()) == "hist" .and. &
            model%monotone_constraint(1) == -1, &
            "histogram monotone metadata", failures)
        do j = 0, 2
            query(:, 2) = -0.75_dp + 0.75_dp*real(j, dp)
            call model%predict(query, prediction, status)
            do i = 1, size(prediction) - 1
                call check(prediction(i + 1) <= prediction(i) + 2.0e-13_dp, &
                    "histogram -1 prediction monotonicity", failures)
            end do
        end do
    end subroutine test_histogram_monotone

    subroutine test_zero_constraint_identity(failures)
        integer, intent(inout) :: failures
        type(xgboost_t) :: unconstrained, constrained
        type(xgboost_options_t) :: options, zero_options
        type(fortnum_status_t) :: status
        real(dp) :: x(12, 2), y(12), prediction_a(12), prediction_b(12)

        call make_fixture(x, y)
        options%n_estimators = 3
        options%max_depth = 2
        options%learning_rate = 0.5_dp
        options%l2 = 0.25_dp
        options%min_child_weight = 0.0_dp
        zero_options = options
        zero_options%monotone_constraints = [0, 0]
        call unconstrained%fit_regression(x, y, status, options)
        call constrained%fit_regression(x, y, status, zero_options)
        call unconstrained%predict(x, prediction_a, status)
        call constrained%predict(x, prediction_b, status)
        call check(status%code == FORTNUM_OK .and. &
            maxval(abs(prediction_a - prediction_b)) < 2.0e-13_dp, &
            "zero constraints preserve unconstrained model", failures)
    end subroutine test_zero_constraint_identity

    subroutine test_constraint_validation(failures)
        integer, intent(inout) :: failures
        type(xgboost_t) :: model
        type(xgboost_options_t) :: options
        type(fortnum_status_t) :: status
        real(dp) :: x(4, 2), y(4)

        x = reshape([0.0_dp, 1.0_dp, 2.0_dp, 3.0_dp, &
            0.0_dp, 1.0_dp, 2.0_dp, 3.0_dp], shape(x))
        y = [0.0_dp, 1.0_dp, 2.0_dp, 3.0_dp]
        options%n_estimators = 1
        options%max_depth = 1
        options%min_child_weight = 0.0_dp
        options%monotone_constraints = [2, 0]
        call model%fit_regression(x, y, status, options)
        call check(status%code == FORTNUM_DOMAIN_ERROR, &
            "invalid monotone constraint value refusal", failures)
        options%monotone_constraints = [1]
        call model%fit_regression(x, y, status, options)
        call check(status%code == FORTNUM_DOMAIN_ERROR, &
            "invalid monotone constraint shape refusal", failures)
    end subroutine test_constraint_validation

    subroutine test_cuda_refusal(failures)
        integer, intent(inout) :: failures
        type(xgboost_t) :: model
        type(xgboost_options_t) :: options
        type(fortml_device_t) :: device
        type(fortnum_status_t) :: status
        real(dp) :: x(4, 1), y(4), prediction(4)

        x(:, 1) = [0.0_dp, 1.0_dp, 2.0_dp, 3.0_dp]
        y = [0.0_dp, 1.0_dp, 2.0_dp, 3.0_dp]
        options%n_estimators = 1
        options%max_depth = 1
        options%min_child_weight = 0.0_dp
        call model%fit_regression(x, y, status, options)
        device%kind = FORTML_DEVICE_CUDA
        device%selected = .true.
        device%available = .true.
        call model%predict_device(device, x, prediction, status)
        call check(status%code == FORTNUM_NOT_IMPLEMENTED .and. &
            .not. model%device_supported(FORTML_DEVICE_CUDA), &
            "CUDA resident tree prediction refusal", failures)
    end subroutine test_cuda_refusal

    subroutine test_large_finite_midpoints(failures)
        integer, intent(inout) :: failures
        type(xgboost_t) :: model
        type(xgboost_options_t) :: options
        type(fortnum_status_t) :: status
        real(dp) :: x(4, 1), y(4), prediction(4)

        ! Adjacent positive values have a finite midpoint but their direct
        ! sum overflows IEEE double precision.  The estimator must retain a
        ! valid split and finite output rather than routing every row left.
        x(:, 1) = [1.50e308_dp, 1.60e308_dp, 1.70e308_dp, 1.75e308_dp]
        y = [0.0_dp, 1.0_dp, 2.0_dp, 3.0_dp]
        options%n_estimators = 1
        options%max_depth = 1
        options%learning_rate = 1.0_dp
        options%l2 = 1.0_dp
        options%min_child_weight = 0.0_dp
        options%monotone_constraints = [1]
        call model%fit_regression(x, y, status, options)
        call model%predict(x, prediction, status)
        call check(status%code == FORTNUM_OK .and. &
            all(ieee_is_finite(prediction)) .and. &
            maxval(prediction(1:3) - prediction(2:4)) <= 2.0e-13_dp, &
            "large finite midpoint stability", failures)
    end subroutine test_large_finite_midpoints

    subroutine test_missing_monotone_policies(failures)
        integer, intent(inout) :: failures
        type(xgboost_t) :: model
        type(xgboost_options_t) :: options
        type(fortnum_status_t) :: status
        real(dp) :: x(8, 2), y(8), query(33, 2), prediction(33)
        character(len=16) :: policies(3)
        real(dp) :: nan
        integer :: i, j

        nan = ieee_value(0.0_dp, ieee_quiet_nan)
        x(:, 1) = [0.0_dp, 1.0_dp, nan, 3.0_dp, 4.0_dp, 5.0_dp, 6.0_dp, 7.0_dp]
        x(:, 2) = [0.0_dp, nan, 2.0_dp, 3.0_dp, 4.0_dp, nan, 6.0_dp, 7.0_dp]
        y = [0.0_dp, 1.0_dp, 2.0_dp, 3.0_dp, 4.0_dp, 5.0_dp, 6.0_dp, 7.0_dp]
        do i = 1, size(query, 1)
            query(i, 1) = 7.0_dp*real(i - 1, dp)/real(size(query, 1) - 1, dp)
            query(i, 2) = 2.5_dp
        end do
        options%n_estimators = 2
        options%max_depth = 2
        options%learning_rate = 0.5_dp
        options%l2 = 0.5_dp
        options%min_child_weight = 0.0_dp
        options%monotone_constraints = [1, 0]
        policies = [character(len=16) :: "learn", "left", "right"]
        do j = 1, size(policies)
            options%missing_policy = policies(j)
            call model%fit_regression(x, y, status, options)
            call model%predict(query, prediction, status)
            call check(status%code == FORTNUM_OK .and. &
                all(ieee_is_finite(prediction)), &
                "missing monotone fit/predict status", failures)
            do i = 1, size(prediction) - 1
                call check(prediction(i + 1) >= prediction(i) - 2.0e-13_dp, &
                    "missing monotone +1 prediction", failures)
            end do
        end do
    end subroutine test_missing_monotone_policies

    subroutine check(condition, label, failures)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: label
        integer, intent(inout) :: failures

        if (.not. condition) then
            write (error_unit, '(a)') "FAIL ["//trim(label)//"]"
            failures = failures + 1
        end if
    end subroutine check

end program test_xgboost_monotonic_constraints
