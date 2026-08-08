program test_radius_neighbors_multioutput_regression
    !! Independent behavioral and device-contract checks for multi-output radius regression.
    use, intrinsic :: iso_fortran_env, only: real64, error_unit
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_DOMAIN_ERROR, &
        FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortml_radius_neighbors_multioutput_regression, only: &
        radius_neighbors_multioutput_regressor_t, &
        RADIUS_MULTI_REGRESSION_WEIGHTS_UNIFORM, &
        RADIUS_MULTI_REGRESSION_WEIGHTS_DISTANCE
    implicit none

    type(radius_neighbors_multioutput_regressor_t) :: model, distance_model, &
        weighted_model, no_outlier_model
    type(fortnum_status_t) :: status
    type(fortml_device_t) :: cuda
    real(real64) :: x(4, 2), targets(4, 2), query(3, 2), predictions(3, 2), &
        tangent(3, 2), predictions_dot(3, 2), x_bar(3, 2), targets_bar(3, 2), &
        cuda_predictions(3, 2), weights(4), outlier(2)
    integer :: failures

    x(:, 1) = [0.0_real64, 1.0_real64, 2.0_real64, 3.0_real64]
    x(:, 2) = 0.0_real64
    targets(:, 1) = [0.0_real64, 10.0_real64, 20.0_real64, 30.0_real64]
    targets(:, 2) = [0.0_real64, 1.0_real64, 4.0_real64, 9.0_real64]
    query(:, 1) = [0.5_real64, 1.5_real64, 5.0_real64]
    query(:, 2) = 0.0_real64
    outlier = [-9.0_real64, -8.0_real64]
    tangent = 1.0_real64
    targets_bar = 1.0_real64
    failures = 0

    call model%fit(x, targets, status, radius=1.1_real64, &
        weights=RADIUS_MULTI_REGRESSION_WEIGHTS_UNIFORM, outlier_value=outlier)
    call check(status_ok(status) .and. model%fitted() .and. &
        model%feature_count() == 2 .and. model%sample_count() == 4 .and. &
        model%output_count() == 2 .and. model%weighting() == &
        RADIUS_MULTI_REGRESSION_WEIGHTS_UNIFORM, "fit metadata", failures)
    call model%predict(query, predictions, status)
    call check(status_ok(status) .and. maxval(abs(predictions - reshape([ &
        5.0_real64, 15.0_real64, -9.0_real64, 0.5_real64, 2.5_real64, &
        -8.0_real64], shape(predictions)))) < 2.0e-14_real64, &
        "uniform multi-output oracle", failures)

    call model%predict_jvp(query, tangent, predictions, predictions_dot, status)
    call check(status_ok(status) .and. maxval(abs(predictions_dot)) == 0.0_real64, &
        "piecewise-constant JVP oracle", failures)
    call model%predict_vjp(query, targets_bar, x_bar, status)
    call check(status_ok(status) .and. maxval(abs(x_bar)) == 0.0_real64, &
        "piecewise-constant VJP oracle", failures)

    query(1, 1) = 1.1_real64
    call model%predict_jvp(query, tangent, predictions, predictions_dot, status)
    call check(status%code == FORTNUM_DOMAIN_ERROR, "radius-boundary JVP refusal", failures)
    call model%predict_vjp(query, targets_bar, x_bar, status)
    call check(status%code == FORTNUM_DOMAIN_ERROR, "radius-boundary VJP refusal", failures)
    query(1, 1) = 0.5_real64

    call distance_model%fit(x, targets, status, radius=1.5_real64, &
        weights=RADIUS_MULTI_REGRESSION_WEIGHTS_DISTANCE)
    query(1, 1) = 1.0_real64
    call distance_model%predict(query(1:1, :), predictions(1:1, :), status)
    call check(status_ok(status) .and. maxval(abs(predictions(1, :) - &
        [10.0_real64, 1.0_real64])) < 2.0e-14_real64, &
        "exact-distance neighbor oracle", failures)
    call distance_model%predict_jvp(query(1:1, :), tangent(1:1, :), &
        predictions(1:1, :), predictions_dot(1:1, :), status)
    call check(status%code == FORTNUM_DOMAIN_ERROR, &
        "exact-distance derivative refusal", failures)
    query(1, 1) = 0.5_real64

    weights = [1.0_real64, 3.0_real64, 1.0_real64, 1.0_real64]
    call weighted_model%fit(x, targets, status, radius=1.1_real64, &
        sample_weight=weights)
    call weighted_model%predict(query(1:1, :), predictions(1:1, :), status)
    call check(status_ok(status) .and. maxval(abs(predictions(1, :) - &
        [7.5_real64, 0.75_real64])) < 2.0e-14_real64, &
        "sample-weighted multi-output oracle", failures)

    call no_outlier_model%fit(x, targets, status, radius=0.1_real64)
    call no_outlier_model%predict(reshape([5.0_real64, 0.0_real64], [1, 2]), &
        predictions(1:1, :), status)
    call check(status%code == FORTNUM_DOMAIN_ERROR, "empty-neighborhood refusal", failures)

    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    cuda_predictions = -37.0_real64
    call model%predict_device(cuda, query, cuda_predictions, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED .and. &
        all(cuda_predictions == -37.0_real64), "CUDA prediction refusal", failures)
    call check(.not. model%device_supported(FORTML_DEVICE_CUDA), &
        "CUDA capability refusal", failures)

    if (failures > 0) then
        write (error_unit, '(a,i0)') &
            "FAIL multi-output radius regression cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS multi-output radius regression independent behavioral oracles"

contains

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failures
        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "FAIL [radius multi-output] "//description
        end if
    end subroutine check

end program test_radius_neighbors_multioutput_regression
