program test_radius_neighbors_regression
    !! Independent hand oracle for radius-neighbor regression and boundaries.
    use, intrinsic :: iso_fortran_env, only: real64
    use fortnum_status, only: status_ok, FORTNUM_DOMAIN_ERROR, &
        FORTNUM_NOT_IMPLEMENTED, fortnum_status_t
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    use fortml_radius_neighbors_regression, only: &
        radius_neighbors_regressor_t, RADIUS_REGRESSION_WEIGHTS_UNIFORM, &
        RADIUS_REGRESSION_WEIGHTS_DISTANCE
    implicit none

    type(radius_neighbors_regressor_t) :: model, distance_model, strict_model
    type(fortnum_status_t) :: status
    type(fortml_device_t) :: cpu, cuda
    real(real64) :: x(3, 1), targets_train(3), query(3, 1), predictions(3)
    real(real64) :: predictions_dot(3), x_dot(3, 1), predictions_bar(3), x_bar(3, 1)
    integer :: failures

    x(:, 1) = [0.0_real64, 1.0_real64, 3.0_real64]
    targets_train = [0.0_real64, 2.0_real64, 6.0_real64]
    query(:, 1) = [0.5_real64, 3.0_real64, 5.0_real64]
    x_dot = 0.0_real64
    predictions_bar = 1.0_real64
    failures = 0

    call model%fit(x, targets_train, status, radius=1.1_real64, &
        weights=RADIUS_REGRESSION_WEIGHTS_UNIFORM, outlier_value=-1.0_real64)
    call check(status_ok(status) .and. model%fitted(), "uniform fit", failures)
    call model%predict(query, predictions, status)
    call check(status_ok(status) .and. &
        maxval(abs(predictions - [1.0_real64, 6.0_real64, -1.0_real64])) < &
        1.0e-14_real64, "uniform and outlier prediction oracle", failures)
    call check(abs(model%radius() - 1.1_real64) < 1.0e-14_real64 .and. &
        model%feature_count() == 1 .and. model%sample_count() == 3, &
        "metadata", failures)

    call distance_model%fit(x, targets_train, status, radius=1.1_real64, &
        weights=RADIUS_REGRESSION_WEIGHTS_DISTANCE)
    call distance_model%predict(reshape([0.25_real64], [1, 1]), &
        predictions(:1), status)
    call check(status_ok(status) .and. abs(predictions(1) - &
        (2.0_real64/sqrt(0.5625_real64))/ &
        (1.0_real64/sqrt(0.0625_real64) + 1.0_real64/sqrt(0.5625_real64))) < &
        1.0e-13_real64, "inverse-distance prediction oracle", failures)

    call strict_model%fit(x, targets_train, status, radius=0.1_real64)
    call strict_model%predict(reshape([5.0_real64], [1, 1]), predictions(:1), status)
    call check(status%code == FORTNUM_DOMAIN_ERROR, &
        "no-neighbor refusal without outlier value", failures)

    call distance_model%predict_jvp(query, x_dot, predictions, predictions_dot, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "selection JVP refusal", failures)
    call distance_model%predict_vjp(query, predictions_bar, x_bar, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "selection VJP refusal", failures)

    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call distance_model%predict_device(cuda, query, predictions, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED .and. &
        .not. distance_model%device_supported(FORTML_DEVICE_CUDA), &
        "CUDA typed refusal", failures)
    call cpu%select(FORTML_DEVICE_CPU, status)
    call distance_model%predict_device(cpu, query(:1, :), predictions(:1), status)
    call check(status_ok(status) .and. abs(predictions(1) - 1.0_real64) < &
        1.0e-14_real64, "CPU device dispatch", failures)

    if (failures /= 0) error stop failures
    print '(a)', "test_radius_neighbors_regression: PASS"

contains

    subroutine check(condition, description, failure_count)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failure_count
        if (.not. condition) then
            failure_count = failure_count + 1
            write (*, '(a)') "FAIL: "//trim(description)
        end if
    end subroutine check

end program test_radius_neighbors_regression
