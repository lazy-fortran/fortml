program test_radius_neighbors_classifier
    !! Independent hand oracle for radius-neighbor votes and device boundaries.
    use, intrinsic :: iso_fortran_env, only: real64
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_DOMAIN_ERROR, &
        FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    use fortml_radius_neighbors_classifier, only: radius_neighbors_classifier_t, &
        RADIUS_WEIGHTS_UNIFORM, RADIUS_WEIGHTS_DISTANCE
    implicit none

    type(radius_neighbors_classifier_t) :: model, distance_model, strict_model
    type(fortnum_status_t) :: status
    type(fortml_device_t) :: cpu, cuda
    real(real64) :: x(3, 1), query(2, 1), probabilities(2, 2)
    real(real64) :: probabilities_dot(2, 2), x_dot(2, 1), probabilities_bar(2, 2)
    real(real64) :: x_bar(2, 1)
    integer :: labels(3), prediction(2), failures

    x(:, 1) = [0.0_real64, 1.0_real64, 3.0_real64]
    labels = [10, 20, 10]
    query(:, 1) = [0.5_real64, 5.0_real64]
    x_dot = 0.0_real64
    probabilities_bar = 1.0_real64
    failures = 0

    call model%fit(x, labels, status, radius=1.1_real64, &
        weights=RADIUS_WEIGHTS_UNIFORM, outlier_label=10)
    call check(status_ok(status) .and. model%fitted(), "uniform fit", failures)
    call model%predict_proba(query, probabilities, status)
    call check(status_ok(status) .and. maxval(abs(probabilities(1, :) - &
        [0.5_real64, 0.5_real64])) < 1.0e-14_real64 .and. &
        maxval(abs(probabilities(2, :) - [1.0_real64, 0.0_real64])) < &
        1.0e-14_real64, "uniform and outlier probability oracle", failures)
    call model%predict(query, prediction, status)
    call check(status_ok(status) .and. all(prediction == [10, 10]), &
        "uniform prediction", failures)
    call check(all(model%classes() == [10, 20]) .and. &
        abs(model%radius() - 1.1_real64) < 1.0e-14_real64, &
        "metadata", failures)

    call distance_model%fit(x, labels, status, radius=1.1_real64, &
        weights=RADIUS_WEIGHTS_DISTANCE, outlier_label=10)
    call distance_model%predict_proba(reshape([0.25_real64], [1, 1]), &
        probabilities(:1, :), status)
    call check(status_ok(status) .and. abs(probabilities(1, 1) - 0.75_real64) < &
        1.0e-13_real64 .and. abs(probabilities(1, 2) - 0.25_real64) < &
        1.0e-13_real64, "inverse-distance probability oracle", failures)

    call strict_model%fit(x, labels, status, radius=0.1_real64)
    call strict_model%predict_proba(reshape([5.0_real64], [1, 1]), &
        probabilities(:1, :), status)
    call check(status%code == FORTNUM_DOMAIN_ERROR, &
        "no-neighbor refusal without outlier label", failures)

    call distance_model%predict_proba_jvp(query, x_dot, probabilities, probabilities_dot, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, "selection JVP refusal", failures)
    call distance_model%predict_proba_vjp(query, probabilities_bar, x_bar, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, "selection VJP refusal", failures)

    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call distance_model%predict_device(cuda, query, prediction, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED .and. &
        .not. distance_model%device_supported(FORTML_DEVICE_CUDA), &
        "CUDA typed refusal", failures)
    call cpu%select(FORTML_DEVICE_CPU, status)
    call distance_model%predict_device(cpu, query, prediction, status)
    call check(status_ok(status) .and. all(prediction == [10, 10]), &
        "CPU device dispatch", failures)

    if (failures /= 0) error stop failures
    print '(a)', "test_radius_neighbors_classifier: PASS"

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

end program test_radius_neighbors_classifier
