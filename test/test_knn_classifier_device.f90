program test_knn_classifier_device
    !! Independent capability and structured-refusal checks for kNN devices.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, &
        FORTML_DEVICE_CUDA
    use fortml_knn_classifier, only: knn_classifier_t, KNN_WEIGHTS_UNIFORM
    use fortnum_status, only: fortnum_status_t, status_ok, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED
    implicit none

    real(dp) :: train(4, 1), query(2, 1)
    integer :: labels(4), predicted(2), failures
    type(knn_classifier_t) :: model
    type(fortml_device_t) :: cpu, cuda, unselected
    type(fortnum_status_t) :: status

    train(:, 1) = [-2.0_dp, -1.0_dp, 1.0_dp, 2.0_dp]
    labels = [-7, -7, 11, 11]
    query(:, 1) = [-1.5_dp, 1.5_dp]
    failures = 0

    call model%fit(train, labels, status, n_neighbors=1, weights=KNN_WEIGHTS_UNIFORM)
    call check(status_ok(status), "kNN host fixture fits", failures)
    call check(model%device_supported(FORTML_DEVICE_CPU), &
        "kNN advertises CPU capability", failures)
    call check(.not. model%device_supported(FORTML_DEVICE_CUDA), &
        "kNN does not overclaim CUDA capability", failures)

    call cpu%select(FORTML_DEVICE_CPU, status)
    call check(status_ok(status), "CPU device selection succeeds", failures)
    predicted = 0
    call model%predict_device(cpu, query, predicted, status)
    call check(status_ok(status) .and. all(predicted == [-7, 11]), &
        "CPU device path matches independent nearest-neighbor oracle", failures)

    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    predicted = [-999, -999]
    call model%predict_device(cuda, query, predicted, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "CUDA kNN path returns explicit resident-kernel refusal", failures)
    call check(all(predicted == [-999, -999]), &
        "CUDA refusal does not fabricate host predictions", failures)

    unselected%kind = FORTML_DEVICE_CPU
    predicted = 0
    call model%predict_device(unselected, query, predicted, status)
    call check(status%code == FORTNUM_DOMAIN_ERROR, &
        "unselected device is rejected", failures)

    if (failures > 0) then
        write (error_unit, '(a,i0)') "FAIL kNN device cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS kNN device capability/refusal oracle"

contains

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "  FAIL [knn-device] "//description
        end if
    end subroutine check

end program test_knn_classifier_device
