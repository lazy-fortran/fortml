program test_knn_classifier_cuda
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA, &
        fortml_cuda_knn_available
    use fortml_knn_classifier, only: knn_classifier_t, KNN_WEIGHTS_UNIFORM
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    real(dp) :: train(4, 1), query(2, 1)
    integer :: labels(4), predicted(2), failures
    type(knn_classifier_t) :: model
    type(fortml_device_t) :: cuda
    type(fortnum_status_t) :: status

    train(:, 1) = [-2.0_dp, -1.0_dp, 1.0_dp, 2.0_dp]
    labels = [-7, -7, 11, 11]
    query(:, 1) = [-1.5_dp, 1.5_dp]
    failures = 0
    if (fortml_cuda_knn_available() == 0) then
        write (*, '(a)') "SKIP FortML CUDA kNN API oracle (native object not linked)"
        stop
    end if
    call model%fit(train, labels, status, n_neighbors=1, weights=KNN_WEIGHTS_UNIFORM)
    call check(status_ok(status), "CUDA fixture fits", failures)
    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    predicted = 0
    call model%predict_device(cuda, query, predicted, status)
    call check(status_ok(status) .and. all(predicted == [-7, 11]), &
        "resident CUDA kNN agrees with nearest-neighbor oracle", failures)
    if (failures > 0) error stop 1
    write (*, '(a)') "PASS FortML CUDA kNN API oracle"

contains

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: description
        integer, intent(inout) :: failures
        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "FAIL "//description
        end if
    end subroutine check

end program test_knn_classifier_cuda
