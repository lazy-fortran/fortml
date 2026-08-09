program test_xgboost_cuda_dispatch
    !! Dispatch contract for the resident numeric XGBoost CUDA plan.  The
    !! ordinary build must refuse CUDA through the typed status boundary;
    !! CUDA builds may return value parity with the validated host route.
    use, intrinsic :: iso_fortran_env, only: real64, error_unit
    use fortml_xgboost, only: xgboost_t, xgboost_options_t
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    use fortnum_status, only: fortnum_status_t, FORTNUM_OK, FORTNUM_NOT_IMPLEMENTED
    implicit none

    type(xgboost_t) :: model
    type(xgboost_options_t) :: options
    type(fortml_device_t) :: cpu, cuda
    type(fortnum_status_t) :: status
    real(real64) :: x(10, 2), target(10), host_prediction(10), device_prediction(10)
    integer :: i, failures

    failures = 0
    do i = 1, 10
        x(i, 1) = real(i - 1, real64)/9.0_real64
        x(i, 2) = 1.0_real64 - x(i, 1)
        target(i) = merge(-1.0_real64, 2.0_real64, i <= 5)
    end do
    options%n_estimators = 3
    options%max_depth = 2
    options%learning_rate = 0.4_real64
    options%l2 = 1.0e-3_real64
    options%min_child_weight = 0.0_real64
    call model%fit(x, target, status, options)
    call check(status%code == FORTNUM_OK, "numeric XGBoost fit", failures)
    call model%predict(x, host_prediction, status)
    call check(status%code == FORTNUM_OK .and. all(host_prediction == host_prediction), &
        "host prediction is finite", failures)

    cpu%kind = FORTML_DEVICE_CPU
    cpu%selected = .true.
    cpu%available = .true.
    call model%predict_device(cpu, x, device_prediction, status)
    call check(status%code == FORTNUM_OK .and. &
        maxval(abs(device_prediction - host_prediction)) < 2.0e-13_real64, &
        "CPU dispatch parity", failures)

    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call model%predict_device(cuda, x, device_prediction, status)
    if (status%code == FORTNUM_OK) then
        call check(model%device_supported(FORTML_DEVICE_CUDA), &
            "native CUDA capability reports supported", failures)
        call check(maxval(abs(device_prediction - host_prediction)) < 2.0e-11_real64, &
            "resident CUDA prediction parity", failures)
    else
        call check(status%code == FORTNUM_NOT_IMPLEMENTED .and. &
            .not. model%device_supported(FORTML_DEVICE_CUDA), &
            "ordinary build gives typed CUDA refusal", failures)
    end if

    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, " XGBoost CUDA dispatch test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS XGBoost resident numeric CUDA dispatch contract"

contains

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failures
        if (.not. condition) then
            write (error_unit, '(a)') "FAIL ["//trim(description)//"]"
            failures = failures + 1
        end if
    end subroutine check

end program test_xgboost_cuda_dispatch
