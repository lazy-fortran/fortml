program fortml_bench_xgboost_cuda_dispatch
    !! Release workload for numeric XGBoost device dispatch.  The CPU row is
    !! a host/device-control parity oracle; CUDA is reported as a typed
    !! unavailable capability unless the resident native plan is linked.
    use, intrinsic :: iso_fortran_env, only: real64, int64
    use fortml_xgboost, only: xgboost_t, xgboost_options_t
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    use fortnum_status, only: fortnum_status_t, FORTNUM_OK, FORTNUM_NOT_IMPLEMENTED
    implicit none

    type(xgboost_t) :: model
    type(xgboost_options_t) :: options
    type(fortml_device_t) :: cpu, cuda
    type(fortnum_status_t) :: status
    real(real64) :: x(128, 3), target(128), host_prediction(128), device_prediction(128)
    real(real64) :: elapsed
    integer(int64) :: started, finished, rate
    integer :: i, repetition

    do i = 1, size(target)
        x(i, 1) = -1.0_real64 + 2.0_real64*real(i - 1, real64)/real(size(target) - 1, real64)
        x(i, 2) = sin(0.13_real64*real(i, real64))
        x(i, 3) = cos(0.07_real64*real(i, real64))
        target(i) = merge(-0.8_real64 + 0.1_real64*x(i, 2), &
            1.2_real64 + 0.2_real64*x(i, 3), x(i, 1) < 0.1_real64)
    end do
    options%n_estimators = 8
    options%max_depth = 2
    options%min_samples_leaf = 2
    options%learning_rate = 0.25_real64
    options%l2 = 1.0_real64
    call model%fit(x, target, status, options)
    if (status%code /= FORTNUM_OK) error stop "XGBoost CUDA dispatch fit failed"
    call model%predict(x, host_prediction, status)
    if (status%code /= FORTNUM_OK) error stop "XGBoost CUDA dispatch host prediction failed"

    cpu%kind = FORTML_DEVICE_CPU
    cpu%selected = .true.
    cpu%available = .true.
    call system_clock(started, rate)
    do repetition = 1, 16
        call model%predict_device(cpu, x, device_prediction, status)
        if (status%code /= FORTNUM_OK) error stop "XGBoost CPU device dispatch failed"
    end do
    call system_clock(finished)
    elapsed = real(finished - started, real64)/real(rate, real64)/16.0_real64
    if (maxval(abs(device_prediction - host_prediction)) > 2.0e-13_real64) &
        error stop "XGBoost CPU dispatch parity failed"
    write (*, '(a,es24.16,a,es24.16)') &
        "xgboost_cuda_dispatch_cpu,", elapsed, ",", &
        maxval(abs(device_prediction - host_prediction))

    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call model%predict_device(cuda, x, device_prediction, status)
    if (status%code == FORTNUM_OK) then
        if (maxval(abs(device_prediction - host_prediction)) > 2.0e-11_real64) &
            error stop "XGBoost resident CUDA parity failed"
        write (*, '(a,es24.16,a,es24.16)') &
            "xgboost_cuda_dispatch_resident,", 1.0_real64, ",", &
            maxval(abs(device_prediction - host_prediction))
    else if (status%code == FORTNUM_NOT_IMPLEMENTED) then
        write (*, '(a)') "xgboost_cuda_dispatch_resident,typed_refusal"
    else
        error stop "XGBoost CUDA dispatch returned unexpected status"
    end if
end program fortml_bench_xgboost_cuda_dispatch
