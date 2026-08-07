program test_cuda_metric_contract
    !! Independent oracle for the weighted CUDA MSE reduction boundary.
    !! The ordinary test build links the typed unavailable stub; a CUDA
    !! harness links the native .cu implementation and exercises parity.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_DOMAIN_ERROR, &
        FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, &
        FORTML_DEVICE_CUDA
    use fortml_cuda_metrics, only: cuda_mean_squared_error, &
        fortml_cuda_mse_available
    implicit none

    type(fortml_device_t) :: cpu, cuda, unselected
    type(fortnum_status_t) :: status
    real(dp) :: target(4, 2), prediction(4, 2), weights(4)
    real(dp) :: value, expected
    integer :: failures

    failures = 0
    target = reshape([1.0_dp, -2.0_dp, 0.5_dp, 4.0_dp, &
        2.0_dp, 1.0_dp, -1.5_dp, 3.0_dp], shape(target))
    prediction = reshape([0.0_dp, -1.0_dp, 1.5_dp, 2.0_dp, &
        1.0_dp, 2.0_dp, -0.5_dp, 4.0_dp], shape(prediction))
    weights = [1.0_dp, 2.0_dp, 0.5_dp, 3.0_dp]
    expected = sum(weights * ((target(:, 1) - prediction(:, 1))**2 + &
        (target(:, 2) - prediction(:, 2))**2)) / (sum(weights)*2.0_dp)

    call cpu%select(FORTML_DEVICE_CPU, status)
    call check(status_ok(status), "CPU device selection", failures)
    value = 1234.0_dp
    call cuda_mean_squared_error(cpu, target, prediction, value, status, weights)
    call check(status%code == FORTNUM_DOMAIN_ERROR .and. value == 0.0_dp, &
        "CUDA metric rejects CPU dispatch", failures)

    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    cuda%device_index = 0
    value = -7.0_dp
    call cuda_mean_squared_error(cuda, target, prediction, value, status, weights)
    if (fortml_cuda_mse_available() /= 0) then
        call check(status_ok(status), "native CUDA weighted MSE status", failures)
        call check(abs(value - expected) < 5.0e-12_dp, &
            "native CUDA weighted MSE matches direct oracle", failures)
    else
        call check(status%code == FORTNUM_NOT_IMPLEMENTED .and. value == 0.0_dp, &
            "unavailable CUDA metric refuses without host fallback", failures)
    end if

    call cuda_mean_squared_error(unselected, target, prediction, value, status, weights)
    call check(status%code == FORTNUM_DOMAIN_ERROR, &
        "unselected device is rejected", failures)

    if (failures /= 0) then
        write (error_unit, '(a,i0)') "FAIL CUDA metric contract cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS CUDA weighted MSE CPU oracle/device contract"

contains

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "  FAIL [cuda-metric] "//description
        end if
    end subroutine check

end program test_cuda_metric_contract
