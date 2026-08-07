program test_device_contract
    use, intrinsic :: iso_fortran_env, only: error_unit, int64
    use fortml_device, only: fortml_device_t, &
        fortml_device_capability_t, fortml_device_available, &
        fortml_query_device, FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA, &
        FORTML_DEVICE_INVALID, FORTML_BACKEND_HOST, FORTML_BACKEND_CUDA
    use fortnum_status, only: fortnum_status_t, status_ok, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED
    implicit none

    type(fortml_device_t) :: device
    type(fortml_device_capability_t) :: capability
    type(fortnum_status_t) :: status
    logical :: available
    integer :: failures

    failures = 0
    call require(device%kind == FORTML_DEVICE_INVALID, &
        "device starts unselected", failures)
    call fortml_device_available(FORTML_DEVICE_CPU, available, status)
    call require(status_ok(status) .and. available, &
        "CPU capability is always available", failures)
    call fortml_query_device(FORTML_DEVICE_CPU, capability, status)
    call require(status_ok(status) .and. capability%host_accessible .and. &
        capability%supports_residency .and. .not. capability%supports_async, &
        "CPU capability reports host residency without asynchronous transfers", &
        failures)

    call device%select(FORTML_DEVICE_CPU, status)
    call require(status_ok(status), "CPU selection succeeds", failures)
    call require(device%selected .and. device%available .and. &
        device%backend == FORTML_BACKEND_HOST .and. &
        trim(device%backend_name()) == "CPU" .and. &
        trim(device%device_name()) == "cpu:0", &
        "CPU selection has explicit backend identity", failures)
    call device%begin_residency(4096_int64, status, owns_data=.true.)
    call require(status_ok(status) .and. device%resident .and. &
        device%owns_residency .and. device%resident_bytes == 4096_int64, &
        "CPU residency records bytes and ownership", failures)
    call device%begin_residency(1_int64, status)
    call require(status%code == FORTNUM_DOMAIN_ERROR, &
        "nested residency is refused", failures)
    call device%record_host_to_device(16_int64, status)
    call require(status%code == FORTNUM_NOT_IMPLEMENTED .and. &
        device%host_to_device_bytes == 0_int64, &
        "CPU host-to-device transfer is an explicit refusal", failures)
    call device%end_residency(status)
    call require(status_ok(status) .and. .not. device%resident .and. &
        device%resident_bytes == 0_int64, &
        "ending residency clears the registered extent", failures)

    call device%select(FORTML_DEVICE_CPU, status, stream_id=1)
    call require(status%code == FORTNUM_NOT_IMPLEMENTED .and. &
        device%selected .and. device%kind == FORTML_DEVICE_CPU, &
        "unsupported CPU stream selection preserves the active context", &
        failures)
    call device%select(99, status)
    call require(status%code == FORTNUM_DOMAIN_ERROR .and. &
        device%kind == FORTML_DEVICE_CPU, &
        "invalid device selection preserves the active context", failures)
    call device%clear(status)
    call require(status_ok(status) .and. .not. device%selected, &
        "clear releases an inactive context", failures)

    call fortml_device_available(FORTML_DEVICE_CUDA, available, status)
    call require(status_ok(status), "CUDA capability query has a status", failures)
    call fortml_query_device(FORTML_DEVICE_CUDA, capability, status)
    call require(status_ok(status) .and. capability%available .eqv. available, &
        "CUDA capability query agrees with availability", failures)
    call device%select(FORTML_DEVICE_CUDA, status)
    if (available) then
        call require(status_ok(status) .and. device%backend == FORTML_BACKEND_CUDA, &
            "available CUDA backend can be selected", failures)
        call require(device%capability%supports_cuda_kernels, &
            "selected CUDA backend advertises CUDA kernels", failures)
        call device%begin_residency(8192_int64, status)
        call require(status_ok(status), "CUDA residency can be registered", failures)
        call device%record_host_to_device(8192_int64, status)
        call require(status_ok(status) .and. device%host_to_device_transfers == 1 &
            .and. device%host_to_device_bytes == 8192_int64, &
            "CUDA host-to-device metadata is accumulated", failures)
        call device%record_device_to_host(128_int64, status)
        call require(status_ok(status) .and. device%device_to_host_transfers == 1 &
            .and. device%device_to_host_bytes == 128_int64, &
            "CUDA device-to-host metadata is accumulated", failures)
        call device%end_residency(status)
        call require(status_ok(status), "CUDA residency can be ended", failures)
    else
        call require(status%code == FORTNUM_NOT_IMPLEMENTED .and. &
            device%kind == FORTML_DEVICE_INVALID .and. .not. device%selected, &
            "unavailable CUDA selection is a recoverable refusal", failures)
    end if

    if (failures > 0) then
        write (error_unit, '(a,i0)') "FAIL: device contract: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS: explicit device contract behavioral tests"

contains

    subroutine require(condition, message, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: message
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "  ", trim(message)
        end if
    end subroutine require

end program test_device_contract
