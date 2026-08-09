program test_lightgbm_cuda_dispatch
    !! Independent device-dispatch oracle for numeric LightGBM trees.
    !!
    !! The four-row fixture has a hand-computable one-split optimum.  CPU
    !! predictions must equal that oracle.  CUDA either reproduces the same
    !! values through the resident additive-tree ABI or returns the typed
    !! unavailable status without changing the caller buffer.
    use, intrinsic :: iso_fortran_env, only: real64, error_unit
    use fortml_lightgbm, only: lightgbm_t, lightgbm_options_t
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    use fortnum_status, only: fortnum_status_t, FORTNUM_OK, FORTNUM_NOT_IMPLEMENTED
    implicit none

    type(lightgbm_t) :: model
    type(lightgbm_options_t) :: options
    type(fortml_device_t) :: cpu, cuda
    type(fortnum_status_t) :: status
    real(real64) :: x(4, 1), y(4), expected(4), host(4), device(4)
    integer :: failures

    failures = 0
    x(:, 1) = [0.0_real64, 1.0_real64, 2.0_real64, 3.0_real64]
    y = [0.0_real64, 0.0_real64, 2.0_real64, 2.0_real64]
    expected = y
    options%n_estimators = 1
    options%num_leaves = 2
    options%min_data_in_leaf = 1
    options%max_bin = 16
    options%learning_rate = 1.0_real64
    options%l2 = 0.0_real64
    call model%fit_regression(x, y, status, options)
    call check(status%code == FORTNUM_OK, "one-split fit", failures)

    call model%predict(x, host, status)
    call check(status%code == FORTNUM_OK, "host prediction", failures)
    call check(maxval(abs(host - expected)) < 1.0e-12_real64, &
        "independent one-split Newton oracle", failures)

    cpu%kind = FORTML_DEVICE_CPU
    cpu%selected = .true.
    cpu%available = .true.
    call model%predict_device(cpu, x, device, status)
    call check(status%code == FORTNUM_OK .and. &
        maxval(abs(device - expected)) < 1.0e-12_real64, &
        "CPU dispatch preserves tree oracle", failures)

    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    device = -77.0_real64
    call model%predict_device(cuda, x, device, status)
    if (status%code == FORTNUM_OK) then
        call check(model%device_supported(FORTML_DEVICE_CUDA), &
            "native CUDA capability reports support", failures)
        call check(maxval(abs(device - expected)) < 2.0e-11_real64, &
            "resident CUDA LightGBM parity", failures)
    else
        call check(status%code == FORTNUM_NOT_IMPLEMENTED .and. &
            all(device == -77.0_real64) .and. &
            .not. model%device_supported(FORTML_DEVICE_CUDA), &
            "ordinary build gives typed CUDA refusal", failures)
    end if

    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, &
            " LightGBM CUDA dispatch test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS LightGBM resident numeric CUDA dispatch oracle"

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

end program test_lightgbm_cuda_dispatch
