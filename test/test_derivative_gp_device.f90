program test_derivative_gp_device
    !! Independent device-boundary checks for mixed value/derivative GPs.
    !! The CUDA case must refuse before touching output arrays; a linked CUDA
    !! optimizer or kernel operator is not evidence of a derivative-GP kernel.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_DOMAIN_ERROR, &
        FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    use fortml_kernels, only: kernel_t, make_rbf_kernel
    use fortml_derivative_gaussian_process, only: gp_derivative_regression_t
    implicit none

    type(gp_derivative_regression_t) :: model
    type(kernel_t) :: kernel
    type(fortml_device_t) :: cpu, cuda, unselected
    type(fortnum_status_t) :: status
    real(dp) :: x(3, 1), y(3, 1), query(2, 1)
    real(dp) :: mean(2, 1), reference_mean(2, 1)
    real(dp) :: variance(2), reference_variance(2)
    real(dp) :: covariance(2, 2), reference_covariance(2, 2)
    integer :: failures

    failures = 0
    x(:, 1) = [0.0_dp, 0.45_dp, 1.05_dp]
    y(:, 1) = [1.2_dp, -0.3_dp, 0.8_dp]
    query(:, 1) = [0.25_dp, 0.8_dp]
    kernel = make_rbf_kernel(1, 1.4_dp, 0.75_dp, status)
    call model%fit(x, [0, 1, 0], y, kernel, 0.07_dp, status, jitter=1.0e-10_dp)
    call check(status_ok(status), "mixed derivative GP fit", failures)
    call check(model%device_supported(FORTML_DEVICE_CPU), &
        "CPU capability metadata", failures)
    call check(.not. model%device_supported(FORTML_DEVICE_CUDA), &
        "CUDA capability metadata refuses absent resident kernel", failures)

    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    mean = 1234.0_dp
    variance = 5678.0_dp
    call model%predict_device(cuda, query, [1, 0], mean, variance, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "CUDA derivative-GP prediction refusal", failures)
    call check(all(mean == 1234.0_dp) .and. all(variance == 5678.0_dp), &
        "CUDA refusal leaves outputs untouched", failures)
    covariance = 4321.0_dp
    call model%joint_covariance_device(cuda, query, [1, 0], covariance, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "CUDA joint covariance refusal", failures)
    call check(all(covariance == 4321.0_dp), &
        "CUDA joint covariance refusal leaves output untouched", failures)

    call cpu%select(FORTML_DEVICE_CPU, status)
    call check(status_ok(status), "CPU device selection", failures)
    call model%predict_device(cpu, query, [1, 0], mean, variance, status)
    call model%predict(query, [1, 0], reference_mean, reference_variance, status)
    call check(status_ok(status) .and. maxval(abs(mean - reference_mean)) < 2.0e-14_dp &
        .and. maxval(abs(variance - reference_variance)) < 2.0e-14_dp, &
        "CPU device dispatch matches reference prediction", failures)
    call model%joint_covariance_device(cpu, query, [1, 0], covariance, status)
    call model%joint_covariance(query, [1, 0], reference_covariance, status)
    call check(status_ok(status) .and. maxval(abs(covariance - reference_covariance)) < 2.0e-14_dp, &
        "CPU joint covariance dispatch matches reference", failures)

    call model%predict_device(unselected, query, [1, 0], mean, variance, status)
    call check(status%code == FORTNUM_DOMAIN_ERROR, &
        "unselected device is rejected", failures)

    if (failures /= 0) then
        write (error_unit, '(a,i0)') "FAIL derivative GP device cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS derivative GP device contract independent oracle"

contains

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "  FAIL [derivative-gp-device] "//description
        end if
    end subroutine check

end program test_derivative_gp_device
