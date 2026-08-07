program test_kernel_clone
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_kernels, only: kernel_t, clone_kernel, kernel_multiply, &
        make_linear_kernel, make_rbf_kernel
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    type(kernel_t) :: left, right, product, copy
    type(fortnum_status_t) :: status
    real(dp) :: original(3), changed(3)
    integer :: failures

    failures = 0
    left = make_rbf_kernel(1, 1.2_dp, 0.7_dp, status)
    right = make_linear_kernel(1, 0.8_dp, status)
    product = kernel_multiply(left, right, status)
    copy = clone_kernel(product)
    original = product%parameters()
    changed = original
    changed(1) = changed(1) + 0.4_dp
    call copy%set_parameters(changed, status)
    if (.not. status_ok(status) .or. maxval(abs(product%parameters() - original)) > 1.0e-14_dp) then
        write (error_unit, '(a)') "FAIL [kernel clone] child mutation leaked into source"
        failures = failures + 1
    end if
    if (abs(copy%value([0.2_dp], [0.9_dp]) - &
        product%value([0.2_dp], [0.9_dp])) < 1.0e-8_dp) then
        write (error_unit, '(a)') "FAIL [kernel clone] clone did not change independently"
        failures = failures + 1
    end if
    if (failures /= 0) error stop 1
    write (*, '(a)') "PASS: kernel clone deep-copy contract"
end program test_kernel_clone
