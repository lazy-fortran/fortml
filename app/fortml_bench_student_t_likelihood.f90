program fortml_bench_student_t_likelihood
    !! Release probe for the fixed-state Student-t likelihood coordinate.
    use, intrinsic :: iso_fortran_env, only: dp => real64
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortml_kernels, only: kernel_t, make_rbf_kernel
    use fortml_student_t_process, only: student_t_process_t
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    type(student_t_process_t) :: process
    type(kernel_t) :: kernel
    type(fortnum_status_t) :: status
    type(fortml_device_t) :: cuda
    real(dp) :: x(7, 1), y(7), value, tangent, vjp(1), hvp(1)
    real(dp) :: theta(1)
    integer :: cuda_jvp_code, cuda_vjp_code, cuda_hvp_code, i

    do i = 1, size(y)
        x(i, 1) = -1.5_dp + 0.5_dp*real(i - 1, dp)
        y(i) = sin(1.7_dp*x(i, 1)) + 0.25_dp*x(i, 1)
    end do
    kernel = make_rbf_kernel(1, 1.0_dp, 0.8_dp, status)
    if (.not. status_ok(status)) error stop "Student-t likelihood kernel failed"
    call process%fit(x, y, kernel, 4.7_dp, 0.01_dp, status)
    if (.not. status_ok(status)) error stop "Student-t likelihood fit failed"
    theta = process%likelihood_parameters()
    call process%log_marginal_likelihood_likelihood_parameter_jvp( &
        y, [1.0_dp], value, tangent, status)
    if (.not. status_ok(status)) error stop "Student-t likelihood JVP failed"
    call process%log_marginal_likelihood_likelihood_parameter_vjp( &
        y, 1.0_dp, vjp, status)
    if (.not. status_ok(status)) error stop "Student-t likelihood VJP failed"
    call process%log_marginal_likelihood_likelihood_parameter_hvp( &
        y, [1.0_dp], hvp, status)
    if (.not. status_ok(status)) error stop "Student-t likelihood HVP failed"
    write (*, '(a,es24.16)') "student_t_likelihood_theta,", theta(1)
    write (*, '(a,es24.16)') "student_t_likelihood_value,", value
    write (*, '(a,es24.16)') "student_t_likelihood_jvp,", tangent
    write (*, '(a,es24.16)') "student_t_likelihood_vjp,", vjp(1)
    write (*, '(a,es24.16)') "student_t_likelihood_hvp,", hvp(1)
    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call process%log_marginal_likelihood_likelihood_parameter_jvp_device( &
        cuda, y, [1.0_dp], value, tangent, status)
    cuda_jvp_code = status%code
    call process%log_marginal_likelihood_likelihood_parameter_vjp_device( &
        cuda, y, 1.0_dp, vjp, status)
    cuda_vjp_code = status%code
    call process%log_marginal_likelihood_likelihood_parameter_hvp_device( &
        cuda, y, [1.0_dp], hvp, status)
    cuda_hvp_code = status%code
    write (*, '(a,i0)') "student_t_likelihood_cuda_jvp,", cuda_jvp_code
    write (*, '(a,i0)') "student_t_likelihood_cuda_vjp,", cuda_vjp_code
    write (*, '(a,i0)') "student_t_likelihood_cuda_hvp,", cuda_hvp_code
end program fortml_bench_student_t_likelihood
