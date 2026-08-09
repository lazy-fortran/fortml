program fortml_bench_basis_blend_pipeline
    !! Correctness-gated learned basis fan-in CPU workload.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortml_basis, only: basis_map_t, make_polynomial_basis, make_fourier_basis
    use fortml_pipeline, only: sequential_basis_pipeline_t
    use fortml_basis_blend_pipeline, only: basis_blend_pipeline_t, &
        make_basis_blend_pipeline
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, &
        FORTML_DEVICE_CUDA
    use fortml_estimator_capabilities, only: FORTML_CAPABILITY_DEVICE_OPENACC
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    implicit none

    integer, parameter :: n_samples = 4096, n_inputs = 2, n_features = 4
    integer, parameter :: repetitions = 64
    real(dp) :: x(n_samples, n_inputs), x_dot(n_samples, n_inputs)
    real(dp) :: y(n_samples, n_features), y_dot(n_samples, n_features)
    real(dp) :: polynomial_y(n_samples, n_features)
    real(dp) :: fourier_y(n_samples, n_features), u(n_samples, n_features)
    real(dp) :: x_bar(n_samples, n_inputs), x_hvp(n_samples, n_inputs)
    real(dp) :: theta_dot(4), theta_bar(4), theta_hvp(4)
    real(dp) :: frequencies(1, 2), elapsed, value_error, adjoint_error
    integer(int64) :: started, finished, rate
    integer :: i, repetition
    type(basis_map_t) :: polynomial_map, fourier_map
    type(sequential_basis_pipeline_t) :: polynomial_branch, fourier_branch
    type(basis_blend_pipeline_t) :: blend
    type(fortml_device_t) :: cpu, accelerator
    type(fortnum_status_t) :: status

    do i = 1, n_samples
        x(i, 1) = sin(0.013_dp*real(i, dp))
        x(i, 2) = cos(0.019_dp*real(i, dp))
        x_dot(i, 1) = 0.2_dp*cos(0.007_dp*real(i, dp))
        x_dot(i, 2) = -0.3_dp*sin(0.011_dp*real(i, dp))
        u(i, :) = [0.13_dp, -0.22_dp, 0.31_dp, -0.17_dp]
    end do
    polynomial_map = make_polynomial_basis(n_inputs, 2, status)
    frequencies = reshape([0.7_dp, 1.1_dp], shape(frequencies))
    fourier_map = make_fourier_basis(n_inputs, frequencies, status)
    call polynomial_branch%initialize(n_inputs, status)
    call polynomial_branch%append(polynomial_map, status, "quadratic")
    call fourier_branch%initialize(n_inputs, status)
    call fourier_branch%append(fourier_map, status, "spectral")
    blend = make_basis_blend_pipeline(n_inputs, status)
    call blend%append(polynomial_branch, 1.25_dp, status, "trend")
    call blend%append(fourier_branch, -0.4_dp, status, "oscillation")
    call blend%fit(x, status)
    if (.not. status_ok(status)) error stop "basis blend benchmark setup failed"
    theta_dot = [-0.05_dp, 0.04_dp, 0.09_dp, -0.07_dp]
    call cpu%select(FORTML_DEVICE_CPU, status)
    if (.not. status_ok(status)) error stop "basis blend CPU selection failed"

    call system_clock(started, rate)
    do repetition = 1, repetitions
        call blend%transform_device(cpu, x, y, status)
    end do
    call system_clock(finished)
    if (.not. status_ok(status)) error stop "basis blend transform failed"
    elapsed = real(finished - started, dp)/real(rate, dp)/real(repetitions, dp)
    call polynomial_map%evaluate(x, polynomial_y, status)
    call fourier_map%evaluate(x, fourier_y, status)
    value_error = maxval(abs(y - (1.25_dp*polynomial_y - 0.4_dp*fourier_y)))
    call blend%jvp(x, theta_dot, x_dot, y, y_dot, status)
    call blend%vjp(x, u, theta_bar, x_bar, status)
    adjoint_error = abs(sum(u*y_dot) - dot_product(theta_bar, theta_dot) - &
        sum(x_bar*x_dot))
    call blend%hvp(x, u, theta_dot, x_dot, theta_hvp, x_hvp, status)
    if (.not. status_ok(status) .or. value_error > 3.0e-13_dp .or. &
        adjoint_error > 2.0e-10_dp) then
        error stop "basis blend correctness gate failed"
    end if

    accelerator%kind = FORTML_DEVICE_CUDA
    accelerator%selected = .true.
    accelerator%available = .true.
    y = 1234.0_dp
    call blend%transform_device(accelerator, x, y, status)
    if (status%code /= FORTNUM_NOT_IMPLEMENTED .or. any(y /= 1234.0_dp)) then
        error stop "basis blend CUDA contract changed unexpectedly"
    end if
    accelerator%kind = FORTML_CAPABILITY_DEVICE_OPENACC
    y = 4321.0_dp
    call blend%transform_device(accelerator, x, y, status)
    if (status%code /= FORTNUM_NOT_IMPLEMENTED .or. any(y /= 4321.0_dp)) then
        error stop "basis blend OpenACC contract changed unexpectedly"
    end if

    write (*, '(a,",",es24.16)') "basis_blend_transform_seconds", elapsed
    write (*, '(a,",",es24.16)') "basis_blend_value_error", value_error
    write (*, '(a,",",es24.16)') "basis_blend_adjoint_error", adjoint_error
    write (*, '(a,",",i0)') "basis_blend_branch_count", blend%branch_count()
    write (*, '(a,",",i0)') "basis_blend_parameter_count", &
        blend%parameter_count()
    write (*, '(a)') "basis_blend_cuda,unavailable"
    write (*, '(a)') "basis_blend_openacc,unavailable"
end program fortml_bench_basis_blend_pipeline
