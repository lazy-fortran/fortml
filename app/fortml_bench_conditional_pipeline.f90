program fortml_bench_conditional_pipeline
    !! Correctness-gated conditional feature-union CPU workload.
    use, intrinsic :: iso_fortran_env, only: real64, int64
    use fortml_basis, only: basis_map_t, make_fourier_basis
    use fortml_conditional_pipeline, only: conditional_basis_pipeline_t, &
        make_conditional_basis_pipeline
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    implicit none

    integer, parameter :: dp = real64
    integer, parameter :: n_samples = 2048, n_inputs = 2, n_features = 4
    real(dp) :: x(n_samples, n_inputs), x_dot(n_samples, n_inputs)
    real(dp) :: features(n_samples, n_features), feature_dot(n_samples, n_features)
    real(dp) :: cotangent(n_samples, n_features), theta_dot(2)
    real(dp) :: theta_bar(2), x_bar(n_samples, n_inputs)
    real(dp) :: theta_hvp(2), x_hvp(n_samples, n_inputs)
    real(dp) :: frequencies(1, 1), max_route_error, elapsed
    integer(int64) :: started, finished, rate
    integer :: i, repetitions
    type(basis_map_t) :: left, right
    type(conditional_basis_pipeline_t) :: pipeline
    type(fortml_device_t) :: cpu, cuda
    type(fortnum_status_t) :: status

    do i = 1, n_samples
        x(i, 1) = -1.0_dp + 2.0_dp*(real(mod(i - 1, 200), dp) + 0.5_dp)/200.0_dp
        x(i, 2) = sin(0.017_dp*real(i, dp))
        x_dot(i, 1) = 0.0_dp
        x_dot(i, 2) = cos(0.017_dp*real(i, dp))
        cotangent(i, :) = [0.13_dp, -0.22_dp, 0.31_dp, -0.17_dp]
    end do
    frequencies(1, 1) = 0.8_dp
    left = make_fourier_basis(1, frequencies, status)
    right = make_fourier_basis(1, frequencies, status)
    pipeline = make_conditional_basis_pipeline(n_inputs, status)
    call pipeline%append(left, [2], 1, -2.0_dp, 0.0_dp, status, name="left")
    call pipeline%append(right, [2], 1, 0.0_dp, 2.0_dp, status, name="right")
    call pipeline%fit(x, status)
    if (.not. status_ok(status)) error stop "conditional pipeline benchmark setup failed"
    theta_dot = [0.17_dp, -0.23_dp]
    call cpu%select(FORTML_DEVICE_CPU, status)
    if (.not. status_ok(status)) error stop "conditional pipeline CPU selection failed"

    call system_clock(started, rate)
    do repetitions = 1, 64
        call pipeline%transform_device(cpu, x, features, status)
    end do
    call system_clock(finished)
    if (.not. status_ok(status)) error stop "conditional pipeline transform failed"
    elapsed = real(finished - started, dp)/real(rate, dp)/64.0_dp
    max_route_error = 0.0_dp
    do i = 1, n_samples
        if (x(i, 1) < 0.0_dp) max_route_error = max(max_route_error, &
            maxval(abs(features(i, 3:4))))
        if (x(i, 1) >= 0.0_dp) max_route_error = max(max_route_error, &
            maxval(abs(features(i, 1:2))))
    end do
    if (max_route_error > 2.0e-13_dp) error stop "conditional route oracle mismatch"
    call pipeline%jvp(x, theta_dot, x_dot, features, feature_dot, status)
    call pipeline%vjp(x, cotangent, theta_bar, x_bar, status)
    call pipeline%hvp(x, cotangent, theta_dot, x_dot, theta_hvp, x_hvp, status)
    if (.not. status_ok(status)) error stop "conditional pipeline product failed"

    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    features = 1234.0_dp
    call pipeline%transform_device(cuda, x, features, status)
    if (status%code /= FORTNUM_NOT_IMPLEMENTED .or. any(features /= 1234.0_dp)) then
        error stop "conditional pipeline CUDA contract changed unexpectedly"
    end if

    write (*, '(a,",",es24.16)') "conditional_pipeline_transform_seconds", elapsed
    write (*, '(a,",",es24.16)') "conditional_pipeline_route_error", max_route_error
    write (*, '(a,",",i0)') "conditional_pipeline_branch_count", pipeline%branch_count()
    write (*, '(a,",",i0)') "conditional_pipeline_feature_count", pipeline%feature_count()
    write (*, '(a)') "conditional_pipeline_cuda,unavailable"
end program fortml_bench_conditional_pipeline
