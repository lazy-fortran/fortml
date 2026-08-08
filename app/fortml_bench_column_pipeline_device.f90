program fortml_bench_column_pipeline_device
    !! Correctness-gated column feature-union device workload.
    use, intrinsic :: iso_fortran_env, only: real64, int64
    use fortml_basis, only: basis_map_t, make_fourier_basis, &
        make_polynomial_basis
    use fortml_column_pipeline, only: column_basis_pipeline_t, &
        make_column_basis_pipeline
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    implicit none

    integer, parameter :: dp = real64
    integer, parameter :: n_samples = 2048, n_inputs = 3, n_features = 4
    real(dp) :: x(n_samples, n_inputs), features(n_samples, n_features)
    real(dp) :: expected(n_samples, n_features), features_sentinel(n_samples, n_features)
    real(dp) :: frequencies(1, 1)
    integer(int64) :: started, finished, rate
    real(dp) :: transform_seconds, max_abs_error
    type(basis_map_t) :: fourier, polynomial
    type(column_basis_pipeline_t) :: pipeline
    type(fortml_device_t) :: cpu, cuda
    type(fortnum_status_t) :: status
    integer :: i, repetitions

    do i = 1, n_samples
        x(i, 1) = -1.0_dp + 2.0_dp*real(mod(i - 1, 101), dp)/100.0_dp
        x(i, 2) = sin(0.017_dp*real(i, dp))
        x(i, 3) = cos(0.013_dp*real(i, dp))
    end do
    frequencies(1, 1) = 0.8_dp
    fourier = make_fourier_basis(1, frequencies, status)
    polynomial = make_polynomial_basis(1, 2, status)
    pipeline = make_column_basis_pipeline(n_inputs, status)
    call pipeline%append(fourier, [3], status, name="seasonal")
    call pipeline%append(polynomial, [1], status, name="trend")
    call pipeline%fit(x, status)
    if (.not. status_ok(status)) error stop "column pipeline benchmark setup failed"

    expected(:, 1) = sin(0.8_dp*x(:, 3))
    expected(:, 2) = cos(0.8_dp*x(:, 3))
    expected(:, 3) = x(:, 1)
    expected(:, 4) = x(:, 1)**2
    call cpu%select(FORTML_DEVICE_CPU, status)
    if (.not. status_ok(status)) error stop "column pipeline CPU selection failed"
    call system_clock(started, rate)
    do repetitions = 1, 64
        call pipeline%transform_device(cpu, x, features, status)
    end do
    call system_clock(finished)
    if (.not. status_ok(status)) error stop "column pipeline CPU transform failed"
    transform_seconds = real(finished - started, dp)/real(rate, dp)/64.0_dp
    max_abs_error = maxval(abs(features - expected))
    if (max_abs_error > 2.0e-13_dp) error stop "column pipeline oracle mismatch"

    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    features_sentinel = 1234.0_dp
    call pipeline%transform_device(cuda, x, features_sentinel, status)
    if (status%code /= FORTNUM_NOT_IMPLEMENTED .or. &
        any(features_sentinel /= 1234.0_dp)) then
        error stop "column pipeline CUDA contract changed unexpectedly"
    end if

    write (*, '(a,es24.16)') "column_pipeline_transform_seconds,", transform_seconds
    write (*, '(a,es24.16)') "column_pipeline_cpu_max_abs_error,", max_abs_error
    write (*, '(a,i0)') "column_pipeline_feature_count,", pipeline%feature_count()
    write (*, '(a)') "column_pipeline_cuda,unavailable"
end program fortml_bench_column_pipeline_device
