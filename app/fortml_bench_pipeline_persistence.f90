program fortml_bench_pipeline_persistence
    !! Release workload for versioned basis-pipeline state persistence.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortml_basis, only: basis_map_t, make_fourier_basis, make_polynomial_basis
    use fortml_pipeline, only: basis_pipeline_t, make_basis_pipeline
    use fortml_pipeline_persistence, only: save_basis_pipeline_text, &
        load_basis_pipeline_text
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer, parameter :: n_samples = 64, n_inputs = 1
    real(dp) :: x(n_samples, n_inputs), phi(n_samples, 4), phi_loaded(n_samples, 4)
    real(dp) :: expected(n_samples, 4), elapsed, roundtrip_error, oracle_error
    real(dp) :: frequencies(1, 1)
    integer(int64) :: clock_start, clock_end, clock_rate
    integer :: i, unit, ios
    type(basis_map_t) :: polynomial, fourier
    type(basis_pipeline_t) :: pipeline, loaded
    type(fortnum_status_t) :: status
    character(len=*), parameter :: path = &
        "/mnt/storage/fortml_pipeline_persistence_benchmark.txt"

    do i = 1, n_samples
        x(i, 1) = -1.0_dp + 2.0_dp*real(i - 1, dp)/real(n_samples - 1, dp)
    end do
    frequencies = reshape([0.8_dp], shape(frequencies))
    polynomial = make_polynomial_basis(n_inputs, 2, status)
    if (.not. status_ok(status)) error stop "persistence benchmark polynomial failed"
    fourier = make_fourier_basis(n_inputs, frequencies, status)
    if (.not. status_ok(status)) error stop "persistence benchmark Fourier failed"
    pipeline = make_basis_pipeline(n_inputs, status)
    call pipeline%append(polynomial, status, name="trend")
    call pipeline%append(fourier, status, name="seasonal")
    call pipeline%fit(x, status)
    call pipeline%transform(x, phi, status)
    if (.not. status_ok(status)) error stop "persistence benchmark setup failed"
    expected(:, 1) = x(:, 1)
    expected(:, 2) = x(:, 1)**2
    expected(:, 3) = sin(0.8_dp*x(:, 1))
    expected(:, 4) = cos(0.8_dp*x(:, 1))
    oracle_error = maxval(abs(phi - expected))

    call system_clock(clock_start, clock_rate)
    call save_basis_pipeline_text(pipeline, path, status)
    if (.not. status_ok(status)) error stop "persistence benchmark save failed"
    loaded = make_basis_pipeline(n_inputs, status)
    call loaded%append(polynomial, status, name="trend")
    call loaded%append(fourier, status, name="seasonal")
    call loaded%fit(x, status)
    call load_basis_pipeline_text(loaded, path, status)
    if (.not. status_ok(status)) error stop "persistence benchmark load failed"
    call loaded%transform(x, phi_loaded, status)
    call system_clock(clock_end)
    if (.not. status_ok(status)) error stop "persistence benchmark transform failed"
    elapsed = real(clock_end - clock_start, dp)/real(clock_rate, dp)
    roundtrip_error = maxval(abs(phi_loaded - phi))
    open (newunit=unit, file=path, status="old", iostat=ios)
    if (ios == 0) close (unit, status="delete")

    write (*, '(a,",",a,",",i0,",",i0,",",i0,",",es24.16,",",es24.16,",",es24.16,",",i0)') &
        "pipeline_persistence", "round_trip", n_samples, loaded%feature_count(), &
        loaded%parameter_count(), elapsed, roundtrip_error, oracle_error, &
        loaded%stage_feature_offset(2) + loaded%stage_parameter_offset(2)
end program fortml_bench_pipeline_persistence
