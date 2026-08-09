program fortml_bench_sequential_pipeline_clone
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortml_basis, only: basis_map_t, make_fourier_basis, &
        make_polynomial_basis
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortml_pipeline, only: sequential_basis_pipeline_t, &
        make_sequential_basis_pipeline
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer, parameter :: n_inputs = 1, n_samples = 64, repetitions = 5000
    real(dp) :: x(n_samples, n_inputs), y(n_samples, 4), clone_y(n_samples, 4)
    real(dp) :: elapsed, checksum
    integer(int64) :: clock_start, clock_end, clock_rate
    integer :: i, clone_code
    type(basis_map_t) :: polynomial, fourier
    type(sequential_basis_pipeline_t) :: pipeline, clone
    type(fortml_device_t) :: device
    type(fortnum_status_t) :: status

    do i = 1, n_samples
        x(i, 1) = -0.9_dp + 1.8_dp * real(i - 1, dp) / real(n_samples - 1, dp)
    end do
    polynomial = make_polynomial_basis(n_inputs, 2, status)
    fourier = make_fourier_basis(2, reshape([0.7_dp, 1.1_dp], [1, 2]), status)
    pipeline = make_sequential_basis_pipeline(n_inputs, status)
    call pipeline%append(polynomial, status, "powers")
    call pipeline%append(fourier, status, "harmonics")
    call pipeline%fit(x, status)
    if (.not. status_ok(status)) error stop "sequential clone benchmark setup failed"

    call system_clock(clock_start, clock_rate)
    do i = 1, repetitions
        call pipeline%clone(clone, status)
        if (.not. status_ok(status)) error stop "sequential clone benchmark clone failed"
    end do
    call system_clock(clock_end)
    elapsed = real(clock_end - clock_start, dp) / real(clock_rate, dp)
    call pipeline%transform(x, y, status)
    call clone%transform(x, clone_y, status)
    checksum = maxval(abs(y - clone_y))

    device%kind = FORTML_DEVICE_CUDA
    device%selected = .true.
    device%available = .true.
    call pipeline%clone_device(device, clone, status)
    clone_code = status%code
    write (*, '(a,i0,a,i0,a,es24.16,a,es24.16,a,i0)') &
        "sequential_pipeline_clone,", pipeline%feature_count(), ",", repetitions, ",", &
        elapsed / real(repetitions, dp), ",", checksum, ",", clone_code
end program fortml_bench_sequential_pipeline_clone
