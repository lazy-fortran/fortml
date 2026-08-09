program fortml_bench_basis_pipeline_clone
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortml_basis, only: basis_map_t, make_polynomial_basis, make_radial_basis
    use fortml_pipeline, only: basis_pipeline_t, make_basis_pipeline
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer, parameter :: n_inputs = 2, repetitions = 5000
    real(dp) :: centers(n_inputs, 1), scales(n_inputs, 1), x(8, n_inputs)
    real(dp) :: phi(8, 6), clone_phi(8, 6), elapsed, checksum
    integer(int64) :: clock_start, clock_end, clock_rate
    integer :: i, clone_code
    type(basis_map_t) :: polynomial, radial
    type(basis_pipeline_t) :: pipeline, clone
    type(fortml_device_t) :: device
    type(fortnum_status_t) :: status

    centers = reshape([0.25_dp, -0.50_dp], shape(centers))
    scales = reshape([0.70_dp, 1.10_dp], shape(scales))
    x = reshape([0.2_dp, -0.3_dp, 0.7_dp, 0.1_dp, -0.4_dp, 0.8_dp, &
        0.5_dp, -0.9_dp, 0.3_dp, 0.2_dp, -0.6_dp, 0.4_dp, &
        0.9_dp, -0.1_dp, -0.2_dp, 0.6_dp], shape(x))
    polynomial = make_polynomial_basis(n_inputs, 2, status, include_intercept=.true.)
    radial = make_radial_basis(n_inputs, centers, scales, status)
    pipeline = make_basis_pipeline(n_inputs, status)
    call pipeline%append(polynomial, status, "polynomial")
    call pipeline%append(radial, status, "radial")
    call pipeline%fit(x, status)
    if (.not. status_ok(status)) error stop "pipeline clone benchmark setup failed"

    call system_clock(clock_start, clock_rate)
    do i = 1, repetitions
        call pipeline%clone(clone, status)
        if (.not. status_ok(status)) error stop "pipeline clone benchmark clone failed"
    end do
    call system_clock(clock_end)
    elapsed = real(clock_end - clock_start, dp)/real(clock_rate, dp)
    call pipeline%transform(x, phi, status)
    call clone%transform(x, clone_phi, status)
    checksum = maxval(abs(phi - clone_phi))

    device%kind = FORTML_DEVICE_CUDA
    device%selected = .true.
    device%available = .true.
    call pipeline%clone_device(device, clone, status)
    clone_code = status%code
    write (*, '(a,i0,a,i0,a,es24.16,a,es24.16,a,i0)') &
        "pipeline_clone,", pipeline%feature_count(), ",", repetitions, ",", &
        elapsed/real(repetitions, dp), ",", checksum, ",", clone_code
end program fortml_bench_basis_pipeline_clone
