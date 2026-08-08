program fortml_bench_pipeline_schema
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortml_basis, only: basis_map_t, make_fourier_basis, &
        make_polynomial_basis
    use fortml_pipeline, only: basis_pipeline_t, make_basis_pipeline
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer, parameter :: n_inputs = 3, repetitions = 10000
    character(len=16) :: names(n_inputs), candidate(n_inputs)
    real(dp) :: frequencies(1, n_inputs), x(1, n_inputs), phi(1, 9)
    real(dp) :: elapsed
    integer(int64) :: clock_start, clock_end, clock_rate
    integer :: i, valid_count
    type(basis_map_t) :: polynomial, fourier
    type(basis_pipeline_t) :: pipeline
    type(fortnum_status_t) :: status

    names = [character(len=16) :: "time", "position", "velocity"]
    candidate = names
    frequencies = reshape([0.7_dp, 1.1_dp, 0.9_dp], shape(frequencies))
    x = reshape([0.2_dp, -0.4_dp, 0.8_dp], shape(x))
    polynomial = make_polynomial_basis(n_inputs, 1, status)
    if (.not. status_ok(status)) error stop "schema benchmark polynomial failed"
    fourier = make_fourier_basis(n_inputs, frequencies, status)
    if (.not. status_ok(status)) error stop "schema benchmark Fourier failed"
    pipeline = make_basis_pipeline(n_inputs, status)
    call pipeline%append(polynomial, status, name="linear")
    call pipeline%append(fourier, status, name="harmonic")
    call pipeline%set_input_schema(names, status)
    call pipeline%validate_input_schema(candidate, status)
    if (.not. status_ok(status)) error stop "schema benchmark setup failed"

    call system_clock(clock_start, clock_rate)
    valid_count = 0
    do i = 1, repetitions
        call pipeline%validate_input_schema(candidate, status)
        if (status_ok(status)) valid_count = valid_count + 1
    end do
    call system_clock(clock_end)
    elapsed = real(clock_end - clock_start, dp)/real(clock_rate, dp)
    write (*, '(a,i0,a,i0,a,i0,a,es24.16,a,i0,a,a)') &
        "pipeline_schema,", n_inputs, ",", pipeline%feature_count(), ",", &
        repetitions, ",", elapsed/real(repetitions, dp), ",", valid_count, ",", &
        trim(pipeline%input_schema_name(1))

    call pipeline%transform(x, phi, status)
    if (.not. status_ok(status)) error stop "schema benchmark transform failed"
end program fortml_bench_pipeline_schema
