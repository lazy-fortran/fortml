program fortml_bench_hamiltonian_structure_gp
    !! Finite-feature GP warm start for a separable Hamiltonian MLP.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortml_hamiltonian_mlp, only: hamiltonian_mlp_t
    use fortml_hamiltonian_structure_gp, only: &
        hamiltonian_structure_gp_initializer_t, hamiltonian_structure_gp_metadata_t
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer, parameter :: n_samples = 256, n_coordinates = 2, n_hidden = 16
    integer, parameter :: repetitions = 8
    real(dp) :: q(n_samples, n_coordinates), p(n_samples, n_coordinates)
    real(dp) :: potential_target(n_samples, 1), kinetic_target(n_samples, 1)
    real(dp), allocatable :: potential(:, :), kinetic(:, :)
    real(dp) :: fit_seconds, predict_seconds
    integer(int64) :: clock_start, clock_end, clock_rate
    integer :: i, j, repetition
    type(hamiltonian_mlp_t) :: model
    type(hamiltonian_structure_gp_initializer_t) :: initializer
    type(hamiltonian_structure_gp_metadata_t) :: metadata
    type(fortnum_status_t) :: status

    do j = 1, n_coordinates
        do i = 1, n_samples
            q(i, j) = sin(0.017_dp*real(i, dp) + 0.053_dp*real(j, dp))
            p(i, j) = cos(0.011_dp*real(i*j, dp))
        end do
    end do
    do i = 1, n_samples
        potential_target(i, 1) = 0.5_dp*sum(q(i, :)**2)
        kinetic_target(i, 1) = 0.5_dp*sum(p(i, :)**2)
    end do

    call model%initialize(n_coordinates, [n_coordinates, n_hidden, 1], &
        [n_coordinates, n_hidden, 1], status, initialization_seed=29)
    if (.not. status_ok(status)) error stop "Hamiltonian structure GP model failed"
    call system_clock(clock_start, clock_rate)
    call initializer%fit_apply(model, q, potential_target, p, kinetic_target, status, 0.1_dp)
    call system_clock(clock_end)
    if (.not. status_ok(status)) error stop "Hamiltonian structure GP fit failed"
    fit_seconds = real(clock_end - clock_start, dp)/real(clock_rate, dp)

    call initializer%predict_components(model, q, p, potential, kinetic, status)
    if (.not. status_ok(status)) error stop "Hamiltonian structure GP prediction failed"
    call system_clock(clock_start, clock_rate)
    do repetition = 1, repetitions
        call initializer%predict_components(model, q, p, potential, kinetic, status)
        if (.not. status_ok(status)) error stop "Hamiltonian structure GP timing failed"
    end do
    call system_clock(clock_end)
    predict_seconds = real(clock_end - clock_start, dp)/real(clock_rate, dp) &
        /real(repetitions, dp)
    metadata = initializer%metadata()
    write (*, '(a,i0,a,i0,a,i0,a,es24.16,a,es24.16,a,es24.16,a,es24.16,a,es24.16)') &
        "hamiltonian_structure_gp,", n_samples, ",", n_coordinates, ",", n_hidden, ",", &
        fit_seconds, ",", predict_seconds, ",", metadata%potential_fit_rmse, ",", &
        metadata%kinetic_fit_rmse, ",", metadata%structure_defect
end program fortml_bench_hamiltonian_structure_gp
