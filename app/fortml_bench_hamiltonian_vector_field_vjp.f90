program fortml_bench_hamiltonian_vector_field_vjp
    !! Release executable for the canonical Hamiltonian vector-field VJP.
    use, intrinsic :: iso_fortran_env, only: output_unit
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_ok
    use fortml_hamiltonian_mlp, only: hamiltonian_mlp_t
    implicit none

    type(hamiltonian_mlp_t) :: model
    type(fortnum_status_t) :: status
    real(dp), allocatable :: theta(:), dtheta(:), parameter_bar(:)
    real(dp) :: state(1, 2), field_bar(1, 2), state_bar(1, 2)
    real(dp) :: dstate(1, 2), field(1, 2), dfield(1, 2)
    real(dp) :: adjoint_error
    integer :: i

    call model%initialize(1, [1, 3, 1], [1, 3, 1], status, initialization_seed=23)
    if (.not. status_ok(status)) error stop 1
    theta = model%parameters()
    allocate(dtheta(size(theta)), parameter_bar(size(theta)))
    dtheta = [(0.006_dp*real(i, dp), i=1, size(theta))]
    state = reshape([0.29_dp, -0.34_dp], shape(state))
    dstate = reshape([-0.08_dp, 0.12_dp], shape(dstate))
    field_bar = reshape([0.61_dp, -0.37_dp], shape(field_bar))

    call model%vector_field_vjp(state, field_bar, parameter_bar, state_bar, status)
    if (.not. status_ok(status)) error stop 1
    call model%vector_field_jvp(state, dtheta, dstate, field, dfield, status)
    if (.not. status_ok(status)) error stop 1
    adjoint_error = abs(sum(field_bar*dfield) - &
        (sum(parameter_bar*dtheta) + sum(state_bar*dstate)))

    write (output_unit, '(a)') &
        "workload,parameter_count,adjoint_error,device,status"
    write (output_unit, '(a,",",i0,",",es24.16,",",a)') &
        "hamiltonian_vector_field_vjp", size(theta), adjoint_error, "cpu,pass"
end program fortml_bench_hamiltonian_vector_field_vjp
