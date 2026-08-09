program fortml_bench_lagrangian_mlp
    !! Correctness-gated scalar Lagrangian residual workload.
    use, intrinsic :: iso_fortran_env, only: dp => real64
    use fortml_lagrangian_mlp, only: lagrangian_mlp_t
    use fortnum_status, only: fortnum_status_t, status_ok
    use fortml_device, only: FORTML_DEVICE_CUDA
    implicit none

    type(lagrangian_mlp_t) :: model
    type(fortnum_status_t) :: status
    real(dp) :: q(1, 1), v(1, 1), acceleration(1, 1), values(1), gradient(1, 2)
    real(dp) :: mass(1, 1, 1), residual(1, 1)
    real(dp), parameter :: theta(9) = [0.7_dp, 0.2_dp, -0.4_dp, 0.3_dp, 0.1_dp, &
        -0.2_dp, 0.4_dp, -0.5_dp, 0.3_dp]

    call model%initialize(1, [2, 1], status, initialization_seed=13)
    if (.not. status_ok(status)) error stop "Lagrangian benchmark initialization failed"
    call model%set_parameters(theta, status)
    if (.not. status_ok(status)) error stop "Lagrangian benchmark parameter setup failed"
    q(1, 1) = 0.3_dp
    v(1, 1) = -0.4_dp
    acceleration(1, 1) = 0.2_dp
    call model%lagrangian_gradient(q, v, values, gradient, status)
    if (.not. status_ok(status)) error stop "Lagrangian benchmark gradient failed"
    call model%mass_matrix(q, v, mass, status)
    if (.not. status_ok(status)) error stop "Lagrangian benchmark mass failed"
    call model%euler_lagrange_residual(q, v, acceleration, residual, status)
    if (.not. status_ok(status)) error stop "Lagrangian benchmark residual failed"
    write (*, '(a,es24.16)') "lagrangian_value,", values(1)
    write (*, '(a,es24.16)') "lagrangian_gradient_q,", gradient(1, 1)
    write (*, '(a,es24.16)') "lagrangian_gradient_v,", gradient(1, 2)
    write (*, '(a,es24.16)') "lagrangian_mass,", mass(1, 1, 1)
    write (*, '(a,es24.16)') "lagrangian_residual,", residual(1, 1)
    call model%select_device(FORTML_DEVICE_CUDA, status)
    if (status_ok(status)) error stop "Lagrangian benchmark CUDA refusal changed"
    write (*, '(a)') "lagrangian_cuda,unavailable"
end program fortml_bench_lagrangian_mlp
