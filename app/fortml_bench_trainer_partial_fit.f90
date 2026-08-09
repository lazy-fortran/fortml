program fortml_bench_trainer_partial_fit
    !! Release workload for the generic trainer partial-fit contract.
    use, intrinsic :: iso_fortran_env, only: int64
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_ok, status_set, FORTNUM_OK
    use fortopt_objective, only: objective_t
    use fortml_device, only: FORTML_DEVICE_CUDA
    use fortml_trainer, only: trainer_t, trainer_options_t, trainer_state_t, &
        FORTML_TRAIN_ADAM
    implicit none

    type(objective_t) :: objective
    type(trainer_t) :: uninterrupted, chunked
    type(trainer_options_t) :: options
    type(trainer_state_t) :: state
    type(fortnum_status_t) :: status
    real(dp) :: elapsed, replay_error
    integer(int64) :: tick_start, tick_end, ticks_per_second

    call objective%initialize(3, diagonal_quadratic, status)
    if (.not. status_ok(status)) error stop 1
    options%optimizer = FORTML_TRAIN_ADAM
    options%learning_rate = 0.05_dp
    options%beta1 = 0.8_dp
    options%beta2 = 0.9_dp
    options%epsilon = 1.0e-9_dp
    options%max_steps = 6
    options%tolerance = 0.0_dp
    options%step_tolerance = 0.0_dp
    options%objective_tolerance = 0.0_dp
    call uninterrupted%initialize(objective, [0.0_dp, 1.0_dp, -1.0_dp], status, options)
    call chunked%initialize(objective, [0.0_dp, 1.0_dp, -1.0_dp], status, options)
    if (.not. status_ok(status)) error stop 1

    call system_clock(tick_start, ticks_per_second)
    call uninterrupted%partial_fit(6, status)
    call chunked%partial_fit(2, status)
    call chunked%partial_fit(4, status)
    if (.not. status_ok(status)) error stop 1
    call system_clock(tick_end)
    elapsed = real(tick_end - tick_start, dp)/real(ticks_per_second, dp)
    replay_error = maxval(abs(uninterrupted%parameters() - chunked%parameters()))
    state = chunked%state_copy()
    write (*, '(a,",",i0,",",es24.16,",",es24.16)') "trainer_partial_fit", &
        state%steps, replay_error, elapsed
    call chunked%partial_fit_device(FORTML_DEVICE_CUDA, 1, status)
    write (*, '(a,",",i0)') "trainer_partial_fit_cuda", status%code

contains

    subroutine diagonal_quadratic(x, value, gradient, status)
        real(dp), intent(in) :: x(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), parameter :: target(3) = [1.5_dp, -0.5_dp, 0.25_dp]
        real(dp), parameter :: curvature(3) = [2.0_dp, 4.0_dp, 1.25_dp]

        if (size(x) /= 3 .or. size(gradient) /= 3) then
            call status_set(status, 1, "quadratic: invalid shape")
            return
        end if
        gradient = curvature*(x - target)
        value = 0.5_dp*sum(curvature*(x - target)**2)
        call status_set(status, FORTNUM_OK, "")
    end subroutine diagonal_quadratic

end program fortml_bench_trainer_partial_fit
