program fortml_bench_trainer_schedule
    !! Release probe for the generic trainer-owned one-cycle schedule.
    use, intrinsic :: iso_fortran_env, only: int64
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_ok, status_set, FORTNUM_OK
    use fortopt_objective, only: objective_t
    use fortml_mlp_schedules, only: make_mlp_schedule_one_cycle
    use fortml_trainer, only: trainer_t, trainer_options_t, trainer_state_t, &
        FORTML_TRAIN_SGD
    implicit none

    type(objective_t) :: objective
    type(trainer_t) :: trainer
    type(trainer_options_t) :: options
    type(trainer_state_t) :: state
    type(fortnum_status_t) :: status
    real(dp) :: elapsed
    integer(int64) :: tick_start, tick_end, ticks_per_second
    integer :: i

    call objective%initialize(2, quadratic_objective, status)
    if (.not. status_ok(status)) error stop "trainer schedule objective failed"
    options%optimizer = FORTML_TRAIN_SGD
    options%learning_rate = 0.1_dp
    options%max_steps = 4
    options%tolerance = 0.0_dp
    options%step_tolerance = 0.0_dp
    options%objective_tolerance = 0.0_dp
    options%use_learning_rate_schedule = .true.
    options%learning_rate_schedule = make_mlp_schedule_one_cycle(2, 4, 2.0_dp, 0.25_dp)
    call trainer%initialize(objective, [0.0_dp, 1.0_dp], status, options)
    if (.not. status_ok(status)) error stop "trainer schedule initialization failed"

    call system_clock(count_rate=ticks_per_second)
    call system_clock(tick_start)
    do i = 1, 4
        call trainer%step(status)
        if (.not. status_ok(status)) error stop "trainer schedule update failed"
    end do
    call system_clock(tick_end)
    elapsed = real(tick_end-tick_start, dp)/real(ticks_per_second, dp)
    state = trainer%state_copy()
    write (*, '(a,",",i0,",",es24.16,",",es24.16,",",es24.16,",",es24.16,",",es24.16)') &
        "trainer_schedule,4", state%steps, state%learning_rate_history(2), &
        state%learning_rate_history(3), state%learning_rate_history(4), &
        state%learning_rate_history(5), elapsed
    write (*, '(a,",",es24.16,",",es24.16)') "trainer_schedule_parameters", &
        state%parameters(1), state%parameters(2)

contains

    subroutine quadratic_objective(x, value, gradient, status)
        real(dp), intent(in) :: x(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status

        if (size(x) /= 2 .or. size(gradient) /= 2) then
            call status_set(status, 1, "quadratic shape")
            return
        end if
        value = (x(1) - 1.5_dp)**2 + 2.0_dp*(x(2) + 0.5_dp)**2
        gradient = [2.0_dp*(x(1) - 1.5_dp), 4.0_dp*(x(2) + 0.5_dp)]
        call status_set(status, FORTNUM_OK, "")
    end subroutine quadratic_objective

end program fortml_bench_trainer_schedule
