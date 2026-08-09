program fortml_bench_trainer_fit_diagnostics
    !! Release workload for generic trainer fit diagnostics.
    use, intrinsic :: iso_fortran_env, only: int64
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_ok, status_set, FORTNUM_OK
    use fortopt_objective, only: objective_t
    use fortml_trainer, only: trainer_t, trainer_options_t, trainer_state_t, &
        FORTML_TRAIN_ADAM, FORTML_TRAIN_LBFGSB
    implicit none

    type(objective_t) :: objective
    type(trainer_t) :: lbfgsb, adam
    type(trainer_options_t) :: options
    type(trainer_state_t) :: state
    type(fortnum_status_t) :: status
    real(dp) :: elapsed, parameter_error
    integer(int64) :: tick_start, tick_end, ticks_per_second

    call objective%initialize(2, quadratic, status)
    if (.not. status_ok(status)) error stop 1

    options = trainer_options_t()
    options%optimizer = FORTML_TRAIN_LBFGSB
    options%max_steps = 64
    options%use_bounds = .true.
    options%lower = [0.0_dp, -1.0_dp]
    options%upper = [2.0_dp, 1.0_dp]
    options%lbfgsb%max_iterations = 64
    options%lbfgsb%gradient_tolerance = 1.0e-10_dp
    options%lbfgsb%step_tolerance = 1.0e-14_dp
    options%lbfgsb%objective_tolerance = 1.0e-14_dp
    call lbfgsb%initialize(objective, [0.0_dp, 1.0_dp], status, options)
    if (.not. status_ok(status)) error stop 1
    call system_clock(tick_start, ticks_per_second)
    call lbfgsb%fit(status)
    call system_clock(tick_end)
    if (.not. status_ok(status)) error stop 1
    elapsed = real(tick_end - tick_start, dp)/real(ticks_per_second, dp)
    state = lbfgsb%state_copy()
    parameter_error = maxval(abs(lbfgsb%parameters() - [1.5_dp, -0.5_dp]))
    write (*, '(a,",",i0,",",i0,",",i0,",",es24.16,",",es24.16)') &
        "trainer_fit_lbfgsb", state%optimizer_iterations, &
        state%line_search_evaluations, state%curvature_updates, parameter_error, elapsed

    options = trainer_options_t()
    options%optimizer = FORTML_TRAIN_ADAM
    options%learning_rate = 0.05_dp
    options%max_steps = 4
    options%tolerance = 0.0_dp
    options%step_tolerance = 0.0_dp
    options%objective_tolerance = 0.0_dp
    options%callback => stop_after_first_step
    call adam%initialize(objective, [0.0_dp, 1.0_dp], status, options)
    if (.not. status_ok(status)) error stop 1
    call system_clock(tick_start, ticks_per_second)
    call adam%fit(status)
    call system_clock(tick_end)
    if (.not. status_ok(status)) error stop 1
    elapsed = real(tick_end - tick_start, dp)/real(ticks_per_second, dp)
    state = adam%state_copy()
    write (*, '(a,",",i0,",",i0,",",i0,",",i0,",",es24.16)') &
        "trainer_fit_adam", state%fit_calls, state%optimizer_iterations, &
        state%line_search_evaluations, state%curvature_updates, elapsed

contains

    subroutine quadratic(x, value, gradient, status)
        real(dp), intent(in) :: x(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status

        if (size(x) /= 2 .or. size(gradient) /= 2) then
            call status_set(status, 1, "quadratic: shape")
            return
        end if
        value = (x(1) - 1.5_dp)**2 + 2.0_dp*(x(2) + 0.5_dp)**2
        gradient = [2.0_dp*(x(1) - 1.5_dp), 4.0_dp*(x(2) + 0.5_dp)]
        call status_set(status, FORTNUM_OK, "")
    end subroutine quadratic

    subroutine stop_after_first_step(step, value, gradient_norm, stop, status)
        integer, intent(in) :: step
        real(dp), intent(in) :: value, gradient_norm
        logical, intent(out) :: stop
        type(fortnum_status_t), intent(out) :: status

        stop = step >= 1
        call status_set(status, FORTNUM_OK, "")
    end subroutine stop_after_first_step

end program fortml_bench_trainer_fit_diagnostics
