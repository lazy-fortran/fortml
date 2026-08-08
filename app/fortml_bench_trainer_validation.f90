program fortml_bench_trainer_validation
    !! Release workload for generic-trainer validation and early stopping.
    use, intrinsic :: iso_fortran_env, only: int64
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_ok, status_set, FORTNUM_OK
    use fortopt_objective, only: objective_t
    use fortml_trainer, only: trainer_t, trainer_options_t, trainer_state_t, &
        FORTML_TRAIN_SGD
    implicit none

    type(objective_t) :: objective
    type(trainer_t) :: trainer, checkpointed, resumed, reference
    type(trainer_options_t) :: options, reference_options
    type(trainer_state_t) :: state, resumed_state
    type(fortnum_status_t) :: status
    real(dp), allocatable :: reference_parameters(:)
    real(dp) :: elapsed, restore_error, resume_error
    integer(int64) :: tick_start, tick_end, ticks_per_second
    character(len=*), parameter :: checkpoint_path = &
        "trainer_validation_release_checkpoint.txt"

    call objective%initialize(2, quadratic_objective, status)
    if (.not. status_ok(status)) error stop 1
    options%optimizer = FORTML_TRAIN_SGD
    options%learning_rate = 0.1_dp
    options%max_steps = 8
    options%tolerance = 0.0_dp
    options%step_tolerance = 0.0_dp
    options%objective_tolerance = 0.0_dp
    options%validation_patience = 2
    options%validation_min_delta = 0.0_dp
    options%validation_restore_best = .true.
    options%validation_callback => validation_curve

    call system_clock(tick_start, ticks_per_second)
    call trainer%initialize(objective, [0.0_dp, 1.0_dp], status, options)
    if (.not. status_ok(status)) error stop 1
    call trainer%fit(status)
    if (.not. status_ok(status)) error stop 1
    call system_clock(tick_end)
    elapsed = real(tick_end - tick_start, dp)/real(ticks_per_second, dp)
    state = trainer%state_copy()

    reference_options = trainer_options_t()
    reference_options%optimizer = FORTML_TRAIN_SGD
    reference_options%learning_rate = 0.1_dp
    reference_options%max_steps = 8
    reference_options%tolerance = 0.0_dp
    reference_options%step_tolerance = 0.0_dp
    reference_options%objective_tolerance = 0.0_dp
    call reference%initialize(objective, [0.0_dp, 1.0_dp], status, reference_options)
    call reference%step(status)
    if (.not. status_ok(status)) error stop 1
    reference_parameters = reference%parameters()
    restore_error = maxval(abs(trainer%parameters() - reference_parameters))

    call checkpointed%initialize(objective, [0.0_dp, 1.0_dp], status, options)
    call checkpointed%step(status)
    call checkpointed%save_checkpoint(checkpoint_path, status)
    if (.not. status_ok(status)) error stop 1
    call resumed%initialize(objective, [0.0_dp, 1.0_dp], status, options)
    call resumed%load_checkpoint(checkpoint_path, status)
    if (.not. status_ok(status)) error stop 1
    call checkpointed%step(status)
    call resumed%step(status)
    call checkpointed%step(status)
    call resumed%step(status)
    if (.not. status_ok(status)) error stop 1
    resume_error = maxval(abs(checkpointed%parameters() - resumed%parameters()))
    resumed_state = resumed%state_copy()

    write (*, '(a,",pass,stopped_step,",i0,",",es24.16)') &
        "trainer_validation", state%steps, elapsed
    write (*, '(a,",pass,best_step,",i0,",",es24.16)') &
        "trainer_validation", state%validation_best_step, elapsed
    write (*, '(a,",pass,best_validation_value,",es24.16,",",es24.16)') &
        "trainer_validation", state%best_validation_value, elapsed
    write (*, '(a,",pass,restore_best_max_abs_error,",es24.16,",",es24.16)') &
        "trainer_validation", restore_error, elapsed
    write (*, '(a,",pass,resume_max_abs_error,",es24.16,",",es24.16)') &
        "trainer_validation", resume_error, elapsed
    write (*, '(a,",pass,resume_best_step,",i0,",",es24.16)') &
        "trainer_validation", resumed_state%validation_best_step, elapsed
    open (unit=91, file=checkpoint_path, status="old")
    close (91, status="delete")

contains

    subroutine quadratic_objective(x, value, gradient, status)
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
    end subroutine quadratic_objective

    subroutine validation_curve(step, parameters, validation_value, status)
        integer, intent(in) :: step
        real(dp), intent(in) :: parameters(:)
        real(dp), intent(out) :: validation_value
        type(fortnum_status_t), intent(out) :: status

        if (size(parameters) /= 2) then
            call status_set(status, 1, "validation curve: parameter shape")
            return
        end if
        select case (step)
        case (0)
            validation_value = 0.4_dp
        case (1)
            validation_value = 0.2_dp
        case (2)
            validation_value = 0.25_dp
        case default
            validation_value = 0.3_dp
        end select
        call status_set(status, FORTNUM_OK, "")
    end subroutine validation_curve

end program fortml_bench_trainer_validation
