program test_trainer
    !! Independent quadratic oracle for the model-agnostic trainer core.
    use, intrinsic :: iso_fortran_env, only: real64, error_unit
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_ok, status_set, FORTNUM_OK
    use fortopt_objective, only: objective_t
    use fortml_trainer, only: trainer_t, trainer_options_t, trainer_state_t, &
        FORTML_TRAIN_SGD, FORTML_TRAIN_ADAM, FORTML_TRAIN_LBFGSB
    implicit none

    type(objective_t) :: objective
    type(trainer_t) :: trainer, clone
    type(trainer_t) :: baseline, checkpointed, resumed
    type(trainer_options_t) :: options
    type(trainer_state_t) :: state
    type(fortnum_status_t) :: status
    real(dp), allocatable :: parameters(:)
    real(dp), allocatable :: before_load(:)
    type(trainer_state_t) :: baseline_state, resumed_state
    character(len=*), parameter :: checkpoint_path = "trainer_checkpoint_test.txt"
    character(len=*), parameter :: truncated_path = "trainer_checkpoint_truncated.txt"
    integer :: failures, i

    failures = 0
    call objective%initialize(2, quadratic_objective, status)
    call check(status_ok(status), "quadratic objective initialization", failures)

    options%optimizer = FORTML_TRAIN_SGD
    options%learning_rate = 0.1_dp
    options%max_steps = 4
    options%tolerance = 0.0_dp
    options%step_tolerance = 0.0_dp
    options%objective_tolerance = 0.0_dp
    options%ema_decay = 0.5_dp
    call trainer%initialize(objective, [0.0_dp, 1.0_dp], status, options)
    call check(status_ok(status), "SGD trainer initialization", failures)
    call trainer%step(status)
    parameters = trainer%parameters()
    state = trainer%state_copy()
    call check(status_ok(status) .and. maxval(abs(parameters - [0.3_dp, 0.4_dp])) < 1.0e-14_dp, &
        "SGD first update matches independent recurrence", failures)
    call check(maxval(abs(state%ema_parameters - [0.15_dp, 0.7_dp])) < 1.0e-14_dp .and. &
        state%steps == 1 .and. state%history_length == 2, &
        "EMA, counters, and history are explicit", failures)

    call trainer%clone(clone, status)
    call check(status_ok(status) .and. clone%initialized(), &
        "trainer clone preserves initialized state", failures)
    call trainer%step(status)
    call clone%step(status)
    call check(maxval(abs(trainer%parameters() - clone%parameters())) < 1.0e-14_dp, &
        "cloned optimizer state resumes identically", failures)

    options = trainer_options_t()
    options%optimizer = FORTML_TRAIN_ADAM
    options%learning_rate = 0.05_dp
    options%max_steps = 1
    options%gradient_clip_norm = 1.0_dp
    call trainer%initialize(objective, [0.0_dp, 1.0_dp], status, options)
    call trainer%step(status)
    state = trainer%state_copy()
    call check(status_ok(status) .and. state%clipped_steps == 1 .and. &
        state%final_value < state%initial_value, &
        "Adam and gradient clipping decrease the objective", failures)

    options = trainer_options_t()
    options%optimizer = FORTML_TRAIN_LBFGSB
    options%use_bounds = .true.
    options%lower = [0.0_dp, -1.0_dp]
    options%upper = [2.0_dp, 1.0_dp]
    options%lbfgsb%max_iterations = 100
    options%lbfgsb%gradient_tolerance = 1.0e-10_dp
    call trainer%initialize(objective, [0.0_dp, 1.0_dp], status, options)
    call trainer%fit(status)
    parameters = trainer%parameters()
    state = trainer%state_copy()
    call check(status_ok(status) .and. state%converged .and. &
        maxval(abs(parameters - [1.5_dp, -0.5_dp])) < 2.0e-7_dp, &
        "bounded L-BFGS-B reaches analytic minimum", failures)

    options = trainer_options_t()
    options%use_bounds = .true.
    options%lower = [0.0_dp]
    options%upper = [1.0_dp]
    call trainer%initialize(objective, [0.0_dp, 1.0_dp], status, options)
    call check(.not. status_ok(status), "inconsistent bound shape refusal", failures)

    ! A file round trip must preserve the optimizer recurrence, not only the
    ! current parameters.  The uninterrupted and resumed trajectories are an
    ! independent behavioral oracle for the serialized Adam moments/history.
    options = trainer_options_t()
    options%optimizer = FORTML_TRAIN_ADAM
    options%learning_rate = 0.05_dp
    options%max_steps = 8
    options%tolerance = 0.0_dp
    options%step_tolerance = 0.0_dp
    options%objective_tolerance = 0.0_dp
    call baseline%initialize(objective, [0.0_dp, 1.0_dp], status, options)
    call checkpointed%initialize(objective, [0.0_dp, 1.0_dp], status, options)
    call resumed%initialize(objective, [0.0_dp, 1.0_dp], status, options)
    do i = 1, 2
        call checkpointed%step(status)
        call baseline%step(status)
    end do
    call checkpointed%save_checkpoint(checkpoint_path, status)
    call check(status_ok(status), "trainer checkpoint save", failures)
    call resumed%load_checkpoint(checkpoint_path, status)
    call check(status_ok(status), "trainer checkpoint load", failures)
    do i = 1, 3
        call baseline%step(status)
        call resumed%step(status)
    end do
    baseline_state = baseline%state_copy()
    resumed_state = resumed%state_copy()
    call check(maxval(abs(baseline%parameters() - resumed%parameters())) < 1.0e-14_dp .and. &
        maxval(abs(baseline_state%value_history(:baseline_state%history_length) - &
        resumed_state%value_history(:resumed_state%history_length))) < 1.0e-14_dp .and. &
        maxval(abs(baseline_state%gradient_norm_history(:baseline_state%history_length) - &
        resumed_state%gradient_norm_history(:resumed_state%history_length))) < 1.0e-14_dp, &
        "save/load continuation matches uninterrupted Adam trajectory", failures)

    ! Extra records and truncation are rejected transactionally; the loaded
    ! destination remains unchanged after each refusal.
    before_load = resumed%parameters()
    open (unit=91, file=checkpoint_path, status="old", position="append", action="write")
    write (91, '(a)') "extra_record 1"
    close (91)
    call resumed%load_checkpoint(checkpoint_path, status)
    call check(.not. status_ok(status) .and. maxval(abs(before_load - resumed%parameters())) < 1.0e-14_dp, &
        "extra checkpoint record refusal is transactional", failures)
    open (unit=91, file=truncated_path, status="replace", action="write")
    write (91, '(a)') "FORTML_TRAINER_CHECKPOINT_TEXT"
    close (91)
    call resumed%load_checkpoint(truncated_path, status)
    call check(.not. status_ok(status) .and. maxval(abs(before_load - resumed%parameters())) < 1.0e-14_dp, &
        "truncated checkpoint refusal is transactional", failures)
    open (unit=91, file=checkpoint_path, status="old")
    close (91, status="delete")
    open (unit=91, file=truncated_path, status="old")
    close (91, status="delete")

    if (failures /= 0) then
        write (error_unit, '(a,i0)') "FAIL trainer cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS model-agnostic trainer independent quadratic oracle"

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

    subroutine check(condition, description, failure_count)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failure_count

        if (.not. condition) then
            failure_count = failure_count + 1
            write (error_unit, '(a)') "  FAIL [trainer] "//description
        end if
    end subroutine check

end program test_trainer
