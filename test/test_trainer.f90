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
    type(trainer_options_t) :: options
    type(trainer_state_t) :: state
    type(fortnum_status_t) :: status
    real(dp), allocatable :: parameters(:)
    integer :: failures

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
