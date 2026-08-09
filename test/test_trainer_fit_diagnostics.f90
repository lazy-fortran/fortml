program test_trainer_fit_diagnostics
    !! Independent behavioral oracle for generic trainer fit diagnostics.
    !!
    !! The quadratic objective is evaluated directly in this fixture.  The
    !! checks therefore verify FortOpt's result counters and the persistent
    !! streaming-fit counters without treating a source snapshot as proof.
    use, intrinsic :: iso_fortran_env, only: error_unit
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_ok, status_set, FORTNUM_OK
    use fortopt_objective, only: objective_t
    use fortml_trainer, only: trainer_t, trainer_options_t, trainer_state_t, &
        FORTML_TRAIN_ADAM, FORTML_TRAIN_LBFGSB, &
        FORTML_TRAINER_CHECKPOINT_SCHEMA_VERSION
    implicit none

    type(objective_t) :: objective
    type(trainer_t) :: lbfgsb, adam, resumed
    type(trainer_options_t) :: options
    type(trainer_state_t) :: state, resumed_state
    type(fortnum_status_t) :: status
    integer :: failures
    character(len=*), parameter :: checkpoint = "trainer_fit_diagnostics_test.txt"

    failures = 0
    call objective%initialize(2, quadratic, status)
    call check(status_ok(status), "quadratic objective initialization", failures)

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
    call check(status_ok(status), "L-BFGS-B trainer initialization", failures)
    call lbfgsb%fit(status)
    state = lbfgsb%state_copy()
    call check(status_ok(status) .and. state%fit_calls == 1, &
        "bounded fit increments the successful fit counter", failures)
    call check(state%optimizer_iterations == state%steps .and. &
        state%optimizer_iterations > 0, &
        "L-BFGS-B iteration diagnostics match its convergence state", failures)
    call check(state%line_search_evaluations > 0 .and. state%curvature_updates > 0, &
        "L-BFGS-B exposes line-search and curvature diagnostics", failures)
    call check(maxval(abs(lbfgsb%parameters() - [1.5_dp, -0.5_dp])) < 2.0e-7_dp, &
        "bounded fit reaches the independent quadratic optimum", failures)
    call check(FORTML_TRAINER_CHECKPOINT_SCHEMA_VERSION == 8, &
        "diagnostics use the versioned schema-8 state contract", failures)

    options = trainer_options_t()
    options%optimizer = FORTML_TRAIN_ADAM
    options%learning_rate = 0.05_dp
    options%max_steps = 4
    options%tolerance = 0.0_dp
    options%step_tolerance = 0.0_dp
    options%objective_tolerance = 0.0_dp
    options%callback => stop_after_first_step
    call adam%initialize(objective, [0.0_dp, 1.0_dp], status, options)
    call check(status_ok(status), "Adam trainer initialization", failures)
    call adam%fit(status)
    state = adam%state_copy()
    call check(status_ok(status) .and. state%fit_calls == 1 .and. &
        state%optimizer_iterations == 1 .and. state%line_search_evaluations == 0 .and. &
        state%curvature_updates == 0, &
        "streaming fit reports one accepted update and no L-BFGS-B counters", failures)

    call adam%save_checkpoint(checkpoint, status)
    call check(status_ok(status), "schema-8 diagnostics checkpoint save", failures)
    call resumed%initialize(objective, [0.0_dp, 1.0_dp], status, options)
    call resumed%load_checkpoint(checkpoint, status)
    resumed_state = resumed%state_copy()
    call check(status_ok(status) .and. resumed_state%fit_calls == state%fit_calls .and. &
        resumed_state%optimizer_iterations == state%optimizer_iterations .and. &
        resumed_state%line_search_evaluations == state%line_search_evaluations .and. &
        resumed_state%curvature_updates == state%curvature_updates, &
        "schema-8 checkpoint preserves fit diagnostics", failures)
    open (unit=91, file=checkpoint, status="old")
    close (91, status="delete")

    if (failures /= 0) then
        write (error_unit, '(a,i0)') "FAIL trainer fit diagnostics cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS trainer fit diagnostics independent quadratic oracle"

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

    subroutine check(condition, description, failure_count)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failure_count

        if (.not. condition) then
            failure_count = failure_count + 1
            write (error_unit, '(a)') "  FAIL [trainer fit diagnostics] "//description
        end if
    end subroutine check

end program test_trainer_fit_diagnostics
