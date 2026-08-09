program test_trainer_schedule
    !! Independent recurrence oracle for generic trainer-owned schedules.
    use, intrinsic :: iso_fortran_env, only: error_unit
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_ok, status_set, FORTNUM_OK
    use fortml_device, only: FORTML_DEVICE_CUDA
    use fortopt_objective, only: objective_t
    use fortml_mlp_schedules, only: make_mlp_schedule_one_cycle, &
        make_mlp_schedule_plateau
    use fortml_trainer, only: trainer_t, trainer_options_t, trainer_state_t, &
        FORTML_TRAIN_SGD
    implicit none

    type(objective_t) :: objective
    type(trainer_t) :: full, split, resumed
    type(trainer_options_t) :: options, invalid_options
    type(trainer_state_t) :: full_state, split_state, resumed_state
    type(fortnum_status_t) :: status
    real(dp) :: expected(2), gradient(2), rates(4)
    integer :: failures, i
    character(*), parameter :: checkpoint_path = "trainer_schedule_checkpoint.txt"

    failures = 0
    call objective%initialize(2, quadratic_objective, status)
    call check(status_ok(status), "quadratic objective initialization", failures)

    options%optimizer = FORTML_TRAIN_SGD
    options%learning_rate = 0.1_dp
    options%max_steps = 4
    options%tolerance = 0.0_dp
    options%step_tolerance = 0.0_dp
    options%objective_tolerance = 0.0_dp
    options%use_learning_rate_schedule = .true.
    options%learning_rate_schedule = make_mlp_schedule_one_cycle(2, 4, 2.0_dp, 0.25_dp)
    call full%initialize(objective, [0.0_dp, 1.0_dp], status, options)
    call check(status_ok(status), "one-cycle trainer initialization", failures)

    ! Independent one-cycle fractions: 1.5, 2.0, 1.125, and 0.25.
    rates = [0.15_dp, 0.2_dp, 0.1125_dp, 0.025_dp]
    expected = [0.0_dp, 1.0_dp]
    do i = 1, 4
        gradient = [2.0_dp*(expected(1) - 1.5_dp), &
            4.0_dp*(expected(2) + 0.5_dp)]
        expected = expected - rates(i)*gradient
        call full%step(status)
        call check(status_ok(status), "scheduled SGD update", failures)
    end do
    full_state = full%state_copy()
    call check(maxval(abs(full%parameters() - expected)) < 2.0e-14_dp, &
        "one-cycle parameters match independent SGD recurrence", failures)
    call check(maxval(abs(full_state%learning_rate_history(2:5) - rates)) < &
        2.0e-14_dp .and. abs(full_state%last_learning_rate - rates(4)) < 2.0e-14_dp, &
        "trainer records every scheduled optimizer rate", failures)

    call split%initialize(objective, [0.0_dp, 1.0_dp], status, options)
    call resumed%initialize(objective, [0.0_dp, 1.0_dp], status, options)
    do i = 1, 2
        call split%step(status)
    end do
    split_state = split%state_copy()
    call split%save_checkpoint(checkpoint_path, status)
    call check(status_ok(status), "scheduled checkpoint save", failures)
    call resumed%load_checkpoint(checkpoint_path, status)
    call check(status_ok(status), "scheduled checkpoint load", failures)
    do i = 1, 2
        call split%step(status)
        call resumed%step(status)
    end do
    split_state = split%state_copy()
    resumed_state = resumed%state_copy()
    call check(maxval(abs(split%parameters() - resumed%parameters())) < 2.0e-14_dp .and. &
        maxval(abs(split_state%learning_rate_history - resumed_state%learning_rate_history)) < &
            2.0e-14_dp, &
        "checkpoint continuation preserves scheduled trajectory", failures)
    open (unit=91, file=checkpoint_path, status="old")
    close (91, status="delete")

    invalid_options = options
    invalid_options%learning_rate_schedule = make_mlp_schedule_plateau(2, 0.0_dp, 0.5_dp)
    call resumed%initialize(objective, [0.0_dp, 1.0_dp], status, invalid_options)
    call check(.not. status_ok(status), "generic trainer refuses metric-aware plateau schedule", failures)
    call check(.not. options%learning_rate_schedule%device_supported(FORTML_DEVICE_CUDA), &
        "schedule reports resident CUDA boundary", failures)

    if (failures /= 0) then
        write (error_unit, '(a,i0)') "FAIL trainer schedule cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS trainer schedule independent recurrence oracle"

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
            write (error_unit, '(a)') "FAIL: "//description
            failure_count = failure_count + 1
        end if
    end subroutine check

end program test_trainer_schedule
