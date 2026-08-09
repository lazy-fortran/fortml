program fortml_bench_trainer_value_clipping
    !! Release-app fixture for the generic trainer value-clipping contract.
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_ok, status_set, FORTNUM_OK
    use fortopt_objective, only: objective_t
    use fortml_trainer, only: trainer_t, trainer_options_t, trainer_state_t, &
        FORTML_TRAIN_SGD
    implicit none

    type(objective_t) :: objective
    type(trainer_t) :: trainer, resumed
    type(trainer_options_t) :: options
    type(trainer_state_t) :: state
    type(fortnum_status_t) :: status
    real(dp), allocatable :: parameters(:)
    character(len=*), parameter :: checkpoint = "trainer_value_clip_bench.txt"

    call objective%initialize(2, quadratic, status)
    if (.not. status_ok(status)) error stop "objective initialization failed"
    options%optimizer = FORTML_TRAIN_SGD
    options%learning_rate = 0.1_dp
    options%max_steps = 2
    options%gradient_clip_value = 1.0_dp
    options%tolerance = 0.0_dp
    options%step_tolerance = 0.0_dp
    options%objective_tolerance = 0.0_dp
    call trainer%initialize(objective, [0.0_dp, 1.0_dp], status, options)
    if (.not. status_ok(status)) error stop "trainer initialization failed"
    call trainer%step(status)
    if (.not. status_ok(status)) error stop "trainer step failed"
    parameters = trainer%parameters()
    state = trainer%state_copy()
    call trainer%save_checkpoint(checkpoint, status)
    if (.not. status_ok(status)) error stop "checkpoint save failed"
    call resumed%initialize(objective, [0.0_dp, 1.0_dp], status, options)
    if (.not. status_ok(status)) error stop "resume initialization failed"
    call resumed%load_checkpoint(checkpoint, status)
    if (.not. status_ok(status)) error stop "checkpoint load failed"
    open (unit=91, file=checkpoint, status="old")
    close (91, status="delete")
    write (*, '(a,",",i0,",",i0,",",es24.16e3,",",es24.16e3,",",i0)') &
        "trainer_value_clipping", state%steps, state%value_clipped_steps, &
        parameters(1), parameters(2), merge(1, 0, all(abs(resumed%parameters() - parameters) < 1.0e-14_dp))

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

end program fortml_bench_trainer_value_clipping
