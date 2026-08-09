program test_trainer_partial_fit
    !! Independent diagonal-quadratic oracle for chunked trainer updates.
    use, intrinsic :: iso_fortran_env, only: error_unit
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_ok, status_set, &
        FORTNUM_OK, FORTNUM_NOT_IMPLEMENTED
    use fortopt_objective, only: objective_t
    use fortml_device, only: FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    use fortml_trainer, only: trainer_t, trainer_options_t, trainer_state_t, &
        FORTML_TRAIN_ADAM
    implicit none

    type(objective_t) :: objective
    type(trainer_t) :: uninterrupted, chunked, resumed
    type(trainer_options_t) :: options
    type(trainer_state_t) :: before, after_uninterrupted, after_chunked
    type(fortnum_status_t) :: status
    real(dp), allocatable :: before_parameters(:)
    integer :: failures

    failures = 0
    call objective%initialize(3, diagonal_quadratic, status)
    call check(status_ok(status), "objective initialization", failures)

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
    call resumed%initialize(objective, [0.0_dp, 1.0_dp, -1.0_dp], status, options)
    call check(status_ok(status), "trainer initialization", failures)

    call uninterrupted%partial_fit(6, status)
    call check(status_ok(status), "uninterrupted six-update chunk", failures)
    call chunked%partial_fit(2, status)
    call check(status_ok(status), "first two-update chunk", failures)
    call chunked%partial_fit(4, status)
    call check(status_ok(status), "second four-update chunk", failures)
    after_uninterrupted = uninterrupted%state_copy()
    after_chunked = chunked%state_copy()
    call check(maxval(abs(uninterrupted%parameters() - chunked%parameters())) < 1.0e-14_dp .and. &
        maxval(abs(after_uninterrupted%value_history - after_chunked%value_history)) < 1.0e-14_dp .and. &
        after_uninterrupted%steps == 6 .and. after_chunked%history_length == 7, &
        "chunked updates reproduce uninterrupted optimizer trajectory", failures)

    call chunked%save_checkpoint("trainer_partial_fit_checkpoint.txt", status)
    call check(status_ok(status), "partial-fit checkpoint save", failures)
    call resumed%load_checkpoint("trainer_partial_fit_checkpoint.txt", status)
    call check(status_ok(status), "partial-fit checkpoint load", failures)
    ! The checkpoint is at the declared budget, so loading it is still valid,
    ! while a further update is rejected without changing the state.
    before_parameters = resumed%parameters()
    before = resumed%state_copy()
    call resumed%partial_fit(1, status)
    after_chunked = resumed%state_copy()
    call check(.not. status_ok(status) .and. maxval(abs(before_parameters - &
        resumed%parameters())) < 1.0e-14_dp .and. after_chunked%steps == before%steps, &
        "budget overflow refusal is transactional", failures)

    call uninterrupted%initialize(objective, [0.0_dp, 1.0_dp, -1.0_dp], status, options)
    call uninterrupted%partial_fit(2, status)
    call check(status_ok(status), "checkpoint prefix chunk", failures)
    call uninterrupted%save_checkpoint("trainer_partial_fit_prefix.txt", status)
    call check(status_ok(status), "prefix checkpoint save", failures)
    call resumed%initialize(objective, [0.0_dp, 1.0_dp, -1.0_dp], status, options)
    call resumed%load_checkpoint("trainer_partial_fit_prefix.txt", status)
    call check(status_ok(status), "prefix checkpoint load", failures)
    call uninterrupted%partial_fit(4, status)
    call check(status_ok(status), "uninterrupted suffix chunk", failures)
    call resumed%partial_fit(4, status)
    call check(status_ok(status) .and. maxval(abs(uninterrupted%parameters() - &
        resumed%parameters())) < 1.0e-14_dp, &
        "checkpoint prefix plus chunk reproduces six-update trajectory", failures)

    before_parameters = resumed%parameters()
    call resumed%partial_fit_device(FORTML_DEVICE_CUDA, 1, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED .and. maxval(abs(&
        before_parameters - resumed%parameters())) < 1.0e-14_dp, &
        "CUDA partial-fit request is a typed refusal without host fallback", failures)
    call resumed%partial_fit_device(FORTML_DEVICE_CPU, 0, status)
    call check(.not. status_ok(status), "non-positive partial-fit count refusal", failures)
    call resumed%partial_fit_device(-99, 1, status)
    call check(.not. status_ok(status), "invalid device partial-fit refusal", failures)

    open (unit=90, file="trainer_partial_fit_checkpoint.txt", status="old")
    close (90, status="delete")
    open (unit=91, file="trainer_partial_fit_prefix.txt", status="old")
    close (91, status="delete")
    if (failures /= 0) then
        write (error_unit, '(a,i0)') "FAIL trainer partial-fit cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS trainer partial-fit independent quadratic oracle"

contains

    subroutine diagonal_quadratic(parameters, value, gradient, status)
        real(dp), intent(in) :: parameters(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), parameter :: target(3) = [1.5_dp, -0.5_dp, 0.25_dp]
        real(dp), parameter :: curvature(3) = [2.0_dp, 4.0_dp, 1.25_dp]

        if (size(parameters) /= 3 .or. size(gradient) /= 3) then
            call status_set(status, 1, "quadratic: invalid parameter shape")
            return
        end if
        gradient = curvature*(parameters - target)
        value = 0.5_dp*sum(curvature*(parameters - target)**2)
        call status_set(status, FORTNUM_OK, "")
    end subroutine diagonal_quadratic

    subroutine check(condition, message, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: message
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "FAIL: "//trim(message)
        end if
    end subroutine check

end program test_trainer_partial_fit
