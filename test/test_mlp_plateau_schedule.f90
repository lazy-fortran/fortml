program test_mlp_plateau_schedule
    !! Independent oracle for metric-aware plateau schedules and persistence.
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_DOMAIN_ERROR, &
        FORTNUM_NOT_IMPLEMENTED
    use fortml_mlp, only: mlp_t, MLP_LINEAR
    use fortml_mlp_schedules, only: mlp_learning_rate_schedule_t, &
        make_mlp_schedule_constant, make_mlp_schedule_plateau, &
        MLP_SCHEDULE_METRIC_MINIMIZE, MLP_SCHEDULE_METRIC_MAXIMIZE, &
        MLP_SCHEDULE_PLATEAU
    use fortml_mlp_training, only: mlp_training_options_t, mlp_training_checkpoint_t, &
        mlp_training_state_t, mlp_train, MLP_OPTIMIZER_SGD
    use fortml_mlp_checkpoint, only: mlp_checkpoint_save, mlp_checkpoint_load
    implicit none

    type(mlp_learning_rate_schedule_t) :: schedule, invalid, constant
    type(fortnum_status_t) :: status
    real(dp) :: rate, best, next_best, dbase, dmetric, dbest, ddelta, dfactor
    real(dp) :: plus, minus, base
    integer :: bad, reductions, next_bad, next_reductions, failures
    logical :: improved, reduced
    type(mlp_t) :: model
    type(mlp_training_options_t) :: options
    type(mlp_training_state_t) :: state
    type(mlp_training_checkpoint_t) :: checkpoint, loaded
    real(dp) :: x(3, 1), target(3, 1)
    character(*), parameter :: path = "test_mlp_plateau_schedule_checkpoint.txt"

    failures = 0
    base = 0.2_dp
    schedule = make_mlp_schedule_plateau(2, 0.05_dp, 0.5_dp, &
        MLP_SCHEDULE_METRIC_MINIMIZE)
    call check(schedule%kind == MLP_SCHEDULE_PLATEAU .and. schedule%valid(), &
        "plateau metadata", failures)

    best = 1.0_dp
    bad = 0
    reductions = 0
    call schedule%rate_with_metric_derivatives(1, base, 1.1_dp, best, bad, reductions, &
        rate, next_best, next_bad, next_reductions, improved, reduced, dbase, dmetric, &
        dbest, ddelta, dfactor, status)
    call check(status_ok(status) .and. .not. improved .and. .not. reduced .and. &
        next_bad == 1 .and. next_reductions == 0 .and. abs(rate-0.2_dp) < 1.0e-15_dp, &
        "plateau first non-improving observation", failures)
    bad = next_bad
    reductions = next_reductions
    call schedule%rate_with_metric(2, base, 1.0_dp, best, bad, reductions, rate, next_best, &
        next_bad, next_reductions, improved, reduced, status)
    call check(status_ok(status) .and. .not. improved .and. reduced .and. &
        next_bad == 0 .and. next_reductions == 1 .and. abs(rate-0.1_dp) < 1.0e-15_dp, &
        "plateau patience reduction", failures)
    bad = next_bad
    reductions = next_reductions
    call schedule%rate_with_metric(3, base, 0.99_dp, best, bad, reductions, rate, next_best, &
        next_bad, next_reductions, improved, reduced, status)
    call check(status_ok(status) .and. .not. improved .and. .not. reduced .and. &
        next_bad == 1 .and. next_reductions == 1 .and. abs(rate-0.1_dp) < 1.0e-15_dp, &
        "plateau state persists after reduction", failures)
    bad = next_bad
    reductions = next_reductions
    call schedule%rate_with_metric(4, base, 0.90_dp, best, bad, reductions, rate, next_best, &
        next_bad, next_reductions, improved, reduced, status)
    best = next_best
    call check(status_ok(status) .and. improved .and. .not. reduced .and. &
        next_bad == 0 .and. next_reductions == 1 .and. abs(best-0.9_dp) < 1.0e-15_dp, &
        "plateau metric improvement resets state", failures)

    ! The active branch is smooth in base rate and factor.  The other
    ! products are the documented zero derivatives of the comparison branch.
    call schedule%rate_with_metric_derivatives(5, base, 0.90_dp, 0.90_dp, 0, 2, rate, &
        next_best, next_bad, next_reductions, improved, reduced, dbase, dmetric, dbest, &
        ddelta, dfactor, status)
    call check(status_ok(status) .and. abs(rate-base*0.25_dp) < 1.0e-15_dp .and. &
        abs(dbase-0.25_dp) < 1.0e-15_dp .and. abs(dmetric) < 1.0e-15_dp .and. &
        abs(dbest) < 1.0e-15_dp .and. abs(ddelta) < 1.0e-15_dp, &
        "plateau active-set derivatives", failures)
    schedule%plateau_factor = schedule%plateau_factor+1.0e-6_dp
    call schedule%rate_with_metric(5, base, 0.90_dp, 0.90_dp, 0, 2, plus, next_best, &
        next_bad, next_reductions, improved, reduced, status)
    schedule%plateau_factor = schedule%plateau_factor-2.0e-6_dp
    call schedule%rate_with_metric(5, base, 0.90_dp, 0.90_dp, 0, 2, minus, next_best, &
        next_bad, next_reductions, improved, reduced, status)
    schedule%plateau_factor = schedule%plateau_factor+1.0e-6_dp
    call check(abs(dfactor-(plus-minus)/(2.0e-6_dp)) < 2.0e-10_dp, &
        "plateau factor finite difference", failures)

    schedule = make_mlp_schedule_plateau(3, 0.0_dp, 0.25_dp, &
        MLP_SCHEDULE_METRIC_MAXIMIZE)
    call schedule%rate_with_metric(1, base, 0.8_dp, 0.7_dp, 0, 0, rate, next_best, &
        next_bad, next_reductions, improved, reduced, status)
    call check(status_ok(status) .and. improved .and. abs(next_best-0.8_dp) < 1.0e-15_dp, &
        "maximizing metric mode", failures)

    call schedule%rate(1, base, rate, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "ordinary rate rejects metric-aware schedule", failures)
    invalid = make_mlp_schedule_plateau(0, 0.0_dp, 0.5_dp)
    call check(.not. invalid%valid(), "invalid plateau patience refusal", failures)
    invalid = make_mlp_schedule_plateau(2, 0.0_dp, 1.0_dp)
    call check(.not. invalid%valid(), "invalid plateau factor refusal", failures)

    ! Persist the typed fields through the versioned checkpoint and verify
    ! malformed schedule fields are rejected before replacing the destination.
    x(:, 1) = [-1.0_dp, 0.0_dp, 1.0_dp]
    target(:, 1) = 2.0_dp*x(:, 1)
    call model%initialize([1, 1], status, output_activation=MLP_LINEAR)
    options%max_epochs = 1
    options%optimizer = MLP_OPTIMIZER_SGD
    options%learning_rate = 0.01_dp
    options%tolerance = 0.0_dp
    options%restore_best = .false.
    constant = make_mlp_schedule_constant()
    options%use_typed_schedule = .true.
    options%typed_schedule = constant
    call mlp_train(model, x, target, status, options, state, checkpoint=checkpoint)
    call check(status_ok(status), "checkpoint fixture training", failures)
    checkpoint%typed_schedule = make_mlp_schedule_plateau(2, 0.02_dp, 0.4_dp, &
        MLP_SCHEDULE_METRIC_MAXIMIZE)
    call mlp_checkpoint_save(checkpoint, path, status)
    call check(status_ok(status), "plateau checkpoint save", failures)
    call mlp_checkpoint_load(loaded, path, status)
    call check(status_ok(status) .and. loaded%typed_schedule%kind == MLP_SCHEDULE_PLATEAU .and. &
        loaded%typed_schedule%metric_mode == MLP_SCHEDULE_METRIC_MAXIMIZE .and. &
        loaded%typed_schedule%patience_updates == 2 .and. &
        abs(loaded%typed_schedule%min_delta-0.02_dp) < 1.0e-15_dp .and. &
        abs(loaded%typed_schedule%plateau_factor-0.4_dp) < 1.0e-15_dp, &
        "plateau checkpoint fields round trip", failures)
    checkpoint%typed_schedule%plateau_factor = 1.0_dp
    call mlp_checkpoint_save(checkpoint, path, status)
    call check(status%code == FORTNUM_DOMAIN_ERROR, &
        "invalid plateau checkpoint refusal", failures)
    options%typed_schedule = make_mlp_schedule_plateau(2, 0.02_dp, 0.4_dp)
    call mlp_train(model, x, target, status, options)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "trainer metric-aware adapter refusal", failures)

    call remove_file(path)
    if (failures > 0) error stop 1
    write (*, '(a)') "PASS MLP plateau schedule independent behavioral oracles"

contains

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: description
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures+1
            write (*, '(a)') "FAIL "//trim(description)
        end if
    end subroutine check

    subroutine remove_file(file_path)
        character(len=*), intent(in) :: file_path
        logical :: exists
        integer :: unit, ios

        inquire(file=file_path, exist=exists)
        if (.not. exists) return
        open(newunit=unit, file=file_path, status="old", iostat=ios)
        if (ios /= 0) return
        close(unit, status="delete", iostat=ios)
    end subroutine remove_file

end program test_mlp_plateau_schedule
