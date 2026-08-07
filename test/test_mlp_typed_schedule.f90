program test_mlp_typed_schedule
    !! Independent oracle for typed schedule training and checkpoint replay.
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_DOMAIN_ERROR
    use fortml_mlp, only: mlp_t, MLP_LINEAR
    use fortml_mlp_schedules, only: mlp_learning_rate_schedule_t, &
        make_mlp_schedule_warmup_cosine
    use fortml_mlp_training, only: mlp_training_options_t, mlp_training_state_t, &
        mlp_training_checkpoint_t, mlp_train, MLP_OPTIMIZER_SGD
    use fortml_mlp_checkpoint, only: mlp_checkpoint_save, mlp_checkpoint_load
    implicit none

    type(mlp_t) :: full_model, split_model, loaded_model
    type(mlp_training_options_t) :: full_options, split_options, bad_options
    type(mlp_training_state_t) :: full_state, split_state
    type(mlp_training_checkpoint_t) :: full_checkpoint, split_checkpoint, loaded_checkpoint
    type(mlp_learning_rate_schedule_t) :: schedule, bad_schedule
    type(fortnum_status_t) :: status
    real(dp) :: x(5, 1), target(5, 1)
    real(dp) :: expected_rate, full_theta(2), split_theta(2)
    character(*), parameter :: path = "test_mlp_typed_schedule_checkpoint.txt"
    integer :: epoch, failures

    failures = 0
    x(:, 1) = [-2.0_dp, -1.0_dp, 0.0_dp, 1.0_dp, 2.0_dp]
    target(:, 1) = 0.7_dp*x(:, 1) - 0.2_dp
    schedule = make_mlp_schedule_warmup_cosine(2, 6, 0.2_dp)

    call full_model%initialize([1, 1], status, output_activation=MLP_LINEAR)
    call split_model%initialize([1, 1], status, output_activation=MLP_LINEAR)
    call loaded_model%initialize([1, 1], status, output_activation=MLP_LINEAR)
    call full_model%set_parameters([0.15_dp, -0.1_dp], status)
    call split_model%set_parameters([0.15_dp, -0.1_dp], status)
    call loaded_model%set_parameters([0.15_dp, -0.1_dp], status)

    full_options%max_epochs = 6
    full_options%optimizer = MLP_OPTIMIZER_SGD
    full_options%learning_rate = 0.12_dp
    full_options%tolerance = 0.0_dp
    full_options%restore_best = .false.
    full_options%use_typed_schedule = .true.
    full_options%typed_schedule = schedule
    split_options = full_options
    split_options%max_epochs = 3

    call mlp_train(full_model, x, target, status, full_options, full_state, &
        checkpoint=full_checkpoint)
    call check(status_ok(status) .and. full_checkpoint%valid(), &
        "typed schedule full fit", failures)
    call check(full_checkpoint%has_typed_schedule .and. &
        full_checkpoint%typed_schedule%kind == schedule%kind, &
        "typed schedule checkpoint metadata", failures)

    do epoch = 1, full_state%epochs
        call schedule%rate(epoch, full_options%learning_rate, expected_rate, status)
        call check(status_ok(status), "typed schedule oracle rate", failures)
        call check(abs(full_state%learning_rate_history(epoch)-expected_rate) < 2.0e-15_dp, &
            "typed schedule learning-rate history", failures)
    end do

    call mlp_train(split_model, x, target, status, split_options, split_state, &
        checkpoint=split_checkpoint)
    call check(status_ok(status) .and. split_checkpoint%valid(), &
        "typed schedule split fit", failures)
    call mlp_checkpoint_save(split_checkpoint, path, status)
    call check(status_ok(status), "typed schedule checkpoint save", failures)
    call mlp_checkpoint_load(loaded_checkpoint, path, status)
    call check(status_ok(status) .and. loaded_checkpoint%valid(), &
        "typed schedule checkpoint load", failures)
    call check(loaded_checkpoint%has_typed_schedule .and. &
        loaded_checkpoint%typed_schedule%total_updates == schedule%total_updates, &
        "typed schedule serialized fields", failures)

    call mlp_train(split_model, x, target, status, full_options, split_state, &
        checkpoint=split_checkpoint)
    call check(status_ok(status), "typed schedule native resume", failures)
    call mlp_train(loaded_model, x, target, status, full_options, split_state, &
        checkpoint=loaded_checkpoint)
    call check(status_ok(status), "typed schedule serialized resume", failures)
    full_theta = full_model%parameters()
    split_theta = split_model%parameters()
    call check(maxval(abs(full_theta-split_theta)) < 2.0e-14_dp, &
        "typed schedule uninterrupted/native replay", failures)
    call check(maxval(abs(full_theta-loaded_model%parameters())) < 2.0e-14_dp, &
        "typed schedule uninterrupted/serialized replay", failures)
    call check(maxval(abs(full_state%learning_rate_history-split_state%learning_rate_history)) &
        < 2.0e-14_dp, "typed schedule replay history", failures)

    bad_schedule = schedule
    bad_schedule%kind = 99
    bad_options = full_options
    bad_options%typed_schedule = bad_schedule
    call mlp_train(full_model, x, target, status, bad_options)
    call check(status%code == FORTNUM_DOMAIN_ERROR, "invalid typed schedule refusal", failures)

    bad_options = full_options
    bad_options%learning_rate_schedule => deterministic_decay
    call mlp_train(full_model, x, target, status, bad_options)
    call check(status%code == FORTNUM_DOMAIN_ERROR, "typed/callback conflict refusal", failures)

    call remove_file(path)
    if (failures > 0) error stop 1
    write (*, '(a)') "PASS MLP typed schedule independent behavioral oracles"

contains

    subroutine deterministic_decay(epoch, update, base_rate, rate)
        integer, intent(in) :: epoch, update
        real(dp), intent(in) :: base_rate
        real(dp), intent(out) :: rate

        rate = base_rate/(1.0_dp + 0.01_dp*real(epoch+update, dp))
    end subroutine deterministic_decay

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: description
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
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

end program test_mlp_typed_schedule
