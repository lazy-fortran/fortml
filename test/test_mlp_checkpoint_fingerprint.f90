program test_mlp_checkpoint_fingerprint
    !! Independent behavioral oracle for deterministic checkpoint identity.
    use, intrinsic :: iso_fortran_env, only: int64
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_ok
    use fortml_device, only: FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    use fortml_mlp, only: mlp_t, MLP_LINEAR
    use fortml_mlp_training, only: mlp_training_options_t, &
        mlp_training_checkpoint_t, mlp_train, MLP_OPTIMIZER_ADAM
    use fortml_mlp_checkpoint, only: mlp_checkpoint_save, mlp_checkpoint_load, &
        mlp_checkpoint_fingerprint, mlp_checkpoint_require_device, &
        mlp_checkpoint_device_supported
    implicit none

    integer :: failures

    failures = 0
    call test_round_trip_and_mutation(failures)
    call test_device_contract(failures)
    if (failures > 0) then
        write (*, '(a,i0)') "FAIL MLP checkpoint fingerprint cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS MLP checkpoint fingerprint behavioral oracles"

contains

    subroutine test_round_trip_and_mutation(failures)
        integer, intent(inout) :: failures
        type(mlp_t) :: model
        type(mlp_training_options_t) :: options
        type(mlp_training_checkpoint_t) :: checkpoint, loaded, mutated
        type(fortnum_status_t) :: status
        real(dp) :: x(4, 1), target(4, 1)
        integer(int64) :: original_fingerprint, loaded_fingerprint
        integer(int64) :: state_fingerprint, metadata_fingerprint
        character(*), parameter :: path = "test_mlp_checkpoint_fingerprint.txt"

        x(:, 1) = [-2.0_dp, -1.0_dp, 1.0_dp, 2.0_dp]
        target(:, 1) = 0.4_dp*x(:, 1) - 0.15_dp
        call model%initialize([1, 1], status, output_activation=MLP_LINEAR)
        call check(status_ok(status), "fingerprint model initialize", failures)
        call model%set_parameters([0.2_dp, -0.1_dp], status)
        call check(status_ok(status), "fingerprint model parameters", failures)

        options%max_epochs = 3
        options%batch_size = 2
        options%shuffle = .true.
        options%shuffle_seed = 23
        options%learning_rate = 0.04_dp
        options%l2 = 0.001_dp
        options%optimizer = MLP_OPTIMIZER_ADAM
        options%tolerance = 0.0_dp
        call mlp_train(model, x, target, status, options, checkpoint=checkpoint)
        call check(status_ok(status) .and. checkpoint%valid(), &
            "fingerprint checkpoint capture", failures)

        original_fingerprint = mlp_checkpoint_fingerprint(checkpoint)
        call check(original_fingerprint /= 0_int64, &
            "valid checkpoint has nonzero fingerprint", failures)
        call mlp_checkpoint_save(checkpoint, path, status)
        call check(status_ok(status), "fingerprint checkpoint save", failures)
        call mlp_checkpoint_load(loaded, path, status)
        call check(status_ok(status) .and. loaded%valid(), &
            "fingerprint checkpoint load", failures)
        loaded_fingerprint = mlp_checkpoint_fingerprint(loaded)
        call check(original_fingerprint == loaded_fingerprint, &
            "formatted round-trip preserves fingerprint", failures)

        mutated = checkpoint
        mutated%second_moment(1) = mutated%second_moment(1) + 1.0e-3_dp
        state_fingerprint = mlp_checkpoint_fingerprint(mutated)
        call check(mutated%valid() .and. state_fingerprint /= original_fingerprint, &
            "optimizer-state mutation changes fingerprint", failures)

        mutated = checkpoint
        mutated%learning_rate = 1.1_dp*mutated%learning_rate
        metadata_fingerprint = mlp_checkpoint_fingerprint(mutated)
        call check(mutated%valid() .and. metadata_fingerprint /= original_fingerprint, &
            "optimizer-metadata mutation changes fingerprint", failures)

        call mutated%clear()
        call check(mlp_checkpoint_fingerprint(mutated) == 0_int64, &
            "invalid checkpoint has zero fingerprint", failures)
        call remove_file(path)
    end subroutine test_round_trip_and_mutation

    subroutine test_device_contract(failures)
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status

        call mlp_checkpoint_require_device(FORTML_DEVICE_CPU, status)
        call check(status_ok(status) .and. &
            mlp_checkpoint_device_supported(FORTML_DEVICE_CPU), &
            "fingerprint CPU capability", failures)
        call mlp_checkpoint_require_device(FORTML_DEVICE_CUDA, status)
        call check(.not. status_ok(status) .and. &
            .not. mlp_checkpoint_device_supported(FORTML_DEVICE_CUDA), &
            "fingerprint CUDA typed refusal", failures)
    end subroutine test_device_contract

    subroutine remove_file(path)
        character(*), intent(in) :: path
        integer :: unit, ios

        open(newunit=unit, file=path, status="old", iostat=ios)
        if (ios /= 0) return
        close(unit, status="delete", iostat=ios)
    end subroutine remove_file

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (*, '(a)') "  FAIL: " // trim(description)
        end if
    end subroutine check

end program test_mlp_checkpoint_fingerprint
