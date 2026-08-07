program test_mlp_ema
    !! Independent behavioral oracle for deterministic MLP EMA state.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_ok
    use fortml_mlp, only: mlp_t, MLP_LINEAR
    use fortml_mlp_training, only: mlp_training_options_t, mlp_training_state_t, &
        mlp_training_checkpoint_t, mlp_train
    use fortml_mlp_checkpoint, only: mlp_checkpoint_save, mlp_checkpoint_load
    implicit none

    integer :: failures

    failures = 0
    call test_ema_recurrence(failures)
    call test_resume_and_file_round_trip(failures)
    if (failures > 0) then
        write (*, '(a,i0)') "FAIL MLP EMA cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS MLP EMA independent behavioral oracles"

contains

    subroutine test_ema_recurrence(failures)
        integer, intent(inout) :: failures
        type(mlp_t) :: model
        type(mlp_training_options_t) :: options
        type(mlp_training_state_t) :: state
        type(mlp_training_checkpoint_t) :: checkpoint
        type(fortnum_status_t) :: status
        real(dp) :: x(3, 1), target(3, 1), theta(2), expected(2), ema(2)
        real(dp) :: first(2), second(2), gradient(2), mhat(2), vhat(2)
        real(dp), parameter :: beta1 = 0.9_dp, beta2 = 0.999_dp
        real(dp), parameter :: epsilon = 1.0e-8_dp, rate = 0.1_dp
        real(dp), parameter :: decay = 0.5_dp
        integer :: step

        x(:, 1) = [-1.0_dp, 0.0_dp, 1.0_dp]
        target(:, 1) = x(:, 1)
        call model%initialize([1, 1], status, output_activation=MLP_LINEAR)
        call check(status_ok(status), "EMA model initialize", failures)
        call model%set_parameters([0.0_dp, 0.0_dp], status)
        options%max_epochs = 2
        options%learning_rate = rate
        options%beta1 = beta1
        options%beta2 = beta2
        options%epsilon = epsilon
        options%ema_decay = decay
        options%tolerance = 0.0_dp
        options%restore_best = .false.
        call mlp_train(model, x, target, status, options, state, checkpoint=checkpoint)
        call check(status_ok(status), "EMA train status", failures)
        call check(state%has_ema .and. allocated(state%ema_parameters), &
            "EMA state is present", failures)
        call check(checkpoint%valid() .and. allocated(checkpoint%ema_parameters), &
            "EMA checkpoint is present", failures)

        expected = 0.0_dp
        ema = expected
        first = 0.0_dp
        second = 0.0_dp
        do step = 1, 2
            gradient = [2.0_dp*(expected(1) - 1.0_dp)/3.0_dp, 0.0_dp]
            first = beta1*first + (1.0_dp-beta1)*gradient
            second = beta2*second + (1.0_dp-beta2)*gradient*gradient
            mhat = first/(1.0_dp-beta1**step)
            vhat = second/(1.0_dp-beta2**step)
            expected = expected - rate*mhat/(sqrt(vhat)+epsilon)
            ema = decay*ema + (1.0_dp-decay)*expected
        end do
        theta = model%parameters()
        call check(maxval(abs(theta-expected)) < 2.0e-14_dp, &
            "Adam parameter recurrence oracle", failures)
        call check(maxval(abs(state%ema_parameters-ema)) < 2.0e-14_dp, &
            "EMA recurrence oracle", failures)
        call check(maxval(abs(checkpoint%ema_parameters-ema)) < 2.0e-14_dp .and. &
            checkpoint%ema_decay == decay .and. all(ieee_is_finite(ema)), &
            "EMA checkpoint values", failures)
    end subroutine test_ema_recurrence

    subroutine test_resume_and_file_round_trip(failures)
        integer, intent(inout) :: failures
        type(mlp_t) :: full_model, split_model, loaded_model
        type(mlp_training_options_t) :: full_options, split_options
        type(mlp_training_state_t) :: full_state, split_state
        type(mlp_training_checkpoint_t) :: full_checkpoint, split_checkpoint, loaded
        type(fortnum_status_t) :: status
        real(dp) :: x(4, 1), target(4, 1)
        character(*), parameter :: path = "test_mlp_ema_checkpoint.txt"

        x(:, 1) = [-2.0_dp, -0.5_dp, 0.5_dp, 2.0_dp]
        target(:, 1) = 0.7_dp*x(:, 1) - 0.2_dp
        call full_model%initialize([1, 2, 1], status, initialization_seed=13)
        call split_model%initialize([1, 2, 1], status, initialization_seed=13)
        call loaded_model%initialize([1, 2, 1], status, initialization_seed=13)
        full_options%max_epochs = 6
        full_options%batch_size = 2
        full_options%shuffle = .true.
        full_options%shuffle_seed = 71
        full_options%learning_rate = 0.01_dp
        full_options%ema_decay = 0.8_dp
        full_options%tolerance = 0.0_dp
        full_options%restore_best = .false.
        split_options = full_options
        split_options%max_epochs = 3
        call mlp_train(full_model, x, target, status, full_options, full_state, &
            checkpoint=full_checkpoint)
        call check(status_ok(status), "full EMA trajectory", failures)
        call mlp_train(split_model, x, target, status, split_options, split_state, &
            checkpoint=split_checkpoint)
        call check(status_ok(status) .and. split_checkpoint%valid(), &
            "split EMA checkpoint", failures)
        call mlp_checkpoint_save(split_checkpoint, path, status)
        call check(status_ok(status), "EMA checkpoint save", failures)
        call mlp_checkpoint_load(loaded, path, status)
        call check(status_ok(status) .and. loaded%valid(), &
            "EMA checkpoint load", failures)
        call check(maxval(abs(loaded%ema_parameters-split_checkpoint%ema_parameters)) < &
            2.0e-14_dp, "EMA file round trip", failures)
        call loaded_model%set_parameters(loaded%parameters, status)
        call mlp_train(split_model, x, target, status, full_options, split_state, &
            checkpoint=split_checkpoint)
        call check(status_ok(status), "native EMA resume", failures)
        call mlp_train(loaded_model, x, target, status, full_options, split_state, &
            checkpoint=loaded)
        call check(status_ok(status), "serialized EMA resume", failures)
        call check(maxval(abs(full_model%parameters()-split_model%parameters())) < &
            2.0e-14_dp .and. maxval(abs(full_model%parameters()-loaded_model%parameters())) < &
            2.0e-14_dp, "EMA resumed parameter trajectory", failures)
        call check(maxval(abs(full_checkpoint%ema_parameters-split_checkpoint%ema_parameters)) < &
            2.0e-14_dp .and. maxval(abs(full_checkpoint%ema_parameters-loaded%ema_parameters)) < &
            2.0e-14_dp, "EMA resumed state trajectory", failures)
        call remove_file(path)
    end subroutine test_resume_and_file_round_trip

    subroutine check(condition, label, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: label
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (*, '(a)') "FAIL: " // trim(label)
        end if
    end subroutine check

    subroutine remove_file(path)
        character(*), intent(in) :: path
        logical :: exists
        integer :: unit, ios

        inquire(file=path, exist=exists)
        if (.not. exists) return
        open(newunit=unit, file=path, status="old", iostat=ios)
        if (ios == 0) close(unit, status="delete")
    end subroutine remove_file

end program test_mlp_ema
