program test_mlp_batch_iterator
    !! Independent behavioral checks for batch cursors and trainer controls.
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_DOMAIN_ERROR, &
        FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    use fortml_mlp, only: mlp_t, MLP_LINEAR
    use fortml_mlp_training, only: mlp_batch_iterator_t, &
        mlp_batch_iterator_cursor_t, mlp_batch_iterator_device_supported, &
        mlp_batch_iterator_require_device, &
        mlp_training_options_t, &
        mlp_training_state_t, mlp_train
    implicit none

    integer :: failures

    failures = 0
    call test_iterator_oracle(failures)
    call test_cursor_replay_oracle(failures)
    call test_gradient_accumulation_oracle(failures)
    call test_training_schedule_and_clipping(failures)
    call test_validation_restore_oracle(failures)
    if (failures > 0) then
        write (*, '(a,i0)') "FAIL MLP batch/trainer cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS MLP batch iterator and trainer controls"

contains

    subroutine test_iterator_oracle(failures)
        integer, intent(inout) :: failures
        type(mlp_batch_iterator_t) :: iterator, iterator_copy
        type(fortnum_status_t) :: status
        integer, allocatable :: first(:), second(:), third(:), repeat(:)
        integer :: expected_first(3), expected_second(3), expected_third(1)
        logical :: has_batch

        expected_first = [7, 2, 5]
        expected_second = [6, 3, 4]
        expected_third = [1]
        call iterator%initialize(7, status, batch_size=3, shuffle=.true., &
            seed=42)
        call check(status_ok(status), "iterator initialize", failures)
        call check(iterator%sample_count() == 7 .and. &
            iterator%batch_count() == 3 .and. iterator%initialized(), &
            "iterator metadata", failures)
        call iterator%reset(status)
        call iterator%next_batch(first, has_batch, status)
        call check(status_ok(status) .and. has_batch .and. &
            all(first == expected_first), "first shuffled batch oracle", failures)
        call iterator%next_batch(second, has_batch, status)
        call check(status_ok(status) .and. has_batch .and. &
            all(second == expected_second), "second shuffled batch oracle", failures)
        call iterator%next_batch(third, has_batch, status)
        call check(status_ok(status) .and. has_batch .and. &
            all(third == expected_third), "uneven final batch oracle", failures)
        call iterator%next_batch(repeat, has_batch, status)
        call check(status_ok(status) .and. .not. has_batch .and. &
            size(repeat) == 0 .and. iterator%current_epoch() == 1, &
            "explicit epoch boundary", failures)
        iterator_copy = iterator
        call iterator%reset(status)
        call iterator%next_batch(repeat, has_batch, status)
        call iterator_copy%reset(status)
        call iterator_copy%next_batch(second, has_batch, status)
        call check(status_ok(status) .and. has_batch .and. &
            all(repeat == second) .and. iterator%current_epoch() == 2, &
            "reproducible copied reset", failures)
        call iterator%initialize(0, status)
        call check(.not. status_ok(status), "invalid iterator refusal", failures)
    end subroutine test_iterator_oracle

    subroutine test_cursor_replay_oracle(failures)
        integer, intent(inout) :: failures
        type(mlp_batch_iterator_t) :: source, replay
        type(mlp_batch_iterator_cursor_t) :: cursor, empty_cursor
        type(fortnum_status_t) :: status
        integer, allocatable :: first(:), expected_next(:), actual_next(:)
        integer :: expected_first(3), expected_second(3)
        logical :: has_batch

        expected_first = [7, 2, 5]
        expected_second = [6, 3, 4]
        call source%initialize(7, status, batch_size=3, shuffle=.true., seed=42)
        call check(status_ok(status), "cursor source initialize", failures)
        call source%reset(status)
        call source%next_batch(first, has_batch, status)
        call check(status_ok(status) .and. has_batch .and. &
            all(first == expected_first), "cursor source prefix oracle", failures)
        call source%capture(cursor, status)
        call check(status_ok(status) .and. cursor%valid(), &
            "capture validates complete cursor", failures)
        call source%next_batch(expected_next, has_batch, status)
        call check(status_ok(status) .and. has_batch .and. &
            all(expected_next == expected_second), &
            "cursor source suffix oracle", failures)

        call replay%initialize(7, status, batch_size=3, shuffle=.true., seed=999)
        call replay%restore(cursor, status)
        call replay%next_batch(actual_next, has_batch, status)
        call check(status_ok(status) .and. has_batch .and. &
            all(actual_next == expected_second), &
            "restored cursor reproduces shuffled suffix", failures)
        call cursor%clear()
        call check(.not. cursor%valid(), "cleared cursor is not restorable", failures)
        call replay%restore(cursor, status)
        call check(status%code == FORTNUM_DOMAIN_ERROR .and. &
            replay%current_epoch() == 1 .and. replay%current_position() == 7, &
            "invalid cursor refusal preserves iterator", failures)
        call source%capture(empty_cursor, status)
        call check(status_ok(status) .and. empty_cursor%valid(), &
            "capture remains usable after independent refusal", failures)

        call check(mlp_batch_iterator_device_supported(FORTML_DEVICE_CPU), &
            "batch cursor CPU capability", failures)
        call check(.not. mlp_batch_iterator_device_supported(FORTML_DEVICE_CUDA), &
            "batch cursor CUDA capability refusal", failures)
        call mlp_batch_iterator_require_device(FORTML_DEVICE_CPU, status)
        call check(status_ok(status), "batch cursor CPU device contract", failures)
        call mlp_batch_iterator_require_device(FORTML_DEVICE_CUDA, status)
        call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
            "batch cursor CUDA device refusal", failures)
    end subroutine test_cursor_replay_oracle

    subroutine test_gradient_accumulation_oracle(failures)
        integer, intent(inout) :: failures
        type(mlp_t) :: full_batch_model, accumulated_model
        type(mlp_training_options_t) :: full_options, accumulated_options
        type(mlp_training_state_t) :: full_state, accumulated_state
        type(fortnum_status_t) :: status
        real(dp) :: x(5, 1), target(5, 1)
        real(dp), allocatable :: full_parameters(:), accumulated_parameters(:)

        x(:, 1) = [1.0_dp, 2.0_dp, 3.0_dp, 4.0_dp, 5.0_dp]
        target(:, 1) = [2.0_dp, 1.0_dp, 3.0_dp, 5.0_dp, 4.0_dp]
        call full_batch_model%initialize([1, 1], status, output_activation=MLP_LINEAR)
        call check(status_ok(status), "full-batch model initialize", failures)
        call full_batch_model%set_parameters([0.25_dp, -0.5_dp], status)
        call check(status_ok(status), "full-batch model parameters", failures)
        call accumulated_model%initialize([1, 1], status, output_activation=MLP_LINEAR)
        call check(status_ok(status), "accumulated model initialize", failures)
        call accumulated_model%set_parameters([0.25_dp, -0.5_dp], status)
        call check(status_ok(status), "accumulated model parameters", failures)

        full_options%max_epochs = 1
        full_options%batch_size = 5
        full_options%accumulation_steps = 1
        full_options%learning_rate = 0.01_dp
        full_options%l2 = 0.2_dp
        full_options%tolerance = 0.0_dp
        full_options%restore_best = .false.
        accumulated_options = full_options
        accumulated_options%batch_size = 2
        accumulated_options%accumulation_steps = 3

        call mlp_train(full_batch_model, x, target, status, full_options, &
            full_state)
        call check(status_ok(status), "full-batch training status", failures)
        call mlp_train(accumulated_model, x, target, status, &
            accumulated_options, accumulated_state)
        call check(status_ok(status), "accumulated training status", failures)
        full_parameters = full_batch_model%parameters()
        accumulated_parameters = accumulated_model%parameters()
        call check(full_state%updates == 1 .and. full_state%microbatches == 1, &
            "full-batch update accounting", failures)
        call check(accumulated_state%updates == 1 .and. &
            accumulated_state%microbatches == 3 .and. &
            accumulated_state%accumulation_steps == 3, &
            "accumulated update accounting", failures)
        call check(size(full_parameters) == size(accumulated_parameters) .and. &
            maxval(abs(full_parameters - accumulated_parameters)) < 1.0e-12_dp, &
            "sample-weighted accumulation matches full batch", failures)

        accumulated_options%accumulation_steps = 0
        call mlp_train(accumulated_model, x, target, status, accumulated_options)
        call check(.not. status_ok(status), "invalid accumulation refusal", failures)
    end subroutine test_gradient_accumulation_oracle

    subroutine test_training_schedule_and_clipping(failures)
        integer, intent(inout) :: failures
        type(mlp_t) :: model
        type(mlp_training_options_t) :: options
        type(mlp_training_state_t) :: state
        type(fortnum_status_t) :: status
        real(dp) :: x(4, 1), target(4, 1)

        x(:, 1) = [1.0_dp, 2.0_dp, 3.0_dp, 4.0_dp]
        target(:, 1) = 1.0_dp
        call model%initialize([1, 1], status, output_activation=MLP_LINEAR)
        call model%set_parameters([0.0_dp, 0.0_dp], status)
        options%max_epochs = 2
        options%batch_size = 4
        options%learning_rate = 0.1_dp
        options%gradient_clip_norm = 0.1_dp
        options%tolerance = 0.0_dp
        options%restore_best = .false.
        options%learning_rate_schedule => halve_by_epoch
        call mlp_train(model, x, target, status, options, state)
        call check(status_ok(status), "scheduled trainer status", failures)
        call check(state%updates == 2 .and. &
            state%gradient_clipped_updates == 2, &
            "gradient clipping update count", failures)
        call check(size(state%learning_rate_history) == 2, &
            "learning-rate history length", failures)
        if (size(state%learning_rate_history) == 2) then
            call check(abs(state%learning_rate_history(1) - 0.1_dp) < &
                1.0e-14_dp .and. abs(state%learning_rate_history(2) - &
                0.05_dp) < 1.0e-14_dp .and. abs(state%last_learning_rate - &
                0.05_dp) < 1.0e-14_dp, "learning-rate schedule history", &
                failures)
        end if
        call check(state%final_loss < state%initial_loss, &
            "scheduled clipped objective decreases", failures)
    end subroutine test_training_schedule_and_clipping

    subroutine test_validation_restore_oracle(failures)
        integer, intent(inout) :: failures
        type(mlp_t) :: model
        type(mlp_training_options_t) :: options
        type(mlp_training_state_t) :: state
        type(fortnum_status_t) :: status
        real(dp) :: x(1, 1), target(1, 1), validation_x(1, 1)
        real(dp) :: validation_target(1, 1), theta(2)
        real(dp) :: expected_loss(5), q, q_history(5), first_moment, second_moment, gradient
        real(dp) :: bias1, bias2, prediction
        integer :: i, best_epoch_expected

        x = 1.0_dp
        target = 3.0_dp
        validation_x = 1.0_dp
        validation_target = 0.5_dp
        call model%initialize([1, 1], status, output_activation=MLP_LINEAR)
        call model%set_parameters([0.0_dp, 0.0_dp], status)
        options%max_epochs = 5
        options%batch_size = 1
        options%learning_rate = 0.1_dp
        options%tolerance = 0.0_dp
        options%restore_best = .true.
        call mlp_train(model, x, target, status, options, state, &
            validation_x, validation_target)
        theta = model%parameters()
        q = 0.0_dp
        first_moment = 0.0_dp
        second_moment = 0.0_dp
        do i = 1, 5
            prediction = 2.0_dp*q
            gradient = prediction - 3.0_dp
            first_moment = 0.9_dp*first_moment + 0.1_dp*gradient
            second_moment = 0.999_dp*second_moment + 0.001_dp*gradient**2
            bias1 = 1.0_dp - 0.9_dp**i
            bias2 = 1.0_dp - 0.999_dp**i
            q = q - 0.1_dp*(first_moment/bias1)/ &
                (sqrt(second_moment/bias2) + 1.0e-8_dp)
            q_history(i) = q
            expected_loss(i) = 0.5_dp*(2.0_dp*q - 0.5_dp)**2
        end do
        best_epoch_expected = minloc(expected_loss, dim=1)
        call check(status_ok(status), "validation trainer status", failures)
        call check(size(state%validation_loss_history) == 5, &
            "validation history length", failures)
        if (size(state%validation_loss_history) == 5) then
            call check(maxval(abs(state%validation_loss_history - &
                expected_loss)) < 2.0e-8_dp, &
                "validation loss independent affine oracle", failures)
        end if
        call check(state%best_epoch == best_epoch_expected .and. &
            state%best_validation_epoch == best_epoch_expected .and. &
            abs(state%best_validation_loss - expected_loss(best_epoch_expected)) < &
            2.0e-8_dp, &
            "validation best-epoch selection", failures)
        call check(maxval(abs(theta - [q_history(best_epoch_expected), &
            q_history(best_epoch_expected)])) < 2.0e-8_dp .and. &
            abs(state%final_validation_loss - state%best_validation_loss) < &
            2.0e-8_dp, "restore-best validation checkpoint", failures)

        options%validation_interval = 0
        call mlp_train(model, x, target, status, options=options, &
            validation_x=validation_x, &
            validation_target=validation_target)
        call check(.not. status_ok(status), &
            "invalid validation interval refusal", failures)
    end subroutine test_validation_restore_oracle

    subroutine halve_by_epoch(epoch, update, base_rate, rate)
        integer, intent(in) :: epoch, update
        real(dp), intent(in) :: base_rate
        real(dp), intent(out) :: rate

        if (update < 1) then
            rate = 0.0_dp
            return
        end if
        rate = base_rate/real(epoch, dp)
    end subroutine halve_by_epoch

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: description
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (*, '(a)') "FAIL ["//description//"]"
        end if
    end subroutine check

end program test_mlp_batch_iterator
