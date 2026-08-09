program test_mlp_checkpoint_io
    !! Behavioral oracle for portable, versioned MLP checkpoint persistence.
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_ok
    use fortml_mlp, only: mlp_t, MLP_LINEAR
    use fortml_mlp_training, only: mlp_training_options_t, &
        mlp_training_state_t, mlp_training_checkpoint_t, mlp_train, &
        MLP_OPTIMIZER_ADAM, MLP_OPTIMIZER_SGD, MLP_OPTIMIZER_ADAMW, &
        MLP_OPTIMIZER_ADAGRAD, MLP_OPTIMIZER_RMSPROP
    use fortml_mlp_checkpoint, only: mlp_checkpoint_save, mlp_checkpoint_load, &
        MLP_CHECKPOINT_MAGIC, MLP_CHECKPOINT_SCHEMA_VERSION, &
        mlp_checkpoint_device_supported, mlp_checkpoint_require_device
    use fortml_device, only: FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    implicit none

    integer :: failures, optimizer
    character(*), parameter :: path = "test_mlp_checkpoint_io.txt"

    failures = 0
    do optimizer = MLP_OPTIMIZER_ADAM, MLP_OPTIMIZER_RMSPROP
        call test_round_trip_and_resume(optimizer, path, failures)
    end do
    call test_refusals(path, failures)
    call test_device_contract(failures)
    call remove_file(path)
    if (failures > 0) then
        write (*, '(a,i0)') "FAIL MLP checkpoint persistence cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS MLP checkpoint persistence behavioral oracles"

contains

    subroutine test_round_trip_and_resume(optimizer, path, failures)
        integer, intent(in) :: optimizer
        character(*), intent(in) :: path
        integer, intent(inout) :: failures
        type(mlp_t) :: full_model, split_model
        type(mlp_t) :: loaded_model
        type(mlp_training_options_t) :: full_options, split_options
        type(mlp_training_state_t) :: full_state, split_state
        type(mlp_training_checkpoint_t) :: full_checkpoint, split_checkpoint
        type(mlp_training_checkpoint_t) :: loaded_checkpoint
        type(fortnum_status_t) :: status
        real(dp) :: x(6, 1), target(6, 1), validation_x(2, 1), validation_target(2, 1)
        integer :: layers(2)

        x(:, 1) = [-3.0_dp, -2.0_dp, -1.0_dp, 1.0_dp, 2.0_dp, 3.0_dp]
        target(:, 1) = 0.35_dp*x(:, 1) + 0.2_dp
        validation_x(:, 1) = [-0.5_dp, 0.5_dp]
        validation_target(:, 1) = 0.35_dp*validation_x(:, 1) + 0.2_dp
        layers = [1, 1]
        call full_model%initialize(layers, status, output_activation=MLP_LINEAR)
        call split_model%initialize(layers, status, output_activation=MLP_LINEAR)
        call loaded_model%initialize(layers, status, output_activation=MLP_LINEAR)
        call full_model%set_parameters([0.1_dp, -0.15_dp], status)
        call split_model%set_parameters([0.1_dp, -0.15_dp], status)
        call loaded_model%set_parameters([0.1_dp, -0.15_dp], status)

        full_options%max_epochs = 5
        full_options%batch_size = 2
        full_options%accumulation_steps = 2
        full_options%validation_interval = 1
        full_options%shuffle = .true.
        full_options%shuffle_seed = 29
        full_options%restore_best = .false.
        full_options%learning_rate = 0.03_dp
        full_options%l2 = 0.001_dp
        full_options%optimizer = optimizer
        full_options%momentum = 0.2_dp
        full_options%nesterov = optimizer == MLP_OPTIMIZER_SGD
        full_options%weight_decay = 0.01_dp
        full_options%rmsprop_decay = 0.8_dp
        full_options%rmsprop_momentum = 0.1_dp
        full_options%rmsprop_centered = .true.
        full_options%epsilon = 1.0e-5_dp
        full_options%tolerance = 0.0_dp
        split_options = full_options
        split_options%max_epochs = 3

        call mlp_train(full_model, x, target, status, full_options, full_state, &
            validation_x=validation_x, validation_target=validation_target, &
            checkpoint=full_checkpoint)
        call check(status_ok(status) .and. full_checkpoint%valid(), &
            "full checkpoint capture", failures)
        call mlp_train(split_model, x, target, status, split_options, split_state, &
            validation_x=validation_x, validation_target=validation_target, &
            checkpoint=split_checkpoint)
        call check(status_ok(status) .and. split_checkpoint%valid(), &
            "split checkpoint capture", failures)

        ! Exercise the weighted microbatch cursor explicitly.  A completed
        ! epoch normally has zero pending mass; a production checkpoint may
        ! also be written between microbatches, so this non-integral value is
        ! an independent persistence oracle for the scalar state.
        split_checkpoint%accumulated_weight_mass = 2.75_dp
        call check(split_checkpoint%valid(), "weighted cursor checkpoint", failures)

        call mlp_checkpoint_save(split_checkpoint, path, status)
        call check(status_ok(status), "checkpoint save", failures)
        call mlp_checkpoint_load(loaded_checkpoint, path, status)
        call check(status_ok(status) .and. loaded_checkpoint%valid(), &
            "checkpoint load", failures)
        call check(same_checkpoint(split_checkpoint, loaded_checkpoint), &
            "exact checkpoint round-trip", failures)

        call loaded_model%set_parameters(loaded_checkpoint%parameters, status)
        call mlp_train(split_model, x, target, status, full_options, split_state, &
            validation_x=validation_x, validation_target=validation_target, &
            checkpoint=split_checkpoint)
        call check(status_ok(status), "native checkpoint resume", failures)
        call mlp_train(loaded_model, x, target, status, full_options, split_state, &
            validation_x=validation_x, validation_target=validation_target, &
            checkpoint=loaded_checkpoint)
        call check(status_ok(status), "serialized checkpoint resume", failures)
        call check(maxval(abs(full_model%parameters() - split_model%parameters())) < &
            2.0e-14_dp, "uninterrupted versus native resume", failures)
        call check(maxval(abs(full_model%parameters() - loaded_model%parameters())) < &
            2.0e-14_dp, "uninterrupted versus serialized resume", failures)
        call check(same_checkpoint(split_checkpoint, loaded_checkpoint), &
            "resumed checkpoint state", failures)
    end subroutine test_round_trip_and_resume

    subroutine test_refusals(path, failures)
        character(*), intent(in) :: path
        integer, intent(inout) :: failures
        type(mlp_training_checkpoint_t) :: checkpoint, loaded
        type(fortnum_status_t) :: status
        integer :: unit, ios

        call mlp_checkpoint_save(checkpoint, path, status)
        call check(.not. status_ok(status), "uninitialized checkpoint refusal", failures)

        open(newunit=unit, file=path, status="replace", action="write", &
            form="formatted", iostat=ios)
        write(unit, '(a)') MLP_CHECKPOINT_MAGIC
        write(unit, '(a)') "schema_version 999"
        close(unit)
        call mlp_checkpoint_load(loaded, path, status)
        call check(.not. status_ok(status), "unknown schema refusal", failures)
        call check(.not. loaded%initialized, &
            "failed load does not initialize destination", failures)

        open(newunit=unit, file=path, status="replace", action="write", &
            form="formatted", iostat=ios)
        write(unit, '(a)') MLP_CHECKPOINT_MAGIC
        write(unit, '(a)') "schema_version 1"
        write(unit, '(a)') "format_version 3"
        close(unit)
        call mlp_checkpoint_load(loaded, path, status)
        call check(.not. status_ok(status), "truncated snapshot refusal", failures)
    end subroutine test_refusals

    subroutine test_device_contract(failures)
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status

        call check(MLP_CHECKPOINT_SCHEMA_VERSION == 12, &
            "checkpoint schema version", failures)
        call mlp_checkpoint_require_device(FORTML_DEVICE_CPU, status)
        call check(status_ok(status) .and. &
            mlp_checkpoint_device_supported(FORTML_DEVICE_CPU), &
            "CPU checkpoint capability", failures)
        call mlp_checkpoint_require_device(FORTML_DEVICE_CUDA, status)
        call check(.not. status_ok(status) .and. &
            .not. mlp_checkpoint_device_supported(FORTML_DEVICE_CUDA), &
            "CUDA checkpoint typed refusal", failures)
        call mlp_checkpoint_require_device(-1, status)
        call check(.not. status_ok(status), "invalid checkpoint device refusal", failures)
    end subroutine test_device_contract

    logical function same_checkpoint(a, b) result(equal)
        type(mlp_training_checkpoint_t), intent(in) :: a, b

        equal = a%format_version == b%format_version .and. &
            a%initialized .eqv. b%initialized .and. a%resume_safe .eqv. b%resume_safe .and. &
            a%n_samples == b%n_samples .and. a%n_features == b%n_features .and. &
            a%n_outputs == b%n_outputs .and. a%n_parameters == b%n_parameters .and. &
            a%epoch == b%epoch .and. a%updates == b%updates .and. &
            a%microbatches == b%microbatches .and. a%microbatch_position == b%microbatch_position .and. &
            a%active_epoch == b%active_epoch .and. a%active_microbatches == b%active_microbatches .and. &
            a%accumulated_samples == b%accumulated_samples .and. &
            a%accumulated_weight_mass == b%accumulated_weight_mass .and. &
            a%iterator_epoch == b%iterator_epoch .and. a%iterator_position == b%iterator_position .and. &
            a%batch_size == b%batch_size .and. a%accumulation_steps == b%accumulation_steps .and. &
            a%shuffle_seed == b%shuffle_seed .and. a%adam_step_count == b%adam_step_count .and. &
            a%optimizer == b%optimizer .and. a%precision_kind == b%precision_kind .and. &
            a%stale_epochs == b%stale_epochs .and. &
            a%gradient_clipped_updates == b%gradient_clipped_updates .and. &
            a%validation_interval == b%validation_interval .and. a%patience == b%patience .and. &
            a%shuffle .eqv. b%shuffle .and. a%has_validation .eqv. b%has_validation .and. &
            a%converged .eqv. b%converged .and. a%early_stopped .eqv. b%early_stopped .and. &
            a%restore_best .eqv. b%restore_best .and. a%shuffle_state == b%shuffle_state .and. &
            (a%has_typed_schedule .eqv. b%has_typed_schedule) .and. &
            a%typed_schedule%kind == b%typed_schedule%kind .and. &
            a%typed_schedule%warmup_updates == b%typed_schedule%warmup_updates .and. &
            a%typed_schedule%total_updates == b%typed_schedule%total_updates .and. &
            a%typed_schedule%min_rate_fraction == b%typed_schedule%min_rate_fraction .and. &
            a%typed_schedule%decay_factor == b%typed_schedule%decay_factor .and. &
            a%typed_schedule%peak_rate_fraction == b%typed_schedule%peak_rate_fraction .and. &
            a%typed_schedule%final_rate_fraction == b%typed_schedule%final_rate_fraction .and. &
            a%learning_rate == b%learning_rate .and. a%beta1 == b%beta1 .and. &
            a%beta2 == b%beta2 .and. a%epsilon == b%epsilon .and. &
            a%rmsprop_decay == b%rmsprop_decay .and. &
            a%rmsprop_momentum == b%rmsprop_momentum .and. &
            a%rmsprop_centered .eqv. b%rmsprop_centered .and. a%momentum == b%momentum .and. &
            a%nesterov .eqv. b%nesterov .and. a%weight_decay == b%weight_decay .and. &
            a%l2 == b%l2 .and. a%tolerance == b%tolerance .and. a%min_delta == b%min_delta .and. &
            a%gradient_clip_norm == b%gradient_clip_norm .and. &
            a%last_learning_rate == b%last_learning_rate .and. &
            a%initial_loss == b%initial_loss .and. a%final_loss == b%final_loss .and. &
            a%best_loss == b%best_loss .and. &
            a%initial_validation_loss == b%initial_validation_loss .and. &
            a%final_validation_loss == b%final_validation_loss .and. &
            a%best_validation_loss == b%best_validation_loss .and. &
            a%best_epoch == b%best_epoch .and. a%best_validation_epoch == b%best_validation_epoch
        if (.not. equal) return
        equal = allocated(a%parameters) .eqv. allocated(b%parameters)
        if (.not. equal) return
        equal = all(a%parameters == b%parameters) .and. &
            all(a%first_moment == b%first_moment) .and. &
            all(a%second_moment == b%second_moment) .and. &
            all(a%best_parameters == b%best_parameters) .and. &
            all(a%accumulated_gradient == b%accumulated_gradient) .and. &
            all(a%iterator_order == b%iterator_order) .and. &
            all(a%loss_history == b%loss_history) .and. &
            all(a%learning_rate_history == b%learning_rate_history)
        if (.not. equal) return
        equal = allocated(a%rmsprop_buffer) .eqv. allocated(b%rmsprop_buffer)
        if (equal .and. allocated(a%rmsprop_buffer)) then
            equal = all(a%rmsprop_buffer == b%rmsprop_buffer)
        end if
        if (.not. equal) return
        equal = allocated(a%validation_loss_history) .eqv. &
            allocated(b%validation_loss_history)
        if (equal .and. allocated(a%validation_loss_history)) then
            equal = all(a%validation_loss_history == b%validation_loss_history)
        end if
    end function same_checkpoint

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

end program test_mlp_checkpoint_io
