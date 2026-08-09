module fortml_mlp_checkpoint
    !! Portable, versioned persistence for MLP training checkpoints.
    !!
    !! The on-disk representation is deliberately formatted text rather than
    !! compiler-dependent unformatted storage.  Every scalar has a named
    !! record and every array has an explicit length followed by one value per
    !! record.  Double precision values use 17 significant decimal digits,
    !! which is sufficient to round-trip an IEEE binary64 value.  A reader
    !! validates the complete snapshot before replacing its destination, so a
    !! truncated or incompatible file cannot leave a usable checkpoint half
    !! updated.
    use, intrinsic :: iso_fortran_env, only: int64, iostat_end
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortml_mlp_training, only: mlp_training_checkpoint_t
    use fortml_device, only: FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    implicit none
    private

    character(*), parameter, public :: MLP_CHECKPOINT_MAGIC = &
        "FORTML_MLP_CHECKPOINT_TEXT"
    integer, parameter, public :: MLP_CHECKPOINT_SCHEMA_VERSION = 12
    integer(int64), parameter :: MLP_CHECKPOINT_FINGERPRINT_MODULUS = 2147483629_int64
    integer(int64), parameter :: MLP_CHECKPOINT_FINGERPRINT_BASE = 131_int64

    public :: mlp_checkpoint_save
    public :: mlp_checkpoint_load
    public :: mlp_checkpoint_device_supported
    public :: mlp_checkpoint_require_device
    public :: mlp_checkpoint_fingerprint

contains

    logical function mlp_checkpoint_device_supported(device_kind) result(supported)
        !! Return whether formatted checkpoint state is supported on a device.
        !!
        !! The text format is a host-owned state boundary.  CUDA resident
        !! trainers must first materialize a complete host snapshot through
        !! their own explicit transfer contract; this predicate prevents a
        !! caller from accidentally treating a host file as resident state.
        integer, intent(in) :: device_kind

        supported = device_kind == FORTML_DEVICE_CPU
    end function mlp_checkpoint_device_supported

    subroutine mlp_checkpoint_require_device(device_kind, status)
        !! Validate the checkpoint/device boundary without a hidden fallback.
        integer, intent(in) :: device_kind
        type(fortnum_status_t), intent(out) :: status

        if (device_kind == FORTML_DEVICE_CPU) then
            call status_set(status, FORTNUM_OK, "")
        else if (device_kind == FORTML_DEVICE_CUDA) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "MLP checkpoint: CUDA-resident serialization requires an explicit host snapshot")
        else
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP checkpoint: invalid device kind")
        end if
    end subroutine mlp_checkpoint_require_device

    integer(int64) function mlp_checkpoint_fingerprint(checkpoint) result(fingerprint)
        !! Return a deterministic identity for a valid host checkpoint.
        !!
        !! The fingerprint covers the schema, continuation metadata, optimizer
        !! hyperparameters, and every serialized state array.  It is a compact
        !! integrity/identity token rather than a cryptographic checksum;
        !! zero means that `checkpoint` is uninitialized or invalid.  The
        !! decimal token stream matches the formatted checkpoint representation,
        !! so the result is independent of compiler binary-layout choices.
        type(mlp_training_checkpoint_t), intent(in) :: checkpoint
        integer(int64) :: hash

        fingerprint = 0_int64
        if (.not. checkpoint%valid()) return

        hash = 1_int64
        call fingerprint_integer(hash, "schema_version", MLP_CHECKPOINT_SCHEMA_VERSION)
        call fingerprint_integer(hash, "format_version", checkpoint%format_version)
        call fingerprint_logical(hash, "initialized", checkpoint%initialized)
        call fingerprint_logical(hash, "resume_safe", checkpoint%resume_safe)
        call fingerprint_integer(hash, "n_samples", checkpoint%n_samples)
        call fingerprint_integer(hash, "n_features", checkpoint%n_features)
        call fingerprint_integer(hash, "n_outputs", checkpoint%n_outputs)
        call fingerprint_integer(hash, "n_parameters", checkpoint%n_parameters)
        call fingerprint_integer(hash, "n_optimizer_groups", checkpoint%n_optimizer_groups)
        call fingerprint_integer(hash, "epoch", checkpoint%epoch)
        call fingerprint_integer(hash, "updates", checkpoint%updates)
        call fingerprint_integer(hash, "microbatches", checkpoint%microbatches)
        call fingerprint_integer(hash, "microbatch_position", checkpoint%microbatch_position)
        call fingerprint_integer(hash, "active_epoch", checkpoint%active_epoch)
        call fingerprint_integer(hash, "active_microbatches", checkpoint%active_microbatches)
        call fingerprint_integer(hash, "accumulated_samples", checkpoint%accumulated_samples)
        call fingerprint_real(hash, "accumulated_weight_mass", &
            checkpoint%accumulated_weight_mass)
        call fingerprint_integer(hash, "iterator_epoch", checkpoint%iterator_epoch)
        call fingerprint_integer(hash, "iterator_position", checkpoint%iterator_position)
        call fingerprint_integer(hash, "batch_size", checkpoint%batch_size)
        call fingerprint_integer(hash, "accumulation_steps", checkpoint%accumulation_steps)
        call fingerprint_integer(hash, "shuffle_seed", checkpoint%shuffle_seed)
        call fingerprint_integer(hash, "adam_step_count", checkpoint%adam_step_count)
        call fingerprint_integer(hash, "optimizer", checkpoint%optimizer)
        call fingerprint_integer(hash, "precision_kind", checkpoint%precision_kind)
        call fingerprint_logical(hash, "loss_scale_enabled", checkpoint%loss_scale%enabled)
        call fingerprint_real(hash, "loss_scale_scale", checkpoint%loss_scale%scale)
        call fingerprint_real(hash, "loss_scale_initial_scale", &
            checkpoint%loss_scale%initial_scale)
        call fingerprint_real(hash, "loss_scale_growth_factor", &
            checkpoint%loss_scale%growth_factor)
        call fingerprint_real(hash, "loss_scale_backoff_factor", &
            checkpoint%loss_scale%backoff_factor)
        call fingerprint_real(hash, "loss_scale_minimum_scale", &
            checkpoint%loss_scale%minimum_scale)
        call fingerprint_real(hash, "loss_scale_maximum_scale", &
            checkpoint%loss_scale%maximum_scale)
        call fingerprint_integer(hash, "loss_scale_growth_interval", &
            checkpoint%loss_scale%growth_interval)
        call fingerprint_integer(hash, "loss_scale_good_steps", checkpoint%loss_scale%good_steps)
        call fingerprint_integer(hash, "loss_scale_overflow_count", &
            checkpoint%loss_scale%overflow_count)
        call fingerprint_integer(hash, "loss_scale_skipped_updates", &
            checkpoint%loss_scale%skipped_updates)
        call fingerprint_integer(hash, "stale_epochs", checkpoint%stale_epochs)
        call fingerprint_integer(hash, "schedule_bad_updates", checkpoint%schedule_bad_updates)
        call fingerprint_integer(hash, "schedule_reductions", checkpoint%schedule_reductions)
        call fingerprint_integer(hash, "gradient_clipped_updates", &
            checkpoint%gradient_clipped_updates)
        call fingerprint_integer(hash, "validation_interval", checkpoint%validation_interval)
        call fingerprint_integer(hash, "patience", checkpoint%patience)
        call fingerprint_logical(hash, "shuffle", checkpoint%shuffle)
        call fingerprint_logical(hash, "has_validation", checkpoint%has_validation)
        call fingerprint_logical(hash, "converged", checkpoint%converged)
        call fingerprint_logical(hash, "early_stopped", checkpoint%early_stopped)
        call fingerprint_logical(hash, "restore_best", checkpoint%restore_best)
        call fingerprint_logical(hash, "has_typed_schedule", checkpoint%has_typed_schedule)
        call fingerprint_logical(hash, "schedule_metric_initialized", &
            checkpoint%schedule_metric_initialized)
        call fingerprint_integer(hash, "typed_schedule_kind", checkpoint%typed_schedule%kind)
        call fingerprint_integer(hash, "typed_schedule_warmup_updates", &
            checkpoint%typed_schedule%warmup_updates)
        call fingerprint_integer(hash, "typed_schedule_total_updates", &
            checkpoint%typed_schedule%total_updates)
        call fingerprint_real(hash, "typed_schedule_min_rate_fraction", &
            checkpoint%typed_schedule%min_rate_fraction)
        call fingerprint_real(hash, "typed_schedule_decay_factor", &
            checkpoint%typed_schedule%decay_factor)
        call fingerprint_real(hash, "typed_schedule_peak_rate_fraction", &
            checkpoint%typed_schedule%peak_rate_fraction)
        call fingerprint_real(hash, "typed_schedule_final_rate_fraction", &
            checkpoint%typed_schedule%final_rate_fraction)
        call fingerprint_integer(hash, "typed_schedule_metric_mode", &
            checkpoint%typed_schedule%metric_mode)
        call fingerprint_integer(hash, "typed_schedule_patience_updates", &
            checkpoint%typed_schedule%patience_updates)
        call fingerprint_real(hash, "typed_schedule_min_delta", &
            checkpoint%typed_schedule%min_delta)
        call fingerprint_real(hash, "typed_schedule_plateau_factor", &
            checkpoint%typed_schedule%plateau_factor)
        call fingerprint_int64(hash, "shuffle_state", checkpoint%shuffle_state)
        call fingerprint_real(hash, "schedule_best_metric", checkpoint%schedule_best_metric)
        call fingerprint_real(hash, "learning_rate", checkpoint%learning_rate)
        call fingerprint_real(hash, "beta1", checkpoint%beta1)
        call fingerprint_real(hash, "beta2", checkpoint%beta2)
        call fingerprint_real(hash, "epsilon", checkpoint%epsilon)
        call fingerprint_real(hash, "adafactor_decay", checkpoint%adafactor_decay)
        call fingerprint_real(hash, "adafactor_clip_threshold", &
            checkpoint%adafactor_clip_threshold)
        call fingerprint_logical(hash, "adafactor_relative_step", &
            checkpoint%adafactor_relative_step)
        call fingerprint_logical(hash, "adafactor_scale_parameter", &
            checkpoint%adafactor_scale_parameter)
        call fingerprint_logical(hash, "adafactor_factored", checkpoint%adafactor_factored)
        call fingerprint_integer(hash, "n_adafactor_blocks", checkpoint%n_adafactor_blocks)
        call fingerprint_real(hash, "rmsprop_decay", checkpoint%rmsprop_decay)
        call fingerprint_real(hash, "rmsprop_momentum", checkpoint%rmsprop_momentum)
        call fingerprint_logical(hash, "rmsprop_centered", checkpoint%rmsprop_centered)
        call fingerprint_real(hash, "momentum", checkpoint%momentum)
        call fingerprint_logical(hash, "nesterov", checkpoint%nesterov)
        call fingerprint_real(hash, "weight_decay", checkpoint%weight_decay)
        call fingerprint_real(hash, "l2", checkpoint%l2)
        call fingerprint_real(hash, "tolerance", checkpoint%tolerance)
        call fingerprint_real(hash, "min_delta", checkpoint%min_delta)
        call fingerprint_real(hash, "gradient_clip_norm", checkpoint%gradient_clip_norm)
        call fingerprint_real(hash, "ema_decay", checkpoint%ema_decay)
        call fingerprint_real(hash, "last_learning_rate", checkpoint%last_learning_rate)
        call fingerprint_real(hash, "initial_loss", checkpoint%initial_loss)
        call fingerprint_real(hash, "final_loss", checkpoint%final_loss)
        call fingerprint_real(hash, "best_loss", checkpoint%best_loss)
        call fingerprint_real(hash, "initial_validation_loss", &
            checkpoint%initial_validation_loss)
        call fingerprint_real(hash, "final_validation_loss", checkpoint%final_validation_loss)
        call fingerprint_real(hash, "best_validation_loss", checkpoint%best_validation_loss)
        call fingerprint_integer(hash, "best_epoch", checkpoint%best_epoch)
        call fingerprint_integer(hash, "best_validation_epoch", checkpoint%best_validation_epoch)

        call fingerprint_real_array(hash, "parameters", checkpoint%parameters)
        call fingerprint_optional_character_array(hash, "optimizer_group_name", &
            checkpoint%optimizer_group_name)
        call fingerprint_optional_integer_array(hash, "optimizer_group_first", &
            checkpoint%optimizer_group_first)
        call fingerprint_optional_integer_array(hash, "optimizer_group_last", &
            checkpoint%optimizer_group_last)
        call fingerprint_optional_real_array(hash, &
            "optimizer_group_learning_rate_multiplier", &
            checkpoint%optimizer_group_learning_rate_multiplier)
        call fingerprint_real_array(hash, "first_moment", checkpoint%first_moment)
        call fingerprint_real_array(hash, "second_moment", checkpoint%second_moment)
        call fingerprint_optional_integer_array(hash, "adafactor_block_first", &
            checkpoint%adafactor_block_first)
        call fingerprint_optional_integer_array(hash, "adafactor_block_last", &
            checkpoint%adafactor_block_last)
        call fingerprint_optional_integer_array(hash, "adafactor_block_rows", &
            checkpoint%adafactor_block_rows)
        call fingerprint_optional_integer_array(hash, "adafactor_block_columns", &
            checkpoint%adafactor_block_columns)
        call fingerprint_optional_integer_array(hash, "adafactor_block_factored", &
            checkpoint%adafactor_block_factored)
        call fingerprint_optional_real_array(hash, "adafactor_row_moment", &
            checkpoint%adafactor_row_moment)
        call fingerprint_optional_real_array(hash, "adafactor_column_moment", &
            checkpoint%adafactor_column_moment)
        call fingerprint_optional_real_array(hash, "adafactor_second_moment", &
            checkpoint%adafactor_second_moment)
        call fingerprint_optional_real_array(hash, "max_second_moment", &
            checkpoint%max_second_moment)
        call fingerprint_optional_real_array(hash, "rmsprop_buffer", &
            checkpoint%rmsprop_buffer)
        call fingerprint_real_array(hash, "best_parameters", checkpoint%best_parameters)
        call fingerprint_real_array(hash, "accumulated_gradient", &
            checkpoint%accumulated_gradient)
        call fingerprint_integer_array(hash, "iterator_order", checkpoint%iterator_order)
        call fingerprint_real_array(hash, "loss_history", checkpoint%loss_history)
        call fingerprint_real_array(hash, "learning_rate_history", &
            checkpoint%learning_rate_history)
        call fingerprint_optional_real_array(hash, "validation_loss_history", &
            checkpoint%validation_loss_history)
        call fingerprint_optional_real_array(hash, "ema_parameters", checkpoint%ema_parameters)

        fingerprint = hash
    end function mlp_checkpoint_fingerprint

    subroutine mlp_checkpoint_save(checkpoint, path, status)
        !! Write a valid checkpoint to a portable formatted-text file.
        type(mlp_training_checkpoint_t), intent(in) :: checkpoint
        character(*), intent(in) :: path
        type(fortnum_status_t), intent(out) :: status
        integer :: unit, ios, close_ios, i

        if (.not. checkpoint%valid()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP checkpoint save: checkpoint is invalid")
            return
        end if

        open(newunit=unit, file=path, status="replace", action="write", &
            form="formatted", access="sequential", iostat=ios)
        if (ios /= 0) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP checkpoint save: cannot open destination")
            return
        end if

        write(unit, "(A)", iostat=ios) MLP_CHECKPOINT_MAGIC
        if (ios == 0) call write_i(unit, "schema_version", &
            MLP_CHECKPOINT_SCHEMA_VERSION, ios)
        if (ios == 0) call write_i(unit, "format_version", &
            checkpoint%format_version, ios)
        if (ios == 0) call write_l(unit, "initialized", checkpoint%initialized, ios)
        if (ios == 0) call write_l(unit, "resume_safe", checkpoint%resume_safe, ios)
        if (ios == 0) call write_i(unit, "n_samples", checkpoint%n_samples, ios)
        if (ios == 0) call write_i(unit, "n_features", checkpoint%n_features, ios)
        if (ios == 0) call write_i(unit, "n_outputs", checkpoint%n_outputs, ios)
        if (ios == 0) call write_i(unit, "n_parameters", checkpoint%n_parameters, ios)
        if (ios == 0) call write_i(unit, "n_optimizer_groups", &
            checkpoint%n_optimizer_groups, ios)
        if (ios == 0) call write_i(unit, "epoch", checkpoint%epoch, ios)
        if (ios == 0) call write_i(unit, "updates", checkpoint%updates, ios)
        if (ios == 0) call write_i(unit, "microbatches", checkpoint%microbatches, ios)
        if (ios == 0) call write_i(unit, "microbatch_position", &
            checkpoint%microbatch_position, ios)
        if (ios == 0) call write_i(unit, "active_epoch", checkpoint%active_epoch, ios)
        if (ios == 0) call write_i(unit, "active_microbatches", &
            checkpoint%active_microbatches, ios)
        if (ios == 0) call write_i(unit, "accumulated_samples", &
            checkpoint%accumulated_samples, ios)
        if (ios == 0) call write_r(unit, "accumulated_weight_mass", &
            checkpoint%accumulated_weight_mass, ios)
        if (ios == 0) call write_i(unit, "iterator_epoch", checkpoint%iterator_epoch, ios)
        if (ios == 0) call write_i(unit, "iterator_position", &
            checkpoint%iterator_position, ios)
        if (ios == 0) call write_i(unit, "batch_size", checkpoint%batch_size, ios)
        if (ios == 0) call write_i(unit, "accumulation_steps", &
            checkpoint%accumulation_steps, ios)
        if (ios == 0) call write_i(unit, "shuffle_seed", checkpoint%shuffle_seed, ios)
        if (ios == 0) call write_i(unit, "adam_step_count", &
            checkpoint%adam_step_count, ios)
        if (ios == 0) call write_i(unit, "optimizer", checkpoint%optimizer, ios)
        if (ios == 0) call write_i(unit, "precision_kind", checkpoint%precision_kind, ios)
        if (ios == 0) call write_l(unit, "loss_scale_enabled", checkpoint%loss_scale%enabled, ios)
        if (ios == 0) call write_r(unit, "loss_scale_scale", checkpoint%loss_scale%scale, ios)
        if (ios == 0) call write_r(unit, "loss_scale_initial_scale", &
            checkpoint%loss_scale%initial_scale, ios)
        if (ios == 0) call write_r(unit, "loss_scale_growth_factor", &
            checkpoint%loss_scale%growth_factor, ios)
        if (ios == 0) call write_r(unit, "loss_scale_backoff_factor", &
            checkpoint%loss_scale%backoff_factor, ios)
        if (ios == 0) call write_r(unit, "loss_scale_minimum_scale", &
            checkpoint%loss_scale%minimum_scale, ios)
        if (ios == 0) call write_r(unit, "loss_scale_maximum_scale", &
            checkpoint%loss_scale%maximum_scale, ios)
        if (ios == 0) call write_i(unit, "loss_scale_growth_interval", &
            checkpoint%loss_scale%growth_interval, ios)
        if (ios == 0) call write_i(unit, "loss_scale_good_steps", &
            checkpoint%loss_scale%good_steps, ios)
        if (ios == 0) call write_i(unit, "loss_scale_overflow_count", &
            checkpoint%loss_scale%overflow_count, ios)
        if (ios == 0) call write_i(unit, "loss_scale_skipped_updates", &
            checkpoint%loss_scale%skipped_updates, ios)
        if (ios == 0) call write_i(unit, "stale_epochs", checkpoint%stale_epochs, ios)
        if (ios == 0) call write_i(unit, "schedule_bad_updates", &
            checkpoint%schedule_bad_updates, ios)
        if (ios == 0) call write_i(unit, "schedule_reductions", &
            checkpoint%schedule_reductions, ios)
        if (ios == 0) call write_i(unit, "gradient_clipped_updates", &
            checkpoint%gradient_clipped_updates, ios)
        if (ios == 0) call write_i(unit, "validation_interval", &
            checkpoint%validation_interval, ios)
        if (ios == 0) call write_i(unit, "patience", checkpoint%patience, ios)
        if (ios == 0) call write_l(unit, "shuffle", checkpoint%shuffle, ios)
        if (ios == 0) call write_l(unit, "has_validation", checkpoint%has_validation, ios)
        if (ios == 0) call write_l(unit, "converged", checkpoint%converged, ios)
        if (ios == 0) call write_l(unit, "early_stopped", checkpoint%early_stopped, ios)
        if (ios == 0) call write_l(unit, "restore_best", checkpoint%restore_best, ios)
        if (ios == 0) call write_l(unit, "has_typed_schedule", &
            checkpoint%has_typed_schedule, ios)
        if (ios == 0) call write_l(unit, "schedule_metric_initialized", &
            checkpoint%schedule_metric_initialized, ios)
        if (ios == 0) call write_i(unit, "typed_schedule_kind", &
            checkpoint%typed_schedule%kind, ios)
        if (ios == 0) call write_i(unit, "typed_schedule_warmup_updates", &
            checkpoint%typed_schedule%warmup_updates, ios)
        if (ios == 0) call write_i(unit, "typed_schedule_total_updates", &
            checkpoint%typed_schedule%total_updates, ios)
        if (ios == 0) call write_r(unit, "typed_schedule_min_rate_fraction", &
            checkpoint%typed_schedule%min_rate_fraction, ios)
        if (ios == 0) call write_r(unit, "typed_schedule_decay_factor", &
            checkpoint%typed_schedule%decay_factor, ios)
        if (ios == 0) call write_r(unit, "typed_schedule_peak_rate_fraction", &
            checkpoint%typed_schedule%peak_rate_fraction, ios)
        if (ios == 0) call write_r(unit, "typed_schedule_final_rate_fraction", &
            checkpoint%typed_schedule%final_rate_fraction, ios)
        if (ios == 0) call write_i(unit, "typed_schedule_metric_mode", &
            checkpoint%typed_schedule%metric_mode, ios)
        if (ios == 0) call write_i(unit, "typed_schedule_patience_updates", &
            checkpoint%typed_schedule%patience_updates, ios)
        if (ios == 0) call write_r(unit, "typed_schedule_min_delta", &
            checkpoint%typed_schedule%min_delta, ios)
        if (ios == 0) call write_r(unit, "typed_schedule_plateau_factor", &
            checkpoint%typed_schedule%plateau_factor, ios)
        if (ios == 0) call write_i8(unit, "shuffle_state", checkpoint%shuffle_state, ios)
        if (ios == 0) call write_r(unit, "schedule_best_metric", &
            checkpoint%schedule_best_metric, ios)
        if (ios == 0) call write_r(unit, "learning_rate", checkpoint%learning_rate, ios)
        if (ios == 0) call write_r(unit, "beta1", checkpoint%beta1, ios)
        if (ios == 0) call write_r(unit, "beta2", checkpoint%beta2, ios)
        if (ios == 0) call write_r(unit, "epsilon", checkpoint%epsilon, ios)
        if (ios == 0) call write_r(unit, "adafactor_decay", checkpoint%adafactor_decay, ios)
        if (ios == 0) call write_r(unit, "adafactor_clip_threshold", &
            checkpoint%adafactor_clip_threshold, ios)
        if (ios == 0) call write_l(unit, "adafactor_relative_step", &
            checkpoint%adafactor_relative_step, ios)
        if (ios == 0) call write_l(unit, "adafactor_scale_parameter", &
            checkpoint%adafactor_scale_parameter, ios)
        if (ios == 0) call write_l(unit, "adafactor_factored", &
            checkpoint%adafactor_factored, ios)
        if (ios == 0) call write_i(unit, "n_adafactor_blocks", &
            checkpoint%n_adafactor_blocks, ios)
        if (ios == 0) call write_r(unit, "rmsprop_decay", checkpoint%rmsprop_decay, ios)
        if (ios == 0) call write_r(unit, "rmsprop_momentum", &
            checkpoint%rmsprop_momentum, ios)
        if (ios == 0) call write_l(unit, "rmsprop_centered", &
            checkpoint%rmsprop_centered, ios)
        if (ios == 0) call write_r(unit, "momentum", checkpoint%momentum, ios)
        if (ios == 0) call write_l(unit, "nesterov", checkpoint%nesterov, ios)
        if (ios == 0) call write_r(unit, "weight_decay", checkpoint%weight_decay, ios)
        if (ios == 0) call write_r(unit, "l2", checkpoint%l2, ios)
        if (ios == 0) call write_r(unit, "tolerance", checkpoint%tolerance, ios)
        if (ios == 0) call write_r(unit, "min_delta", checkpoint%min_delta, ios)
        if (ios == 0) call write_r(unit, "gradient_clip_norm", &
            checkpoint%gradient_clip_norm, ios)
        if (ios == 0) call write_r(unit, "ema_decay", checkpoint%ema_decay, ios)
        if (ios == 0) call write_r(unit, "last_learning_rate", &
            checkpoint%last_learning_rate, ios)
        if (ios == 0) call write_r(unit, "initial_loss", checkpoint%initial_loss, ios)
        if (ios == 0) call write_r(unit, "final_loss", checkpoint%final_loss, ios)
        if (ios == 0) call write_r(unit, "best_loss", checkpoint%best_loss, ios)
        if (ios == 0) call write_r(unit, "initial_validation_loss", &
            checkpoint%initial_validation_loss, ios)
        if (ios == 0) call write_r(unit, "final_validation_loss", &
            checkpoint%final_validation_loss, ios)
        if (ios == 0) call write_r(unit, "best_validation_loss", &
            checkpoint%best_validation_loss, ios)
        if (ios == 0) call write_i(unit, "best_epoch", checkpoint%best_epoch, ios)
        if (ios == 0) call write_i(unit, "best_validation_epoch", &
            checkpoint%best_validation_epoch, ios)

        if (ios == 0) call write_r_array(unit, "parameters", checkpoint%parameters, ios)
        if (ios == 0) call write_optional_c_array(unit, "optimizer_group_name", &
            checkpoint%optimizer_group_name, ios)
        if (ios == 0) call write_optional_i_array(unit, "optimizer_group_first", &
            checkpoint%optimizer_group_first, ios)
        if (ios == 0) call write_optional_i_array(unit, "optimizer_group_last", &
            checkpoint%optimizer_group_last, ios)
        if (ios == 0) call write_optional_r_array(unit, &
            "optimizer_group_learning_rate_multiplier", &
            checkpoint%optimizer_group_learning_rate_multiplier, ios)
        if (ios == 0) call write_r_array(unit, "first_moment", checkpoint%first_moment, ios)
        if (ios == 0) call write_r_array(unit, "second_moment", checkpoint%second_moment, ios)
        if (ios == 0) call write_optional_i_array(unit, "adafactor_block_first", &
            checkpoint%adafactor_block_first, ios)
        if (ios == 0) call write_optional_i_array(unit, "adafactor_block_last", &
            checkpoint%adafactor_block_last, ios)
        if (ios == 0) call write_optional_i_array(unit, "adafactor_block_rows", &
            checkpoint%adafactor_block_rows, ios)
        if (ios == 0) call write_optional_i_array(unit, "adafactor_block_columns", &
            checkpoint%adafactor_block_columns, ios)
        if (ios == 0) call write_optional_i_array(unit, "adafactor_block_factored", &
            checkpoint%adafactor_block_factored, ios)
        if (ios == 0) call write_optional_r_array(unit, "adafactor_row_moment", &
            checkpoint%adafactor_row_moment, ios)
        if (ios == 0) call write_optional_r_array(unit, "adafactor_column_moment", &
            checkpoint%adafactor_column_moment, ios)
        if (ios == 0) call write_optional_r_array(unit, "adafactor_second_moment", &
            checkpoint%adafactor_second_moment, ios)
        if (ios == 0) call write_optional_r_array(unit, "max_second_moment", &
            checkpoint%max_second_moment, ios)
        if (ios == 0) call write_optional_r_array(unit, "rmsprop_buffer", &
            checkpoint%rmsprop_buffer, ios)
        if (ios == 0) call write_r_array(unit, "best_parameters", &
            checkpoint%best_parameters, ios)
        if (ios == 0) call write_r_array(unit, "accumulated_gradient", &
            checkpoint%accumulated_gradient, ios)
        if (ios == 0) call write_i_array(unit, "iterator_order", checkpoint%iterator_order, ios)
        if (ios == 0) call write_r_array(unit, "loss_history", checkpoint%loss_history, ios)
        if (ios == 0) call write_r_array(unit, "learning_rate_history", &
            checkpoint%learning_rate_history, ios)
        if (ios == 0) call write_optional_r_array(unit, "validation_loss_history", &
            checkpoint%validation_loss_history, ios)
        if (ios == 0) call write_optional_r_array(unit, "ema_parameters", &
            checkpoint%ema_parameters, ios)

        close_ios = 0
        close(unit, iostat=close_ios)
        if (ios /= 0 .or. close_ios /= 0) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP checkpoint save: formatted write failed")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_checkpoint_save

    subroutine mlp_checkpoint_load(checkpoint, path, status)
        !! Read and validate a portable checkpoint without partial mutation.
        type(mlp_training_checkpoint_t), intent(inout) :: checkpoint
        character(*), intent(in) :: path
        type(fortnum_status_t), intent(out) :: status
        type(mlp_training_checkpoint_t) :: candidate
        character(len=256) :: line
        character(len=64) :: key
        integer :: unit, ios, close_ios, schema

        open(newunit=unit, file=path, status="old", action="read", &
            form="formatted", access="sequential", iostat=ios)
        if (ios /= 0) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP checkpoint load: cannot open source")
            return
        end if

        read(unit, "(A)", iostat=ios) line
        if (ios /= 0 .or. trim(line) /= MLP_CHECKPOINT_MAGIC) goto 900
        call read_i(unit, "schema_version", schema, ios)
        if (ios /= 0 .or. schema /= MLP_CHECKPOINT_SCHEMA_VERSION) then
            close(unit)
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP checkpoint load: unsupported schema version")
            return
        end if

        call read_i(unit, "format_version", candidate%format_version, ios)
        if (ios == 0) call read_l(unit, "initialized", candidate%initialized, ios)
        if (ios == 0) call read_l(unit, "resume_safe", candidate%resume_safe, ios)
        if (ios == 0) call read_i(unit, "n_samples", candidate%n_samples, ios)
        if (ios == 0) call read_i(unit, "n_features", candidate%n_features, ios)
        if (ios == 0) call read_i(unit, "n_outputs", candidate%n_outputs, ios)
        if (ios == 0) call read_i(unit, "n_parameters", candidate%n_parameters, ios)
        if (ios == 0) call read_i(unit, "n_optimizer_groups", &
            candidate%n_optimizer_groups, ios)
        if (ios == 0) call read_i(unit, "epoch", candidate%epoch, ios)
        if (ios == 0) call read_i(unit, "updates", candidate%updates, ios)
        if (ios == 0) call read_i(unit, "microbatches", candidate%microbatches, ios)
        if (ios == 0) call read_i(unit, "microbatch_position", &
            candidate%microbatch_position, ios)
        if (ios == 0) call read_i(unit, "active_epoch", candidate%active_epoch, ios)
        if (ios == 0) call read_i(unit, "active_microbatches", &
            candidate%active_microbatches, ios)
        if (ios == 0) call read_i(unit, "accumulated_samples", &
            candidate%accumulated_samples, ios)
        if (ios == 0) call read_r(unit, "accumulated_weight_mass", &
            candidate%accumulated_weight_mass, ios)
        if (ios == 0) call read_i(unit, "iterator_epoch", candidate%iterator_epoch, ios)
        if (ios == 0) call read_i(unit, "iterator_position", &
            candidate%iterator_position, ios)
        if (ios == 0) call read_i(unit, "batch_size", candidate%batch_size, ios)
        if (ios == 0) call read_i(unit, "accumulation_steps", &
            candidate%accumulation_steps, ios)
        if (ios == 0) call read_i(unit, "shuffle_seed", candidate%shuffle_seed, ios)
        if (ios == 0) call read_i(unit, "adam_step_count", &
            candidate%adam_step_count, ios)
        if (ios == 0) call read_i(unit, "optimizer", candidate%optimizer, ios)
        if (ios == 0) call read_i(unit, "precision_kind", candidate%precision_kind, ios)
        if (ios == 0) call read_l(unit, "loss_scale_enabled", candidate%loss_scale%enabled, ios)
        if (ios == 0) call read_r(unit, "loss_scale_scale", candidate%loss_scale%scale, ios)
        if (ios == 0) call read_r(unit, "loss_scale_initial_scale", &
            candidate%loss_scale%initial_scale, ios)
        if (ios == 0) call read_r(unit, "loss_scale_growth_factor", &
            candidate%loss_scale%growth_factor, ios)
        if (ios == 0) call read_r(unit, "loss_scale_backoff_factor", &
            candidate%loss_scale%backoff_factor, ios)
        if (ios == 0) call read_r(unit, "loss_scale_minimum_scale", &
            candidate%loss_scale%minimum_scale, ios)
        if (ios == 0) call read_r(unit, "loss_scale_maximum_scale", &
            candidate%loss_scale%maximum_scale, ios)
        if (ios == 0) call read_i(unit, "loss_scale_growth_interval", &
            candidate%loss_scale%growth_interval, ios)
        if (ios == 0) call read_i(unit, "loss_scale_good_steps", &
            candidate%loss_scale%good_steps, ios)
        if (ios == 0) call read_i(unit, "loss_scale_overflow_count", &
            candidate%loss_scale%overflow_count, ios)
        if (ios == 0) call read_i(unit, "loss_scale_skipped_updates", &
            candidate%loss_scale%skipped_updates, ios)
        if (ios == 0) call read_i(unit, "stale_epochs", candidate%stale_epochs, ios)
        if (ios == 0) call read_i(unit, "schedule_bad_updates", &
            candidate%schedule_bad_updates, ios)
        if (ios == 0) call read_i(unit, "schedule_reductions", &
            candidate%schedule_reductions, ios)
        if (ios == 0) call read_i(unit, "gradient_clipped_updates", &
            candidate%gradient_clipped_updates, ios)
        if (ios == 0) call read_i(unit, "validation_interval", &
            candidate%validation_interval, ios)
        if (ios == 0) call read_i(unit, "patience", candidate%patience, ios)
        if (ios == 0) call read_l(unit, "shuffle", candidate%shuffle, ios)
        if (ios == 0) call read_l(unit, "has_validation", candidate%has_validation, ios)
        if (ios == 0) call read_l(unit, "converged", candidate%converged, ios)
        if (ios == 0) call read_l(unit, "early_stopped", candidate%early_stopped, ios)
        if (ios == 0) call read_l(unit, "restore_best", candidate%restore_best, ios)
        if (ios == 0) call read_l(unit, "has_typed_schedule", &
            candidate%has_typed_schedule, ios)
        if (ios == 0) call read_l(unit, "schedule_metric_initialized", &
            candidate%schedule_metric_initialized, ios)
        if (ios == 0) call read_i(unit, "typed_schedule_kind", &
            candidate%typed_schedule%kind, ios)
        if (ios == 0) call read_i(unit, "typed_schedule_warmup_updates", &
            candidate%typed_schedule%warmup_updates, ios)
        if (ios == 0) call read_i(unit, "typed_schedule_total_updates", &
            candidate%typed_schedule%total_updates, ios)
        if (ios == 0) call read_r(unit, "typed_schedule_min_rate_fraction", &
            candidate%typed_schedule%min_rate_fraction, ios)
        if (ios == 0) call read_r(unit, "typed_schedule_decay_factor", &
            candidate%typed_schedule%decay_factor, ios)
        if (ios == 0) call read_r(unit, "typed_schedule_peak_rate_fraction", &
            candidate%typed_schedule%peak_rate_fraction, ios)
        if (ios == 0) call read_r(unit, "typed_schedule_final_rate_fraction", &
            candidate%typed_schedule%final_rate_fraction, ios)
        if (ios == 0) call read_i(unit, "typed_schedule_metric_mode", &
            candidate%typed_schedule%metric_mode, ios)
        if (ios == 0) call read_i(unit, "typed_schedule_patience_updates", &
            candidate%typed_schedule%patience_updates, ios)
        if (ios == 0) call read_r(unit, "typed_schedule_min_delta", &
            candidate%typed_schedule%min_delta, ios)
        if (ios == 0) call read_r(unit, "typed_schedule_plateau_factor", &
            candidate%typed_schedule%plateau_factor, ios)
        if (ios == 0) call read_i8(unit, "shuffle_state", candidate%shuffle_state, ios)
        if (ios == 0) call read_r(unit, "schedule_best_metric", &
            candidate%schedule_best_metric, ios)
        if (ios == 0) call read_r(unit, "learning_rate", candidate%learning_rate, ios)
        if (ios == 0) call read_r(unit, "beta1", candidate%beta1, ios)
        if (ios == 0) call read_r(unit, "beta2", candidate%beta2, ios)
        if (ios == 0) call read_r(unit, "epsilon", candidate%epsilon, ios)
        if (ios == 0) call read_r(unit, "adafactor_decay", candidate%adafactor_decay, ios)
        if (ios == 0) call read_r(unit, "adafactor_clip_threshold", &
            candidate%adafactor_clip_threshold, ios)
        if (ios == 0) call read_l(unit, "adafactor_relative_step", &
            candidate%adafactor_relative_step, ios)
        if (ios == 0) call read_l(unit, "adafactor_scale_parameter", &
            candidate%adafactor_scale_parameter, ios)
        if (ios == 0) call read_l(unit, "adafactor_factored", &
            candidate%adafactor_factored, ios)
        if (ios == 0) call read_i(unit, "n_adafactor_blocks", &
            candidate%n_adafactor_blocks, ios)
        if (ios == 0) call read_r(unit, "rmsprop_decay", candidate%rmsprop_decay, ios)
        if (ios == 0) call read_r(unit, "rmsprop_momentum", &
            candidate%rmsprop_momentum, ios)
        if (ios == 0) call read_l(unit, "rmsprop_centered", &
            candidate%rmsprop_centered, ios)
        if (ios == 0) call read_r(unit, "momentum", candidate%momentum, ios)
        if (ios == 0) call read_l(unit, "nesterov", candidate%nesterov, ios)
        if (ios == 0) call read_r(unit, "weight_decay", candidate%weight_decay, ios)
        if (ios == 0) call read_r(unit, "l2", candidate%l2, ios)
        if (ios == 0) call read_r(unit, "tolerance", candidate%tolerance, ios)
        if (ios == 0) call read_r(unit, "min_delta", candidate%min_delta, ios)
        if (ios == 0) call read_r(unit, "gradient_clip_norm", &
            candidate%gradient_clip_norm, ios)
        if (ios == 0) call read_r(unit, "ema_decay", candidate%ema_decay, ios)
        if (ios == 0) call read_r(unit, "last_learning_rate", &
            candidate%last_learning_rate, ios)
        if (ios == 0) call read_r(unit, "initial_loss", candidate%initial_loss, ios)
        if (ios == 0) call read_r(unit, "final_loss", candidate%final_loss, ios)
        if (ios == 0) call read_r(unit, "best_loss", candidate%best_loss, ios)
        if (ios == 0) call read_r(unit, "initial_validation_loss", &
            candidate%initial_validation_loss, ios)
        if (ios == 0) call read_r(unit, "final_validation_loss", &
            candidate%final_validation_loss, ios)
        if (ios == 0) call read_r(unit, "best_validation_loss", &
            candidate%best_validation_loss, ios)
        if (ios == 0) call read_i(unit, "best_epoch", candidate%best_epoch, ios)
        if (ios == 0) call read_i(unit, "best_validation_epoch", &
            candidate%best_validation_epoch, ios)
        if (ios /= 0) goto 900

        call read_r_array(unit, "parameters_count", "parameters_item", &
            candidate%n_parameters, candidate%parameters, ios)
        if (ios == 0) call read_optional_c_array(unit, "optimizer_group_name_present", &
            "optimizer_group_name_count", "optimizer_group_name_item", &
            candidate%n_optimizer_groups, candidate%optimizer_group_name, ios)
        if (ios == 0) call read_optional_i_array(unit, "optimizer_group_first_present", &
            "optimizer_group_first_count", "optimizer_group_first_item", &
            candidate%n_optimizer_groups, candidate%optimizer_group_first, ios)
        if (ios == 0) call read_optional_i_array(unit, "optimizer_group_last_present", &
            "optimizer_group_last_count", "optimizer_group_last_item", &
            candidate%n_optimizer_groups, candidate%optimizer_group_last, ios)
        if (ios == 0) call read_optional_r_array(unit, &
            "optimizer_group_learning_rate_multiplier_present", &
            "optimizer_group_learning_rate_multiplier_count", &
            "optimizer_group_learning_rate_multiplier_item", &
            candidate%n_optimizer_groups, &
            candidate%optimizer_group_learning_rate_multiplier, ios)
        if (ios == 0) call read_r_array(unit, "first_moment_count", &
            "first_moment_item", candidate%n_parameters, candidate%first_moment, ios)
        if (ios == 0) call read_r_array(unit, "second_moment_count", &
            "second_moment_item", candidate%n_parameters, candidate%second_moment, ios)
        if (ios == 0) call read_optional_i_array(unit, "adafactor_block_first_present", &
            "adafactor_block_first_count", "adafactor_block_first_item", &
            candidate%n_adafactor_blocks, candidate%adafactor_block_first, ios)
        if (ios == 0) call read_optional_i_array(unit, "adafactor_block_last_present", &
            "adafactor_block_last_count", "adafactor_block_last_item", &
            candidate%n_adafactor_blocks, candidate%adafactor_block_last, ios)
        if (ios == 0) call read_optional_i_array(unit, "adafactor_block_rows_present", &
            "adafactor_block_rows_count", "adafactor_block_rows_item", &
            candidate%n_adafactor_blocks, candidate%adafactor_block_rows, ios)
        if (ios == 0) call read_optional_i_array(unit, "adafactor_block_columns_present", &
            "adafactor_block_columns_count", "adafactor_block_columns_item", &
            candidate%n_adafactor_blocks, candidate%adafactor_block_columns, ios)
        if (ios == 0) call read_optional_i_array(unit, "adafactor_block_factored_present", &
            "adafactor_block_factored_count", "adafactor_block_factored_item", &
            candidate%n_adafactor_blocks, candidate%adafactor_block_factored, ios)
        if (ios == 0) call read_optional_r_array(unit, "adafactor_row_moment_present", &
            "adafactor_row_moment_count", "adafactor_row_moment_item", -1, &
            candidate%adafactor_row_moment, ios)
        if (ios == 0) call read_optional_r_array(unit, "adafactor_column_moment_present", &
            "adafactor_column_moment_count", "adafactor_column_moment_item", -1, &
            candidate%adafactor_column_moment, ios)
        if (ios == 0) call read_optional_r_array(unit, "adafactor_second_moment_present", &
            "adafactor_second_moment_count", "adafactor_second_moment_item", -1, &
            candidate%adafactor_second_moment, ios)
        if (ios == 0) call read_optional_r_array(unit, "max_second_moment_present", &
            "max_second_moment_count", "max_second_moment_item", &
            candidate%n_parameters, candidate%max_second_moment, ios)
        if (ios == 0) call read_optional_r_array(unit, "rmsprop_buffer_present", &
            "rmsprop_buffer_count", "rmsprop_buffer_item", &
            candidate%n_parameters, candidate%rmsprop_buffer, ios)
        if (ios == 0) call read_r_array(unit, "best_parameters_count", &
            "best_parameters_item", candidate%n_parameters, candidate%best_parameters, ios)
        if (ios == 0) call read_r_array(unit, "accumulated_gradient_count", &
            "accumulated_gradient_item", candidate%n_parameters, &
            candidate%accumulated_gradient, ios)
        if (ios == 0) call read_i_array(unit, "iterator_order_count", &
            "iterator_order_item", candidate%n_samples, candidate%iterator_order, ios)
        if (ios == 0) call read_r_array(unit, "loss_history_count", "loss_history_item", &
            candidate%epoch, candidate%loss_history, ios)
        if (ios == 0) call read_r_array(unit, "learning_rate_history_count", &
            "learning_rate_history_item", candidate%epoch, &
            candidate%learning_rate_history, ios)
        if (ios == 0) call read_optional_r_array(unit, &
            "validation_loss_history_present", "validation_loss_history_count", &
            "validation_loss_history_item", candidate%epoch, &
            candidate%validation_loss_history, ios)
        if (ios == 0) call read_optional_r_array(unit, &
            "ema_parameters_present", "ema_parameters_count", &
            "ema_parameters_item", candidate%n_parameters, &
            candidate%ema_parameters, ios)
        if (ios /= 0) goto 900

        read(unit, "(A)", iostat=ios) line
        if (ios == 0) goto 900
        if (ios /= iostat_end) goto 900
        close_ios = 0
        close(unit, iostat=close_ios)
        if (close_ios /= 0 .or. .not. candidate%valid()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP checkpoint load: malformed, truncated, or invalid snapshot")
            return
        end if

        call checkpoint%clear()
        checkpoint = candidate
        call status_set(status, FORTNUM_OK, "")
        return

        900   continue
        close_ios = 0
        close(unit, iostat=close_ios)
        call status_set(status, FORTNUM_DOMAIN_ERROR, &
            "MLP checkpoint load: malformed, truncated, or invalid snapshot")
    end subroutine mlp_checkpoint_load

    subroutine fingerprint_integer(hash, label, value)
        integer(int64), intent(inout) :: hash
        character(*), intent(in) :: label
        integer, intent(in) :: value
        character(len=64) :: token

        write(token, '(I0)') value
        call fingerprint_token(hash, label)
        call fingerprint_token(hash, trim(token))
    end subroutine fingerprint_integer

    subroutine fingerprint_int64(hash, label, value)
        integer(int64), intent(inout) :: hash
        character(*), intent(in) :: label
        integer(int64), intent(in) :: value
        character(len=64) :: token

        write(token, '(I0)') value
        call fingerprint_token(hash, label)
        call fingerprint_token(hash, trim(token))
    end subroutine fingerprint_int64

    subroutine fingerprint_logical(hash, label, value)
        integer(int64), intent(inout) :: hash
        character(*), intent(in) :: label
        logical, intent(in) :: value

        call fingerprint_integer(hash, label, merge(1, 0, value))
    end subroutine fingerprint_logical

    subroutine fingerprint_real(hash, label, value)
        integer(int64), intent(inout) :: hash
        character(*), intent(in) :: label
        real(dp), intent(in) :: value
        character(len=64) :: token

        write(token, '(ES26.17E3)') value
        call fingerprint_token(hash, label)
        call fingerprint_token(hash, trim(token))
    end subroutine fingerprint_real

    subroutine fingerprint_token(hash, value)
        integer(int64), intent(inout) :: hash
        character(*), intent(in) :: value
        integer :: i

        do i = 1, len_trim(value)
            hash = modulo(MLP_CHECKPOINT_FINGERPRINT_BASE*hash + &
                int(iachar(value(i:i)), int64), MLP_CHECKPOINT_FINGERPRINT_MODULUS)
        end do
        hash = modulo(MLP_CHECKPOINT_FINGERPRINT_BASE*hash + 10_int64, &
            MLP_CHECKPOINT_FINGERPRINT_MODULUS)
    end subroutine fingerprint_token

    subroutine fingerprint_real_array(hash, label, values)
        integer(int64), intent(inout) :: hash
        character(*), intent(in) :: label
        real(dp), intent(in) :: values(:)
        integer :: i

        call fingerprint_integer(hash, label//"_count", size(values))
        do i = 1, size(values)
            call fingerprint_real(hash, label//"_item", values(i))
        end do
    end subroutine fingerprint_real_array

    subroutine fingerprint_integer_array(hash, label, values)
        integer(int64), intent(inout) :: hash
        character(*), intent(in) :: label
        integer, intent(in) :: values(:)
        integer :: i

        call fingerprint_integer(hash, label//"_count", size(values))
        do i = 1, size(values)
            call fingerprint_integer(hash, label//"_item", values(i))
        end do
    end subroutine fingerprint_integer_array

    subroutine fingerprint_character_array(hash, label, values)
        integer(int64), intent(inout) :: hash
        character(*), intent(in) :: label
        character(len=64), intent(in) :: values(:)
        integer :: i

        call fingerprint_integer(hash, label//"_count", size(values))
        do i = 1, size(values)
            call fingerprint_token(hash, label//"_item")
            call fingerprint_token(hash, trim(values(i)))
        end do
    end subroutine fingerprint_character_array

    subroutine fingerprint_optional_integer_array(hash, label, values)
        integer(int64), intent(inout) :: hash
        character(*), intent(in) :: label
        integer, allocatable, intent(in) :: values(:)

        call fingerprint_logical(hash, label//"_present", allocated(values))
        if (allocated(values)) call fingerprint_integer_array(hash, label, values)
    end subroutine fingerprint_optional_integer_array

    subroutine fingerprint_optional_real_array(hash, label, values)
        integer(int64), intent(inout) :: hash
        character(*), intent(in) :: label
        real(dp), allocatable, intent(in) :: values(:)

        call fingerprint_logical(hash, label//"_present", allocated(values))
        if (allocated(values)) call fingerprint_real_array(hash, label, values)
    end subroutine fingerprint_optional_real_array

    subroutine fingerprint_optional_character_array(hash, label, values)
        integer(int64), intent(inout) :: hash
        character(*), intent(in) :: label
        character(len=64), allocatable, intent(in) :: values(:)

        call fingerprint_logical(hash, label//"_present", allocated(values))
        if (allocated(values)) call fingerprint_character_array(hash, label, values)
    end subroutine fingerprint_optional_character_array

    subroutine write_i(unit, key, value, ios)
        integer, intent(in) :: unit, value
        character(*), intent(in) :: key
        integer, intent(out) :: ios

        write(unit, "(A,1X,I0)", iostat=ios) trim(key), value
    end subroutine write_i

    subroutine write_i8(unit, key, value, ios)
        integer, intent(in) :: unit
        integer(int64), intent(in) :: value
        character(*), intent(in) :: key
        integer, intent(out) :: ios

        write(unit, "(A,1X,I0)", iostat=ios) trim(key), value
    end subroutine write_i8

    subroutine write_l(unit, key, value, ios)
        integer, intent(in) :: unit
        logical, intent(in) :: value
        character(*), intent(in) :: key
        integer, intent(out) :: ios

        write(unit, "(A,1X,I0)", iostat=ios) trim(key), merge(1, 0, value)
    end subroutine write_l

    subroutine write_r(unit, key, value, ios)
        integer, intent(in) :: unit
        real(dp), intent(in) :: value
        character(*), intent(in) :: key
        integer, intent(out) :: ios

        write(unit, "(A,1X,ES26.17E3)", iostat=ios) trim(key), value
    end subroutine write_r

    subroutine write_c(unit, key, value, ios)
        integer, intent(in) :: unit
        character(*), intent(in) :: key, value
        integer, intent(out) :: ios

        write(unit, '(A,1X,A)', iostat=ios) trim(key), '"' // trim(value) // '"'
    end subroutine write_c

    subroutine write_r_array(unit, key, values, ios)
        integer, intent(in) :: unit
        real(dp), intent(in) :: values(:)
        character(*), intent(in) :: key
        integer, intent(out) :: ios
        integer :: i
        character(len=80) :: count_key, item_key

        write(count_key, '(A,"_count")') trim(key)
        write(item_key, '(A,"_item")') trim(key)
        call write_i(unit, trim(count_key), size(values), ios)
        do i = 1, size(values)
            if (ios /= 0) return
            call write_r(unit, trim(item_key), values(i), ios)
        end do
    end subroutine write_r_array

    subroutine write_i_array(unit, key, values, ios)
        integer, intent(in) :: unit
        integer, intent(in) :: values(:)
        character(*), intent(in) :: key
        integer, intent(out) :: ios
        integer :: i
        character(len=80) :: count_key, item_key

        write(count_key, '(A,"_count")') trim(key)
        write(item_key, '(A,"_item")') trim(key)
        call write_i(unit, trim(count_key), size(values), ios)
        do i = 1, size(values)
            if (ios /= 0) return
            call write_i(unit, trim(item_key), values(i), ios)
        end do
    end subroutine write_i_array

    subroutine write_optional_i_array(unit, key, values, ios)
        integer, intent(in) :: unit
        integer, allocatable, intent(in) :: values(:)
        character(*), intent(in) :: key
        integer, intent(out) :: ios
        character(len=80) :: present_key

        write(present_key, '(A,"_present")') trim(key)
        call write_l(unit, trim(present_key), allocated(values), ios)
        if (ios /= 0 .or. .not. allocated(values)) return
        call write_i_array(unit, key, values, ios)
    end subroutine write_optional_i_array

    subroutine write_optional_c_array(unit, key, values, ios)
        integer, intent(in) :: unit
        character(len=64), allocatable, intent(in) :: values(:)
        character(*), intent(in) :: key
        integer, intent(out) :: ios
        character(len=80) :: present_key
        integer :: i

        write(present_key, '(A,"_present")') trim(key)
        call write_l(unit, trim(present_key), allocated(values), ios)
        if (ios /= 0 .or. .not. allocated(values)) return
        call write_i(unit, trim(key)//"_count", size(values), ios)
        do i = 1, size(values)
            if (ios /= 0) return
            call write_c(unit, trim(key)//"_item", values(i), ios)
        end do
    end subroutine write_optional_c_array

    subroutine write_optional_r_array(unit, key, values, ios)
        integer, intent(in) :: unit
        real(dp), allocatable, intent(in) :: values(:)
        character(*), intent(in) :: key
        integer, intent(out) :: ios
        integer :: i
        character(len=80) :: present_key

        write(present_key, '(A,"_present")') trim(key)
        call write_l(unit, trim(present_key), allocated(values), ios)
        if (ios /= 0 .or. .not. allocated(values)) return
        call write_r_array(unit, key, values, ios)
    end subroutine write_optional_r_array

    subroutine read_i(unit, expected, value, ios)
        integer, intent(in) :: unit
        character(*), intent(in) :: expected
        integer, intent(out) :: value
        integer, intent(out) :: ios
        character(len=80) :: key

        read(unit, *, iostat=ios) key, value
        if (ios == 0 .and. trim(key) /= trim(expected)) ios = 1
    end subroutine read_i

    subroutine read_i8(unit, expected, value, ios)
        integer, intent(in) :: unit
        character(*), intent(in) :: expected
        integer(int64), intent(out) :: value
        integer, intent(out) :: ios
        character(len=80) :: key

        read(unit, *, iostat=ios) key, value
        if (ios == 0 .and. trim(key) /= trim(expected)) ios = 1
    end subroutine read_i8

    subroutine read_l(unit, expected, value, ios)
        integer, intent(in) :: unit
        character(*), intent(in) :: expected
        logical, intent(out) :: value
        integer, intent(out) :: ios
        character(len=80) :: key
        integer :: encoded

        encoded = 0
        read(unit, *, iostat=ios) key, encoded
        if (ios == 0 .and. trim(key) /= trim(expected)) ios = 1
        if (ios == 0 .and. encoded /= 0 .and. encoded /= 1) ios = 1
        value = encoded == 1
    end subroutine read_l

    subroutine read_r(unit, expected, value, ios)
        integer, intent(in) :: unit
        character(*), intent(in) :: expected
        real(dp), intent(out) :: value
        integer, intent(out) :: ios
        character(len=80) :: key

        read(unit, *, iostat=ios) key, value
        if (ios == 0 .and. trim(key) /= trim(expected)) ios = 1
    end subroutine read_r

    subroutine read_c(unit, expected, value, ios)
        integer, intent(in) :: unit
        character(*), intent(in) :: expected
        character(*), intent(out) :: value
        integer, intent(out) :: ios
        character(len=80) :: key

        read(unit, *, iostat=ios) key, value
        if (ios == 0 .and. trim(key) /= trim(expected)) ios = 1
    end subroutine read_c

    subroutine read_r_array(unit, count_key, item_key, expected_count, values, ios)
        integer, intent(in) :: unit, expected_count
        character(*), intent(in) :: count_key, item_key
        real(dp), allocatable, intent(out) :: values(:)
        integer, intent(out) :: ios
        integer :: count, i, alloc_status

        call read_i(unit, count_key, count, ios)
        if (ios /= 0 .or. count < 0 .or. &
            (expected_count >= 0 .and. count /= expected_count)) then
            ios = 1
            return
        end if
        allocate(values(count), stat=alloc_status)
        if (alloc_status /= 0) then
            ios = 1
            return
        end if
        do i = 1, count
            call read_r(unit, item_key, values(i), ios)
            if (ios /= 0) return
        end do
    end subroutine read_r_array

    subroutine read_i_array(unit, count_key, item_key, expected_count, values, ios)
        integer, intent(in) :: unit, expected_count
        character(*), intent(in) :: count_key, item_key
        integer, allocatable, intent(out) :: values(:)
        integer, intent(out) :: ios
        integer :: count, i, alloc_status

        call read_i(unit, count_key, count, ios)
        if (ios /= 0 .or. count < 0 .or. &
            (expected_count >= 0 .and. count /= expected_count)) then
            ios = 1
            return
        end if
        allocate(values(count), stat=alloc_status)
        if (alloc_status /= 0) then
            ios = 1
            return
        end if
        do i = 1, count
            call read_i(unit, item_key, values(i), ios)
            if (ios /= 0) return
        end do
    end subroutine read_i_array

    subroutine read_optional_i_array(unit, present_key, count_key, item_key, &
            expected_count, values, ios)
        integer, intent(in) :: unit, expected_count
        character(*), intent(in) :: present_key, count_key, item_key
        integer, allocatable, intent(out) :: values(:)
        integer, intent(out) :: ios
        logical :: present

        call read_l(unit, present_key, present, ios)
        if (ios /= 0) return
        if (.not. present) then
            if (allocated(values)) deallocate(values)
            return
        end if
        call read_i_array(unit, count_key, item_key, expected_count, values, ios)
    end subroutine read_optional_i_array

    subroutine read_optional_c_array(unit, present_key, count_key, item_key, &
            expected_count, values, ios)
        integer, intent(in) :: unit, expected_count
        character(*), intent(in) :: present_key, count_key, item_key
        character(len=64), allocatable, intent(out) :: values(:)
        integer, intent(out) :: ios
        logical :: present
        integer :: count, i, alloc_status

        call read_l(unit, present_key, present, ios)
        if (ios /= 0) return
        if (.not. present) then
            if (allocated(values)) deallocate(values)
            return
        end if
        call read_i(unit, count_key, count, ios)
        if (ios /= 0 .or. count < 0 .or. &
            (expected_count >= 0 .and. count /= expected_count)) then
            ios = 1
            return
        end if
        allocate(values(count), stat=alloc_status)
        if (alloc_status /= 0) then
            ios = 1
            return
        end if
        do i = 1, count
            call read_c(unit, item_key, values(i), ios)
            if (ios /= 0) return
        end do
    end subroutine read_optional_c_array

    subroutine read_optional_r_array(unit, present_key, count_key, item_key, &
            expected_count, values, ios)
        integer, intent(in) :: unit, expected_count
        character(*), intent(in) :: present_key, count_key, item_key
        real(dp), allocatable, intent(out) :: values(:)
        integer, intent(out) :: ios
        logical :: present

        call read_l(unit, present_key, present, ios)
        if (ios /= 0) return
        if (.not. present) then
            if (allocated(values)) deallocate(values)
            return
        end if
        call read_r_array(unit, count_key, item_key, expected_count, values, ios)
    end subroutine read_optional_r_array

end module fortml_mlp_checkpoint
