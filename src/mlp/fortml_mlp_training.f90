module fortml_mlp_training
    !! Deterministic, optimizer-facing training products for dense MLPs.
    !!
    !! `mlp_train` deliberately keeps the data contract explicit: rows are
    !! samples, columns are features (or outputs), and the objective is the
    !! mean squared error with an optional L2 penalty.  The same loss product
    !! is used for full-batch diagnostics and every mini-batch update, so the
    !! reported gradient is the derivative of the reported objective.  The
    !! scalar L2 derivative is returned as a first-class hyperparameter
    !! product, which lets an outer optimizer differentiate this objective
    !! without finite-difference plumbing.
    use, intrinsic :: iso_fortran_env, only: int64, real32
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED, FORTNUM_CONVERGENCE_ERROR
    use fortml_mlp, only: mlp_t, mlp_parameter_block_t
    use fortml_mlp_schedules, only: mlp_learning_rate_schedule_t, &
        MLP_SCHEDULE_PLATEAU
    use fortml_losses, only: weighted_mse_loss_value, weighted_mse_loss_vjp, &
        weighted_mse_loss_hvp
    use fortopt_objective, only: objective_t
    use fortopt_adam, only: adam_t
    use fortopt_adamw, only: adamw_t
    use fortopt_adagrad, only: adagrad_t
    use fortopt_rmsprop, only: rmsprop_t
    use fortopt_sgd, only: sgd_t
    use fortml_adafactor, only: adafactor_t
    use fortml_adafactor_factored, only: adafactor_factored_t, &
        adafactor_block_spec_t
    use fortml_amsgrad, only: amsgrad_t
    use fortml_radam, only: radam_t
    use fortopt_lbfgsb, only: lbfgsb_t, lbfgsb_options_t, lbfgsb_result_t
    implicit none
    private

    integer, parameter, public :: MLP_REDUCTION_MEAN = 1
    integer, parameter, public :: MLP_REDUCTION_SUM = 2
    integer, parameter, public :: MLP_OPTIMIZER_ADAM = 1
    integer, parameter, public :: MLP_OPTIMIZER_SGD = 2
    integer, parameter, public :: MLP_OPTIMIZER_ADAMW = 3
    integer, parameter, public :: MLP_OPTIMIZER_ADAGRAD = 4
    integer, parameter, public :: MLP_OPTIMIZER_RMSPROP = 5
    integer, parameter, public :: MLP_OPTIMIZER_ADAFACTOR = 6
    integer, parameter, public :: MLP_OPTIMIZER_AMSGRAD = 7
    integer, parameter, public :: MLP_OPTIMIZER_RADAM = 8
    integer, parameter, public :: MLP_OPTIMIZER_LION = 9

    ! Precision is a first-class training contract.  FP64 is the deterministic
    ! reference path.  FP32 uses DP master parameters and a single-precision
    ! model/data boundary with transactional loss scaling.  FP16 and BF16 are
    ! named so callers receive a typed refusal until their storage and kernels
    ! are available.
    integer, parameter, public :: MLP_PRECISION_FP64 = 1
    integer, parameter, public :: MLP_PRECISION_FP32 = 2
    integer, parameter, public :: MLP_PRECISION_FP16 = 3
    integer, parameter, public :: MLP_PRECISION_BF16 = 4

    type, public :: mlp_loss_scale_state_t
        !! Explicit automatic-mixed-precision loss-scale state.
        !!
        !! The state is deliberately independent of a device backend.  It is
        !! used by both the FP64 reference path and the CPU FP32 master-weight
        !! path.  Lower-precision resident kernels retain a typed refusal until
        !! their storage and execution contracts are present.
        logical :: enabled = .false.
        real(dp) :: scale = 1.0_dp
        real(dp) :: initial_scale = 1.0_dp
        real(dp) :: growth_factor = 2.0_dp
        real(dp) :: backoff_factor = 0.5_dp
        real(dp) :: minimum_scale = 1.0_dp
        real(dp) :: maximum_scale = 16777216.0_dp
        integer :: growth_interval = 2000
        integer :: good_steps = 0
        integer :: overflow_count = 0
        integer :: skipped_updates = 0
    contains
        procedure, public :: initialize => loss_scale_initialize
        procedure, public :: observe => loss_scale_observe
        procedure, public :: scale_gradient => loss_scale_scale_gradient
        procedure, public :: unscale_gradient => loss_scale_unscale_gradient
        procedure, public :: scaled_gradient_finite => loss_scale_scaled_gradient_finite
        procedure, public :: valid => loss_scale_valid
        procedure, public :: compatible => loss_scale_compatible
    end type mlp_loss_scale_state_t

    integer, parameter, public :: MLP_EVENT_TRAIN_BEGIN = 1
    integer, parameter, public :: MLP_EVENT_UPDATE = 2
    integer, parameter, public :: MLP_EVENT_VALIDATION = 3
    integer, parameter, public :: MLP_EVENT_EPOCH_END = 4
    integer, parameter, public :: MLP_EVENT_CHECKPOINT = 5
    integer, parameter, public :: MLP_EVENT_TRAIN_END = 6

    abstract interface
        subroutine mlp_epoch_callback_proc(epoch, loss, gradient_norm, stop)
            import :: dp
            integer, intent(in) :: epoch
            real(dp), intent(in) :: loss, gradient_norm
            logical, intent(out) :: stop
        end subroutine mlp_epoch_callback_proc

        subroutine mlp_learning_rate_schedule_proc(epoch, update, base_rate, rate)
            import :: dp
            integer, intent(in) :: epoch, update
            real(dp), intent(in) :: base_rate
            real(dp), intent(out) :: rate
        end subroutine mlp_learning_rate_schedule_proc

        subroutine mlp_training_event_proc(event, epoch, update, loss, &
                validation_loss, gradient_norm, learning_rate, stop, status)
            import :: dp, fortnum_status_t
            integer, intent(in) :: event, epoch, update
            real(dp), intent(in) :: loss, validation_loss, gradient_norm
            real(dp), intent(in) :: learning_rate
            logical, intent(out) :: stop
            type(fortnum_status_t), intent(out) :: status
        end subroutine mlp_training_event_proc
    end interface

    type, public :: mlp_optimizer_group_t
        !! A named, contiguous parameter slice with its own update scale.
        !!
        !! The multiplier is applied to the complete optimizer delta after
        !! the optimizer updates its shared moments.  This gives every
        !! optimizer the same deterministic group contract (including
        !! AdamW's decoupled decay) without maintaining a second hidden
        !! optimizer state per group.  Parameters not covered by a group use
        !! multiplier one.
        character(len=64) :: name = ""
        integer :: first = 0
        integer :: last = -1
        real(dp) :: learning_rate_multiplier = 1.0_dp
    contains
        procedure, public :: initialize => mlp_optimizer_group_initialize
        procedure, public :: size => mlp_optimizer_group_size
        procedure, public :: initialized => mlp_optimizer_group_initialized
    end type mlp_optimizer_group_t

    type, public :: mlp_training_options_t
        integer :: max_epochs = 1000
        integer :: batch_size = 0
        integer :: accumulation_steps = 1
        integer :: validation_interval = 1
        integer :: patience = 0
        integer :: shuffle_seed = 17
        logical :: shuffle = .false.
        logical :: restore_best = .true.
        real(dp) :: learning_rate = 1.0e-3_dp
        real(dp) :: beta1 = 0.9_dp
        real(dp) :: beta2 = 0.999_dp
        real(dp) :: epsilon = 1.0e-8_dp
        real(dp) :: adafactor_decay = 0.999_dp
        real(dp) :: adafactor_clip_threshold = 1.0_dp
        logical :: adafactor_relative_step = .false.
        logical :: adafactor_scale_parameter = .false.
        !! Use row/column factored state for matrix-shaped MLP weight blocks.
        !! Checkpoint/resume is a typed refusal until its ragged factor state
        !! is serialized; the default vector recurrence remains resumable.
        logical :: adafactor_factored = .false.
        real(dp) :: rmsprop_decay = 0.99_dp
        real(dp) :: rmsprop_momentum = 0.0_dp
        logical :: rmsprop_centered = .false.
        integer :: optimizer = MLP_OPTIMIZER_ADAM
        integer :: precision_kind = MLP_PRECISION_FP64
        !! Loss scaling is explicit for FP64 and FP32.  FP32 keeps the
        !! optimizer and checkpoint parameters in binary64 while the model
        !! boundary is rounded through `real(real32)`.  FP16/BF16 remain typed
        !! refusals until their storage and kernels are available.
        type(mlp_loss_scale_state_t) :: loss_scale
        real(dp) :: momentum = 0.0_dp
        logical :: nesterov = .false.
        real(dp) :: weight_decay = 0.0_dp
        real(dp) :: l2 = 0.0_dp
        real(dp) :: tolerance = 1.0e-8_dp
        real(dp) :: min_delta = 0.0_dp
        real(dp) :: gradient_clip_norm = 0.0_dp
        !! Exponential moving average of post-update parameters.  A value of
        !! zero disables EMA; otherwise the decay must lie in [0,1).
        real(dp) :: ema_decay = 0.0_dp
        !! Optional non-overlapping parameter groups.  A group's multiplier
        !! scales its post-optimizer update; omitted groups use one.
        type(mlp_optimizer_group_t), allocatable :: optimizer_groups(:)
        procedure(mlp_epoch_callback_proc), pointer, nopass :: callback => null()
        procedure(mlp_learning_rate_schedule_proc), pointer, nopass :: &
            learning_rate_schedule => null()
        !! A typed schedule is stateless and can be persisted in a checkpoint.
        !! Set `use_typed_schedule` to select it; a custom callback and typed
        !! schedule cannot be active simultaneously.
        logical :: use_typed_schedule = .false.
        type(mlp_learning_rate_schedule_t) :: typed_schedule
        procedure(mlp_training_event_proc), pointer, nopass :: &
            event_callback => null()
    end type mlp_training_options_t

    type, public :: mlp_training_state_t
        integer :: epochs = 0
        integer :: updates = 0
        integer :: microbatches = 0
        integer :: accumulation_steps = 1
        integer :: precision_kind = MLP_PRECISION_FP64
        type(mlp_loss_scale_state_t) :: loss_scale
        integer :: best_epoch = 0
        integer :: best_validation_epoch = 0
        integer :: schedule_bad_updates = 0
        integer :: schedule_reductions = 0
        logical :: converged = .false.
        logical :: early_stopped = .false.
        logical :: schedule_metric_initialized = .false.
        integer :: gradient_clipped_updates = 0
        real(dp) :: initial_loss = huge(1.0_dp)
        real(dp) :: final_loss = huge(1.0_dp)
        real(dp) :: best_loss = huge(1.0_dp)
        real(dp) :: initial_validation_loss = huge(1.0_dp)
        real(dp) :: final_validation_loss = huge(1.0_dp)
        real(dp) :: best_validation_loss = huge(1.0_dp)
        real(dp) :: schedule_best_metric = huge(1.0_dp)
        real(dp) :: gradient_norm = huge(1.0_dp)
        real(dp) :: last_learning_rate = 0.0_dp
        logical :: has_ema = .false.
        real(dp), allocatable :: ema_parameters(:)
        real(dp), allocatable :: loss_history(:)
        real(dp), allocatable :: learning_rate_history(:)
        real(dp), allocatable :: validation_loss_history(:)
    end type mlp_training_state_t

    type, public :: mlp_training_checkpoint_t
        !! Complete in-memory optimizer training snapshot.
        !!
        !! A checkpoint is produced at every completed epoch when it is passed
        !! to `mlp_train`.  Calling `mlp_train` again with the same initialized
        !! checkpoint resumes at `epoch + 1`; `max_epochs` is interpreted as
        !! the total target epoch, so a resumed call may increase it.  The
        !! model parameters, Adam moments and step counter, local shuffle
        !! stream, exact permutation cursor, schedule position, validation and
        !! best-state bookkeeping are all copied. A typed schedule is copied
        !! with its structural and continuous fields. Procedure pointers
        !! (custom schedules and callbacks) are intentionally not copied: the
        !! caller must install deterministic procedures again on resumed options.
        integer :: format_version = 11
        logical :: initialized = .false.
        logical :: resume_safe = .true.
        integer :: n_samples = 0
        integer :: n_features = 0
        integer :: n_outputs = 0
        integer :: n_parameters = 0
        integer :: n_optimizer_groups = 0
        integer :: epoch = 0
        integer :: updates = 0
        integer :: microbatches = 0
        integer :: microbatch_position = 1
        integer :: active_epoch = 0
        integer :: active_microbatches = 0
        integer :: accumulated_samples = 0
        real(dp) :: accumulated_weight_mass = 0.0_dp
        integer :: iterator_epoch = 0
        integer :: iterator_position = 1
        integer :: batch_size = 0
        integer :: accumulation_steps = 1
        integer :: shuffle_seed = 17
        integer :: adam_step_count = 0
        integer :: optimizer = MLP_OPTIMIZER_ADAM
        integer :: precision_kind = MLP_PRECISION_FP64
        type(mlp_loss_scale_state_t) :: loss_scale
        integer :: stale_epochs = 0
        integer :: schedule_bad_updates = 0
        integer :: schedule_reductions = 0
        integer :: gradient_clipped_updates = 0
        integer :: validation_interval = 1
        integer :: patience = 0
        logical :: shuffle = .false.
        logical :: has_validation = .false.
        logical :: converged = .false.
        logical :: early_stopped = .false.
        logical :: restore_best = .true.
        logical :: has_typed_schedule = .false.
        logical :: schedule_metric_initialized = .false.
        integer(int64) :: shuffle_state = 1_int64
        real(dp) :: learning_rate = 1.0e-3_dp
        real(dp) :: beta1 = 0.9_dp
        real(dp) :: beta2 = 0.999_dp
        real(dp) :: epsilon = 1.0e-8_dp
        real(dp) :: adafactor_decay = 0.999_dp
        real(dp) :: adafactor_clip_threshold = 1.0_dp
        logical :: adafactor_relative_step = .false.
        logical :: adafactor_scale_parameter = .false.
        logical :: adafactor_factored = .false.
        real(dp) :: rmsprop_decay = 0.99_dp
        real(dp) :: rmsprop_momentum = 0.0_dp
        logical :: rmsprop_centered = .false.
        real(dp) :: momentum = 0.0_dp
        logical :: nesterov = .false.
        real(dp) :: weight_decay = 0.0_dp
        real(dp) :: l2 = 0.0_dp
        real(dp) :: tolerance = 1.0e-8_dp
        real(dp) :: min_delta = 0.0_dp
        real(dp) :: gradient_clip_norm = 0.0_dp
        real(dp) :: ema_decay = 0.0_dp
        type(mlp_learning_rate_schedule_t) :: typed_schedule
        real(dp) :: last_learning_rate = 0.0_dp
        real(dp) :: schedule_best_metric = huge(1.0_dp)
        real(dp) :: initial_loss = huge(1.0_dp)
        real(dp) :: final_loss = huge(1.0_dp)
        real(dp) :: best_loss = huge(1.0_dp)
        real(dp) :: initial_validation_loss = huge(1.0_dp)
        real(dp) :: final_validation_loss = huge(1.0_dp)
        real(dp) :: best_validation_loss = huge(1.0_dp)
        integer :: best_epoch = 0
        integer :: best_validation_epoch = 0
        real(dp), allocatable :: parameters(:)
        character(len=64), allocatable :: optimizer_group_name(:)
        integer, allocatable :: optimizer_group_first(:)
        integer, allocatable :: optimizer_group_last(:)
        real(dp), allocatable :: optimizer_group_learning_rate_multiplier(:)
        real(dp), allocatable :: first_moment(:)
        real(dp), allocatable :: second_moment(:)
        real(dp), allocatable :: max_second_moment(:)
        real(dp), allocatable :: rmsprop_buffer(:)
        !! Matrix-factored Adafactor state is flattened with explicit block
        !! metadata so a checkpoint can reject a shape-compatible but
        !! differently laid-out network rather than silently remapping rows.
        integer :: n_adafactor_blocks = 0
        integer, allocatable :: adafactor_block_first(:)
        integer, allocatable :: adafactor_block_last(:)
        integer, allocatable :: adafactor_block_rows(:)
        integer, allocatable :: adafactor_block_columns(:)
        integer, allocatable :: adafactor_block_factored(:)
        real(dp), allocatable :: adafactor_row_moment(:)
        real(dp), allocatable :: adafactor_column_moment(:)
        real(dp), allocatable :: adafactor_second_moment(:)
        real(dp), allocatable :: ema_parameters(:)
        real(dp), allocatable :: best_parameters(:)
        real(dp), allocatable :: accumulated_gradient(:)
        integer, allocatable :: iterator_order(:)
        real(dp), allocatable :: loss_history(:)
        real(dp), allocatable :: learning_rate_history(:)
        real(dp), allocatable :: validation_loss_history(:)
    contains
        procedure, public :: clear => mlp_checkpoint_clear
        procedure, public :: valid => mlp_checkpoint_valid
    end type mlp_training_checkpoint_t

    type, public :: mlp_loss_diagnostics_t
        !! Named scalar diagnostics for the MLP MSE objective.
        real(dp) :: data_loss = 0.0_dp
        real(dp) :: regularization_loss = 0.0_dp
        real(dp) :: weight_mass = 0.0_dp
        integer :: sample_count = 0
    end type mlp_loss_diagnostics_t

    type, public :: mlp_batch_iterator_t
        !! Deterministic row-index batches with explicit epoch boundaries.
        !!
        !! `next_batch` never advances implicitly to the next epoch.  A caller
        !! must call `reset`, which makes the RNG boundary and final-batch
        !! behavior explicit and makes an iterator copy a resumable in-memory
        !! cursor.  The order is one-based to match Fortran array indexing.
        private
        integer :: n_samples = 0
        integer :: batch_size = 0
        integer :: position = 1
        integer :: epoch_number = 0
        integer(int64) :: shuffle_state = 1_int64
        logical :: shuffle = .false.
        logical :: ready = .false.
        integer, allocatable :: order(:)
    contains
        procedure, public :: initialize => batch_iterator_initialize
        procedure, public :: reset => batch_iterator_reset
        procedure, public :: next_batch => batch_iterator_next
        procedure, public :: batch_count => batch_iterator_batch_count
        procedure, public :: sample_count => batch_iterator_sample_count
        procedure, public :: current_epoch => batch_iterator_epoch
        procedure, public :: current_position => batch_iterator_position
        procedure, public :: initialized => batch_iterator_initialized
    end type mlp_batch_iterator_t

    type, public :: mlp_training_objective_t
        !! A value/gradient/HVP adapter suitable for FortOpt.
        !!
        !! The packed variable is the network parameter vector.  When
        !! `optimize_l2` is true, one final scalar component is appended and
        !! represents the non-negative L2 coefficient.  Every evaluation
        !! updates the live model, so a FortOpt optimizer and direct products
        !! use exactly the same loss implementation.
        private
        type(mlp_t), pointer :: model => null()
        real(dp), allocatable :: features(:, :), targets(:, :)
        real(dp), allocatable :: sample_weight(:)
        real(dp) :: l2 = 0.0_dp
        logical :: optimize_l2 = .false.
    contains
        procedure, public :: initialize => mlp_objective_initialize
        procedure, public :: parameter_count => mlp_objective_parameter_count
        procedure, public :: parameters => mlp_objective_parameters
        procedure, public :: value_gradient => mlp_objective_value_gradient
        procedure, public :: jvp => mlp_objective_jvp
        procedure, public :: vjp => mlp_objective_vjp
        procedure, public :: hvp => mlp_objective_hvp
        procedure, public :: fortopt => mlp_objective_fortopt
    end type mlp_training_objective_t

    type, public :: mlp_lbfgsb_options_t
        !! Bounds and convergence controls for deterministic MLP L-BFGS-B.
        !!
        !! The network parameter block is always optimized.  When
        !! `optimize_l2` is true, one additional bounded parameter is appended
        !! to that block and the analytic L2 derivative is optimized together
        !! with the weights.  Bounds are deliberately explicit: silently
        !! clipping a model parameter outside the caller's intended domain is
        !! a common source of irreproducible training.
        integer :: memory = 10
        integer :: max_iterations = 100
        integer :: max_line_search = 40
        real(dp) :: gradient_tolerance = 1.0e-8_dp
        real(dp) :: step_tolerance = 1.0e-12_dp
        real(dp) :: objective_tolerance = 1.0e-12_dp
        real(dp) :: lower_bound = -20.0_dp
        real(dp) :: upper_bound = 20.0_dp
        real(dp) :: l2 = 0.0_dp
        real(dp) :: l2_lower_bound = 0.0_dp
        real(dp) :: l2_upper_bound = 20.0_dp
        logical :: optimize_l2 = .false.
    end type mlp_lbfgsb_options_t

    type, public :: mlp_lbfgsb_result_t
        !! Objective and optimizer diagnostics returned by
        !! `mlp_optimize_lbfgsb`.
        logical :: converged = .false.
        integer :: iterations = 0
        integer :: line_search_evaluations = 0
        real(dp) :: objective = huge(1.0_dp)
        real(dp) :: gradient_norm = huge(1.0_dp)
        real(dp) :: l2 = 0.0_dp
    end type mlp_lbfgsb_result_t

    public :: mlp_epoch_callback_proc
    public :: mlp_learning_rate_schedule_proc
    public :: mlp_training_event_proc
    public :: mlp_loss_value_gradient
    public :: mlp_loss_hvp
    public :: mlp_train
    public :: mlp_optimize_lbfgsb
    public :: mlp_precision_name

contains

    pure function mlp_precision_name(kind) result(name)
        !! Stable human-readable spelling for the training precision contract.
        integer, intent(in) :: kind
        character(len=8) :: name

        select case (kind)
        case (MLP_PRECISION_FP64)
            name = "fp64"
        case (MLP_PRECISION_FP32)
            name = "fp32"
        case (MLP_PRECISION_FP16)
            name = "fp16"
        case (MLP_PRECISION_BF16)
            name = "bf16"
        case default
            name = "unknown"
        end select
    end function mlp_precision_name

    subroutine loss_scale_initialize(self, status, enabled, initial_scale, growth_factor, &
            backoff_factor, growth_interval, minimum_scale, maximum_scale)
        class(mlp_loss_scale_state_t), intent(out) :: self
        type(fortnum_status_t), intent(out) :: status
        logical, intent(in), optional :: enabled
        real(dp), intent(in), optional :: initial_scale, growth_factor, backoff_factor
        integer, intent(in), optional :: growth_interval
        real(dp), intent(in), optional :: minimum_scale, maximum_scale

        self%enabled = .false.
        self%scale = 1.0_dp
        self%initial_scale = 1.0_dp
        self%growth_factor = 2.0_dp
        self%backoff_factor = 0.5_dp
        self%minimum_scale = 1.0_dp
        self%maximum_scale = 16777216.0_dp
        self%growth_interval = 2000
        self%good_steps = 0
        self%overflow_count = 0
        self%skipped_updates = 0
        if (present(enabled)) self%enabled = enabled
        if (present(initial_scale)) self%initial_scale = initial_scale
        if (present(growth_factor)) self%growth_factor = growth_factor
        if (present(backoff_factor)) self%backoff_factor = backoff_factor
        if (present(growth_interval)) self%growth_interval = growth_interval
        if (present(minimum_scale)) self%minimum_scale = minimum_scale
        if (present(maximum_scale)) self%maximum_scale = maximum_scale
        self%scale = self%initial_scale
        if (.not. self%enabled) then
            self%scale = 1.0_dp
            self%initial_scale = 1.0_dp
        end if
        if (.not. self%valid()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP loss scale: invalid initialization parameters")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine loss_scale_initialize

    subroutine loss_scale_observe(self, finite_gradient, update_applied, status)
        class(mlp_loss_scale_state_t), intent(inout) :: self
        logical, intent(in) :: finite_gradient, update_applied
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: next_scale

        if (.not. self%valid()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP loss scale: state is invalid")
            return
        end if
        if (.not. self%enabled) then
            call status_set(status, FORTNUM_OK, "")
            return
        end if
        if (.not. finite_gradient) then
            self%overflow_count = self%overflow_count + 1
            self%skipped_updates = self%skipped_updates + 1
            self%good_steps = 0
            next_scale = self%scale*self%backoff_factor
            self%scale = max(self%minimum_scale, next_scale)
        else if (update_applied) then
            self%good_steps = self%good_steps + 1
            if (self%good_steps >= self%growth_interval) then
                next_scale = self%scale*self%growth_factor
                self%scale = min(self%maximum_scale, next_scale)
                self%good_steps = 0
            end if
        end if
        if (.not. self%valid()) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "MLP loss scale: update produced invalid state")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine loss_scale_observe

    subroutine loss_scale_scale_gradient(self, gradient, scaled_gradient, status)
        !! Apply the current loss scale without hiding overflow from callers.
        !!
        !! This is deliberately a separate product from `observe`: a scaled
        !! vector may contain IEEE infinities even when the input vector is
        !! finite.  Callers should use `scaled_gradient_finite` before
        !! unscaling and committing an optimizer update.
        class(mlp_loss_scale_state_t), intent(in) :: self
        real(dp), intent(in) :: gradient(:)
        real(dp), intent(out) :: scaled_gradient(:)
        type(fortnum_status_t), intent(out) :: status

        scaled_gradient = 0.0_dp
        if (.not. self%valid()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP loss scale: cannot scale with invalid state")
            return
        end if
        if (size(scaled_gradient) /= size(gradient)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP loss scale: scaled-gradient shape mismatch")
            return
        end if
        ! Do not reject a non-finite input here: preserving it in the output
        ! lets the caller route both source-gradient and scale-induced
        ! overflows through the same `scaled_gradient_finite` branch.
        scaled_gradient = gradient
        if (self%enabled) scaled_gradient = self%scale*gradient
        call status_set(status, FORTNUM_OK, "")
    end subroutine loss_scale_scale_gradient

    subroutine loss_scale_unscale_gradient(self, scaled_gradient, gradient, status)
        !! Undo the current loss scale after a finite scaled computation.
        class(mlp_loss_scale_state_t), intent(in) :: self
        real(dp), intent(in) :: scaled_gradient(:)
        real(dp), intent(out) :: gradient(:)
        type(fortnum_status_t), intent(out) :: status

        gradient = 0.0_dp
        if (.not. self%valid()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP loss scale: cannot unscale with invalid state")
            return
        end if
        if (size(gradient) /= size(scaled_gradient)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP loss scale: gradient shape mismatch")
            return
        end if
        if (.not. self%scaled_gradient_finite(scaled_gradient)) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "MLP loss scale: refusing to unscale a non-finite gradient")
            return
        end if
        gradient = scaled_gradient
        if (self%enabled) gradient = scaled_gradient/self%scale
        if (any(.not. ieee_is_finite(gradient))) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "MLP loss scale: unscaled gradient is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine loss_scale_unscale_gradient

    logical function loss_scale_scaled_gradient_finite(self, scaled_gradient) result(finite)
        !! Return whether a scaled gradient is safe to commit.
        class(mlp_loss_scale_state_t), intent(in) :: self
        real(dp), intent(in) :: scaled_gradient(:)

        finite = self%valid()
        if (.not. finite) return
        finite = all(ieee_is_finite(scaled_gradient))
    end function loss_scale_scaled_gradient_finite

    logical function loss_scale_valid(self) result(valid)
        class(mlp_loss_scale_state_t), intent(in) :: self

        valid = self%growth_interval > 0 .and. self%good_steps >= 0 .and. &
            self%overflow_count >= 0 .and. self%skipped_updates >= 0 .and. &
            ieee_is_finite(self%scale) .and. ieee_is_finite(self%initial_scale) .and. &
            ieee_is_finite(self%growth_factor) .and. ieee_is_finite(self%backoff_factor) .and. &
            ieee_is_finite(self%minimum_scale) .and. ieee_is_finite(self%maximum_scale) .and. &
            self%minimum_scale > 0.0_dp .and. self%maximum_scale >= self%minimum_scale .and. &
            self%initial_scale >= self%minimum_scale .and. &
            self%initial_scale <= self%maximum_scale .and. &
            self%scale >= self%minimum_scale .and. self%scale <= self%maximum_scale .and. &
            self%growth_factor >= 1.0_dp .and. self%backoff_factor > 0.0_dp .and. &
            self%backoff_factor <= 1.0_dp
        if (.not. valid) return
        if (.not. self%enabled) valid = self%scale == 1.0_dp .and. self%initial_scale == 1.0_dp
    end function loss_scale_valid

    logical function loss_scale_compatible(self, other) result(equal)
        class(mlp_loss_scale_state_t), intent(in) :: self
        type(mlp_loss_scale_state_t), intent(in) :: other

        equal = self%enabled .eqv. other%enabled .and. &
            self%initial_scale == other%initial_scale .and. &
            self%growth_factor == other%growth_factor .and. &
            self%backoff_factor == other%backoff_factor .and. &
            self%minimum_scale == other%minimum_scale .and. &
            self%maximum_scale == other%maximum_scale .and. &
            self%growth_interval == other%growth_interval
    end function loss_scale_compatible

    subroutine mlp_optimizer_group_initialize(self, name, first, last, &
            learning_rate_multiplier, status)
        class(mlp_optimizer_group_t), intent(out) :: self
        character(*), intent(in) :: name
        integer, intent(in) :: first, last
        real(dp), intent(in) :: learning_rate_multiplier
        type(fortnum_status_t), intent(out) :: status

        self%name = ""
        self%first = 0
        self%last = -1
        self%learning_rate_multiplier = 1.0_dp
        if (len_trim(name) == 0 .or. len_trim(name) > len(self%name) .or. &
            index(name, '"') > 0 .or. &
            first < 1 .or. last < first .or. &
            .not. ieee_is_finite(learning_rate_multiplier) .or. &
            learning_rate_multiplier <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP optimizer group: invalid name, range, or multiplier")
            return
        end if
        self%name = trim(name)
        self%first = first
        self%last = last
        self%learning_rate_multiplier = learning_rate_multiplier
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_optimizer_group_initialize

    integer function mlp_optimizer_group_size(self) result(n)
        class(mlp_optimizer_group_t), intent(in) :: self

        n = 0
        if (self%initialized()) n = self%last - self%first + 1
    end function mlp_optimizer_group_size

    logical function mlp_optimizer_group_initialized(self) result(yes)
        class(mlp_optimizer_group_t), intent(in) :: self

        yes = len_trim(self%name) > 0 .and. self%first >= 1 .and. &
            self%last >= self%first .and. &
            ieee_is_finite(self%learning_rate_multiplier) .and. &
            self%learning_rate_multiplier > 0.0_dp
    end function mlp_optimizer_group_initialized

    subroutine batch_iterator_initialize(self, n_samples, status, batch_size, &
            shuffle, seed)
        class(mlp_batch_iterator_t), intent(out) :: self
        integer, intent(in) :: n_samples
        type(fortnum_status_t), intent(out) :: status
        integer, intent(in), optional :: batch_size
        logical, intent(in), optional :: shuffle
        integer, intent(in), optional :: seed
        integer :: requested_batch, requested_seed, first
        logical :: requested_shuffle

        requested_batch = n_samples
        if (present(batch_size)) requested_batch = batch_size
        requested_shuffle = .false.
        if (present(shuffle)) requested_shuffle = shuffle
        requested_seed = 17
        if (present(seed)) requested_seed = seed
        if (n_samples < 1 .or. requested_batch < 1 .or. &
            (requested_shuffle .and. requested_seed <= 0)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP batch iterator: invalid sample, batch, or seed value")
            return
        end if

        self%n_samples = n_samples
        self%batch_size = min(requested_batch, n_samples)
        self%shuffle = requested_shuffle
        self%shuffle_state = int(requested_seed, int64)
        self%position = n_samples + 1
        self%epoch_number = 0
        self%ready = .true.
        allocate(self%order(n_samples))
        self%order = [(first, first=1, n_samples)]
        call status_set(status, FORTNUM_OK, "")
    end subroutine batch_iterator_initialize

    subroutine batch_iterator_reset(self, status)
        class(mlp_batch_iterator_t), intent(inout) :: self
        type(fortnum_status_t), intent(out) :: status
        integer :: first

        if (.not. self%ready .or. .not. allocated(self%order)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP batch iterator: reset before initialize")
            return
        end if
        if (self%epoch_number == huge(self%epoch_number)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP batch iterator: epoch counter overflow")
            return
        end if
        self%epoch_number = self%epoch_number + 1
        self%position = 1
        self%order = [(first, first=1, self%n_samples)]
        if (self%shuffle) call shuffle_order(self%order, self%shuffle_state)
        call status_set(status, FORTNUM_OK, "")
    end subroutine batch_iterator_reset

    subroutine batch_iterator_next(self, indices, has_batch, status)
        class(mlp_batch_iterator_t), intent(inout) :: self
        integer, allocatable, intent(out) :: indices(:)
        logical, intent(out) :: has_batch
        type(fortnum_status_t), intent(out) :: status
        integer :: last

        has_batch = .false.
        if (.not. self%ready .or. .not. allocated(self%order)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP batch iterator: next_batch before initialize")
            return
        end if
        if (self%position > self%n_samples) then
            allocate(indices(0))
            call status_set(status, FORTNUM_OK, "")
            return
        end if
        last = min(self%position + self%batch_size - 1, self%n_samples)
        allocate(indices(last - self%position + 1))
        indices = self%order(self%position:last)
        self%position = last + 1
        has_batch = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine batch_iterator_next

    integer function batch_iterator_batch_count(self) result(count)
        class(mlp_batch_iterator_t), intent(in) :: self

        count = 0
        if (.not. self%ready) return
        count = (self%n_samples + self%batch_size - 1)/self%batch_size
    end function batch_iterator_batch_count

    integer function batch_iterator_sample_count(self) result(count)
        class(mlp_batch_iterator_t), intent(in) :: self

        count = self%n_samples
    end function batch_iterator_sample_count

    integer function batch_iterator_epoch(self) result(epoch)
        class(mlp_batch_iterator_t), intent(in) :: self

        epoch = self%epoch_number
    end function batch_iterator_epoch

    integer function batch_iterator_position(self) result(position)
        class(mlp_batch_iterator_t), intent(in) :: self

        position = self%position
    end function batch_iterator_position

    logical function batch_iterator_initialized(self) result(initialized)
        class(mlp_batch_iterator_t), intent(in) :: self

        initialized = self%ready
    end function batch_iterator_initialized

    subroutine mlp_checkpoint_clear(self)
        class(mlp_training_checkpoint_t), intent(inout) :: self
        !! Default-initialized instances, standing in for empty
        !! structure constructors: nvfortran rejects `T()` outright,
        !! and a declared local carries the same default init.
        type(mlp_learning_rate_schedule_t) :: mlp_learning_rate_schedule_t_default
        type(mlp_loss_scale_state_t) :: mlp_loss_scale_state_t_default

        if (allocated(self%parameters)) deallocate(self%parameters)
        if (allocated(self%optimizer_group_name)) deallocate(self%optimizer_group_name)
        if (allocated(self%optimizer_group_first)) deallocate(self%optimizer_group_first)
        if (allocated(self%optimizer_group_last)) deallocate(self%optimizer_group_last)
        if (allocated(self%optimizer_group_learning_rate_multiplier)) then
            deallocate(self%optimizer_group_learning_rate_multiplier)
        end if
        if (allocated(self%first_moment)) deallocate(self%first_moment)
        if (allocated(self%second_moment)) deallocate(self%second_moment)
        if (allocated(self%max_second_moment)) deallocate(self%max_second_moment)
        if (allocated(self%rmsprop_buffer)) deallocate(self%rmsprop_buffer)
        if (allocated(self%adafactor_block_first)) deallocate(self%adafactor_block_first)
        if (allocated(self%adafactor_block_last)) deallocate(self%adafactor_block_last)
        if (allocated(self%adafactor_block_rows)) deallocate(self%adafactor_block_rows)
        if (allocated(self%adafactor_block_columns)) deallocate(self%adafactor_block_columns)
        if (allocated(self%adafactor_block_factored)) deallocate(self%adafactor_block_factored)
        if (allocated(self%adafactor_row_moment)) deallocate(self%adafactor_row_moment)
        if (allocated(self%adafactor_column_moment)) deallocate(self%adafactor_column_moment)
        if (allocated(self%adafactor_second_moment)) deallocate(self%adafactor_second_moment)
        if (allocated(self%ema_parameters)) deallocate(self%ema_parameters)
        if (allocated(self%best_parameters)) deallocate(self%best_parameters)
        if (allocated(self%accumulated_gradient)) then
            deallocate(self%accumulated_gradient)
        end if
        if (allocated(self%iterator_order)) deallocate(self%iterator_order)
        if (allocated(self%loss_history)) deallocate(self%loss_history)
        if (allocated(self%learning_rate_history)) then
            deallocate(self%learning_rate_history)
        end if
        if (allocated(self%validation_loss_history)) then
            deallocate(self%validation_loss_history)
        end if
        self%format_version = 11
        self%initialized = .false.
        self%resume_safe = .true.
        self%n_samples = 0
        self%n_features = 0
        self%n_outputs = 0
        self%n_parameters = 0
        self%n_optimizer_groups = 0
        self%epoch = 0
        self%updates = 0
        self%microbatches = 0
        self%microbatch_position = 1
        self%active_epoch = 0
        self%active_microbatches = 0
        self%accumulated_samples = 0
        self%accumulated_weight_mass = 0.0_dp
        self%iterator_epoch = 0
        self%iterator_position = 1
        self%batch_size = 0
        self%accumulation_steps = 1
        self%precision_kind = MLP_PRECISION_FP64
        self%loss_scale = mlp_loss_scale_state_t_default
        self%shuffle_seed = 17
        self%adam_step_count = 0
        self%optimizer = MLP_OPTIMIZER_ADAM
        self%stale_epochs = 0
        self%schedule_bad_updates = 0
        self%schedule_reductions = 0
        self%gradient_clipped_updates = 0
        self%validation_interval = 1
        self%patience = 0
        self%shuffle = .false.
        self%has_validation = .false.
        self%converged = .false.
        self%early_stopped = .false.
        self%restore_best = .true.
        self%has_typed_schedule = .false.
        self%schedule_metric_initialized = .false.
        self%shuffle_state = 1_int64
        self%learning_rate = 1.0e-3_dp
        self%beta1 = 0.9_dp
        self%beta2 = 0.999_dp
        self%epsilon = 1.0e-8_dp
        self%adafactor_decay = 0.999_dp
        self%adafactor_clip_threshold = 1.0_dp
        self%adafactor_relative_step = .false.
        self%adafactor_scale_parameter = .false.
        self%adafactor_factored = .false.
        self%n_adafactor_blocks = 0
        self%rmsprop_decay = 0.99_dp
        self%rmsprop_momentum = 0.0_dp
        self%rmsprop_centered = .false.
        self%momentum = 0.0_dp
        self%nesterov = .false.
        self%weight_decay = 0.0_dp
        self%l2 = 0.0_dp
        self%tolerance = 1.0e-8_dp
        self%min_delta = 0.0_dp
        self%gradient_clip_norm = 0.0_dp
        self%ema_decay = 0.0_dp
        self%typed_schedule = mlp_learning_rate_schedule_t_default
        self%last_learning_rate = 0.0_dp
        self%schedule_best_metric = huge(1.0_dp)
        self%initial_loss = huge(1.0_dp)
        self%final_loss = huge(1.0_dp)
        self%best_loss = huge(1.0_dp)
        self%initial_validation_loss = huge(1.0_dp)
        self%final_validation_loss = huge(1.0_dp)
        self%best_validation_loss = huge(1.0_dp)
        self%best_epoch = 0
        self%best_validation_epoch = 0
    end subroutine mlp_checkpoint_clear

    logical function mlp_checkpoint_valid(self) result(valid)
        class(mlp_training_checkpoint_t), intent(in) :: self

        valid = self%initialized .and. self%format_version == 11 .and. &
            self%n_samples > 0 .and. self%n_features > 0 .and. &
            self%n_outputs > 0 .and. self%n_parameters > 0 .and. &
            self%epoch >= 0 .and. self%updates >= 0 .and. &
            self%microbatches >= 0 .and. self%microbatch_position >= 1 .and. &
            self%active_epoch >= self%epoch .and. &
            self%active_microbatches >= 0 .and. self%accumulated_samples >= 0 .and. &
            self%iterator_epoch >= 0 .and. self%iterator_position >= 1 .and. &
            self%iterator_position <= self%n_samples + 1 .and. &
            self%accumulated_samples <= self%n_samples .and. &
            self%active_microbatches <= self%accumulation_steps .and. &
            self%batch_size > 0 .and. self%accumulation_steps > 0 .and. &
            self%shuffle_seed > 0 .and. self%adam_step_count >= 0 .and. &
            self%precision_kind >= MLP_PRECISION_FP64 .and. &
            self%precision_kind <= MLP_PRECISION_BF16 .and. &
            self%loss_scale%valid() .and. &
            (self%optimizer == MLP_OPTIMIZER_ADAM .or. &
            self%optimizer == MLP_OPTIMIZER_SGD .or. &
            self%optimizer == MLP_OPTIMIZER_ADAMW .or. &
            self%optimizer == MLP_OPTIMIZER_ADAGRAD .or. &
            self%optimizer == MLP_OPTIMIZER_RMSPROP .or. &
            self%optimizer == MLP_OPTIMIZER_ADAFACTOR .or. &
            self%optimizer == MLP_OPTIMIZER_AMSGRAD .or. &
            self%optimizer == MLP_OPTIMIZER_RADAM .or. &
            self%optimizer == MLP_OPTIMIZER_LION) .and. &
            self%validation_interval > 0 .and. self%patience >= 0 .and. &
            self%gradient_clipped_updates >= 0 .and. &
            self%schedule_bad_updates >= 0 .and. self%schedule_reductions >= 0 .and. &
            allocated(self%parameters) .and. allocated(self%first_moment) .and. &
            allocated(self%second_moment) .and. allocated(self%best_parameters) &
            .and. allocated(self%accumulated_gradient) .and. &
            allocated(self%iterator_order) .and. &
            allocated(self%loss_history) .and. &
            allocated(self%learning_rate_history)
        if (.not. valid) return
        valid = ieee_is_finite(self%accumulated_weight_mass) .and. &
            self%accumulated_weight_mass >= 0.0_dp
        if (.not. valid) return
        valid = size(self%parameters) == self%n_parameters .and. &
            size(self%first_moment) == self%n_parameters .and. &
            size(self%second_moment) == self%n_parameters .and. &
            size(self%best_parameters) == self%n_parameters .and. &
            size(self%accumulated_gradient) == self%n_parameters .and. &
            size(self%iterator_order) == self%n_samples .and. &
            size(self%loss_history) == self%epoch .and. &
            size(self%learning_rate_history) == self%epoch .and. &
            all(self%iterator_order >= 1) .and. &
            all(self%iterator_order <= self%n_samples)
        if (.not. valid) return
        if (self%optimizer == MLP_OPTIMIZER_RMSPROP) then
            if (.not. allocated(self%rmsprop_buffer)) then
                valid = .false.
                return
            end if
            if (size(self%rmsprop_buffer) /= self%n_parameters) then
                valid = .false.
                return
            end if
        else if (allocated(self%rmsprop_buffer)) then
            valid = .false.
            return
        end if
        if (self%optimizer == MLP_OPTIMIZER_AMSGRAD) then
            if (.not. allocated(self%max_second_moment)) then
                valid = .false.
                return
            end if
            if (size(self%max_second_moment) /= self%n_parameters) then
                valid = .false.
                return
            end if
        else if (allocated(self%max_second_moment)) then
            valid = .false.
            return
        end if
        if (self%optimizer == MLP_OPTIMIZER_ADAFACTOR .and. self%adafactor_factored) then
            if (self%n_adafactor_blocks < 1 .or. &
                .not. allocated(self%adafactor_block_first) .or. &
                .not. allocated(self%adafactor_block_last) .or. &
                .not. allocated(self%adafactor_block_rows) .or. &
                .not. allocated(self%adafactor_block_columns) .or. &
                .not. allocated(self%adafactor_block_factored) .or. &
                .not. allocated(self%adafactor_row_moment) .or. &
                .not. allocated(self%adafactor_column_moment) .or. &
                .not. allocated(self%adafactor_second_moment)) then
                valid = .false.
                return
            end if
            valid = size(self%adafactor_block_first) == self%n_adafactor_blocks .and. &
                size(self%adafactor_block_last) == self%n_adafactor_blocks .and. &
                size(self%adafactor_block_rows) == self%n_adafactor_blocks .and. &
                size(self%adafactor_block_columns) == self%n_adafactor_blocks .and. &
                size(self%adafactor_block_factored) == self%n_adafactor_blocks
            if (.not. valid) return
            valid = all(self%adafactor_block_first >= 1) .and. &
                all(self%adafactor_block_last >= self%adafactor_block_first) .and. &
                all(self%adafactor_block_rows >= 1) .and. &
                all(self%adafactor_block_columns >= 1) .and. &
                all((self%adafactor_block_factored == 0) .or. &
                (self%adafactor_block_factored == 1))
            if (.not. valid) return
            valid = all(self%adafactor_block_last - self%adafactor_block_first + 1 == &
                self%adafactor_block_rows*self%adafactor_block_columns)
            if (.not. valid) return
            valid = sum(merge(self%adafactor_block_rows, 0, &
                self%adafactor_block_factored == 1)) == size(self%adafactor_row_moment) .and. &
                sum(merge(self%adafactor_block_columns, 0, &
                self%adafactor_block_factored == 1)) == size(self%adafactor_column_moment) .and. &
                sum(merge(self%adafactor_block_last - self%adafactor_block_first + 1, 0, &
                self%adafactor_block_factored == 0)) == size(self%adafactor_second_moment)
            if (.not. valid) return
            valid = all(ieee_is_finite(self%adafactor_row_moment)) .and. &
                all(ieee_is_finite(self%adafactor_column_moment)) .and. &
                all(ieee_is_finite(self%adafactor_second_moment)) .and. &
                all(self%adafactor_row_moment >= 0.0_dp) .and. &
                all(self%adafactor_column_moment >= 0.0_dp) .and. &
                all(self%adafactor_second_moment >= 0.0_dp)
        else
            valid = .not. self%adafactor_factored .and. self%n_adafactor_blocks == 0 .and. &
                .not. allocated(self%adafactor_block_first) .and. &
                .not. allocated(self%adafactor_block_last) .and. &
                .not. allocated(self%adafactor_block_rows) .and. &
                .not. allocated(self%adafactor_block_columns) .and. &
                .not. allocated(self%adafactor_block_factored) .and. &
                .not. allocated(self%adafactor_row_moment) .and. &
                .not. allocated(self%adafactor_column_moment) .and. &
                .not. allocated(self%adafactor_second_moment)
        end if
        if (.not. valid) return
        if (self%ema_decay > 0.0_dp) then
            if (.not. allocated(self%ema_parameters)) then
                valid = .false.
                return
            end if
            if (size(self%ema_parameters) /= self%n_parameters) then
                valid = .false.
                return
            end if
        else if (allocated(self%ema_parameters)) then
            valid = .false.
            return
        end if
        if (self%has_validation) then
            valid = allocated(self%validation_loss_history) .and. &
                size(self%validation_loss_history) == self%epoch
        else
            valid = .not. allocated(self%validation_loss_history)
        end if
        if (.not. valid) return
        valid = all(ieee_is_finite(self%parameters)) .and. &
            all(ieee_is_finite(self%first_moment)) .and. &
            all(ieee_is_finite(self%second_moment)) .and. &
            all(ieee_is_finite(self%best_parameters)) .and. &
            all(ieee_is_finite(self%accumulated_gradient)) .and. &
            all(ieee_is_finite(self%loss_history)) .and. &
            all(ieee_is_finite(self%learning_rate_history))
        if (self%optimizer == MLP_OPTIMIZER_RMSPROP) valid = valid .and. &
            all(ieee_is_finite(self%rmsprop_buffer))
        if (self%optimizer == MLP_OPTIMIZER_AMSGRAD) valid = valid .and. &
            all(ieee_is_finite(self%max_second_moment))
        if (self%ema_decay > 0.0_dp) valid = valid .and. &
            all(ieee_is_finite(self%ema_parameters))
        if (self%has_validation) valid = valid .and. &
            all(ieee_is_finite(self%validation_loss_history))
        valid = valid .and. ieee_is_finite(self%learning_rate) .and. &
            ieee_is_finite(self%beta1) .and. ieee_is_finite(self%beta2) .and. &
            ieee_is_finite(self%epsilon) .and. ieee_is_finite(self%l2) .and. &
            ieee_is_finite(self%rmsprop_decay) .and. &
            ieee_is_finite(self%rmsprop_momentum) .and. &
            ieee_is_finite(self%momentum) .and. &
            ieee_is_finite(self%weight_decay) .and. &
            ieee_is_finite(self%last_learning_rate) .and. &
            ieee_is_finite(self%schedule_best_metric) .and. &
            ieee_is_finite(self%initial_loss) .and. ieee_is_finite(self%final_loss) &
            .and. ieee_is_finite(self%best_loss) .and. &
            ieee_is_finite(self%tolerance) .and. &
            ieee_is_finite(self%min_delta) .and. &
            ieee_is_finite(self%gradient_clip_norm) .and. &
            self%tolerance >= 0.0_dp .and. self%min_delta >= 0.0_dp .and. &
            self%gradient_clip_norm >= 0.0_dp .and. &
            ieee_is_finite(self%ema_decay) .and. self%ema_decay >= 0.0_dp .and. &
            self%ema_decay < 1.0_dp
        valid = valid .and. self%momentum >= 0.0_dp .and. self%momentum < 1.0_dp
        valid = valid .and. self%rmsprop_decay >= 0.0_dp .and. &
            self%rmsprop_decay < 1.0_dp .and. self%rmsprop_momentum >= 0.0_dp .and. &
            self%rmsprop_momentum < 1.0_dp
        valid = valid .and. self%adafactor_decay >= 0.0_dp .and. &
            self%adafactor_decay < 1.0_dp .and. self%adafactor_clip_threshold > 0.0_dp .and. &
            ieee_is_finite(self%adafactor_decay) .and. ieee_is_finite(self%adafactor_clip_threshold)
        valid = valid .and. self%weight_decay >= 0.0_dp
        if (self%nesterov) valid = valid .and. self%optimizer == MLP_OPTIMIZER_SGD .and. &
            self%momentum > 0.0_dp
        if (self%has_validation) valid = valid .and. &
            ieee_is_finite(self%initial_validation_loss) .and. &
            ieee_is_finite(self%final_validation_loss) .and. &
            ieee_is_finite(self%best_validation_loss)
        if (self%has_typed_schedule) valid = valid .and. self%typed_schedule%valid()
        if (self%has_typed_schedule .and. self%typed_schedule%kind == MLP_SCHEDULE_PLATEAU) then
            valid = valid .and. self%schedule_metric_initialized
        end if
        if (self%n_optimizer_groups < 0) then
            valid = .false.
        else if (self%n_optimizer_groups == 0) then
            valid = valid .and. .not. allocated(self%optimizer_group_name) .and. &
                .not. allocated(self%optimizer_group_first) .and. &
                .not. allocated(self%optimizer_group_last) .and. &
                .not. allocated(self%optimizer_group_learning_rate_multiplier)
        else
            valid = valid .and. allocated(self%optimizer_group_name) .and. &
                allocated(self%optimizer_group_first) .and. &
                allocated(self%optimizer_group_last) .and. &
                allocated(self%optimizer_group_learning_rate_multiplier)
            if (.not. valid) return
            valid = size(self%optimizer_group_name) == self%n_optimizer_groups .and. &
                size(self%optimizer_group_first) == self%n_optimizer_groups .and. &
                size(self%optimizer_group_last) == self%n_optimizer_groups .and. &
                size(self%optimizer_group_learning_rate_multiplier) == self%n_optimizer_groups
            if (.not. valid) return
            valid = all(len_trim(self%optimizer_group_name) > 0) .and. &
                all(self%optimizer_group_first >= 1) .and. &
                all(self%optimizer_group_last >= self%optimizer_group_first) .and. &
                all(ieee_is_finite(self%optimizer_group_learning_rate_multiplier)) .and. &
                all(self%optimizer_group_learning_rate_multiplier > 0.0_dp)
            if (.not. valid) return
            valid = optimizer_group_ranges_valid(self%n_parameters, &
                self%optimizer_group_first, self%optimizer_group_last)
        end if
    end function mlp_checkpoint_valid

    subroutine mlp_loss_value_gradient(model, x, target, l2, value, gradient, &
            l2_gradient, status, sample_weight, reduction, diagnostics)
        !! MSE value, network-parameter gradient, and derivative with respect
        !! to the scalar L2 coefficient.  The optional `sample_weight` and
        !! `reduction` arguments make the data term's weighting explicit:
        !! `MLP_REDUCTION_MEAN` divides by positive weight mass, while
        !! `MLP_REDUCTION_SUM` leaves the weighted sum unnormalized.  L2 is
        !! always added once as a parameter regularizer.  `diagnostics`, when
        !! present, reports the two scalar objective components and weight
        !! mass used by the reduction.
        class(mlp_t), intent(in) :: model
        real(dp), intent(in) :: x(:, :), target(:, :), l2
        real(dp), intent(out) :: value, gradient(:), l2_gradient
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:)
        integer, intent(in), optional :: reduction
        type(mlp_loss_diagnostics_t), intent(out), optional :: diagnostics
        real(dp), allocatable :: prediction(:, :), x_bar(:, :)
        real(dp), allocatable :: weighted_residual(:, :), effective_weight(:)
        real(dp), allocatable :: theta(:)
        integer :: n_samples, reduction_kind
        real(dp) :: weight_mass, data_loss, regularization_loss

        value = 0.0_dp
        l2_gradient = 0.0_dp
        if (present(diagnostics)) diagnostics = mlp_loss_diagnostics_t()
        if (.not. valid_data(model, x, target) .or. l2 < 0.0_dp .or. &
            size(gradient) /= model%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP loss: model, data, penalty, or gradient shape is invalid")
            return
        end if
        n_samples = size(x, 1)
        reduction_kind = MLP_REDUCTION_MEAN
        if (present(reduction)) reduction_kind = reduction
        if (reduction_kind /= MLP_REDUCTION_MEAN .and. &
            reduction_kind /= MLP_REDUCTION_SUM) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP loss: reduction must be mean or sum")
            return
        end if
        if (present(sample_weight)) then
            if (size(sample_weight) /= n_samples .or. &
                any(.not. ieee_is_finite(sample_weight)) .or. &
                any(sample_weight < 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP loss: sample weights must be finite and non-negative")
                return
            end if
            weight_mass = sum(sample_weight)
        else
            weight_mass = real(n_samples, dp)
        end if
        if (.not. ieee_is_finite(weight_mass) .or. weight_mass <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP loss: sample weights have zero support")
            return
        end if
        allocate(prediction(size(target, 1), size(target, 2)))
        allocate(weighted_residual, mold=prediction)
        allocate(effective_weight(n_samples))
        if (present(sample_weight)) then
            effective_weight = sample_weight
        else
            effective_weight = 1.0_dp
        end if
        allocate(x_bar, mold=x)
        call model%predict(x, prediction, status)
        if (status%code /= FORTNUM_OK) return
        call weighted_mse_loss_value(prediction, target, effective_weight, &
            data_loss, status, reduction_kind)
        if (status%code /= FORTNUM_OK) return
        call weighted_mse_loss_vjp(prediction, target, effective_weight, 1.0_dp, &
            weighted_residual, status, reduction_kind)
        if (status%code /= FORTNUM_OK) return
        call model%vjp(x, weighted_residual, gradient, x_bar, status)
        if (status%code /= FORTNUM_OK) return
        theta = model%parameters()
        regularization_loss = 0.5_dp*l2*sum(theta*theta)
        value = data_loss + regularization_loss
        l2_gradient = 0.5_dp*sum(theta*theta)
        gradient = gradient + l2*theta
        if (present(diagnostics)) then
            diagnostics%data_loss = data_loss
            diagnostics%regularization_loss = regularization_loss
            diagnostics%weight_mass = weight_mass
            diagnostics%sample_count = n_samples
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_loss_value_gradient

    subroutine mlp_loss_hvp(model, x, target, l2, dtheta, l2_direction, &
            parameter_hvp, l2_hvp, status, sample_weight, reduction)
        !! Hessian-vector product for the MSE+L2 objective.
        !!
        !! The joint direction is `(dtheta,l2_direction)`.  The returned
        !! `parameter_hvp` is the parameter block of the joint Hessian-vector
        !! product and `l2_hvp` is the scalar L2 block.  Thus this product can
        !! be passed directly to a second-order outer hyperparameter method:
        !!
        !!   H * (dtheta, dl2) =
        !!     ( H_theta_theta*dtheta + theta*dl2, theta^T*dtheta ).
        !!
        !! The data term uses the network's JVP/VJP/HVP products; no finite
        !! differences are used in the implementation.
        class(mlp_t), intent(in) :: model
        real(dp), intent(in) :: x(:, :), target(:, :), l2, dtheta(:)
        real(dp), intent(in) :: l2_direction
        real(dp), intent(out) :: parameter_hvp(:), l2_hvp
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:)
        integer, intent(in), optional :: reduction
        real(dp), allocatable :: prediction(:, :), residual(:, :)
        real(dp), allocatable :: output_tangent(:, :), zero_input(:, :)
        real(dp), allocatable :: x_bar(:, :), jtj_product(:), curvature(:)
        real(dp), allocatable :: theta(:), effective_weight(:), output_hvp(:, :)
        integer :: n_samples, n_parameters, reduction_kind, i
        real(dp) :: weight_mass, normalization

        parameter_hvp = 0.0_dp
        l2_hvp = 0.0_dp
        n_parameters = model%parameter_count()
        if (.not. valid_data(model, x, target) .or. l2 < 0.0_dp .or. &
            size(dtheta) /= n_parameters .or. &
            size(parameter_hvp) /= n_parameters) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP loss HVP: model, data, penalty, or direction shape is invalid")
            return
        end if

        n_samples = size(x, 1)
        reduction_kind = MLP_REDUCTION_MEAN
        if (present(reduction)) reduction_kind = reduction
        if (reduction_kind /= MLP_REDUCTION_MEAN .and. &
            reduction_kind /= MLP_REDUCTION_SUM) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP loss HVP: reduction must be mean or sum")
            return
        end if
        allocate(effective_weight(n_samples))
        if (present(sample_weight)) then
            if (size(sample_weight) /= n_samples .or. &
                any(.not. ieee_is_finite(sample_weight)) .or. &
                any(sample_weight < 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP loss HVP: sample weights are invalid")
                return
            end if
            effective_weight = sample_weight
        else
            effective_weight = 1.0_dp
        end if
        weight_mass = sum(effective_weight)
        if (.not. ieee_is_finite(weight_mass) .or. weight_mass <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP loss HVP: sample weights have zero support")
            return
        end if
        normalization = 1.0_dp
        if (reduction_kind == MLP_REDUCTION_MEAN) normalization = weight_mass
        allocate(prediction(size(target, 1), size(target, 2)))
        allocate(residual, mold=prediction)
        allocate(output_tangent, mold=prediction)
        allocate(output_hvp, mold=prediction)
        allocate(zero_input, mold=x)
        allocate(x_bar, mold=x)
        allocate(jtj_product(n_parameters), curvature(n_parameters))
        call model%predict(x, prediction, status)
        if (status%code /= FORTNUM_OK) return
        residual = prediction - target
        zero_input = 0.0_dp
        call model%jvp(x, dtheta, zero_input, prediction, output_tangent, status)
        if (status%code /= FORTNUM_OK) return
        call weighted_mse_loss_hvp(prediction, target, effective_weight, &
            output_tangent, output_hvp, status, reduction_kind)
        if (status%code /= FORTNUM_OK) return
        call model%vjp(x, output_hvp, jtj_product, &
            x_bar, status)
        if (status%code /= FORTNUM_OK) return
        residual = prediction - target
        do i = 1, n_samples
            residual(i, :) = effective_weight(i)*residual(i, :)/normalization
        end do
        call model%hvp(x, residual, dtheta, zero_input, &
            curvature, x_bar, status)
        if (status%code /= FORTNUM_OK) return
        theta = model%parameters()
        parameter_hvp = jtj_product + curvature + l2*dtheta + &
            l2_direction*theta
        l2_hvp = dot_product(theta, dtheta)
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_loss_hvp

    subroutine mlp_objective_initialize(self, model, x, target, l2, status, &
            optimize_l2, sample_weight)
        class(mlp_training_objective_t), intent(out) :: self
        type(mlp_t), target, intent(inout) :: model
        real(dp), intent(in) :: x(:, :), target(:, :), l2
        type(fortnum_status_t), intent(out) :: status
        logical, intent(in), optional :: optimize_l2
        real(dp), intent(in), optional :: sample_weight(:)
        real(dp) :: weight_mass

        self%l2 = 0.0_dp
        self%optimize_l2 = .false.
        if (present(optimize_l2)) self%optimize_l2 = optimize_l2
        if (.not. valid_data(model, x, target)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP objective: model or data is invalid")
            return
        end if
        if (.not. ieee_is_finite(l2) .or. l2 < 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP objective: L2 coefficient is invalid")
            return
        end if
        if (present(sample_weight)) then
            call validate_sample_weight(sample_weight, size(x, 1), weight_mass, &
                status, "MLP objective")
            if (status%code /= FORTNUM_OK) return
        end if
        self%model => model
        allocate(self%features, source=x)
        allocate(self%targets, source=target)
        if (present(sample_weight)) allocate(self%sample_weight, source=sample_weight)
        self%l2 = l2
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_objective_initialize

    integer function mlp_objective_parameter_count(self) result(count)
        class(mlp_training_objective_t), intent(in) :: self

        count = 0
        if (.not. associated(self%model)) return
        count = self%model%parameter_count()
        if (self%optimize_l2) count = count + 1
    end function mlp_objective_parameter_count

    function mlp_objective_parameters(self) result(parameters)
        class(mlp_training_objective_t), intent(in) :: self
        real(dp), allocatable :: parameters(:)
        integer :: n_model

        allocate(parameters(self%parameter_count()))
        if (.not. associated(self%model)) return
        n_model = self%model%parameter_count()
        parameters(:n_model) = self%model%parameters()
        if (self%optimize_l2) parameters(n_model + 1) = self%l2
    end function mlp_objective_parameters

    subroutine mlp_objective_value_gradient(self, parameters, value, gradient, &
            status)
        class(mlp_training_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: l2, l2_gradient
        integer :: n_model

        value = huge(1.0_dp)
        gradient = 0.0_dp
        if (.not. associated(self%model)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP objective: adapter is not initialized")
            return
        end if
        n_model = self%model%parameter_count()
        if (size(parameters) /= self%parameter_count() .or. &
            size(gradient) /= size(parameters) .or. &
            any(.not. ieee_is_finite(parameters))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP objective: parameter or gradient shape/value is invalid")
            return
        end if
        l2 = self%l2
        if (self%optimize_l2) then
            l2 = parameters(n_model + 1)
            if (.not. ieee_is_finite(l2) .or. l2 < 0.0_dp) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP objective: optimized L2 coefficient is invalid")
                return
            end if
        end if
        call self%model%set_parameters(parameters(:n_model), status)
        if (status%code /= FORTNUM_OK) return
        if (allocated(self%sample_weight)) then
            call mlp_loss_value_gradient(self%model, self%features, self%targets, &
                l2, value, gradient(:n_model), l2_gradient, status, &
                sample_weight=self%sample_weight)
        else
            call mlp_loss_value_gradient(self%model, self%features, self%targets, &
                l2, value, gradient(:n_model), l2_gradient, status)
        end if
        if (status%code /= FORTNUM_OK) return
        if (self%optimize_l2) gradient(n_model + 1) = l2_gradient
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_objective_value_gradient

    subroutine mlp_objective_jvp(self, parameters, direction, value, tangent, &
            status)
        !! Exact scalar JVP for the MSE+L2 training objective.
        !!
        !! The output is scalar, so the product is the contraction of the
        !! analytic objective gradient with `direction`.  Keeping this
        !! operation beside `value_gradient` gives callers a complete forward
        !! differentiation contract without finite-difference plumbing.
        class(mlp_training_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:), direction(:)
        real(dp), intent(out) :: value, tangent
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: gradient(:)

        value = huge(1.0_dp)
        tangent = 0.0_dp
        if (size(direction) /= size(parameters) .or. &
            any(.not. ieee_is_finite(direction))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP objective JVP: direction shape/value is invalid")
            return
        end if
        allocate(gradient(size(parameters)))
        call self%value_gradient(parameters, value, gradient, status)
        if (status%code /= FORTNUM_OK) return
        tangent = dot_product(gradient, direction)
        if (.not. ieee_is_finite(tangent)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP objective JVP: product is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_objective_jvp

    subroutine mlp_objective_vjp(self, parameters, output_bar, gradient, status)
        !! Exact scalar VJP for the MSE+L2 training objective.
        !!
        !! `output_bar` is the cotangent of the scalar objective.  The
        !! returned packed vector is `output_bar * d objective / d parameters`.
        class(mlp_training_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:), output_bar
        real(dp), intent(out) :: gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: value

        gradient = 0.0_dp
        if (.not. ieee_is_finite(output_bar)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP objective VJP: output cotangent is not finite")
            return
        end if
        call self%value_gradient(parameters, value, gradient, status)
        if (status%code /= FORTNUM_OK) return
        gradient = output_bar*gradient
        if (any(.not. ieee_is_finite(gradient))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP objective VJP: product is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_objective_vjp

    subroutine mlp_objective_hvp(self, parameters, direction, product, status)
        class(mlp_training_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:), direction(:)
        real(dp), intent(out) :: product(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: l2, l2_direction, l2_hvp
        real(dp), allocatable :: parameter_hvp(:)
        integer :: n_model

        product = 0.0_dp
        if (.not. associated(self%model)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP objective HVP: adapter is not initialized")
            return
        end if
        n_model = self%model%parameter_count()
        if (size(parameters) /= self%parameter_count() .or. &
            size(direction) /= size(parameters) .or. &
            size(product) /= size(parameters)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP objective HVP: parameter or direction shape is invalid")
            return
        end if
        l2 = self%l2
        l2_direction = 0.0_dp
        if (self%optimize_l2) then
            l2 = parameters(n_model + 1)
            l2_direction = direction(n_model + 1)
            if (.not. ieee_is_finite(l2) .or. l2 < 0.0_dp) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP objective HVP: optimized L2 coefficient is invalid")
                return
            end if
        end if
        call self%model%set_parameters(parameters(:n_model), status)
        if (status%code /= FORTNUM_OK) return
        allocate(parameter_hvp(n_model))
        if (allocated(self%sample_weight)) then
            call mlp_loss_hvp(self%model, self%features, self%targets, l2, &
                direction(:n_model), l2_direction, parameter_hvp, l2_hvp, status, &
                sample_weight=self%sample_weight)
        else
            call mlp_loss_hvp(self%model, self%features, self%targets, l2, &
                direction(:n_model), l2_direction, parameter_hvp, l2_hvp, status)
        end if
        if (status%code /= FORTNUM_OK) return
        product(:n_model) = parameter_hvp
        if (self%optimize_l2) product(n_model + 1) = l2_hvp
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_objective_hvp

    subroutine mlp_objective_fortopt(self, objective, status)
        class(mlp_training_objective_t), target, intent(inout) :: self
        type(objective_t), intent(out) :: objective
        type(fortnum_status_t), intent(out) :: status

        if (.not. associated(self%model)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP objective: adapter is not initialized")
            return
        end if
        call objective%initialize_context(self%parameter_count(), self, &
            mlp_objective_context_callback, status)
    end subroutine mlp_objective_fortopt

    subroutine mlp_objective_context_callback(context, parameters, value, &
            gradient, status)
        class(*), intent(inout) :: context
        real(dp), intent(in) :: parameters(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status

        select type (adapter => context)
            type is (mlp_training_objective_t)
            call adapter%value_gradient(parameters, value, gradient, status)
        class default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP objective: context has the wrong type")
        end select
    end subroutine mlp_objective_context_callback

    subroutine mlp_optimize_lbfgsb(model, x, target, options, result, status, &
            sample_weight)
        !! Optimize an MLP MSE+L2 objective with FortOpt L-BFGS-B.
        !!
        !! This is a full-batch objective adapter, intended for deterministic
        !! parameter fitting and outer hyperparameter experiments.  The
        !! network weights and biases are differentiated analytically through
        !! the MLP VJP; with `optimize_l2`, the scalar L2 block uses the exact
        !! derivative returned by `mlp_loss_value_gradient`.  Bounds are
        !! applied by FortOpt's projected L-BFGS-B implementation and are
        !! never implemented by a finite-difference or clipping wrapper.
        class(mlp_t), target, intent(inout) :: model
        real(dp), intent(in) :: x(:, :), target(:, :)
        type(mlp_lbfgsb_options_t), intent(in) :: options
        type(mlp_lbfgsb_result_t), intent(out) :: result
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:)
        type(mlp_training_objective_t), target :: adapter
        type(objective_t) :: objective
        type(lbfgsb_t) :: optimizer
        type(lbfgsb_options_t) :: optimizer_options
        type(lbfgsb_result_t) :: optimizer_result
        real(dp), allocatable :: parameters(:), lower(:), upper(:), gradient(:)
        integer :: n_model, n_parameters
        !! Default-initialized instances, standing in for empty
        !! structure constructors: nvfortran rejects `T()` outright,
        !! and a declared local carries the same default init.
        type(mlp_lbfgsb_result_t) :: mlp_lbfgsb_result_t_default

        result = mlp_lbfgsb_result_t_default
        if (.not. valid_lbfgsb_options(options) .or. &
            .not. valid_data(model, x, target)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP L-BFGS-B: model, data, or options are invalid")
            return
        end if

        n_model = model%parameter_count()
        if (n_model < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP L-BFGS-B: model has no parameters")
            return
        end if
        if (present(sample_weight)) then
            call adapter%initialize(model, x, target, options%l2, status, &
                optimize_l2=options%optimize_l2, sample_weight=sample_weight)
        else
            call adapter%initialize(model, x, target, options%l2, status, &
                optimize_l2=options%optimize_l2)
        end if
        if (status%code /= FORTNUM_OK) return
        n_parameters = adapter%parameter_count()
        parameters = adapter%parameters()
        allocate(lower(n_parameters), upper(n_parameters), gradient(n_parameters))
        lower(:n_model) = options%lower_bound
        upper(:n_model) = options%upper_bound
        if (options%optimize_l2) then
            lower(n_model + 1) = options%l2_lower_bound
            upper(n_model + 1) = options%l2_upper_bound
        end if

        call adapter%fortopt(objective, status)
        if (status%code /= FORTNUM_OK) return
        optimizer_options%memory = options%memory
        optimizer_options%max_iterations = options%max_iterations
        optimizer_options%max_line_search = options%max_line_search
        optimizer_options%gradient_tolerance = options%gradient_tolerance
        optimizer_options%step_tolerance = options%step_tolerance
        optimizer_options%objective_tolerance = options%objective_tolerance
        call optimizer%minimize(objective, parameters, lower, upper, &
            optimizer_options, optimizer_result, status)
        if (status%code /= FORTNUM_OK) return

        call model%set_parameters(parameters(:n_model), status)
        if (status%code /= FORTNUM_OK) return
        call adapter%value_gradient(parameters, result%objective, gradient, status)
        if (status%code /= FORTNUM_OK) return
        result%converged = optimizer_result%state%converged
        result%iterations = optimizer_result%state%iteration
        result%line_search_evaluations = optimizer_result%line_search_evaluations
        result%gradient_norm = sqrt(sum(gradient*gradient))
        result%l2 = options%l2
        if (options%optimize_l2) result%l2 = parameters(n_model + 1)
        if (.not. ieee_is_finite(result%objective) .or. &
            .not. ieee_is_finite(result%gradient_norm) .or. &
            .not. ieee_is_finite(result%l2)) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "MLP L-BFGS-B: result is not finite")
            return
        end if
        if (.not. result%converged) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "MLP L-BFGS-B: iteration limit reached")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_optimize_lbfgsb

    subroutine mlp_train(model, x, target, status, options, state, &
            validation_x, validation_target, checkpoint, sample_weight, &
            validation_weight)
        !! Train `model` with deterministic Adam, AMSGrad, AdamW, Adagrad, RMSprop, or
        !! SGD updates, as selected by `options%optimizer`.
        !!
        !! A zero batch size selects full-batch updates.  When shuffling is
        !! enabled, an explicit Park--Miller stream seeded by `shuffle_seed`
        !! drives Fisher--Yates permutations; no process-global RNG state is
        !! touched.  Callback execution occurs once per completed epoch.
        !! Optional validation arrays are evaluated at `validation_interval`;
        !! patience and best-state restoration then monitor that held-out
        !! objective without using validation rows for updates.  When
        !! `checkpoint` is present, a fresh snapshot is written after each
        !! completed epoch and an initialized snapshot resumes the same
        !! trajectory.  Set `options%use_typed_schedule` and
        !! `options%typed_schedule` to select one of the stateless built-in
        !! schedules; its fields are serialized with the checkpoint. Custom
        !! schedule callbacks and typed schedules are mutually exclusive.
        !! `options%max_epochs` is a total target epoch on both fresh and
        !! resumed calls.
        !! Optional `sample_weight` values are finite, non-negative row weights
        !! with positive total mass.  Minibatch accumulation forms the exact
        !! weighted mean by accumulating each microbatch gradient with its
        !! weight mass.  `validation_weight` applies the same contract to the
        !! optional validation objective.
        class(mlp_t), intent(inout) :: model
        real(dp), intent(in) :: x(:, :), target(:, :)
        type(fortnum_status_t), intent(out) :: status
        type(mlp_training_options_t), intent(in), optional :: options
        type(mlp_training_state_t), intent(out), optional :: state
        real(dp), intent(in), optional :: validation_x(:, :), validation_target(:, :)
        type(mlp_training_checkpoint_t), intent(inout), optional :: checkpoint
        real(dp), intent(in), optional :: sample_weight(:), validation_weight(:)
        type(mlp_training_options_t) :: config
        type(mlp_training_state_t) :: result
        type(adam_t) :: optimizer
        type(adamw_t) :: adamw_optimizer
        type(adagrad_t) :: adagrad_optimizer
        type(amsgrad_t) :: amsgrad_optimizer
        type(rmsprop_t) :: rmsprop_optimizer
        type(sgd_t) :: sgd_optimizer
        type(adafactor_t) :: adafactor_optimizer
        type(adafactor_factored_t) :: adafactor_factored_optimizer
        type(mlp_batch_iterator_t) :: iterator
        type(radam_t) :: radam_optimizer
        real(dp), allocatable :: theta(:), theta_before(:), best_theta(:), gradient(:)
        real(dp), allocatable :: scaled_gradient(:)
        real(dp), allocatable :: accumulated_gradient(:)
        real(dp), allocatable :: ema_parameters(:)
        real(dp), allocatable :: lion_momentum(:)
        real(dp), allocatable :: x_batch(:, :), target_batch(:, :)
        real(dp), allocatable :: weight_batch(:)
        real(dp), allocatable :: x_training(:, :), target_training(:, :)
        real(dp), allocatable :: validation_x_training(:, :), validation_target_training(:, :)
        integer, allocatable :: batch_indices(:)
        real(dp) :: loss, l2_gradient, gradient_norm, improvement
        real(dp) :: best_loss
        real(dp) :: validation_loss, monitored_loss
        real(dp) :: effective_rate, raw_gradient_norm
        real(dp) :: batch_weight_mass, accumulated_weight_mass
        real(dp) :: validation_weight_mass
        integer :: n_samples, n_outputs, n_parameters
        integer :: batch, epoch
        integer :: microbatch_count, accumulated_samples
        integer :: stale_epochs
        integer :: schedule_bad_updates, schedule_reductions
        integer :: next_schedule_bad_updates, next_schedule_reductions
        integer :: start_epoch, history_length
        logical :: stop_now, event_stop, has_batch, resuming, resume_active_epoch
        logical :: incompatible_checkpoint
        logical :: has_typed_schedule
        logical :: schedule_metric_initialized, schedule_improved, schedule_reduced
        real(dp) :: schedule_best_metric, next_schedule_best_metric
        type(mlp_learning_rate_schedule_t) :: schedule_config
        type(mlp_loss_scale_state_t) :: loss_scaler
        type(mlp_parameter_block_t), allocatable :: parameter_layout(:)
        type(adafactor_block_spec_t), allocatable :: adafactor_specs(:)

        resuming = .false.
        validation_loss = huge(1.0_dp)
        gradient_norm = 0.0_dp
        schedule_bad_updates = 0
        schedule_reductions = 0
        schedule_best_metric = huge(1.0_dp)
        schedule_metric_initialized = .false.
        if (present(checkpoint)) resuming = checkpoint%initialized
        resume_active_epoch = .false.
        if (present(options)) config = options
        result%precision_kind = config%precision_kind
        loss_scaler = config%loss_scale
        result%loss_scale = loss_scaler
        has_typed_schedule = config%use_typed_schedule
        schedule_config = config%typed_schedule
        if (has_typed_schedule .and. .not. schedule_config%valid()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP train: typed schedule is invalid")
            if (present(state)) state = result
            return
        end if
        if (.not. valid_options(config) .or. &
            .not. valid_data(model, x, target) .or. &
            (present(validation_x) .neqv. present(validation_target))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP train: invalid model, data, or options")
            if (present(state)) state = result
            return
        end if
        if (config%precision_kind == MLP_PRECISION_FP16 .or. &
            config%precision_kind == MLP_PRECISION_BF16) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "MLP train: fp16/bf16 storage and kernels are unavailable")
            if (present(state)) state = result
            return
        end if
        if (present(validation_x)) then
            if (.not. valid_data(model, validation_x, validation_target)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP train: validation data is invalid")
                if (present(state)) state = result
                return
            end if
        end if

        n_samples = size(x, 1)
        allocate(x_training(n_samples, size(x, 2)), target_training(size(target, 1), size(target, 2)))
        x_training = x
        target_training = target
        if (config%precision_kind == MLP_PRECISION_FP32) then
            call quantize_matrix_fp32(x_training, status, "MLP train features")
            if (status%code /= FORTNUM_OK) then
                if (present(state)) state = result
                return
            end if
            call quantize_matrix_fp32(target_training, status, "MLP train targets")
            if (status%code /= FORTNUM_OK) then
                if (present(state)) state = result
                return
            end if
        end if
        if (present(validation_x)) then
            allocate(validation_x_training(size(validation_x, 1), size(validation_x, 2)), &
                validation_target_training(size(validation_target, 1), size(validation_target, 2)))
            validation_x_training = validation_x
            validation_target_training = validation_target
            if (config%precision_kind == MLP_PRECISION_FP32) then
                call quantize_matrix_fp32(validation_x_training, status, "MLP validation features")
                if (status%code /= FORTNUM_OK) then
                    if (present(state)) state = result
                    return
                end if
                call quantize_matrix_fp32(validation_target_training, status, "MLP validation targets")
                if (status%code /= FORTNUM_OK) then
                    if (present(state)) state = result
                    return
                end if
            end if
        end if
        if (present(sample_weight)) then
            call validate_sample_weight(sample_weight, n_samples, batch_weight_mass, &
                status, "MLP train")
            if (status%code /= FORTNUM_OK) then
                if (present(state)) state = result
                return
            end if
        else
            batch_weight_mass = real(n_samples, dp)
        end if
        if (present(validation_weight)) then
            if (.not. present(validation_x)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP train: validation weights require validation data")
                if (present(state)) state = result
                return
            end if
            call validate_sample_weight(validation_weight, size(validation_x, 1), &
                validation_weight_mass, status, "MLP train validation")
            if (status%code /= FORTNUM_OK) then
                if (present(state)) state = result
                return
            end if
        else if (present(validation_x)) then
            validation_weight_mass = real(size(validation_x, 1), dp)
        end if
        n_outputs = size(target, 2)
        n_parameters = model%parameter_count()
        if (.not. optimizer_groups_fit(n_parameters, config%optimizer_groups)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP train: optimizer group range exceeds model parameters")
            if (present(state)) state = result
            return
        end if
        batch = config%batch_size
        if (batch == 0) batch = n_samples
        batch = min(batch, n_samples)

        if (resuming) then
            incompatible_checkpoint = .not. checkpoint%valid()
            if (.not. checkpoint%resume_safe) incompatible_checkpoint = .true.
            if (checkpoint%n_samples /= n_samples) incompatible_checkpoint = .true.
            if (checkpoint%n_features /= size(x, 2)) incompatible_checkpoint = .true.
            if (checkpoint%n_outputs /= n_outputs) incompatible_checkpoint = .true.
            if (checkpoint%n_parameters /= n_parameters) incompatible_checkpoint = .true.
            if (checkpoint%precision_kind /= config%precision_kind) then
                incompatible_checkpoint = .true.
            end if
            if (.not. checkpoint%loss_scale%compatible(config%loss_scale)) then
                incompatible_checkpoint = .true.
            end if
            if (checkpoint%batch_size /= batch) incompatible_checkpoint = .true.
            if (checkpoint%accumulation_steps /= config%accumulation_steps) then
                incompatible_checkpoint = .true.
            end if
            if (checkpoint%shuffle .neqv. config%shuffle) incompatible_checkpoint = .true.
            if (checkpoint%shuffle_seed /= config%shuffle_seed) incompatible_checkpoint = .true.
            if (checkpoint%learning_rate /= config%learning_rate) incompatible_checkpoint = .true.
            if (checkpoint%beta1 /= config%beta1) incompatible_checkpoint = .true.
            if (checkpoint%beta2 /= config%beta2) incompatible_checkpoint = .true.
            if (checkpoint%epsilon /= config%epsilon) incompatible_checkpoint = .true.
            if (checkpoint%adafactor_decay /= config%adafactor_decay) then
                incompatible_checkpoint = .true.
            end if
            if (checkpoint%adafactor_clip_threshold /= config%adafactor_clip_threshold) then
                incompatible_checkpoint = .true.
            end if
            if (checkpoint%adafactor_relative_step .neqv. config%adafactor_relative_step) then
                incompatible_checkpoint = .true.
            end if
            if (checkpoint%adafactor_scale_parameter .neqv. config%adafactor_scale_parameter) then
                incompatible_checkpoint = .true.
            end if
            if (checkpoint%adafactor_factored .neqv. config%adafactor_factored) then
                incompatible_checkpoint = .true.
            end if
            if (checkpoint%rmsprop_decay /= config%rmsprop_decay) then
                incompatible_checkpoint = .true.
            end if
            if (checkpoint%rmsprop_momentum /= config%rmsprop_momentum) then
                incompatible_checkpoint = .true.
            end if
            if (checkpoint%rmsprop_centered .neqv. config%rmsprop_centered) then
                incompatible_checkpoint = .true.
            end if
            if (checkpoint%optimizer /= config%optimizer) incompatible_checkpoint = .true.
            if (checkpoint%momentum /= config%momentum) incompatible_checkpoint = .true.
            if (checkpoint%nesterov .neqv. config%nesterov) incompatible_checkpoint = .true.
            if (checkpoint%weight_decay /= config%weight_decay) then
                incompatible_checkpoint = .true.
            end if
            if (checkpoint%l2 /= config%l2) incompatible_checkpoint = .true.
            if (checkpoint%validation_interval /= config%validation_interval) then
                incompatible_checkpoint = .true.
            end if
            if (checkpoint%patience /= config%patience) incompatible_checkpoint = .true.
            if (checkpoint%restore_best .neqv. config%restore_best) then
                incompatible_checkpoint = .true.
            end if
            if (checkpoint%tolerance /= config%tolerance) incompatible_checkpoint = .true.
            if (checkpoint%min_delta /= config%min_delta) incompatible_checkpoint = .true.
            if (checkpoint%gradient_clip_norm /= config%gradient_clip_norm) then
                incompatible_checkpoint = .true.
            end if
            if (checkpoint%ema_decay /= config%ema_decay) incompatible_checkpoint = .true.
            if (.not. optimizer_groups_equal(checkpoint, config%optimizer_groups)) then
                incompatible_checkpoint = .true.
            end if
            if (checkpoint%has_typed_schedule .neqv. has_typed_schedule) then
                incompatible_checkpoint = .true.
            end if
            if (checkpoint%has_typed_schedule) then
                if (.not. schedules_equal(checkpoint%typed_schedule, schedule_config)) then
                    incompatible_checkpoint = .true.
                end if
                if (schedule_config%kind == MLP_SCHEDULE_PLATEAU .and. &
                    .not. checkpoint%schedule_metric_initialized) then
                    incompatible_checkpoint = .true.
                end if
            end if
            if (checkpoint%epoch > config%max_epochs) incompatible_checkpoint = .true.
            if (incompatible_checkpoint) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP train: checkpoint is invalid or incompatible")
                if (present(state)) state = result
                return
            end if
            resume_active_epoch = checkpoint%iterator_position <= n_samples .and. &
                checkpoint%active_epoch > checkpoint%epoch
            if (checkpoint%has_validation .neqv. present(validation_x)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP train: checkpoint validation contract differs")
                if (present(state)) state = result
                return
            end if
        end if

        if (resuming) then
            theta = checkpoint%parameters
            call set_model_parameters_for_precision(model, theta, config%precision_kind, status)
            if (status%code /= FORTNUM_OK) then
                if (present(state)) state = result
                return
            end if
            allocate(best_theta, source=checkpoint%best_parameters)
            if (config%ema_decay > 0.0_dp) then
                allocate(ema_parameters, source=checkpoint%ema_parameters)
            end if
            if (config%optimizer == MLP_OPTIMIZER_LION) then
                allocate(lion_momentum, source=checkpoint%first_moment)
            end if
            schedule_bad_updates = checkpoint%schedule_bad_updates
            schedule_reductions = checkpoint%schedule_reductions
            schedule_best_metric = checkpoint%schedule_best_metric
            schedule_metric_initialized = checkpoint%schedule_metric_initialized
            loss_scaler = checkpoint%loss_scale
        else
            theta = model%parameters()
            call set_model_parameters_for_precision(model, theta, config%precision_kind, status)
            if (status%code /= FORTNUM_OK) then
                if (present(state)) state = result
                return
            end if
            allocate(best_theta, source=theta)
            if (config%ema_decay > 0.0_dp) then
                allocate(ema_parameters, source=theta)
            end if
            if (config%optimizer == MLP_OPTIMIZER_LION) then
                allocate(lion_momentum(n_parameters))
                lion_momentum = 0.0_dp
            end if
        end if
        result%loss_scale = loss_scaler
        result%has_ema = config%ema_decay > 0.0_dp
        if (result%has_ema) allocate(result%ema_parameters, source=ema_parameters)
        allocate(gradient(n_parameters))
        if (loss_scaler%enabled) allocate(scaled_gradient(n_parameters))
        if (allocated(config%optimizer_groups)) allocate(theta_before(n_parameters))
        allocate(accumulated_gradient(n_parameters))
        history_length = config%max_epochs
        allocate(result%loss_history(history_length))
        allocate(result%learning_rate_history(history_length))
        result%loss_history = huge(1.0_dp)
        result%learning_rate_history = 0.0_dp
        if (present(validation_x)) then
            allocate(result%validation_loss_history(history_length))
            result%validation_loss_history = huge(1.0_dp)
        end if
        result%accumulation_steps = config%accumulation_steps
        if (resuming) then
            result%epochs = checkpoint%epoch
            result%updates = checkpoint%updates
            result%microbatches = checkpoint%microbatches
            result%gradient_clipped_updates = checkpoint%gradient_clipped_updates
            result%converged = .false.
            result%early_stopped = .false.
            result%best_epoch = checkpoint%best_epoch
            result%best_validation_epoch = checkpoint%best_validation_epoch
            result%initial_loss = checkpoint%initial_loss
            result%final_loss = checkpoint%final_loss
            result%best_loss = checkpoint%best_loss
            result%initial_validation_loss = checkpoint%initial_validation_loss
            result%final_validation_loss = checkpoint%final_validation_loss
            result%best_validation_loss = checkpoint%best_validation_loss
            result%last_learning_rate = checkpoint%last_learning_rate
            result%schedule_bad_updates = checkpoint%schedule_bad_updates
            result%schedule_reductions = checkpoint%schedule_reductions
            result%schedule_best_metric = checkpoint%schedule_best_metric
            result%schedule_metric_initialized = checkpoint%schedule_metric_initialized
            if (result%epochs > 0) then
                result%loss_history(:result%epochs) = checkpoint%loss_history
                result%learning_rate_history(:result%epochs) = &
                    checkpoint%learning_rate_history
                if (present(validation_x)) result%validation_loss_history( &
                    :result%epochs) = checkpoint%validation_loss_history
            end if
            best_loss = checkpoint%best_loss
            stale_epochs = checkpoint%stale_epochs
            monitored_loss = checkpoint%best_loss
        else
            if (present(sample_weight)) then
                call mlp_loss_value_gradient(model, x_training, target_training, config%l2, loss, &
                    gradient, l2_gradient, status, sample_weight=sample_weight)
            else
                call mlp_loss_value_gradient(model, x_training, target_training, config%l2, loss, &
                    gradient, l2_gradient, status)
            end if
            if (status%code /= FORTNUM_OK) then
                if (present(state)) state = result
                return
            end if
            call round_gradient_for_precision(gradient, config%precision_kind, status)
            if (status%code /= FORTNUM_OK) then
                if (present(state)) state = result
                return
            end if
            result%initial_loss = loss
            best_loss = loss
            result%best_loss = loss
            monitored_loss = loss
            if (present(validation_x)) then
                if (present(validation_weight)) then
                    call mlp_loss_value_gradient(model, validation_x_training, validation_target_training, &
                        config%l2, validation_loss, gradient, l2_gradient, status, &
                        sample_weight=validation_weight)
                else
                    call mlp_loss_value_gradient(model, validation_x_training, validation_target_training, &
                        config%l2, validation_loss, gradient, l2_gradient, status)
                end if
                if (status%code /= FORTNUM_OK) then
                    if (present(state)) state = result
                    return
                end if
                call round_gradient_for_precision(gradient, config%precision_kind, status)
                if (status%code /= FORTNUM_OK) then
                    if (present(state)) state = result
                    return
                end if
                result%initial_validation_loss = validation_loss
                result%best_validation_loss = validation_loss
                best_loss = validation_loss
                monitored_loss = validation_loss
            end if
        end if
        if (has_typed_schedule .and. schedule_config%kind == MLP_SCHEDULE_PLATEAU .and. &
            .not. schedule_metric_initialized) then
            schedule_best_metric = monitored_loss
            schedule_metric_initialized = .true.
        end if
        result%schedule_bad_updates = schedule_bad_updates
        result%schedule_reductions = schedule_reductions
        result%schedule_best_metric = schedule_best_metric
        result%schedule_metric_initialized = schedule_metric_initialized
        call emit_training_event(config, MLP_EVENT_TRAIN_BEGIN, 0, result%updates, &
            result%initial_loss, result%initial_validation_loss, 0.0_dp, &
            config%learning_rate, event_stop, status)
        if (status%code /= FORTNUM_OK) then
            if (present(state)) state = result
            return
        end if
        if (event_stop) then
            result%early_stopped = .true.
            if (present(state)) state = result
            call status_set(status, FORTNUM_OK, "")
            return
        end if
        if (config%optimizer == MLP_OPTIMIZER_ADAM) then
            call optimizer%initialize(n_parameters, status, &
                learning_rate=config%learning_rate, beta1=config%beta1, &
                beta2=config%beta2, epsilon=config%epsilon)
        else if (config%optimizer == MLP_OPTIMIZER_SGD) then
            call sgd_optimizer%initialize(n_parameters, status, &
                learning_rate=config%learning_rate, momentum=config%momentum, &
                nesterov=config%nesterov)
        else if (config%optimizer == MLP_OPTIMIZER_ADAMW) then
            call adamw_optimizer%initialize(n_parameters, status, &
                learning_rate=config%learning_rate, beta1=config%beta1, &
                beta2=config%beta2, epsilon=config%epsilon, &
                weight_decay=config%weight_decay)
        else if (config%optimizer == MLP_OPTIMIZER_ADAGRAD) then
            call adagrad_optimizer%initialize(n_parameters, status, &
                learning_rate=config%learning_rate, epsilon=config%epsilon)
        else if (config%optimizer == MLP_OPTIMIZER_ADAFACTOR) then
            if (config%adafactor_factored) then
                parameter_layout = model%parameter_layout()
                allocate(adafactor_specs(size(parameter_layout)))
                call adafactor_specs_from_layout(parameter_layout, adafactor_specs, status)
                if (status%code /= FORTNUM_OK) then
                    if (present(state)) state = result
                    return
                end if
                call adafactor_factored_optimizer%initialize(n_parameters, adafactor_specs, status, &
                    learning_rate=config%learning_rate, decay=config%adafactor_decay, &
                    epsilon=config%epsilon, clip_threshold=config%adafactor_clip_threshold, &
                    relative_step=config%adafactor_relative_step, &
                    scale_parameter=config%adafactor_scale_parameter)
            else
                call adafactor_optimizer%initialize(n_parameters, status, &
                    learning_rate=config%learning_rate, decay=config%adafactor_decay, &
                    epsilon=config%epsilon, clip_threshold=config%adafactor_clip_threshold, &
                    relative_step=config%adafactor_relative_step, &
                    scale_parameter=config%adafactor_scale_parameter)
            end if
        else if (config%optimizer == MLP_OPTIMIZER_AMSGRAD) then
            call amsgrad_optimizer%initialize(n_parameters, status, &
                learning_rate=config%learning_rate, beta1=config%beta1, &
                beta2=config%beta2, epsilon=config%epsilon)
        else if (config%optimizer == MLP_OPTIMIZER_RADAM) then
            call radam_optimizer%initialize(n_parameters, status, &
                learning_rate=config%learning_rate, beta1=config%beta1, &
                beta2=config%beta2, epsilon=config%epsilon)
        else if (config%optimizer == MLP_OPTIMIZER_LION) then
            ! Lion keeps one interpolated-gradient momentum vector.  It is
            ! initialized above and is restored from checkpoint%first_moment.
            call status_set(status, FORTNUM_OK, "")
        else
            call rmsprop_optimizer%initialize(n_parameters, status, &
                learning_rate=config%learning_rate, decay=config%rmsprop_decay, &
                epsilon=config%epsilon, momentum=config%rmsprop_momentum, &
                centered=config%rmsprop_centered)
        end if
        if (status%code /= FORTNUM_OK) then
            if (present(state)) state = result
            return
        end if
        if (resuming) then
            if (config%optimizer == MLP_OPTIMIZER_ADAM) then
                optimizer%first_moment = checkpoint%first_moment
                optimizer%second_moment = checkpoint%second_moment
                optimizer%step_count = checkpoint%adam_step_count
                optimizer%learning_rate = checkpoint%last_learning_rate
                if (optimizer%learning_rate <= 0.0_dp) then
                    optimizer%learning_rate = config%learning_rate
                end if
            else if (config%optimizer == MLP_OPTIMIZER_SGD) then
                sgd_optimizer%velocity = checkpoint%first_moment
                sgd_optimizer%step_count = checkpoint%adam_step_count
                sgd_optimizer%learning_rate = checkpoint%last_learning_rate
                if (sgd_optimizer%learning_rate <= 0.0_dp) then
                    sgd_optimizer%learning_rate = config%learning_rate
                end if
            else if (config%optimizer == MLP_OPTIMIZER_ADAMW) then
                adamw_optimizer%first_moment = checkpoint%first_moment
                adamw_optimizer%second_moment = checkpoint%second_moment
                adamw_optimizer%step_count = checkpoint%adam_step_count
                adamw_optimizer%learning_rate = checkpoint%last_learning_rate
                if (adamw_optimizer%learning_rate <= 0.0_dp) then
                    adamw_optimizer%learning_rate = config%learning_rate
                end if
            else if (config%optimizer == MLP_OPTIMIZER_ADAGRAD) then
                adagrad_optimizer%accumulated_square = checkpoint%first_moment
                adagrad_optimizer%step_count = checkpoint%adam_step_count
                adagrad_optimizer%learning_rate = checkpoint%last_learning_rate
                if (adagrad_optimizer%learning_rate <= 0.0_dp) then
                    adagrad_optimizer%learning_rate = config%learning_rate
                end if
            else if (config%optimizer == MLP_OPTIMIZER_ADAFACTOR .and. &
                    .not. config%adafactor_factored) then
                adafactor_optimizer%second_moment = checkpoint%first_moment
                adafactor_optimizer%step_count = checkpoint%adam_step_count
                adafactor_optimizer%learning_rate = checkpoint%last_learning_rate
                if (adafactor_optimizer%learning_rate <= 0.0_dp) then
                    adafactor_optimizer%learning_rate = config%learning_rate
                end if
            else if (config%optimizer == MLP_OPTIMIZER_ADAFACTOR .and. &
                    config%adafactor_factored) then
                call restore_factored_adafactor_checkpoint(checkpoint, &
                    adafactor_factored_optimizer, adafactor_specs, status)
                if (status%code /= FORTNUM_OK) then
                    if (present(state)) state = result
                    return
                end if
                adafactor_factored_optimizer%step_count = checkpoint%adam_step_count
                adafactor_factored_optimizer%learning_rate = checkpoint%last_learning_rate
                if (adafactor_factored_optimizer%learning_rate <= 0.0_dp) then
                    adafactor_factored_optimizer%learning_rate = config%learning_rate
                end if
            else if (config%optimizer == MLP_OPTIMIZER_AMSGRAD) then
                amsgrad_optimizer%first_moment = checkpoint%first_moment
                amsgrad_optimizer%second_moment = checkpoint%second_moment
                amsgrad_optimizer%max_second_moment = checkpoint%max_second_moment
                amsgrad_optimizer%step_count = checkpoint%adam_step_count
                amsgrad_optimizer%learning_rate = checkpoint%last_learning_rate
                if (amsgrad_optimizer%learning_rate <= 0.0_dp) then
                    amsgrad_optimizer%learning_rate = config%learning_rate
                end if
            else if (config%optimizer == MLP_OPTIMIZER_RADAM) then
                radam_optimizer%first_moment = checkpoint%first_moment
                radam_optimizer%second_moment = checkpoint%second_moment
                radam_optimizer%step_count = checkpoint%adam_step_count
                radam_optimizer%learning_rate = checkpoint%last_learning_rate
                if (radam_optimizer%learning_rate <= 0.0_dp) then
                    radam_optimizer%learning_rate = config%learning_rate
                end if
            else if (config%optimizer == MLP_OPTIMIZER_LION) then
                ! Lion has no bias-correction counter; the first-moment
                ! checkpoint slot stores its single momentum vector.
            else
                rmsprop_optimizer%square_average = checkpoint%first_moment
                rmsprop_optimizer%gradient_average = checkpoint%second_moment
                rmsprop_optimizer%momentum_buffer = checkpoint%rmsprop_buffer
                rmsprop_optimizer%step_count = checkpoint%adam_step_count
                rmsprop_optimizer%learning_rate = checkpoint%last_learning_rate
                if (rmsprop_optimizer%learning_rate <= 0.0_dp) then
                    rmsprop_optimizer%learning_rate = config%learning_rate
                end if
            end if
        end if
        call iterator%initialize(n_samples, status, batch_size=batch, &
            shuffle=config%shuffle, seed=config%shuffle_seed)
        if (status%code /= FORTNUM_OK) then
            if (present(state)) state = result
            return
        end if
        if (resuming) then
            iterator%order = checkpoint%iterator_order
            iterator%position = checkpoint%iterator_position
            iterator%epoch_number = checkpoint%iterator_epoch
            iterator%shuffle_state = checkpoint%shuffle_state
        end if

        start_epoch = 1
        if (resuming) start_epoch = checkpoint%epoch + 1
        if (resuming .and. resume_active_epoch) start_epoch = checkpoint%active_epoch
        do epoch = start_epoch, config%max_epochs
            if (.not. (resuming .and. resume_active_epoch .and. &
                epoch == start_epoch)) then
                call iterator%reset(status)
                if (status%code /= FORTNUM_OK) then
                    if (present(state)) state = result
                    return
                end if
            end if
            accumulated_gradient = 0.0_dp
            accumulated_samples = 0
            accumulated_weight_mass = 0.0_dp
            microbatch_count = 0
            if (resuming .and. resume_active_epoch .and. epoch == start_epoch) then
                accumulated_gradient = checkpoint%accumulated_gradient
                accumulated_samples = checkpoint%accumulated_samples
                accumulated_weight_mass = checkpoint%accumulated_weight_mass
                if (accumulated_weight_mass <= 0.0_dp .and. accumulated_samples > 0) then
                    accumulated_weight_mass = real(accumulated_samples, dp)
                end if
                microbatch_count = checkpoint%active_microbatches
                resume_active_epoch = .false.
            end if
            has_batch = .true.
            do while (has_batch)
                call iterator%next_batch(batch_indices, has_batch, status)
                if (status%code /= FORTNUM_OK) then
                    if (present(state)) state = result
                    return
                end if
                if (.not. has_batch) exit
                allocate(x_batch(size(batch_indices), size(x, 2)))
                allocate(target_batch(size(batch_indices), n_outputs))
                allocate(weight_batch(size(batch_indices)))
                x_batch = x_training(batch_indices, :)
                target_batch = target_training(batch_indices, :)
                if (present(sample_weight)) then
                    weight_batch = sample_weight(batch_indices)
                else
                    weight_batch = 1.0_dp
                end if
                batch_weight_mass = sum(weight_batch)
                ! Accumulate only the data term.  Add the L2 penalty once at
                ! the optimizer boundary so it is not counted once per
                ! microbatch.
                if (batch_weight_mass > 0.0_dp) then
                    call mlp_loss_value_gradient(model, x_batch, target_batch, &
                        0.0_dp, loss, gradient, l2_gradient, status, &
                        sample_weight=weight_batch)
                else
                    loss = 0.0_dp
                    gradient = 0.0_dp
                    l2_gradient = 0.0_dp
                    call status_set(status, FORTNUM_OK, "")
                end if
                deallocate(x_batch, target_batch, weight_batch)
                if (status%code /= FORTNUM_OK) then
                    if (present(state)) state = result
                    return
                end if
                call round_gradient_for_precision(gradient, config%precision_kind, status)
                if (status%code /= FORTNUM_OK) then
                    if (present(state)) state = result
                    return
                end if
                accumulated_gradient = accumulated_gradient + &
                    batch_weight_mass*gradient
                if (config%precision_kind == MLP_PRECISION_FP32) then
                    call round_gradient_for_precision(accumulated_gradient, &
                        config%precision_kind, status)
                    if (status%code /= FORTNUM_OK) then
                        if (present(state)) state = result
                        return
                    end if
                end if
                accumulated_samples = accumulated_samples + size(batch_indices)
                accumulated_weight_mass = accumulated_weight_mass + batch_weight_mass
                microbatch_count = microbatch_count + 1
                result%microbatches = result%microbatches + 1

                ! Flush at the configured boundary or at the uneven end of an
                ! epoch.  The latter keeps every sample in the final update.
                if (microbatch_count >= config%accumulation_steps .or. &
                    iterator%current_position() > n_samples) then
                    if (microbatch_count > 0) then
                        if (accumulated_weight_mass > 0.0_dp) then
                            gradient = accumulated_gradient/accumulated_weight_mass
                            if (config%precision_kind == MLP_PRECISION_FP64) then
                                theta = model%parameters()
                            end if
                            gradient = gradient + config%l2*theta
                            call round_gradient_for_precision(gradient, config%precision_kind, status)
                            if (status%code /= FORTNUM_OK) then
                                if (present(state)) state = result
                                return
                            end if
                            if (loss_scaler%enabled) then
                                call loss_scaler%scale_gradient(gradient, scaled_gradient, status)
                                if (status%code /= FORTNUM_OK) then
                                    if (present(state)) state = result
                                    return
                                end if
                                if (.not. loss_scaler%scaled_gradient_finite(scaled_gradient)) then
                                    call loss_scaler%observe(.false., .false., status)
                                    if (status%code /= FORTNUM_OK) then
                                        if (present(state)) state = result
                                        return
                                    end if
                                    result%loss_scale = loss_scaler
                                    accumulated_gradient = 0.0_dp
                                    accumulated_samples = 0
                                    accumulated_weight_mass = 0.0_dp
                                    microbatch_count = 0
                                    cycle
                                end if
                                call loss_scaler%unscale_gradient(scaled_gradient, gradient, status)
                                if (status%code /= FORTNUM_OK) then
                                    if (present(state)) state = result
                                    return
                                end if
                            end if
                            raw_gradient_norm = sqrt(sum(gradient*gradient))
                            if (config%gradient_clip_norm > 0.0_dp .and. &
                                raw_gradient_norm > config%gradient_clip_norm) then
                                gradient = gradient*config%gradient_clip_norm/ &
                                    raw_gradient_norm
                                result%gradient_clipped_updates = &
                                    result%gradient_clipped_updates + 1
                            end if
                            effective_rate = config%learning_rate
                            if (has_typed_schedule) then
                                if (schedule_config%kind == MLP_SCHEDULE_PLATEAU) then
                                    effective_rate = config%learning_rate*schedule_config%plateau_factor** &
                                        schedule_reductions
                                    if (.not. ieee_is_finite(effective_rate)) then
                                        call status_set(status, FORTNUM_DOMAIN_ERROR, &
                                            "MLP train: plateau schedule rate is not finite")
                                        if (present(state)) state = result
                                        return
                                    end if
                                else
                                    call schedule_config%rate(result%updates + 1, &
                                        config%learning_rate, effective_rate, status)
                                end if
                                if (status%code /= FORTNUM_OK) then
                                    if (present(state)) state = result
                                    return
                                end if
                            else if (associated(config%learning_rate_schedule)) then
                                call config%learning_rate_schedule(epoch, &
                                    result%updates + 1, config%learning_rate, &
                                    effective_rate)
                            end if
                            if (.not. ieee_is_finite(effective_rate) .or. &
                                effective_rate <= 0.0_dp) then
                                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                                    "MLP train: schedule returned an invalid learning rate")
                                if (present(state)) state = result
                                return
                            end if
                            if (config%optimizer == MLP_OPTIMIZER_ADAM) then
                                optimizer%learning_rate = effective_rate
                            else if (config%optimizer == MLP_OPTIMIZER_SGD) then
                                sgd_optimizer%learning_rate = effective_rate
                            else if (config%optimizer == MLP_OPTIMIZER_ADAMW) then
                                adamw_optimizer%learning_rate = effective_rate
                            else if (config%optimizer == MLP_OPTIMIZER_ADAGRAD) then
                                adagrad_optimizer%learning_rate = effective_rate
                            else if (config%optimizer == MLP_OPTIMIZER_ADAFACTOR) then
                                adafactor_optimizer%learning_rate = effective_rate
                                if (config%adafactor_factored) then
                                    adafactor_factored_optimizer%learning_rate = effective_rate
                                end if
                            else if (config%optimizer == MLP_OPTIMIZER_AMSGRAD) then
                                amsgrad_optimizer%learning_rate = effective_rate
                            else if (config%optimizer == MLP_OPTIMIZER_RADAM) then
                                radam_optimizer%learning_rate = effective_rate
                            else if (config%optimizer == MLP_OPTIMIZER_LION) then
                                ! Lion consumes the effective rate directly below.
                            else
                                rmsprop_optimizer%learning_rate = effective_rate
                            end if
                            result%last_learning_rate = effective_rate
                            if (allocated(theta_before)) theta_before = theta
                            if (config%optimizer == MLP_OPTIMIZER_ADAM) then
                                call optimizer%step(theta, gradient, status)
                            else if (config%optimizer == MLP_OPTIMIZER_SGD) then
                                call sgd_optimizer%step(theta, gradient, status)
                            else if (config%optimizer == MLP_OPTIMIZER_ADAMW) then
                                call adamw_optimizer%step(theta, gradient, status)
                            else if (config%optimizer == MLP_OPTIMIZER_ADAGRAD) then
                                call adagrad_optimizer%step(theta, gradient, status)
                            else if (config%optimizer == MLP_OPTIMIZER_ADAFACTOR) then
                                if (config%adafactor_factored) then
                                    call adafactor_factored_optimizer%step(theta, gradient, status)
                                else
                                    call adafactor_optimizer%step(theta, gradient, status)
                                end if
                            else if (config%optimizer == MLP_OPTIMIZER_AMSGRAD) then
                                call amsgrad_optimizer%step(theta, gradient, status)
                            else if (config%optimizer == MLP_OPTIMIZER_RADAM) then
                                call radam_optimizer%step(theta, gradient, status)
                            else if (config%optimizer == MLP_OPTIMIZER_LION) then
                                call lion_step(theta, gradient, lion_momentum, config%beta1, &
                                    config%beta2, effective_rate, config%weight_decay, status)
                            else
                                call rmsprop_optimizer%step(theta, gradient, status)
                            end if
                            if (status%code /= FORTNUM_OK) then
                                if (present(state)) state = result
                                return
                            end if
                            if (allocated(config%optimizer_groups)) then
                                call apply_optimizer_group_scales(theta, theta_before, &
                                    config%optimizer_groups, status)
                                if (status%code /= FORTNUM_OK) then
                                    if (present(state)) state = result
                                    return
                                end if
                            end if
                            call set_model_parameters_for_precision(model, theta, &
                                config%precision_kind, status)
                            if (status%code /= FORTNUM_OK) then
                                if (present(state)) state = result
                                return
                            end if
                            if (config%ema_decay > 0.0_dp) then
                                ema_parameters = config%ema_decay*ema_parameters + &
                                    (1.0_dp-config%ema_decay)*theta
                                result%ema_parameters = ema_parameters
                            end if
                            call loss_scaler%observe(.true., .true., status)
                            if (status%code /= FORTNUM_OK) then
                                if (present(state)) state = result
                                return
                            end if
                            result%loss_scale = loss_scaler
                            result%updates = result%updates + 1
                            call emit_training_event(config, MLP_EVENT_UPDATE, epoch, &
                                result%updates, loss, validation_loss, &
                                raw_gradient_norm, effective_rate, event_stop, status)
                            if (status%code /= FORTNUM_OK) then
                                if (present(state)) state = result
                                return
                            end if
                            if (event_stop) result%early_stopped = .true.
                        end if
                        accumulated_gradient = 0.0_dp
                        accumulated_samples = 0
                        accumulated_weight_mass = 0.0_dp
                        microbatch_count = 0
                    end if
                end if
                if (present(checkpoint)) then
                    call checkpoint_capture(checkpoint, x, target, config, result, &
                        loss_scaler, &
                        iterator, optimizer, adamw_optimizer, adagrad_optimizer, &
                        rmsprop_optimizer, adafactor_optimizer, adafactor_factored_optimizer, &
                        amsgrad_optimizer, &
                        radam_optimizer, sgd_optimizer, lion_momentum, theta, &
                        best_theta, stale_epochs, &
                        epoch, microbatch_count, accumulated_samples, &
                        accumulated_weight_mass, &
                        accumulated_gradient, ema_parameters, has_typed_schedule, &
                        schedule_config, present(validation_x), schedule_bad_updates, &
                        schedule_reductions, schedule_best_metric, &
                        schedule_metric_initialized, status)
                    if (status%code /= FORTNUM_OK) then
                        if (present(state)) state = result
                        return
                    end if
                    call emit_training_event(config, MLP_EVENT_CHECKPOINT, epoch, &
                        result%updates, loss, validation_loss, gradient_norm, &
                        result%last_learning_rate, event_stop, status)
                    if (status%code /= FORTNUM_OK) then
                        if (present(state)) state = result
                        return
                    end if
                    if (event_stop) result%early_stopped = .true.
                end if
            end do

            if (present(sample_weight)) then
                call mlp_loss_value_gradient(model, x_training, target_training, config%l2, loss, &
                    gradient, l2_gradient, status, sample_weight=sample_weight)
            else
                call mlp_loss_value_gradient(model, x_training, target_training, config%l2, loss, &
                    gradient, l2_gradient, status)
            end if
            if (status%code /= FORTNUM_OK) then
                if (present(state)) state = result
                return
            end if
            call round_gradient_for_precision(gradient, config%precision_kind, status)
            if (status%code /= FORTNUM_OK) then
                if (present(state)) state = result
                return
            end if
            gradient_norm = sqrt(sum(gradient*gradient))
            result%epochs = epoch
            result%loss_history(epoch) = loss
            result%learning_rate_history(epoch) = result%last_learning_rate
            result%gradient_norm = gradient_norm
            if (present(validation_x)) then
                if (mod(epoch, config%validation_interval) == 0) then
                    if (present(validation_weight)) then
                        call mlp_loss_value_gradient(model, validation_x_training, validation_target_training, &
                            config%l2, validation_loss, gradient, l2_gradient, status, &
                            sample_weight=validation_weight)
                    else
                        call mlp_loss_value_gradient(model, validation_x_training, validation_target_training, &
                            config%l2, validation_loss, gradient, l2_gradient, status)
                    end if
                    if (status%code /= FORTNUM_OK) then
                        if (present(state)) state = result
                        return
                    end if
                    call round_gradient_for_precision(gradient, config%precision_kind, status)
                    if (status%code /= FORTNUM_OK) then
                        if (present(state)) state = result
                        return
                    end if
                    result%validation_loss_history(epoch) = validation_loss
                    monitored_loss = validation_loss
                    call emit_training_event(config, MLP_EVENT_VALIDATION, epoch, &
                        result%updates, loss, validation_loss, gradient_norm, &
                        result%last_learning_rate, event_stop, status)
                    if (status%code /= FORTNUM_OK) then
                        if (present(state)) state = result
                        return
                    end if
                    if (event_stop) result%early_stopped = .true.
                end if
            else
                monitored_loss = loss
            end if
            if (has_typed_schedule .and. schedule_config%kind == MLP_SCHEDULE_PLATEAU) then
                call schedule_config%rate_with_metric(epoch, config%learning_rate, &
                    monitored_loss, schedule_best_metric, schedule_bad_updates, &
                    schedule_reductions, effective_rate, next_schedule_best_metric, &
                    next_schedule_bad_updates, next_schedule_reductions, schedule_improved, &
                    schedule_reduced, status)
                if (status%code /= FORTNUM_OK) then
                    if (present(state)) state = result
                    return
                end if
                schedule_best_metric = next_schedule_best_metric
                schedule_bad_updates = next_schedule_bad_updates
                schedule_reductions = next_schedule_reductions
                schedule_metric_initialized = .true.
                result%schedule_bad_updates = schedule_bad_updates
                result%schedule_reductions = schedule_reductions
                result%schedule_best_metric = schedule_best_metric
                result%schedule_metric_initialized = schedule_metric_initialized
                result%last_learning_rate = effective_rate
                result%learning_rate_history(epoch) = effective_rate
            end if
            if (present(validation_x)) then
                if (mod(epoch, config%validation_interval) /= 0) then
                    improvement = -1.0_dp
                else
                    improvement = best_loss - monitored_loss
                end if
            else
                improvement = best_loss - monitored_loss
            end if
            if (improvement > config%min_delta) then
                best_loss = monitored_loss
                result%best_loss = monitored_loss
                result%best_epoch = epoch
                if (present(validation_x)) then
                    result%best_validation_loss = monitored_loss
                    result%best_validation_epoch = epoch
                end if
                best_theta = theta
                stale_epochs = 0
            else
                if (present(validation_x)) then
                    if (mod(epoch, config%validation_interval) == 0) then
                        stale_epochs = stale_epochs + 1
                    end if
                else
                    stale_epochs = stale_epochs + 1
                end if
            end if
            stop_now = result%early_stopped
            if (associated(config%callback)) then
                call config%callback(epoch, loss, gradient_norm, stop_now)
            end if
            call emit_training_event(config, MLP_EVENT_EPOCH_END, epoch, &
                result%updates, loss, validation_loss, gradient_norm, &
                result%last_learning_rate, event_stop, status)
            if (status%code /= FORTNUM_OK) then
                if (present(state)) state = result
                return
            end if
            if (event_stop) stop_now = .true.
            if (gradient_norm <= config%tolerance) then
                result%converged = .true.
            end if
            if (stop_now) then
                result%early_stopped = .true.
            end if
            if (config%patience > 0 .and. stale_epochs >= config%patience) then
                result%early_stopped = .true.
            end if
            if (present(checkpoint)) then
                call checkpoint_capture(checkpoint, x, target, config, result, &
                    loss_scaler, &
                    iterator, optimizer, adamw_optimizer, adagrad_optimizer, &
                    rmsprop_optimizer, adafactor_optimizer, adafactor_factored_optimizer, &
                    amsgrad_optimizer, &
                    radam_optimizer, sgd_optimizer, lion_momentum, theta, &
                    best_theta, stale_epochs, &
                    epoch, microbatch_count, accumulated_samples, &
                    accumulated_weight_mass, &
                    accumulated_gradient, ema_parameters, has_typed_schedule, &
                    schedule_config, present(validation_x), schedule_bad_updates, &
                    schedule_reductions, schedule_best_metric, &
                    schedule_metric_initialized, status)
                if (status%code /= FORTNUM_OK) then
                    if (present(state)) state = result
                    return
                end if
                call emit_training_event(config, MLP_EVENT_CHECKPOINT, epoch, &
                    result%updates, loss, validation_loss, gradient_norm, &
                    result%last_learning_rate, event_stop, status)
                if (status%code /= FORTNUM_OK) then
                    if (present(state)) state = result
                    return
                end if
                if (event_stop) result%early_stopped = .true.
            end if
            if (result%converged .or. result%early_stopped) then
                exit
            end if
        end do

        call shrink_history(result%loss_history, result%epochs)
        call shrink_history(result%learning_rate_history, result%epochs)
        if (present(validation_x)) then
            call shrink_history(result%validation_loss_history, result%epochs)
        end if
        if (config%restore_best .and. result%best_epoch < result%epochs) then
            theta = best_theta
            call set_model_parameters_for_precision(model, theta, config%precision_kind, status)
            if (status%code /= FORTNUM_OK) then
                if (present(state)) state = result
                return
            end if
            if (present(sample_weight)) then
                call mlp_loss_value_gradient(model, x_training, target_training, config%l2, loss, &
                    gradient, l2_gradient, status, sample_weight=sample_weight)
            else
                call mlp_loss_value_gradient(model, x_training, target_training, config%l2, loss, &
                    gradient, l2_gradient, status)
            end if
            if (status%code /= FORTNUM_OK) then
                if (present(state)) state = result
                return
            end if
            call round_gradient_for_precision(gradient, config%precision_kind, status)
            if (status%code /= FORTNUM_OK) then
                if (present(state)) state = result
                return
            end if
            result%gradient_norm = sqrt(sum(gradient*gradient))
            if (present(checkpoint)) checkpoint%resume_safe = .false.
        end if
        result%final_loss = loss
        if (present(validation_x)) then
            if (present(validation_weight)) then
                call mlp_loss_value_gradient(model, validation_x_training, validation_target_training, &
                    config%l2, validation_loss, gradient, l2_gradient, status, &
                    sample_weight=validation_weight)
            else
                call mlp_loss_value_gradient(model, validation_x_training, validation_target_training, &
                    config%l2, validation_loss, gradient, l2_gradient, status)
            end if
            if (status%code /= FORTNUM_OK) then
                if (present(state)) state = result
                return
            end if
            call round_gradient_for_precision(gradient, config%precision_kind, status)
            if (status%code /= FORTNUM_OK) then
                if (present(state)) state = result
                return
            end if
            result%final_validation_loss = validation_loss
        end if
        ! Keep the public model at the binary64 master state after the final
        ! FP32 evaluation.  The forward model is rounded again at the start
        ! of every resumed call, while callers retain the usual model API.
        if (config%precision_kind == MLP_PRECISION_FP32) then
            call model%set_parameters(theta, status)
            if (status%code /= FORTNUM_OK) then
                if (present(state)) state = result
                return
            end if
        end if
        call emit_training_event(config, MLP_EVENT_TRAIN_END, result%epochs, &
            result%updates, result%final_loss, result%final_validation_loss, &
            result%gradient_norm, result%last_learning_rate, event_stop, status)
        if (status%code /= FORTNUM_OK) then
            if (present(state)) state = result
            return
        end if
        if (present(state)) state = result
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_train

    subroutine lion_step(theta, gradient, momentum, beta1, beta2, learning_rate, &
            weight_decay, status)
        !! One Lion update with the canonical sign/interpolation recurrence.
        !!
        !! `momentum` is the beta2 exponential moving average.  The update
        !! direction uses the beta1 interpolation before the momentum state is
        !! advanced.  Weight decay is decoupled and therefore does not alter
        !! the sign branch; a zero interpolated component has a zero update.
        real(dp), intent(inout) :: theta(:), momentum(:)
        real(dp), intent(in) :: gradient(:), beta1, beta2, learning_rate
        real(dp), intent(in) :: weight_decay
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: interpolated(:), update(:)
        integer :: i

        if (size(theta) < 1 .or. size(momentum) /= size(theta) .or. &
            size(gradient) /= size(theta) .or. beta1 < 0.0_dp .or. &
            beta1 >= 1.0_dp .or. beta2 < 0.0_dp .or. beta2 >= 1.0_dp .or. &
            learning_rate <= 0.0_dp .or. weight_decay < 0.0_dp .or. &
            .not. all(ieee_is_finite(theta)) .or. &
            .not. all(ieee_is_finite(gradient)) .or. &
            .not. all(ieee_is_finite(momentum))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP Lion: update state or options are invalid")
            return
        end if
        allocate(interpolated(size(theta)), update(size(theta)))
        interpolated = beta1*momentum + (1.0_dp-beta1)*gradient
        do i = 1, size(theta)
            if (interpolated(i) > 0.0_dp) then
                update(i) = 1.0_dp
            else if (interpolated(i) < 0.0_dp) then
                update(i) = -1.0_dp
            else
                update(i) = 0.0_dp
            end if
        end do
        theta = theta - learning_rate*(update + weight_decay*theta)
        momentum = beta2*momentum + (1.0_dp-beta2)*gradient
        if (any(.not. ieee_is_finite(theta)) .or. any(.not. ieee_is_finite(momentum))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP Lion: update produced a non-finite state")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine lion_step

    subroutine checkpoint_capture(checkpoint, x, target, config, result, loss_scaler, iterator, &
            optimizer, adamw_optimizer, adagrad_optimizer, rmsprop_optimizer, &
            adafactor_optimizer, adafactor_factored_optimizer, amsgrad_optimizer, &
            radam_optimizer, sgd_optimizer, lion_momentum, theta, best_theta, &
            stale_epochs, active_epoch, &
            active_microbatches, accumulated_samples, accumulated_weight_mass, &
            accumulated_gradient, &
            ema_parameters, has_typed_schedule, typed_schedule, has_validation, &
            schedule_bad_updates, schedule_reductions, schedule_best_metric, &
            schedule_metric_initialized, status)
        type(mlp_training_checkpoint_t), intent(inout) :: checkpoint
        real(dp), intent(in) :: x(:, :), target(:, :)
        type(mlp_training_options_t), intent(in) :: config
        type(mlp_training_state_t), intent(in) :: result
        type(mlp_loss_scale_state_t), intent(in) :: loss_scaler
        type(mlp_batch_iterator_t), intent(in) :: iterator
        type(adam_t), intent(in) :: optimizer
        type(adamw_t), intent(in) :: adamw_optimizer
        type(adagrad_t), intent(in) :: adagrad_optimizer
        type(rmsprop_t), intent(in) :: rmsprop_optimizer
        type(adafactor_t), intent(in) :: adafactor_optimizer
        type(adafactor_factored_t), intent(in) :: adafactor_factored_optimizer
        type(amsgrad_t), intent(in) :: amsgrad_optimizer
        type(radam_t), intent(in) :: radam_optimizer
        type(sgd_t), intent(in) :: sgd_optimizer
        real(dp), allocatable, intent(in) :: lion_momentum(:)
        real(dp), intent(in) :: theta(:), best_theta(:)
        integer, intent(in) :: stale_epochs
        integer, intent(in) :: active_epoch, active_microbatches, accumulated_samples
        real(dp), intent(in) :: accumulated_weight_mass
        real(dp), intent(in) :: accumulated_gradient(:)
        real(dp), allocatable, intent(in) :: ema_parameters(:)
        logical, intent(in) :: has_typed_schedule
        type(mlp_learning_rate_schedule_t), intent(in) :: typed_schedule
        logical, intent(in) :: has_validation
        integer, intent(in) :: schedule_bad_updates, schedule_reductions
        real(dp), intent(in) :: schedule_best_metric
        logical, intent(in) :: schedule_metric_initialized
        type(fortnum_status_t), intent(out) :: status
        integer :: n
        logical :: invalid_state

        invalid_state = size(theta) < 1 .or. size(best_theta) /= size(theta) .or. &
            size(accumulated_gradient) /= size(theta) .or. &
            active_epoch < result%epochs .or. active_microbatches < 0 .or. &
            accumulated_samples < 0 .or. .not. iterator%initialized() .or. &
            schedule_bad_updates < 0 .or. schedule_reductions < 0 .or. &
            .not. ieee_is_finite(schedule_best_metric)
        if (has_typed_schedule .and. typed_schedule%kind == MLP_SCHEDULE_PLATEAU .and. &
            .not. schedule_metric_initialized) invalid_state = .true.
        if (config%ema_decay > 0.0_dp) then
            if (.not. allocated(ema_parameters)) then
                invalid_state = .true.
            else if (size(ema_parameters) /= size(theta)) then
                invalid_state = .true.
            end if
        else if (allocated(ema_parameters)) then
            invalid_state = .true.
        end if
        if (config%optimizer == MLP_OPTIMIZER_ADAM) then
            if (.not. allocated(optimizer%first_moment)) invalid_state = .true.
            if (.not. allocated(optimizer%second_moment)) invalid_state = .true.
            if (allocated(optimizer%first_moment)) then
                if (size(optimizer%first_moment) /= size(theta)) invalid_state = .true.
            end if
            if (allocated(optimizer%second_moment)) then
                if (size(optimizer%second_moment) /= size(theta)) invalid_state = .true.
            end if
        else if (config%optimizer == MLP_OPTIMIZER_SGD) then
            if (.not. allocated(sgd_optimizer%velocity)) invalid_state = .true.
            if (allocated(sgd_optimizer%velocity)) then
                if (size(sgd_optimizer%velocity) /= size(theta)) invalid_state = .true.
            end if
        else if (config%optimizer == MLP_OPTIMIZER_ADAMW) then
            if (.not. allocated(adamw_optimizer%first_moment)) invalid_state = .true.
            if (.not. allocated(adamw_optimizer%second_moment)) invalid_state = .true.
            if (allocated(adamw_optimizer%first_moment)) then
                if (size(adamw_optimizer%first_moment) /= size(theta)) then
                    invalid_state = .true.
                end if
            end if
            if (allocated(adamw_optimizer%second_moment)) then
                if (size(adamw_optimizer%second_moment) /= size(theta)) then
                    invalid_state = .true.
                end if
            end if
        else if (config%optimizer == MLP_OPTIMIZER_ADAGRAD) then
            if (.not. allocated(adagrad_optimizer%accumulated_square)) then
                invalid_state = .true.
            end if
            if (allocated(adagrad_optimizer%accumulated_square)) then
                if (size(adagrad_optimizer%accumulated_square) /= size(theta)) then
                    invalid_state = .true.
                end if
            end if
        else if (config%optimizer == MLP_OPTIMIZER_ADAFACTOR) then
            if (config%adafactor_factored) then
                if (.not. adafactor_factored_optimizer%initialized()) then
                    invalid_state = .true.
                end if
            else if (.not. allocated(adafactor_optimizer%second_moment)) then
                invalid_state = .true.
            else if (size(adafactor_optimizer%second_moment) /= size(theta)) then
                invalid_state = .true.
            end if
        else if (config%optimizer == MLP_OPTIMIZER_AMSGRAD) then
            if (.not. allocated(amsgrad_optimizer%first_moment) .or. &
                .not. allocated(amsgrad_optimizer%second_moment) .or. &
                .not. allocated(amsgrad_optimizer%max_second_moment)) then
                invalid_state = .true.
            end if
            if (allocated(amsgrad_optimizer%first_moment)) then
                if (size(amsgrad_optimizer%first_moment) /= size(theta)) invalid_state = .true.
            end if
            if (allocated(amsgrad_optimizer%second_moment)) then
                if (size(amsgrad_optimizer%second_moment) /= size(theta)) invalid_state = .true.
            end if
            if (allocated(amsgrad_optimizer%max_second_moment)) then
                if (size(amsgrad_optimizer%max_second_moment) /= size(theta)) invalid_state = .true.
            end if
        else if (config%optimizer == MLP_OPTIMIZER_RADAM) then
            if (.not. allocated(radam_optimizer%first_moment) .or. &
                .not. allocated(radam_optimizer%second_moment)) then
                invalid_state = .true.
            end if
            if (allocated(radam_optimizer%first_moment)) then
                if (size(radam_optimizer%first_moment) /= size(theta)) invalid_state = .true.
            end if
            if (allocated(radam_optimizer%second_moment)) then
                if (size(radam_optimizer%second_moment) /= size(theta)) invalid_state = .true.
            end if
        else if (config%optimizer == MLP_OPTIMIZER_LION) then
            if (.not. allocated(lion_momentum)) then
                invalid_state = .true.
            else if (size(lion_momentum) /= size(theta)) then
                invalid_state = .true.
            end if
        else if (config%optimizer == MLP_OPTIMIZER_RMSPROP) then
            if (.not. allocated(rmsprop_optimizer%square_average) .or. &
                .not. allocated(rmsprop_optimizer%gradient_average) .or. &
                .not. allocated(rmsprop_optimizer%momentum_buffer)) then
                invalid_state = .true.
            end if
            if (allocated(rmsprop_optimizer%square_average)) then
                if (size(rmsprop_optimizer%square_average) /= size(theta)) then
                    invalid_state = .true.
                end if
            end if
            if (allocated(rmsprop_optimizer%gradient_average)) then
                if (size(rmsprop_optimizer%gradient_average) /= size(theta)) then
                    invalid_state = .true.
                end if
            end if
            if (allocated(rmsprop_optimizer%momentum_buffer)) then
                if (size(rmsprop_optimizer%momentum_buffer) /= size(theta)) then
                    invalid_state = .true.
                end if
            end if
        end if
        if (invalid_state) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP checkpoint: optimizer or parameter state is invalid")
            return
        end if
        call checkpoint%clear()
        checkpoint%format_version = 11
        checkpoint%initialized = .true.
        checkpoint%resume_safe = .true.
        checkpoint%n_samples = size(x, 1)
        checkpoint%n_features = size(x, 2)
        checkpoint%n_outputs = size(target, 2)
        checkpoint%n_parameters = size(theta)
        if (allocated(config%optimizer_groups)) then
            checkpoint%n_optimizer_groups = size(config%optimizer_groups)
            if (checkpoint%n_optimizer_groups > 0) then
                allocate(checkpoint%optimizer_group_name(checkpoint%n_optimizer_groups), &
                    checkpoint%optimizer_group_first(checkpoint%n_optimizer_groups), &
                    checkpoint%optimizer_group_last(checkpoint%n_optimizer_groups), &
                    checkpoint%optimizer_group_learning_rate_multiplier( &
                    checkpoint%n_optimizer_groups))
                checkpoint%optimizer_group_name = [(config%optimizer_groups(n)%name, &
                    n=1, checkpoint%n_optimizer_groups)]
                checkpoint%optimizer_group_first = [(config%optimizer_groups(n)%first, &
                    n=1, checkpoint%n_optimizer_groups)]
                checkpoint%optimizer_group_last = [(config%optimizer_groups(n)%last, &
                    n=1, checkpoint%n_optimizer_groups)]
                checkpoint%optimizer_group_learning_rate_multiplier = &
                    [(config%optimizer_groups(n)%learning_rate_multiplier, &
                    n=1, checkpoint%n_optimizer_groups)]
            end if
        end if
        checkpoint%epoch = result%epochs
        checkpoint%updates = result%updates
        checkpoint%microbatches = result%microbatches
        checkpoint%gradient_clipped_updates = result%gradient_clipped_updates
        checkpoint%microbatch_position = iterator%position
        checkpoint%active_epoch = active_epoch
        checkpoint%active_microbatches = active_microbatches
        checkpoint%accumulated_samples = accumulated_samples
        checkpoint%accumulated_weight_mass = accumulated_weight_mass
        checkpoint%iterator_epoch = iterator%epoch_number
        checkpoint%iterator_position = iterator%position
        checkpoint%batch_size = iterator%batch_size
        checkpoint%accumulation_steps = config%accumulation_steps
        checkpoint%precision_kind = config%precision_kind
        checkpoint%loss_scale = loss_scaler
        checkpoint%shuffle_seed = config%shuffle_seed
        checkpoint%optimizer = config%optimizer
        if (config%optimizer == MLP_OPTIMIZER_ADAM) then
            checkpoint%adam_step_count = optimizer%step_count
        else if (config%optimizer == MLP_OPTIMIZER_SGD) then
            checkpoint%adam_step_count = sgd_optimizer%step_count
        else if (config%optimizer == MLP_OPTIMIZER_ADAMW) then
            checkpoint%adam_step_count = adamw_optimizer%step_count
        else if (config%optimizer == MLP_OPTIMIZER_ADAGRAD) then
            checkpoint%adam_step_count = adagrad_optimizer%step_count
        else if (config%optimizer == MLP_OPTIMIZER_ADAFACTOR) then
            if (config%adafactor_factored) then
                checkpoint%adam_step_count = adafactor_factored_optimizer%step_count
            else
                checkpoint%adam_step_count = adafactor_optimizer%step_count
            end if
        else if (config%optimizer == MLP_OPTIMIZER_AMSGRAD) then
            checkpoint%adam_step_count = amsgrad_optimizer%step_count
        else if (config%optimizer == MLP_OPTIMIZER_RADAM) then
            checkpoint%adam_step_count = radam_optimizer%step_count
        else
            checkpoint%adam_step_count = rmsprop_optimizer%step_count
        end if
        checkpoint%stale_epochs = stale_epochs
        checkpoint%schedule_bad_updates = schedule_bad_updates
        checkpoint%schedule_reductions = schedule_reductions
        checkpoint%validation_interval = config%validation_interval
        checkpoint%patience = config%patience
        checkpoint%shuffle = config%shuffle
        checkpoint%has_validation = has_validation
        checkpoint%converged = result%converged
        checkpoint%early_stopped = result%early_stopped
        checkpoint%restore_best = config%restore_best
        checkpoint%has_typed_schedule = has_typed_schedule
        checkpoint%schedule_metric_initialized = schedule_metric_initialized
        checkpoint%typed_schedule = typed_schedule
        checkpoint%schedule_best_metric = schedule_best_metric
        checkpoint%shuffle_state = iterator%shuffle_state
        checkpoint%learning_rate = config%learning_rate
        checkpoint%beta1 = config%beta1
        checkpoint%beta2 = config%beta2
        checkpoint%epsilon = config%epsilon
        checkpoint%adafactor_decay = config%adafactor_decay
        checkpoint%adafactor_clip_threshold = config%adafactor_clip_threshold
        checkpoint%adafactor_relative_step = config%adafactor_relative_step
        checkpoint%adafactor_scale_parameter = config%adafactor_scale_parameter
        checkpoint%adafactor_factored = config%adafactor_factored
        checkpoint%rmsprop_decay = config%rmsprop_decay
        checkpoint%rmsprop_momentum = config%rmsprop_momentum
        checkpoint%rmsprop_centered = config%rmsprop_centered
        checkpoint%momentum = config%momentum
        checkpoint%nesterov = config%nesterov
        checkpoint%weight_decay = config%weight_decay
        checkpoint%l2 = config%l2
        checkpoint%tolerance = config%tolerance
        checkpoint%min_delta = config%min_delta
        checkpoint%gradient_clip_norm = config%gradient_clip_norm
        checkpoint%ema_decay = config%ema_decay
        checkpoint%last_learning_rate = result%last_learning_rate
        checkpoint%initial_loss = result%initial_loss
        checkpoint%final_loss = result%final_loss
        checkpoint%best_loss = result%best_loss
        checkpoint%initial_validation_loss = result%initial_validation_loss
        checkpoint%final_validation_loss = result%final_validation_loss
        checkpoint%best_validation_loss = result%best_validation_loss
        checkpoint%best_epoch = result%best_epoch
        checkpoint%best_validation_epoch = result%best_validation_epoch
        allocate(checkpoint%parameters, source=theta)
        if (config%optimizer == MLP_OPTIMIZER_ADAM) then
            allocate(checkpoint%first_moment, source=optimizer%first_moment)
            allocate(checkpoint%second_moment, source=optimizer%second_moment)
        else if (config%optimizer == MLP_OPTIMIZER_SGD) then
            allocate(checkpoint%first_moment, source=sgd_optimizer%velocity)
            allocate(checkpoint%second_moment(size(theta)))
            checkpoint%second_moment = 0.0_dp
        else if (config%optimizer == MLP_OPTIMIZER_ADAMW) then
            allocate(checkpoint%first_moment, source=adamw_optimizer%first_moment)
            allocate(checkpoint%second_moment, source=adamw_optimizer%second_moment)
        else if (config%optimizer == MLP_OPTIMIZER_ADAGRAD) then
            allocate(checkpoint%first_moment, source=adagrad_optimizer%accumulated_square)
            allocate(checkpoint%second_moment(size(theta)))
            checkpoint%second_moment = 0.0_dp
        else if (config%optimizer == MLP_OPTIMIZER_ADAFACTOR) then
            if (config%adafactor_factored) then
                allocate(checkpoint%first_moment(size(theta)))
                checkpoint%first_moment = 0.0_dp
                allocate(checkpoint%second_moment(size(theta)))
                checkpoint%second_moment = 0.0_dp
                call capture_factored_adafactor_state(checkpoint, &
                    adafactor_factored_optimizer, status)
                if (status%code /= FORTNUM_OK) return
            else
                allocate(checkpoint%first_moment, source=adafactor_optimizer%second_moment)
                allocate(checkpoint%second_moment(size(theta)))
                checkpoint%second_moment = 0.0_dp
            end if
        else if (config%optimizer == MLP_OPTIMIZER_AMSGRAD) then
            allocate(checkpoint%first_moment, source=amsgrad_optimizer%first_moment)
            allocate(checkpoint%second_moment, source=amsgrad_optimizer%second_moment)
            allocate(checkpoint%max_second_moment, source=amsgrad_optimizer%max_second_moment)
        else if (config%optimizer == MLP_OPTIMIZER_RADAM) then
            allocate(checkpoint%first_moment, source=radam_optimizer%first_moment)
            allocate(checkpoint%second_moment, source=radam_optimizer%second_moment)
        else if (config%optimizer == MLP_OPTIMIZER_LION) then
            allocate(checkpoint%first_moment, source=lion_momentum)
            allocate(checkpoint%second_moment(size(theta)))
            checkpoint%second_moment = 0.0_dp
        else
            allocate(checkpoint%first_moment, source=rmsprop_optimizer%square_average)
            allocate(checkpoint%second_moment, source=rmsprop_optimizer%gradient_average)
            allocate(checkpoint%rmsprop_buffer, source=rmsprop_optimizer%momentum_buffer)
        end if
        if (config%ema_decay > 0.0_dp) then
            allocate(checkpoint%ema_parameters, source=ema_parameters)
        end if
        allocate(checkpoint%best_parameters, source=best_theta)
        allocate(checkpoint%accumulated_gradient, source=accumulated_gradient)
        allocate(checkpoint%iterator_order, source=iterator%order)
        allocate(checkpoint%loss_history(result%epochs))
        allocate(checkpoint%learning_rate_history(result%epochs))
        n = result%epochs
        if (n > 0) then
            checkpoint%loss_history = result%loss_history(:n)
            checkpoint%learning_rate_history = result%learning_rate_history(:n)
        end if
        if (has_validation) then
            allocate(checkpoint%validation_loss_history(result%epochs))
            if (n > 0) checkpoint%validation_loss_history = &
                result%validation_loss_history(:n)
        end if
        if (.not. checkpoint%valid()) then
            call checkpoint%clear()
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP checkpoint: captured state failed validation")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine checkpoint_capture

    subroutine emit_training_event(config, event, epoch, update, loss, &
            validation_loss, gradient_norm, learning_rate, stop, status)
        type(mlp_training_options_t), intent(in) :: config
        integer, intent(in) :: event, epoch, update
        real(dp), intent(in) :: loss, validation_loss, gradient_norm
        real(dp), intent(in) :: learning_rate
        logical, intent(out) :: stop
        type(fortnum_status_t), intent(out) :: status

        stop = .false.
        if (associated(config%event_callback)) then
            call config%event_callback(event, epoch, update, loss, &
                validation_loss, gradient_norm, learning_rate, stop, status)
        else
            call status_set(status, FORTNUM_OK, "")
        end if
    end subroutine emit_training_event

    logical function valid_options(options) result(valid)
        type(mlp_training_options_t), intent(in) :: options

        valid = options%max_epochs >= 1 .and. options%batch_size >= 0 .and. &
            options%accumulation_steps >= 1 .and. &
            options%patience >= 0 .and. options%learning_rate > 0.0_dp .and. &
            options%precision_kind >= MLP_PRECISION_FP64 .and. &
            options%precision_kind <= MLP_PRECISION_BF16 .and. &
            (options%optimizer == MLP_OPTIMIZER_ADAM .or. &
            options%optimizer == MLP_OPTIMIZER_SGD .or. &
            options%optimizer == MLP_OPTIMIZER_ADAMW .or. &
            options%optimizer == MLP_OPTIMIZER_ADAGRAD .or. &
            options%optimizer == MLP_OPTIMIZER_RMSPROP .or. &
            options%optimizer == MLP_OPTIMIZER_ADAFACTOR .or. &
            options%optimizer == MLP_OPTIMIZER_AMSGRAD .or. &
            options%optimizer == MLP_OPTIMIZER_RADAM .or. &
            options%optimizer == MLP_OPTIMIZER_LION) .and. &
            options%beta1 >= 0.0_dp .and. options%beta1 < 1.0_dp .and. &
            options%beta2 >= 0.0_dp .and. options%beta2 < 1.0_dp .and. &
            options%epsilon > 0.0_dp .and. options%l2 >= 0.0_dp .and. &
            options%momentum >= 0.0_dp .and. options%momentum < 1.0_dp .and. &
            options%tolerance >= 0.0_dp .and. options%min_delta >= 0.0_dp .and. &
            options%gradient_clip_norm >= 0.0_dp .and. &
            options%ema_decay >= 0.0_dp .and. options%ema_decay < 1.0_dp .and. &
            options%validation_interval >= 1
        valid = valid .and. ieee_is_finite(options%learning_rate) .and. &
            ieee_is_finite(options%beta1) .and. ieee_is_finite(options%beta2) .and. &
            ieee_is_finite(options%epsilon) .and. ieee_is_finite(options%l2) .and. &
            ieee_is_finite(options%adafactor_decay) .and. &
            ieee_is_finite(options%adafactor_clip_threshold) .and. &
            ieee_is_finite(options%rmsprop_decay) .and. &
            ieee_is_finite(options%rmsprop_momentum) .and. &
            ieee_is_finite(options%momentum) .and. &
            ieee_is_finite(options%weight_decay) .and. &
            ieee_is_finite(options%tolerance) .and. &
            ieee_is_finite(options%min_delta) .and. &
            ieee_is_finite(options%gradient_clip_norm) .and. &
            ieee_is_finite(options%ema_decay)
        valid = valid .and. options%weight_decay >= 0.0_dp
        valid = valid .and. options%loss_scale%valid()
        valid = valid .and. options%rmsprop_decay >= 0.0_dp .and. &
            options%rmsprop_decay < 1.0_dp .and. options%rmsprop_momentum >= 0.0_dp .and. &
            options%rmsprop_momentum < 1.0_dp
        valid = valid .and. options%adafactor_decay >= 0.0_dp .and. &
            options%adafactor_decay < 1.0_dp .and. options%adafactor_clip_threshold > 0.0_dp
        if (options%shuffle) valid = valid .and. options%shuffle_seed > 0
        if (options%nesterov) valid = valid .and. &
            options%optimizer == MLP_OPTIMIZER_SGD .and. options%momentum > 0.0_dp
        if (options%use_typed_schedule) then
            valid = valid .and. options%typed_schedule%valid()
            if (associated(options%learning_rate_schedule)) valid = .false.
        end if
        valid = valid .and. valid_optimizer_groups(options%optimizer_groups)
    end function valid_options

    logical function valid_optimizer_groups(groups) result(valid)
        type(mlp_optimizer_group_t), allocatable, intent(in) :: groups(:)
        integer :: i, j

        valid = .true.
        if (.not. allocated(groups)) return
        do i = 1, size(groups)
            if (.not. groups(i)%initialized()) then
                valid = .false.
                return
            end if
            do j = 1, i - 1
                if (trim(groups(i)%name) == trim(groups(j)%name) .or. &
                    ranges_overlap(groups(i)%first, groups(i)%last, &
                    groups(j)%first, groups(j)%last)) then
                    valid = .false.
                    return
                end if
            end do
        end do
    end function valid_optimizer_groups

    logical function optimizer_group_ranges_valid(n_parameters, first, last) result(valid)
        integer, intent(in) :: n_parameters
        integer, intent(in) :: first(:), last(:)
        integer :: i, j

        valid = n_parameters > 0 .and. size(first) == size(last)
        if (.not. valid) return
        do i = 1, size(first)
            if (first(i) < 1 .or. last(i) < first(i) .or. last(i) > n_parameters) then
                valid = .false.
                return
            end if
            do j = 1, i - 1
                if (ranges_overlap(first(i), last(i), first(j), last(j))) then
                    valid = .false.
                    return
                end if
            end do
        end do
    end function optimizer_group_ranges_valid

    logical function optimizer_groups_fit(n_parameters, groups) result(valid)
        integer, intent(in) :: n_parameters
        type(mlp_optimizer_group_t), allocatable, intent(in) :: groups(:)
        integer, allocatable :: first(:), last(:)
        integer :: i

        valid = .true.
        if (.not. allocated(groups)) return
        if (size(groups) == 0) return
        allocate(first(size(groups)), last(size(groups)))
        do i = 1, size(groups)
            first(i) = groups(i)%first
            last(i) = groups(i)%last
        end do
        valid = optimizer_group_ranges_valid(n_parameters, first, last)
    end function optimizer_groups_fit

    logical function optimizer_groups_equal(checkpoint, groups) result(equal)
        type(mlp_training_checkpoint_t), intent(in) :: checkpoint
        type(mlp_optimizer_group_t), allocatable, intent(in) :: groups(:)
        integer :: i

        if (.not. allocated(groups)) then
            equal = checkpoint%n_optimizer_groups == 0
            return
        end if
        equal = checkpoint%n_optimizer_groups == size(groups)
        if (.not. equal) return
        if (size(groups) == 0) return
        if (.not. allocated(checkpoint%optimizer_group_name) .or. &
            .not. allocated(checkpoint%optimizer_group_first) .or. &
            .not. allocated(checkpoint%optimizer_group_last) .or. &
            .not. allocated(checkpoint%optimizer_group_learning_rate_multiplier)) then
            equal = .false.
            return
        end if
        do i = 1, size(groups)
            equal = trim(checkpoint%optimizer_group_name(i)) == trim(groups(i)%name) .and. &
                checkpoint%optimizer_group_first(i) == groups(i)%first .and. &
                checkpoint%optimizer_group_last(i) == groups(i)%last .and. &
                checkpoint%optimizer_group_learning_rate_multiplier(i) == &
                groups(i)%learning_rate_multiplier
            if (.not. equal) return
        end do
    end function optimizer_groups_equal

    subroutine apply_optimizer_group_scales(theta, theta_before, groups, status)
        real(dp), intent(inout) :: theta(:)
        real(dp), intent(in) :: theta_before(:)
        type(mlp_optimizer_group_t), allocatable, intent(in) :: groups(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, first, last

        if (size(theta) /= size(theta_before) .or. &
            .not. optimizer_groups_fit(size(theta), groups)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP optimizer groups: parameter state is incompatible")
            return
        end if
        do i = 1, size(groups)
            first = groups(i)%first
            last = groups(i)%last
            theta(first:last) = theta_before(first:last) + &
                groups(i)%learning_rate_multiplier* &
                (theta(first:last) - theta_before(first:last))
        end do
        if (any(.not. ieee_is_finite(theta))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP optimizer groups: scaled update is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine apply_optimizer_group_scales

    logical function ranges_overlap(first_a, last_a, first_b, last_b) result(overlap)
        integer, intent(in) :: first_a, last_a, first_b, last_b

        overlap = first_a <= last_b .and. first_b <= last_a
    end function ranges_overlap

    logical function schedules_equal(first, second) result(equal)
        type(mlp_learning_rate_schedule_t), intent(in) :: first, second

        equal = first%kind == second%kind .and. &
            first%warmup_updates == second%warmup_updates .and. &
            first%total_updates == second%total_updates .and. &
            first%min_rate_fraction == second%min_rate_fraction .and. &
            first%decay_factor == second%decay_factor .and. &
            first%peak_rate_fraction == second%peak_rate_fraction .and. &
            first%final_rate_fraction == second%final_rate_fraction .and. &
            first%metric_mode == second%metric_mode .and. &
            first%patience_updates == second%patience_updates .and. &
            first%min_delta == second%min_delta .and. &
            first%plateau_factor == second%plateau_factor
    end function schedules_equal

    subroutine adafactor_specs_from_layout(layout, specs, status)
        !! Convert the MLP's named dense layout into explicit Adafactor blocks.
        type(mlp_parameter_block_t), intent(in) :: layout(:)
        type(adafactor_block_spec_t), intent(out) :: specs(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: i

        if (size(layout) < 1 .or. size(specs) /= size(layout)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP Adafactor: parameter layout is invalid")
            return
        end if
        do i = 1, size(layout)
            specs(i)%first = layout(i)%first
            specs(i)%last = layout(i)%last
            specs(i)%rows = layout(i)%rows
            specs(i)%columns = layout(i)%columns
            specs(i)%factored = trim(layout(i)%kind) == "weight" .and. &
                layout(i)%rows > 1 .and. layout(i)%columns > 1
            if (.not. specs(i)%valid()) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP Adafactor: parameter layout block is invalid")
                return
            end if
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine adafactor_specs_from_layout

    subroutine capture_factored_adafactor_state(checkpoint, optimizer, status)
        type(mlp_training_checkpoint_t), intent(inout) :: checkpoint
        type(adafactor_factored_t), intent(in) :: optimizer
        type(fortnum_status_t), intent(out) :: status
        integer :: i, row_total, column_total, second_total
        integer :: row_position, column_position, second_position

        if (.not. optimizer%initialized() .or. .not. allocated(optimizer%blocks) .or. &
            .not. allocated(optimizer%state)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP checkpoint: factored Adafactor is not initialized")
            return
        end if
        checkpoint%n_adafactor_blocks = size(optimizer%blocks)
        allocate(checkpoint%adafactor_block_first(checkpoint%n_adafactor_blocks), &
            checkpoint%adafactor_block_last(checkpoint%n_adafactor_blocks), &
            checkpoint%adafactor_block_rows(checkpoint%n_adafactor_blocks), &
            checkpoint%adafactor_block_columns(checkpoint%n_adafactor_blocks), &
            checkpoint%adafactor_block_factored(checkpoint%n_adafactor_blocks))
        row_total = 0
        column_total = 0
        second_total = 0
        do i = 1, checkpoint%n_adafactor_blocks
            checkpoint%adafactor_block_first(i) = optimizer%blocks(i)%first
            checkpoint%adafactor_block_last(i) = optimizer%blocks(i)%last
            checkpoint%adafactor_block_rows(i) = optimizer%blocks(i)%rows
            checkpoint%adafactor_block_columns(i) = optimizer%blocks(i)%columns
            checkpoint%adafactor_block_factored(i) = merge(1, 0, optimizer%blocks(i)%factored)
            if (optimizer%blocks(i)%factored) then
                row_total = row_total + optimizer%blocks(i)%rows
                column_total = column_total + optimizer%blocks(i)%columns
            else
                second_total = second_total + optimizer%blocks(i)%last - &
                    optimizer%blocks(i)%first + 1
            end if
        end do
        allocate(checkpoint%adafactor_row_moment(row_total), &
            checkpoint%adafactor_column_moment(column_total), &
            checkpoint%adafactor_second_moment(second_total))
        row_position = 1
        column_position = 1
        second_position = 1
        do i = 1, checkpoint%n_adafactor_blocks
            if (optimizer%blocks(i)%factored) then
                checkpoint%adafactor_row_moment(row_position:row_position + &
                    optimizer%blocks(i)%rows - 1) = optimizer%state(i)%row_moment
                checkpoint%adafactor_column_moment(column_position:column_position + &
                    optimizer%blocks(i)%columns - 1) = optimizer%state(i)%column_moment
                row_position = row_position + optimizer%blocks(i)%rows
                column_position = column_position + optimizer%blocks(i)%columns
            else
                checkpoint%adafactor_second_moment(second_position:second_position + &
                    optimizer%blocks(i)%last - optimizer%blocks(i)%first) = &
                    optimizer%state(i)%second_moment
                second_position = second_position + optimizer%blocks(i)%last - &
                    optimizer%blocks(i)%first + 1
            end if
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine capture_factored_adafactor_state

    subroutine restore_factored_adafactor_checkpoint(checkpoint, optimizer, specs, status)
        type(mlp_training_checkpoint_t), intent(in) :: checkpoint
        type(adafactor_factored_t), intent(inout) :: optimizer
        type(adafactor_block_spec_t), intent(in) :: specs(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, row_total, column_total, second_total
        integer :: row_position, column_position, second_position

        if (.not. checkpoint%adafactor_factored .or. checkpoint%n_adafactor_blocks /= size(specs) .or. &
            .not. optimizer%initialized() .or. size(optimizer%blocks) /= size(specs)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP checkpoint: factored Adafactor layout is incompatible")
            return
        end if
        do i = 1, size(specs)
            if (checkpoint%adafactor_block_first(i) /= specs(i)%first .or. &
                checkpoint%adafactor_block_last(i) /= specs(i)%last .or. &
                checkpoint%adafactor_block_rows(i) /= specs(i)%rows .or. &
                checkpoint%adafactor_block_columns(i) /= specs(i)%columns .or. &
                (checkpoint%adafactor_block_factored(i) == 1) .neqv. specs(i)%factored) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP checkpoint: factored Adafactor layout is incompatible")
                return
            end if
        end do
        row_total = sum(merge(specs%rows, 0, specs%factored))
        column_total = sum(merge(specs%columns, 0, specs%factored))
        second_total = sum(merge(specs%last - specs%first + 1, 0, .not. specs%factored))
        if (.not. allocated(checkpoint%adafactor_row_moment) .or. &
            .not. allocated(checkpoint%adafactor_column_moment) .or. &
            .not. allocated(checkpoint%adafactor_second_moment) .or. &
            size(checkpoint%adafactor_row_moment) /= row_total .or. &
            size(checkpoint%adafactor_column_moment) /= column_total .or. &
            size(checkpoint%adafactor_second_moment) /= second_total) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP checkpoint: factored Adafactor state is malformed")
            return
        end if
        row_position = 1
        column_position = 1
        second_position = 1
        do i = 1, size(specs)
            if (specs(i)%factored) then
                optimizer%state(i)%row_moment = checkpoint%adafactor_row_moment(row_position: &
                    row_position + specs(i)%rows - 1)
                optimizer%state(i)%column_moment = checkpoint%adafactor_column_moment(column_position: &
                    column_position + specs(i)%columns - 1)
                row_position = row_position + specs(i)%rows
                column_position = column_position + specs(i)%columns
            else
                optimizer%state(i)%second_moment = checkpoint%adafactor_second_moment(second_position: &
                    second_position + specs(i)%last - specs(i)%first)
                second_position = second_position + specs(i)%last - specs(i)%first + 1
            end if
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine restore_factored_adafactor_checkpoint

    logical function valid_lbfgsb_options(options) result(valid)
        type(mlp_lbfgsb_options_t), intent(in) :: options

        valid = options%memory >= 1 .and. options%max_iterations >= 1 .and. &
            options%max_line_search >= 1 .and. options%lower_bound <= &
            options%upper_bound .and. options%l2_lower_bound <= &
            options%l2_upper_bound .and. options%l2 >= 0.0_dp .and. &
            options%l2_lower_bound >= 0.0_dp
        valid = valid .and. ieee_is_finite(options%gradient_tolerance) .and. &
            ieee_is_finite(options%step_tolerance) .and. &
            ieee_is_finite(options%objective_tolerance) .and. &
            ieee_is_finite(options%lower_bound) .and. &
            ieee_is_finite(options%upper_bound) .and. &
            ieee_is_finite(options%l2) .and. &
            ieee_is_finite(options%l2_lower_bound) .and. &
            ieee_is_finite(options%l2_upper_bound) .and. &
            options%gradient_tolerance >= 0.0_dp .and. &
            options%step_tolerance >= 0.0_dp .and. &
            options%objective_tolerance >= 0.0_dp
        if (options%optimize_l2) valid = valid .and. &
            options%l2 >= options%l2_lower_bound .and. &
            options%l2 <= options%l2_upper_bound
    end function valid_lbfgsb_options

    logical function valid_data(model, x, target) result(valid)
        class(mlp_t), intent(in) :: model
        real(dp), intent(in) :: x(:, :), target(:, :)

        valid = size(x, 1) > 0 .and. size(target, 1) == size(x, 1)
        if (.not. valid) return
        valid = size(x, 2) > 0 .and. size(target, 2) > 0
        if (.not. valid) return
        valid = all(ieee_is_finite(x)) .and. all(ieee_is_finite(target))
        if (.not. valid) return
        ! The model itself performs the authoritative feature/output shape
        ! check; this call avoids exposing its private allocation predicate.
        valid = model%parameter_count() > 0
    end function valid_data

    subroutine set_model_parameters_for_precision(model, parameters, precision_kind, status)
        !! Load the forward model from a binary64 master vector.
        !!
        !! FP32 deliberately has no second parameter storage in `mlp_t` yet.
        !! The model boundary is therefore a transactional binary32 cast.  The
        !! caller keeps `parameters` as the optimizer/master vector and this
        !! routine leaves it untouched.  A value outside the finite FP32 range
        !! is rejected before the model can be partially updated.
        class(mlp_t), intent(inout) :: model
        real(dp), intent(in) :: parameters(:)
        integer, intent(in) :: precision_kind
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: rounded(:)

        if (precision_kind == MLP_PRECISION_FP32) then
            allocate(rounded(size(parameters)))
            rounded = real(real(parameters, real32), dp)
            if (any(.not. ieee_is_finite(rounded))) then
                call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                    "MLP train: master parameter is outside finite FP32 range")
                return
            end if
            call model%set_parameters(rounded, status)
            return
        end if
        call model%set_parameters(parameters, status)
    end subroutine set_model_parameters_for_precision

    subroutine quantize_matrix_fp32(values, status, context)
        !! Round a training data matrix through the IEEE binary32 kind.
        real(dp), intent(inout) :: values(:, :)
        type(fortnum_status_t), intent(out) :: status
        character(len=*), intent(in) :: context
        real(dp), allocatable :: rounded(:, :)

        allocate(rounded(size(values, 1), size(values, 2)))
        rounded = real(real(values, real32), dp)
        if (any(.not. ieee_is_finite(rounded))) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                trim(context)//" exceeds finite FP32 range")
            return
        end if
        values = rounded
        call status_set(status, FORTNUM_OK, "")
    end subroutine quantize_matrix_fp32

    subroutine round_gradient_for_precision(gradient, precision_kind, status)
        !! Keep the gradient boundary identical to the FP32 forward boundary.
        real(dp), intent(inout) :: gradient(:)
        integer, intent(in) :: precision_kind
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: rounded(:)

        if (precision_kind == MLP_PRECISION_FP32) then
            allocate(rounded(size(gradient)))
            rounded = real(real(gradient, real32), dp)
            if (any(.not. ieee_is_finite(rounded))) then
                call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                    "MLP train: gradient exceeds finite FP32 range")
                return
            end if
            gradient = rounded
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine round_gradient_for_precision

    subroutine validate_sample_weight(sample_weight, n_samples, weight_mass, &
            status, context)
        real(dp), intent(in) :: sample_weight(:)
        integer, intent(in) :: n_samples
        real(dp), intent(out) :: weight_mass
        type(fortnum_status_t), intent(out) :: status
        character(len=*), intent(in) :: context

        weight_mass = 0.0_dp
        if (size(sample_weight) /= n_samples .or. &
            any(.not. ieee_is_finite(sample_weight)) .or. &
            any(sample_weight < 0.0_dp)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                trim(context)//": sample weights must be finite and non-negative")
            return
        end if
        weight_mass = sum(sample_weight)
        if (.not. ieee_is_finite(weight_mass) .or. weight_mass <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                trim(context)//": sample weights must have positive mass")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine validate_sample_weight

    subroutine shuffle_order(order, generator)
        integer, intent(inout) :: order(:)
        integer(int64), intent(inout) :: generator
        integer :: i, j, temporary

        do i = size(order), 2, -1
            generator = mod(48271_int64*generator, 2147483647_int64)
            j = 1 + int(mod(generator, int(i, int64)))
            temporary = order(i)
            order(i) = order(j)
            order(j) = temporary
        end do
    end subroutine shuffle_order

    subroutine shrink_history(history, length)
        real(dp), allocatable, intent(inout) :: history(:)
        integer, intent(in) :: length
        real(dp), allocatable :: compact(:)

        if (size(history) == length) return
        allocate(compact(max(0, length)))
        if (length > 0) compact = history(:length)
        call move_alloc(compact, history)
    end subroutine shrink_history

end module fortml_mlp_training
