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
    use, intrinsic :: iso_fortran_env, only: int64
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_CONVERGENCE_ERROR
    use fortml_mlp, only: mlp_t
    use fortopt_objective, only: objective_t
    use fortopt_adam, only: adam_t
    use fortopt_adamw, only: adamw_t
    use fortopt_sgd, only: sgd_t
    use fortopt_lbfgsb, only: lbfgsb_t, lbfgsb_options_t, lbfgsb_result_t
    implicit none
    private

    integer, parameter, public :: MLP_REDUCTION_MEAN = 1
    integer, parameter, public :: MLP_REDUCTION_SUM = 2
    integer, parameter, public :: MLP_OPTIMIZER_ADAM = 1
    integer, parameter, public :: MLP_OPTIMIZER_SGD = 2
    integer, parameter, public :: MLP_OPTIMIZER_ADAMW = 3

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
    end interface

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
        integer :: optimizer = MLP_OPTIMIZER_ADAM
        real(dp) :: momentum = 0.0_dp
        logical :: nesterov = .false.
        real(dp) :: weight_decay = 0.0_dp
        real(dp) :: l2 = 0.0_dp
        real(dp) :: tolerance = 1.0e-8_dp
        real(dp) :: min_delta = 0.0_dp
        real(dp) :: gradient_clip_norm = 0.0_dp
        procedure(mlp_epoch_callback_proc), pointer, nopass :: callback => null()
        procedure(mlp_learning_rate_schedule_proc), pointer, nopass :: &
            learning_rate_schedule => null()
    end type mlp_training_options_t

    type, public :: mlp_training_state_t
        integer :: epochs = 0
        integer :: updates = 0
        integer :: microbatches = 0
        integer :: accumulation_steps = 1
        integer :: best_epoch = 0
        integer :: best_validation_epoch = 0
        logical :: converged = .false.
        logical :: early_stopped = .false.
        integer :: gradient_clipped_updates = 0
        real(dp) :: initial_loss = huge(1.0_dp)
        real(dp) :: final_loss = huge(1.0_dp)
        real(dp) :: best_loss = huge(1.0_dp)
        real(dp) :: initial_validation_loss = huge(1.0_dp)
        real(dp) :: final_validation_loss = huge(1.0_dp)
        real(dp) :: best_validation_loss = huge(1.0_dp)
        real(dp) :: gradient_norm = huge(1.0_dp)
        real(dp) :: last_learning_rate = 0.0_dp
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
        !! best-state bookkeeping are all copied.  Procedure pointers (custom
        !! schedules and callbacks) are intentionally not copied: the caller
        !! must install deterministic procedures again on the resumed options.
        integer :: format_version = 2
        logical :: initialized = .false.
        logical :: resume_safe = .true.
        integer :: n_samples = 0
        integer :: n_features = 0
        integer :: n_outputs = 0
        integer :: n_parameters = 0
        integer :: epoch = 0
        integer :: updates = 0
        integer :: microbatches = 0
        integer :: microbatch_position = 1
        integer :: active_epoch = 0
        integer :: active_microbatches = 0
        integer :: accumulated_samples = 0
        integer :: iterator_epoch = 0
        integer :: iterator_position = 1
        integer :: batch_size = 0
        integer :: accumulation_steps = 1
        integer :: shuffle_seed = 17
        integer :: adam_step_count = 0
        integer :: optimizer = MLP_OPTIMIZER_ADAM
        integer :: stale_epochs = 0
        integer :: gradient_clipped_updates = 0
        integer :: validation_interval = 1
        integer :: patience = 0
        logical :: shuffle = .false.
        logical :: has_validation = .false.
        logical :: converged = .false.
        logical :: early_stopped = .false.
        logical :: restore_best = .true.
        integer(int64) :: shuffle_state = 1_int64
        real(dp) :: learning_rate = 1.0e-3_dp
        real(dp) :: beta1 = 0.9_dp
        real(dp) :: beta2 = 0.999_dp
        real(dp) :: epsilon = 1.0e-8_dp
        real(dp) :: momentum = 0.0_dp
        logical :: nesterov = .false.
        real(dp) :: weight_decay = 0.0_dp
        real(dp) :: l2 = 0.0_dp
        real(dp) :: tolerance = 1.0e-8_dp
        real(dp) :: min_delta = 0.0_dp
        real(dp) :: gradient_clip_norm = 0.0_dp
        real(dp) :: last_learning_rate = 0.0_dp
        real(dp) :: initial_loss = huge(1.0_dp)
        real(dp) :: final_loss = huge(1.0_dp)
        real(dp) :: best_loss = huge(1.0_dp)
        real(dp) :: initial_validation_loss = huge(1.0_dp)
        real(dp) :: final_validation_loss = huge(1.0_dp)
        real(dp) :: best_validation_loss = huge(1.0_dp)
        integer :: best_epoch = 0
        integer :: best_validation_epoch = 0
        real(dp), allocatable :: parameters(:)
        real(dp), allocatable :: first_moment(:)
        real(dp), allocatable :: second_moment(:)
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
        real(dp) :: l2 = 0.0_dp
        logical :: optimize_l2 = .false.
    contains
        procedure, public :: initialize => mlp_objective_initialize
        procedure, public :: parameter_count => mlp_objective_parameter_count
        procedure, public :: parameters => mlp_objective_parameters
        procedure, public :: value_gradient => mlp_objective_value_gradient
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
    public :: mlp_loss_value_gradient
    public :: mlp_loss_hvp
    public :: mlp_train
    public :: mlp_optimize_lbfgsb

contains

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

        if (allocated(self%parameters)) deallocate(self%parameters)
        if (allocated(self%first_moment)) deallocate(self%first_moment)
        if (allocated(self%second_moment)) deallocate(self%second_moment)
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
        self%format_version = 2
        self%initialized = .false.
        self%resume_safe = .true.
        self%n_samples = 0
        self%n_features = 0
        self%n_outputs = 0
        self%n_parameters = 0
        self%epoch = 0
        self%updates = 0
        self%microbatches = 0
        self%microbatch_position = 1
        self%active_epoch = 0
        self%active_microbatches = 0
        self%accumulated_samples = 0
        self%iterator_epoch = 0
        self%iterator_position = 1
        self%batch_size = 0
        self%accumulation_steps = 1
        self%shuffle_seed = 17
        self%adam_step_count = 0
        self%optimizer = MLP_OPTIMIZER_ADAM
        self%stale_epochs = 0
        self%gradient_clipped_updates = 0
        self%validation_interval = 1
        self%patience = 0
        self%shuffle = .false.
        self%has_validation = .false.
        self%converged = .false.
        self%early_stopped = .false.
        self%restore_best = .true.
        self%shuffle_state = 1_int64
        self%learning_rate = 1.0e-3_dp
        self%beta1 = 0.9_dp
        self%beta2 = 0.999_dp
        self%epsilon = 1.0e-8_dp
        self%momentum = 0.0_dp
        self%nesterov = .false.
        self%weight_decay = 0.0_dp
        self%l2 = 0.0_dp
        self%tolerance = 1.0e-8_dp
        self%min_delta = 0.0_dp
        self%gradient_clip_norm = 0.0_dp
        self%last_learning_rate = 0.0_dp
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

        valid = self%initialized .and. self%format_version == 2 .and. &
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
            (self%optimizer == MLP_OPTIMIZER_ADAM .or. &
            self%optimizer == MLP_OPTIMIZER_SGD .or. &
            self%optimizer == MLP_OPTIMIZER_ADAMW) .and. &
            self%validation_interval > 0 .and. self%patience >= 0 .and. &
            self%gradient_clipped_updates >= 0 .and. &
            allocated(self%parameters) .and. allocated(self%first_moment) .and. &
            allocated(self%second_moment) .and. allocated(self%best_parameters) &
            .and. allocated(self%accumulated_gradient) .and. &
            allocated(self%iterator_order) .and. &
            allocated(self%loss_history) .and. &
            allocated(self%learning_rate_history)
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
        if (self%has_validation) valid = valid .and. &
            all(ieee_is_finite(self%validation_loss_history))
        valid = valid .and. ieee_is_finite(self%learning_rate) .and. &
            ieee_is_finite(self%beta1) .and. ieee_is_finite(self%beta2) .and. &
            ieee_is_finite(self%epsilon) .and. ieee_is_finite(self%l2) .and. &
            ieee_is_finite(self%momentum) .and. &
            ieee_is_finite(self%weight_decay) .and. &
            ieee_is_finite(self%last_learning_rate) .and. &
            ieee_is_finite(self%initial_loss) .and. ieee_is_finite(self%final_loss) &
            .and. ieee_is_finite(self%best_loss) .and. &
            ieee_is_finite(self%tolerance) .and. &
            ieee_is_finite(self%min_delta) .and. &
            ieee_is_finite(self%gradient_clip_norm) .and. &
            self%tolerance >= 0.0_dp .and. self%min_delta >= 0.0_dp .and. &
            self%gradient_clip_norm >= 0.0_dp
        valid = valid .and. self%momentum >= 0.0_dp .and. self%momentum < 1.0_dp
        valid = valid .and. self%weight_decay >= 0.0_dp
        if (self%nesterov) valid = valid .and. self%optimizer == MLP_OPTIMIZER_SGD .and. &
            self%momentum > 0.0_dp
        if (self%has_validation) valid = valid .and. &
            ieee_is_finite(self%initial_validation_loss) .and. &
            ieee_is_finite(self%final_validation_loss) .and. &
            ieee_is_finite(self%best_validation_loss)
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
        real(dp), allocatable :: prediction(:, :), residual(:, :), x_bar(:, :)
        real(dp), allocatable :: weighted_residual(:, :)
        real(dp), allocatable :: theta(:)
        integer :: n_samples, reduction_kind, i
        real(dp) :: weight_mass, normalization, data_loss, regularization_loss

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
        normalization = 1.0_dp
        if (reduction_kind == MLP_REDUCTION_MEAN) normalization = weight_mass
        allocate(prediction(size(target, 1), size(target, 2)))
        allocate(residual, mold=prediction)
        allocate(weighted_residual, mold=prediction)
        allocate(x_bar, mold=x)
        call model%predict(x, prediction, status)
        if (status%code /= FORTNUM_OK) return
        residual = prediction - target
        weighted_residual = residual
        if (present(sample_weight)) then
            do i = 1, n_samples
                weighted_residual(i, :) = sample_weight(i)*residual(i, :)
            end do
        end if
        data_loss = 0.5_dp*sum(residual*weighted_residual)/normalization
        call model%vjp(x, weighted_residual/normalization, gradient, x_bar, status)
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
            parameter_hvp, l2_hvp, status)
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
        real(dp), allocatable :: prediction(:, :), residual(:, :)
        real(dp), allocatable :: output_tangent(:, :), zero_input(:, :)
        real(dp), allocatable :: x_bar(:, :), jtj_product(:), curvature(:)
        real(dp), allocatable :: theta(:)
        integer :: n_samples, n_parameters

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
        allocate(prediction(size(target, 1), size(target, 2)))
        allocate(residual, mold=prediction)
        allocate(output_tangent, mold=prediction)
        allocate(zero_input, mold=x)
        allocate(x_bar, mold=x)
        allocate(jtj_product(n_parameters), curvature(n_parameters))
        call model%predict(x, prediction, status)
        if (status%code /= FORTNUM_OK) return
        residual = prediction - target
        zero_input = 0.0_dp
        call model%jvp(x, dtheta, zero_input, prediction, output_tangent, status)
        if (status%code /= FORTNUM_OK) return
        call model%vjp(x, output_tangent/real(n_samples, dp), jtj_product, &
            x_bar, status)
        if (status%code /= FORTNUM_OK) return
        call model%hvp(x, residual/real(n_samples, dp), dtheta, zero_input, &
            curvature, x_bar, status)
        if (status%code /= FORTNUM_OK) return
        theta = model%parameters()
        parameter_hvp = jtj_product + curvature + l2*dtheta + &
            l2_direction*theta
        l2_hvp = dot_product(theta, dtheta)
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_loss_hvp

    subroutine mlp_objective_initialize(self, model, x, target, l2, status, &
            optimize_l2)
        class(mlp_training_objective_t), intent(out) :: self
        type(mlp_t), target, intent(inout) :: model
        real(dp), intent(in) :: x(:, :), target(:, :), l2
        type(fortnum_status_t), intent(out) :: status
        logical, intent(in), optional :: optimize_l2

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
        self%model => model
        allocate(self%features, source=x)
        allocate(self%targets, source=target)
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
            size(gradient) /= size(parameters)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP objective: parameter or gradient shape is invalid")
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
        call mlp_loss_value_gradient(self%model, self%features, self%targets, &
            l2, value, gradient(:n_model), l2_gradient, status)
        if (status%code /= FORTNUM_OK) return
        if (self%optimize_l2) gradient(n_model + 1) = l2_gradient
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_objective_value_gradient

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
        call mlp_loss_hvp(self%model, self%features, self%targets, l2, &
            direction(:n_model), l2_direction, parameter_hvp, l2_hvp, status)
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

    subroutine mlp_optimize_lbfgsb(model, x, target, options, result, status)
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
        type(mlp_training_objective_t), target :: adapter
        type(objective_t) :: objective
        type(lbfgsb_t) :: optimizer
        type(lbfgsb_options_t) :: optimizer_options
        type(lbfgsb_result_t) :: optimizer_result
        real(dp), allocatable :: parameters(:), lower(:), upper(:), gradient(:)
        integer :: n_model, n_parameters

        result = mlp_lbfgsb_result_t()
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
        call adapter%initialize(model, x, target, options%l2, status, &
            optimize_l2=options%optimize_l2)
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
            validation_x, validation_target, checkpoint)
        !! Train `model` with deterministic Adam updates.
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
        !! trajectory.  `options%max_epochs` is a total target epoch on both
        !! fresh and resumed calls.
        class(mlp_t), intent(inout) :: model
        real(dp), intent(in) :: x(:, :), target(:, :)
        type(fortnum_status_t), intent(out) :: status
        type(mlp_training_options_t), intent(in), optional :: options
        type(mlp_training_state_t), intent(out), optional :: state
        real(dp), intent(in), optional :: validation_x(:, :), validation_target(:, :)
        type(mlp_training_checkpoint_t), intent(inout), optional :: checkpoint
        type(mlp_training_options_t) :: config
        type(mlp_training_state_t) :: result
        type(adam_t) :: optimizer
        type(adamw_t) :: adamw_optimizer
        type(sgd_t) :: sgd_optimizer
        type(mlp_batch_iterator_t) :: iterator
        real(dp), allocatable :: theta(:), best_theta(:), gradient(:)
        real(dp), allocatable :: accumulated_gradient(:)
        real(dp), allocatable :: x_batch(:, :), target_batch(:, :)
        integer, allocatable :: batch_indices(:)
        real(dp) :: loss, l2_gradient, gradient_norm, improvement
        real(dp) :: best_loss
        real(dp) :: validation_loss, monitored_loss
        real(dp) :: effective_rate, raw_gradient_norm
        integer :: n_samples, n_outputs, n_parameters
        integer :: batch, epoch
        integer :: microbatch_count, accumulated_samples
        integer :: stale_epochs
        integer :: start_epoch, history_length
        logical :: stop_now, has_batch, resuming, resume_active_epoch
        logical :: incompatible_checkpoint

        resuming = .false.
        if (present(checkpoint)) resuming = checkpoint%initialized
        resume_active_epoch = .false.
        if (present(options)) config = options
        if (.not. valid_options(config) .or. &
            .not. valid_data(model, x, target) .or. &
            (present(validation_x) .neqv. present(validation_target))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP train: invalid model, data, or options")
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
        n_outputs = size(target, 2)
        n_parameters = model%parameter_count()
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
            call model%set_parameters(theta, status)
            if (status%code /= FORTNUM_OK) then
                if (present(state)) state = result
                return
            end if
            allocate(best_theta, source=checkpoint%best_parameters)
        else
            theta = model%parameters()
            allocate(best_theta, source=theta)
        end if
        allocate(gradient(n_parameters))
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
            call mlp_loss_value_gradient(model, x, target, config%l2, loss, &
                gradient, l2_gradient, status)
            if (status%code /= FORTNUM_OK) then
                if (present(state)) state = result
                return
            end if
            result%initial_loss = loss
            best_loss = loss
            result%best_loss = loss
            monitored_loss = loss
            if (present(validation_x)) then
                call mlp_loss_value_gradient(model, validation_x, validation_target, &
                    config%l2, validation_loss, gradient, l2_gradient, status)
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
        if (config%optimizer == MLP_OPTIMIZER_ADAM) then
            call optimizer%initialize(n_parameters, status, &
                learning_rate=config%learning_rate, beta1=config%beta1, &
                beta2=config%beta2, epsilon=config%epsilon)
        else if (config%optimizer == MLP_OPTIMIZER_SGD) then
            call sgd_optimizer%initialize(n_parameters, status, &
                learning_rate=config%learning_rate, momentum=config%momentum, &
                nesterov=config%nesterov)
        else
            call adamw_optimizer%initialize(n_parameters, status, &
                learning_rate=config%learning_rate, beta1=config%beta1, &
                beta2=config%beta2, epsilon=config%epsilon, &
                weight_decay=config%weight_decay)
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
            else
                adamw_optimizer%first_moment = checkpoint%first_moment
                adamw_optimizer%second_moment = checkpoint%second_moment
                adamw_optimizer%step_count = checkpoint%adam_step_count
                adamw_optimizer%learning_rate = checkpoint%last_learning_rate
                if (adamw_optimizer%learning_rate <= 0.0_dp) then
                    adamw_optimizer%learning_rate = config%learning_rate
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
            microbatch_count = 0
            if (resuming .and. resume_active_epoch .and. epoch == start_epoch) then
                accumulated_gradient = checkpoint%accumulated_gradient
                accumulated_samples = checkpoint%accumulated_samples
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
                x_batch = x(batch_indices, :)
                target_batch = target(batch_indices, :)
                ! Accumulate only the data term.  Add the L2 penalty once at
                ! the optimizer boundary so it is not counted once per
                ! microbatch.
                call mlp_loss_value_gradient(model, x_batch, target_batch, &
                    0.0_dp, loss, gradient, l2_gradient, status)
                deallocate(x_batch, target_batch)
                if (status%code /= FORTNUM_OK) then
                    if (present(state)) state = result
                    return
                end if
                accumulated_gradient = accumulated_gradient + &
                    real(size(batch_indices), dp)*gradient
                accumulated_samples = accumulated_samples + size(batch_indices)
                microbatch_count = microbatch_count + 1
                result%microbatches = result%microbatches + 1

                ! Flush at the configured boundary or at the uneven end of an
                ! epoch.  The latter keeps every sample in the final update.
                if (microbatch_count >= config%accumulation_steps .or. &
                    iterator%current_position() > n_samples) then
                    if (microbatch_count > 0) then
                        gradient = accumulated_gradient/ &
                            real(accumulated_samples, dp)
                        theta = model%parameters()
                        gradient = gradient + config%l2*theta
                        raw_gradient_norm = sqrt(sum(gradient*gradient))
                        if (config%gradient_clip_norm > 0.0_dp .and. &
                            raw_gradient_norm > config%gradient_clip_norm) then
                            gradient = gradient*config%gradient_clip_norm/ &
                                raw_gradient_norm
                            result%gradient_clipped_updates = &
                                result%gradient_clipped_updates + 1
                        end if
                        effective_rate = config%learning_rate
                        if (associated(config%learning_rate_schedule)) then
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
                        else
                            adamw_optimizer%learning_rate = effective_rate
                        end if
                        result%last_learning_rate = effective_rate
                        if (config%optimizer == MLP_OPTIMIZER_ADAM) then
                            call optimizer%step(theta, gradient, status)
                        else if (config%optimizer == MLP_OPTIMIZER_SGD) then
                            call sgd_optimizer%step(theta, gradient, status)
                        else
                            call adamw_optimizer%step(theta, gradient, status)
                        end if
                        if (status%code /= FORTNUM_OK) then
                            if (present(state)) state = result
                            return
                        end if
                        call model%set_parameters(theta, status)
                        if (status%code /= FORTNUM_OK) then
                            if (present(state)) state = result
                            return
                        end if
                        result%updates = result%updates + 1
                        accumulated_gradient = 0.0_dp
                        accumulated_samples = 0
                        microbatch_count = 0
                    end if
                end if
                if (present(checkpoint)) then
                    call checkpoint_capture(checkpoint, x, target, config, result, &
                        iterator, optimizer, adamw_optimizer, sgd_optimizer, theta, &
                        best_theta, stale_epochs, &
                        epoch, microbatch_count, accumulated_samples, &
                        accumulated_gradient, present(validation_x), status)
                    if (status%code /= FORTNUM_OK) then
                        if (present(state)) state = result
                        return
                    end if
                end if
            end do

            call mlp_loss_value_gradient(model, x, target, config%l2, loss, &
                gradient, l2_gradient, status)
            if (status%code /= FORTNUM_OK) then
                if (present(state)) state = result
                return
            end if
            gradient_norm = sqrt(sum(gradient*gradient))
            result%epochs = epoch
            result%loss_history(epoch) = loss
            result%learning_rate_history(epoch) = result%last_learning_rate
            result%gradient_norm = gradient_norm
            if (present(validation_x) .and. &
                mod(epoch, config%validation_interval) == 0) then
                call mlp_loss_value_gradient(model, validation_x, validation_target, &
                    config%l2, validation_loss, gradient, l2_gradient, status)
                if (status%code /= FORTNUM_OK) then
                    if (present(state)) state = result
                    return
                end if
                result%validation_loss_history(epoch) = validation_loss
                monitored_loss = validation_loss
            else
                monitored_loss = loss
            end if
            if (present(validation_x) .and. &
                mod(epoch, config%validation_interval) /= 0) then
                improvement = -1.0_dp
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
                if (.not. present(validation_x) .or. &
                    mod(epoch, config%validation_interval) == 0) then
                    stale_epochs = stale_epochs + 1
                end if
            end if
            stop_now = .false.
            if (associated(config%callback)) then
                call config%callback(epoch, loss, gradient_norm, stop_now)
            end if
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
                    iterator, optimizer, adamw_optimizer, sgd_optimizer, theta, &
                    best_theta, stale_epochs, &
                    epoch, microbatch_count, accumulated_samples, &
                    accumulated_gradient, present(validation_x), status)
                if (status%code /= FORTNUM_OK) then
                    if (present(state)) state = result
                    return
                end if
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
            call model%set_parameters(theta, status)
            if (status%code /= FORTNUM_OK) then
                if (present(state)) state = result
                return
            end if
            call mlp_loss_value_gradient(model, x, target, config%l2, loss, &
                gradient, l2_gradient, status)
            if (status%code /= FORTNUM_OK) then
                if (present(state)) state = result
                return
            end if
            result%gradient_norm = sqrt(sum(gradient*gradient))
            if (present(checkpoint)) checkpoint%resume_safe = .false.
        end if
        result%final_loss = loss
        if (present(validation_x)) then
            call mlp_loss_value_gradient(model, validation_x, validation_target, &
                config%l2, validation_loss, gradient, l2_gradient, status)
            if (status%code /= FORTNUM_OK) then
                if (present(state)) state = result
                return
            end if
            result%final_validation_loss = validation_loss
        end if
        if (present(state)) state = result
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_train

    subroutine checkpoint_capture(checkpoint, x, target, config, result, iterator, &
            optimizer, adamw_optimizer, sgd_optimizer, theta, best_theta, &
            stale_epochs, active_epoch, &
            active_microbatches, accumulated_samples, accumulated_gradient, &
            has_validation, status)
        type(mlp_training_checkpoint_t), intent(inout) :: checkpoint
        real(dp), intent(in) :: x(:, :), target(:, :)
        type(mlp_training_options_t), intent(in) :: config
        type(mlp_training_state_t), intent(in) :: result
        type(mlp_batch_iterator_t), intent(in) :: iterator
        type(adam_t), intent(in) :: optimizer
        type(adamw_t), intent(in) :: adamw_optimizer
        type(sgd_t), intent(in) :: sgd_optimizer
        real(dp), intent(in) :: theta(:), best_theta(:)
        integer, intent(in) :: stale_epochs
        integer, intent(in) :: active_epoch, active_microbatches, accumulated_samples
        real(dp), intent(in) :: accumulated_gradient(:)
        logical, intent(in) :: has_validation
        type(fortnum_status_t), intent(out) :: status
        integer :: n
        logical :: invalid_state

        invalid_state = size(theta) < 1 .or. size(best_theta) /= size(theta) .or. &
            size(accumulated_gradient) /= size(theta) .or. &
            active_epoch < result%epochs .or. active_microbatches < 0 .or. &
            accumulated_samples < 0 .or. .not. iterator%initialized()
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
        else
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
        end if
        if (invalid_state) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP checkpoint: optimizer or parameter state is invalid")
            return
        end if
        call checkpoint%clear()
        checkpoint%format_version = 2
        checkpoint%initialized = .true.
        checkpoint%resume_safe = .true.
        checkpoint%n_samples = size(x, 1)
        checkpoint%n_features = size(x, 2)
        checkpoint%n_outputs = size(target, 2)
        checkpoint%n_parameters = size(theta)
        checkpoint%epoch = result%epochs
        checkpoint%updates = result%updates
        checkpoint%microbatches = result%microbatches
        checkpoint%gradient_clipped_updates = result%gradient_clipped_updates
        checkpoint%microbatch_position = iterator%position
        checkpoint%active_epoch = active_epoch
        checkpoint%active_microbatches = active_microbatches
        checkpoint%accumulated_samples = accumulated_samples
        checkpoint%iterator_epoch = iterator%epoch_number
        checkpoint%iterator_position = iterator%position
        checkpoint%batch_size = iterator%batch_size
        checkpoint%accumulation_steps = config%accumulation_steps
        checkpoint%shuffle_seed = config%shuffle_seed
        checkpoint%optimizer = config%optimizer
        if (config%optimizer == MLP_OPTIMIZER_ADAM) then
            checkpoint%adam_step_count = optimizer%step_count
        else if (config%optimizer == MLP_OPTIMIZER_SGD) then
            checkpoint%adam_step_count = sgd_optimizer%step_count
        else
            checkpoint%adam_step_count = adamw_optimizer%step_count
        end if
        checkpoint%stale_epochs = stale_epochs
        checkpoint%validation_interval = config%validation_interval
        checkpoint%patience = config%patience
        checkpoint%shuffle = config%shuffle
        checkpoint%has_validation = has_validation
        checkpoint%converged = result%converged
        checkpoint%early_stopped = result%early_stopped
        checkpoint%restore_best = config%restore_best
        checkpoint%shuffle_state = iterator%shuffle_state
        checkpoint%learning_rate = config%learning_rate
        checkpoint%beta1 = config%beta1
        checkpoint%beta2 = config%beta2
        checkpoint%epsilon = config%epsilon
        checkpoint%momentum = config%momentum
        checkpoint%nesterov = config%nesterov
        checkpoint%weight_decay = config%weight_decay
        checkpoint%l2 = config%l2
        checkpoint%tolerance = config%tolerance
        checkpoint%min_delta = config%min_delta
        checkpoint%gradient_clip_norm = config%gradient_clip_norm
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
        else
            allocate(checkpoint%first_moment, source=adamw_optimizer%first_moment)
            allocate(checkpoint%second_moment, source=adamw_optimizer%second_moment)
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

    logical function valid_options(options) result(valid)
        type(mlp_training_options_t), intent(in) :: options

        valid = options%max_epochs >= 1 .and. options%batch_size >= 0 .and. &
            options%accumulation_steps >= 1 .and. &
            options%patience >= 0 .and. options%learning_rate > 0.0_dp .and. &
            (options%optimizer == MLP_OPTIMIZER_ADAM .or. &
            options%optimizer == MLP_OPTIMIZER_SGD .or. &
            options%optimizer == MLP_OPTIMIZER_ADAMW) .and. &
            options%beta1 >= 0.0_dp .and. options%beta1 < 1.0_dp .and. &
            options%beta2 >= 0.0_dp .and. options%beta2 < 1.0_dp .and. &
            options%epsilon > 0.0_dp .and. options%l2 >= 0.0_dp .and. &
            options%momentum >= 0.0_dp .and. options%momentum < 1.0_dp .and. &
            options%tolerance >= 0.0_dp .and. options%min_delta >= 0.0_dp .and. &
            options%gradient_clip_norm >= 0.0_dp .and. &
            options%validation_interval >= 1
        valid = valid .and. ieee_is_finite(options%learning_rate) .and. &
            ieee_is_finite(options%beta1) .and. ieee_is_finite(options%beta2) .and. &
            ieee_is_finite(options%epsilon) .and. ieee_is_finite(options%l2) .and. &
            ieee_is_finite(options%momentum) .and. &
            ieee_is_finite(options%weight_decay) .and. &
            ieee_is_finite(options%tolerance) .and. &
            ieee_is_finite(options%min_delta) .and. &
            ieee_is_finite(options%gradient_clip_norm)
        valid = valid .and. options%weight_decay >= 0.0_dp
        if (options%shuffle) valid = valid .and. options%shuffle_seed > 0
        if (options%nesterov) valid = valid .and. &
            options%optimizer == MLP_OPTIMIZER_SGD .and. options%momentum > 0.0_dp
    end function valid_options

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
