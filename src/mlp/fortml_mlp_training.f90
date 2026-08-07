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
        FORTNUM_DOMAIN_ERROR
    use fortml_mlp, only: mlp_t
    use fortopt_objective, only: objective_t
    use fortopt_adam, only: adam_t
    implicit none
    private

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
        integer :: patience = 0
        integer :: shuffle_seed = 17
        logical :: shuffle = .false.
        logical :: restore_best = .true.
        real(dp) :: learning_rate = 1.0e-3_dp
        real(dp) :: beta1 = 0.9_dp
        real(dp) :: beta2 = 0.999_dp
        real(dp) :: epsilon = 1.0e-8_dp
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
        logical :: converged = .false.
        logical :: early_stopped = .false.
        integer :: gradient_clipped_updates = 0
        real(dp) :: initial_loss = huge(1.0_dp)
        real(dp) :: final_loss = huge(1.0_dp)
        real(dp) :: best_loss = huge(1.0_dp)
        real(dp) :: gradient_norm = huge(1.0_dp)
        real(dp) :: last_learning_rate = 0.0_dp
        real(dp), allocatable :: loss_history(:)
        real(dp), allocatable :: learning_rate_history(:)
    end type mlp_training_state_t

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

    public :: mlp_epoch_callback_proc
    public :: mlp_learning_rate_schedule_proc
    public :: mlp_loss_value_gradient
    public :: mlp_loss_hvp
    public :: mlp_train

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

    subroutine mlp_loss_value_gradient(model, x, target, l2, value, gradient, &
            l2_gradient, status)
        !! MSE value, network-parameter gradient, and derivative with respect
        !! to the scalar L2 coefficient.
        class(mlp_t), intent(in) :: model
        real(dp), intent(in) :: x(:, :), target(:, :), l2
        real(dp), intent(out) :: value, gradient(:), l2_gradient
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: prediction(:, :), residual(:, :), x_bar(:, :)
        real(dp), allocatable :: theta(:)
        integer :: n_samples

        value = 0.0_dp
        l2_gradient = 0.0_dp
        if (.not. valid_data(model, x, target) .or. l2 < 0.0_dp .or. &
            size(gradient) /= model%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP loss: model, data, penalty, or gradient shape is invalid")
            return
        end if
        n_samples = size(x, 1)
        allocate(prediction(size(target, 1), size(target, 2)))
        allocate(residual, mold=prediction)
        allocate(x_bar, mold=x)
        call model%predict(x, prediction, status)
        if (status%code /= FORTNUM_OK) return
        residual = prediction - target
        value = 0.5_dp*sum(residual*residual)/real(n_samples, dp)
        call model%vjp(x, residual/real(n_samples, dp), gradient, x_bar, status)
        if (status%code /= FORTNUM_OK) return
        theta = model%parameters()
        value = value + 0.5_dp*l2*sum(theta*theta)
        l2_gradient = 0.5_dp*sum(theta*theta)
        gradient = gradient + l2*theta
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

    subroutine mlp_train(model, x, target, status, options, state)
        !! Train `model` with deterministic Adam updates.
        !!
        !! A zero batch size selects full-batch updates.  When shuffling is
        !! enabled, an explicit Park--Miller stream seeded by `shuffle_seed`
        !! drives Fisher--Yates permutations; no process-global RNG state is
        !! touched.  Callback execution occurs once per completed epoch.
        class(mlp_t), intent(inout) :: model
        real(dp), intent(in) :: x(:, :), target(:, :)
        type(fortnum_status_t), intent(out) :: status
        type(mlp_training_options_t), intent(in), optional :: options
        type(mlp_training_state_t), intent(out), optional :: state
        type(mlp_training_options_t) :: config
        type(mlp_training_state_t) :: result
        type(adam_t) :: optimizer
        type(mlp_batch_iterator_t) :: iterator
        real(dp), allocatable :: theta(:), best_theta(:), gradient(:)
        real(dp), allocatable :: accumulated_gradient(:)
        real(dp), allocatable :: x_batch(:, :), target_batch(:, :)
        integer, allocatable :: batch_indices(:)
        real(dp) :: loss, l2_gradient, gradient_norm, improvement
        real(dp) :: best_loss
        real(dp) :: effective_rate, raw_gradient_norm
        integer :: n_samples, n_outputs, n_parameters
        integer :: batch, epoch
        integer :: microbatch_count, accumulated_samples
        integer :: stale_epochs
        logical :: stop_now, has_batch

        if (present(options)) config = options
        if (.not. valid_options(config) .or. &
            .not. valid_data(model, x, target)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP train: invalid model, data, or options")
            if (present(state)) state = result
            return
        end if

        n_samples = size(x, 1)
        n_outputs = size(target, 2)
        n_parameters = model%parameter_count()
        batch = config%batch_size
        if (batch == 0) batch = n_samples
        batch = min(batch, n_samples)

        theta = model%parameters()
        allocate(best_theta, source=theta)
        allocate(gradient(n_parameters))
        allocate(accumulated_gradient(n_parameters))
        allocate(result%loss_history(config%max_epochs))
        allocate(result%learning_rate_history(config%max_epochs))
        result%accumulation_steps = config%accumulation_steps
        call mlp_loss_value_gradient(model, x, target, config%l2, loss, &
            gradient, l2_gradient, status)
        if (status%code /= FORTNUM_OK) then
            if (present(state)) state = result
            return
        end if
        result%initial_loss = loss
        best_loss = loss
        result%best_loss = loss
        stale_epochs = 0
        call optimizer%initialize(n_parameters, status, &
            learning_rate=config%learning_rate, beta1=config%beta1, &
            beta2=config%beta2, epsilon=config%epsilon)
        if (status%code /= FORTNUM_OK) then
            if (present(state)) state = result
            return
        end if
        call iterator%initialize(n_samples, status, batch_size=batch, &
            shuffle=config%shuffle, seed=config%shuffle_seed)
        if (status%code /= FORTNUM_OK) then
            if (present(state)) state = result
            return
        end if

        do epoch = 1, config%max_epochs
            call iterator%reset(status)
            if (status%code /= FORTNUM_OK) then
                if (present(state)) state = result
                return
            end if
            accumulated_gradient = 0.0_dp
            accumulated_samples = 0
            microbatch_count = 0
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
                        optimizer%learning_rate = effective_rate
                        result%last_learning_rate = effective_rate
                        call optimizer%step(theta, gradient, status)
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
            improvement = best_loss - loss
            if (improvement > config%min_delta) then
                best_loss = loss
                result%best_loss = loss
                result%best_epoch = epoch
                best_theta = theta
                stale_epochs = 0
            else
                stale_epochs = stale_epochs + 1
            end if
            stop_now = .false.
            if (associated(config%callback)) then
                call config%callback(epoch, loss, gradient_norm, stop_now)
            end if
            if (gradient_norm <= config%tolerance) then
                result%converged = .true.
                exit
            end if
            if (stop_now) then
                result%early_stopped = .true.
                exit
            end if
            if (config%patience > 0 .and. stale_epochs >= config%patience) then
                result%early_stopped = .true.
                exit
            end if
        end do

        call shrink_history(result%loss_history, result%epochs)
        call shrink_history(result%learning_rate_history, result%epochs)
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
        end if
        result%final_loss = loss
        if (present(state)) state = result
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_train

    logical function valid_options(options) result(valid)
        type(mlp_training_options_t), intent(in) :: options

        valid = options%max_epochs >= 1 .and. options%batch_size >= 0 .and. &
            options%accumulation_steps >= 1 .and. &
            options%patience >= 0 .and. options%learning_rate > 0.0_dp .and. &
            options%beta1 >= 0.0_dp .and. options%beta1 < 1.0_dp .and. &
            options%beta2 >= 0.0_dp .and. options%beta2 < 1.0_dp .and. &
            options%epsilon > 0.0_dp .and. options%l2 >= 0.0_dp .and. &
            options%tolerance >= 0.0_dp .and. options%min_delta >= 0.0_dp .and. &
            options%gradient_clip_norm >= 0.0_dp
        valid = valid .and. ieee_is_finite(options%learning_rate) .and. &
            ieee_is_finite(options%beta1) .and. ieee_is_finite(options%beta2) .and. &
            ieee_is_finite(options%epsilon) .and. ieee_is_finite(options%l2) .and. &
            ieee_is_finite(options%tolerance) .and. &
            ieee_is_finite(options%min_delta) .and. &
            ieee_is_finite(options%gradient_clip_norm)
        if (options%shuffle) valid = valid .and. options%shuffle_seed > 0
    end function valid_options

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
