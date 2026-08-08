module fortml_mlp_minibatch_hypergradient
    !! Exact hypergradients through a deterministic mini-batch SGD trajectory.
    !!
    !! The inner update is
    !!
    !!   theta_(k+1) = theta_k - eta * grad L_batch(theta_k, l2)
    !!
    !! for a fixed epoch/batch order.  The outer objective is validation MSE
    !! after the final update.  The packed search variable is
    !! `[log(learning_rate), log(l2)]`; both value/JVP/VJP products are
    !! analytic and use the MLP HVP for the trajectory sensitivity.  This
    !! provides a deterministic mini-batch boundary for FortOpt L-BFGS-B
    !! without pretending that stochastic or resident-CUDA trajectories are
    !! differentiable.  Such paths return a typed refusal.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use, intrinsic :: iso_fortran_env, only: int64
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, status_ok, &
        FORTNUM_OK, FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED, &
        FORTNUM_CONVERGENCE_ERROR
    use fortml_device, only: FORTML_DEVICE_CPU
    use fortml_mlp, only: mlp_t
    use fortml_mlp_training, only: mlp_loss_value_gradient, mlp_loss_hvp
    use fortopt_objective, only: objective_t
    use fortopt_lbfgsb, only: lbfgsb_t, lbfgsb_options_t, lbfgsb_result_t
    implicit none
    private

    integer, parameter, public :: MLP_MINIBATCH_HYPERPARAMETER_COUNT = 2
    integer, parameter, public :: MLP_MINIBATCH_LOG_LEARNING_RATE = 1
    integer, parameter, public :: MLP_MINIBATCH_LOG_L2 = 2

    type, public :: mlp_minibatch_hypergradient_metadata_t
        integer :: parameter_count = MLP_MINIBATCH_HYPERPARAMETER_COUNT
        integer :: log_learning_rate_index = MLP_MINIBATCH_LOG_LEARNING_RATE
        integer :: log_l2_index = MLP_MINIBATCH_LOG_L2
        integer :: epochs = 0
        integer :: batch_size = 0
        integer :: steps = 0
        integer :: samples_per_epoch = 0
        logical :: shuffle = .false.
    end type mlp_minibatch_hypergradient_metadata_t

    type, public :: mlp_minibatch_hypergradient_options_t
        !! Fixed mini-batch SGD trajectory configuration.
        integer :: epochs = 4
        integer :: batch_size = 0
        logical :: shuffle = .false.
        integer :: shuffle_seed = 17
        real(dp) :: learning_rate = 1.0e-2_dp
        real(dp) :: l2 = 1.0e-4_dp
        real(dp) :: lower_log_learning_rate = -12.0_dp
        real(dp) :: upper_log_learning_rate = 2.0_dp
        real(dp) :: lower_log_l2 = -20.0_dp
        real(dp) :: upper_log_l2 = 2.0_dp
        integer :: device_kind = FORTML_DEVICE_CPU
        integer :: memory = 8
        integer :: max_iterations = 100
        integer :: max_line_search = 40
        real(dp) :: gradient_tolerance = 1.0e-6_dp
        real(dp) :: step_tolerance = 1.0e-12_dp
        real(dp) :: objective_tolerance = 1.0e-12_dp
    end type mlp_minibatch_hypergradient_options_t

    type, public :: mlp_minibatch_hypergradient_result_t
        logical :: converged = .false.
        integer :: iterations = 0
        integer :: line_search_evaluations = 0
        real(dp) :: objective = huge(1.0_dp)
        real(dp) :: gradient_norm = huge(1.0_dp)
        real(dp) :: log_learning_rate = 0.0_dp
        real(dp) :: log_l2 = 0.0_dp
        real(dp) :: learning_rate = 0.0_dp
        real(dp) :: l2 = 0.0_dp
    end type mlp_minibatch_hypergradient_result_t

    type, public :: mlp_minibatch_hypergradient_objective_t
        !! FortOpt-compatible fixed-batch trajectory objective.
        private
        type(mlp_t), pointer :: model => null()
        real(dp), allocatable :: train_x(:, :), train_target(:, :)
        real(dp), allocatable :: validation_x(:, :), validation_target(:, :)
        real(dp), allocatable :: initial_parameters(:)
        integer, allocatable :: batch_indices(:), batch_offsets(:), batch_lengths(:)
        type(mlp_minibatch_hypergradient_metadata_t) :: layout
        real(dp) :: initial_log_learning_rate = 0.0_dp
        real(dp) :: initial_log_l2 = 0.0_dp
        logical :: initialized = .false.
    contains
        procedure, public :: initialize => minibatch_hypergradient_initialize
        procedure, public :: parameter_count => minibatch_hypergradient_parameter_count
        procedure, public :: metadata => minibatch_hypergradient_metadata
        procedure, public :: parameters => minibatch_hypergradient_parameters
        procedure, public :: value_gradient => minibatch_hypergradient_value_gradient
        procedure, public :: jvp => minibatch_hypergradient_jvp
        procedure, public :: vjp => minibatch_hypergradient_vjp
        procedure, public :: hvp => minibatch_hypergradient_hvp
        procedure, public :: fortopt => minibatch_hypergradient_fortopt
        procedure, public :: is_initialized => minibatch_hypergradient_is_initialized
    end type mlp_minibatch_hypergradient_objective_t

    public :: mlp_optimize_minibatch_hyperparameters

contains

    subroutine minibatch_hypergradient_initialize(self, model, train_x, train_target, &
            validation_x, validation_target, options, status)
        class(mlp_minibatch_hypergradient_objective_t), intent(out) :: self
        type(mlp_t), target, intent(inout) :: model
        real(dp), intent(in) :: train_x(:, :), train_target(:, :)
        real(dp), intent(in) :: validation_x(:, :), validation_target(:, :)
        type(mlp_minibatch_hypergradient_options_t), intent(in) :: options
        type(fortnum_status_t), intent(out) :: status
        integer :: n_samples, batch_size, batches_per_epoch, step, epoch
        integer :: source_offset, batch_start, batch_length, sample
        integer, allocatable :: order(:)
        integer(int64) :: shuffle_state
        !! Default-initialized instances, standing in for empty
        !! structure constructors: nvfortran rejects `T()` outright,
        !! and a declared local carries the same default init.
        type(mlp_minibatch_hypergradient_metadata_t) :: mlp_minibatch_hypergradient_metadata_t_default

        self%initialized = .false.
        self%layout = mlp_minibatch_hypergradient_metadata_t_default
        if (options%device_kind /= FORTML_DEVICE_CPU) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "MLP mini-batch hypergradient: CUDA trajectory is not resident")
            return
        end if
        if (.not. valid_options(options)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP mini-batch hypergradient: options are invalid")
            return
        end if
        if (.not. valid_data(model, train_x, train_target) .or. &
            .not. valid_data(model, validation_x, validation_target)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP mini-batch hypergradient: model or data is invalid")
            return
        end if

        n_samples = size(train_x, 1)
        batch_size = options%batch_size
        if (batch_size == 0) batch_size = n_samples
        if (batch_size > n_samples) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP mini-batch hypergradient: batch size exceeds sample count")
            return
        end if
        batches_per_epoch = (n_samples + batch_size - 1)/batch_size
        self%layout%epochs = options%epochs
        self%layout%batch_size = batch_size
        self%layout%steps = options%epochs*batches_per_epoch
        self%layout%samples_per_epoch = n_samples
        self%layout%shuffle = options%shuffle

        self%model => model
        allocate(self%train_x, source=train_x)
        allocate(self%train_target, source=train_target)
        allocate(self%validation_x, source=validation_x)
        allocate(self%validation_target, source=validation_target)
        allocate(self%initial_parameters, source=model%parameters())
        allocate(self%batch_indices(options%epochs*n_samples))
        allocate(self%batch_offsets(self%layout%steps))
        allocate(self%batch_lengths(self%layout%steps))
        allocate(order(n_samples))
        shuffle_state = int(options%shuffle_seed, int64)
        source_offset = 0
        step = 0
        do epoch = 1, options%epochs
            order = [(sample, sample=1, n_samples)]
            if (options%shuffle) call shuffle_order(order, shuffle_state)
            batch_start = 1
            do while (batch_start <= n_samples)
                step = step + 1
                batch_length = min(batch_size, n_samples - batch_start + 1)
                self%batch_offsets(step) = source_offset + 1
                self%batch_lengths(step) = batch_length
                self%batch_indices(source_offset + 1:source_offset + batch_length) = &
                    order(batch_start:batch_start + batch_length - 1)
                source_offset = source_offset + batch_length
                batch_start = batch_start + batch_length
            end do
        end do
        self%initial_log_learning_rate = log(options%learning_rate)
        self%initial_log_l2 = log(options%l2)
        self%initialized = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine minibatch_hypergradient_initialize

    integer function minibatch_hypergradient_parameter_count(self) result(count)
        class(mlp_minibatch_hypergradient_objective_t), intent(in) :: self

        count = 0
        if (self%initialized) count = self%layout%parameter_count
    end function minibatch_hypergradient_parameter_count

    function minibatch_hypergradient_metadata(self) result(layout)
        class(mlp_minibatch_hypergradient_objective_t), intent(in) :: self
        type(mlp_minibatch_hypergradient_metadata_t) :: layout

        layout = self%layout
    end function minibatch_hypergradient_metadata

    function minibatch_hypergradient_parameters(self) result(parameters)
        class(mlp_minibatch_hypergradient_objective_t), intent(in) :: self
        real(dp), allocatable :: parameters(:)

        allocate(parameters(MLP_MINIBATCH_HYPERPARAMETER_COUNT))
        parameters = [self%initial_log_learning_rate, self%initial_log_l2]
    end function minibatch_hypergradient_parameters

    logical function minibatch_hypergradient_is_initialized(self) result(yes)
        class(mlp_minibatch_hypergradient_objective_t), intent(in) :: self

        yes = self%initialized .and. associated(self%model) .and. &
            allocated(self%initial_parameters) .and. allocated(self%batch_indices)
    end function minibatch_hypergradient_is_initialized

    subroutine minibatch_hypergradient_value_gradient(self, parameters, value, &
            gradient, status)
        class(mlp_minibatch_hypergradient_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status

        value = huge(1.0_dp)
        gradient = 0.0_dp
        if (.not. self%is_initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP mini-batch hypergradient: objective is not initialized")
            return
        end if
        if (.not. valid_parameter_vector(parameters, gradient)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP mini-batch hypergradient: packed shape is invalid")
            return
        end if
        call reverse_value_gradient(self, parameters, value, gradient, status)
    end subroutine minibatch_hypergradient_value_gradient

    subroutine minibatch_hypergradient_jvp(self, parameters, direction, value, &
            tangent, status)
        class(mlp_minibatch_hypergradient_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:), direction(:)
        real(dp), intent(out) :: value, tangent
        type(fortnum_status_t), intent(out) :: status

        value = huge(1.0_dp)
        tangent = 0.0_dp
        if (.not. self%is_initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP mini-batch hypergradient JVP: objective is not initialized")
            return
        end if
        if (size(parameters) /= MLP_MINIBATCH_HYPERPARAMETER_COUNT .or. &
            size(direction) /= MLP_MINIBATCH_HYPERPARAMETER_COUNT .or. &
            any(.not. ieee_is_finite(parameters)) .or. &
            any(.not. ieee_is_finite(direction))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP mini-batch hypergradient JVP: packed shape is invalid")
            return
        end if
        call forward_jvp(self, parameters, direction, value, tangent, status)
    end subroutine minibatch_hypergradient_jvp

    subroutine minibatch_hypergradient_vjp(self, parameters, output_bar, gradient, &
            status)
        class(mlp_minibatch_hypergradient_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:), output_bar
        real(dp), intent(out) :: gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: value

        gradient = 0.0_dp
        if (.not. ieee_is_finite(output_bar)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP mini-batch hypergradient VJP: cotangent is invalid")
            return
        end if
        call self%value_gradient(parameters, value, gradient, status)
        if (.not. status_ok(status)) return
        gradient = output_bar*gradient
        if (any(.not. ieee_is_finite(gradient))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP mini-batch hypergradient VJP: product is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine minibatch_hypergradient_vjp

    subroutine minibatch_hypergradient_hvp(self, parameters, direction, product, status)
        !! Outer HVPs require third network derivatives and are refused exactly.
        class(mlp_minibatch_hypergradient_objective_t), intent(in) :: self
        real(dp), intent(in) :: parameters(:), direction(:)
        real(dp), intent(out) :: product(:)
        type(fortnum_status_t), intent(out) :: status

        product = 0.0_dp
        if (size(parameters) /= MLP_MINIBATCH_HYPERPARAMETER_COUNT .or. &
            size(direction) /= MLP_MINIBATCH_HYPERPARAMETER_COUNT .or. &
            size(product) /= MLP_MINIBATCH_HYPERPARAMETER_COUNT .or. &
            any(.not. ieee_is_finite(parameters)) .or. &
            any(.not. ieee_is_finite(direction))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP mini-batch hypergradient HVP: packed shape is invalid")
            return
        end if
        if (.not. self%is_initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP mini-batch hypergradient HVP: objective is not initialized")
            return
        end if
        call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
            "MLP mini-batch hypergradient HVP requires third network derivatives")
    end subroutine minibatch_hypergradient_hvp

    subroutine minibatch_hypergradient_fortopt(self, objective, status)
        class(mlp_minibatch_hypergradient_objective_t), target, intent(inout) :: self
        type(objective_t), intent(out) :: objective
        type(fortnum_status_t), intent(out) :: status

        if (.not. self%is_initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP mini-batch hypergradient: objective is not initialized")
            return
        end if
        call objective%initialize_context(MLP_MINIBATCH_HYPERPARAMETER_COUNT, self, &
            minibatch_hypergradient_context_callback, status)
    end subroutine minibatch_hypergradient_fortopt

    subroutine minibatch_hypergradient_context_callback(context, parameters, value, &
            gradient, status)
        class(*), intent(inout) :: context
        real(dp), intent(in) :: parameters(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status

        select type (adapter => context)
            type is (mlp_minibatch_hypergradient_objective_t)
            call adapter%value_gradient(parameters, value, gradient, status)
        class default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP mini-batch hypergradient: context has the wrong type")
        end select
    end subroutine minibatch_hypergradient_context_callback

    subroutine mlp_optimize_minibatch_hyperparameters(model, train_x, train_target, &
            validation_x, validation_target, options, result, status)
        type(mlp_t), target, intent(inout) :: model
        real(dp), intent(in) :: train_x(:, :), train_target(:, :)
        real(dp), intent(in) :: validation_x(:, :), validation_target(:, :)
        type(mlp_minibatch_hypergradient_options_t), intent(in) :: options
        type(mlp_minibatch_hypergradient_result_t), intent(out) :: result
        type(fortnum_status_t), intent(out) :: status
        type(mlp_minibatch_hypergradient_objective_t), target :: adapter
        type(objective_t) :: objective
        type(lbfgsb_t) :: optimizer
        type(lbfgsb_options_t) :: optimizer_options
        type(lbfgsb_result_t) :: optimizer_result
        real(dp) :: parameters(MLP_MINIBATCH_HYPERPARAMETER_COUNT)
        real(dp) :: lower(MLP_MINIBATCH_HYPERPARAMETER_COUNT)
        real(dp) :: upper(MLP_MINIBATCH_HYPERPARAMETER_COUNT)
        real(dp) :: gradient(MLP_MINIBATCH_HYPERPARAMETER_COUNT)
        !! Default-initialized instances, standing in for empty
        !! structure constructors: nvfortran rejects `T()` outright,
        !! and a declared local carries the same default init.
        type(mlp_minibatch_hypergradient_result_t) :: mlp_minibatch_hypergradient_result_t_default

        result = mlp_minibatch_hypergradient_result_t_default
        if (options%device_kind /= FORTML_DEVICE_CPU) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "MLP mini-batch hyperparameter optimization: CUDA is not resident")
            return
        end if
        if (.not. valid_options(options) .or. &
            .not. valid_data(model, train_x, train_target) .or. &
            .not. valid_data(model, validation_x, validation_target)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP mini-batch hyperparameter optimization: inputs are invalid")
            return
        end if
        call adapter%initialize(model, train_x, train_target, validation_x, &
            validation_target, options, status)
        if (.not. status_ok(status)) return
        parameters = adapter%parameters()
        lower = [options%lower_log_learning_rate, options%lower_log_l2]
        upper = [options%upper_log_learning_rate, options%upper_log_l2]
        call adapter%fortopt(objective, status)
        if (.not. status_ok(status)) return
        optimizer_options%memory = options%memory
        optimizer_options%max_iterations = options%max_iterations
        optimizer_options%max_line_search = options%max_line_search
        optimizer_options%gradient_tolerance = options%gradient_tolerance
        optimizer_options%step_tolerance = options%step_tolerance
        optimizer_options%objective_tolerance = options%objective_tolerance
        call optimizer%minimize(objective, parameters, lower, upper, &
            optimizer_options, optimizer_result, status)
        if (.not. status_ok(status)) return
        call adapter%value_gradient(parameters, result%objective, gradient, status)
        if (.not. status_ok(status)) return
        result%converged = optimizer_result%state%converged
        result%iterations = optimizer_result%state%iteration
        result%line_search_evaluations = optimizer_result%line_search_evaluations
        result%gradient_norm = sqrt(sum(gradient*gradient))
        result%log_learning_rate = parameters(MLP_MINIBATCH_LOG_LEARNING_RATE)
        result%log_l2 = parameters(MLP_MINIBATCH_LOG_L2)
        result%learning_rate = exp(result%log_learning_rate)
        result%l2 = exp(result%log_l2)
        if (.not. result%converged) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "MLP mini-batch hyperparameter optimization: iteration limit reached")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_optimize_minibatch_hyperparameters

    subroutine reverse_value_gradient(self, parameters, value, gradient, status)
        class(mlp_minibatch_hypergradient_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: theta_history(:, :), gradient_history(:, :)
        real(dp), allocatable :: theta_bar(:), gradient_theta(:), hvp(:)
        real(dp), allocatable :: x_batch(:, :), target_batch(:, :)
        real(dp) :: learning_rate, l2, l2_gradient, validation_l2_gradient
        real(dp) :: train_value, coefficient
        integer :: n_parameters, step

        value = huge(1.0_dp)
        gradient = 0.0_dp
        call unpack_parameters(parameters, learning_rate, l2, status)
        if (.not. status_ok(status)) return
        n_parameters = size(self%initial_parameters)
        allocate(theta_history(n_parameters, self%layout%steps + 1))
        allocate(gradient_history(n_parameters, self%layout%steps))
        allocate(gradient_theta(n_parameters), hvp(n_parameters))
        theta_history(:, 1) = self%initial_parameters
        do step = 1, self%layout%steps
            call self%model%set_parameters(theta_history(:, step), status)
            if (.not. status_ok(status)) return
            call make_batch(self, step, x_batch, target_batch)
            call mlp_loss_value_gradient(self%model, x_batch, target_batch, l2, &
                train_value, gradient_theta, l2_gradient, status)
            deallocate(x_batch, target_batch)
            if (.not. status_ok(status)) return
            gradient_history(:, step) = gradient_theta
            theta_history(:, step + 1) = theta_history(:, step) - &
                learning_rate*gradient_theta
            if (any(.not. ieee_is_finite(theta_history(:, step + 1)))) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP mini-batch hypergradient: trajectory is not finite")
                return
            end if
        end do
        call self%model%set_parameters(theta_history(:, self%layout%steps + 1), status)
        if (.not. status_ok(status)) return
        call mlp_loss_value_gradient(self%model, self%validation_x, self%validation_target, &
            0.0_dp, value, gradient_theta, validation_l2_gradient, status)
        if (.not. status_ok(status)) return
        allocate(theta_bar(n_parameters))
        theta_bar = gradient_theta
        do step = self%layout%steps, 1, -1
            call self%model%set_parameters(theta_history(:, step), status)
            if (.not. status_ok(status)) return
            call make_batch(self, step, x_batch, target_batch)
            call mlp_loss_hvp(self%model, x_batch, target_batch, l2, theta_bar, &
                0.0_dp, hvp, l2_gradient, status)
            deallocate(x_batch, target_batch)
            if (.not. status_ok(status)) return
            coefficient = dot_product(theta_bar, gradient_history(:, step))
            gradient(MLP_MINIBATCH_LOG_LEARNING_RATE) = &
                gradient(MLP_MINIBATCH_LOG_LEARNING_RATE) - learning_rate*coefficient
            gradient(MLP_MINIBATCH_LOG_L2) = gradient(MLP_MINIBATCH_LOG_L2) - &
                learning_rate*l2*dot_product(theta_bar, theta_history(:, step))
            theta_bar = theta_bar - learning_rate*hvp
        end do
        if (.not. ieee_is_finite(value) .or. any(.not. ieee_is_finite(gradient))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP mini-batch hypergradient: product is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine reverse_value_gradient

    subroutine forward_jvp(self, parameters, direction, value, tangent, status)
        class(mlp_minibatch_hypergradient_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:), direction(:)
        real(dp), intent(out) :: value, tangent
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: theta(:), theta_dot(:), gradient(:), hvp(:)
        real(dp), allocatable :: x_batch(:, :), target_batch(:, :)
        real(dp) :: learning_rate, l2, learning_rate_dot, l2_dot
        real(dp) :: train_value, l2_gradient, validation_l2_gradient
        integer :: n_parameters, step

        value = huge(1.0_dp)
        tangent = 0.0_dp
        call unpack_parameters(parameters, learning_rate, l2, status)
        if (.not. status_ok(status)) return
        learning_rate_dot = learning_rate*direction(MLP_MINIBATCH_LOG_LEARNING_RATE)
        l2_dot = l2*direction(MLP_MINIBATCH_LOG_L2)
        n_parameters = size(self%initial_parameters)
        allocate(theta, source=self%initial_parameters)
        allocate(theta_dot(n_parameters), gradient(n_parameters), hvp(n_parameters))
        theta_dot = 0.0_dp
        do step = 1, self%layout%steps
            call self%model%set_parameters(theta, status)
            if (.not. status_ok(status)) return
            call make_batch(self, step, x_batch, target_batch)
            call mlp_loss_value_gradient(self%model, x_batch, target_batch, l2, &
                train_value, gradient, l2_gradient, status)
            if (.not. status_ok(status)) then
                deallocate(x_batch, target_batch)
                return
            end if
            call mlp_loss_hvp(self%model, x_batch, target_batch, l2, theta_dot, &
                l2_dot, hvp, l2_gradient, status)
            deallocate(x_batch, target_batch)
            if (.not. status_ok(status)) return
            theta_dot = theta_dot - learning_rate*hvp - learning_rate_dot*gradient
            theta = theta - learning_rate*gradient
            if (any(.not. ieee_is_finite(theta)) .or. &
                any(.not. ieee_is_finite(theta_dot))) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP mini-batch hypergradient JVP: trajectory is not finite")
                return
            end if
        end do
        call self%model%set_parameters(theta, status)
        if (.not. status_ok(status)) return
        call mlp_loss_value_gradient(self%model, self%validation_x, self%validation_target, &
            0.0_dp, value, gradient, validation_l2_gradient, status)
        if (.not. status_ok(status)) return
        tangent = dot_product(gradient, theta_dot)
        if (.not. ieee_is_finite(value) .or. .not. ieee_is_finite(tangent)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP mini-batch hypergradient JVP: product is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine forward_jvp

    subroutine make_batch(self, step, x_batch, target_batch)
        class(mlp_minibatch_hypergradient_objective_t), intent(in) :: self
        integer, intent(in) :: step
        real(dp), allocatable, intent(out) :: x_batch(:, :), target_batch(:, :)
        integer :: offset, length, i, source

        offset = self%batch_offsets(step)
        length = self%batch_lengths(step)
        allocate(x_batch(length, size(self%train_x, 2)))
        allocate(target_batch(length, size(self%train_target, 2)))
        do i = 1, length
            source = self%batch_indices(offset + i - 1)
            x_batch(i, :) = self%train_x(source, :)
            target_batch(i, :) = self%train_target(source, :)
        end do
    end subroutine make_batch

    subroutine unpack_parameters(parameters, learning_rate, l2, status)
        real(dp), intent(in) :: parameters(:)
        real(dp), intent(out) :: learning_rate, l2
        type(fortnum_status_t), intent(out) :: status

        learning_rate = 0.0_dp
        l2 = 0.0_dp
        if (size(parameters) /= MLP_MINIBATCH_HYPERPARAMETER_COUNT .or. &
            any(.not. ieee_is_finite(parameters))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP mini-batch hypergradient: packed parameters are invalid")
            return
        end if
        learning_rate = exp(parameters(MLP_MINIBATCH_LOG_LEARNING_RATE))
        l2 = exp(parameters(MLP_MINIBATCH_LOG_L2))
        if (.not. ieee_is_finite(learning_rate) .or. .not. ieee_is_finite(l2) .or. &
            learning_rate <= 0.0_dp .or. l2 <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP mini-batch hypergradient: positive log parameters required")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine unpack_parameters

    logical function valid_parameter_vector(parameters, gradient) result(valid)
        real(dp), intent(in) :: parameters(:), gradient(:)

        valid = size(parameters) == MLP_MINIBATCH_HYPERPARAMETER_COUNT .and. &
            size(gradient) == MLP_MINIBATCH_HYPERPARAMETER_COUNT .and. &
            all(ieee_is_finite(parameters))
    end function valid_parameter_vector

    logical function valid_options(options) result(valid)
        type(mlp_minibatch_hypergradient_options_t), intent(in) :: options

        valid = .false.
        if (options%epochs < 1 .or. options%batch_size < 0 .or. &
            options%shuffle_seed <= 0) return
        if (.not. ieee_is_finite(options%learning_rate) .or. &
            .not. ieee_is_finite(options%l2)) return
        if (options%learning_rate <= 0.0_dp .or. options%l2 <= 0.0_dp) return
        if (.not. ieee_is_finite(options%lower_log_learning_rate) .or. &
            .not. ieee_is_finite(options%upper_log_learning_rate) .or. &
            .not. ieee_is_finite(options%lower_log_l2) .or. &
            .not. ieee_is_finite(options%upper_log_l2)) return
        if (options%lower_log_learning_rate > options%upper_log_learning_rate .or. &
            options%lower_log_l2 > options%upper_log_l2) return
        if (log(options%learning_rate) < options%lower_log_learning_rate .or. &
            log(options%learning_rate) > options%upper_log_learning_rate .or. &
            log(options%l2) < options%lower_log_l2 .or. &
            log(options%l2) > options%upper_log_l2) return
        if (options%memory < 1 .or. options%max_iterations < 1 .or. &
            options%max_line_search < 1) return
        if (.not. ieee_is_finite(options%gradient_tolerance) .or. &
            .not. ieee_is_finite(options%step_tolerance) .or. &
            .not. ieee_is_finite(options%objective_tolerance)) return
        if (options%gradient_tolerance < 0.0_dp .or. &
            options%step_tolerance < 0.0_dp .or. &
            options%objective_tolerance < 0.0_dp) return
        valid = .true.
    end function valid_options

    logical function valid_data(model, x, target) result(valid)
        class(mlp_t), intent(in) :: model
        real(dp), intent(in) :: x(:, :), target(:, :)

        valid = .false.
        if (.not. allocated(model%layer_sizes)) return
        if (size(model%layer_sizes) < 2 .or. model%parameter_count() <= 0) return
        if (size(x, 1) < 1 .or. size(x, 2) /= model%layer_sizes(1)) return
        if (size(target, 1) /= size(x, 1) .or. &
            size(target, 2) /= model%layer_sizes(size(model%layer_sizes))) return
        valid = all(ieee_is_finite(x)) .and. all(ieee_is_finite(target))
    end function valid_data

    subroutine shuffle_order(order, state)
        integer, intent(inout) :: order(:)
        integer(int64), intent(inout) :: state
        integer :: i, j, temporary

        do i = size(order), 2, -1
            state = modulo(48271_int64*state, 2147483647_int64)
            j = 1 + int(modulo(state, int(i, int64)))
            temporary = order(i)
            order(i) = order(j)
            order(j) = temporary
        end do
    end subroutine shuffle_order

end module fortml_mlp_minibatch_hypergradient
