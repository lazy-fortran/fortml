module fortml_mlp_sgd_momentum_hypergradient
    !! Exact fixed full-batch SGD momentum/Nesterov trajectory products.
    !!
    !! The packed outer vector is
    !! `[log(learning_rate), log(l2), momentum]`.  The inner trajectory starts
    !! from the parameters present when `initialize` is called and performs a
    !! fixed number of full-batch updates.  Both classical momentum and the
    !! Nesterov look-ahead update use the same recurrence as `fortopt_sgd`.
    !! Forward sensitivities use the MLP analytic Hessian-vector product; no
    !! finite differences or optimizer fallback are used.  CUDA and stochastic
    !! trajectories are explicit typed refusals until resident state products
    !! are available.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED, FORTNUM_CONVERGENCE_ERROR
    use fortml_device, only: FORTML_DEVICE_CPU
    use fortml_mlp, only: mlp_t
    use fortml_mlp_training, only: mlp_loss_value_gradient, mlp_loss_hvp, &
        MLP_OPTIMIZER_SGD
    use fortopt_objective, only: objective_t
    use fortopt_lbfgsb, only: lbfgsb_t, lbfgsb_options_t, lbfgsb_result_t
    implicit none
    private

    integer, parameter, public :: MLP_SGD_MOMENTUM_HYPERPARAMETER_COUNT = 3
    integer, parameter, public :: MLP_SGD_LOG_LEARNING_RATE = 1
    integer, parameter, public :: MLP_SGD_LOG_L2 = 2
    integer, parameter, public :: MLP_SGD_MOMENTUM = 3

    type, public :: mlp_sgd_momentum_hypergradient_metadata_t
        integer :: parameter_count = MLP_SGD_MOMENTUM_HYPERPARAMETER_COUNT
        integer :: log_learning_rate_index = MLP_SGD_LOG_LEARNING_RATE
        integer :: log_l2_index = MLP_SGD_LOG_L2
        integer :: momentum_index = MLP_SGD_MOMENTUM
        integer :: inner_steps = 0
        logical :: nesterov = .false.
    end type mlp_sgd_momentum_hypergradient_metadata_t

    type, public :: mlp_sgd_momentum_hypergradient_options_t
        !! Fixed full-batch SGD momentum/Nesterov trajectory configuration.
        integer :: steps = 8
        real(dp) :: learning_rate = 1.0e-2_dp
        real(dp) :: l2 = 1.0e-4_dp
        real(dp) :: momentum = 0.9_dp
        logical :: nesterov = .false.
        real(dp) :: lower_log_learning_rate = -12.0_dp
        real(dp) :: upper_log_learning_rate = 2.0_dp
        real(dp) :: lower_log_l2 = -20.0_dp
        real(dp) :: upper_log_l2 = 2.0_dp
        real(dp) :: lower_momentum = 0.0_dp
        real(dp) :: upper_momentum = 0.999999_dp
        integer :: optimizer = MLP_OPTIMIZER_SGD
        integer :: device_kind = FORTML_DEVICE_CPU
        integer :: memory = 8
        integer :: max_iterations = 100
        integer :: max_line_search = 40
        real(dp) :: gradient_tolerance = 1.0e-6_dp
        real(dp) :: step_tolerance = 1.0e-12_dp
        real(dp) :: objective_tolerance = 1.0e-12_dp
    end type mlp_sgd_momentum_hypergradient_options_t

    type, public :: mlp_sgd_momentum_hypergradient_result_t
        logical :: converged = .false.
        integer :: iterations = 0
        integer :: line_search_evaluations = 0
        real(dp) :: objective = huge(1.0_dp)
        real(dp) :: gradient_norm = huge(1.0_dp)
        real(dp) :: log_learning_rate = 0.0_dp
        real(dp) :: log_l2 = 0.0_dp
        real(dp) :: momentum = 0.0_dp
        real(dp) :: learning_rate = 0.0_dp
        real(dp) :: l2 = 0.0_dp
    end type mlp_sgd_momentum_hypergradient_result_t

    type, public :: mlp_sgd_momentum_hypergradient_objective_t
        !! Exact value/gradient/JVP/VJP products through the SGD state.
        private
        type(mlp_t), pointer :: model => null()
        real(dp), allocatable :: train_x(:, :), train_target(:, :)
        real(dp), allocatable :: validation_x(:, :), validation_target(:, :)
        real(dp), allocatable :: initial_parameters(:)
        type(mlp_sgd_momentum_hypergradient_metadata_t) :: layout
        real(dp) :: initial_log_learning_rate = 0.0_dp
        real(dp) :: initial_log_l2 = 0.0_dp
        real(dp) :: initial_momentum = 0.0_dp
        logical :: initialized = .false.
    contains
        procedure, public :: initialize => mlp_sgd_momentum_hypergradient_initialize
        procedure, public :: parameter_count => mlp_sgd_momentum_hypergradient_parameter_count
        procedure, public :: metadata => mlp_sgd_momentum_hypergradient_metadata
        procedure, public :: parameters => mlp_sgd_momentum_hypergradient_parameters
        procedure, public :: value_gradient => mlp_sgd_momentum_hypergradient_value_gradient
        procedure, public :: jvp => mlp_sgd_momentum_hypergradient_jvp
        procedure, public :: vjp => mlp_sgd_momentum_hypergradient_vjp
        procedure, public :: fortopt => mlp_sgd_momentum_hypergradient_fortopt
        procedure, public :: is_initialized => mlp_sgd_momentum_hypergradient_is_initialized
    end type mlp_sgd_momentum_hypergradient_objective_t

    public :: mlp_optimize_sgd_momentum_hyperparameters

contains

    subroutine mlp_sgd_momentum_hypergradient_initialize(self, model, train_x, &
            train_target, validation_x, validation_target, options, status)
        class(mlp_sgd_momentum_hypergradient_objective_t), intent(out) :: self
        type(mlp_t), target, intent(inout) :: model
        real(dp), intent(in) :: train_x(:, :), train_target(:, :)
        real(dp), intent(in) :: validation_x(:, :), validation_target(:, :)
        type(mlp_sgd_momentum_hypergradient_options_t), intent(in) :: options
        type(fortnum_status_t), intent(out) :: status

        self%initialized = .false.
        self%layout = mlp_sgd_momentum_hypergradient_metadata_t()
        if (.not. valid_options(options)) then
            if (options%optimizer /= MLP_OPTIMIZER_SGD .or. &
                options%device_kind /= FORTML_DEVICE_CPU) then
                call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                    "MLP SGD momentum hypergradient: optimizer or device is unsupported")
            else
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP SGD momentum hypergradient: options are invalid")
            end if
            return
        end if
        if (.not. valid_data(model, train_x, train_target) .or. &
            .not. valid_data(model, validation_x, validation_target)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP SGD momentum hypergradient: model or data is invalid")
            return
        end if

        self%model => model
        allocate(self%train_x, source=train_x)
        allocate(self%train_target, source=train_target)
        allocate(self%validation_x, source=validation_x)
        allocate(self%validation_target, source=validation_target)
        allocate(self%initial_parameters, source=model%parameters())
        self%layout%inner_steps = options%steps
        self%layout%nesterov = options%nesterov
        self%initial_log_learning_rate = log(options%learning_rate)
        self%initial_log_l2 = log(options%l2)
        self%initial_momentum = options%momentum
        self%initialized = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_sgd_momentum_hypergradient_initialize

    integer function mlp_sgd_momentum_hypergradient_parameter_count(self) result(count)
        class(mlp_sgd_momentum_hypergradient_objective_t), intent(in) :: self

        count = 0
        if (self%initialized) count = self%layout%parameter_count
    end function mlp_sgd_momentum_hypergradient_parameter_count

    function mlp_sgd_momentum_hypergradient_metadata(self) result(layout)
        class(mlp_sgd_momentum_hypergradient_objective_t), intent(in) :: self
        type(mlp_sgd_momentum_hypergradient_metadata_t) :: layout

        layout = self%layout
    end function mlp_sgd_momentum_hypergradient_metadata

    function mlp_sgd_momentum_hypergradient_parameters(self) result(parameters)
        class(mlp_sgd_momentum_hypergradient_objective_t), intent(in) :: self
        real(dp), allocatable :: parameters(:)

        allocate(parameters(MLP_SGD_MOMENTUM_HYPERPARAMETER_COUNT))
        parameters = [self%initial_log_learning_rate, self%initial_log_l2, &
            self%initial_momentum]
    end function mlp_sgd_momentum_hypergradient_parameters

    logical function mlp_sgd_momentum_hypergradient_is_initialized(self) result(yes)
        class(mlp_sgd_momentum_hypergradient_objective_t), intent(in) :: self

        yes = self%initialized .and. associated(self%model) .and. &
            allocated(self%initial_parameters)
    end function mlp_sgd_momentum_hypergradient_is_initialized

    subroutine mlp_sgd_momentum_hypergradient_value_gradient(self, parameters, value, &
            gradient, status)
        class(mlp_sgd_momentum_hypergradient_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: direction(MLP_SGD_MOMENTUM_HYPERPARAMETER_COUNT), tangent

        value = huge(1.0_dp)
        gradient = 0.0_dp
        direction = 0.0_dp
        if (.not. self%is_initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP SGD momentum hypergradient: objective is not initialized")
            return
        end if
        if (.not. valid_parameter_vector(parameters, gradient)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP SGD momentum hypergradient: packed shape is invalid")
            return
        end if
        call sgd_momentum_forward(self, parameters, direction, value, tangent, &
            gradient, status)
    end subroutine mlp_sgd_momentum_hypergradient_value_gradient

    subroutine mlp_sgd_momentum_hypergradient_jvp(self, parameters, direction, value, &
            tangent, status)
        class(mlp_sgd_momentum_hypergradient_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:), direction(:)
        real(dp), intent(out) :: value, tangent
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: gradient(MLP_SGD_MOMENTUM_HYPERPARAMETER_COUNT)

        value = huge(1.0_dp)
        tangent = 0.0_dp
        gradient = 0.0_dp
        if (.not. self%is_initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP SGD momentum hypergradient JVP: objective is not initialized")
            return
        end if
        if (size(parameters) /= MLP_SGD_MOMENTUM_HYPERPARAMETER_COUNT .or. &
            size(direction) /= MLP_SGD_MOMENTUM_HYPERPARAMETER_COUNT .or. &
            any(.not. ieee_is_finite(parameters)) .or. &
            any(.not. ieee_is_finite(direction))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP SGD momentum hypergradient JVP: packed shape is invalid")
            return
        end if
        call sgd_momentum_forward(self, parameters, direction, value, tangent, &
            gradient, status)
    end subroutine mlp_sgd_momentum_hypergradient_jvp

    subroutine mlp_sgd_momentum_hypergradient_vjp(self, parameters, output_bar, &
            gradient, status)
        class(mlp_sgd_momentum_hypergradient_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:), output_bar
        real(dp), intent(out) :: gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: value

        gradient = 0.0_dp
        if (.not. ieee_is_finite(output_bar)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP SGD momentum hypergradient VJP: cotangent is invalid")
            return
        end if
        call self%value_gradient(parameters, value, gradient, status)
        if (status%code /= FORTNUM_OK) return
        gradient = output_bar*gradient
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_sgd_momentum_hypergradient_vjp

    subroutine mlp_sgd_momentum_hypergradient_fortopt(self, objective, status)
        class(mlp_sgd_momentum_hypergradient_objective_t), target, intent(inout) :: self
        type(objective_t), intent(out) :: objective
        type(fortnum_status_t), intent(out) :: status

        if (.not. self%is_initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP SGD momentum hypergradient: objective is not initialized")
            return
        end if
        call objective%initialize_context(MLP_SGD_MOMENTUM_HYPERPARAMETER_COUNT, self, &
            mlp_sgd_momentum_hypergradient_context_callback, status)
    end subroutine mlp_sgd_momentum_hypergradient_fortopt

    subroutine mlp_sgd_momentum_hypergradient_context_callback(context, parameters, &
            value, gradient, status)
        class(*), intent(inout) :: context
        real(dp), intent(in) :: parameters(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status

        select type (adapter => context)
            type is (mlp_sgd_momentum_hypergradient_objective_t)
            call adapter%value_gradient(parameters, value, gradient, status)
        class default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP SGD momentum hypergradient: context has the wrong type")
        end select
    end subroutine mlp_sgd_momentum_hypergradient_context_callback

    subroutine mlp_optimize_sgd_momentum_hyperparameters(model, train_x, train_target, &
            validation_x, validation_target, options, result, status)
        type(mlp_t), target, intent(inout) :: model
        real(dp), intent(in) :: train_x(:, :), train_target(:, :)
        real(dp), intent(in) :: validation_x(:, :), validation_target(:, :)
        type(mlp_sgd_momentum_hypergradient_options_t), intent(in) :: options
        type(mlp_sgd_momentum_hypergradient_result_t), intent(out) :: result
        type(fortnum_status_t), intent(out) :: status
        type(mlp_sgd_momentum_hypergradient_objective_t), target :: adapter
        type(objective_t) :: objective
        type(lbfgsb_t) :: optimizer
        type(lbfgsb_options_t) :: optimizer_options
        type(lbfgsb_result_t) :: optimizer_result
        real(dp) :: parameters(MLP_SGD_MOMENTUM_HYPERPARAMETER_COUNT)
        real(dp) :: lower(MLP_SGD_MOMENTUM_HYPERPARAMETER_COUNT)
        real(dp) :: upper(MLP_SGD_MOMENTUM_HYPERPARAMETER_COUNT)
        real(dp) :: gradient(MLP_SGD_MOMENTUM_HYPERPARAMETER_COUNT)

        result = mlp_sgd_momentum_hypergradient_result_t()
        if (.not. valid_options(options)) then
            if (options%optimizer /= MLP_OPTIMIZER_SGD .or. &
                options%device_kind /= FORTML_DEVICE_CPU) then
                call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                    "MLP SGD momentum hyperparameter optimization: unsupported optimizer/device")
            else
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP SGD momentum hyperparameter optimization: options are invalid")
            end if
            return
        end if
        call adapter%initialize(model, train_x, train_target, validation_x, &
            validation_target, options, status)
        if (status%code /= FORTNUM_OK) return
        parameters = adapter%parameters()
        lower = [options%lower_log_learning_rate, options%lower_log_l2, &
            options%lower_momentum]
        upper = [options%upper_log_learning_rate, options%upper_log_l2, &
            options%upper_momentum]
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
        call adapter%value_gradient(parameters, result%objective, gradient, status)
        if (status%code /= FORTNUM_OK) return
        result%converged = optimizer_result%state%converged
        result%iterations = optimizer_result%state%iteration
        result%line_search_evaluations = optimizer_result%line_search_evaluations
        result%gradient_norm = sqrt(sum(gradient*gradient))
        result%log_learning_rate = parameters(MLP_SGD_LOG_LEARNING_RATE)
        result%log_l2 = parameters(MLP_SGD_LOG_L2)
        result%momentum = parameters(MLP_SGD_MOMENTUM)
        result%learning_rate = exp(result%log_learning_rate)
        result%l2 = exp(result%log_l2)
        if (.not. result%converged) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "MLP SGD momentum hyperparameter optimization: iteration limit reached")
            return
        end if
        if (.not. ieee_is_finite(result%objective) .or. &
            .not. ieee_is_finite(result%gradient_norm)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP SGD momentum hyperparameter optimization: result is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_optimize_sgd_momentum_hyperparameters

    subroutine sgd_momentum_forward(self, parameters, direction, value, tangent, &
            gradient, status)
        class(mlp_sgd_momentum_hypergradient_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:), direction(:)
        real(dp), intent(out) :: value, tangent, gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: theta(:), theta_dot(:, :), velocity(:), velocity_old(:)
        real(dp), allocatable :: velocity_dot(:, :), velocity_dot_new(:, :)
        real(dp), allocatable :: raw_gradient(:), raw_gradient_dot(:), hvp(:)
        real(dp), allocatable :: validation_gradient(:), update(:), update_dot(:)
        real(dp) :: learning_rate, l2, momentum, learning_rate_dot, l2_dot, momentum_dot
        real(dp) :: train_value, l2_gradient, scalar_hvp
        integer :: n_parameters, step, parameter_index

        value = huge(1.0_dp)
        tangent = 0.0_dp
        gradient = 0.0_dp
        if (.not. finite_parameters(parameters, learning_rate, l2, momentum) .or. &
            (self%layout%nesterov .and. momentum <= 0.0_dp)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP SGD momentum hypergradient: packed parameters are invalid")
            return
        end if
        n_parameters = size(self%initial_parameters)
        allocate(theta, source=self%initial_parameters)
        allocate(theta_dot(n_parameters, MLP_SGD_MOMENTUM_HYPERPARAMETER_COUNT))
        allocate(velocity(n_parameters), velocity_old(n_parameters))
        allocate(velocity_dot(n_parameters, MLP_SGD_MOMENTUM_HYPERPARAMETER_COUNT))
        allocate(velocity_dot_new(n_parameters, MLP_SGD_MOMENTUM_HYPERPARAMETER_COUNT))
        allocate(raw_gradient(n_parameters), raw_gradient_dot(n_parameters), hvp(n_parameters))
        allocate(update(n_parameters), update_dot(n_parameters))
        theta_dot = 0.0_dp
        velocity = 0.0_dp
        velocity_dot = 0.0_dp
        do step = 1, self%layout%inner_steps
            call self%model%set_parameters(theta, status)
            if (status%code /= FORTNUM_OK) return
            call mlp_loss_value_gradient(self%model, self%train_x, self%train_target, &
                l2, train_value, raw_gradient, l2_gradient, status)
            if (status%code /= FORTNUM_OK) return
            velocity_old = velocity
            velocity = momentum*velocity_old + raw_gradient
            update = raw_gradient
            if (.not. self%layout%nesterov) update = velocity
            if (self%layout%nesterov) update = raw_gradient + momentum*velocity
            do parameter_index = 1, MLP_SGD_MOMENTUM_HYPERPARAMETER_COUNT
                learning_rate_dot = 0.0_dp
                l2_dot = 0.0_dp
                momentum_dot = 0.0_dp
                if (parameter_index == MLP_SGD_LOG_LEARNING_RATE) then
                    learning_rate_dot = learning_rate
                else if (parameter_index == MLP_SGD_LOG_L2) then
                    l2_dot = l2
                else
                    momentum_dot = 1.0_dp
                end if
                call mlp_loss_hvp(self%model, self%train_x, self%train_target, l2, &
                    theta_dot(:, parameter_index), l2_dot, hvp, scalar_hvp, status)
                if (status%code /= FORTNUM_OK) return
                raw_gradient_dot = hvp
                velocity_dot_new(:, parameter_index) = momentum* &
                    velocity_dot(:, parameter_index) + momentum_dot*velocity_old + &
                    raw_gradient_dot
                if (self%layout%nesterov) then
                    update_dot = raw_gradient_dot + momentum_dot*velocity + &
                        momentum*velocity_dot_new(:, parameter_index)
                else
                    update_dot = velocity_dot_new(:, parameter_index)
                end if
                theta_dot(:, parameter_index) = theta_dot(:, parameter_index) - &
                    learning_rate_dot*update - learning_rate*update_dot
            end do
            velocity_dot = velocity_dot_new
            theta = theta-learning_rate*update
            if (any(.not. ieee_is_finite(theta)) .or. &
                any(.not. ieee_is_finite(theta_dot)) .or. &
                any(.not. ieee_is_finite(velocity_dot))) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP SGD momentum hypergradient: trajectory is not finite")
                return
            end if
        end do
        call self%model%set_parameters(theta, status)
        if (status%code /= FORTNUM_OK) return
        allocate(validation_gradient(n_parameters))
        call mlp_loss_value_gradient(self%model, self%validation_x, self%validation_target, &
            0.0_dp, value, validation_gradient, l2_gradient, status)
        if (status%code /= FORTNUM_OK) return
        do parameter_index = 1, MLP_SGD_MOMENTUM_HYPERPARAMETER_COUNT
            gradient(parameter_index) = dot_product(validation_gradient, &
                theta_dot(:, parameter_index))
        end do
        tangent = dot_product(gradient, direction)
        if (.not. ieee_is_finite(value) .or. any(.not. ieee_is_finite(gradient)) .or. &
            .not. ieee_is_finite(tangent)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP SGD momentum hypergradient: forward product is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine sgd_momentum_forward

    logical function valid_parameter_vector(parameters, gradient) result(valid)
        real(dp), intent(in) :: parameters(:), gradient(:)

        valid = size(parameters) == MLP_SGD_MOMENTUM_HYPERPARAMETER_COUNT .and. &
            size(gradient) == MLP_SGD_MOMENTUM_HYPERPARAMETER_COUNT .and. &
            all(ieee_is_finite(parameters))
    end function valid_parameter_vector

    logical function finite_parameters(parameters, learning_rate, l2, momentum) result(valid)
        real(dp), intent(in) :: parameters(:)
        real(dp), intent(out) :: learning_rate, l2, momentum

        learning_rate = 0.0_dp
        l2 = 0.0_dp
        momentum = 0.0_dp
        valid = size(parameters) == MLP_SGD_MOMENTUM_HYPERPARAMETER_COUNT .and. &
            all(ieee_is_finite(parameters))
        if (.not. valid) return
        learning_rate = exp(parameters(MLP_SGD_LOG_LEARNING_RATE))
        l2 = exp(parameters(MLP_SGD_LOG_L2))
        momentum = parameters(MLP_SGD_MOMENTUM)
        valid = ieee_is_finite(learning_rate) .and. ieee_is_finite(l2) .and. &
            ieee_is_finite(momentum) .and. learning_rate > 0.0_dp .and. &
            l2 > 0.0_dp .and. momentum >= 0.0_dp .and. momentum < 1.0_dp
    end function finite_parameters

    logical function valid_options(options) result(valid)
        type(mlp_sgd_momentum_hypergradient_options_t), intent(in) :: options

        valid = options%steps >= 1 .and. options%optimizer == MLP_OPTIMIZER_SGD .and. &
            options%device_kind == FORTML_DEVICE_CPU .and. &
            ieee_is_finite(options%learning_rate) .and. ieee_is_finite(options%l2) .and. &
            ieee_is_finite(options%momentum) .and. options%learning_rate > 0.0_dp .and. &
            options%l2 > 0.0_dp .and. options%momentum >= 0.0_dp .and. &
            options%momentum < 1.0_dp .and. ieee_is_finite(options%lower_log_learning_rate) .and. &
            ieee_is_finite(options%upper_log_learning_rate) .and. &
            ieee_is_finite(options%lower_log_l2) .and. ieee_is_finite(options%upper_log_l2) .and. &
            ieee_is_finite(options%lower_momentum) .and. ieee_is_finite(options%upper_momentum) .and. &
            options%lower_log_learning_rate <= options%upper_log_learning_rate .and. &
            options%lower_log_l2 <= options%upper_log_l2 .and. &
            options%lower_momentum <= options%upper_momentum .and. &
            options%lower_momentum >= 0.0_dp .and. options%upper_momentum < 1.0_dp .and. &
            log(options%learning_rate) >= options%lower_log_learning_rate .and. &
            log(options%learning_rate) <= options%upper_log_learning_rate .and. &
            log(options%l2) >= options%lower_log_l2 .and. log(options%l2) <= options%upper_log_l2 .and. &
            options%momentum >= options%lower_momentum .and. &
            options%momentum <= options%upper_momentum .and. options%memory >= 1 .and. &
            options%max_iterations >= 1 .and. options%max_line_search >= 1 .and. &
            ieee_is_finite(options%gradient_tolerance) .and. &
            ieee_is_finite(options%step_tolerance) .and. &
            ieee_is_finite(options%objective_tolerance) .and. &
            options%gradient_tolerance >= 0.0_dp .and. options%step_tolerance >= 0.0_dp .and. &
            options%objective_tolerance >= 0.0_dp
        if (options%nesterov) valid = valid .and. options%momentum > 0.0_dp .and. &
            options%lower_momentum > 0.0_dp
    end function valid_options

    logical function valid_data(model, x, target) result(valid)
        class(mlp_t), intent(in) :: model
        real(dp), intent(in) :: x(:, :), target(:, :)

        valid = .false.
        if (.not. allocated(model%layer_sizes)) return
        if (size(model%layer_sizes) < 2) return
        if (model%parameter_count() <= 0) return
        if (size(x, 1) < 1 .or. size(x, 2) /= model%layer_sizes(1)) return
        if (size(target, 1) /= size(x, 1) .or. &
            size(target, 2) /= model%layer_sizes(size(model%layer_sizes))) return
        valid = all(ieee_is_finite(x)) .and. all(ieee_is_finite(target))
    end function valid_data

end module fortml_mlp_sgd_momentum_hypergradient
