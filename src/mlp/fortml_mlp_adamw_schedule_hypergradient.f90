module fortml_mlp_adamw_schedule_hypergradient
    !! Exact CPU products through a fixed full-batch scheduled AdamW trajectory.
    !!
    !! The packed outer coordinates are
    !! [log(base_rate), log(l2), log(weight_decay), logit(beta1),
    !!  logit(beta2), log(epsilon), logit(min_fraction), logit(decay_factor)].
    !! The schedule shape is fixed and its continuous fields are differentiated
    !! analytically.  The state recurrence is the AdamW recurrence used by
    !! FortOpt: moments receive the objective gradient, while weight decay is
    !! applied directly to the parameter state.  No finite-difference training
    !! run is used by value, JVP, or VJP products.
    !!
    !! This is deliberately a CPU reference path.  CUDA, mixed precision, and
    !! outer HVP requests return typed refusals until resident state and third
    !! network derivatives have independent contracts.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, status_ok, &
        FORTNUM_OK, FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED, &
        FORTNUM_CONVERGENCE_ERROR
    use fortml_device, only: FORTML_DEVICE_CPU
    use fortml_mlp, only: mlp_t
    use fortml_mlp_schedules, only: mlp_learning_rate_schedule_t, &
        MLP_SCHEDULE_CONSTANT, MLP_SCHEDULE_COSINE_DECAY, &
        MLP_SCHEDULE_WARMUP_COSINE, MLP_SCHEDULE_EXPONENTIAL_DECAY
    use fortml_mlp_training, only: mlp_loss_value_gradient, mlp_loss_hvp, &
        MLP_OPTIMIZER_ADAMW, MLP_PRECISION_FP64
    use fortopt_objective, only: objective_t
    use fortopt_lbfgsb, only: lbfgsb_t, lbfgsb_options_t, lbfgsb_result_t
    implicit none
    private

    integer, parameter, public :: MLP_ADAMW_SCHEDULE_HYPERPARAMETER_COUNT = 8
    integer, parameter, public :: MLP_ADAMW_SCHEDULE_LOG_BASE_RATE = 1
    integer, parameter, public :: MLP_ADAMW_SCHEDULE_LOG_L2 = 2
    integer, parameter, public :: MLP_ADAMW_SCHEDULE_LOG_WEIGHT_DECAY = 3
    integer, parameter, public :: MLP_ADAMW_SCHEDULE_LOGIT_BETA1 = 4
    integer, parameter, public :: MLP_ADAMW_SCHEDULE_LOGIT_BETA2 = 5
    integer, parameter, public :: MLP_ADAMW_SCHEDULE_LOG_EPSILON = 6
    integer, parameter, public :: MLP_ADAMW_SCHEDULE_LOGIT_MIN_FRACTION = 7
    integer, parameter, public :: MLP_ADAMW_SCHEDULE_LOGIT_DECAY_FACTOR = 8

    type, public :: mlp_adamw_schedule_hypergradient_metadata_t
        integer :: parameter_count = MLP_ADAMW_SCHEDULE_HYPERPARAMETER_COUNT
        integer :: log_base_rate_index = MLP_ADAMW_SCHEDULE_LOG_BASE_RATE
        integer :: log_l2_index = MLP_ADAMW_SCHEDULE_LOG_L2
        integer :: log_weight_decay_index = MLP_ADAMW_SCHEDULE_LOG_WEIGHT_DECAY
        integer :: logit_beta1_index = MLP_ADAMW_SCHEDULE_LOGIT_BETA1
        integer :: logit_beta2_index = MLP_ADAMW_SCHEDULE_LOGIT_BETA2
        integer :: log_epsilon_index = MLP_ADAMW_SCHEDULE_LOG_EPSILON
        integer :: logit_min_fraction_index = MLP_ADAMW_SCHEDULE_LOGIT_MIN_FRACTION
        integer :: logit_decay_factor_index = MLP_ADAMW_SCHEDULE_LOGIT_DECAY_FACTOR
        integer :: inner_steps = 0
        integer :: schedule_kind = MLP_SCHEDULE_CONSTANT
        integer :: warmup_updates = 0
        integer :: total_updates = 0
    end type mlp_adamw_schedule_hypergradient_metadata_t

    type, public :: mlp_adamw_schedule_hypergradient_options_t
        integer :: steps = 8
        type(mlp_learning_rate_schedule_t) :: schedule
        real(dp) :: base_rate = 1.0e-2_dp
        real(dp) :: l2 = 1.0e-4_dp
        real(dp) :: weight_decay = 1.0e-2_dp
        real(dp) :: beta1 = 0.9_dp
        real(dp) :: beta2 = 0.999_dp
        real(dp) :: epsilon = 1.0e-8_dp
        real(dp) :: lower_log_base_rate = -12.0_dp
        real(dp) :: upper_log_base_rate = 2.0_dp
        real(dp) :: lower_log_l2 = -20.0_dp
        real(dp) :: upper_log_l2 = 2.0_dp
        real(dp) :: lower_log_weight_decay = -20.0_dp
        real(dp) :: upper_log_weight_decay = 2.0_dp
        real(dp) :: lower_logit_beta1 = -12.0_dp
        real(dp) :: upper_logit_beta1 = 12.0_dp
        real(dp) :: lower_logit_beta2 = -12.0_dp
        real(dp) :: upper_logit_beta2 = 12.0_dp
        real(dp) :: lower_log_epsilon = -30.0_dp
        real(dp) :: upper_log_epsilon = 2.0_dp
        real(dp) :: lower_logit_min_fraction = -12.0_dp
        real(dp) :: upper_logit_min_fraction = 12.0_dp
        real(dp) :: lower_logit_decay_factor = -12.0_dp
        real(dp) :: upper_logit_decay_factor = 12.0_dp
        integer :: optimizer = MLP_OPTIMIZER_ADAMW
        integer :: precision_kind = MLP_PRECISION_FP64
        integer :: device_kind = FORTML_DEVICE_CPU
        integer :: memory = 8
        integer :: max_iterations = 100
        integer :: max_line_search = 40
        real(dp) :: gradient_tolerance = 1.0e-6_dp
        real(dp) :: step_tolerance = 1.0e-12_dp
        real(dp) :: objective_tolerance = 1.0e-12_dp
    end type mlp_adamw_schedule_hypergradient_options_t

    type, public :: mlp_adamw_schedule_hypergradient_result_t
        logical :: converged = .false.
        integer :: iterations = 0
        integer :: line_search_evaluations = 0
        real(dp) :: objective = huge(1.0_dp)
        real(dp) :: gradient_norm = huge(1.0_dp)
        real(dp) :: log_base_rate = 0.0_dp
        real(dp) :: log_l2 = 0.0_dp
        real(dp) :: log_weight_decay = 0.0_dp
        real(dp) :: logit_beta1 = 0.0_dp
        real(dp) :: logit_beta2 = 0.0_dp
        real(dp) :: log_epsilon = 0.0_dp
        real(dp) :: logit_min_fraction = 0.0_dp
        real(dp) :: logit_decay_factor = 0.0_dp
        real(dp) :: base_rate = 0.0_dp
        real(dp) :: l2 = 0.0_dp
        real(dp) :: weight_decay = 0.0_dp
        real(dp) :: beta1 = 0.0_dp
        real(dp) :: beta2 = 0.0_dp
        real(dp) :: epsilon = 0.0_dp
        real(dp) :: min_rate_fraction = 0.0_dp
        real(dp) :: decay_factor = 0.0_dp
    end type mlp_adamw_schedule_hypergradient_result_t

    type, public :: mlp_adamw_schedule_hypergradient_objective_t
        private
        type(mlp_t), pointer :: model => null()
        real(dp), allocatable :: train_x(:, :), train_target(:, :)
        real(dp), allocatable :: validation_x(:, :), validation_target(:, :)
        real(dp), allocatable :: initial_parameters(:)
        type(mlp_learning_rate_schedule_t) :: schedule
        type(mlp_adamw_schedule_hypergradient_metadata_t) :: layout
        real(dp) :: initial_log_base_rate = 0.0_dp
        real(dp) :: initial_log_l2 = 0.0_dp
        real(dp) :: initial_log_weight_decay = 0.0_dp
        real(dp) :: initial_logit_beta1 = 0.0_dp
        real(dp) :: initial_logit_beta2 = 0.0_dp
        real(dp) :: initial_log_epsilon = 0.0_dp
        real(dp) :: initial_logit_min_fraction = 0.0_dp
        real(dp) :: initial_logit_decay_factor = 0.0_dp
        logical :: initialized = .false.
    contains
        procedure, public :: initialize => adamw_schedule_initialize
        procedure, public :: parameter_count => adamw_schedule_parameter_count
        procedure, public :: metadata => adamw_schedule_metadata
        procedure, public :: parameters => adamw_schedule_parameters
        procedure, public :: value_gradient => adamw_schedule_value_gradient
        procedure, public :: jvp => adamw_schedule_jvp
        procedure, public :: vjp => adamw_schedule_vjp
        procedure, public :: hvp => adamw_schedule_hvp
        procedure, public :: fortopt => adamw_schedule_fortopt
        procedure, public :: is_initialized => adamw_schedule_is_initialized
    end type mlp_adamw_schedule_hypergradient_objective_t

    public :: mlp_optimize_adamw_schedule_hyperparameters

contains

    subroutine adamw_schedule_initialize(self, model, train_x, train_target, &
            validation_x, validation_target, options, status)
        class(mlp_adamw_schedule_hypergradient_objective_t), intent(out) :: self
        type(mlp_t), target, intent(inout) :: model
        real(dp), intent(in) :: train_x(:, :), train_target(:, :)
        real(dp), intent(in) :: validation_x(:, :), validation_target(:, :)
        type(mlp_adamw_schedule_hypergradient_options_t), intent(in) :: options
        type(fortnum_status_t), intent(out) :: status
        type(mlp_adamw_schedule_hypergradient_metadata_t) :: default_layout

        self%initialized = .false.
        self%layout = default_layout
        if (.not. valid_options(options)) then
            if (options%optimizer /= MLP_OPTIMIZER_ADAMW .or. &
                options%precision_kind /= MLP_PRECISION_FP64 .or. &
                options%device_kind /= FORTML_DEVICE_CPU) then
                call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                    "MLP scheduled AdamW hypergradient: optimizer or device is unsupported")
            else
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP scheduled AdamW hypergradient: options are invalid")
            end if
            return
        end if
        if (.not. valid_data(model, train_x, train_target) .or. &
            .not. valid_data(model, validation_x, validation_target)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP scheduled AdamW hypergradient: model or data is invalid")
            return
        end if
        self%model => model
        allocate(self%train_x, source=train_x)
        allocate(self%train_target, source=train_target)
        allocate(self%validation_x, source=validation_x)
        allocate(self%validation_target, source=validation_target)
        allocate(self%initial_parameters, source=model%parameters())
        self%schedule = options%schedule
        self%layout%inner_steps = options%steps
        self%layout%schedule_kind = options%schedule%kind
        self%layout%warmup_updates = options%schedule%warmup_updates
        self%layout%total_updates = options%schedule%total_updates
        self%initial_log_base_rate = log(options%base_rate)
        self%initial_log_l2 = log(options%l2)
        self%initial_log_weight_decay = log(options%weight_decay)
        self%initial_logit_beta1 = probability_logit(options%beta1)
        self%initial_logit_beta2 = probability_logit(options%beta2)
        self%initial_log_epsilon = log(options%epsilon)
        self%initial_logit_min_fraction = probability_logit( &
            interior_probability(options%schedule%min_rate_fraction))
        self%initial_logit_decay_factor = probability_logit( &
            interior_probability(options%schedule%decay_factor))
        self%initialized = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine adamw_schedule_initialize

    integer function adamw_schedule_parameter_count(self) result(count)
        class(mlp_adamw_schedule_hypergradient_objective_t), intent(in) :: self

        count = 0
        if (self%initialized) count = self%layout%parameter_count
    end function adamw_schedule_parameter_count

    function adamw_schedule_metadata(self) result(layout)
        class(mlp_adamw_schedule_hypergradient_objective_t), intent(in) :: self
        type(mlp_adamw_schedule_hypergradient_metadata_t) :: layout

        layout = self%layout
    end function adamw_schedule_metadata

    function adamw_schedule_parameters(self) result(parameters)
        class(mlp_adamw_schedule_hypergradient_objective_t), intent(in) :: self
        real(dp), allocatable :: parameters(:)

        allocate(parameters(MLP_ADAMW_SCHEDULE_HYPERPARAMETER_COUNT))
        parameters = [self%initial_log_base_rate, self%initial_log_l2, &
            self%initial_log_weight_decay, self%initial_logit_beta1, &
            self%initial_logit_beta2, self%initial_log_epsilon, &
            self%initial_logit_min_fraction, self%initial_logit_decay_factor]
    end function adamw_schedule_parameters

    logical function adamw_schedule_is_initialized(self) result(yes)
        class(mlp_adamw_schedule_hypergradient_objective_t), intent(in) :: self

        yes = self%initialized .and. associated(self%model) .and. &
            allocated(self%initial_parameters)
    end function adamw_schedule_is_initialized

    subroutine adamw_schedule_value_gradient(self, parameters, value, gradient, status)
        class(mlp_adamw_schedule_hypergradient_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: direction(MLP_ADAMW_SCHEDULE_HYPERPARAMETER_COUNT), tangent

        value = huge(1.0_dp)
        gradient = 0.0_dp
        direction = 0.0_dp
        if (.not. self%is_initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP scheduled AdamW hypergradient: objective is not initialized")
            return
        end if
        if (size(parameters) /= MLP_ADAMW_SCHEDULE_HYPERPARAMETER_COUNT .or. &
            size(gradient) /= MLP_ADAMW_SCHEDULE_HYPERPARAMETER_COUNT .or. &
            any(.not. ieee_is_finite(parameters))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP scheduled AdamW hypergradient: packed shape is invalid")
            return
        end if
        call adamw_schedule_forward(self, parameters, direction, value, tangent, &
            gradient, status)
    end subroutine adamw_schedule_value_gradient

    subroutine adamw_schedule_jvp(self, parameters, direction, value, tangent, status)
        class(mlp_adamw_schedule_hypergradient_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:), direction(:)
        real(dp), intent(out) :: value, tangent
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: gradient(MLP_ADAMW_SCHEDULE_HYPERPARAMETER_COUNT)

        value = huge(1.0_dp)
        tangent = 0.0_dp
        gradient = 0.0_dp
        if (.not. self%is_initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP scheduled AdamW hypergradient JVP: objective is not initialized")
            return
        end if
        if (size(parameters) /= MLP_ADAMW_SCHEDULE_HYPERPARAMETER_COUNT .or. &
            size(direction) /= MLP_ADAMW_SCHEDULE_HYPERPARAMETER_COUNT .or. &
            any(.not. ieee_is_finite(parameters)) .or. &
            any(.not. ieee_is_finite(direction))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP scheduled AdamW hypergradient JVP: packed shape is invalid")
            return
        end if
        call adamw_schedule_forward(self, parameters, direction, value, tangent, &
            gradient, status)
    end subroutine adamw_schedule_jvp

    subroutine adamw_schedule_vjp(self, parameters, output_bar, gradient, status)
        class(mlp_adamw_schedule_hypergradient_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:), output_bar
        real(dp), intent(out) :: gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: value

        gradient = 0.0_dp
        if (.not. ieee_is_finite(output_bar)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP scheduled AdamW hypergradient VJP: cotangent is invalid")
            return
        end if
        call self%value_gradient(parameters, value, gradient, status)
        if (.not. status_ok(status)) return
        gradient = output_bar*gradient
        call status_set(status, FORTNUM_OK, "")
    end subroutine adamw_schedule_vjp

    subroutine adamw_schedule_hvp(self, parameters, direction, product, status)
        class(mlp_adamw_schedule_hypergradient_objective_t), intent(in) :: self
        real(dp), intent(in) :: parameters(:), direction(:)
        real(dp), intent(out) :: product(:)
        type(fortnum_status_t), intent(out) :: status

        product = 0.0_dp
        if (size(parameters) /= MLP_ADAMW_SCHEDULE_HYPERPARAMETER_COUNT .or. &
            size(direction) /= MLP_ADAMW_SCHEDULE_HYPERPARAMETER_COUNT .or. &
            size(product) /= MLP_ADAMW_SCHEDULE_HYPERPARAMETER_COUNT .or. &
            any(.not. ieee_is_finite(parameters)) .or. &
            any(.not. ieee_is_finite(direction))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP scheduled AdamW hypergradient HVP: packed shape is invalid")
            return
        end if
        if (.not. self%is_initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP scheduled AdamW hypergradient HVP: objective is not initialized")
            return
        end if
        call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
            "MLP scheduled AdamW hypergradient HVP requires third network derivatives")
    end subroutine adamw_schedule_hvp

    subroutine adamw_schedule_fortopt(self, objective, status)
        class(mlp_adamw_schedule_hypergradient_objective_t), target, intent(inout) :: self
        type(objective_t), intent(out) :: objective
        type(fortnum_status_t), intent(out) :: status

        if (.not. self%is_initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP scheduled AdamW hypergradient: objective is not initialized")
            return
        end if
        call objective%initialize_context(MLP_ADAMW_SCHEDULE_HYPERPARAMETER_COUNT, self, &
            adamw_schedule_context_callback, status)
    end subroutine adamw_schedule_fortopt

    subroutine adamw_schedule_context_callback(context, parameters, value, gradient, status)
        class(*), intent(inout) :: context
        real(dp), intent(in) :: parameters(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status

        select type (adapter => context)
            type is (mlp_adamw_schedule_hypergradient_objective_t)
            call adapter%value_gradient(parameters, value, gradient, status)
        class default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP scheduled AdamW hypergradient: context has the wrong type")
        end select
    end subroutine adamw_schedule_context_callback

    subroutine mlp_optimize_adamw_schedule_hyperparameters(model, train_x, train_target, &
            validation_x, validation_target, options, result, status)
        type(mlp_t), target, intent(inout) :: model
        real(dp), intent(in) :: train_x(:, :), train_target(:, :)
        real(dp), intent(in) :: validation_x(:, :), validation_target(:, :)
        type(mlp_adamw_schedule_hypergradient_options_t), intent(in) :: options
        type(mlp_adamw_schedule_hypergradient_result_t), intent(out) :: result
        type(fortnum_status_t), intent(out) :: status
        type(mlp_adamw_schedule_hypergradient_objective_t), target :: adapter
        type(objective_t) :: objective
        type(lbfgsb_t) :: optimizer
        type(lbfgsb_options_t) :: optimizer_options
        type(lbfgsb_result_t) :: optimizer_result
        real(dp) :: parameters(MLP_ADAMW_SCHEDULE_HYPERPARAMETER_COUNT)
        real(dp) :: lower(MLP_ADAMW_SCHEDULE_HYPERPARAMETER_COUNT)
        real(dp) :: upper(MLP_ADAMW_SCHEDULE_HYPERPARAMETER_COUNT)
        real(dp) :: gradient(MLP_ADAMW_SCHEDULE_HYPERPARAMETER_COUNT)
        type(mlp_adamw_schedule_hypergradient_result_t) :: default_result

        result = default_result
        if (.not. valid_options(options)) then
            if (options%optimizer /= MLP_OPTIMIZER_ADAMW .or. &
                options%precision_kind /= MLP_PRECISION_FP64 .or. &
                options%device_kind /= FORTML_DEVICE_CPU) then
                call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                    "MLP scheduled AdamW optimization: optimizer or device is unsupported")
            else
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP scheduled AdamW optimization: options are invalid")
            end if
            return
        end if
        call adapter%initialize(model, train_x, train_target, validation_x, &
            validation_target, options, status)
        if (.not. status_ok(status)) return
        parameters = adapter%parameters()
        lower = [options%lower_log_base_rate, options%lower_log_l2, &
            options%lower_log_weight_decay, options%lower_logit_beta1, &
            options%lower_logit_beta2, options%lower_log_epsilon, &
            options%lower_logit_min_fraction, options%lower_logit_decay_factor]
        upper = [options%upper_log_base_rate, options%upper_log_l2, &
            options%upper_log_weight_decay, options%upper_logit_beta1, &
            options%upper_logit_beta2, options%upper_log_epsilon, &
            options%upper_logit_min_fraction, options%upper_logit_decay_factor]
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
        result%log_base_rate = parameters(MLP_ADAMW_SCHEDULE_LOG_BASE_RATE)
        result%log_l2 = parameters(MLP_ADAMW_SCHEDULE_LOG_L2)
        result%log_weight_decay = parameters(MLP_ADAMW_SCHEDULE_LOG_WEIGHT_DECAY)
        result%logit_beta1 = parameters(MLP_ADAMW_SCHEDULE_LOGIT_BETA1)
        result%logit_beta2 = parameters(MLP_ADAMW_SCHEDULE_LOGIT_BETA2)
        result%log_epsilon = parameters(MLP_ADAMW_SCHEDULE_LOG_EPSILON)
        result%logit_min_fraction = parameters(MLP_ADAMW_SCHEDULE_LOGIT_MIN_FRACTION)
        result%logit_decay_factor = parameters(MLP_ADAMW_SCHEDULE_LOGIT_DECAY_FACTOR)
        result%base_rate = exp(result%log_base_rate)
        result%l2 = exp(result%log_l2)
        result%weight_decay = exp(result%log_weight_decay)
        result%beta1 = sigmoid(result%logit_beta1)
        result%beta2 = sigmoid(result%logit_beta2)
        result%epsilon = exp(result%log_epsilon)
        result%min_rate_fraction = sigmoid(result%logit_min_fraction)
        result%decay_factor = sigmoid(result%logit_decay_factor)
        if (.not. result%converged) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "MLP scheduled AdamW optimization: iteration limit reached")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_optimize_adamw_schedule_hyperparameters

    subroutine adamw_schedule_forward(self, parameters, direction, value, tangent, &
            gradient, status)
        class(mlp_adamw_schedule_hypergradient_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:), direction(:)
        real(dp), intent(out) :: value, tangent, gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: base_rate, l2, weight_decay, beta1, beta2, epsilon
        real(dp) :: min_fraction, decay_factor, rate, dbase, dmin, ddecay
        real(dp) :: base_dot, l2_dot, weight_decay_dot, beta1_dot, beta2_dot
        real(dp) :: epsilon_dot, min_dot, decay_dot, rate_dot, decay_multiplier
        real(dp) :: decay_multiplier_dot, c1, c2, c1_dot, c2_dot
        real(dp) :: train_value, l2_gradient, validation_l2_gradient
        real(dp), allocatable :: theta(:), theta_dot(:, :), first(:), second(:)
        real(dp), allocatable :: first_previous(:), second_previous(:)
        real(dp), allocatable :: first_dot(:, :), second_dot(:, :)
        real(dp), allocatable :: raw_gradient(:), gradient_dot(:), hvp(:)
        real(dp), allocatable :: first_hat(:), second_hat(:), first_hat_dot(:), second_hat_dot(:)
        real(dp), allocatable :: square_root(:), denominator(:), denominator_dot(:)
        real(dp), allocatable :: update(:), update_dot(:), validation_gradient(:)
        integer :: n_parameters, step, index

        value = huge(1.0_dp)
        tangent = 0.0_dp
        gradient = 0.0_dp
        if (.not. unpack(parameters, base_rate, l2, weight_decay, beta1, beta2, &
            epsilon, min_fraction, decay_factor)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP scheduled AdamW hypergradient: transformed parameters are invalid")
            return
        end if
        n_parameters = size(self%initial_parameters)
        allocate(theta, source=self%initial_parameters)
        allocate(theta_dot(n_parameters, MLP_ADAMW_SCHEDULE_HYPERPARAMETER_COUNT))
        allocate(first(n_parameters), second(n_parameters), first_previous(n_parameters), &
            second_previous(n_parameters))
        allocate(first_dot(n_parameters, MLP_ADAMW_SCHEDULE_HYPERPARAMETER_COUNT), &
            second_dot(n_parameters, MLP_ADAMW_SCHEDULE_HYPERPARAMETER_COUNT))
        allocate(raw_gradient(n_parameters), gradient_dot(n_parameters), hvp(n_parameters))
        allocate(first_hat(n_parameters), second_hat(n_parameters), first_hat_dot(n_parameters), &
            second_hat_dot(n_parameters), square_root(n_parameters), denominator(n_parameters), &
            denominator_dot(n_parameters), update(n_parameters), update_dot(n_parameters), &
            validation_gradient(n_parameters))
        theta_dot = 0.0_dp
        first = 0.0_dp
        second = 0.0_dp
        first_dot = 0.0_dp
        second_dot = 0.0_dp
        do step = 1, self%layout%inner_steps
            call self%model%set_parameters(theta, status)
            if (.not. status_ok(status)) return
            call mlp_loss_value_gradient(self%model, self%train_x, self%train_target, &
                l2, train_value, raw_gradient, l2_gradient, status)
            if (.not. status_ok(status)) return
            call configured_rate(self%schedule, step, base_rate, min_fraction, decay_factor, &
                rate, dbase, dmin, ddecay, status)
            if (.not. status_ok(status)) return
            first_previous = first
            second_previous = second
            first = beta1*first_previous + (1.0_dp-beta1)*raw_gradient
            second = beta2*second_previous + (1.0_dp-beta2)*raw_gradient*raw_gradient
            do index = 1, MLP_ADAMW_SCHEDULE_HYPERPARAMETER_COUNT
                l2_dot = 0.0_dp
                beta1_dot = 0.0_dp
                beta2_dot = 0.0_dp
                if (index == MLP_ADAMW_SCHEDULE_LOG_L2) l2_dot = l2
                if (index == MLP_ADAMW_SCHEDULE_LOGIT_BETA1) beta1_dot = beta1*(1.0_dp-beta1)
                if (index == MLP_ADAMW_SCHEDULE_LOGIT_BETA2) beta2_dot = beta2*(1.0_dp-beta2)
                call mlp_loss_hvp(self%model, self%train_x, self%train_target, l2, &
                    theta_dot(:, index), l2_dot, hvp, l2_gradient, status)
                if (.not. status_ok(status)) return
                gradient_dot = hvp
                first_dot(:, index) = beta1*first_dot(:, index) + &
                    (1.0_dp-beta1)*gradient_dot + beta1_dot*(first_previous-raw_gradient)
                second_dot(:, index) = beta2*second_dot(:, index) + &
                    (1.0_dp-beta2)*2.0_dp*raw_gradient*gradient_dot + &
                    beta2_dot*(second_previous-raw_gradient*raw_gradient)
            end do
            c1 = 1.0_dp-beta1**step
            c2 = 1.0_dp-beta2**step
            first_hat = first/c1
            second_hat = second/c2
            if (any(second_hat < 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP scheduled AdamW hypergradient: second moment is negative")
                return
            end if
            square_root = sqrt(second_hat)
            if (any(square_root == 0.0_dp)) then
                call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                    "MLP scheduled AdamW hypergradient: zero second-moment derivative")
                return
            end if
            denominator = square_root + epsilon
            update = first_hat/denominator
            do index = 1, MLP_ADAMW_SCHEDULE_HYPERPARAMETER_COUNT
                base_dot = 0.0_dp
                l2_dot = 0.0_dp
                weight_decay_dot = 0.0_dp
                beta1_dot = 0.0_dp
                beta2_dot = 0.0_dp
                epsilon_dot = 0.0_dp
                min_dot = 0.0_dp
                decay_dot = 0.0_dp
                if (index == MLP_ADAMW_SCHEDULE_LOG_BASE_RATE) base_dot = base_rate
                if (index == MLP_ADAMW_SCHEDULE_LOG_L2) l2_dot = l2
                if (index == MLP_ADAMW_SCHEDULE_LOG_WEIGHT_DECAY) then
                    weight_decay_dot = weight_decay
                end if
                if (index == MLP_ADAMW_SCHEDULE_LOGIT_BETA1) then
                    beta1_dot = beta1*(1.0_dp-beta1)
                end if
                if (index == MLP_ADAMW_SCHEDULE_LOGIT_BETA2) then
                    beta2_dot = beta2*(1.0_dp-beta2)
                end if
                if (index == MLP_ADAMW_SCHEDULE_LOG_EPSILON) epsilon_dot = epsilon
                if (index == MLP_ADAMW_SCHEDULE_LOGIT_MIN_FRACTION) then
                    min_dot = min_fraction*(1.0_dp-min_fraction)
                end if
                if (index == MLP_ADAMW_SCHEDULE_LOGIT_DECAY_FACTOR) then
                    decay_dot = decay_factor*(1.0_dp-decay_factor)
                end if
                c1_dot = -real(step, dp)*beta1**(step-1)*beta1_dot
                c2_dot = -real(step, dp)*beta2**(step-1)*beta2_dot
                first_hat_dot = first_dot(:, index)/c1 - first*c1_dot/(c1*c1)
                second_hat_dot = second_dot(:, index)/c2 - second*c2_dot/(c2*c2)
                denominator_dot = second_hat_dot/(2.0_dp*square_root) + epsilon_dot
                update_dot = first_hat_dot/denominator - &
                    first_hat*denominator_dot/(denominator*denominator)
                rate_dot = dbase*base_dot + dmin*min_dot + ddecay*decay_dot
                decay_multiplier = 1.0_dp-rate*weight_decay
                decay_multiplier_dot = -(rate_dot*weight_decay + rate*weight_decay_dot)
                theta_dot(:, index) = decay_multiplier*theta_dot(:, index) + &
                    decay_multiplier_dot*theta - rate_dot*update - rate*update_dot
            end do
            decay_multiplier = 1.0_dp-rate*weight_decay
            theta = decay_multiplier*theta-rate*update
            if (any(.not. ieee_is_finite(theta)) .or. any(.not. ieee_is_finite(theta_dot)) .or. &
                any(.not. ieee_is_finite(first)) .or. any(.not. ieee_is_finite(second))) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP scheduled AdamW hypergradient: trajectory is not finite")
                return
            end if
        end do
        call self%model%set_parameters(theta, status)
        if (.not. status_ok(status)) return
        call mlp_loss_value_gradient(self%model, self%validation_x, self%validation_target, &
            0.0_dp, value, validation_gradient, l2_gradient, status)
        if (.not. status_ok(status)) return
        do index = 1, MLP_ADAMW_SCHEDULE_HYPERPARAMETER_COUNT
            gradient(index) = dot_product(validation_gradient, theta_dot(:, index))
        end do
        tangent = dot_product(gradient, direction)
        if (.not. ieee_is_finite(value) .or. any(.not. ieee_is_finite(gradient)) .or. &
            .not. ieee_is_finite(tangent)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP scheduled AdamW hypergradient: product is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine adamw_schedule_forward

    subroutine configured_rate(base_schedule, update, base_rate, min_fraction, decay_factor, &
            rate, dbase, dmin, ddecay, status)
        type(mlp_learning_rate_schedule_t), intent(in) :: base_schedule
        integer, intent(in) :: update
        real(dp), intent(in) :: base_rate, min_fraction, decay_factor
        real(dp), intent(out) :: rate, dbase, dmin, ddecay
        type(fortnum_status_t), intent(out) :: status
        type(mlp_learning_rate_schedule_t) :: schedule
        real(dp) :: ignored_peak, ignored_final

        schedule = base_schedule
        schedule%min_rate_fraction = min_fraction
        schedule%decay_factor = decay_factor
        call schedule%rate_with_full_derivatives(update, base_rate, rate, dbase, dmin, &
            ddecay, ignored_peak, ignored_final, status)
    end subroutine configured_rate

    logical function unpack(parameters, base_rate, l2, weight_decay, beta1, beta2, &
            epsilon, min_fraction, decay_factor) result(valid)
        real(dp), intent(in) :: parameters(:)
        real(dp), intent(out) :: base_rate, l2, weight_decay, beta1, beta2, epsilon
        real(dp), intent(out) :: min_fraction, decay_factor

        base_rate = 0.0_dp
        l2 = 0.0_dp
        weight_decay = 0.0_dp
        beta1 = 0.0_dp
        beta2 = 0.0_dp
        epsilon = 0.0_dp
        min_fraction = 0.0_dp
        decay_factor = 0.0_dp
        valid = size(parameters) == MLP_ADAMW_SCHEDULE_HYPERPARAMETER_COUNT
        if (.not. valid) return
        if (any(.not. ieee_is_finite(parameters))) then
            valid = .false.
            return
        end if
        base_rate = exp(parameters(MLP_ADAMW_SCHEDULE_LOG_BASE_RATE))
        l2 = exp(parameters(MLP_ADAMW_SCHEDULE_LOG_L2))
        weight_decay = exp(parameters(MLP_ADAMW_SCHEDULE_LOG_WEIGHT_DECAY))
        beta1 = sigmoid(parameters(MLP_ADAMW_SCHEDULE_LOGIT_BETA1))
        beta2 = sigmoid(parameters(MLP_ADAMW_SCHEDULE_LOGIT_BETA2))
        epsilon = exp(parameters(MLP_ADAMW_SCHEDULE_LOG_EPSILON))
        min_fraction = sigmoid(parameters(MLP_ADAMW_SCHEDULE_LOGIT_MIN_FRACTION))
        decay_factor = sigmoid(parameters(MLP_ADAMW_SCHEDULE_LOGIT_DECAY_FACTOR))
        valid = ieee_is_finite(base_rate) .and. ieee_is_finite(l2) .and. &
            ieee_is_finite(weight_decay) .and. ieee_is_finite(beta1) .and. &
            ieee_is_finite(beta2) .and. ieee_is_finite(epsilon) .and. &
            ieee_is_finite(min_fraction) .and. ieee_is_finite(decay_factor) .and. &
            base_rate > 0.0_dp .and. l2 > 0.0_dp .and. weight_decay > 0.0_dp .and. &
            beta1 > 0.0_dp .and. beta1 < 1.0_dp .and. beta2 > 0.0_dp .and. beta2 < 1.0_dp .and. &
            epsilon > 0.0_dp .and. min_fraction >= 0.0_dp .and. min_fraction <= 1.0_dp .and. &
            decay_factor > 0.0_dp .and. decay_factor < 1.0_dp
    end function unpack

    logical function valid_options(options) result(valid)
        type(mlp_adamw_schedule_hypergradient_options_t), intent(in) :: options

        valid = options%steps >= 1 .and. options%optimizer == MLP_OPTIMIZER_ADAMW .and. &
            options%precision_kind == MLP_PRECISION_FP64 .and. &
            options%device_kind == FORTML_DEVICE_CPU .and. options%base_rate > 0.0_dp .and. &
            options%l2 > 0.0_dp .and. options%weight_decay > 0.0_dp .and. &
            options%beta1 > 0.0_dp .and. options%beta1 < 1.0_dp .and. &
            options%beta2 > 0.0_dp .and. options%beta2 < 1.0_dp .and. options%epsilon > 0.0_dp
        if (.not. valid) return
        valid = ieee_is_finite(options%base_rate) .and. ieee_is_finite(options%l2) .and. &
            ieee_is_finite(options%weight_decay) .and. ieee_is_finite(options%beta1) .and. &
            ieee_is_finite(options%beta2) .and. ieee_is_finite(options%epsilon) .and. &
            options%schedule%valid()
        if (.not. valid) return
        select case (options%schedule%kind)
        case (MLP_SCHEDULE_CONSTANT, MLP_SCHEDULE_COSINE_DECAY, &
                MLP_SCHEDULE_WARMUP_COSINE, MLP_SCHEDULE_EXPONENTIAL_DECAY)
            continue
        case default
            valid = .false.
            return
        end select
        valid = bounds_valid(options%lower_log_base_rate, options%upper_log_base_rate, &
            log(options%base_rate)) .and. bounds_valid(options%lower_log_l2, &
            options%upper_log_l2, log(options%l2)) .and. bounds_valid( &
            options%lower_log_weight_decay, options%upper_log_weight_decay, &
            log(options%weight_decay))
        valid = valid .and. bounds_valid(options%lower_logit_beta1, options%upper_logit_beta1, &
            probability_logit(options%beta1)) .and. bounds_valid(options%lower_logit_beta2, &
            options%upper_logit_beta2, probability_logit(options%beta2)) .and. bounds_valid( &
            options%lower_log_epsilon, options%upper_log_epsilon, log(options%epsilon))
        valid = valid .and. bounds_valid(options%lower_logit_min_fraction, &
            options%upper_logit_min_fraction, probability_logit(interior_probability( &
            options%schedule%min_rate_fraction))) .and. bounds_valid( &
            options%lower_logit_decay_factor, options%upper_logit_decay_factor, &
            probability_logit(interior_probability(options%schedule%decay_factor)))
        valid = valid .and. options%memory >= 1 .and. options%max_iterations >= 1 .and. &
            options%max_line_search >= 1 .and. ieee_is_finite(options%gradient_tolerance) .and. &
            ieee_is_finite(options%step_tolerance) .and. &
            ieee_is_finite(options%objective_tolerance) .and. &
            options%gradient_tolerance >= 0.0_dp .and. options%step_tolerance >= 0.0_dp .and. &
            options%objective_tolerance >= 0.0_dp
    end function valid_options

    logical function bounds_valid(lower, upper, value) result(valid)
        real(dp), intent(in) :: lower, upper, value

        valid = ieee_is_finite(lower) .and. ieee_is_finite(upper) .and. &
            ieee_is_finite(value) .and. &
            lower <= upper .and. value >= lower .and. value <= upper
    end function bounds_valid

    logical function valid_data(model, x, target) result(valid)
        type(mlp_t), intent(in) :: model
        real(dp), intent(in) :: x(:, :), target(:, :)

        valid = .false.
        if (.not. allocated(model%layer_sizes)) return
        if (size(model%layer_sizes) < 2 .or. model%parameter_count() <= 0) return
        if (size(x, 1) < 1 .or. size(target, 1) /= size(x, 1)) return
        if (size(x, 2) /= model%layer_sizes(1)) return
        if (size(target, 2) /= model%layer_sizes(size(model%layer_sizes))) return
        if (any(.not. ieee_is_finite(x)) .or. any(.not. ieee_is_finite(target))) return
        valid = .true.
    end function valid_data

    pure real(dp) function sigmoid(value) result(output)
        real(dp), intent(in) :: value

        if (value >= 0.0_dp) then
            output = 1.0_dp/(1.0_dp+exp(-value))
        else
            output = exp(value)/(1.0_dp+exp(value))
        end if
    end function sigmoid

    pure real(dp) function probability_logit(value) result(output)
        real(dp), intent(in) :: value

        output = log(value/(1.0_dp-value))
    end function probability_logit

    pure real(dp) function interior_probability(value) result(output)
        real(dp), intent(in) :: value

        output = min(1.0_dp-1.0e-5_dp, max(1.0e-5_dp, value))
    end function interior_probability

end module fortml_mlp_adamw_schedule_hypergradient
