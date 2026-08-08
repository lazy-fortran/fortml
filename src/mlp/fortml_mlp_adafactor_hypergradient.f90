module fortml_mlp_adafactor_hypergradient
    !! Exact fixed full-batch hypergradients through vector Adafactor.
    !!
    !! The supported branch matches `fortml_adafactor` with an unfactored
    !! second-moment vector, fixed full-batch updates, and both relative-step
    !! and parameter-scaling modes disabled.  The packed outer variable is
    !! `[log(learning_rate), log(l2), decay, log(epsilon),
    !!   log(clip_threshold)]`.
    !!
    !! The clip and square-root transitions are differentiated piecewise.  A
    !! trajectory that lands on a clip boundary is refused with a typed status
    !! instead of silently differentiating an active-set change.  Relative-step
    !! and parameter-scaling branches have their own discrete state and are
    !! refused until their state derivatives receive a separate contract.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED, FORTNUM_CONVERGENCE_ERROR
    use fortml_device, only: FORTML_DEVICE_CPU
    use fortml_mlp, only: mlp_t
    use fortml_mlp_training, only: mlp_loss_value_gradient, mlp_loss_hvp, &
        MLP_OPTIMIZER_ADAFACTOR
    use fortopt_objective, only: objective_t
    use fortopt_lbfgsb, only: lbfgsb_t, lbfgsb_options_t, lbfgsb_result_t
    implicit none
    private

    integer, parameter, public :: MLP_ADAFACTOR_HYPERPARAMETER_COUNT = 5
    integer, parameter, public :: MLP_ADAFACTOR_LOG_LEARNING_RATE = 1
    integer, parameter, public :: MLP_ADAFACTOR_LOG_L2 = 2
    integer, parameter, public :: MLP_ADAFACTOR_DECAY = 3
    integer, parameter, public :: MLP_ADAFACTOR_LOG_EPSILON = 4
    integer, parameter, public :: MLP_ADAFACTOR_LOG_CLIP_THRESHOLD = 5
    real(dp), parameter :: ACTIVE_SET_TOLERANCE = 32.0_dp*epsilon(1.0_dp)

    type, public :: mlp_adafactor_hypergradient_metadata_t
        integer :: parameter_count = MLP_ADAFACTOR_HYPERPARAMETER_COUNT
        integer :: log_learning_rate_index = MLP_ADAFACTOR_LOG_LEARNING_RATE
        integer :: log_l2_index = MLP_ADAFACTOR_LOG_L2
        integer :: decay_index = MLP_ADAFACTOR_DECAY
        integer :: log_epsilon_index = MLP_ADAFACTOR_LOG_EPSILON
        integer :: log_clip_threshold_index = MLP_ADAFACTOR_LOG_CLIP_THRESHOLD
        integer :: inner_steps = 0
        logical :: relative_step = .false.
        logical :: scale_parameter = .false.
    end type mlp_adafactor_hypergradient_metadata_t

    type, public :: mlp_adafactor_hypergradient_options_t
        !! Smooth fixed-trajectory vector Adafactor configuration.
        integer :: steps = 8
        real(dp) :: learning_rate = 1.0e-2_dp
        real(dp) :: l2 = 1.0e-4_dp
        real(dp) :: decay = 0.999_dp
        real(dp) :: epsilon = 1.0e-8_dp
        real(dp) :: clip_threshold = 1.0_dp
        logical :: relative_step = .false.
        logical :: scale_parameter = .false.
        real(dp) :: lower_log_learning_rate = -12.0_dp
        real(dp) :: upper_log_learning_rate = 2.0_dp
        real(dp) :: lower_log_l2 = -20.0_dp
        real(dp) :: upper_log_l2 = 2.0_dp
        real(dp) :: lower_decay = 0.0_dp
        real(dp) :: upper_decay = 0.999999_dp
        real(dp) :: lower_log_epsilon = -30.0_dp
        real(dp) :: upper_log_epsilon = 2.0_dp
        real(dp) :: lower_log_clip_threshold = -20.0_dp
        real(dp) :: upper_log_clip_threshold = 20.0_dp
        integer :: optimizer = MLP_OPTIMIZER_ADAFACTOR
        integer :: device_kind = FORTML_DEVICE_CPU
        integer :: memory = 8
        integer :: max_iterations = 100
        integer :: max_line_search = 40
        real(dp) :: gradient_tolerance = 1.0e-6_dp
        real(dp) :: step_tolerance = 1.0e-12_dp
        real(dp) :: objective_tolerance = 1.0e-12_dp
    end type mlp_adafactor_hypergradient_options_t

    type, public :: mlp_adafactor_hypergradient_result_t
        logical :: converged = .false.
        integer :: iterations = 0
        integer :: line_search_evaluations = 0
        real(dp) :: objective = huge(1.0_dp)
        real(dp) :: gradient_norm = huge(1.0_dp)
        real(dp) :: log_learning_rate = 0.0_dp
        real(dp) :: log_l2 = 0.0_dp
        real(dp) :: decay = 0.0_dp
        real(dp) :: log_epsilon = 0.0_dp
        real(dp) :: log_clip_threshold = 0.0_dp
        real(dp) :: learning_rate = 0.0_dp
        real(dp) :: l2 = 0.0_dp
        real(dp) :: epsilon = 0.0_dp
        real(dp) :: clip_threshold = 0.0_dp
    end type mlp_adafactor_hypergradient_result_t

    type, public :: mlp_adafactor_hypergradient_objective_t
        !! FortOpt-compatible validation objective through vector Adafactor.
        private
        type(mlp_t), pointer :: model => null()
        real(dp), allocatable :: train_x(:, :), train_target(:, :)
        real(dp), allocatable :: validation_x(:, :), validation_target(:, :)
        real(dp), allocatable :: initial_parameters(:)
        type(mlp_adafactor_hypergradient_metadata_t) :: layout
        real(dp) :: initial_log_learning_rate = 0.0_dp
        real(dp) :: initial_log_l2 = 0.0_dp
        real(dp) :: initial_decay = 0.999_dp
        real(dp) :: initial_log_epsilon = -18.420680743952367_dp
        real(dp) :: initial_log_clip_threshold = 0.0_dp
        logical :: initialized = .false.
    contains
        procedure, public :: initialize => mlp_adafactor_hypergradient_initialize
        procedure, public :: parameter_count => mlp_adafactor_hypergradient_parameter_count
        procedure, public :: metadata => mlp_adafactor_hypergradient_metadata
        procedure, public :: parameters => mlp_adafactor_hypergradient_parameters
        procedure, public :: value_gradient => mlp_adafactor_hypergradient_value_gradient
        procedure, public :: jvp => mlp_adafactor_hypergradient_jvp
        procedure, public :: vjp => mlp_adafactor_hypergradient_vjp
        procedure, public :: fortopt => mlp_adafactor_hypergradient_fortopt
        procedure, public :: is_initialized => mlp_adafactor_hypergradient_is_initialized
    end type mlp_adafactor_hypergradient_objective_t

    public :: mlp_optimize_adafactor_hyperparameters

contains

    subroutine mlp_adafactor_hypergradient_initialize(self, model, train_x, train_target, &
            validation_x, validation_target, options, status)
        class(mlp_adafactor_hypergradient_objective_t), intent(out) :: self
        type(mlp_t), target, intent(inout) :: model
        real(dp), intent(in) :: train_x(:, :), train_target(:, :)
        real(dp), intent(in) :: validation_x(:, :), validation_target(:, :)
        type(mlp_adafactor_hypergradient_options_t), intent(in) :: options
        type(fortnum_status_t), intent(out) :: status

        self%initialized = .false.
        self%layout = mlp_adafactor_hypergradient_metadata_t()
        if (options%relative_step .or. options%scale_parameter) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "MLP Adafactor hypergradient: relative-step or parameter-scaling state is unsupported")
            return
        end if
        if (.not. valid_options(options)) then
            if (options%optimizer /= MLP_OPTIMIZER_ADAFACTOR .or. &
                    options%device_kind /= FORTML_DEVICE_CPU) then
                call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                    "MLP Adafactor hypergradient: optimizer or device is unsupported")
            else
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP Adafactor hypergradient: options are invalid")
            end if
            return
        end if
        if (.not. valid_data(model, train_x, train_target) .or. &
                .not. valid_data(model, validation_x, validation_target)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP Adafactor hypergradient: model or data shape is invalid")
            return
        end if
        self%model => model
        allocate(self%train_x, source=train_x)
        allocate(self%train_target, source=train_target)
        allocate(self%validation_x, source=validation_x)
        allocate(self%validation_target, source=validation_target)
        allocate(self%initial_parameters, source=model%parameters())
        self%layout%inner_steps = options%steps
        self%layout%relative_step = options%relative_step
        self%layout%scale_parameter = options%scale_parameter
        self%initial_log_learning_rate = log(options%learning_rate)
        self%initial_log_l2 = log(options%l2)
        self%initial_decay = options%decay
        self%initial_log_epsilon = log(options%epsilon)
        self%initial_log_clip_threshold = log(options%clip_threshold)
        self%initialized = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_adafactor_hypergradient_initialize

    integer function mlp_adafactor_hypergradient_parameter_count(self) result(count)
        class(mlp_adafactor_hypergradient_objective_t), intent(in) :: self

        count = 0
        if (self%initialized) count = self%layout%parameter_count
    end function mlp_adafactor_hypergradient_parameter_count

    function mlp_adafactor_hypergradient_metadata(self) result(layout)
        class(mlp_adafactor_hypergradient_objective_t), intent(in) :: self
        type(mlp_adafactor_hypergradient_metadata_t) :: layout

        layout = self%layout
    end function mlp_adafactor_hypergradient_metadata

    function mlp_adafactor_hypergradient_parameters(self) result(parameters)
        class(mlp_adafactor_hypergradient_objective_t), intent(in) :: self
        real(dp), allocatable :: parameters(:)

        allocate(parameters(MLP_ADAFACTOR_HYPERPARAMETER_COUNT))
        parameters = [self%initial_log_learning_rate, self%initial_log_l2, &
            self%initial_decay, self%initial_log_epsilon, self%initial_log_clip_threshold]
    end function mlp_adafactor_hypergradient_parameters

    logical function mlp_adafactor_hypergradient_is_initialized(self) result(yes)
        class(mlp_adafactor_hypergradient_objective_t), intent(in) :: self

        yes = self%initialized .and. associated(self%model) .and. &
            allocated(self%initial_parameters)
    end function mlp_adafactor_hypergradient_is_initialized

    subroutine mlp_adafactor_hypergradient_value_gradient(self, parameters, value, &
            gradient, status)
        class(mlp_adafactor_hypergradient_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: direction(MLP_ADAFACTOR_HYPERPARAMETER_COUNT), tangent

        value = huge(1.0_dp)
        gradient = 0.0_dp
        direction = 0.0_dp
        if (.not. self%is_initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP Adafactor hypergradient: objective is not initialized")
            return
        end if
        if (size(parameters) /= MLP_ADAFACTOR_HYPERPARAMETER_COUNT .or. &
                size(gradient) /= MLP_ADAFACTOR_HYPERPARAMETER_COUNT .or. &
                any(.not. ieee_is_finite(parameters))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP Adafactor hypergradient: packed shape is invalid")
            return
        end if
        call adafactor_forward(self, parameters, direction, value, tangent, gradient, status)
    end subroutine mlp_adafactor_hypergradient_value_gradient

    subroutine mlp_adafactor_hypergradient_jvp(self, parameters, direction, value, &
            tangent, status)
        class(mlp_adafactor_hypergradient_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:), direction(:)
        real(dp), intent(out) :: value, tangent
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: gradient(MLP_ADAFACTOR_HYPERPARAMETER_COUNT)

        value = huge(1.0_dp)
        tangent = 0.0_dp
        gradient = 0.0_dp
        if (.not. self%is_initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP Adafactor hypergradient JVP: objective is not initialized")
            return
        end if
        if (size(parameters) /= MLP_ADAFACTOR_HYPERPARAMETER_COUNT .or. &
                size(direction) /= MLP_ADAFACTOR_HYPERPARAMETER_COUNT .or. &
                any(.not. ieee_is_finite(parameters)) .or. &
                any(.not. ieee_is_finite(direction))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP Adafactor hypergradient JVP: packed shape is invalid")
            return
        end if
        call adafactor_forward(self, parameters, direction, value, tangent, gradient, status)
    end subroutine mlp_adafactor_hypergradient_jvp

    subroutine mlp_adafactor_hypergradient_vjp(self, parameters, output_bar, gradient, &
            status)
        class(mlp_adafactor_hypergradient_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:), output_bar
        real(dp), intent(out) :: gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: value

        gradient = 0.0_dp
        if (.not. ieee_is_finite(output_bar)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP Adafactor hypergradient VJP: cotangent is invalid")
            return
        end if
        call self%value_gradient(parameters, value, gradient, status)
        if (status%code /= FORTNUM_OK) return
        gradient = output_bar*gradient
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_adafactor_hypergradient_vjp

    subroutine mlp_adafactor_hypergradient_fortopt(self, objective, status)
        class(mlp_adafactor_hypergradient_objective_t), target, intent(inout) :: self
        type(objective_t), intent(out) :: objective
        type(fortnum_status_t), intent(out) :: status

        if (.not. self%is_initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP Adafactor hypergradient: objective is not initialized")
            return
        end if
        call objective%initialize_context(MLP_ADAFACTOR_HYPERPARAMETER_COUNT, self, &
            mlp_adafactor_hypergradient_context_callback, status)
    end subroutine mlp_adafactor_hypergradient_fortopt

    subroutine mlp_adafactor_hypergradient_context_callback(context, parameters, value, &
            gradient, status)
        class(*), intent(inout) :: context
        real(dp), intent(in) :: parameters(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status

        select type (adapter => context)
            type is (mlp_adafactor_hypergradient_objective_t)
            call adapter%value_gradient(parameters, value, gradient, status)
        class default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP Adafactor hypergradient: context has the wrong type")
        end select
    end subroutine mlp_adafactor_hypergradient_context_callback

    subroutine mlp_optimize_adafactor_hyperparameters(model, train_x, train_target, &
            validation_x, validation_target, options, result, status)
        type(mlp_t), target, intent(inout) :: model
        real(dp), intent(in) :: train_x(:, :), train_target(:, :)
        real(dp), intent(in) :: validation_x(:, :), validation_target(:, :)
        type(mlp_adafactor_hypergradient_options_t), intent(in) :: options
        type(mlp_adafactor_hypergradient_result_t), intent(out) :: result
        type(fortnum_status_t), intent(out) :: status
        type(mlp_adafactor_hypergradient_objective_t), target :: adapter
        type(objective_t) :: objective
        type(lbfgsb_t) :: optimizer
        type(lbfgsb_options_t) :: optimizer_options
        type(lbfgsb_result_t) :: optimizer_result
        real(dp) :: parameters(MLP_ADAFACTOR_HYPERPARAMETER_COUNT)
        real(dp) :: lower(MLP_ADAFACTOR_HYPERPARAMETER_COUNT)
        real(dp) :: upper(MLP_ADAFACTOR_HYPERPARAMETER_COUNT)
        real(dp) :: gradient(MLP_ADAFACTOR_HYPERPARAMETER_COUNT)

        result = mlp_adafactor_hypergradient_result_t()
        if (.not. valid_options(options) .or. options%relative_step .or. &
                options%scale_parameter) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP Adafactor hyperparameter optimization: options are invalid")
            return
        end if
        call adapter%initialize(model, train_x, train_target, validation_x, &
            validation_target, options, status)
        if (status%code /= FORTNUM_OK) return
        parameters = adapter%parameters()
        lower = [options%lower_log_learning_rate, options%lower_log_l2, options%lower_decay, &
            options%lower_log_epsilon, options%lower_log_clip_threshold]
        upper = [options%upper_log_learning_rate, options%upper_log_l2, options%upper_decay, &
            options%upper_log_epsilon, options%upper_log_clip_threshold]
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
        result%log_learning_rate = parameters(MLP_ADAFACTOR_LOG_LEARNING_RATE)
        result%log_l2 = parameters(MLP_ADAFACTOR_LOG_L2)
        result%decay = parameters(MLP_ADAFACTOR_DECAY)
        result%log_epsilon = parameters(MLP_ADAFACTOR_LOG_EPSILON)
        result%log_clip_threshold = parameters(MLP_ADAFACTOR_LOG_CLIP_THRESHOLD)
        result%learning_rate = exp(result%log_learning_rate)
        result%l2 = exp(result%log_l2)
        result%epsilon = exp(result%log_epsilon)
        result%clip_threshold = exp(result%log_clip_threshold)
        if (.not. result%converged) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "MLP Adafactor hyperparameter optimization: iteration limit reached")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_optimize_adafactor_hyperparameters

    subroutine adafactor_forward(self, parameters, direction, value, tangent, gradient, status)
        class(mlp_adafactor_hypergradient_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:), direction(:)
        real(dp), intent(out) :: value, tangent, gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: theta(:), theta_dot(:, :), second(:), second_previous(:)
        real(dp), allocatable :: second_dot(:, :), raw_gradient(:), gradient_dot(:), hvp(:)
        real(dp), allocatable :: validation_gradient(:), sqrt_second(:), denominator(:)
        real(dp), allocatable :: update(:), update_dot(:), denominator_dot(:)
        real(dp) :: learning_rate, l2, decay, epsilon_value, clip_threshold
        real(dp) :: learning_rate_dot, l2_dot, decay_dot, epsilon_dot, clip_dot
        real(dp) :: train_value, l2_gradient, scalar_hvp
        real(dp) :: update_rms, update_rms_dot, clip_scale, clip_scale_dot
        real(dp) :: denominator_value, sqrt_value
        integer :: n_parameters, step, parameter_index

        value = huge(1.0_dp)
        tangent = 0.0_dp
        gradient = 0.0_dp
        if (.not. finite_parameters(parameters, learning_rate, l2, decay, &
                epsilon_value, clip_threshold)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP Adafactor hypergradient: packed parameters are invalid")
            return
        end if
        n_parameters = size(self%initial_parameters)
        allocate(theta, source=self%initial_parameters)
        allocate(theta_dot(n_parameters, MLP_ADAFACTOR_HYPERPARAMETER_COUNT))
        allocate(second(n_parameters), second_previous(n_parameters))
        allocate(second_dot(n_parameters, MLP_ADAFACTOR_HYPERPARAMETER_COUNT))
        allocate(raw_gradient(n_parameters), gradient_dot(n_parameters), hvp(n_parameters))
        allocate(validation_gradient(n_parameters))
        allocate(sqrt_second(n_parameters), denominator(n_parameters), denominator_dot(n_parameters))
        allocate(update(n_parameters), update_dot(n_parameters))
        theta_dot = 0.0_dp
        second = 0.0_dp
        second_dot = 0.0_dp

        do step = 1, self%layout%inner_steps
            call self%model%set_parameters(theta, status)
            if (status%code /= FORTNUM_OK) return
            call mlp_loss_value_gradient(self%model, self%train_x, self%train_target, l2, &
                train_value, raw_gradient, l2_gradient, status)
            if (status%code /= FORTNUM_OK) return
            second_previous = second
            second = decay*second_previous + (1.0_dp-decay)*raw_gradient*raw_gradient
            if (any(second < 0.0_dp) .or. any(.not. ieee_is_finite(second))) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP Adafactor hypergradient: second-moment state is invalid")
                return
            end if
            update_rms = sqrt(sum(second)/real(n_parameters, dp))
            if (abs(update_rms-clip_threshold) <= ACTIVE_SET_TOLERANCE*max(1.0_dp, clip_threshold)) then
                call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                    "MLP Adafactor hypergradient: clip active-set boundary is nondifferentiable")
                return
            end if
            clip_scale = max(1.0_dp, update_rms/clip_threshold)
            sqrt_second = sqrt(max(second, 0.0_dp))
            denominator = sqrt_second+epsilon_value
            update = raw_gradient/clip_scale/denominator
            do parameter_index = 1, MLP_ADAFACTOR_HYPERPARAMETER_COUNT
                l2_dot = 0.0_dp
                decay_dot = 0.0_dp
                epsilon_dot = 0.0_dp
                clip_dot = 0.0_dp
                learning_rate_dot = 0.0_dp
                if (parameter_index == MLP_ADAFACTOR_LOG_LEARNING_RATE) learning_rate_dot = learning_rate
                if (parameter_index == MLP_ADAFACTOR_LOG_L2) l2_dot = l2
                if (parameter_index == MLP_ADAFACTOR_DECAY) decay_dot = 1.0_dp
                if (parameter_index == MLP_ADAFACTOR_LOG_EPSILON) epsilon_dot = epsilon_value
                if (parameter_index == MLP_ADAFACTOR_LOG_CLIP_THRESHOLD) clip_dot = clip_threshold
                call mlp_loss_hvp(self%model, self%train_x, self%train_target, l2, &
                    theta_dot(:, parameter_index), l2_dot, hvp, scalar_hvp, status)
                if (status%code /= FORTNUM_OK) return
                gradient_dot = hvp
                second_dot(:, parameter_index) = decay*second_dot(:, parameter_index) + &
                    (1.0_dp-decay)*2.0_dp*raw_gradient*gradient_dot + &
                    decay_dot*(second_previous-raw_gradient*raw_gradient)
                update_rms_dot = 0.0_dp
                if (update_rms > 0.0_dp) update_rms_dot = &
                    sum(second_dot(:, parameter_index))/(2.0_dp*real(n_parameters, dp)*update_rms)
                clip_scale_dot = 0.0_dp
                if (update_rms > clip_threshold) clip_scale_dot = &
                    update_rms_dot/clip_threshold - update_rms*clip_dot/(clip_threshold*clip_threshold)
                denominator_dot = 0.0_dp
                where (sqrt_second > 0.0_dp)
                    denominator_dot = second_dot(:, parameter_index)/(2.0_dp*sqrt_second)
                end where
                denominator_dot = denominator_dot+epsilon_dot
                update_dot = gradient_dot/clip_scale/denominator - &
                    raw_gradient*clip_scale_dot/(clip_scale*clip_scale*denominator) - &
                    raw_gradient*denominator_dot/(clip_scale*denominator*denominator)
                theta_dot(:, parameter_index) = theta_dot(:, parameter_index) - &
                    learning_rate_dot*update-learning_rate*update_dot
            end do
            theta = theta-learning_rate*update
            if (any(.not. ieee_is_finite(theta)) .or. any(.not. ieee_is_finite(theta_dot)) .or. &
                    any(.not. ieee_is_finite(second_dot))) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP Adafactor hypergradient: trajectory is not finite")
                return
            end if
        end do
        call self%model%set_parameters(theta, status)
        if (status%code /= FORTNUM_OK) return
        call mlp_loss_value_gradient(self%model, self%validation_x, self%validation_target, &
            0.0_dp, value, validation_gradient, l2_gradient, status)
        if (status%code /= FORTNUM_OK) return
        do parameter_index = 1, MLP_ADAFACTOR_HYPERPARAMETER_COUNT
            gradient(parameter_index) = dot_product(validation_gradient, theta_dot(:, parameter_index))
        end do
        tangent = dot_product(gradient, direction)
        if (.not. ieee_is_finite(value) .or. any(.not. ieee_is_finite(gradient)) .or. &
                .not. ieee_is_finite(tangent)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP Adafactor hypergradient: trajectory product is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine adafactor_forward

    logical function finite_parameters(parameters, learning_rate, l2, decay, epsilon_value, &
            clip_threshold) result(valid)
        real(dp), intent(in) :: parameters(:)
        real(dp), intent(out) :: learning_rate, l2, decay, epsilon_value, clip_threshold

        learning_rate = 0.0_dp
        l2 = 0.0_dp
        decay = 0.0_dp
        epsilon_value = 0.0_dp
        clip_threshold = 0.0_dp
        valid = size(parameters) == MLP_ADAFACTOR_HYPERPARAMETER_COUNT .and. &
            all(ieee_is_finite(parameters))
        if (.not. valid) return
        learning_rate = exp(parameters(MLP_ADAFACTOR_LOG_LEARNING_RATE))
        l2 = exp(parameters(MLP_ADAFACTOR_LOG_L2))
        decay = parameters(MLP_ADAFACTOR_DECAY)
        epsilon_value = exp(parameters(MLP_ADAFACTOR_LOG_EPSILON))
        clip_threshold = exp(parameters(MLP_ADAFACTOR_LOG_CLIP_THRESHOLD))
        valid = ieee_is_finite(learning_rate) .and. ieee_is_finite(l2) .and. &
            ieee_is_finite(decay) .and. ieee_is_finite(epsilon_value) .and. &
            ieee_is_finite(clip_threshold) .and. learning_rate > 0.0_dp .and. &
            l2 > 0.0_dp .and. decay >= 0.0_dp .and. decay < 1.0_dp .and. &
            epsilon_value > 0.0_dp .and. clip_threshold > 0.0_dp
    end function finite_parameters

    logical function valid_options(options) result(valid)
        type(mlp_adafactor_hypergradient_options_t), intent(in) :: options

        valid = options%steps >= 1 .and. options%optimizer == MLP_OPTIMIZER_ADAFACTOR .and. &
            options%device_kind == FORTML_DEVICE_CPU .and. .not. options%relative_step .and. &
            .not. options%scale_parameter .and. ieee_is_finite(options%learning_rate) .and. &
            ieee_is_finite(options%l2) .and. ieee_is_finite(options%decay) .and. &
            ieee_is_finite(options%epsilon) .and. ieee_is_finite(options%clip_threshold) .and. &
            options%learning_rate > 0.0_dp .and. options%l2 > 0.0_dp .and. &
            options%decay >= 0.0_dp .and. options%decay < 1.0_dp .and. &
            options%epsilon > 0.0_dp .and. options%clip_threshold > 0.0_dp .and. &
            ieee_is_finite(options%lower_log_learning_rate) .and. &
            ieee_is_finite(options%upper_log_learning_rate) .and. &
            ieee_is_finite(options%lower_log_l2) .and. ieee_is_finite(options%upper_log_l2) .and. &
            ieee_is_finite(options%lower_decay) .and. ieee_is_finite(options%upper_decay) .and. &
            ieee_is_finite(options%lower_log_epsilon) .and. ieee_is_finite(options%upper_log_epsilon) .and. &
            ieee_is_finite(options%lower_log_clip_threshold) .and. &
            ieee_is_finite(options%upper_log_clip_threshold) .and. &
            options%lower_log_learning_rate <= options%upper_log_learning_rate .and. &
            options%lower_log_l2 <= options%upper_log_l2 .and. options%lower_decay <= options%upper_decay .and. &
            options%lower_log_epsilon <= options%upper_log_epsilon .and. &
            options%lower_log_clip_threshold <= options%upper_log_clip_threshold .and. &
            log(options%learning_rate) >= options%lower_log_learning_rate .and. &
            log(options%learning_rate) <= options%upper_log_learning_rate .and. &
            log(options%l2) >= options%lower_log_l2 .and. log(options%l2) <= options%upper_log_l2 .and. &
            options%decay >= options%lower_decay .and. options%decay <= options%upper_decay .and. &
            log(options%epsilon) >= options%lower_log_epsilon .and. log(options%epsilon) <= options%upper_log_epsilon .and. &
            log(options%clip_threshold) >= options%lower_log_clip_threshold .and. &
            log(options%clip_threshold) <= options%upper_log_clip_threshold .and. options%memory >= 1 .and. &
            options%max_iterations >= 1 .and. options%max_line_search >= 1 .and. &
            ieee_is_finite(options%gradient_tolerance) .and. ieee_is_finite(options%step_tolerance) .and. &
            ieee_is_finite(options%objective_tolerance) .and. options%gradient_tolerance >= 0.0_dp .and. &
            options%step_tolerance >= 0.0_dp .and. options%objective_tolerance >= 0.0_dp
    end function valid_options

    logical function valid_data(model, x, target) result(valid)
        type(mlp_t), intent(in) :: model
        real(dp), intent(in) :: x(:, :), target(:, :)

        valid = allocated(model%layer_sizes) .and. size(model%layer_sizes) >= 2 .and. &
            size(x, 1) >= 1 .and. size(x, 2) == model%layer_sizes(1) .and. &
            size(target, 1) == size(x, 1) .and. &
            size(target, 2) == model%layer_sizes(size(model%layer_sizes)) .and. &
            all(ieee_is_finite(x)) .and. all(ieee_is_finite(target))
    end function valid_data

end module fortml_mlp_adafactor_hypergradient
