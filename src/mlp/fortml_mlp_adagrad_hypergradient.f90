module fortml_mlp_adagrad_hypergradient
    !! Exact hypergradients through a fixed full-batch Adagrad trajectory.
    !!
    !! The packed outer vector is
    !! `[log(learning_rate), log(l2), log(epsilon)]`.  The validation MSE
    !! after a fixed number of inner updates is differentiated with the MLP's
    !! analytic HVP; no finite-difference or optimizer fallback is used.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED, FORTNUM_CONVERGENCE_ERROR
    use fortml_device, only: FORTML_DEVICE_CPU
    use fortml_mlp, only: mlp_t
    use fortml_mlp_training, only: mlp_loss_value_gradient, mlp_loss_hvp, &
        MLP_OPTIMIZER_ADAGRAD
    use fortopt_objective, only: objective_t
    use fortopt_lbfgsb, only: lbfgsb_t, lbfgsb_options_t, lbfgsb_result_t
    implicit none
    private

    integer, parameter, public :: MLP_ADAGRAD_HYPERPARAMETER_COUNT = 3
    integer, parameter, public :: MLP_ADAGRAD_LOG_LEARNING_RATE = 1
    integer, parameter, public :: MLP_ADAGRAD_LOG_L2 = 2
    integer, parameter, public :: MLP_ADAGRAD_LOG_EPSILON = 3

    type, public :: mlp_adagrad_hypergradient_metadata_t
        integer :: parameter_count = MLP_ADAGRAD_HYPERPARAMETER_COUNT
        integer :: log_learning_rate_index = MLP_ADAGRAD_LOG_LEARNING_RATE
        integer :: log_l2_index = MLP_ADAGRAD_LOG_L2
        integer :: log_epsilon_index = MLP_ADAGRAD_LOG_EPSILON
        integer :: inner_steps = 0
    end type mlp_adagrad_hypergradient_metadata_t

    type, public :: mlp_adagrad_hypergradient_options_t
        !! Fixed full-batch Adagrad trajectory configuration.
        integer :: steps = 8
        real(dp) :: learning_rate = 1.0e-2_dp
        real(dp) :: l2 = 1.0e-4_dp
        real(dp) :: epsilon = 1.0e-8_dp
        real(dp) :: lower_log_learning_rate = -12.0_dp
        real(dp) :: upper_log_learning_rate = 2.0_dp
        real(dp) :: lower_log_l2 = -20.0_dp
        real(dp) :: upper_log_l2 = 2.0_dp
        real(dp) :: lower_log_epsilon = -30.0_dp
        real(dp) :: upper_log_epsilon = 2.0_dp
        integer :: optimizer = MLP_OPTIMIZER_ADAGRAD
        integer :: device_kind = FORTML_DEVICE_CPU
        integer :: memory = 8
        integer :: max_iterations = 100
        integer :: max_line_search = 40
        real(dp) :: gradient_tolerance = 1.0e-6_dp
        real(dp) :: step_tolerance = 1.0e-12_dp
        real(dp) :: objective_tolerance = 1.0e-12_dp
    end type mlp_adagrad_hypergradient_options_t

    type, public :: mlp_adagrad_hypergradient_result_t
        logical :: converged = .false.
        integer :: iterations = 0
        integer :: line_search_evaluations = 0
        real(dp) :: objective = huge(1.0_dp)
        real(dp) :: gradient_norm = huge(1.0_dp)
        real(dp) :: log_learning_rate = 0.0_dp
        real(dp) :: log_l2 = 0.0_dp
        real(dp) :: log_epsilon = 0.0_dp
        real(dp) :: learning_rate = 0.0_dp
        real(dp) :: l2 = 0.0_dp
        real(dp) :: epsilon = 0.0_dp
    end type mlp_adagrad_hypergradient_result_t

    type, public :: mlp_adagrad_hypergradient_objective_t
        !! Exact value/gradient/JVP/VJP products through Adagrad state.
        private
        type(mlp_t), pointer :: model => null()
        real(dp), allocatable :: train_x(:, :), train_target(:, :)
        real(dp), allocatable :: validation_x(:, :), validation_target(:, :)
        real(dp), allocatable :: initial_parameters(:)
        type(mlp_adagrad_hypergradient_metadata_t) :: layout
        real(dp) :: initial_log_learning_rate = 0.0_dp
        real(dp) :: initial_log_l2 = 0.0_dp
        real(dp) :: initial_log_epsilon = -18.420680743952367_dp
        logical :: initialized = .false.
    contains
        procedure, public :: initialize => mlp_adagrad_hypergradient_initialize
        procedure, public :: parameter_count => mlp_adagrad_hypergradient_parameter_count
        procedure, public :: metadata => mlp_adagrad_hypergradient_metadata
        procedure, public :: parameters => mlp_adagrad_hypergradient_parameters
        procedure, public :: value_gradient => mlp_adagrad_hypergradient_value_gradient
        procedure, public :: jvp => mlp_adagrad_hypergradient_jvp
        procedure, public :: vjp => mlp_adagrad_hypergradient_vjp
        procedure, public :: fortopt => mlp_adagrad_hypergradient_fortopt
        procedure, public :: is_initialized => mlp_adagrad_hypergradient_is_initialized
    end type mlp_adagrad_hypergradient_objective_t

    public :: mlp_optimize_adagrad_hyperparameters

contains

    subroutine mlp_adagrad_hypergradient_initialize(self, model, train_x, train_target, &
            validation_x, validation_target, options, status)
        class(mlp_adagrad_hypergradient_objective_t), intent(out) :: self
        type(mlp_t), target, intent(inout) :: model
        real(dp), intent(in) :: train_x(:, :), train_target(:, :)
        real(dp), intent(in) :: validation_x(:, :), validation_target(:, :)
        type(mlp_adagrad_hypergradient_options_t), intent(in) :: options
        type(fortnum_status_t), intent(out) :: status

        self%initialized = .false.
        self%layout = mlp_adagrad_hypergradient_metadata_t()
        if (.not. valid_options(options)) then
            if (options%optimizer /= MLP_OPTIMIZER_ADAGRAD .or. &
                options%device_kind /= FORTML_DEVICE_CPU) then
                call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                    "MLP Adagrad hypergradient: optimizer or device is unsupported")
            else
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP Adagrad hypergradient: options are invalid")
            end if
            return
        end if
        if (.not. valid_data(model, train_x, train_target) .or. &
            .not. valid_data(model, validation_x, validation_target)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP Adagrad hypergradient: model or data is invalid")
            return
        end if

        self%model => model
        allocate(self%train_x, source=train_x)
        allocate(self%train_target, source=train_target)
        allocate(self%validation_x, source=validation_x)
        allocate(self%validation_target, source=validation_target)
        allocate(self%initial_parameters, source=model%parameters())
        self%layout%inner_steps = options%steps
        self%initial_log_learning_rate = log(options%learning_rate)
        self%initial_log_l2 = log(options%l2)
        self%initial_log_epsilon = log(options%epsilon)
        self%initialized = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_adagrad_hypergradient_initialize

    integer function mlp_adagrad_hypergradient_parameter_count(self) result(count)
        class(mlp_adagrad_hypergradient_objective_t), intent(in) :: self

        count = 0
        if (self%initialized) count = self%layout%parameter_count
    end function mlp_adagrad_hypergradient_parameter_count

    function mlp_adagrad_hypergradient_metadata(self) result(layout)
        class(mlp_adagrad_hypergradient_objective_t), intent(in) :: self
        type(mlp_adagrad_hypergradient_metadata_t) :: layout

        layout = self%layout
    end function mlp_adagrad_hypergradient_metadata

    function mlp_adagrad_hypergradient_parameters(self) result(parameters)
        class(mlp_adagrad_hypergradient_objective_t), intent(in) :: self
        real(dp), allocatable :: parameters(:)

        allocate(parameters(MLP_ADAGRAD_HYPERPARAMETER_COUNT))
        parameters = [self%initial_log_learning_rate, self%initial_log_l2, &
            self%initial_log_epsilon]
    end function mlp_adagrad_hypergradient_parameters

    logical function mlp_adagrad_hypergradient_is_initialized(self) result(yes)
        class(mlp_adagrad_hypergradient_objective_t), intent(in) :: self

        yes = self%initialized .and. associated(self%model) .and. &
            allocated(self%initial_parameters)
    end function mlp_adagrad_hypergradient_is_initialized

    subroutine mlp_adagrad_hypergradient_value_gradient(self, parameters, value, &
            gradient, status)
        class(mlp_adagrad_hypergradient_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: direction(MLP_ADAGRAD_HYPERPARAMETER_COUNT), tangent

        value = huge(1.0_dp)
        gradient = 0.0_dp
        direction = 0.0_dp
        if (.not. self%is_initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP Adagrad hypergradient: objective is not initialized")
            return
        end if
        if (size(parameters) /= MLP_ADAGRAD_HYPERPARAMETER_COUNT .or. &
            size(gradient) /= MLP_ADAGRAD_HYPERPARAMETER_COUNT .or. &
            any(.not. ieee_is_finite(parameters))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP Adagrad hypergradient: packed shape is invalid")
            return
        end if
        call adagrad_forward(self, parameters, direction, value, tangent, gradient, status)
    end subroutine mlp_adagrad_hypergradient_value_gradient

    subroutine mlp_adagrad_hypergradient_jvp(self, parameters, direction, value, &
            tangent, status)
        class(mlp_adagrad_hypergradient_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:), direction(:)
        real(dp), intent(out) :: value, tangent
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: gradient(MLP_ADAGRAD_HYPERPARAMETER_COUNT)

        value = huge(1.0_dp)
        tangent = 0.0_dp
        gradient = 0.0_dp
        if (.not. self%is_initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP Adagrad hypergradient JVP: objective is not initialized")
            return
        end if
        if (size(parameters) /= MLP_ADAGRAD_HYPERPARAMETER_COUNT .or. &
            size(direction) /= MLP_ADAGRAD_HYPERPARAMETER_COUNT .or. &
            any(.not. ieee_is_finite(parameters)) .or. &
            any(.not. ieee_is_finite(direction))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP Adagrad hypergradient JVP: packed shape is invalid")
            return
        end if
        call adagrad_forward(self, parameters, direction, value, tangent, gradient, status)
    end subroutine mlp_adagrad_hypergradient_jvp

    subroutine mlp_adagrad_hypergradient_vjp(self, parameters, output_bar, gradient, &
            status)
        class(mlp_adagrad_hypergradient_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:), output_bar
        real(dp), intent(out) :: gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: value

        gradient = 0.0_dp
        if (.not. ieee_is_finite(output_bar)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP Adagrad hypergradient VJP: cotangent is invalid")
            return
        end if
        call self%value_gradient(parameters, value, gradient, status)
        if (status%code /= FORTNUM_OK) return
        gradient = output_bar*gradient
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_adagrad_hypergradient_vjp

    subroutine mlp_adagrad_hypergradient_fortopt(self, objective, status)
        class(mlp_adagrad_hypergradient_objective_t), target, intent(inout) :: self
        type(objective_t), intent(out) :: objective
        type(fortnum_status_t), intent(out) :: status

        if (.not. self%is_initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP Adagrad hypergradient: objective is not initialized")
            return
        end if
        call objective%initialize_context(MLP_ADAGRAD_HYPERPARAMETER_COUNT, self, &
            mlp_adagrad_hypergradient_context_callback, status)
    end subroutine mlp_adagrad_hypergradient_fortopt

    subroutine mlp_adagrad_hypergradient_context_callback(context, parameters, value, &
            gradient, status)
        class(*), intent(inout) :: context
        real(dp), intent(in) :: parameters(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status

        select type (adapter => context)
            type is (mlp_adagrad_hypergradient_objective_t)
            call adapter%value_gradient(parameters, value, gradient, status)
        class default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP Adagrad hypergradient: context has the wrong type")
        end select
    end subroutine mlp_adagrad_hypergradient_context_callback

    subroutine mlp_optimize_adagrad_hyperparameters(model, train_x, train_target, &
            validation_x, validation_target, options, result, status)
        !! Optimize Adagrad trajectory hyperparameters with L-BFGS-B.
        type(mlp_t), target, intent(inout) :: model
        real(dp), intent(in) :: train_x(:, :), train_target(:, :)
        real(dp), intent(in) :: validation_x(:, :), validation_target(:, :)
        type(mlp_adagrad_hypergradient_options_t), intent(in) :: options
        type(mlp_adagrad_hypergradient_result_t), intent(out) :: result
        type(fortnum_status_t), intent(out) :: status
        type(mlp_adagrad_hypergradient_objective_t), target :: adapter
        type(objective_t) :: objective
        type(lbfgsb_t) :: optimizer
        type(lbfgsb_options_t) :: optimizer_options
        type(lbfgsb_result_t) :: optimizer_result
        real(dp) :: parameters(MLP_ADAGRAD_HYPERPARAMETER_COUNT)
        real(dp) :: lower(MLP_ADAGRAD_HYPERPARAMETER_COUNT)
        real(dp) :: upper(MLP_ADAGRAD_HYPERPARAMETER_COUNT)
        real(dp) :: gradient(MLP_ADAGRAD_HYPERPARAMETER_COUNT)

        result = mlp_adagrad_hypergradient_result_t()
        if (.not. valid_options(options)) then
            if (options%optimizer /= MLP_OPTIMIZER_ADAGRAD .or. &
                options%device_kind /= FORTML_DEVICE_CPU) then
                call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                    "MLP Adagrad hyperparameter optimization: unsupported optimizer/device")
            else
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP Adagrad hyperparameter optimization: options are invalid")
            end if
            return
        end if
        call adapter%initialize(model, train_x, train_target, validation_x, &
            validation_target, options, status)
        if (status%code /= FORTNUM_OK) return
        parameters = adapter%parameters()
        lower = [options%lower_log_learning_rate, options%lower_log_l2, &
            options%lower_log_epsilon]
        upper = [options%upper_log_learning_rate, options%upper_log_l2, &
            options%upper_log_epsilon]
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
        result%log_learning_rate = parameters(MLP_ADAGRAD_LOG_LEARNING_RATE)
        result%log_l2 = parameters(MLP_ADAGRAD_LOG_L2)
        result%log_epsilon = parameters(MLP_ADAGRAD_LOG_EPSILON)
        result%learning_rate = exp(result%log_learning_rate)
        result%l2 = exp(result%log_l2)
        result%epsilon = exp(result%log_epsilon)
        if (.not. result%converged) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "MLP Adagrad hyperparameter optimization: iteration limit reached")
            return
        end if
        if (.not. ieee_is_finite(result%objective) .or. &
            .not. ieee_is_finite(result%gradient_norm) .or. &
            .not. ieee_is_finite(result%learning_rate) .or. &
            .not. ieee_is_finite(result%l2) .or. &
            .not. ieee_is_finite(result%epsilon)) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "MLP Adagrad hyperparameter optimization: result is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_optimize_adagrad_hyperparameters

    subroutine adagrad_forward(self, parameters, direction, value, tangent, gradient, &
            status)
        class(mlp_adagrad_hypergradient_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:), direction(:)
        real(dp), intent(out) :: value, tangent, gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: learning_rate, l2, epsilon
        real(dp) :: learning_rate_dot, l2_dot, epsilon_dot
        real(dp) :: train_value, l2_gradient, scalar_hvp
        real(dp), allocatable :: theta(:), theta_dot(:, :), accumulator(:), accumulator_dot(:, :)
        real(dp), allocatable :: raw_gradient(:), gradient_dot(:, :), hvp(:)
        real(dp), allocatable :: denominator(:), denominator_dot(:), step(:), step_dot(:, :)
        real(dp), allocatable :: validation_gradient(:)
        integer :: n_parameters, step_index, parameter_index

        value = huge(1.0_dp)
        tangent = 0.0_dp
        gradient = 0.0_dp
        if (.not. finite_parameters(parameters, learning_rate, l2, epsilon)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP Adagrad hypergradient: log hyperparameters are invalid")
            return
        end if
        n_parameters = size(self%initial_parameters)
        allocate(theta, source=self%initial_parameters)
        allocate(theta_dot(n_parameters, MLP_ADAGRAD_HYPERPARAMETER_COUNT))
        allocate(accumulator(n_parameters), accumulator_dot(n_parameters, &
            MLP_ADAGRAD_HYPERPARAMETER_COUNT))
        allocate(raw_gradient(n_parameters), gradient_dot(n_parameters, &
            MLP_ADAGRAD_HYPERPARAMETER_COUNT), hvp(n_parameters))
        allocate(denominator(n_parameters), denominator_dot(n_parameters))
        allocate(step(n_parameters), step_dot(n_parameters, MLP_ADAGRAD_HYPERPARAMETER_COUNT))
        theta_dot = 0.0_dp
        accumulator = 0.0_dp
        accumulator_dot = 0.0_dp

        do step_index = 1, self%layout%inner_steps
            call self%model%set_parameters(theta, status)
            if (status%code /= FORTNUM_OK) return
            call mlp_loss_value_gradient(self%model, self%train_x, self%train_target, &
                l2, train_value, raw_gradient, l2_gradient, status)
            if (status%code /= FORTNUM_OK) return
            do parameter_index = 1, MLP_ADAGRAD_HYPERPARAMETER_COUNT
                l2_dot = 0.0_dp
                if (parameter_index == MLP_ADAGRAD_LOG_L2) l2_dot = l2
                call mlp_loss_hvp(self%model, self%train_x, self%train_target, l2, &
                    theta_dot(:, parameter_index), l2_dot, hvp, scalar_hvp, status)
                if (status%code /= FORTNUM_OK) return
                gradient_dot(:, parameter_index) = hvp
                accumulator_dot(:, parameter_index) = accumulator_dot(:, parameter_index) + &
                    2.0_dp*raw_gradient*gradient_dot(:, parameter_index)
            end do
            accumulator = accumulator + raw_gradient*raw_gradient
            denominator = sqrt(accumulator) + epsilon
            step = raw_gradient/denominator
            do parameter_index = 1, MLP_ADAGRAD_HYPERPARAMETER_COUNT
                denominator_dot = 0.0_dp
                where (accumulator > 0.0_dp)
                    denominator_dot = accumulator_dot(:, parameter_index) / &
                        (2.0_dp*sqrt(accumulator))
                end where
                epsilon_dot = 0.0_dp
                if (parameter_index == MLP_ADAGRAD_LOG_EPSILON) epsilon_dot = epsilon
                denominator_dot = denominator_dot + epsilon_dot
                step_dot(:, parameter_index) = (gradient_dot(:, parameter_index)*denominator - &
                    raw_gradient*denominator_dot)/(denominator*denominator)
                learning_rate_dot = 0.0_dp
                if (parameter_index == MLP_ADAGRAD_LOG_LEARNING_RATE) then
                    learning_rate_dot = learning_rate
                end if
                theta_dot(:, parameter_index) = theta_dot(:, parameter_index) - &
                    learning_rate_dot*step - learning_rate*step_dot(:, parameter_index)
            end do
            theta = theta-learning_rate*step
            if (any(.not. ieee_is_finite(theta)) .or. &
                any(.not. ieee_is_finite(theta_dot)) .or. &
                any(.not. ieee_is_finite(accumulator))) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP Adagrad hypergradient: trajectory is not finite")
                return
            end if
        end do

        call self%model%set_parameters(theta, status)
        if (status%code /= FORTNUM_OK) return
        allocate(validation_gradient(n_parameters))
        call mlp_loss_value_gradient(self%model, self%validation_x, self%validation_target, &
            0.0_dp, value, validation_gradient, l2_gradient, status)
        if (status%code /= FORTNUM_OK) return
        do parameter_index = 1, MLP_ADAGRAD_HYPERPARAMETER_COUNT
            gradient(parameter_index) = dot_product(validation_gradient, &
                theta_dot(:, parameter_index))
        end do
        tangent = dot_product(gradient, direction)
        if (.not. ieee_is_finite(value) .or. any(.not. ieee_is_finite(gradient)) .or. &
            .not. ieee_is_finite(tangent)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP Adagrad hypergradient: objective product is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine adagrad_forward

    logical function finite_parameters(parameters, learning_rate, l2, epsilon) result(valid)
        real(dp), intent(in) :: parameters(:)
        real(dp), intent(out) :: learning_rate, l2, epsilon

        learning_rate = 0.0_dp
        l2 = 0.0_dp
        epsilon = 0.0_dp
        valid = size(parameters) == MLP_ADAGRAD_HYPERPARAMETER_COUNT
        if (.not. valid) return
        if (any(.not. ieee_is_finite(parameters))) then
            valid = .false.
            return
        end if
        learning_rate = exp(parameters(MLP_ADAGRAD_LOG_LEARNING_RATE))
        l2 = exp(parameters(MLP_ADAGRAD_LOG_L2))
        epsilon = exp(parameters(MLP_ADAGRAD_LOG_EPSILON))
        valid = ieee_is_finite(learning_rate) .and. ieee_is_finite(l2) .and. &
            ieee_is_finite(epsilon) .and. learning_rate > 0.0_dp .and. &
            l2 > 0.0_dp .and. epsilon > 0.0_dp
    end function finite_parameters

    logical function valid_options(options) result(valid)
        type(mlp_adagrad_hypergradient_options_t), intent(in) :: options

        valid = options%steps >= 1 .and. options%optimizer == MLP_OPTIMIZER_ADAGRAD .and. &
            options%device_kind == FORTML_DEVICE_CPU
        if (.not. valid) return
        valid = ieee_is_finite(options%learning_rate) .and. ieee_is_finite(options%l2) .and. &
            ieee_is_finite(options%epsilon) .and. options%learning_rate > 0.0_dp .and. &
            options%l2 > 0.0_dp .and. options%epsilon > 0.0_dp
        if (.not. valid) return
        valid = ieee_is_finite(options%lower_log_learning_rate) .and. &
            ieee_is_finite(options%upper_log_learning_rate) .and. &
            ieee_is_finite(options%lower_log_l2) .and. ieee_is_finite(options%upper_log_l2) .and. &
            ieee_is_finite(options%lower_log_epsilon) .and. &
            ieee_is_finite(options%upper_log_epsilon) .and. &
            options%lower_log_learning_rate <= options%upper_log_learning_rate .and. &
            options%lower_log_l2 <= options%upper_log_l2 .and. &
            options%lower_log_epsilon <= options%upper_log_epsilon
        if (.not. valid) return
        valid = log(options%learning_rate) >= options%lower_log_learning_rate .and. &
            log(options%learning_rate) <= options%upper_log_learning_rate .and. &
            log(options%l2) >= options%lower_log_l2 .and. log(options%l2) <= options%upper_log_l2 .and. &
            log(options%epsilon) >= options%lower_log_epsilon .and. &
            log(options%epsilon) <= options%upper_log_epsilon
        if (.not. valid) return
        valid = options%memory >= 1 .and. options%max_iterations >= 1 .and. &
            options%max_line_search >= 1 .and. ieee_is_finite(options%gradient_tolerance) .and. &
            ieee_is_finite(options%step_tolerance) .and. &
            ieee_is_finite(options%objective_tolerance) .and. &
            options%gradient_tolerance >= 0.0_dp .and. options%step_tolerance >= 0.0_dp .and. &
            options%objective_tolerance >= 0.0_dp
    end function valid_options

    logical function valid_data(model, x, target) result(valid)
        class(mlp_t), intent(in) :: model
        real(dp), intent(in) :: x(:, :), target(:, :)

        valid = .false.
        if (.not. allocated(model%layer_sizes)) return
        if (size(model%layer_sizes) < 2) return
        if (model%parameter_count() <= 0) return
        if (size(x, 1) <= 0 .or. size(target, 1) /= size(x, 1)) return
        if (size(x, 2) /= model%layer_sizes(1)) return
        if (size(target, 2) /= model%layer_sizes(size(model%layer_sizes))) return
        if (any(.not. ieee_is_finite(x)) .or. any(.not. ieee_is_finite(target))) return
        valid = .true.
    end function valid_data

end module fortml_mlp_adagrad_hypergradient
