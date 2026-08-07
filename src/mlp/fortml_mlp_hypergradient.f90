module fortml_mlp_hypergradient
    !! Exact hypergradients through a fixed full-batch MLP SGD trajectory.
    !!
    !! The deliberately small contract in this module is useful as a reliable
    !! building block for differentiable hyperparameter search.  The packed
    !! outer variable is
    !!
    !!     [ log(learning_rate), log(l2) ]
    !!
    !! and the inner model starts from the parameters present at
    !! `initialize`.  Each objective evaluation then performs exactly
    !! `options%steps` full-batch gradient-descent updates.  The validation
    !! MSE after the final update is the scalar outer objective; validation
    !! rows never participate in an update or a regularization term.
    !!
    !! Forward JVPs use the MLP's analytic Hessian-vector product.  The
    !! value/gradient path uses an exact reverse adjoint through the same
    !! trajectory.  No finite differences, optimizer fallback, or hidden host
    !! device transfer is used.  Adam, momentum/Nesterov SGD, and CUDA are
    !! refused explicitly until their complete hypergradient contracts exist.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED, FORTNUM_CONVERGENCE_ERROR
    use fortml_device, only: FORTML_DEVICE_CPU
    use fortml_mlp, only: mlp_t
    use fortml_mlp_training, only: mlp_loss_value_gradient, mlp_loss_hvp, &
        MLP_OPTIMIZER_SGD, MLP_OPTIMIZER_ADAMW
    use fortopt_objective, only: objective_t
    use fortopt_lbfgsb, only: lbfgsb_t, lbfgsb_options_t, lbfgsb_result_t
    implicit none
    private

    integer, parameter, public :: MLP_HYPERPARAMETER_COUNT = 2
    integer, parameter, public :: MLP_LOG_LEARNING_RATE = 1
    integer, parameter, public :: MLP_LOG_L2 = 2
    integer, parameter, public :: MLP_ADAMW_HYPERPARAMETER_COUNT = 3
    integer, parameter, public :: MLP_ADAMW_LOG_LEARNING_RATE = 1
    integer, parameter, public :: MLP_ADAMW_LOG_L2 = 2
    integer, parameter, public :: MLP_ADAMW_LOG_WEIGHT_DECAY = 3

    type, public :: mlp_hyperparameter_metadata_t
        !! Stable packed layout for the outer search variable.
        integer :: parameter_count = MLP_HYPERPARAMETER_COUNT
        integer :: log_learning_rate_index = MLP_LOG_LEARNING_RATE
        integer :: log_l2_index = MLP_LOG_L2
        integer :: inner_steps = 0
    end type mlp_hyperparameter_metadata_t

    type, public :: mlp_hypergradient_options_t
        !! Fixed-trajectory hyperparameter search configuration.
        integer :: steps = 8
        real(dp) :: learning_rate = 1.0e-2_dp
        real(dp) :: l2 = 1.0e-4_dp
        real(dp) :: lower_log_learning_rate = -12.0_dp
        real(dp) :: upper_log_learning_rate = 2.0_dp
        real(dp) :: lower_log_l2 = -20.0_dp
        real(dp) :: upper_log_l2 = 2.0_dp
        integer :: optimizer = MLP_OPTIMIZER_SGD
        real(dp) :: momentum = 0.0_dp
        logical :: nesterov = .false.
        integer :: device_kind = FORTML_DEVICE_CPU
        integer :: memory = 8
        integer :: max_iterations = 100
        integer :: max_line_search = 40
        real(dp) :: gradient_tolerance = 1.0e-6_dp
        real(dp) :: step_tolerance = 1.0e-12_dp
        real(dp) :: objective_tolerance = 1.0e-12_dp
    end type mlp_hypergradient_options_t

    type, public :: mlp_hypergradient_result_t
        logical :: converged = .false.
        integer :: iterations = 0
        integer :: line_search_evaluations = 0
        real(dp) :: objective = huge(1.0_dp)
        real(dp) :: gradient_norm = huge(1.0_dp)
        real(dp) :: log_learning_rate = 0.0_dp
        real(dp) :: log_l2 = 0.0_dp
        real(dp) :: learning_rate = 0.0_dp
        real(dp) :: l2 = 0.0_dp
    end type mlp_hypergradient_result_t

    type, public :: mlp_adamw_hypergradient_metadata_t
        integer :: parameter_count = MLP_ADAMW_HYPERPARAMETER_COUNT
        integer :: log_learning_rate_index = MLP_ADAMW_LOG_LEARNING_RATE
        integer :: log_l2_index = MLP_ADAMW_LOG_L2
        integer :: log_weight_decay_index = MLP_ADAMW_LOG_WEIGHT_DECAY
        integer :: inner_steps = 0
    end type mlp_adamw_hypergradient_metadata_t

    type, public :: mlp_adamw_hypergradient_options_t
        !! Fixed full-batch AdamW trajectory configuration.
        integer :: steps = 8
        real(dp) :: learning_rate = 1.0e-2_dp
        real(dp) :: l2 = 1.0e-4_dp
        real(dp) :: weight_decay = 1.0e-2_dp
        real(dp) :: beta1 = 0.9_dp
        real(dp) :: beta2 = 0.999_dp
        real(dp) :: epsilon = 1.0e-8_dp
        real(dp) :: lower_log_learning_rate = -12.0_dp
        real(dp) :: upper_log_learning_rate = 2.0_dp
        real(dp) :: lower_log_l2 = -20.0_dp
        real(dp) :: upper_log_l2 = 2.0_dp
        real(dp) :: lower_log_weight_decay = -20.0_dp
        real(dp) :: upper_log_weight_decay = 2.0_dp
        integer :: optimizer = MLP_OPTIMIZER_ADAMW
        integer :: device_kind = FORTML_DEVICE_CPU
        integer :: memory = 8
        integer :: max_iterations = 100
        integer :: max_line_search = 40
        real(dp) :: gradient_tolerance = 1.0e-6_dp
        real(dp) :: step_tolerance = 1.0e-12_dp
        real(dp) :: objective_tolerance = 1.0e-12_dp
    end type mlp_adamw_hypergradient_options_t

    type, public :: mlp_adamw_hypergradient_result_t
        logical :: converged = .false.
        integer :: iterations = 0
        integer :: line_search_evaluations = 0
        real(dp) :: objective = huge(1.0_dp)
        real(dp) :: gradient_norm = huge(1.0_dp)
        real(dp) :: log_learning_rate = 0.0_dp
        real(dp) :: log_l2 = 0.0_dp
        real(dp) :: log_weight_decay = 0.0_dp
        real(dp) :: learning_rate = 0.0_dp
        real(dp) :: l2 = 0.0_dp
        real(dp) :: weight_decay = 0.0_dp
    end type mlp_adamw_hypergradient_result_t

    type, public :: mlp_hypergradient_objective_t
        !! FortOpt-compatible validation objective over two log parameters.
        private
        type(mlp_t), pointer :: model => null()
        real(dp), allocatable :: train_x(:, :), train_target(:, :)
        real(dp), allocatable :: validation_x(:, :), validation_target(:, :)
        real(dp), allocatable :: initial_parameters(:)
        type(mlp_hyperparameter_metadata_t) :: layout
        real(dp) :: initial_log_learning_rate = 0.0_dp
        real(dp) :: initial_log_l2 = 0.0_dp
        logical :: initialized = .false.
    contains
        procedure, public :: initialize => mlp_hypergradient_initialize
        procedure, public :: parameter_count => mlp_hypergradient_parameter_count
        procedure, public :: metadata => mlp_hypergradient_metadata
        procedure, public :: parameters => mlp_hypergradient_parameters
        procedure, public :: value_gradient => mlp_hypergradient_value_gradient
        procedure, public :: jvp => mlp_hypergradient_jvp
        procedure, public :: vjp => mlp_hypergradient_vjp
        procedure, public :: fortopt => mlp_hypergradient_fortopt
        procedure, public :: is_initialized => mlp_hypergradient_is_initialized
    end type mlp_hypergradient_objective_t

    type, public :: mlp_adamw_hypergradient_objective_t
        !! Exact value/gradient products through a fixed AdamW trajectory.
        private
        type(mlp_t), pointer :: model => null()
        real(dp), allocatable :: train_x(:, :), train_target(:, :)
        real(dp), allocatable :: validation_x(:, :), validation_target(:, :)
        real(dp), allocatable :: initial_parameters(:)
        type(mlp_adamw_hypergradient_metadata_t) :: layout
        real(dp) :: initial_log_learning_rate = 0.0_dp
        real(dp) :: initial_log_l2 = 0.0_dp
        real(dp) :: initial_log_weight_decay = 0.0_dp
        real(dp) :: beta1 = 0.9_dp
        real(dp) :: beta2 = 0.999_dp
        real(dp) :: epsilon = 1.0e-8_dp
        logical :: initialized = .false.
    contains
        procedure, public :: initialize => mlp_adamw_hypergradient_initialize
        procedure, public :: parameter_count => mlp_adamw_hypergradient_parameter_count
        procedure, public :: metadata => mlp_adamw_hypergradient_metadata
        procedure, public :: parameters => mlp_adamw_hypergradient_parameters
        procedure, public :: value_gradient => mlp_adamw_hypergradient_value_gradient
        procedure, public :: jvp => mlp_adamw_hypergradient_jvp
        procedure, public :: vjp => mlp_adamw_hypergradient_vjp
        procedure, public :: fortopt => mlp_adamw_hypergradient_fortopt
        procedure, public :: is_initialized => mlp_adamw_hypergradient_is_initialized
    end type mlp_adamw_hypergradient_objective_t

    public :: mlp_optimize_hyperparameters
    public :: mlp_optimize_adamw_hyperparameters

contains

    subroutine mlp_hypergradient_initialize(self, model, train_x, train_target, &
            validation_x, validation_target, options, status)
        class(mlp_hypergradient_objective_t), intent(out) :: self
        type(mlp_t), target, intent(inout) :: model
        real(dp), intent(in) :: train_x(:, :), train_target(:, :)
        real(dp), intent(in) :: validation_x(:, :), validation_target(:, :)
        type(mlp_hypergradient_options_t), intent(in) :: options
        type(fortnum_status_t), intent(out) :: status

        self%initialized = .false.
        self%layout = mlp_hyperparameter_metadata_t()
        if (.not. valid_options(options)) then
            if (options%optimizer /= MLP_OPTIMIZER_SGD .or. &
                options%momentum /= 0.0_dp .or. options%nesterov .or. &
                options%device_kind /= FORTML_DEVICE_CPU) then
                call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                    "MLP hypergradient: optimizer or device is unsupported")
            else
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP hypergradient: options are invalid")
            end if
            return
        end if
        if (.not. valid_data(model, train_x, train_target) .or. &
            .not. valid_data(model, validation_x, validation_target)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP hypergradient: model or data shape is invalid")
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
        self%initialized = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_hypergradient_initialize

    integer function mlp_hypergradient_parameter_count(self) result(count)
        class(mlp_hypergradient_objective_t), intent(in) :: self

        count = 0
        if (self%initialized) count = self%layout%parameter_count
    end function mlp_hypergradient_parameter_count

    function mlp_hypergradient_metadata(self) result(layout)
        class(mlp_hypergradient_objective_t), intent(in) :: self
        type(mlp_hyperparameter_metadata_t) :: layout

        layout = self%layout
    end function mlp_hypergradient_metadata

    function mlp_hypergradient_parameters(self) result(parameters)
        class(mlp_hypergradient_objective_t), intent(in) :: self
        real(dp), allocatable :: parameters(:)

        allocate(parameters(MLP_HYPERPARAMETER_COUNT))
        parameters = [self%initial_log_learning_rate, self%initial_log_l2]
    end function mlp_hypergradient_parameters

    logical function mlp_hypergradient_is_initialized(self) result(yes)
        class(mlp_hypergradient_objective_t), intent(in) :: self

        yes = self%initialized .and. associated(self%model) .and. &
            allocated(self%initial_parameters)
    end function mlp_hypergradient_is_initialized

    subroutine mlp_hypergradient_value_gradient(self, parameters, value, gradient, &
            status)
        class(mlp_hypergradient_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status

        value = huge(1.0_dp)
        gradient = 0.0_dp
        if (.not. self%is_initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP hypergradient: objective is not initialized")
            return
        end if
        if (.not. valid_parameter_vector(parameters, gradient)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP hypergradient: packed parameter shape is invalid")
            return
        end if
        call reverse_value_gradient(self, parameters, value, gradient, status)
    end subroutine mlp_hypergradient_value_gradient

    subroutine mlp_hypergradient_jvp(self, parameters, direction, value, tangent, &
            status)
        !! Exact scalar JVP of the final validation objective.
        class(mlp_hypergradient_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:), direction(:)
        real(dp), intent(out) :: value, tangent
        type(fortnum_status_t), intent(out) :: status

        value = huge(1.0_dp)
        tangent = 0.0_dp
        if (.not. self%is_initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP hypergradient JVP: objective is not initialized")
            return
        end if
        if (size(parameters) /= MLP_HYPERPARAMETER_COUNT .or. &
            size(direction) /= MLP_HYPERPARAMETER_COUNT .or. &
            any(.not. ieee_is_finite(parameters)) .or. &
            any(.not. ieee_is_finite(direction))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP hypergradient JVP: packed shape or direction is invalid")
            return
        end if
        call forward_jvp(self, parameters, direction, value, tangent, status)
    end subroutine mlp_hypergradient_jvp

    subroutine mlp_hypergradient_vjp(self, parameters, output_bar, gradient, status)
        !! Exact reverse VJP for a scalar cotangent.
        class(mlp_hypergradient_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:), output_bar
        real(dp), intent(out) :: gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: value

        gradient = 0.0_dp
        if (.not. self%is_initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP hypergradient VJP: objective is not initialized")
            return
        end if
        if (size(parameters) /= MLP_HYPERPARAMETER_COUNT .or. &
            size(gradient) /= MLP_HYPERPARAMETER_COUNT .or. &
            .not. ieee_is_finite(output_bar)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP hypergradient VJP: packed shape or cotangent is invalid")
            return
        end if
        call reverse_value_gradient(self, parameters, value, gradient, status)
        if (status%code /= FORTNUM_OK) return
        gradient = output_bar*gradient
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_hypergradient_vjp

    subroutine mlp_hypergradient_fortopt(self, objective, status)
        class(mlp_hypergradient_objective_t), target, intent(inout) :: self
        type(objective_t), intent(out) :: objective
        type(fortnum_status_t), intent(out) :: status

        if (.not. self%is_initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP hypergradient: objective is not initialized")
            return
        end if
        call objective%initialize_context(MLP_HYPERPARAMETER_COUNT, self, &
            mlp_hypergradient_context_callback, status)
    end subroutine mlp_hypergradient_fortopt

    subroutine mlp_hypergradient_context_callback(context, parameters, value, &
            gradient, status)
        class(*), intent(inout) :: context
        real(dp), intent(in) :: parameters(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status

        select type (adapter => context)
            type is (mlp_hypergradient_objective_t)
            call adapter%value_gradient(parameters, value, gradient, status)
        class default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP hypergradient: context has the wrong type")
        end select
    end subroutine mlp_hypergradient_context_callback

    subroutine mlp_optimize_hyperparameters(model, train_x, train_target, &
            validation_x, validation_target, options, result, status)
        !! Optimize the two log hyperparameters with FortOpt L-BFGS-B.
        type(mlp_t), target, intent(inout) :: model
        real(dp), intent(in) :: train_x(:, :), train_target(:, :)
        real(dp), intent(in) :: validation_x(:, :), validation_target(:, :)
        type(mlp_hypergradient_options_t), intent(in) :: options
        type(mlp_hypergradient_result_t), intent(out) :: result
        type(fortnum_status_t), intent(out) :: status
        type(mlp_hypergradient_objective_t), target :: adapter
        type(objective_t) :: objective
        type(lbfgsb_t) :: optimizer
        type(lbfgsb_options_t) :: optimizer_options
        type(lbfgsb_result_t) :: optimizer_result
        real(dp) :: parameters(MLP_HYPERPARAMETER_COUNT)
        real(dp) :: lower(MLP_HYPERPARAMETER_COUNT), upper(MLP_HYPERPARAMETER_COUNT)
        real(dp) :: gradient(MLP_HYPERPARAMETER_COUNT)

        result = mlp_hypergradient_result_t()
        if (.not. valid_options(options)) then
            if (options%optimizer /= MLP_OPTIMIZER_SGD .or. &
                options%momentum /= 0.0_dp .or. options%nesterov .or. &
                options%device_kind /= FORTML_DEVICE_CPU) then
                call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                    "MLP hyperparameter optimization: unsupported optimizer/device")
            else
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP hyperparameter optimization: options are invalid")
            end if
            return
        end if
        call adapter%initialize(model, train_x, train_target, validation_x, &
            validation_target, options, status)
        if (status%code /= FORTNUM_OK) return
        parameters = adapter%parameters()
        lower = [options%lower_log_learning_rate, options%lower_log_l2]
        upper = [options%upper_log_learning_rate, options%upper_log_l2]
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
        result%log_learning_rate = parameters(MLP_LOG_LEARNING_RATE)
        result%log_l2 = parameters(MLP_LOG_L2)
        result%learning_rate = exp(result%log_learning_rate)
        result%l2 = exp(result%log_l2)
        if (.not. result%converged) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "MLP hyperparameter optimization: iteration limit reached")
            return
        end if
        if (.not. ieee_is_finite(result%objective) .or. &
            .not. ieee_is_finite(result%gradient_norm) .or. &
            .not. ieee_is_finite(result%learning_rate) .or. &
            .not. ieee_is_finite(result%l2)) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "MLP hyperparameter optimization: result is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_optimize_hyperparameters

    subroutine mlp_adamw_hypergradient_initialize(self, model, train_x, train_target, &
            validation_x, validation_target, options, status)
        class(mlp_adamw_hypergradient_objective_t), intent(out) :: self
        type(mlp_t), target, intent(inout) :: model
        real(dp), intent(in) :: train_x(:, :), train_target(:, :)
        real(dp), intent(in) :: validation_x(:, :), validation_target(:, :)
        type(mlp_adamw_hypergradient_options_t), intent(in) :: options
        type(fortnum_status_t), intent(out) :: status

        self%initialized = .false.
        self%layout = mlp_adamw_hypergradient_metadata_t()
        if (.not. valid_adamw_options(options)) then
            if (options%optimizer /= MLP_OPTIMIZER_ADAMW .or. &
                options%device_kind /= FORTML_DEVICE_CPU) then
                call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                    "MLP AdamW hypergradient: optimizer or device is unsupported")
            else
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP AdamW hypergradient: options are invalid")
            end if
            return
        end if
        if (.not. valid_data(model, train_x, train_target) .or. &
            .not. valid_data(model, validation_x, validation_target)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP AdamW hypergradient: model or data shape is invalid")
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
        self%initial_log_weight_decay = log(options%weight_decay)
        self%beta1 = options%beta1
        self%beta2 = options%beta2
        self%epsilon = options%epsilon
        self%initialized = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_adamw_hypergradient_initialize

    integer function mlp_adamw_hypergradient_parameter_count(self) result(count)
        class(mlp_adamw_hypergradient_objective_t), intent(in) :: self

        count = 0
        if (self%initialized) count = self%layout%parameter_count
    end function mlp_adamw_hypergradient_parameter_count

    function mlp_adamw_hypergradient_metadata(self) result(layout)
        class(mlp_adamw_hypergradient_objective_t), intent(in) :: self
        type(mlp_adamw_hypergradient_metadata_t) :: layout

        layout = self%layout
    end function mlp_adamw_hypergradient_metadata

    function mlp_adamw_hypergradient_parameters(self) result(parameters)
        class(mlp_adamw_hypergradient_objective_t), intent(in) :: self
        real(dp), allocatable :: parameters(:)

        allocate(parameters(MLP_ADAMW_HYPERPARAMETER_COUNT))
        parameters = [self%initial_log_learning_rate, self%initial_log_l2, &
            self%initial_log_weight_decay]
    end function mlp_adamw_hypergradient_parameters

    logical function mlp_adamw_hypergradient_is_initialized(self) result(yes)
        class(mlp_adamw_hypergradient_objective_t), intent(in) :: self

        yes = self%initialized .and. associated(self%model) .and. &
            allocated(self%initial_parameters)
    end function mlp_adamw_hypergradient_is_initialized

    subroutine mlp_adamw_hypergradient_value_gradient(self, parameters, value, &
            gradient, status)
        class(mlp_adamw_hypergradient_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: direction(MLP_ADAMW_HYPERPARAMETER_COUNT), tangent

        value = huge(1.0_dp)
        gradient = 0.0_dp
        if (.not. self%is_initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP AdamW hypergradient: objective is not initialized")
            return
        end if
        if (size(parameters) /= MLP_ADAMW_HYPERPARAMETER_COUNT .or. &
            size(gradient) /= MLP_ADAMW_HYPERPARAMETER_COUNT .or. &
            any(.not. ieee_is_finite(parameters))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP AdamW hypergradient: packed parameter shape is invalid")
            return
        end if
        direction = 0.0_dp
        call adamw_forward(self, parameters, direction, value, tangent, gradient, status)
    end subroutine mlp_adamw_hypergradient_value_gradient

    subroutine mlp_adamw_hypergradient_jvp(self, parameters, direction, value, &
            tangent, status)
        class(mlp_adamw_hypergradient_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:), direction(:)
        real(dp), intent(out) :: value, tangent
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: gradient(MLP_ADAMW_HYPERPARAMETER_COUNT)

        value = huge(1.0_dp)
        tangent = 0.0_dp
        gradient = 0.0_dp
        if (.not. self%is_initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP AdamW hypergradient JVP: objective is not initialized")
            return
        end if
        if (size(parameters) /= MLP_ADAMW_HYPERPARAMETER_COUNT .or. &
            size(direction) /= MLP_ADAMW_HYPERPARAMETER_COUNT .or. &
            any(.not. ieee_is_finite(parameters)) .or. &
            any(.not. ieee_is_finite(direction))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP AdamW hypergradient JVP: packed shape is invalid")
            return
        end if
        call adamw_forward(self, parameters, direction, value, tangent, gradient, status)
    end subroutine mlp_adamw_hypergradient_jvp

    subroutine mlp_adamw_hypergradient_vjp(self, parameters, output_bar, gradient, status)
        class(mlp_adamw_hypergradient_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:), output_bar
        real(dp), intent(out) :: gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: value, tangent, direction(MLP_ADAMW_HYPERPARAMETER_COUNT)

        gradient = 0.0_dp
        value = huge(1.0_dp)
        tangent = 0.0_dp
        direction = 0.0_dp
        if (.not. ieee_is_finite(output_bar)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP AdamW hypergradient VJP: cotangent is invalid")
            return
        end if
        call self%value_gradient(parameters, value, gradient, status)
        if (status%code /= FORTNUM_OK) return
        gradient = output_bar*gradient
    end subroutine mlp_adamw_hypergradient_vjp

    subroutine mlp_adamw_hypergradient_fortopt(self, objective, status)
        class(mlp_adamw_hypergradient_objective_t), target, intent(inout) :: self
        type(objective_t), intent(out) :: objective
        type(fortnum_status_t), intent(out) :: status

        if (.not. self%is_initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP AdamW hypergradient: objective is not initialized")
            return
        end if
        call objective%initialize_context(MLP_ADAMW_HYPERPARAMETER_COUNT, self, &
            mlp_adamw_hypergradient_context_callback, status)
    end subroutine mlp_adamw_hypergradient_fortopt

    subroutine mlp_adamw_hypergradient_context_callback(context, parameters, value, &
            gradient, status)
        class(*), intent(inout) :: context
        real(dp), intent(in) :: parameters(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status

        select type (adapter => context)
            type is (mlp_adamw_hypergradient_objective_t)
            call adapter%value_gradient(parameters, value, gradient, status)
        class default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP AdamW hypergradient: context has the wrong type")
        end select
    end subroutine mlp_adamw_hypergradient_context_callback

    subroutine mlp_optimize_adamw_hyperparameters(model, train_x, train_target, &
            validation_x, validation_target, options, result, status)
        type(mlp_t), target, intent(inout) :: model
        real(dp), intent(in) :: train_x(:, :), train_target(:, :)
        real(dp), intent(in) :: validation_x(:, :), validation_target(:, :)
        type(mlp_adamw_hypergradient_options_t), intent(in) :: options
        type(mlp_adamw_hypergradient_result_t), intent(out) :: result
        type(fortnum_status_t), intent(out) :: status
        type(mlp_adamw_hypergradient_objective_t), target :: adapter
        type(objective_t) :: objective
        type(lbfgsb_t) :: optimizer
        type(lbfgsb_options_t) :: optimizer_options
        type(lbfgsb_result_t) :: optimizer_result
        real(dp) :: parameters(MLP_ADAMW_HYPERPARAMETER_COUNT)
        real(dp) :: lower(MLP_ADAMW_HYPERPARAMETER_COUNT)
        real(dp) :: upper(MLP_ADAMW_HYPERPARAMETER_COUNT)
        real(dp) :: gradient(MLP_ADAMW_HYPERPARAMETER_COUNT)

        result = mlp_adamw_hypergradient_result_t()
        if (.not. valid_adamw_options(options)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP AdamW hyperparameter optimization: options are invalid")
            return
        end if
        call adapter%initialize(model, train_x, train_target, validation_x, &
            validation_target, options, status)
        if (status%code /= FORTNUM_OK) return
        parameters = adapter%parameters()
        lower = [options%lower_log_learning_rate, options%lower_log_l2, &
            options%lower_log_weight_decay]
        upper = [options%upper_log_learning_rate, options%upper_log_l2, &
            options%upper_log_weight_decay]
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
        result%log_learning_rate = parameters(1)
        result%log_l2 = parameters(2)
        result%log_weight_decay = parameters(3)
        result%learning_rate = exp(parameters(1))
        result%l2 = exp(parameters(2))
        result%weight_decay = exp(parameters(3))
        if (.not. result%converged) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "MLP AdamW hyperparameter optimization: iteration limit reached")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_optimize_adamw_hyperparameters

    subroutine adamw_forward(self, parameters, direction, value, tangent, gradient, &
            status)
        !! Propagate exact forward sensitivities through full-batch AdamW.
        class(mlp_adamw_hypergradient_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:), direction(:)
        real(dp), intent(out) :: value, tangent, gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: theta(:), theta_dot(:, :), first(:), second(:)
        real(dp), allocatable :: first_dot(:, :), second_dot(:, :)
        real(dp), allocatable :: raw_gradient(:), hvp(:)
        real(dp), allocatable :: validation_gradient(:), zero_input(:, :)
        real(dp) :: learning_rate, l2, weight_decay
        real(dp) :: learning_rate_dot, l2_dot, weight_decay_dot
        real(dp) :: train_value, l2_gradient, scalar_hvp
        real(dp) :: first_hat, second_hat
        real(dp), allocatable :: first_hat_dot(:), second_hat_dot(:)
        real(dp), allocatable :: denominator(:), sqrt_second(:), update(:), update_dot(:)
        integer :: n_parameters, step, parameter_index

        value = huge(1.0_dp)
        tangent = 0.0_dp
        gradient = 0.0_dp
        if (.not. finite_adamw_logs(parameters, learning_rate, l2, weight_decay)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP AdamW hypergradient: log hyperparameters are invalid")
            return
        end if
        n_parameters = size(self%initial_parameters)
        allocate(theta, source=self%initial_parameters)
        allocate(theta_dot(n_parameters, MLP_ADAMW_HYPERPARAMETER_COUNT))
        allocate(first(n_parameters), second(n_parameters))
        allocate(first_dot(n_parameters, MLP_ADAMW_HYPERPARAMETER_COUNT))
        allocate(second_dot(n_parameters, MLP_ADAMW_HYPERPARAMETER_COUNT))
        allocate(raw_gradient(n_parameters), hvp(n_parameters))
        allocate(first_hat_dot(n_parameters), second_hat_dot(n_parameters), &
            denominator(n_parameters), sqrt_second(n_parameters), &
            update(n_parameters), update_dot(n_parameters))
        theta_dot = 0.0_dp
        first = 0.0_dp
        second = 0.0_dp
        first_dot = 0.0_dp
        second_dot = 0.0_dp
        do step = 1, self%layout%inner_steps
            call self%model%set_parameters(theta, status)
            if (status%code /= FORTNUM_OK) return
            call mlp_loss_value_gradient(self%model, self%train_x, self%train_target, &
                l2, train_value, raw_gradient, l2_gradient, status)
            if (status%code /= FORTNUM_OK) return
            first = self%beta1*first + (1.0_dp-self%beta1)*raw_gradient
            second = self%beta2*second + (1.0_dp-self%beta2)*raw_gradient*raw_gradient
            do parameter_index = 1, MLP_ADAMW_HYPERPARAMETER_COUNT
                learning_rate_dot = 0.0_dp
                l2_dot = 0.0_dp
                weight_decay_dot = 0.0_dp
                if (parameter_index == MLP_ADAMW_LOG_LEARNING_RATE) then
                    learning_rate_dot = learning_rate
                else if (parameter_index == MLP_ADAMW_LOG_L2) then
                    l2_dot = l2
                else
                    weight_decay_dot = weight_decay
                end if
                call mlp_loss_hvp(self%model, self%train_x, self%train_target, l2, &
                    theta_dot(:, parameter_index), l2_dot, hvp, scalar_hvp, status)
                if (status%code /= FORTNUM_OK) return
                first_dot(:, parameter_index) = self%beta1*first_dot(:, parameter_index) &
                    + (1.0_dp-self%beta1)*hvp
                second_dot(:, parameter_index) = self%beta2*second_dot(:, parameter_index) &
                    + (1.0_dp-self%beta2)*2.0_dp*raw_gradient*hvp
            end do
            do parameter_index = 1, MLP_ADAMW_HYPERPARAMETER_COUNT
                learning_rate_dot = 0.0_dp
                weight_decay_dot = 0.0_dp
                if (parameter_index == MLP_ADAMW_LOG_LEARNING_RATE) then
                    learning_rate_dot = learning_rate
                else if (parameter_index == MLP_ADAMW_LOG_WEIGHT_DECAY) then
                    weight_decay_dot = weight_decay
                end if
                first_hat = 1.0_dp/(1.0_dp-self%beta1**step)
                second_hat = 1.0_dp/(1.0_dp-self%beta2**step)
                first_hat_dot = first_dot(:, parameter_index)*first_hat
                second_hat_dot = second_dot(:, parameter_index)*second_hat
                denominator = sqrt(second*second_hat) + self%epsilon
                sqrt_second = sqrt(second*second_hat)
                update = (first*first_hat)/denominator
                update_dot = first_hat_dot/denominator
                where (sqrt_second > 0.0_dp)
                    update_dot = update_dot - (first*first_hat)* &
                        (0.5_dp*second_hat_dot/sqrt_second)/(denominator*denominator)
                end where
                theta_dot(:, parameter_index) = (1.0_dp-learning_rate*weight_decay) * &
                    theta_dot(:, parameter_index) - theta*(learning_rate_dot*weight_decay + &
                    learning_rate*weight_decay_dot) - learning_rate_dot*update - &
                    learning_rate*update_dot
            end do
            theta = (1.0_dp-learning_rate*weight_decay)*theta - &
                learning_rate*(first/(1.0_dp-self%beta1**step))/ &
                (sqrt(second/(1.0_dp-self%beta2**step)) + self%epsilon)
            if (any(.not. ieee_is_finite(theta)) .or. &
                any(.not. ieee_is_finite(theta_dot))) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP AdamW hypergradient: trajectory is not finite")
                return
            end if
        end do
        call self%model%set_parameters(theta, status)
        if (status%code /= FORTNUM_OK) return
        allocate(validation_gradient(n_parameters))
        allocate(zero_input(size(self%validation_x, 1), size(self%validation_x, 2)))
        call mlp_loss_value_gradient(self%model, self%validation_x, &
            self%validation_target, 0.0_dp, value, validation_gradient, &
            l2_gradient, status)
        if (status%code /= FORTNUM_OK) return
        do parameter_index = 1, MLP_ADAMW_HYPERPARAMETER_COUNT
            gradient(parameter_index) = dot_product(validation_gradient, &
                theta_dot(:, parameter_index))
        end do
        tangent = dot_product(gradient, direction)
        if (.not. ieee_is_finite(value) .or. any(.not. ieee_is_finite(gradient)) .or. &
            .not. ieee_is_finite(tangent)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP AdamW hypergradient: objective product is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine adamw_forward

    subroutine reverse_value_gradient(self, parameters, value, gradient, status)
        class(mlp_hypergradient_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: theta_history(:, :), gradient_history(:, :)
        real(dp), allocatable :: theta_bar(:), gradient_theta(:), hvp(:)
        real(dp), allocatable :: zero_input(:, :)
        real(dp) :: learning_rate, l2, train_value, l2_gradient
        real(dp) :: validation_l2_gradient, direct_learning_rate
        real(dp) :: direct_l2, scalar_hvp
        integer :: n_parameters, step

        value = huge(1.0_dp)
        gradient = 0.0_dp
        if (.not. finite_logs(parameters, learning_rate, l2)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP hypergradient: log hyperparameters are invalid")
            return
        end if
        n_parameters = size(self%initial_parameters)
        allocate(theta_history(n_parameters, self%layout%inner_steps + 1))
        allocate(gradient_history(n_parameters, self%layout%inner_steps))
        theta_history(:, 1) = self%initial_parameters
        allocate(gradient_theta(n_parameters), hvp(n_parameters))
        do step = 1, self%layout%inner_steps
            call self%model%set_parameters(theta_history(:, step), status)
            if (status%code /= FORTNUM_OK) return
            call mlp_loss_value_gradient(self%model, self%train_x, &
                self%train_target, l2, train_value, gradient_theta, &
                l2_gradient, status)
            if (status%code /= FORTNUM_OK) return
            gradient_history(:, step) = gradient_theta
            theta_history(:, step + 1) = theta_history(:, step) - &
                learning_rate*gradient_theta
            if (any(.not. ieee_is_finite(theta_history(:, step + 1)))) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP hypergradient: inner trajectory is not finite")
                return
            end if
        end do

        call self%model%set_parameters(theta_history(:, self%layout%inner_steps + 1), &
            status)
        if (status%code /= FORTNUM_OK) return
        allocate(zero_input(size(self%validation_x, 1), size(self%validation_x, 2)))
        call mlp_loss_value_gradient(self%model, self%validation_x, &
            self%validation_target, 0.0_dp, value, gradient_theta, &
            validation_l2_gradient, status)
        if (status%code /= FORTNUM_OK) return
        theta_bar = gradient_theta
        do step = self%layout%inner_steps, 1, -1
            call self%model%set_parameters(theta_history(:, step), status)
            if (status%code /= FORTNUM_OK) return
            call mlp_loss_hvp(self%model, self%train_x, self%train_target, l2, &
                theta_bar, 0.0_dp, hvp, scalar_hvp, status)
            if (status%code /= FORTNUM_OK) return
            direct_learning_rate = -dot_product(theta_bar, gradient_history(:, step))&
                *learning_rate
            direct_l2 = -dot_product(theta_bar, theta_history(:, step))*learning_rate*l2
            gradient(MLP_LOG_LEARNING_RATE) = &
                gradient(MLP_LOG_LEARNING_RATE) + direct_learning_rate
            gradient(MLP_LOG_L2) = gradient(MLP_LOG_L2) + direct_l2
            theta_bar = theta_bar - learning_rate*hvp
        end do
        if (any(.not. ieee_is_finite(gradient)) .or. &
            .not. ieee_is_finite(value)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP hypergradient: objective product is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine reverse_value_gradient

    subroutine forward_jvp(self, parameters, direction, value, tangent, status)
        class(mlp_hypergradient_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:), direction(:)
        real(dp), intent(out) :: value, tangent
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: theta(:), theta_dot(:), gradient(:), hvp(:)
        real(dp), allocatable :: zero_input(:, :)
        real(dp) :: learning_rate, l2, learning_rate_dot, l2_dot
        real(dp) :: train_value, l2_gradient, validation_l2_gradient
        integer :: n_parameters, step

        value = huge(1.0_dp)
        tangent = 0.0_dp
        if (.not. finite_logs(parameters, learning_rate, l2)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP hypergradient JVP: log hyperparameters are invalid")
            return
        end if
        learning_rate_dot = learning_rate*direction(MLP_LOG_LEARNING_RATE)
        l2_dot = l2*direction(MLP_LOG_L2)
        n_parameters = size(self%initial_parameters)
        allocate(theta, source=self%initial_parameters)
        allocate(theta_dot(n_parameters), gradient(n_parameters), hvp(n_parameters))
        theta_dot = 0.0_dp
        allocate(zero_input(size(self%train_x, 1), size(self%train_x, 2)))
        do step = 1, self%layout%inner_steps
            call self%model%set_parameters(theta, status)
            if (status%code /= FORTNUM_OK) return
            call mlp_loss_value_gradient(self%model, self%train_x, &
                self%train_target, l2, train_value, gradient, l2_gradient, status)
            if (status%code /= FORTNUM_OK) return
            call mlp_loss_hvp(self%model, self%train_x, self%train_target, l2, &
                theta_dot, l2_dot, hvp, l2_gradient, status)
            if (status%code /= FORTNUM_OK) return
            theta_dot = theta_dot - learning_rate*hvp - learning_rate_dot*gradient
            theta = theta - learning_rate*gradient
            if (any(.not. ieee_is_finite(theta)) .or. &
                any(.not. ieee_is_finite(theta_dot))) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP hypergradient JVP: inner tangent is not finite")
                return
            end if
        end do
        call self%model%set_parameters(theta, status)
        if (status%code /= FORTNUM_OK) return
        call mlp_loss_value_gradient(self%model, self%validation_x, &
            self%validation_target, 0.0_dp, value, gradient, &
            validation_l2_gradient, status)
        if (status%code /= FORTNUM_OK) return
        tangent = dot_product(gradient, theta_dot)
        if (.not. ieee_is_finite(value) .or. .not. ieee_is_finite(tangent)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP hypergradient JVP: objective product is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine forward_jvp

    logical function valid_parameter_vector(parameters, gradient) result(valid)
        real(dp), intent(in) :: parameters(:), gradient(:)

        valid = size(parameters) == MLP_HYPERPARAMETER_COUNT .and. &
            size(gradient) == MLP_HYPERPARAMETER_COUNT .and. &
            all(ieee_is_finite(parameters))
    end function valid_parameter_vector

    logical function finite_logs(parameters, learning_rate, l2) result(valid)
        real(dp), intent(in) :: parameters(:)
        real(dp), intent(out) :: learning_rate, l2

        learning_rate = 0.0_dp
        l2 = 0.0_dp
        valid = size(parameters) == MLP_HYPERPARAMETER_COUNT .and. &
            all(ieee_is_finite(parameters))
        if (.not. valid) return
        learning_rate = exp(parameters(MLP_LOG_LEARNING_RATE))
        l2 = exp(parameters(MLP_LOG_L2))
        valid = ieee_is_finite(learning_rate) .and. ieee_is_finite(l2) .and. &
            learning_rate > 0.0_dp .and. l2 > 0.0_dp
    end function finite_logs

    logical function finite_adamw_logs(parameters, learning_rate, l2, weight_decay) &
            result(valid)
        real(dp), intent(in) :: parameters(:)
        real(dp), intent(out) :: learning_rate, l2, weight_decay

        learning_rate = 0.0_dp
        l2 = 0.0_dp
        weight_decay = 0.0_dp
        valid = size(parameters) == MLP_ADAMW_HYPERPARAMETER_COUNT .and. &
            all(ieee_is_finite(parameters))
        if (.not. valid) return
        learning_rate = exp(parameters(MLP_ADAMW_LOG_LEARNING_RATE))
        l2 = exp(parameters(MLP_ADAMW_LOG_L2))
        weight_decay = exp(parameters(MLP_ADAMW_LOG_WEIGHT_DECAY))
        valid = ieee_is_finite(learning_rate) .and. ieee_is_finite(l2) .and. &
            ieee_is_finite(weight_decay) .and. learning_rate > 0.0_dp .and. &
            l2 > 0.0_dp .and. weight_decay > 0.0_dp
    end function finite_adamw_logs

    logical function valid_adamw_options(options) result(valid)
        type(mlp_adamw_hypergradient_options_t), intent(in) :: options

        valid = options%steps >= 1 .and. options%optimizer == MLP_OPTIMIZER_ADAMW .and. &
            options%device_kind == FORTML_DEVICE_CPU .and. &
            ieee_is_finite(options%learning_rate) .and. ieee_is_finite(options%l2) .and. &
            ieee_is_finite(options%weight_decay) .and. options%learning_rate > 0.0_dp .and. &
            options%l2 > 0.0_dp .and. options%weight_decay > 0.0_dp .and. &
            ieee_is_finite(options%beta1) .and. ieee_is_finite(options%beta2) .and. &
            ieee_is_finite(options%epsilon) .and. options%beta1 >= 0.0_dp .and. &
            options%beta1 < 1.0_dp .and. options%beta2 >= 0.0_dp .and. &
            options%beta2 < 1.0_dp .and. options%epsilon > 0.0_dp .and. &
            ieee_is_finite(options%lower_log_learning_rate) .and. &
            ieee_is_finite(options%upper_log_learning_rate) .and. &
            ieee_is_finite(options%lower_log_l2) .and. ieee_is_finite(options%upper_log_l2) .and. &
            ieee_is_finite(options%lower_log_weight_decay) .and. &
            ieee_is_finite(options%upper_log_weight_decay) .and. &
            options%lower_log_learning_rate <= options%upper_log_learning_rate .and. &
            options%lower_log_l2 <= options%upper_log_l2 .and. &
            options%lower_log_weight_decay <= options%upper_log_weight_decay .and. &
            log(options%learning_rate) >= options%lower_log_learning_rate .and. &
            log(options%learning_rate) <= options%upper_log_learning_rate .and. &
            log(options%l2) >= options%lower_log_l2 .and. log(options%l2) <= options%upper_log_l2 .and. &
            log(options%weight_decay) >= options%lower_log_weight_decay .and. &
            log(options%weight_decay) <= options%upper_log_weight_decay .and. &
            options%memory >= 1 .and. options%max_iterations >= 1 .and. &
            options%max_line_search >= 1 .and. ieee_is_finite(options%gradient_tolerance) .and. &
            ieee_is_finite(options%step_tolerance) .and. ieee_is_finite(options%objective_tolerance) .and. &
            options%gradient_tolerance >= 0.0_dp .and. options%step_tolerance >= 0.0_dp .and. &
            options%objective_tolerance >= 0.0_dp
    end function valid_adamw_options

    logical function valid_options(options) result(valid)
        type(mlp_hypergradient_options_t), intent(in) :: options

        valid = options%steps >= 1 .and. options%optimizer == MLP_OPTIMIZER_SGD .and. &
            options%momentum == 0.0_dp .and. .not. options%nesterov .and. &
            options%device_kind == FORTML_DEVICE_CPU .and. &
            ieee_is_finite(options%learning_rate) .and. &
            ieee_is_finite(options%l2) .and. options%learning_rate > 0.0_dp .and. &
            options%l2 > 0.0_dp .and. &
            ieee_is_finite(options%lower_log_learning_rate) .and. &
            ieee_is_finite(options%upper_log_learning_rate) .and. &
            ieee_is_finite(options%lower_log_l2) .and. &
            ieee_is_finite(options%upper_log_l2) .and. &
            options%lower_log_learning_rate <= options%upper_log_learning_rate .and. &
            options%lower_log_l2 <= options%upper_log_l2 .and. &
            log(options%learning_rate) >= options%lower_log_learning_rate .and. &
            log(options%learning_rate) <= options%upper_log_learning_rate .and. &
            log(options%l2) >= options%lower_log_l2 .and. &
            log(options%l2) <= options%upper_log_l2 .and. &
            options%memory >= 1 .and. options%max_iterations >= 1 .and. &
            options%max_line_search >= 1 .and. &
            ieee_is_finite(options%gradient_tolerance) .and. &
            ieee_is_finite(options%step_tolerance) .and. &
            ieee_is_finite(options%objective_tolerance) .and. &
            options%gradient_tolerance >= 0.0_dp .and. &
            options%step_tolerance >= 0.0_dp .and. &
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
        if (any(.not. ieee_is_finite(x)) .or. &
            any(.not. ieee_is_finite(target))) return
        valid = .true.
    end function valid_data

end module fortml_mlp_hypergradient
