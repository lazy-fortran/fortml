module fortml_mlp_amsgrad_hypergradient
    !! Exact products through a fixed full-batch AMSGrad trajectory.
    !!
    !! The packed outer coordinates are
    !! `[log(learning_rate), log(l2), logit(beta1), logit(beta2),
    !!   log(epsilon)]`.  The objective differentiates the complete
    !! bias-corrected first/second moments and the elementwise maximum second
    !! moment.  It does not finite-difference an inner training run.  AMSGrad's
    !! max active-set boundary and a zero second-moment square root are
    !! mathematically nonsmooth and return a typed refusal.  CUDA requests
    !! likewise return `FORTNUM_NOT_IMPLEMENTED` until a resident trajectory
    !! state is available.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED, FORTNUM_CONVERGENCE_ERROR
    use fortml_device, only: FORTML_DEVICE_CPU
    use fortml_mlp, only: mlp_t
    use fortml_mlp_training, only: mlp_loss_value_gradient, mlp_loss_hvp, &
        MLP_OPTIMIZER_AMSGRAD
    use fortopt_objective, only: objective_t
    use fortopt_lbfgsb, only: lbfgsb_t, lbfgsb_options_t, lbfgsb_result_t
    implicit none
    private

    integer, parameter, public :: MLP_AMSGRAD_HYPERPARAMETER_COUNT = 5
    integer, parameter, public :: MLP_AMSGRAD_LOG_LEARNING_RATE = 1
    integer, parameter, public :: MLP_AMSGRAD_LOG_L2 = 2
    integer, parameter, public :: MLP_AMSGRAD_LOGIT_BETA1 = 3
    integer, parameter, public :: MLP_AMSGRAD_LOGIT_BETA2 = 4
    integer, parameter, public :: MLP_AMSGRAD_LOG_EPSILON = 5

    type, public :: mlp_amsgrad_hypergradient_metadata_t
        integer :: parameter_count = MLP_AMSGRAD_HYPERPARAMETER_COUNT
        integer :: log_learning_rate_index = MLP_AMSGRAD_LOG_LEARNING_RATE
        integer :: log_l2_index = MLP_AMSGRAD_LOG_L2
        integer :: logit_beta1_index = MLP_AMSGRAD_LOGIT_BETA1
        integer :: logit_beta2_index = MLP_AMSGRAD_LOGIT_BETA2
        integer :: log_epsilon_index = MLP_AMSGRAD_LOG_EPSILON
        integer :: inner_steps = 0
        real(dp) :: max_boundary_tolerance = 0.0_dp
    end type mlp_amsgrad_hypergradient_metadata_t

    type, public :: mlp_amsgrad_hypergradient_options_t
        !! Fixed full-batch AMSGrad trajectory configuration.
        integer :: steps = 8
        real(dp) :: learning_rate = 1.0e-2_dp
        real(dp) :: l2 = 1.0e-4_dp
        real(dp) :: beta1 = 0.9_dp
        real(dp) :: beta2 = 0.999_dp
        real(dp) :: epsilon = 1.0e-8_dp
        !! Products are refused when the second-moment max active set is this
        !! close to changing.  The scale is `max(1, abs(a), abs(b))`.
        real(dp) :: max_boundary_tolerance = 1.0e-10_dp
        real(dp) :: lower_log_learning_rate = -12.0_dp
        real(dp) :: upper_log_learning_rate = 2.0_dp
        real(dp) :: lower_log_l2 = -20.0_dp
        real(dp) :: upper_log_l2 = 2.0_dp
        real(dp) :: lower_logit_beta1 = -12.0_dp
        real(dp) :: upper_logit_beta1 = 12.0_dp
        real(dp) :: lower_logit_beta2 = -12.0_dp
        real(dp) :: upper_logit_beta2 = 12.0_dp
        real(dp) :: lower_log_epsilon = -30.0_dp
        real(dp) :: upper_log_epsilon = 2.0_dp
        integer :: optimizer = MLP_OPTIMIZER_AMSGRAD
        integer :: device_kind = FORTML_DEVICE_CPU
        integer :: memory = 8
        integer :: max_iterations = 100
        integer :: max_line_search = 40
        real(dp) :: gradient_tolerance = 1.0e-6_dp
        real(dp) :: step_tolerance = 1.0e-12_dp
        real(dp) :: objective_tolerance = 1.0e-12_dp
    end type mlp_amsgrad_hypergradient_options_t

    type, public :: mlp_amsgrad_hypergradient_result_t
        logical :: converged = .false.
        integer :: iterations = 0
        integer :: line_search_evaluations = 0
        real(dp) :: objective = huge(1.0_dp)
        real(dp) :: gradient_norm = huge(1.0_dp)
        real(dp) :: log_learning_rate = 0.0_dp
        real(dp) :: log_l2 = 0.0_dp
        real(dp) :: logit_beta1 = 0.0_dp
        real(dp) :: logit_beta2 = 0.0_dp
        real(dp) :: log_epsilon = 0.0_dp
        real(dp) :: learning_rate = 0.0_dp
        real(dp) :: l2 = 0.0_dp
        real(dp) :: beta1 = 0.0_dp
        real(dp) :: beta2 = 0.0_dp
        real(dp) :: epsilon = 0.0_dp
    end type mlp_amsgrad_hypergradient_result_t

    type, public :: mlp_amsgrad_hypergradient_objective_t
        !! FortOpt-compatible validation objective through AMSGrad state.
        private
        type(mlp_t), pointer :: model => null()
        real(dp), allocatable :: train_x(:, :), train_target(:, :)
        real(dp), allocatable :: validation_x(:, :), validation_target(:, :)
        real(dp), allocatable :: initial_parameters(:)
        type(mlp_amsgrad_hypergradient_metadata_t) :: layout
        real(dp) :: initial_log_learning_rate = 0.0_dp
        real(dp) :: initial_log_l2 = 0.0_dp
        real(dp) :: initial_logit_beta1 = 0.0_dp
        real(dp) :: initial_logit_beta2 = 0.0_dp
        real(dp) :: initial_log_epsilon = 0.0_dp
        logical :: initialized = .false.
    contains
        procedure, public :: initialize => mlp_amsgrad_hypergradient_initialize
        procedure, public :: parameter_count => mlp_amsgrad_hypergradient_parameter_count
        procedure, public :: metadata => mlp_amsgrad_hypergradient_metadata
        procedure, public :: parameters => mlp_amsgrad_hypergradient_parameters
        procedure, public :: value_gradient => mlp_amsgrad_hypergradient_value_gradient
        procedure, public :: jvp => mlp_amsgrad_hypergradient_jvp
        procedure, public :: vjp => mlp_amsgrad_hypergradient_vjp
        procedure, public :: fortopt => mlp_amsgrad_hypergradient_fortopt
        procedure, public :: is_initialized => mlp_amsgrad_hypergradient_is_initialized
    end type mlp_amsgrad_hypergradient_objective_t

    public :: mlp_optimize_amsgrad_hyperparameters

contains

    subroutine mlp_amsgrad_hypergradient_initialize(self, model, train_x, train_target, &
            validation_x, validation_target, options, status)
        class(mlp_amsgrad_hypergradient_objective_t), intent(out) :: self
        type(mlp_t), target, intent(inout) :: model
        real(dp), intent(in) :: train_x(:, :), train_target(:, :)
        real(dp), intent(in) :: validation_x(:, :), validation_target(:, :)
        type(mlp_amsgrad_hypergradient_options_t), intent(in) :: options
        type(fortnum_status_t), intent(out) :: status
        !! Default-initialized instances, standing in for empty
        !! structure constructors: nvfortran rejects `T()` outright,
        !! and a declared local carries the same default init.
        type(mlp_amsgrad_hypergradient_metadata_t) :: mlp_amsgrad_hypergradient_metadata_t_default

        self%initialized = .false.
        self%layout = mlp_amsgrad_hypergradient_metadata_t_default
        if (.not. valid_options(options)) then
            if (options%optimizer /= MLP_OPTIMIZER_AMSGRAD .or. &
                options%device_kind /= FORTML_DEVICE_CPU) then
                call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                    "MLP AMSGrad hypergradient: optimizer or device is unsupported")
            else
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP AMSGrad hypergradient: options are invalid")
            end if
            return
        end if
        if (.not. valid_data(model, train_x, train_target)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP AMSGrad hypergradient: training model or data is invalid")
            return
        end if
        if (.not. valid_data(model, validation_x, validation_target)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP AMSGrad hypergradient: validation model or data is invalid")
            return
        end if

        self%model => model
        allocate(self%train_x, source=train_x)
        allocate(self%train_target, source=train_target)
        allocate(self%validation_x, source=validation_x)
        allocate(self%validation_target, source=validation_target)
        allocate(self%initial_parameters, source=model%parameters())
        self%layout%inner_steps = options%steps
        self%layout%max_boundary_tolerance = options%max_boundary_tolerance
        self%initial_log_learning_rate = log(options%learning_rate)
        self%initial_log_l2 = log(options%l2)
        self%initial_logit_beta1 = probability_logit(options%beta1)
        self%initial_logit_beta2 = probability_logit(options%beta2)
        self%initial_log_epsilon = log(options%epsilon)
        self%initialized = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_amsgrad_hypergradient_initialize

    integer function mlp_amsgrad_hypergradient_parameter_count(self) result(count)
        class(mlp_amsgrad_hypergradient_objective_t), intent(in) :: self

        count = 0
        if (self%initialized) count = self%layout%parameter_count
    end function mlp_amsgrad_hypergradient_parameter_count

    function mlp_amsgrad_hypergradient_metadata(self) result(layout)
        class(mlp_amsgrad_hypergradient_objective_t), intent(in) :: self
        type(mlp_amsgrad_hypergradient_metadata_t) :: layout

        layout = self%layout
    end function mlp_amsgrad_hypergradient_metadata

    function mlp_amsgrad_hypergradient_parameters(self) result(parameters)
        class(mlp_amsgrad_hypergradient_objective_t), intent(in) :: self
        real(dp), allocatable :: parameters(:)

        allocate(parameters(MLP_AMSGRAD_HYPERPARAMETER_COUNT))
        parameters = [self%initial_log_learning_rate, self%initial_log_l2, &
            self%initial_logit_beta1, self%initial_logit_beta2, self%initial_log_epsilon]
    end function mlp_amsgrad_hypergradient_parameters

    logical function mlp_amsgrad_hypergradient_is_initialized(self) result(yes)
        class(mlp_amsgrad_hypergradient_objective_t), intent(in) :: self

        yes = self%initialized
        if (.not. yes) return
        yes = associated(self%model)
        if (.not. yes) return
        yes = allocated(self%initial_parameters)
    end function mlp_amsgrad_hypergradient_is_initialized

    subroutine mlp_amsgrad_hypergradient_value_gradient(self, parameters, value, &
            gradient, status)
        class(mlp_amsgrad_hypergradient_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: direction(MLP_AMSGRAD_HYPERPARAMETER_COUNT), tangent

        value = huge(1.0_dp)
        gradient = 0.0_dp
        direction = 0.0_dp
        if (.not. self%is_initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP AMSGrad hypergradient: objective is not initialized")
            return
        end if
        if (size(parameters) /= MLP_AMSGRAD_HYPERPARAMETER_COUNT) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP AMSGrad hypergradient: packed parameter shape is invalid")
            return
        end if
        if (size(gradient) /= MLP_AMSGRAD_HYPERPARAMETER_COUNT) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP AMSGrad hypergradient: packed gradient shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(parameters))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP AMSGrad hypergradient: packed parameters are not finite")
            return
        end if
        call amsgrad_forward(self, parameters, direction, value, tangent, gradient, status)
    end subroutine mlp_amsgrad_hypergradient_value_gradient

    subroutine mlp_amsgrad_hypergradient_jvp(self, parameters, direction, value, &
            tangent, status)
        class(mlp_amsgrad_hypergradient_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:), direction(:)
        real(dp), intent(out) :: value, tangent
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: gradient(MLP_AMSGRAD_HYPERPARAMETER_COUNT)

        value = huge(1.0_dp)
        tangent = 0.0_dp
        gradient = 0.0_dp
        if (.not. self%is_initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP AMSGrad hypergradient JVP: objective is not initialized")
            return
        end if
        if (size(parameters) /= MLP_AMSGRAD_HYPERPARAMETER_COUNT .or. &
            size(direction) /= MLP_AMSGRAD_HYPERPARAMETER_COUNT) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP AMSGrad hypergradient JVP: packed shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(parameters)) .or. &
            any(.not. ieee_is_finite(direction))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP AMSGrad hypergradient JVP: packed values are not finite")
            return
        end if
        call amsgrad_forward(self, parameters, direction, value, tangent, gradient, status)
    end subroutine mlp_amsgrad_hypergradient_jvp

    subroutine mlp_amsgrad_hypergradient_vjp(self, parameters, output_bar, gradient, &
            status)
        class(mlp_amsgrad_hypergradient_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:), output_bar
        real(dp), intent(out) :: gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: value

        gradient = 0.0_dp
        if (.not. ieee_is_finite(output_bar)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP AMSGrad hypergradient VJP: cotangent is invalid")
            return
        end if
        call self%value_gradient(parameters, value, gradient, status)
        if (status%code /= FORTNUM_OK) return
        gradient = output_bar*gradient
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_amsgrad_hypergradient_vjp

    subroutine mlp_amsgrad_hypergradient_fortopt(self, objective, status)
        class(mlp_amsgrad_hypergradient_objective_t), target, intent(inout) :: self
        type(objective_t), intent(out) :: objective
        type(fortnum_status_t), intent(out) :: status

        if (.not. self%is_initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP AMSGrad hypergradient: objective is not initialized")
            return
        end if
        call objective%initialize_context(MLP_AMSGRAD_HYPERPARAMETER_COUNT, self, &
            mlp_amsgrad_hypergradient_context_callback, status)
    end subroutine mlp_amsgrad_hypergradient_fortopt

    subroutine mlp_amsgrad_hypergradient_context_callback(context, parameters, value, &
            gradient, status)
        class(*), intent(inout) :: context
        real(dp), intent(in) :: parameters(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status

        select type (adapter => context)
            type is (mlp_amsgrad_hypergradient_objective_t)
            call adapter%value_gradient(parameters, value, gradient, status)
        class default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP AMSGrad hypergradient: context has the wrong type")
        end select
    end subroutine mlp_amsgrad_hypergradient_context_callback

    subroutine mlp_optimize_amsgrad_hyperparameters(model, train_x, train_target, &
            validation_x, validation_target, options, result, status)
        !! Optimize AMSGrad trajectory hyperparameters with FortOpt L-BFGS-B.
        type(mlp_t), target, intent(inout) :: model
        real(dp), intent(in) :: train_x(:, :), train_target(:, :)
        real(dp), intent(in) :: validation_x(:, :), validation_target(:, :)
        type(mlp_amsgrad_hypergradient_options_t), intent(in) :: options
        type(mlp_amsgrad_hypergradient_result_t), intent(out) :: result
        type(fortnum_status_t), intent(out) :: status
        type(mlp_amsgrad_hypergradient_objective_t), target :: adapter
        type(objective_t) :: objective
        type(lbfgsb_t) :: optimizer
        type(lbfgsb_options_t) :: optimizer_options
        type(lbfgsb_result_t) :: optimizer_result
        real(dp) :: parameters(MLP_AMSGRAD_HYPERPARAMETER_COUNT)
        real(dp) :: lower(MLP_AMSGRAD_HYPERPARAMETER_COUNT)
        real(dp) :: upper(MLP_AMSGRAD_HYPERPARAMETER_COUNT)
        real(dp) :: gradient(MLP_AMSGRAD_HYPERPARAMETER_COUNT)
        !! Default-initialized instances, standing in for empty
        !! structure constructors: nvfortran rejects `T()` outright,
        !! and a declared local carries the same default init.
        type(mlp_amsgrad_hypergradient_result_t) :: mlp_amsgrad_hypergradient_result_t_default

        result = mlp_amsgrad_hypergradient_result_t_default
        if (.not. valid_options(options)) then
            if (options%optimizer /= MLP_OPTIMIZER_AMSGRAD .or. &
                options%device_kind /= FORTML_DEVICE_CPU) then
                call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                    "MLP AMSGrad hyperparameter optimization: unsupported optimizer/device")
            else
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP AMSGrad hyperparameter optimization: options are invalid")
            end if
            return
        end if
        call adapter%initialize(model, train_x, train_target, validation_x, &
            validation_target, options, status)
        if (status%code /= FORTNUM_OK) return
        parameters = adapter%parameters()
        lower = [options%lower_log_learning_rate, options%lower_log_l2, &
            options%lower_logit_beta1, options%lower_logit_beta2, options%lower_log_epsilon]
        upper = [options%upper_log_learning_rate, options%upper_log_l2, &
            options%upper_logit_beta1, options%upper_logit_beta2, options%upper_log_epsilon]
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
        result%log_learning_rate = parameters(MLP_AMSGRAD_LOG_LEARNING_RATE)
        result%log_l2 = parameters(MLP_AMSGRAD_LOG_L2)
        result%logit_beta1 = parameters(MLP_AMSGRAD_LOGIT_BETA1)
        result%logit_beta2 = parameters(MLP_AMSGRAD_LOGIT_BETA2)
        result%log_epsilon = parameters(MLP_AMSGRAD_LOG_EPSILON)
        result%learning_rate = exp(result%log_learning_rate)
        result%l2 = exp(result%log_l2)
        result%beta1 = logit_to_probability(result%logit_beta1)
        result%beta2 = logit_to_probability(result%logit_beta2)
        result%epsilon = exp(result%log_epsilon)
        if (.not. result%converged) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "MLP AMSGrad hyperparameter optimization: iteration limit reached")
            return
        end if
        if (.not. ieee_is_finite(result%objective) .or. &
            .not. ieee_is_finite(result%gradient_norm)) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "MLP AMSGrad hyperparameter optimization: result is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_optimize_amsgrad_hyperparameters

    subroutine amsgrad_forward(self, parameters, direction, value, tangent, gradient, status)
        !! Forward-mode propagation of all five packed hyperparameter tangents.
        class(mlp_amsgrad_hypergradient_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:), direction(:)
        real(dp), intent(out) :: value, tangent, gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: theta(:), theta_dot(:, :)
        real(dp), allocatable :: first(:), second(:), max_second(:)
        real(dp), allocatable :: first_dot(:, :), second_dot(:, :), max_second_dot(:, :)
        real(dp), allocatable :: raw_gradient(:), gradient_dot(:), hvp(:)
        real(dp), allocatable :: first_hat(:), second_hat(:), first_hat_dot(:), second_hat_dot(:)
        real(dp), allocatable :: sqrt_second(:), denominator(:), denominator_dot(:)
        real(dp), allocatable :: update(:), update_dot(:), validation_gradient(:)
        real(dp) :: learning_rate, l2, beta1, beta2, epsilon
        real(dp) :: learning_rate_dot, l2_dot, beta1_dot, beta2_dot, epsilon_dot
        real(dp), allocatable :: first_previous(:), second_previous(:), max_previous(:)
        real(dp), allocatable :: max_previous_dot(:, :)
        real(dp) :: c1, c2, c1_dot, c2_dot
        real(dp) :: train_value, l2_gradient, scalar_hvp, active_scale
        integer :: n_parameters, step, parameter_index, parameter_index_inner

        value = huge(1.0_dp)
        tangent = 0.0_dp
        gradient = 0.0_dp
        if (.not. finite_parameters(parameters, learning_rate, l2, beta1, beta2, epsilon)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP AMSGrad hypergradient: packed parameters are invalid")
            return
        end if
        n_parameters = size(self%initial_parameters)
        allocate(theta, source=self%initial_parameters)
        allocate(theta_dot(n_parameters, MLP_AMSGRAD_HYPERPARAMETER_COUNT))
        allocate(first(n_parameters), second(n_parameters), max_second(n_parameters))
        allocate(first_dot(n_parameters, MLP_AMSGRAD_HYPERPARAMETER_COUNT), &
            second_dot(n_parameters, MLP_AMSGRAD_HYPERPARAMETER_COUNT), &
            max_second_dot(n_parameters, MLP_AMSGRAD_HYPERPARAMETER_COUNT))
        allocate(raw_gradient(n_parameters), gradient_dot(n_parameters), hvp(n_parameters))
        allocate(first_hat(n_parameters), second_hat(n_parameters), &
            first_hat_dot(n_parameters), second_hat_dot(n_parameters))
        allocate(sqrt_second(n_parameters), denominator(n_parameters), &
            denominator_dot(n_parameters), update(n_parameters), update_dot(n_parameters))
        allocate(first_previous(n_parameters), second_previous(n_parameters), &
            max_previous(n_parameters))
        allocate(max_previous_dot(n_parameters, MLP_AMSGRAD_HYPERPARAMETER_COUNT))
        theta_dot = 0.0_dp
        first = 0.0_dp
        second = 0.0_dp
        max_second = 0.0_dp
        first_dot = 0.0_dp
        second_dot = 0.0_dp
        max_second_dot = 0.0_dp

        do step = 1, self%layout%inner_steps
            call self%model%set_parameters(theta, status)
            if (status%code /= FORTNUM_OK) return
            call mlp_loss_value_gradient(self%model, self%train_x, self%train_target, &
                l2, train_value, raw_gradient, l2_gradient, status)
            if (status%code /= FORTNUM_OK) return
            first_previous = first
            second_previous = second
            max_previous = max_second
            max_previous_dot = max_second_dot
            first = beta1*first_previous + (1.0_dp-beta1)*raw_gradient
            second = beta2*second_previous + (1.0_dp-beta2)*raw_gradient*raw_gradient
            do parameter_index = 1, MLP_AMSGRAD_HYPERPARAMETER_COUNT
                l2_dot = 0.0_dp
                beta1_dot = 0.0_dp
                beta2_dot = 0.0_dp
                if (parameter_index == MLP_AMSGRAD_LOG_L2) l2_dot = l2
                if (parameter_index == MLP_AMSGRAD_LOGIT_BETA1) beta1_dot = beta1*(1.0_dp-beta1)
                if (parameter_index == MLP_AMSGRAD_LOGIT_BETA2) beta2_dot = beta2*(1.0_dp-beta2)
                call mlp_loss_hvp(self%model, self%train_x, self%train_target, l2, &
                    theta_dot(:, parameter_index), l2_dot, hvp, scalar_hvp, status)
                if (status%code /= FORTNUM_OK) return
                gradient_dot = hvp
                first_dot(:, parameter_index) = beta1*first_dot(:, parameter_index) + &
                    (1.0_dp-beta1)*gradient_dot + beta1_dot*(first_previous-raw_gradient)
                second_dot(:, parameter_index) = beta2*second_dot(:, parameter_index) + &
                    (1.0_dp-beta2)*2.0_dp*raw_gradient*gradient_dot + &
                    beta2_dot*(second_previous-raw_gradient*raw_gradient)
            end do

            c1 = 1.0_dp-beta1**step
            c2 = 1.0_dp-beta2**step
            if (c1 <= 0.0_dp .or. c2 <= 0.0_dp .or. &
                .not. ieee_is_finite(c1) .or. .not. ieee_is_finite(c2)) then
                call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                    "MLP AMSGrad hypergradient: zero bias-correction denominator")
                return
            end if
            max_second = max(max_previous, second)
            do parameter_index_inner = 1, n_parameters
                active_scale = max(1.0_dp, abs(max_previous(parameter_index_inner)), &
                    abs(second(parameter_index_inner)))
                if (abs(second(parameter_index_inner)-max_previous(parameter_index_inner)) <= &
                    self%layout%max_boundary_tolerance*active_scale) then
                    call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                        "MLP AMSGrad hypergradient: max-second-moment active-set tie")
                    return
                end if
            end do
            do parameter_index = 1, MLP_AMSGRAD_HYPERPARAMETER_COUNT
                where (second > max_previous)
                    max_second_dot(:, parameter_index) = second_dot(:, parameter_index)
                elsewhere
                    max_second_dot(:, parameter_index) = max_previous_dot(:, parameter_index)
                end where
            end do
            first_hat = first/c1
            second_hat = max_second/c2
            if (any(second_hat < 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP AMSGrad hypergradient: second moment is negative")
                return
            end if
            sqrt_second = sqrt(second_hat)
            if (any(sqrt_second == 0.0_dp)) then
                call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                    "MLP AMSGrad hypergradient: zero second-moment square-root derivative")
                return
            end if
            denominator = sqrt_second+epsilon
            if (any(denominator <= 0.0_dp) .or. any(.not. ieee_is_finite(denominator))) then
                call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                    "MLP AMSGrad hypergradient: zero update denominator")
                return
            end if
            update = first_hat/denominator
            do parameter_index = 1, MLP_AMSGRAD_HYPERPARAMETER_COUNT
                learning_rate_dot = 0.0_dp
                l2_dot = 0.0_dp
                beta1_dot = 0.0_dp
                beta2_dot = 0.0_dp
                epsilon_dot = 0.0_dp
                if (parameter_index == MLP_AMSGRAD_LOG_LEARNING_RATE) learning_rate_dot = learning_rate
                if (parameter_index == MLP_AMSGRAD_LOG_L2) l2_dot = l2
                if (parameter_index == MLP_AMSGRAD_LOGIT_BETA1) beta1_dot = beta1*(1.0_dp-beta1)
                if (parameter_index == MLP_AMSGRAD_LOGIT_BETA2) beta2_dot = beta2*(1.0_dp-beta2)
                if (parameter_index == MLP_AMSGRAD_LOG_EPSILON) epsilon_dot = epsilon
                c1_dot = -real(step,dp)*beta1**(step-1)*beta1_dot
                c2_dot = -real(step,dp)*beta2**(step-1)*beta2_dot
                first_hat_dot = first_dot(:, parameter_index)/c1 - &
                    first*c1_dot/(c1*c1)
                second_hat_dot = max_second_dot(:, parameter_index)/c2 - &
                    max_second*c2_dot/(c2*c2)
                denominator_dot = second_hat_dot/(2.0_dp*sqrt_second) + epsilon_dot
                update_dot = first_hat_dot/denominator - &
                    first_hat*denominator_dot/(denominator*denominator)
                theta_dot(:, parameter_index) = theta_dot(:, parameter_index) - &
                    learning_rate_dot*update-learning_rate*update_dot
            end do
            theta = theta-learning_rate*update
            if (any(.not. ieee_is_finite(theta)) .or. &
                any(.not. ieee_is_finite(theta_dot)) .or. &
                any(.not. ieee_is_finite(first)) .or. any(.not. ieee_is_finite(second)) .or. &
                any(.not. ieee_is_finite(max_second))) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP AMSGrad hypergradient: trajectory is not finite")
                return
            end if
        end do
        call self%model%set_parameters(theta, status)
        if (status%code /= FORTNUM_OK) return
        allocate(validation_gradient(n_parameters))
        call mlp_loss_value_gradient(self%model, self%validation_x, self%validation_target, &
            0.0_dp, value, validation_gradient, l2_gradient, status)
        if (status%code /= FORTNUM_OK) return
        do parameter_index = 1, MLP_AMSGRAD_HYPERPARAMETER_COUNT
            gradient(parameter_index) = dot_product(validation_gradient, &
                theta_dot(:, parameter_index))
        end do
        tangent = dot_product(gradient, direction)
        if (.not. ieee_is_finite(value) .or. any(.not. ieee_is_finite(gradient)) .or. &
            .not. ieee_is_finite(tangent)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP AMSGrad hypergradient: trajectory product is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine amsgrad_forward

    logical function finite_parameters(parameters, learning_rate, l2, beta1, beta2, epsilon) result(valid)
        real(dp), intent(in) :: parameters(:)
        real(dp), intent(out) :: learning_rate, l2, beta1, beta2, epsilon

        learning_rate = 0.0_dp
        l2 = 0.0_dp
        beta1 = 0.0_dp
        beta2 = 0.0_dp
        epsilon = 0.0_dp
        valid = size(parameters) == MLP_AMSGRAD_HYPERPARAMETER_COUNT
        if (.not. valid) return
        if (any(.not. ieee_is_finite(parameters))) then
            valid = .false.
            return
        end if
        learning_rate = exp(parameters(MLP_AMSGRAD_LOG_LEARNING_RATE))
        l2 = exp(parameters(MLP_AMSGRAD_LOG_L2))
        beta1 = logit_to_probability(parameters(MLP_AMSGRAD_LOGIT_BETA1))
        beta2 = logit_to_probability(parameters(MLP_AMSGRAD_LOGIT_BETA2))
        epsilon = exp(parameters(MLP_AMSGRAD_LOG_EPSILON))
        valid = ieee_is_finite(learning_rate) .and. ieee_is_finite(l2) .and. &
            ieee_is_finite(beta1) .and. ieee_is_finite(beta2) .and. ieee_is_finite(epsilon) .and. &
            learning_rate > 0.0_dp .and. l2 >= 0.0_dp .and. beta1 > 0.0_dp .and. beta1 < 1.0_dp .and. &
            beta2 > 0.0_dp .and. beta2 < 1.0_dp .and. epsilon > 0.0_dp
    end function finite_parameters

    logical function valid_options(options) result(valid)
        type(mlp_amsgrad_hypergradient_options_t), intent(in) :: options
        real(dp) :: logit1, logit2

        valid = .false.
        if (options%steps < 1) return
        if (options%optimizer /= MLP_OPTIMIZER_AMSGRAD) return
        if (options%device_kind /= FORTML_DEVICE_CPU) return
        if (options%learning_rate <= 0.0_dp .or. options%l2 < 0.0_dp .or. &
            options%beta1 <= 0.0_dp .or. options%beta1 >= 1.0_dp .or. &
            options%beta2 <= 0.0_dp .or. options%beta2 >= 1.0_dp .or. &
            options%epsilon <= 0.0_dp .or. options%max_boundary_tolerance < 0.0_dp) return
        if (.not. ieee_is_finite(options%learning_rate) .or. &
            .not. ieee_is_finite(options%l2) .or. .not. ieee_is_finite(options%beta1) .or. &
            .not. ieee_is_finite(options%beta2) .or. .not. ieee_is_finite(options%epsilon) .or. &
            .not. ieee_is_finite(options%max_boundary_tolerance)) return
        logit1 = probability_logit(options%beta1)
        logit2 = probability_logit(options%beta2)
        valid = bounds_valid(options%lower_log_learning_rate, options%upper_log_learning_rate, &
            log(options%learning_rate))
        if (.not. valid) return
        valid = bounds_valid(options%lower_log_l2, options%upper_log_l2, log(max(options%l2, tiny(1.0_dp))))
        if (options%l2 == 0.0_dp) valid = .false.
        if (.not. valid) return
        valid = bounds_valid(options%lower_logit_beta1, options%upper_logit_beta1, logit1)
        if (.not. valid) return
        valid = bounds_valid(options%lower_logit_beta2, options%upper_logit_beta2, logit2)
        if (.not. valid) return
        valid = bounds_valid(options%lower_log_epsilon, options%upper_log_epsilon, log(options%epsilon))
        if (.not. valid) return
        valid = options%memory >= 1 .and. options%max_iterations >= 1 .and. &
            options%max_line_search >= 1 .and. ieee_is_finite(options%gradient_tolerance) .and. &
            ieee_is_finite(options%step_tolerance) .and. ieee_is_finite(options%objective_tolerance) .and. &
            options%gradient_tolerance >= 0.0_dp .and. options%step_tolerance >= 0.0_dp .and. &
            options%objective_tolerance >= 0.0_dp
    end function valid_options

    logical function bounds_valid(lower, upper, value) result(valid)
        real(dp), intent(in) :: lower, upper, value

        valid = ieee_is_finite(lower) .and. ieee_is_finite(upper) .and. &
            lower <= upper .and. value >= lower .and. value <= upper
    end function bounds_valid

    logical function valid_data(model, x, target) result(valid)
        type(mlp_t), intent(in) :: model
        real(dp), intent(in) :: x(:, :), target(:, :)

        valid = .false.
        if (.not. allocated(model%layer_sizes)) return
        if (size(model%layer_sizes) < 2) return
        if (model%parameter_count() <= 0) return
        if (size(x, 1) < 1 .or. size(target, 1) /= size(x, 1)) return
        if (size(x, 2) /= model%layer_sizes(1)) return
        if (size(target, 2) /= model%layer_sizes(size(model%layer_sizes))) return
        if (any(.not. ieee_is_finite(x)) .or. any(.not. ieee_is_finite(target))) return
        valid = .true.
    end function valid_data

    pure real(dp) function logit_to_probability(logit) result(probability)
        real(dp), intent(in) :: logit

        if (logit >= 0.0_dp) then
            probability = 1.0_dp/(1.0_dp+exp(-logit))
        else
            probability = exp(logit)/(1.0_dp+exp(logit))
        end if
    end function logit_to_probability

    pure real(dp) function probability_logit(probability) result(logit)
        real(dp), intent(in) :: probability

        logit = log(probability/(1.0_dp-probability))
    end function probability_logit

end module fortml_mlp_amsgrad_hypergradient
