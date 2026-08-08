module fortml_mlp_lion_hypergradient
    !! Exact fixed full-batch Lion trajectory products.
    !!
    !! Lion (Evolved Sign Momentum) uses a sign update and is therefore only
    !! piecewise differentiable.  This module differentiates the deterministic
    !! trajectory on a declared nonzero-sign branch; if an interpolated update
    !! reaches the configured sign margin, the objective returns a typed
    !! `FORTNUM_NOT_IMPLEMENTED` refusal instead of pretending that a
    !! derivative exists at the branch boundary.  The packed outer vector is
    !! `[log(learning_rate), log(l2), logit(beta1), logit(beta2)]`.
    !!
    !! The implementation uses the analytic MLP Hessian-vector product for
    !! each tangent state and the same fixed initial parameters for every
    !! evaluation.  It is CPU-only until the model, Lion state, and branch
    !! metadata can remain resident on a device.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED, FORTNUM_CONVERGENCE_ERROR
    use fortml_device, only: FORTML_DEVICE_CPU
    use fortml_mlp, only: mlp_t
    use fortml_mlp_training, only: mlp_loss_value_gradient, mlp_loss_hvp
    use fortopt_objective, only: objective_t
    use fortopt_lbfgsb, only: lbfgsb_t, lbfgsb_options_t, lbfgsb_result_t
    implicit none
    private

    integer, parameter, public :: MLP_OPTIMIZER_LION = 7
    integer, parameter, public :: MLP_LION_HYPERPARAMETER_COUNT = 4
    integer, parameter, public :: MLP_LION_LOG_LEARNING_RATE = 1
    integer, parameter, public :: MLP_LION_LOG_L2 = 2
    integer, parameter, public :: MLP_LION_LOGIT_BETA1 = 3
    integer, parameter, public :: MLP_LION_LOGIT_BETA2 = 4

    type, public :: mlp_lion_hypergradient_metadata_t
        integer :: parameter_count = MLP_LION_HYPERPARAMETER_COUNT
        integer :: log_learning_rate_index = MLP_LION_LOG_LEARNING_RATE
        integer :: log_l2_index = MLP_LION_LOG_L2
        integer :: logit_beta1_index = MLP_LION_LOGIT_BETA1
        integer :: logit_beta2_index = MLP_LION_LOGIT_BETA2
        integer :: inner_steps = 0
        real(dp) :: sign_margin = 1.0e-14_dp
    end type mlp_lion_hypergradient_metadata_t

    type, public :: mlp_lion_hypergradient_options_t
        integer :: steps = 8
        real(dp) :: learning_rate = 1.0e-3_dp
        real(dp) :: l2 = 1.0e-4_dp
        real(dp) :: beta1 = 0.9_dp
        real(dp) :: beta2 = 0.99_dp
        !! A sign branch closer than this threshold is refused because its
        !! derivative is not defined.  The threshold is not an outer variable.
        real(dp) :: sign_margin = 1.0e-14_dp
        real(dp) :: lower_log_learning_rate = -16.0_dp
        real(dp) :: upper_log_learning_rate = 1.0_dp
        real(dp) :: lower_log_l2 = -24.0_dp
        real(dp) :: upper_log_l2 = 1.0_dp
        real(dp) :: lower_logit_beta1 = -12.0_dp
        real(dp) :: upper_logit_beta1 = 12.0_dp
        real(dp) :: lower_logit_beta2 = -12.0_dp
        real(dp) :: upper_logit_beta2 = 12.0_dp
        integer :: optimizer = MLP_OPTIMIZER_LION
        integer :: device_kind = FORTML_DEVICE_CPU
        integer :: memory = 8
        integer :: max_iterations = 100
        integer :: max_line_search = 40
        real(dp) :: gradient_tolerance = 1.0e-6_dp
        real(dp) :: step_tolerance = 1.0e-12_dp
        real(dp) :: objective_tolerance = 1.0e-12_dp
    end type mlp_lion_hypergradient_options_t

    type, public :: mlp_lion_hypergradient_result_t
        logical :: converged = .false.
        integer :: iterations = 0
        integer :: line_search_evaluations = 0
        real(dp) :: objective = huge(1.0_dp)
        real(dp) :: gradient_norm = huge(1.0_dp)
        real(dp) :: log_learning_rate = 0.0_dp
        real(dp) :: log_l2 = 0.0_dp
        real(dp) :: logit_beta1 = 0.0_dp
        real(dp) :: logit_beta2 = 0.0_dp
        real(dp) :: learning_rate = 0.0_dp
        real(dp) :: l2 = 0.0_dp
        real(dp) :: beta1 = 0.0_dp
        real(dp) :: beta2 = 0.0_dp
    end type mlp_lion_hypergradient_result_t

    type, public :: mlp_lion_hypergradient_objective_t
        private
        type(mlp_t), pointer :: model => null()
        real(dp), allocatable :: train_x(:, :), train_target(:, :)
        real(dp), allocatable :: validation_x(:, :), validation_target(:, :)
        real(dp), allocatable :: initial_parameters(:)
        type(mlp_lion_hypergradient_metadata_t) :: layout
        real(dp) :: initial_log_learning_rate = 0.0_dp
        real(dp) :: initial_log_l2 = 0.0_dp
        real(dp) :: initial_logit_beta1 = 0.0_dp
        real(dp) :: initial_logit_beta2 = 0.0_dp
        logical :: initialized = .false.
    contains
        procedure, public :: initialize => mlp_lion_hypergradient_initialize
        procedure, public :: parameter_count => mlp_lion_hypergradient_parameter_count
        procedure, public :: metadata => mlp_lion_hypergradient_metadata
        procedure, public :: parameters => mlp_lion_hypergradient_parameters
        procedure, public :: value_gradient => mlp_lion_hypergradient_value_gradient
        procedure, public :: jvp => mlp_lion_hypergradient_jvp
        procedure, public :: vjp => mlp_lion_hypergradient_vjp
        procedure, public :: fortopt => mlp_lion_hypergradient_fortopt
        procedure, public :: is_initialized => mlp_lion_hypergradient_is_initialized
    end type mlp_lion_hypergradient_objective_t

    public :: mlp_optimize_lion_hyperparameters

contains

    subroutine mlp_lion_hypergradient_initialize(self, model, train_x, train_target, &
            validation_x, validation_target, options, status)
        class(mlp_lion_hypergradient_objective_t), intent(out) :: self
        type(mlp_t), target, intent(inout) :: model
        real(dp), intent(in) :: train_x(:, :), train_target(:, :)
        real(dp), intent(in) :: validation_x(:, :), validation_target(:, :)
        type(mlp_lion_hypergradient_options_t), intent(in) :: options
        type(fortnum_status_t), intent(out) :: status
        !! Default-initialized instances, standing in for empty
        !! structure constructors: nvfortran rejects `T()` outright,
        !! and a declared local carries the same default init.
        type(mlp_lion_hypergradient_metadata_t) :: mlp_lion_hypergradient_metadata_t_default

        self%initialized = .false.
        self%layout = mlp_lion_hypergradient_metadata_t_default
        if (.not. valid_options(options)) then
            if (options%optimizer /= MLP_OPTIMIZER_LION .or. &
                options%device_kind /= FORTML_DEVICE_CPU) then
                call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                    "MLP Lion hypergradient: optimizer or device is unsupported")
            else
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP Lion hypergradient: options are invalid")
            end if
            return
        end if
        if (.not. valid_data(model, train_x, train_target) .or. &
            .not. valid_data(model, validation_x, validation_target)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP Lion hypergradient: model or data is invalid")
            return
        end if

        self%model => model
        allocate(self%train_x, source=train_x)
        allocate(self%train_target, source=train_target)
        allocate(self%validation_x, source=validation_x)
        allocate(self%validation_target, source=validation_target)
        allocate(self%initial_parameters, source=model%parameters())
        self%layout%inner_steps = options%steps
        self%layout%sign_margin = options%sign_margin
        self%initial_log_learning_rate = log(options%learning_rate)
        self%initial_log_l2 = log(options%l2)
        self%initial_logit_beta1 = logit_probability(options%beta1)
        self%initial_logit_beta2 = logit_probability(options%beta2)
        self%initialized = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_lion_hypergradient_initialize

    integer function mlp_lion_hypergradient_parameter_count(self) result(count)
        class(mlp_lion_hypergradient_objective_t), intent(in) :: self

        count = 0
        if (self%initialized) count = self%layout%parameter_count
    end function mlp_lion_hypergradient_parameter_count

    function mlp_lion_hypergradient_metadata(self) result(layout)
        class(mlp_lion_hypergradient_objective_t), intent(in) :: self
        type(mlp_lion_hypergradient_metadata_t) :: layout

        layout = self%layout
    end function mlp_lion_hypergradient_metadata

    function mlp_lion_hypergradient_parameters(self) result(parameters)
        class(mlp_lion_hypergradient_objective_t), intent(in) :: self
        real(dp), allocatable :: parameters(:)

        allocate(parameters(MLP_LION_HYPERPARAMETER_COUNT))
        parameters = [self%initial_log_learning_rate, self%initial_log_l2, &
            self%initial_logit_beta1, self%initial_logit_beta2]
    end function mlp_lion_hypergradient_parameters

    logical function mlp_lion_hypergradient_is_initialized(self) result(yes)
        class(mlp_lion_hypergradient_objective_t), intent(in) :: self

        yes = self%initialized .and. associated(self%model) .and. &
            allocated(self%initial_parameters)
    end function mlp_lion_hypergradient_is_initialized

    subroutine mlp_lion_hypergradient_value_gradient(self, parameters, value, &
            gradient, status)
        class(mlp_lion_hypergradient_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: direction(MLP_LION_HYPERPARAMETER_COUNT), tangent

        value = huge(1.0_dp)
        gradient = 0.0_dp
        direction = 0.0_dp
        if (.not. self%is_initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP Lion hypergradient: objective is not initialized")
            return
        end if
        if (.not. valid_parameter_vector(parameters, gradient)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP Lion hypergradient: packed shape is invalid")
            return
        end if
        call lion_forward(self, parameters, direction, value, tangent, gradient, status)
    end subroutine mlp_lion_hypergradient_value_gradient

    subroutine mlp_lion_hypergradient_jvp(self, parameters, direction, value, tangent, &
            status)
        class(mlp_lion_hypergradient_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:), direction(:)
        real(dp), intent(out) :: value, tangent
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: gradient(MLP_LION_HYPERPARAMETER_COUNT)

        value = huge(1.0_dp)
        tangent = 0.0_dp
        gradient = 0.0_dp
        if (.not. self%is_initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP Lion hypergradient JVP: objective is not initialized")
            return
        end if
        if (size(parameters) /= MLP_LION_HYPERPARAMETER_COUNT .or. &
            size(direction) /= MLP_LION_HYPERPARAMETER_COUNT .or. &
            any(.not. ieee_is_finite(parameters)) .or. &
            any(.not. ieee_is_finite(direction))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP Lion hypergradient JVP: packed shape is invalid")
            return
        end if
        call lion_forward(self, parameters, direction, value, tangent, gradient, status)
    end subroutine mlp_lion_hypergradient_jvp

    subroutine mlp_lion_hypergradient_vjp(self, parameters, output_bar, gradient, status)
        class(mlp_lion_hypergradient_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:), output_bar
        real(dp), intent(out) :: gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: value

        gradient = 0.0_dp
        if (.not. ieee_is_finite(output_bar)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP Lion hypergradient VJP: cotangent is invalid")
            return
        end if
        call self%value_gradient(parameters, value, gradient, status)
        if (status%code /= FORTNUM_OK) return
        gradient = output_bar*gradient
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_lion_hypergradient_vjp

    subroutine mlp_lion_hypergradient_fortopt(self, objective, status)
        class(mlp_lion_hypergradient_objective_t), target, intent(inout) :: self
        type(objective_t), intent(out) :: objective
        type(fortnum_status_t), intent(out) :: status

        if (.not. self%is_initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP Lion hypergradient: objective is not initialized")
            return
        end if
        call objective%initialize_context(MLP_LION_HYPERPARAMETER_COUNT, self, &
            mlp_lion_hypergradient_context_callback, status)
    end subroutine mlp_lion_hypergradient_fortopt

    subroutine mlp_lion_hypergradient_context_callback(context, parameters, value, &
            gradient, status)
        class(*), intent(inout) :: context
        real(dp), intent(in) :: parameters(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status

        select type (adapter => context)
            type is (mlp_lion_hypergradient_objective_t)
            call adapter%value_gradient(parameters, value, gradient, status)
        class default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP Lion hypergradient: context has the wrong type")
        end select
    end subroutine mlp_lion_hypergradient_context_callback

    subroutine mlp_optimize_lion_hyperparameters(model, train_x, train_target, &
            validation_x, validation_target, options, result, status)
        type(mlp_t), target, intent(inout) :: model
        real(dp), intent(in) :: train_x(:, :), train_target(:, :)
        real(dp), intent(in) :: validation_x(:, :), validation_target(:, :)
        type(mlp_lion_hypergradient_options_t), intent(in) :: options
        type(mlp_lion_hypergradient_result_t), intent(out) :: result
        type(fortnum_status_t), intent(out) :: status
        type(mlp_lion_hypergradient_objective_t), target :: adapter
        type(objective_t) :: objective
        type(lbfgsb_t) :: optimizer
        type(lbfgsb_options_t) :: optimizer_options
        type(lbfgsb_result_t) :: optimizer_result
        real(dp) :: parameters(MLP_LION_HYPERPARAMETER_COUNT)
        real(dp) :: lower(MLP_LION_HYPERPARAMETER_COUNT)
        real(dp) :: upper(MLP_LION_HYPERPARAMETER_COUNT)
        real(dp) :: gradient(MLP_LION_HYPERPARAMETER_COUNT)
        !! Default-initialized instances, standing in for empty
        !! structure constructors: nvfortran rejects `T()` outright,
        !! and a declared local carries the same default init.
        type(mlp_lion_hypergradient_result_t) :: mlp_lion_hypergradient_result_t_default

        result = mlp_lion_hypergradient_result_t_default
        if (.not. valid_options(options)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP Lion hyperparameter optimization: options are invalid")
            return
        end if
        call adapter%initialize(model, train_x, train_target, validation_x, &
            validation_target, options, status)
        if (status%code /= FORTNUM_OK) return
        parameters = adapter%parameters()
        lower = [options%lower_log_learning_rate, options%lower_log_l2, &
            options%lower_logit_beta1, options%lower_logit_beta2]
        upper = [options%upper_log_learning_rate, options%upper_log_l2, &
            options%upper_logit_beta1, options%upper_logit_beta2]
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
        result%log_learning_rate = parameters(MLP_LION_LOG_LEARNING_RATE)
        result%log_l2 = parameters(MLP_LION_LOG_L2)
        result%logit_beta1 = parameters(MLP_LION_LOGIT_BETA1)
        result%logit_beta2 = parameters(MLP_LION_LOGIT_BETA2)
        result%learning_rate = exp(result%log_learning_rate)
        result%l2 = exp(result%log_l2)
        result%beta1 = logit_to_probability(result%logit_beta1)
        result%beta2 = logit_to_probability(result%logit_beta2)
        if (.not. result%converged) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "MLP Lion hyperparameter optimization: iteration limit reached")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_optimize_lion_hyperparameters

    subroutine lion_forward(self, parameters, direction, value, tangent, gradient, status)
        class(mlp_lion_hypergradient_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:), direction(:)
        real(dp), intent(out) :: value, tangent, gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: theta(:), theta_dot(:, :), momentum(:), momentum_dot(:, :)
        real(dp), allocatable :: raw_gradient(:), gradient_dot(:, :), hvp(:)
        real(dp), allocatable :: interpolated(:), sign_update(:)
        real(dp), allocatable :: validation_gradient(:)
        real(dp) :: learning_rate, l2, beta1, beta2
        real(dp) :: learning_rate_dot, l2_dot, beta2_dot
        real(dp) :: train_value, l2_gradient, scalar_hvp
        integer :: n_parameters, step, parameter_index

        value = huge(1.0_dp)
        tangent = 0.0_dp
        gradient = 0.0_dp
        if (.not. finite_parameters(parameters, learning_rate, l2, beta1, beta2)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP Lion hypergradient: packed values are invalid")
            return
        end if
        n_parameters = size(self%initial_parameters)
        allocate(theta, source=self%initial_parameters)
        allocate(theta_dot(n_parameters, MLP_LION_HYPERPARAMETER_COUNT))
        allocate(momentum(n_parameters), momentum_dot(n_parameters, &
            MLP_LION_HYPERPARAMETER_COUNT))
        allocate(raw_gradient(n_parameters), gradient_dot(n_parameters, &
            MLP_LION_HYPERPARAMETER_COUNT), hvp(n_parameters))
        allocate(interpolated(n_parameters), sign_update(n_parameters))
        theta_dot = 0.0_dp
        momentum = 0.0_dp
        momentum_dot = 0.0_dp

        do step = 1, self%layout%inner_steps
            call self%model%set_parameters(theta, status)
            if (status%code /= FORTNUM_OK) return
            call mlp_loss_value_gradient(self%model, self%train_x, self%train_target, &
                l2, train_value, raw_gradient, l2_gradient, status)
            if (status%code /= FORTNUM_OK) return
            interpolated = beta1*momentum + (1.0_dp-beta1)*raw_gradient
            do parameter_index = 1, MLP_LION_HYPERPARAMETER_COUNT
                l2_dot = 0.0_dp
                if (parameter_index == MLP_LION_LOG_L2) l2_dot = l2
                call mlp_loss_hvp(self%model, self%train_x, self%train_target, l2, &
                    theta_dot(:, parameter_index), l2_dot, hvp, scalar_hvp, status)
                if (status%code /= FORTNUM_OK) return
                gradient_dot(:, parameter_index) = hvp
                ! The sign map has zero derivative on this fixed branch.  Its
                ! candidate tangent is therefore not needed by the recurrence.
            end do
            do parameter_index = 1, n_parameters
                if (abs(interpolated(parameter_index)) <= self%layout%sign_margin) then
                    call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                        "MLP Lion hypergradient: sign branch is nondifferentiable")
                    return
                end if
                sign_update(parameter_index) = merge(1.0_dp, -1.0_dp, &
                    interpolated(parameter_index) > 0.0_dp)
            end do
            do parameter_index = 1, MLP_LION_HYPERPARAMETER_COUNT
                beta2_dot = 0.0_dp
                if (parameter_index == MLP_LION_LOGIT_BETA2) beta2_dot = beta2*(1.0_dp-beta2)
                momentum_dot(:, parameter_index) = beta2*momentum_dot(:, parameter_index) + &
                    (1.0_dp-beta2)*gradient_dot(:, parameter_index) + &
                    beta2_dot*(momentum-raw_gradient)
            end do
            momentum = beta2*momentum + (1.0_dp-beta2)*raw_gradient
            do parameter_index = 1, MLP_LION_HYPERPARAMETER_COUNT
                learning_rate_dot = 0.0_dp
                if (parameter_index == MLP_LION_LOG_LEARNING_RATE) learning_rate_dot = learning_rate
                theta_dot(:, parameter_index) = theta_dot(:, parameter_index) - &
                    learning_rate_dot*sign_update
            end do
            theta = theta-learning_rate*sign_update
            if (any(.not. ieee_is_finite(theta)) .or. &
                any(.not. ieee_is_finite(theta_dot)) .or. &
                any(.not. ieee_is_finite(momentum))) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP Lion hypergradient: trajectory is not finite")
                return
            end if
        end do

        call self%model%set_parameters(theta, status)
        if (status%code /= FORTNUM_OK) return
        allocate(validation_gradient(n_parameters))
        call mlp_loss_value_gradient(self%model, self%validation_x, &
            self%validation_target, 0.0_dp, value, validation_gradient, &
            l2_gradient, status)
        if (status%code /= FORTNUM_OK) return
        do parameter_index = 1, MLP_LION_HYPERPARAMETER_COUNT
            gradient(parameter_index) = dot_product(validation_gradient, &
                theta_dot(:, parameter_index))
        end do
        tangent = dot_product(gradient, direction)
        if (.not. ieee_is_finite(value) .or. any(.not. ieee_is_finite(gradient)) .or. &
            .not. ieee_is_finite(tangent)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP Lion hypergradient: objective product is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine lion_forward

    logical function finite_parameters(parameters, learning_rate, l2, beta1, beta2) result(valid)
        real(dp), intent(in) :: parameters(:)
        real(dp), intent(out) :: learning_rate, l2, beta1, beta2

        learning_rate = 0.0_dp
        l2 = 0.0_dp
        beta1 = 0.0_dp
        beta2 = 0.0_dp
        valid = .false.
        if (size(parameters) /= MLP_LION_HYPERPARAMETER_COUNT .or. &
            any(.not. ieee_is_finite(parameters))) return
        learning_rate = exp(parameters(MLP_LION_LOG_LEARNING_RATE))
        l2 = exp(parameters(MLP_LION_LOG_L2))
        beta1 = logit_to_probability(parameters(MLP_LION_LOGIT_BETA1))
        beta2 = logit_to_probability(parameters(MLP_LION_LOGIT_BETA2))
        valid = ieee_is_finite(learning_rate) .and. ieee_is_finite(l2) .and. &
            ieee_is_finite(beta1) .and. ieee_is_finite(beta2) .and. &
            learning_rate > 0.0_dp .and. l2 > 0.0_dp .and. beta1 > 0.0_dp .and. &
            beta1 < 1.0_dp .and. beta2 > 0.0_dp .and. beta2 < 1.0_dp
    end function finite_parameters

    logical function valid_parameter_vector(parameters, gradient) result(valid)
        real(dp), intent(in) :: parameters(:), gradient(:)

        valid = size(parameters) == MLP_LION_HYPERPARAMETER_COUNT .and. &
            size(gradient) == MLP_LION_HYPERPARAMETER_COUNT .and. &
            all(ieee_is_finite(parameters))
    end function valid_parameter_vector

    logical function valid_options(options) result(valid)
        type(mlp_lion_hypergradient_options_t), intent(in) :: options
        real(dp) :: logit1, logit2

        valid = .false.
        if (options%beta1 <= 0.0_dp .or. options%beta1 >= 1.0_dp .or. &
            options%beta2 <= 0.0_dp .or. options%beta2 >= 1.0_dp) return
        logit1 = logit_probability(options%beta1)
        logit2 = logit_probability(options%beta2)
        valid = options%steps >= 1 .and. options%optimizer == MLP_OPTIMIZER_LION .and. &
            options%device_kind == FORTML_DEVICE_CPU .and. &
            ieee_is_finite(options%learning_rate) .and. ieee_is_finite(options%l2) .and. &
            options%learning_rate > 0.0_dp .and. options%l2 > 0.0_dp .and. &
            ieee_is_finite(options%sign_margin) .and. options%sign_margin >= 0.0_dp .and. &
            ieee_is_finite(options%lower_log_learning_rate) .and. &
            ieee_is_finite(options%upper_log_learning_rate) .and. &
            ieee_is_finite(options%lower_log_l2) .and. ieee_is_finite(options%upper_log_l2) .and. &
            ieee_is_finite(options%lower_logit_beta1) .and. ieee_is_finite(options%upper_logit_beta1) .and. &
            ieee_is_finite(options%lower_logit_beta2) .and. ieee_is_finite(options%upper_logit_beta2) .and. &
            options%lower_log_learning_rate <= options%upper_log_learning_rate .and. &
            options%lower_log_l2 <= options%upper_log_l2 .and. &
            options%lower_logit_beta1 <= options%upper_logit_beta1 .and. &
            options%lower_logit_beta2 <= options%upper_logit_beta2 .and. &
            log(options%learning_rate) >= options%lower_log_learning_rate .and. &
            log(options%learning_rate) <= options%upper_log_learning_rate .and. &
            log(options%l2) >= options%lower_log_l2 .and. log(options%l2) <= options%upper_log_l2 .and. &
            logit1 >= options%lower_logit_beta1 .and. logit1 <= options%upper_logit_beta1 .and. &
            logit2 >= options%lower_logit_beta2 .and. logit2 <= options%upper_logit_beta2 .and. &
            options%memory >= 1 .and. options%max_iterations >= 1 .and. &
            options%max_line_search >= 1 .and. ieee_is_finite(options%gradient_tolerance) .and. &
            ieee_is_finite(options%step_tolerance) .and. ieee_is_finite(options%objective_tolerance) .and. &
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

    pure real(dp) function logit_probability(probability) result(value)
        real(dp), intent(in) :: probability

        value = log(probability) - log(1.0_dp-probability)
    end function logit_probability

    pure real(dp) function logit_to_probability(logit) result(value)
        real(dp), intent(in) :: logit

        value = 1.0_dp/(1.0_dp+exp(-logit))
    end function logit_to_probability

end module fortml_mlp_lion_hypergradient
