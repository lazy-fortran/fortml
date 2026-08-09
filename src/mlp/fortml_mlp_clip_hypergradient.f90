module fortml_mlp_clip_hypergradient
    !! Exact first-order products through fixed full-batch SGD with global-norm
    !! gradient clipping.
    !!
    !! The packed outer vector is
    !! `[log(learning_rate), log(l2), log(gradient_clip_norm)]`.  The active
    !! clipping branch differentiates both the raw-gradient norm and the clip
    !! threshold.  A trajectory that lands on the clipping kink has no unique
    !! derivative and returns `FORTNUM_NOT_IMPLEMENTED`.  CUDA trajectories
    !! remain unavailable until the resident MLP derivative graph is linked.
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

    integer, parameter, public :: MLP_CLIP_HYPERPARAMETER_COUNT = 3
    integer, parameter, public :: MLP_CLIP_LOG_LEARNING_RATE = 1
    integer, parameter, public :: MLP_CLIP_LOG_L2 = 2
    integer, parameter, public :: MLP_CLIP_LOG_NORM = 3

    type, public :: mlp_clip_hypergradient_metadata_t
        integer :: parameter_count = MLP_CLIP_HYPERPARAMETER_COUNT
        integer :: log_learning_rate_index = MLP_CLIP_LOG_LEARNING_RATE
        integer :: log_l2_index = MLP_CLIP_LOG_L2
        integer :: log_clip_norm_index = MLP_CLIP_LOG_NORM
        integer :: inner_steps = 0
    end type mlp_clip_hypergradient_metadata_t

    type, public :: mlp_clip_hypergradient_options_t
        integer :: steps = 8
        real(dp) :: learning_rate = 1.0e-2_dp
        real(dp) :: l2 = 1.0e-4_dp
        real(dp) :: gradient_clip_norm = 1.0_dp
        real(dp) :: lower_log_learning_rate = -12.0_dp
        real(dp) :: upper_log_learning_rate = 2.0_dp
        real(dp) :: lower_log_l2 = -20.0_dp
        real(dp) :: upper_log_l2 = 2.0_dp
        real(dp) :: lower_log_clip_norm = -12.0_dp
        real(dp) :: upper_log_clip_norm = 8.0_dp
        integer :: optimizer = MLP_OPTIMIZER_SGD
        integer :: device_kind = FORTML_DEVICE_CPU
        integer :: memory = 8
        integer :: max_iterations = 100
        integer :: max_line_search = 40
        real(dp) :: gradient_tolerance = 1.0e-6_dp
        real(dp) :: step_tolerance = 1.0e-12_dp
        real(dp) :: objective_tolerance = 1.0e-12_dp
    end type mlp_clip_hypergradient_options_t

    type, public :: mlp_clip_hypergradient_result_t
        logical :: converged = .false.
        integer :: iterations = 0
        integer :: line_search_evaluations = 0
        real(dp) :: objective = huge(1.0_dp)
        real(dp) :: gradient_norm = huge(1.0_dp)
        real(dp) :: log_learning_rate = 0.0_dp
        real(dp) :: log_l2 = 0.0_dp
        real(dp) :: log_clip_norm = 0.0_dp
        real(dp) :: learning_rate = 0.0_dp
        real(dp) :: l2 = 0.0_dp
        real(dp) :: gradient_clip_norm = 0.0_dp
    end type mlp_clip_hypergradient_result_t

    type, public :: mlp_clip_hypergradient_objective_t
        private
        type(mlp_t), pointer :: model => null()
        real(dp), allocatable :: train_x(:, :), train_target(:, :)
        real(dp), allocatable :: validation_x(:, :), validation_target(:, :)
        real(dp), allocatable :: initial_parameters(:)
        type(mlp_clip_hypergradient_metadata_t) :: layout
        real(dp) :: initial_log_learning_rate = 0.0_dp
        real(dp) :: initial_log_l2 = 0.0_dp
        real(dp) :: initial_log_clip_norm = 0.0_dp
        logical :: initialized = .false.
    contains
        procedure, public :: initialize => mlp_clip_hypergradient_initialize
        procedure, public :: parameter_count => mlp_clip_hypergradient_parameter_count
        procedure, public :: metadata => mlp_clip_hypergradient_metadata
        procedure, public :: parameters => mlp_clip_hypergradient_parameters
        procedure, public :: value_gradient => mlp_clip_hypergradient_value_gradient
        procedure, public :: jvp => mlp_clip_hypergradient_jvp
        procedure, public :: vjp => mlp_clip_hypergradient_vjp
        procedure, public :: hvp => mlp_clip_hypergradient_hvp
        procedure, public :: fortopt => mlp_clip_hypergradient_fortopt
        procedure, public :: is_initialized => mlp_clip_hypergradient_is_initialized
    end type mlp_clip_hypergradient_objective_t

    public :: mlp_optimize_clip_hyperparameters

contains

    subroutine mlp_clip_hypergradient_initialize(self, model, train_x, train_target, &
            validation_x, validation_target, options, status)
        class(mlp_clip_hypergradient_objective_t), intent(out) :: self
        type(mlp_t), target, intent(inout) :: model
        real(dp), intent(in) :: train_x(:, :), train_target(:, :)
        real(dp), intent(in) :: validation_x(:, :), validation_target(:, :)
        type(mlp_clip_hypergradient_options_t), intent(in) :: options
        type(fortnum_status_t), intent(out) :: status
        type(mlp_clip_hypergradient_metadata_t) :: default_layout

        self%initialized = .false.
        self%layout = default_layout
        if (.not. valid_options(options)) then
            if (options%optimizer /= MLP_OPTIMIZER_SGD .or. &
                options%device_kind /= FORTML_DEVICE_CPU) then
                call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                    "MLP clip hypergradient: optimizer or device is unsupported")
            else
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP clip hypergradient: options are invalid")
            end if
            return
        end if
        if (.not. valid_data(model, train_x, train_target)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP clip hypergradient: training data are invalid")
            return
        end if
        if (.not. valid_data(model, validation_x, validation_target)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP clip hypergradient: validation data are invalid")
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
        self%initial_log_clip_norm = log(options%gradient_clip_norm)
        self%initialized = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_clip_hypergradient_initialize

    integer function mlp_clip_hypergradient_parameter_count(self) result(count)
        class(mlp_clip_hypergradient_objective_t), intent(in) :: self

        count = 0
        if (self%initialized) count = MLP_CLIP_HYPERPARAMETER_COUNT
    end function mlp_clip_hypergradient_parameter_count

    function mlp_clip_hypergradient_metadata(self) result(layout)
        class(mlp_clip_hypergradient_objective_t), intent(in) :: self
        type(mlp_clip_hypergradient_metadata_t) :: layout

        layout = self%layout
    end function mlp_clip_hypergradient_metadata

    function mlp_clip_hypergradient_parameters(self) result(parameters)
        class(mlp_clip_hypergradient_objective_t), intent(in) :: self
        real(dp), allocatable :: parameters(:)

        allocate(parameters(MLP_CLIP_HYPERPARAMETER_COUNT))
        parameters = [self%initial_log_learning_rate, self%initial_log_l2, &
            self%initial_log_clip_norm]
    end function mlp_clip_hypergradient_parameters

    logical function mlp_clip_hypergradient_is_initialized(self) result(initialized)
        class(mlp_clip_hypergradient_objective_t), intent(in) :: self

        initialized = self%initialized .and. associated(self%model) .and. &
            allocated(self%initial_parameters)
    end function mlp_clip_hypergradient_is_initialized

    subroutine mlp_clip_hypergradient_value_gradient(self, parameters, value, &
            gradient, status)
        class(mlp_clip_hypergradient_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: direction(MLP_CLIP_HYPERPARAMETER_COUNT), tangent

        direction = 0.0_dp
        call clip_forward(self, parameters, direction, value, tangent, gradient, status)
    end subroutine mlp_clip_hypergradient_value_gradient

    subroutine mlp_clip_hypergradient_jvp(self, parameters, direction, value, &
            tangent, status)
        class(mlp_clip_hypergradient_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:), direction(:)
        real(dp), intent(out) :: value, tangent
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: gradient(MLP_CLIP_HYPERPARAMETER_COUNT)

        value = huge(1.0_dp)
        tangent = 0.0_dp
        if (size(direction) /= MLP_CLIP_HYPERPARAMETER_COUNT) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP clip hypergradient JVP: direction shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(direction))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP clip hypergradient JVP: direction is not finite")
            return
        end if
        call clip_forward(self, parameters, direction, value, tangent, gradient, status)
    end subroutine mlp_clip_hypergradient_jvp

    subroutine mlp_clip_hypergradient_vjp(self, parameters, output_bar, &
            parameter_bar, status)
        class(mlp_clip_hypergradient_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:), output_bar
        real(dp), intent(out) :: parameter_bar(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: value, gradient(MLP_CLIP_HYPERPARAMETER_COUNT)

        parameter_bar = 0.0_dp
        if (size(parameter_bar) /= MLP_CLIP_HYPERPARAMETER_COUNT .or. &
            .not. ieee_is_finite(output_bar)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP clip hypergradient VJP: cotangent is invalid")
            return
        end if
        call self%value_gradient(parameters, value, gradient, status)
        if (status%code /= FORTNUM_OK) return
        parameter_bar = output_bar*gradient
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_clip_hypergradient_vjp

    subroutine mlp_clip_hypergradient_hvp(self, parameters, direction, product, status)
        class(mlp_clip_hypergradient_objective_t), intent(in) :: self
        real(dp), intent(in) :: parameters(:), direction(:)
        real(dp), intent(out) :: product(:)
        type(fortnum_status_t), intent(out) :: status

        product = 0.0_dp
        if (size(parameters) /= MLP_CLIP_HYPERPARAMETER_COUNT .or. &
            size(direction) /= MLP_CLIP_HYPERPARAMETER_COUNT .or. &
            size(product) /= MLP_CLIP_HYPERPARAMETER_COUNT) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP clip hypergradient HVP: packed shape is invalid")
            return
        end if
        if (.not. self%is_initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP clip hypergradient HVP: objective is not initialized")
            return
        end if
        call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
            "MLP clip hypergradient HVP requires third network derivatives")
    end subroutine mlp_clip_hypergradient_hvp

    subroutine mlp_clip_hypergradient_fortopt(self, objective, status)
        class(mlp_clip_hypergradient_objective_t), target, intent(inout) :: self
        type(objective_t), intent(out) :: objective
        type(fortnum_status_t), intent(out) :: status

        if (.not. self%is_initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP clip hypergradient: objective is not initialized")
            return
        end if
        call objective%initialize_context(MLP_CLIP_HYPERPARAMETER_COUNT, self, &
            clip_context_callback, status)
    end subroutine mlp_clip_hypergradient_fortopt

    subroutine clip_context_callback(context, parameters, value, gradient, status)
        class(*), intent(inout) :: context
        real(dp), intent(in) :: parameters(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status

        select type (adapter => context)
            type is (mlp_clip_hypergradient_objective_t)
            call adapter%value_gradient(parameters, value, gradient, status)
        class default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP clip hypergradient: context has the wrong type")
        end select
    end subroutine clip_context_callback

    subroutine mlp_optimize_clip_hyperparameters(model, train_x, train_target, &
            validation_x, validation_target, options, result, status)
        type(mlp_t), target, intent(inout) :: model
        real(dp), intent(in) :: train_x(:, :), train_target(:, :)
        real(dp), intent(in) :: validation_x(:, :), validation_target(:, :)
        type(mlp_clip_hypergradient_options_t), intent(in) :: options
        type(mlp_clip_hypergradient_result_t), intent(out) :: result
        type(fortnum_status_t), intent(out) :: status
        type(mlp_clip_hypergradient_objective_t), target :: adapter
        type(mlp_clip_hypergradient_result_t) :: default_result
        type(objective_t) :: objective
        type(lbfgsb_t) :: optimizer
        type(lbfgsb_options_t) :: optimizer_options
        type(lbfgsb_result_t) :: optimizer_result
        real(dp) :: parameters(MLP_CLIP_HYPERPARAMETER_COUNT)
        real(dp) :: lower(MLP_CLIP_HYPERPARAMETER_COUNT)
        real(dp) :: upper(MLP_CLIP_HYPERPARAMETER_COUNT)
        real(dp) :: gradient(MLP_CLIP_HYPERPARAMETER_COUNT)

        result = default_result
        call adapter%initialize(model, train_x, train_target, validation_x, &
            validation_target, options, status)
        if (status%code /= FORTNUM_OK) return
        parameters = adapter%parameters()
        lower = [options%lower_log_learning_rate, options%lower_log_l2, &
            options%lower_log_clip_norm]
        upper = [options%upper_log_learning_rate, options%upper_log_l2, &
            options%upper_log_clip_norm]
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
        result%log_learning_rate = parameters(MLP_CLIP_LOG_LEARNING_RATE)
        result%log_l2 = parameters(MLP_CLIP_LOG_L2)
        result%log_clip_norm = parameters(MLP_CLIP_LOG_NORM)
        result%learning_rate = exp(result%log_learning_rate)
        result%l2 = exp(result%log_l2)
        result%gradient_clip_norm = exp(result%log_clip_norm)
        if (.not. result%converged) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "MLP clip hyperparameter optimization: iteration limit reached")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_optimize_clip_hyperparameters

    subroutine clip_forward(self, parameters, direction, value, tangent, gradient, status)
        class(mlp_clip_hypergradient_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:), direction(:)
        real(dp), intent(out) :: value, tangent, gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: theta(:), theta_dot(:, :), raw_gradient(:)
        real(dp), allocatable :: raw_gradient_dot(:), clipped_gradient(:)
        real(dp), allocatable :: clipped_gradient_dot(:), validation_gradient(:)
        real(dp) :: learning_rate, l2, clip_norm, raw_norm, scale, scale_dot
        real(dp) :: learning_rate_dot, l2_dot, clip_norm_dot
        real(dp) :: train_value, l2_gradient, l2_hvp, kink_tolerance
        integer :: n_parameters, step, index

        value = huge(1.0_dp)
        tangent = 0.0_dp
        gradient = 0.0_dp
        if (.not. self%is_initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP clip hypergradient: objective is not initialized")
            return
        end if
        if (size(gradient) /= MLP_CLIP_HYPERPARAMETER_COUNT .or. &
            size(direction) /= MLP_CLIP_HYPERPARAMETER_COUNT) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP clip hypergradient: packed shape is invalid")
            return
        end if
        if (.not. finite_parameters(parameters, learning_rate, l2, clip_norm)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP clip hypergradient: packed parameters are invalid")
            return
        end if

        n_parameters = size(self%initial_parameters)
        allocate(theta, source=self%initial_parameters)
        allocate(theta_dot(n_parameters, MLP_CLIP_HYPERPARAMETER_COUNT))
        allocate(raw_gradient(n_parameters), raw_gradient_dot(n_parameters))
        allocate(clipped_gradient(n_parameters), clipped_gradient_dot(n_parameters))
        allocate(validation_gradient(n_parameters))
        theta_dot = 0.0_dp
        do step = 1, self%layout%inner_steps
            call self%model%set_parameters(theta, status)
            if (status%code /= FORTNUM_OK) return
            call mlp_loss_value_gradient(self%model, self%train_x, self%train_target, &
                l2, train_value, raw_gradient, l2_gradient, status)
            if (status%code /= FORTNUM_OK) return
            raw_norm = sqrt(sum(raw_gradient*raw_gradient))
            kink_tolerance = 64.0_dp*epsilon(1.0_dp)* &
                max(1.0_dp, raw_norm, clip_norm)
            if (abs(raw_norm-clip_norm) <= kink_tolerance) then
                call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                    "MLP clip hypergradient: clipping active-set boundary")
                return
            end if
            if (raw_norm > clip_norm) then
                scale = clip_norm/raw_norm
                clipped_gradient = scale*raw_gradient
            else
                scale = 1.0_dp
                clipped_gradient = raw_gradient
            end if
            do index = 1, MLP_CLIP_HYPERPARAMETER_COUNT
                learning_rate_dot = 0.0_dp
                l2_dot = 0.0_dp
                clip_norm_dot = 0.0_dp
                if (index == MLP_CLIP_LOG_LEARNING_RATE) then
                    learning_rate_dot = learning_rate
                else if (index == MLP_CLIP_LOG_L2) then
                    l2_dot = l2
                else
                    clip_norm_dot = clip_norm
                end if
                call mlp_loss_hvp(self%model, self%train_x, self%train_target, &
                    l2, theta_dot(:, index), l2_dot, raw_gradient_dot, l2_hvp, status)
                if (status%code /= FORTNUM_OK) return
                if (raw_norm > clip_norm) then
                    scale_dot = clip_norm_dot/raw_norm - &
                        clip_norm*dot_product(raw_gradient, raw_gradient_dot)/raw_norm**3
                    clipped_gradient_dot = scale*raw_gradient_dot + &
                        scale_dot*raw_gradient
                else
                    clipped_gradient_dot = raw_gradient_dot
                end if
                theta_dot(:, index) = theta_dot(:, index) - &
                    learning_rate_dot*clipped_gradient - &
                    learning_rate*clipped_gradient_dot
            end do
            theta = theta-learning_rate*clipped_gradient
            if (any(.not. ieee_is_finite(theta))) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP clip hypergradient: parameter trajectory is not finite")
                return
            end if
            if (any(.not. ieee_is_finite(theta_dot))) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP clip hypergradient: tangent trajectory is not finite")
                return
            end if
        end do
        call self%model%set_parameters(theta, status)
        if (status%code /= FORTNUM_OK) return
        call mlp_loss_value_gradient(self%model, self%validation_x, &
            self%validation_target, 0.0_dp, value, validation_gradient, &
            l2_gradient, status)
        if (status%code /= FORTNUM_OK) return
        do index = 1, MLP_CLIP_HYPERPARAMETER_COUNT
            gradient(index) = dot_product(validation_gradient, theta_dot(:, index))
        end do
        tangent = dot_product(gradient, direction)
        if (.not. ieee_is_finite(value) .or. &
            any(.not. ieee_is_finite(gradient)) .or. &
            .not. ieee_is_finite(tangent)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP clip hypergradient: objective products are not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine clip_forward

    logical function finite_parameters(parameters, learning_rate, l2, &
            clip_norm) result(valid)
        real(dp), intent(in) :: parameters(:)
        real(dp), intent(out) :: learning_rate, l2, clip_norm

        learning_rate = 0.0_dp
        l2 = 0.0_dp
        clip_norm = 0.0_dp
        valid = size(parameters) == MLP_CLIP_HYPERPARAMETER_COUNT
        if (.not. valid) return
        valid = all(ieee_is_finite(parameters))
        if (.not. valid) return
        learning_rate = exp(parameters(MLP_CLIP_LOG_LEARNING_RATE))
        l2 = exp(parameters(MLP_CLIP_LOG_L2))
        clip_norm = exp(parameters(MLP_CLIP_LOG_NORM))
        valid = ieee_is_finite(learning_rate) .and. ieee_is_finite(l2) .and. &
            ieee_is_finite(clip_norm) .and. learning_rate > 0.0_dp .and. &
            l2 > 0.0_dp .and. clip_norm > 0.0_dp
    end function finite_parameters

    logical function valid_options(options) result(valid)
        type(mlp_clip_hypergradient_options_t), intent(in) :: options

        valid = options%steps >= 1 .and. options%optimizer == MLP_OPTIMIZER_SGD .and. &
            options%device_kind == FORTML_DEVICE_CPU .and. &
            ieee_is_finite(options%learning_rate) .and. options%learning_rate > 0.0_dp .and. &
            ieee_is_finite(options%l2) .and. options%l2 > 0.0_dp .and. &
            ieee_is_finite(options%gradient_clip_norm) .and. &
            options%gradient_clip_norm > 0.0_dp .and. &
            ieee_is_finite(options%lower_log_learning_rate) .and. &
            ieee_is_finite(options%upper_log_learning_rate) .and. &
            ieee_is_finite(options%lower_log_l2) .and. &
            ieee_is_finite(options%upper_log_l2) .and. &
            ieee_is_finite(options%lower_log_clip_norm) .and. &
            ieee_is_finite(options%upper_log_clip_norm) .and. &
            options%lower_log_learning_rate <= options%upper_log_learning_rate .and. &
            options%lower_log_l2 <= options%upper_log_l2 .and. &
            options%lower_log_clip_norm <= options%upper_log_clip_norm .and. &
            log(options%learning_rate) >= options%lower_log_learning_rate .and. &
            log(options%learning_rate) <= options%upper_log_learning_rate .and. &
            log(options%l2) >= options%lower_log_l2 .and. &
            log(options%l2) <= options%upper_log_l2 .and. &
            log(options%gradient_clip_norm) >= options%lower_log_clip_norm .and. &
            log(options%gradient_clip_norm) <= options%upper_log_clip_norm .and. &
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
        if (size(x, 1) < 1) return
        if (size(x, 2) /= model%layer_sizes(1)) return
        if (size(target, 1) /= size(x, 1)) return
        if (size(target, 2) /= model%layer_sizes(size(model%layer_sizes))) return
        valid = all(ieee_is_finite(x)) .and. all(ieee_is_finite(target))
    end function valid_data

end module fortml_mlp_clip_hypergradient
