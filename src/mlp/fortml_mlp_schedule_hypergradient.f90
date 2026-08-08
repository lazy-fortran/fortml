module fortml_mlp_schedule_hypergradient
    !! Exact hypergradients through a scheduled full-batch MLP trajectory.
    !!
    !! The inner recurrence is
    !!
    !!     theta_(k+1) = theta_k - rate(k; h) * grad_theta L(theta_k, l2)
    !!
    !! where `rate` is one of the typed stateless schedules from
    !! `fortml_mlp_schedules`.  The outer variable is
    !! `[log(base_rate), log(l2), logit(min_fraction), logit(decay_factor)]`
    !! for ordinary schedules.  For one-cycle schedules the final two
    !! coordinates are `[log(peak_rate_fraction), log(final_rate_fraction)]`.
    !! Schedule shape (kind and integer update counts) is deliberately fixed;
    !! all continuous fields have exact forward JVP and reverse VJP products.
    !! The objective is a FortOpt context, so the same reverse product is used
    !! by L-BFGS-B without finite differences or an optimizer fallback.
    !!
    !! CUDA is an explicit refusal until a resident MLP trajectory kernel is
    !! linked.  This prevents a host round trip from being reported as a GPU
    !! derivative or training result.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, status_ok, &
        FORTNUM_OK, FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED, &
        FORTNUM_CONVERGENCE_ERROR
    use fortml_device, only: FORTML_DEVICE_CPU
    use fortml_mlp, only: mlp_t, MLP_LINEAR
    use fortml_mlp_schedules, only: mlp_learning_rate_schedule_t, &
        MLP_SCHEDULE_CONSTANT, MLP_SCHEDULE_ONE_CYCLE
    use fortml_mlp_training, only: mlp_loss_value_gradient, mlp_loss_hvp
    use fortopt_objective, only: objective_t
    use fortopt_lbfgsb, only: lbfgsb_t, lbfgsb_options_t, lbfgsb_result_t
    implicit none
    private

    integer, parameter, public :: MLP_SCHEDULE_HYPERPARAMETER_COUNT = 4
    integer, parameter, public :: MLP_SCHEDULE_LOG_BASE_RATE = 1
    integer, parameter, public :: MLP_SCHEDULE_LOG_L2 = 2
    integer, parameter, public :: MLP_SCHEDULE_LOGIT_MIN_FRACTION = 3
    integer, parameter, public :: MLP_SCHEDULE_LOGIT_DECAY_FACTOR = 4
    integer, parameter, public :: MLP_SCHEDULE_LOG_PEAK_FRACTION = 3
    integer, parameter, public :: MLP_SCHEDULE_LOG_FINAL_FRACTION = 4

    type, public :: mlp_schedule_hypergradient_metadata_t
        integer :: parameter_count = MLP_SCHEDULE_HYPERPARAMETER_COUNT
        integer :: log_base_rate_index = MLP_SCHEDULE_LOG_BASE_RATE
        integer :: log_l2_index = MLP_SCHEDULE_LOG_L2
        integer :: logit_min_fraction_index = MLP_SCHEDULE_LOGIT_MIN_FRACTION
        integer :: logit_decay_factor_index = MLP_SCHEDULE_LOGIT_DECAY_FACTOR
        integer :: inner_steps = 0
        integer :: schedule_kind = 0
        integer :: warmup_updates = 0
        integer :: total_updates = 0
        logical :: one_cycle_coordinates = .false.
    end type mlp_schedule_hypergradient_metadata_t

    type, public :: mlp_schedule_hypergradient_options_t
        !! Fixed-shape scheduled full-batch trajectory configuration.
        integer :: steps = 8
        type(mlp_learning_rate_schedule_t) :: schedule
        real(dp) :: base_rate = 1.0e-2_dp
        real(dp) :: l2 = 1.0e-4_dp
        real(dp) :: lower_log_base_rate = -12.0_dp
        real(dp) :: upper_log_base_rate = 2.0_dp
        real(dp) :: lower_log_l2 = -20.0_dp
        real(dp) :: upper_log_l2 = 2.0_dp
        real(dp) :: lower_logit_min_fraction = -12.0_dp
        real(dp) :: upper_logit_min_fraction = 12.0_dp
        real(dp) :: lower_logit_decay_factor = -12.0_dp
        real(dp) :: upper_logit_decay_factor = 12.0_dp
        integer :: device_kind = FORTML_DEVICE_CPU
        integer :: memory = 8
        integer :: max_iterations = 100
        integer :: max_line_search = 40
        real(dp) :: gradient_tolerance = 1.0e-6_dp
        real(dp) :: step_tolerance = 1.0e-12_dp
        real(dp) :: objective_tolerance = 1.0e-12_dp
    end type mlp_schedule_hypergradient_options_t

    type, public :: mlp_schedule_hypergradient_result_t
        logical :: converged = .false.
        integer :: iterations = 0
        integer :: line_search_evaluations = 0
        real(dp) :: objective = huge(1.0_dp)
        real(dp) :: gradient_norm = huge(1.0_dp)
        real(dp) :: log_base_rate = 0.0_dp
        real(dp) :: log_l2 = 0.0_dp
        real(dp) :: logit_min_fraction = 0.0_dp
        real(dp) :: logit_decay_factor = 0.0_dp
        real(dp) :: base_rate = 0.0_dp
        real(dp) :: l2 = 0.0_dp
        real(dp) :: min_rate_fraction = 0.0_dp
        real(dp) :: decay_factor = 0.0_dp
    end type mlp_schedule_hypergradient_result_t

    type, public :: mlp_schedule_hypergradient_objective_t
        !! FortOpt-compatible scheduled trajectory objective.
        private
        type(mlp_t), pointer :: model => null()
        real(dp), allocatable :: train_x(:, :), train_target(:, :)
        real(dp), allocatable :: validation_x(:, :), validation_target(:, :)
        real(dp), allocatable :: initial_parameters(:)
        type(mlp_learning_rate_schedule_t) :: schedule
        type(mlp_schedule_hypergradient_metadata_t) :: layout
        real(dp) :: initial_log_base_rate = 0.0_dp
        real(dp) :: initial_log_l2 = 0.0_dp
        real(dp) :: initial_logit_min_fraction = 0.0_dp
        real(dp) :: initial_logit_decay_factor = 0.0_dp
        logical :: initialized = .false.
    contains
        procedure, public :: initialize => schedule_hypergradient_initialize
        procedure, public :: parameter_count => schedule_hypergradient_parameter_count
        procedure, public :: metadata => schedule_hypergradient_metadata
        procedure, public :: parameters => schedule_hypergradient_parameters
        procedure, public :: value_gradient => schedule_hypergradient_value_gradient
        procedure, public :: jvp => schedule_hypergradient_jvp
        procedure, public :: vjp => schedule_hypergradient_vjp
        procedure, public :: hvp => schedule_hypergradient_hvp
        procedure, public :: fortopt => schedule_hypergradient_fortopt
        procedure, public :: is_initialized => schedule_hypergradient_is_initialized
    end type mlp_schedule_hypergradient_objective_t

    public :: mlp_optimize_schedule_hyperparameters

contains

    subroutine schedule_hypergradient_initialize(self, model, train_x, train_target, &
            validation_x, validation_target, options, status)
        class(mlp_schedule_hypergradient_objective_t), intent(out) :: self
        type(mlp_t), target, intent(inout) :: model
        real(dp), intent(in) :: train_x(:, :), train_target(:, :)
        real(dp), intent(in) :: validation_x(:, :), validation_target(:, :)
        type(mlp_schedule_hypergradient_options_t), intent(in) :: options
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: min_fraction, decay_factor
        !! Default-initialized instances, standing in for empty
        !! structure constructors: nvfortran rejects `T()` outright,
        !! and a declared local carries the same default init.
        type(mlp_schedule_hypergradient_metadata_t) :: mlp_schedule_hypergradient_metadata_t_default

        self%initialized = .false.
        self%layout = mlp_schedule_hypergradient_metadata_t_default
        if (.not. valid_options(options)) then
            if (options%device_kind /= FORTML_DEVICE_CPU) then
                call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                    "MLP schedule hypergradient: device is unsupported")
            else
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP schedule hypergradient: options are invalid")
            end if
            return
        end if
        if (.not. valid_data(model, train_x, train_target) .or. &
            .not. valid_data(model, validation_x, validation_target)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP schedule hypergradient: model or data shape is invalid")
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
        if (options%schedule%kind == MLP_SCHEDULE_ONE_CYCLE) then
            ! The coordinate slots are logarithms for positive one-cycle
            ! peak/final fractions.  The schedule validator enforces
            ! peak >= 1 and final <= peak.
            min_fraction = max(options%schedule%peak_rate_fraction, 1.0e-12_dp)
            decay_factor = max(options%schedule%final_rate_fraction, 1.0e-12_dp)
            self%layout%one_cycle_coordinates = .true.
            self%initial_logit_min_fraction = log(min_fraction)
            self%initial_logit_decay_factor = log(decay_factor)
        else
            min_fraction = interior_probability(options%schedule%min_rate_fraction)
            decay_factor = interior_probability(options%schedule%decay_factor)
            self%initial_logit_min_fraction = logit(min_fraction)
            self%initial_logit_decay_factor = logit(decay_factor)
        end if
        self%initialized = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine schedule_hypergradient_initialize

    integer function schedule_hypergradient_parameter_count(self) result(count)
        class(mlp_schedule_hypergradient_objective_t), intent(in) :: self

        count = 0
        if (self%initialized) count = self%layout%parameter_count
    end function schedule_hypergradient_parameter_count

    function schedule_hypergradient_metadata(self) result(layout)
        class(mlp_schedule_hypergradient_objective_t), intent(in) :: self
        type(mlp_schedule_hypergradient_metadata_t) :: layout

        layout = self%layout
    end function schedule_hypergradient_metadata

    function schedule_hypergradient_parameters(self) result(parameters)
        class(mlp_schedule_hypergradient_objective_t), intent(in) :: self
        real(dp), allocatable :: parameters(:)

        allocate(parameters(MLP_SCHEDULE_HYPERPARAMETER_COUNT))
        parameters = [self%initial_log_base_rate, self%initial_log_l2, &
            self%initial_logit_min_fraction, self%initial_logit_decay_factor]
    end function schedule_hypergradient_parameters

    logical function schedule_hypergradient_is_initialized(self) result(yes)
        class(mlp_schedule_hypergradient_objective_t), intent(in) :: self

        yes = self%initialized .and. associated(self%model) .and. &
            allocated(self%initial_parameters)
    end function schedule_hypergradient_is_initialized

    subroutine schedule_hypergradient_value_gradient(self, parameters, value, &
            gradient, status)
        class(mlp_schedule_hypergradient_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status

        value = huge(1.0_dp)
        gradient = 0.0_dp
        if (.not. self%is_initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP schedule hypergradient: objective is not initialized")
            return
        end if
        if (.not. valid_parameter_vector(parameters, gradient)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP schedule hypergradient: packed shape is invalid")
            return
        end if
        call reverse_value_gradient(self, parameters, value, gradient, status)
    end subroutine schedule_hypergradient_value_gradient

    subroutine schedule_hypergradient_jvp(self, parameters, direction, value, &
            tangent, status)
        !! Exact scalar JVP through every schedule and optimizer update.
        class(mlp_schedule_hypergradient_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:), direction(:)
        real(dp), intent(out) :: value, tangent
        type(fortnum_status_t), intent(out) :: status

        value = huge(1.0_dp)
        tangent = 0.0_dp
        if (.not. self%is_initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP schedule hypergradient JVP: objective is not initialized")
            return
        end if
        if (size(parameters) /= MLP_SCHEDULE_HYPERPARAMETER_COUNT .or. &
            size(direction) /= MLP_SCHEDULE_HYPERPARAMETER_COUNT .or. &
            any(.not. ieee_is_finite(parameters)) .or. &
            any(.not. ieee_is_finite(direction))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP schedule hypergradient JVP: packed shape is invalid")
            return
        end if
        call forward_jvp(self, parameters, direction, value, tangent, status)
    end subroutine schedule_hypergradient_jvp

    subroutine schedule_hypergradient_vjp(self, parameters, output_bar, gradient, &
            status)
        !! Exact reverse VJP for a scalar cotangent.
        class(mlp_schedule_hypergradient_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:), output_bar
        real(dp), intent(out) :: gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: value

        gradient = 0.0_dp
        if (.not. self%is_initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP schedule hypergradient VJP: objective is not initialized")
            return
        end if
        if (size(parameters) /= MLP_SCHEDULE_HYPERPARAMETER_COUNT .or. &
            size(gradient) /= MLP_SCHEDULE_HYPERPARAMETER_COUNT .or. &
            .not. ieee_is_finite(output_bar)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP schedule hypergradient VJP: packed shape is invalid")
            return
        end if
        call reverse_value_gradient(self, parameters, value, gradient, status)
        if (.not. status_ok(status)) return
        gradient = output_bar*gradient
        call status_set(status, FORTNUM_OK, "")
    end subroutine schedule_hypergradient_vjp

    subroutine schedule_hypergradient_hvp(self, parameters, direction, product, status)
        !! Exact outer HVP for a constant-rate affine trajectory.
        !!
        !! A one-layer linear MLP has a parameter-independent MSE Hessian, so
        !! mixed second tangents through the fixed constant-rate schedule can
        !! be propagated analytically.  Other schedule families retain a typed
        !! refusal until their rate second products are specified; nonlinear
        !! networks retain the third-derivative boundary.
        class(mlp_schedule_hypergradient_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:), direction(:)
        real(dp), intent(out) :: product(:)
        type(fortnum_status_t), intent(out) :: status

        product = 0.0_dp
        if (size(parameters) /= MLP_SCHEDULE_HYPERPARAMETER_COUNT .or. &
            size(direction) /= MLP_SCHEDULE_HYPERPARAMETER_COUNT .or. &
            size(product) /= MLP_SCHEDULE_HYPERPARAMETER_COUNT .or. &
            any(.not. ieee_is_finite(parameters)) .or. &
            any(.not. ieee_is_finite(direction))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP schedule hypergradient HVP: packed shape is invalid")
            return
        end if
        if (.not. self%is_initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP schedule hypergradient HVP: objective is not initialized")
            return
        end if
        if (self%layout%schedule_kind /= MLP_SCHEDULE_CONSTANT) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "MLP schedule hypergradient HVP requires schedule rate second products")
            return
        end if
        if (.not. affine_one_layer(self%model)) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "MLP schedule hypergradient HVP requires one linear dense layer")
            return
        end if
        call constant_schedule_affine_hvp(self, parameters, direction, product, status)
    end subroutine schedule_hypergradient_hvp

    subroutine constant_schedule_affine_hvp(self, parameters, direction, product, status)
        !! Mixed second tangent for the constant-rate affine branch.
        class(mlp_schedule_hypergradient_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:), direction(:)
        real(dp), intent(out) :: product(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: theta(:), theta_dot(:, :), theta_ddot(:, :)
        real(dp), allocatable :: gradient(:), gradient_dot(:, :), gradient_ddot(:)
        real(dp), allocatable :: theta_direction(:), gradient_direction(:)
        real(dp), allocatable :: validation_gradient(:), validation_hvp(:)
        real(dp) :: base_rate, l2, min_fraction, decay_factor
        real(dp) :: base_direction, l2_direction, base_second, l2_second
        real(dp) :: base_tangent, l2_tangent
        real(dp) :: train_value, validation_value, l2_gradient, scalar_hvp
        integer :: n_parameters, step, i

        product = 0.0_dp
        call unpack_parameters(parameters, base_rate, l2, min_fraction, decay_factor, status)
        if (.not. status_ok(status)) return
        n_parameters = size(self%initial_parameters)
        allocate(theta, source=self%initial_parameters)
        allocate(theta_dot(n_parameters, MLP_SCHEDULE_HYPERPARAMETER_COUNT))
        allocate(theta_ddot(n_parameters, MLP_SCHEDULE_HYPERPARAMETER_COUNT))
        allocate(gradient(n_parameters), gradient_dot(n_parameters, &
            MLP_SCHEDULE_HYPERPARAMETER_COUNT), gradient_ddot(n_parameters))
        allocate(theta_direction(n_parameters), gradient_direction(n_parameters))
        theta_dot = 0.0_dp
        theta_ddot = 0.0_dp
        do step = 1, self%layout%inner_steps
            call self%model%set_parameters(theta, status)
            if (.not. status_ok(status)) return
            call mlp_loss_value_gradient(self%model, self%train_x, self%train_target, &
                l2, train_value, gradient, l2_gradient, status)
            if (.not. status_ok(status)) return
            theta_direction = matmul(theta_dot, direction)
            gradient_direction = 0.0_dp
            do i = 1, MLP_SCHEDULE_HYPERPARAMETER_COUNT
                base_tangent = 0.0_dp
                l2_tangent = 0.0_dp
                if (i == MLP_SCHEDULE_LOG_BASE_RATE) then
                    base_tangent = base_rate
                else if (i == MLP_SCHEDULE_LOG_L2) then
                    l2_tangent = l2
                end if
                call mlp_loss_hvp(self%model, self%train_x, self%train_target, l2, &
                    theta_dot(:, i), l2_tangent, gradient_dot(:, i), scalar_hvp, status)
                if (.not. status_ok(status)) return
                gradient_direction = gradient_direction + direction(i)*gradient_dot(:, i)
            end do
            base_direction = base_rate*direction(MLP_SCHEDULE_LOG_BASE_RATE)
            l2_direction = l2*direction(MLP_SCHEDULE_LOG_L2)
            do i = 1, MLP_SCHEDULE_HYPERPARAMETER_COUNT
                base_tangent = 0.0_dp
                l2_tangent = 0.0_dp
                base_second = 0.0_dp
                l2_second = 0.0_dp
                if (i == MLP_SCHEDULE_LOG_BASE_RATE) then
                    base_tangent = base_rate
                    base_second = base_rate*direction(MLP_SCHEDULE_LOG_BASE_RATE)
                else if (i == MLP_SCHEDULE_LOG_L2) then
                    l2_tangent = l2
                    l2_second = l2*direction(MLP_SCHEDULE_LOG_L2)
                end if
                call mlp_loss_hvp(self%model, self%train_x, self%train_target, l2, &
                    theta_ddot(:, i), l2_second, gradient_ddot, scalar_hvp, status)
                if (.not. status_ok(status)) return
                gradient_ddot = gradient_ddot + l2_direction*theta_dot(:, i) + &
                    l2_tangent*theta_direction
                theta_ddot(:, i) = theta_ddot(:, i) - base_second*gradient - &
                    base_tangent*gradient_direction - base_direction*gradient_dot(:, i) - &
                    base_rate*gradient_ddot
                theta_dot(:, i) = theta_dot(:, i) - base_tangent*gradient - &
                    base_rate*gradient_dot(:, i)
            end do
            theta = theta - base_rate*gradient
            if (any(.not. ieee_is_finite(theta)) .or. &
                any(.not. ieee_is_finite(theta_dot)) .or. &
                any(.not. ieee_is_finite(theta_ddot))) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP schedule hypergradient HVP: trajectory is not finite")
                return
            end if
        end do
        call self%model%set_parameters(theta, status)
        if (.not. status_ok(status)) return
        allocate(validation_gradient(n_parameters), validation_hvp(n_parameters))
        call mlp_loss_value_gradient(self%model, self%validation_x, self%validation_target, &
            0.0_dp, validation_value, validation_gradient, l2_gradient, status)
        if (.not. status_ok(status)) return
        theta_direction = matmul(theta_dot, direction)
        call mlp_loss_hvp(self%model, self%validation_x, self%validation_target, 0.0_dp, &
            theta_direction, 0.0_dp, validation_hvp, scalar_hvp, status)
        if (.not. status_ok(status)) return
        do i = 1, MLP_SCHEDULE_HYPERPARAMETER_COUNT
            product(i) = dot_product(validation_hvp, theta_dot(:, i)) + &
                dot_product(validation_gradient, theta_ddot(:, i))
        end do
        if (any(.not. ieee_is_finite(product))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP schedule hypergradient HVP: product is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine constant_schedule_affine_hvp

    subroutine schedule_hypergradient_fortopt(self, objective, status)
        class(mlp_schedule_hypergradient_objective_t), target, intent(inout) :: self
        type(objective_t), intent(out) :: objective
        type(fortnum_status_t), intent(out) :: status

        if (.not. self%is_initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP schedule hypergradient: objective is not initialized")
            return
        end if
        call objective%initialize_context(MLP_SCHEDULE_HYPERPARAMETER_COUNT, self, &
            schedule_hypergradient_context_callback, status)
    end subroutine schedule_hypergradient_fortopt

    subroutine schedule_hypergradient_context_callback(context, parameters, value, &
            gradient, status)
        class(*), intent(inout) :: context
        real(dp), intent(in) :: parameters(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status

        select type (adapter => context)
            type is (mlp_schedule_hypergradient_objective_t)
            call adapter%value_gradient(parameters, value, gradient, status)
        class default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP schedule hypergradient: context has the wrong type")
        end select
    end subroutine schedule_hypergradient_context_callback

    subroutine mlp_optimize_schedule_hyperparameters(model, train_x, train_target, &
            validation_x, validation_target, options, result, status)
        !! Optimize the four continuous schedule/regularization variables with
        !! FortOpt's projected L-BFGS-B implementation.
        type(mlp_t), target, intent(inout) :: model
        real(dp), intent(in) :: train_x(:, :), train_target(:, :)
        real(dp), intent(in) :: validation_x(:, :), validation_target(:, :)
        type(mlp_schedule_hypergradient_options_t), intent(in) :: options
        type(mlp_schedule_hypergradient_result_t), intent(out) :: result
        type(fortnum_status_t), intent(out) :: status
        type(mlp_schedule_hypergradient_objective_t), target :: adapter
        type(objective_t) :: objective
        type(lbfgsb_t) :: optimizer
        type(lbfgsb_options_t) :: optimizer_options
        type(lbfgsb_result_t) :: optimizer_result
        real(dp) :: parameters(MLP_SCHEDULE_HYPERPARAMETER_COUNT)
        real(dp) :: lower(MLP_SCHEDULE_HYPERPARAMETER_COUNT)
        real(dp) :: upper(MLP_SCHEDULE_HYPERPARAMETER_COUNT)
        real(dp) :: gradient(MLP_SCHEDULE_HYPERPARAMETER_COUNT)
        !! Default-initialized instances, standing in for empty
        !! structure constructors: nvfortran rejects `T()` outright,
        !! and a declared local carries the same default init.
        type(mlp_schedule_hypergradient_result_t) :: mlp_schedule_hypergradient_result_t_default

        result = mlp_schedule_hypergradient_result_t_default
        if (.not. valid_options(options)) then
            if (options%device_kind /= FORTML_DEVICE_CPU) then
                call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                    "MLP schedule hyperparameter optimization: device unsupported")
            else
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP schedule hyperparameter optimization: options invalid")
            end if
            return
        end if
        call adapter%initialize(model, train_x, train_target, validation_x, &
            validation_target, options, status)
        if (.not. status_ok(status)) return
        parameters = adapter%parameters()
        lower = [options%lower_log_base_rate, options%lower_log_l2, &
            options%lower_logit_min_fraction, options%lower_logit_decay_factor]
        upper = [options%upper_log_base_rate, options%upper_log_l2, &
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
        call unpack_parameters(parameters, result%base_rate, result%l2, &
            result%min_rate_fraction, result%decay_factor, status)
        if (.not. status_ok(status)) return
        if (options%schedule%kind == MLP_SCHEDULE_ONE_CYCLE) then
            result%min_rate_fraction = exp(parameters(MLP_SCHEDULE_LOG_PEAK_FRACTION))
            result%decay_factor = exp(parameters(MLP_SCHEDULE_LOG_FINAL_FRACTION))
        end if
        result%log_base_rate = parameters(MLP_SCHEDULE_LOG_BASE_RATE)
        result%log_l2 = parameters(MLP_SCHEDULE_LOG_L2)
        result%logit_min_fraction = parameters(MLP_SCHEDULE_LOGIT_MIN_FRACTION)
        result%logit_decay_factor = parameters(MLP_SCHEDULE_LOGIT_DECAY_FACTOR)
        if (.not. result%converged) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "MLP schedule hyperparameter optimization: iteration limit reached")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_optimize_schedule_hyperparameters

    subroutine reverse_value_gradient(self, parameters, value, gradient, status)
        class(mlp_schedule_hypergradient_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: theta_history(:, :), gradient_history(:, :)
        real(dp), allocatable :: theta_bar(:), gradient_theta(:), hvp(:)
        real(dp), allocatable :: rate_history(:), dbase_history(:)
        real(dp), allocatable :: dmin_history(:), ddecay_history(:)
        real(dp) :: base_rate, l2, min_fraction, decay_factor
        real(dp) :: l2_gradient, validation_l2_gradient, scalar_hvp
        real(dp) :: schedule_rate, dbase, dmin, ddecay, coefficient
        integer :: n_parameters, step

        value = huge(1.0_dp)
        gradient = 0.0_dp
        call unpack_parameters(parameters, base_rate, l2, min_fraction, &
            decay_factor, status)
        if (.not. status_ok(status)) return
        if (self%layout%one_cycle_coordinates) then
            min_fraction = exp(parameters(MLP_SCHEDULE_LOG_PEAK_FRACTION))
            decay_factor = exp(parameters(MLP_SCHEDULE_LOG_FINAL_FRACTION))
            if (.not. ieee_is_finite(min_fraction) .or. &
                .not. ieee_is_finite(decay_factor) .or. min_fraction < 1.0_dp .or. &
                decay_factor <= 0.0_dp .or. decay_factor > min_fraction) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP schedule hypergradient: one-cycle fractions are invalid")
                return
            end if
        end if
        n_parameters = size(self%initial_parameters)
        allocate(theta_history(n_parameters, self%layout%inner_steps + 1))
        allocate(gradient_history(n_parameters, self%layout%inner_steps))
        allocate(rate_history(self%layout%inner_steps))
        allocate(dbase_history(self%layout%inner_steps))
        allocate(dmin_history(self%layout%inner_steps))
        allocate(ddecay_history(self%layout%inner_steps))
        theta_history(:, 1) = self%initial_parameters
        allocate(gradient_theta(n_parameters), hvp(n_parameters))
        do step = 1, self%layout%inner_steps
            call self%model%set_parameters(theta_history(:, step), status)
            if (.not. status_ok(status)) return
            call mlp_loss_value_gradient(self%model, self%train_x, &
                self%train_target, l2, coefficient, gradient_theta, l2_gradient, status)
            if (.not. status_ok(status)) return
            call schedule_at(self%schedule, step, base_rate, min_fraction, decay_factor, &
                schedule_rate, dbase, dmin, ddecay, status)
            if (.not. status_ok(status)) return
            gradient_history(:, step) = gradient_theta
            rate_history(step) = schedule_rate
            dbase_history(step) = dbase
            dmin_history(step) = dmin
            ddecay_history(step) = ddecay
            theta_history(:, step + 1) = theta_history(:, step) - &
                schedule_rate*gradient_theta
            if (any(.not. ieee_is_finite(theta_history(:, step + 1)))) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP schedule hypergradient: trajectory is not finite")
                return
            end if
        end do

        call self%model%set_parameters(theta_history(:, self%layout%inner_steps + 1), &
            status)
        if (.not. status_ok(status)) return
        call mlp_loss_value_gradient(self%model, self%validation_x, &
            self%validation_target, 0.0_dp, value, gradient_theta, &
            validation_l2_gradient, status)
        if (.not. status_ok(status)) return
        allocate(theta_bar(n_parameters))
        theta_bar = gradient_theta
        do step = self%layout%inner_steps, 1, -1
            call self%model%set_parameters(theta_history(:, step), status)
            if (.not. status_ok(status)) return
            call mlp_loss_hvp(self%model, self%train_x, self%train_target, l2, &
                theta_bar, 0.0_dp, hvp, scalar_hvp, status)
            if (.not. status_ok(status)) return
            coefficient = -dot_product(theta_bar, gradient_history(:, step))
            gradient(MLP_SCHEDULE_LOG_BASE_RATE) = &
                gradient(MLP_SCHEDULE_LOG_BASE_RATE) + coefficient* &
                dbase_history(step)*base_rate
            gradient(MLP_SCHEDULE_LOG_L2) = gradient(MLP_SCHEDULE_LOG_L2) - &
                rate_history(step)*dot_product(theta_bar, theta_history(:, step))*l2
            if (self%layout%one_cycle_coordinates) then
                gradient(MLP_SCHEDULE_LOG_PEAK_FRACTION) = &
                    gradient(MLP_SCHEDULE_LOG_PEAK_FRACTION) + coefficient* &
                    dmin_history(step)*min_fraction
                gradient(MLP_SCHEDULE_LOG_FINAL_FRACTION) = &
                    gradient(MLP_SCHEDULE_LOG_FINAL_FRACTION) + coefficient* &
                    ddecay_history(step)*decay_factor
            else
                gradient(MLP_SCHEDULE_LOGIT_MIN_FRACTION) = &
                    gradient(MLP_SCHEDULE_LOGIT_MIN_FRACTION) + coefficient* &
                    dmin_history(step)*min_fraction*(1.0_dp-min_fraction)
                gradient(MLP_SCHEDULE_LOGIT_DECAY_FACTOR) = &
                    gradient(MLP_SCHEDULE_LOGIT_DECAY_FACTOR) + coefficient* &
                    ddecay_history(step)*decay_factor*(1.0_dp-decay_factor)
            end if
            theta_bar = theta_bar - rate_history(step)*hvp
        end do
        if (.not. ieee_is_finite(value) .or. any(.not. ieee_is_finite(gradient))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP schedule hypergradient: product is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine reverse_value_gradient

    subroutine forward_jvp(self, parameters, direction, value, tangent, status)
        class(mlp_schedule_hypergradient_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:), direction(:)
        real(dp), intent(out) :: value, tangent
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: theta(:), theta_dot(:), gradient(:), hvp(:)
        real(dp) :: base_rate, l2, min_fraction, decay_factor
        real(dp) :: base_dot, l2_dot, min_dot, decay_dot, rate_dot
        real(dp) :: schedule_rate, dbase, dmin, ddecay
        real(dp) :: train_value, l2_gradient, validation_l2_gradient
        integer :: n_parameters, step

        value = huge(1.0_dp)
        tangent = 0.0_dp
        call unpack_parameters(parameters, base_rate, l2, min_fraction, &
            decay_factor, status)
        if (.not. status_ok(status)) return
        if (self%layout%one_cycle_coordinates) then
            min_fraction = exp(parameters(MLP_SCHEDULE_LOG_PEAK_FRACTION))
            decay_factor = exp(parameters(MLP_SCHEDULE_LOG_FINAL_FRACTION))
            if (.not. ieee_is_finite(min_fraction) .or. &
                .not. ieee_is_finite(decay_factor) .or. min_fraction < 1.0_dp .or. &
                decay_factor <= 0.0_dp .or. decay_factor > min_fraction) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP schedule hypergradient: one-cycle fractions are invalid")
                return
            end if
        end if
        base_dot = base_rate*direction(MLP_SCHEDULE_LOG_BASE_RATE)
        l2_dot = l2*direction(MLP_SCHEDULE_LOG_L2)
        if (self%layout%one_cycle_coordinates) then
            min_dot = min_fraction*direction(MLP_SCHEDULE_LOG_PEAK_FRACTION)
            decay_dot = decay_factor*direction(MLP_SCHEDULE_LOG_FINAL_FRACTION)
        else
            min_dot = min_fraction*(1.0_dp-min_fraction)* &
                direction(MLP_SCHEDULE_LOGIT_MIN_FRACTION)
            decay_dot = decay_factor*(1.0_dp-decay_factor)* &
                direction(MLP_SCHEDULE_LOGIT_DECAY_FACTOR)
        end if
        n_parameters = size(self%initial_parameters)
        allocate(theta, source=self%initial_parameters)
        allocate(theta_dot(n_parameters), gradient(n_parameters), hvp(n_parameters))
        theta_dot = 0.0_dp
        do step = 1, self%layout%inner_steps
            call self%model%set_parameters(theta, status)
            if (.not. status_ok(status)) return
            call mlp_loss_value_gradient(self%model, self%train_x, &
                self%train_target, l2, train_value, gradient, l2_gradient, status)
            if (.not. status_ok(status)) return
            call mlp_loss_hvp(self%model, self%train_x, self%train_target, l2, &
                theta_dot, l2_dot, hvp, l2_gradient, status)
            if (.not. status_ok(status)) return
            call schedule_at(self%schedule, step, base_rate, min_fraction, decay_factor, &
                schedule_rate, dbase, dmin, ddecay, status)
            if (.not. status_ok(status)) return
            rate_dot = dbase*base_dot + dmin*min_dot + ddecay*decay_dot
            theta_dot = theta_dot - schedule_rate*hvp - rate_dot*gradient
            theta = theta - schedule_rate*gradient
            if (any(.not. ieee_is_finite(theta)) .or. &
                any(.not. ieee_is_finite(theta_dot))) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP schedule hypergradient JVP: trajectory is not finite")
                return
            end if
        end do
        call self%model%set_parameters(theta, status)
        if (.not. status_ok(status)) return
        call mlp_loss_value_gradient(self%model, self%validation_x, &
            self%validation_target, 0.0_dp, value, gradient, &
            validation_l2_gradient, status)
        if (.not. status_ok(status)) return
        tangent = dot_product(gradient, theta_dot)
        if (.not. ieee_is_finite(value) .or. .not. ieee_is_finite(tangent)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP schedule hypergradient JVP: product is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine forward_jvp

    subroutine schedule_at(base_schedule, update, base_rate, min_fraction, &
            decay_factor, rate, dbase, dmin, ddecay, status)
        type(mlp_learning_rate_schedule_t), intent(in) :: base_schedule
        integer, intent(in) :: update
        real(dp), intent(in) :: base_rate, min_fraction, decay_factor
        real(dp), intent(out) :: rate, dbase, dmin, ddecay
        type(fortnum_status_t), intent(out) :: status
        type(mlp_learning_rate_schedule_t) :: schedule
        real(dp) :: dpeak, dfinal

        schedule = base_schedule
        if (base_schedule%kind == MLP_SCHEDULE_ONE_CYCLE) then
            schedule%peak_rate_fraction = min_fraction
            schedule%final_rate_fraction = decay_factor
        else
            schedule%min_rate_fraction = min_fraction
            schedule%decay_factor = decay_factor
        end if
        ! The legacy three-derivative wrapper intentionally omits the peak and
        ! final-fraction products used by one-cycle schedules.  Request the
        ! complete derivative set and map those coordinates onto the common
        ! hypergradient slots.
        call schedule%rate_with_full_derivatives(update, base_rate, rate, dbase, dmin, &
            ddecay, dpeak, dfinal, status)
        if (base_schedule%kind == MLP_SCHEDULE_ONE_CYCLE) then
            dmin = dpeak
            ddecay = dfinal
        end if
    end subroutine schedule_at

    subroutine unpack_parameters(parameters, base_rate, l2, min_fraction, &
            decay_factor, status)
        real(dp), intent(in) :: parameters(:)
        real(dp), intent(out) :: base_rate, l2, min_fraction, decay_factor
        type(fortnum_status_t), intent(out) :: status

        base_rate = 0.0_dp
        l2 = 0.0_dp
        min_fraction = 0.0_dp
        decay_factor = 0.0_dp
        if (size(parameters) /= MLP_SCHEDULE_HYPERPARAMETER_COUNT .or. &
            any(.not. ieee_is_finite(parameters))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP schedule hypergradient: packed values are invalid")
            return
        end if
        base_rate = exp(parameters(MLP_SCHEDULE_LOG_BASE_RATE))
        l2 = exp(parameters(MLP_SCHEDULE_LOG_L2))
        ! The one-cycle branch is unpacked by the trajectory caller because
        ! this helper intentionally remains schedule-family agnostic.
        min_fraction = sigmoid(parameters(MLP_SCHEDULE_LOGIT_MIN_FRACTION))
        decay_factor = sigmoid(parameters(MLP_SCHEDULE_LOGIT_DECAY_FACTOR))
        if (.not. ieee_is_finite(base_rate) .or. .not. ieee_is_finite(l2) .or. &
            base_rate <= 0.0_dp .or. l2 <= 0.0_dp .or. &
            .not. ieee_is_finite(min_fraction) .or. &
            .not. ieee_is_finite(decay_factor)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP schedule hypergradient: transformed values are invalid")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine unpack_parameters

    logical function valid_options(options) result(valid)
        type(mlp_schedule_hypergradient_options_t), intent(in) :: options

        valid = options%steps >= 1 .and. options%device_kind == FORTML_DEVICE_CPU .and. &
            options%memory >= 1 .and. options%max_iterations >= 1 .and. &
            options%max_line_search >= 1 .and. options%base_rate > 0.0_dp .and. &
            options%l2 > 0.0_dp .and. options%schedule%valid() .and. &
            ieee_is_finite(options%base_rate) .and. ieee_is_finite(options%l2)
        if (.not. valid) return
        valid = finite_bounds(options%lower_log_base_rate, options%upper_log_base_rate) &
            .and. options%lower_log_base_rate <= options%upper_log_base_rate
        valid = valid .and. finite_bounds(options%lower_log_l2, options%upper_log_l2) &
            .and. options%lower_log_l2 <= options%upper_log_l2
        valid = valid .and. finite_bounds(options%lower_logit_min_fraction, &
            options%upper_logit_min_fraction) .and. &
            options%lower_logit_min_fraction <= options%upper_logit_min_fraction
        valid = valid .and. finite_bounds(options%lower_logit_decay_factor, &
            options%upper_logit_decay_factor) .and. &
            options%lower_logit_decay_factor <= options%upper_logit_decay_factor
        valid = valid .and. ieee_is_finite(options%gradient_tolerance) .and. &
            ieee_is_finite(options%step_tolerance) .and. &
            ieee_is_finite(options%objective_tolerance) .and. &
            options%gradient_tolerance >= 0.0_dp .and. &
            options%step_tolerance >= 0.0_dp .and. options%objective_tolerance >= 0.0_dp
    end function valid_options

    logical function valid_data(model, x, target) result(valid)
        class(mlp_t), intent(in) :: model
        real(dp), intent(in) :: x(:, :), target(:, :)

        valid = model%parameter_count() > 0
        if (.not. valid) return
        if (.not. allocated(model%layer_sizes)) then
            valid = .false.
            return
        end if
        valid = size(x, 1) > 0
        if (.not. valid) return
        valid = size(x, 2) == model%layer_sizes(1)
        if (.not. valid) return
        valid = size(target, 1) == size(x, 1)
        if (.not. valid) return
        valid = size(target, 2) == model%layer_sizes(size(model%layer_sizes))
        if (.not. valid) return
        valid = all(ieee_is_finite(x)) .and. all(ieee_is_finite(target))
    end function valid_data

    logical function valid_parameter_vector(parameters, gradient) result(valid)
        real(dp), intent(in) :: parameters(:), gradient(:)

        valid = size(parameters) == MLP_SCHEDULE_HYPERPARAMETER_COUNT .and. &
            size(gradient) == MLP_SCHEDULE_HYPERPARAMETER_COUNT .and. &
            all(ieee_is_finite(parameters))
    end function valid_parameter_vector

    logical function affine_one_layer(model) result(valid)
        !! Whether the model is a single dense affine map.
        class(mlp_t), intent(in) :: model

        valid = allocated(model%layer_sizes) .and. allocated(model%layer)
        if (.not. valid) return
        ! With one layer there is no hidden activation to apply; only the
        ! output activation controls whether the map is affine.
        valid = size(model%layer_sizes) == 2 .and. size(model%layer) == 1 .and. &
            model%output_activation == MLP_LINEAR
    end function affine_one_layer

    logical function finite_bounds(lower, upper) result(valid)
        real(dp), intent(in) :: lower, upper

        valid = ieee_is_finite(lower) .and. ieee_is_finite(upper)
    end function finite_bounds

    real(dp) function sigmoid(value) result(output)
        real(dp), intent(in) :: value

        if (value >= 0.0_dp) then
            output = 1.0_dp/(1.0_dp + exp(-value))
        else
            output = exp(value)/(1.0_dp + exp(value))
        end if
    end function sigmoid

    real(dp) function logit(value) result(output)
        real(dp), intent(in) :: value

        output = log(value/(1.0_dp-value))
    end function logit

    real(dp) function interior_probability(value) result(output)
        real(dp), intent(in) :: value

        output = min(1.0_dp-1.0e-8_dp, max(1.0e-8_dp, value))
    end function interior_probability

end module fortml_mlp_schedule_hypergradient
