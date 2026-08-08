module fortml_mlp_optimizer_group_hypergradient
    !! Fixed full-batch MLP trajectory products for learning-rate groups.
    !!
    !! The production trainer applies each optimizer group's multiplier to its
    !! post-optimizer update.  This objective exposes the same contract to a
    !! differentiable outer search.  The packed variable is
    !! `[log(learning_rate), log(l2), log(multiplier_1), ...]`; group ranges
    !! and names are fixed discrete metadata captured at initialization.
    !!
    !! The trajectory is plain full-batch SGD, which is the first complete
    !! group-product adapter.  MLP analytic HVPs propagate every smooth outer
    !! coordinate, and FortOpt consumes the resulting value/gradient callback.
    !! The outer HVP is intentionally a typed refusal: producing it would
    !! require third network derivatives (the inner recurrence already uses
    !! an MLP HVP).  Keeping the method on the public objective gives callers
    !! one stable derivative surface and prevents accidental finite-difference
    !! fallbacks.
    !! CUDA is an explicit refusal until the complete model, update, and group
    !! metadata are resident on a device.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED, FORTNUM_CONVERGENCE_ERROR
    use fortml_device, only: FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    use fortml_mlp, only: mlp_t
    use fortml_mlp_training, only: mlp_loss_value_gradient, mlp_loss_hvp, &
        mlp_optimizer_group_t, MLP_OPTIMIZER_SGD
    use fortopt_objective, only: objective_t
    use fortopt_lbfgsb, only: lbfgsb_t, lbfgsb_options_t, lbfgsb_result_t
    implicit none
    private

    integer, parameter, public :: MLP_OPTIMIZER_GROUP_BASE_COUNT = 2
    integer, parameter, public :: MLP_OPTIMIZER_GROUP_LOG_LEARNING_RATE = 1
    integer, parameter, public :: MLP_OPTIMIZER_GROUP_LOG_L2 = 2
    integer, parameter, public :: MLP_OPTIMIZER_GROUP_FIRST_LOG_MULTIPLIER = 3
    integer, parameter, public :: MLP_OPTIMIZER_GROUP_OPTIMIZER = MLP_OPTIMIZER_SGD
    real(dp), parameter :: LOG_PARAMETER_MIN = -40.0_dp
    real(dp), parameter :: LOG_PARAMETER_MAX = 40.0_dp

    type, public :: mlp_optimizer_group_hypergradient_metadata_t
        integer :: parameter_count = 0
        integer :: group_count = 0
        integer :: inner_steps = 0
        integer :: log_learning_rate_index = MLP_OPTIMIZER_GROUP_LOG_LEARNING_RATE
        integer :: log_l2_index = MLP_OPTIMIZER_GROUP_LOG_L2
        integer :: first_log_multiplier_index = MLP_OPTIMIZER_GROUP_FIRST_LOG_MULTIPLIER
    end type mlp_optimizer_group_hypergradient_metadata_t

    type, public :: mlp_optimizer_group_hypergradient_options_t
        integer :: steps = 8
        real(dp) :: learning_rate = 1.0e-2_dp
        real(dp) :: l2 = 1.0e-4_dp
        type(mlp_optimizer_group_t), allocatable :: groups(:)
        real(dp) :: lower_log_learning_rate = -16.0_dp
        real(dp) :: upper_log_learning_rate = 1.0_dp
        real(dp) :: lower_log_l2 = -24.0_dp
        real(dp) :: upper_log_l2 = 1.0_dp
        real(dp) :: lower_log_multiplier = -8.0_dp
        real(dp) :: upper_log_multiplier = 8.0_dp
        integer :: optimizer = MLP_OPTIMIZER_GROUP_OPTIMIZER
        integer :: device_kind = FORTML_DEVICE_CPU
        integer :: memory = 8
        integer :: max_iterations = 100
        integer :: max_line_search = 40
        real(dp) :: gradient_tolerance = 1.0e-6_dp
        real(dp) :: step_tolerance = 1.0e-12_dp
        real(dp) :: objective_tolerance = 1.0e-12_dp
    end type mlp_optimizer_group_hypergradient_options_t

    type, public :: mlp_optimizer_group_hypergradient_result_t
        logical :: converged = .false.
        integer :: iterations = 0
        integer :: line_search_evaluations = 0
        real(dp) :: objective = huge(1.0_dp)
        real(dp) :: gradient_norm = huge(1.0_dp)
        real(dp) :: log_learning_rate = 0.0_dp
        real(dp) :: log_l2 = 0.0_dp
        real(dp) :: learning_rate = 0.0_dp
        real(dp) :: l2 = 0.0_dp
        real(dp), allocatable :: log_multiplier(:)
        real(dp), allocatable :: multiplier(:)
    end type mlp_optimizer_group_hypergradient_result_t

    type, public :: mlp_optimizer_group_hypergradient_objective_t
        private
        type(mlp_t), pointer :: model => null()
        real(dp), allocatable :: train_x(:, :), train_target(:, :)
        real(dp), allocatable :: validation_x(:, :), validation_target(:, :)
        real(dp), allocatable :: initial_parameters(:)
        type(mlp_optimizer_group_t), allocatable :: groups(:)
        type(mlp_optimizer_group_hypergradient_metadata_t) :: layout
        real(dp) :: initial_log_learning_rate = 0.0_dp
        real(dp) :: initial_log_l2 = 0.0_dp
        logical :: initialized = .false.
    contains
        procedure, public :: initialize => mlp_optimizer_group_hypergradient_initialize
        procedure, public :: parameter_count => mlp_optimizer_group_hypergradient_parameter_count
        procedure, public :: group_count => mlp_optimizer_group_hypergradient_group_count
        procedure, public :: group_name => mlp_optimizer_group_hypergradient_group_name
        procedure, public :: group_range => mlp_optimizer_group_hypergradient_group_range
        procedure, public :: metadata => mlp_optimizer_group_hypergradient_metadata
        procedure, public :: parameters => mlp_optimizer_group_hypergradient_parameters
        procedure, public :: value_gradient => mlp_optimizer_group_hypergradient_value_gradient
        procedure, public :: jvp => mlp_optimizer_group_hypergradient_jvp
        procedure, public :: vjp => mlp_optimizer_group_hypergradient_vjp
        procedure, public :: hvp => mlp_optimizer_group_hypergradient_hvp
        procedure, public :: fortopt => mlp_optimizer_group_hypergradient_fortopt
        procedure, public :: is_initialized => mlp_optimizer_group_hypergradient_is_initialized
    end type mlp_optimizer_group_hypergradient_objective_t

    public :: mlp_optimize_optimizer_group_hyperparameters

contains

    subroutine mlp_optimizer_group_hypergradient_initialize(self, model, train_x, &
            train_target, validation_x, validation_target, options, status)
        class(mlp_optimizer_group_hypergradient_objective_t), intent(out) :: self
        type(mlp_t), target, intent(inout) :: model
        real(dp), intent(in) :: train_x(:, :), train_target(:, :)
        real(dp), intent(in) :: validation_x(:, :), validation_target(:, :)
        type(mlp_optimizer_group_hypergradient_options_t), intent(in) :: options
        type(fortnum_status_t), intent(out) :: status
        integer :: i, j, n_model, n_groups

        self%initialized = .false.
        self%layout = mlp_optimizer_group_hypergradient_metadata_t()
        if (.not. valid_options(options)) then
            if (options%optimizer /= MLP_OPTIMIZER_GROUP_OPTIMIZER .or. &
                options%device_kind == FORTML_DEVICE_CUDA) then
                call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                    "MLP optimizer-group hypergradient: optimizer or device is unsupported")
            else
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP optimizer-group hypergradient: options are invalid")
            end if
            return
        end if
        if (.not. valid_data(model, train_x, train_target) .or. &
            .not. valid_data(model, validation_x, validation_target)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP optimizer-group hypergradient: model or data is invalid")
            return
        end if
        n_model = model%parameter_count()
        if (.not. allocated(options%groups)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP optimizer-group hypergradient: groups are required")
            return
        end if
        n_groups = size(options%groups)
        if (n_groups < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP optimizer-group hypergradient: groups are required")
            return
        end if
        do i = 1, n_groups
            if (.not. options%groups(i)%initialized() .or. &
                options%groups(i)%last > n_model) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP optimizer-group hypergradient: group range is invalid")
                return
            end if
            if (log(options%groups(i)%learning_rate_multiplier) < &
                options%lower_log_multiplier .or. &
                log(options%groups(i)%learning_rate_multiplier) > &
                options%upper_log_multiplier) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP optimizer-group hypergradient: multiplier is outside bounds")
                return
            end if
            do j = 1, i - 1
                if (trim(options%groups(i)%name) == trim(options%groups(j)%name) .or. &
                    ranges_overlap(options%groups(i), options%groups(j))) then
                    call status_set(status, FORTNUM_DOMAIN_ERROR, &
                        "MLP optimizer-group hypergradient: names and ranges must be unique")
                    return
                end if
            end do
        end do

        self%model => model
        allocate(self%train_x, source=train_x)
        allocate(self%train_target, source=train_target)
        allocate(self%validation_x, source=validation_x)
        allocate(self%validation_target, source=validation_target)
        allocate(self%initial_parameters, source=model%parameters())
        allocate(self%groups, source=options%groups)
        self%layout%group_count = n_groups
        self%layout%parameter_count = MLP_OPTIMIZER_GROUP_BASE_COUNT + n_groups
        self%layout%inner_steps = options%steps
        self%initial_log_learning_rate = log(options%learning_rate)
        self%initial_log_l2 = log(options%l2)
        self%initialized = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_optimizer_group_hypergradient_initialize

    integer function mlp_optimizer_group_hypergradient_parameter_count(self) result(count)
        class(mlp_optimizer_group_hypergradient_objective_t), intent(in) :: self

        count = 0
        if (self%initialized) count = self%layout%parameter_count
    end function mlp_optimizer_group_hypergradient_parameter_count

    integer function mlp_optimizer_group_hypergradient_group_count(self) result(count)
        class(mlp_optimizer_group_hypergradient_objective_t), intent(in) :: self

        count = 0
        if (allocated(self%groups)) count = size(self%groups)
    end function mlp_optimizer_group_hypergradient_group_count

    function mlp_optimizer_group_hypergradient_group_name(self, index) result(name)
        class(mlp_optimizer_group_hypergradient_objective_t), intent(in) :: self
        integer, intent(in) :: index
        character(len=64) :: name

        name = ""
        if (.not. allocated(self%groups)) return
        if (index < 1 .or. index > size(self%groups)) return
        name = self%groups(index)%name
    end function mlp_optimizer_group_hypergradient_group_name

    subroutine mlp_optimizer_group_hypergradient_group_range(self, index, first, last, status)
        class(mlp_optimizer_group_hypergradient_objective_t), intent(in) :: self
        integer, intent(in) :: index
        integer, intent(out) :: first, last
        type(fortnum_status_t), intent(out) :: status

        first = 0
        last = -1
        if (.not. allocated(self%groups)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP optimizer-group hypergradient: groups are not initialized")
            return
        end if
        if (index < 1 .or. index > size(self%groups)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP optimizer-group hypergradient: group index is invalid")
            return
        end if
        first = self%groups(index)%first
        last = self%groups(index)%last
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_optimizer_group_hypergradient_group_range

    function mlp_optimizer_group_hypergradient_metadata(self) result(layout)
        class(mlp_optimizer_group_hypergradient_objective_t), intent(in) :: self
        type(mlp_optimizer_group_hypergradient_metadata_t) :: layout

        layout = self%layout
    end function mlp_optimizer_group_hypergradient_metadata

    function mlp_optimizer_group_hypergradient_parameters(self) result(parameters)
        class(mlp_optimizer_group_hypergradient_objective_t), intent(in) :: self
        real(dp), allocatable :: parameters(:)
        integer :: i

        allocate(parameters(self%layout%parameter_count))
        parameters = 0.0_dp
        if (.not. self%initialized) return
        parameters(MLP_OPTIMIZER_GROUP_LOG_LEARNING_RATE) = self%initial_log_learning_rate
        parameters(MLP_OPTIMIZER_GROUP_LOG_L2) = self%initial_log_l2
        do i = 1, self%layout%group_count
            parameters(MLP_OPTIMIZER_GROUP_FIRST_LOG_MULTIPLIER + i - 1) = &
                log(self%groups(i)%learning_rate_multiplier)
        end do
    end function mlp_optimizer_group_hypergradient_parameters

    logical function mlp_optimizer_group_hypergradient_is_initialized(self) result(yes)
        class(mlp_optimizer_group_hypergradient_objective_t), intent(in) :: self

        yes = self%initialized .and. associated(self%model) .and. &
            allocated(self%initial_parameters) .and. allocated(self%groups)
    end function mlp_optimizer_group_hypergradient_is_initialized

    subroutine mlp_optimizer_group_hypergradient_value_gradient(self, parameters, value, &
            gradient, status)
        class(mlp_optimizer_group_hypergradient_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: direction(self%layout%parameter_count), tangent

        value = huge(1.0_dp)
        gradient = 0.0_dp
        direction = 0.0_dp
        if (.not. self%is_initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP optimizer-group hypergradient: objective is not initialized")
            return
        end if
        if (.not. valid_parameter_vector(parameters, gradient, self%layout%parameter_count)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP optimizer-group hypergradient: packed shape is invalid")
            return
        end if
        call optimizer_group_forward(self, parameters, direction, value, tangent, gradient, status)
    end subroutine mlp_optimizer_group_hypergradient_value_gradient

    subroutine mlp_optimizer_group_hypergradient_jvp(self, parameters, direction, value, &
            tangent, status)
        class(mlp_optimizer_group_hypergradient_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:), direction(:)
        real(dp), intent(out) :: value, tangent
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: gradient(:)

        value = huge(1.0_dp)
        tangent = 0.0_dp
        if (.not. self%is_initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP optimizer-group hypergradient JVP: objective is not initialized")
            return
        end if
        if (size(direction) /= self%layout%parameter_count .or. &
            any(.not. ieee_is_finite(direction))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP optimizer-group hypergradient JVP: direction is invalid")
            return
        end if
        allocate(gradient(self%layout%parameter_count))
        call optimizer_group_forward(self, parameters, direction, value, tangent, gradient, status)
    end subroutine mlp_optimizer_group_hypergradient_jvp

    subroutine mlp_optimizer_group_hypergradient_vjp(self, parameters, output_bar, &
            gradient, status)
        class(mlp_optimizer_group_hypergradient_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:), output_bar
        real(dp), intent(out) :: gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: value

        gradient = 0.0_dp
        if (.not. ieee_is_finite(output_bar)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP optimizer-group hypergradient VJP: cotangent is invalid")
            return
        end if
        call self%value_gradient(parameters, value, gradient, status)
        if (status%code /= FORTNUM_OK) return
        gradient = output_bar*gradient
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_optimizer_group_hypergradient_vjp

    subroutine mlp_optimizer_group_hypergradient_hvp(self, parameters, direction, product, status)
        !! Return a typed refusal until third network derivatives are available.
        !!
        !! `value_gradient`, `jvp`, and `vjp` are exact.  An outer HVP would
        !! differentiate the inner MLP HVP with respect to the trajectory and
        !! therefore needs third derivatives of the network loss.  Returning
        !! zero with `FORTNUM_NOT_IMPLEMENTED` makes the unsupported product
        !! explicit and keeps FortOpt from silently consuming a numerical
        !! approximation.
        class(mlp_optimizer_group_hypergradient_objective_t), intent(in) :: self
        real(dp), intent(in) :: parameters(:), direction(:)
        real(dp), intent(out) :: product(:)
        type(fortnum_status_t), intent(out) :: status

        product = 0.0_dp
        if (size(parameters) /= self%layout%parameter_count .or. &
                size(direction) /= self%layout%parameter_count .or. &
                size(product) /= self%layout%parameter_count .or. &
                any(.not. ieee_is_finite(parameters)) .or. &
                any(.not. ieee_is_finite(direction))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP optimizer-group hypergradient HVP: packed shape is invalid")
            return
        end if
        if (.not. self%is_initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP optimizer-group hypergradient HVP: objective is not initialized")
            return
        end if
        call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
            "MLP optimizer-group hypergradient HVP requires third network derivatives")
    end subroutine mlp_optimizer_group_hypergradient_hvp

    subroutine mlp_optimizer_group_hypergradient_fortopt(self, objective, status)
        class(mlp_optimizer_group_hypergradient_objective_t), target, intent(inout) :: self
        type(objective_t), intent(out) :: objective
        type(fortnum_status_t), intent(out) :: status

        if (.not. self%is_initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP optimizer-group hypergradient: objective is not initialized")
            return
        end if
        call objective%initialize_context(self%layout%parameter_count, self, &
            optimizer_group_context_callback, status)
    end subroutine mlp_optimizer_group_hypergradient_fortopt

    subroutine optimizer_group_context_callback(context, parameters, value, gradient, status)
        class(*), intent(inout) :: context
        real(dp), intent(in) :: parameters(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status

        select type (adapter => context)
            type is (mlp_optimizer_group_hypergradient_objective_t)
            call adapter%value_gradient(parameters, value, gradient, status)
        class default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP optimizer-group hypergradient: context has the wrong type")
        end select
    end subroutine optimizer_group_context_callback

    subroutine mlp_optimize_optimizer_group_hyperparameters(model, train_x, train_target, &
            validation_x, validation_target, options, result, status)
        type(mlp_t), target, intent(inout) :: model
        real(dp), intent(in) :: train_x(:, :), train_target(:, :)
        real(dp), intent(in) :: validation_x(:, :), validation_target(:, :)
        type(mlp_optimizer_group_hypergradient_options_t), intent(in) :: options
        type(mlp_optimizer_group_hypergradient_result_t), intent(out) :: result
        type(fortnum_status_t), intent(out) :: status
        type(mlp_optimizer_group_hypergradient_objective_t), target :: adapter
        type(objective_t) :: objective
        type(lbfgsb_t) :: optimizer
        type(lbfgsb_options_t) :: optimizer_options
        type(lbfgsb_result_t) :: optimizer_result
        real(dp), allocatable :: parameters(:), lower(:), upper(:), gradient(:)
        integer :: n_parameters, n_groups, i

        result = mlp_optimizer_group_hypergradient_result_t()
        if (.not. valid_options(options)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP optimizer-group hyperparameter optimization: options are invalid")
            return
        end if
        call adapter%initialize(model, train_x, train_target, validation_x, &
            validation_target, options, status)
        if (status%code /= FORTNUM_OK) return
        n_parameters = adapter%parameter_count()
        n_groups = adapter%group_count()
        allocate(parameters(n_parameters), lower(n_parameters), upper(n_parameters), &
            gradient(n_parameters), result%log_multiplier(n_groups), result%multiplier(n_groups))
        parameters = adapter%parameters()
        lower(1) = options%lower_log_learning_rate
        upper(1) = options%upper_log_learning_rate
        lower(2) = options%lower_log_l2
        upper(2) = options%upper_log_l2
        do i = 1, n_groups
            lower(2+i) = options%lower_log_multiplier
            upper(2+i) = options%upper_log_multiplier
        end do
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
        result%learning_rate = exp(result%log_learning_rate)
        result%l2 = exp(result%log_l2)
        result%log_multiplier = parameters(3:)
        result%multiplier = exp(result%log_multiplier)
        if (.not. result%converged) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "MLP optimizer-group hyperparameter optimization: iteration limit reached")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_optimize_optimizer_group_hyperparameters

    subroutine optimizer_group_forward(self, parameters, direction, value, tangent, &
            gradient, status)
        class(mlp_optimizer_group_hypergradient_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:), direction(:)
        real(dp), intent(out) :: value, tangent, gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: theta(:), theta_dot(:, :), raw_gradient(:)
        real(dp), allocatable :: gradient_dot(:, :), hvp(:), scale(:), scale_dot(:, :)
        real(dp), allocatable :: validation_gradient(:)
        real(dp) :: learning_rate, l2, train_value, l2_gradient, scalar_hvp
        real(dp) :: learning_rate_dot, l2_dot, multiplier
        integer :: n_model, n_outer, step, parameter_index, group_index, first, last

        value = huge(1.0_dp)
        tangent = 0.0_dp
        gradient = 0.0_dp
        if (.not. valid_trajectory_parameters(self, parameters, learning_rate, l2)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP optimizer-group hypergradient: packed values are invalid")
            return
        end if
        n_model = size(self%initial_parameters)
        n_outer = self%layout%parameter_count
        allocate(theta, source=self%initial_parameters)
        allocate(theta_dot(n_model, n_outer), raw_gradient(n_model))
        allocate(gradient_dot(n_model, n_outer), hvp(n_model))
        allocate(scale(n_model), scale_dot(n_model, n_outer))
        theta_dot = 0.0_dp
        scale = 1.0_dp
        scale_dot = 0.0_dp
        do group_index = 1, self%layout%group_count
            first = self%groups(group_index)%first
            last = self%groups(group_index)%last
            multiplier = exp(parameters(MLP_OPTIMIZER_GROUP_FIRST_LOG_MULTIPLIER + &
                group_index - 1))
            scale(first:last) = multiplier
            scale_dot(first:last, MLP_OPTIMIZER_GROUP_FIRST_LOG_MULTIPLIER + &
                group_index - 1) = multiplier
        end do

        do step = 1, self%layout%inner_steps
            call self%model%set_parameters(theta, status)
            if (status%code /= FORTNUM_OK) return
            call mlp_loss_value_gradient(self%model, self%train_x, self%train_target, &
                l2, train_value, raw_gradient, l2_gradient, status)
            if (status%code /= FORTNUM_OK) return
            do parameter_index = 1, n_outer
                l2_dot = 0.0_dp
                if (parameter_index == MLP_OPTIMIZER_GROUP_LOG_L2) l2_dot = l2
                call mlp_loss_hvp(self%model, self%train_x, self%train_target, l2, &
                    theta_dot(:, parameter_index), l2_dot, hvp, scalar_hvp, status)
                if (status%code /= FORTNUM_OK) return
                gradient_dot(:, parameter_index) = hvp
                learning_rate_dot = 0.0_dp
                if (parameter_index == MLP_OPTIMIZER_GROUP_LOG_LEARNING_RATE) then
                    learning_rate_dot = learning_rate
                end if
                theta_dot(:, parameter_index) = theta_dot(:, parameter_index) - &
                    learning_rate_dot*scale*raw_gradient - &
                    learning_rate*scale_dot(:, parameter_index)*raw_gradient - &
                    learning_rate*scale*gradient_dot(:, parameter_index)
            end do
            theta = theta-learning_rate*scale*raw_gradient
            if (any(.not. ieee_is_finite(theta)) .or. &
                any(.not. ieee_is_finite(theta_dot))) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP optimizer-group hypergradient: trajectory is not finite")
                return
            end if
        end do

        call self%model%set_parameters(theta, status)
        if (status%code /= FORTNUM_OK) return
        allocate(validation_gradient(n_model))
        call mlp_loss_value_gradient(self%model, self%validation_x, &
            self%validation_target, 0.0_dp, value, validation_gradient, &
            l2_gradient, status)
        if (status%code /= FORTNUM_OK) return
        do parameter_index = 1, n_outer
            gradient(parameter_index) = dot_product(validation_gradient, &
                theta_dot(:, parameter_index))
        end do
        tangent = dot_product(gradient, direction)
        if (.not. ieee_is_finite(value) .or. any(.not. ieee_is_finite(gradient)) .or. &
            .not. ieee_is_finite(tangent)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP optimizer-group hypergradient: objective product is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine optimizer_group_forward

    logical function valid_trajectory_parameters(self, parameters, learning_rate, l2) result(valid)
        class(mlp_optimizer_group_hypergradient_objective_t), intent(in) :: self
        real(dp), intent(in) :: parameters(:)
        real(dp), intent(out) :: learning_rate, l2

        learning_rate = 0.0_dp
        l2 = 0.0_dp
        valid = .false.
        if (size(parameters) /= self%layout%parameter_count) return
        if (any(.not. ieee_is_finite(parameters))) return
        if (any(parameters < LOG_PARAMETER_MIN) .or. any(parameters > LOG_PARAMETER_MAX)) return
        learning_rate = exp(parameters(1))
        l2 = exp(parameters(2))
        valid = ieee_is_finite(learning_rate) .and. ieee_is_finite(l2) .and. &
            learning_rate > 0.0_dp .and. l2 > 0.0_dp
    end function valid_trajectory_parameters

    logical function valid_parameter_vector(parameters, gradient, count) result(valid)
        real(dp), intent(in) :: parameters(:), gradient(:)
        integer, intent(in) :: count

        valid = size(parameters) == count .and. size(gradient) == count .and. &
            all(ieee_is_finite(parameters))
    end function valid_parameter_vector

    logical function valid_options(options) result(valid)
        type(mlp_optimizer_group_hypergradient_options_t), intent(in) :: options

        valid = options%steps >= 1 .and. options%optimizer == MLP_OPTIMIZER_GROUP_OPTIMIZER .and. &
            options%device_kind == FORTML_DEVICE_CPU .and. &
            ieee_is_finite(options%learning_rate) .and. ieee_is_finite(options%l2) .and. &
            options%learning_rate > 0.0_dp .and. options%l2 > 0.0_dp .and. &
            ieee_is_finite(options%lower_log_learning_rate) .and. &
            ieee_is_finite(options%upper_log_learning_rate) .and. &
            ieee_is_finite(options%lower_log_l2) .and. ieee_is_finite(options%upper_log_l2) .and. &
            ieee_is_finite(options%lower_log_multiplier) .and. &
            ieee_is_finite(options%upper_log_multiplier) .and. &
            options%lower_log_learning_rate <= options%upper_log_learning_rate .and. &
            options%lower_log_l2 <= options%upper_log_l2 .and. &
            options%lower_log_multiplier <= options%upper_log_multiplier .and. &
            log(options%learning_rate) >= options%lower_log_learning_rate .and. &
            log(options%learning_rate) <= options%upper_log_learning_rate .and. &
            log(options%l2) >= options%lower_log_l2 .and. log(options%l2) <= options%upper_log_l2 .and. &
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

    logical function ranges_overlap(a, b) result(overlap)
        type(mlp_optimizer_group_t), intent(in) :: a, b

        overlap = a%first <= b%last .and. b%first <= a%last
    end function ranges_overlap

end module fortml_mlp_optimizer_group_hypergradient
