module fortml_mlp_poisson
    !! Differentiable Poisson negative-log-likelihood objective for MLPs.
    !!
    !! The network emits one log-rate per sample.  The objective delegates
    !! the scalar loss derivatives to the stable Poisson products in
    !! `fortml_losses` and composes them with the MLP JVP/VJP/HVP products.
    !! An optional final packed coordinate exposes the non-negative L2
    !! coefficient to FortOpt, including the exact mixed HVP block.
    !!
    !! This is a CPU objective.  A requested CUDA device is rejected before
    !! state is installed; no host fallback is hidden behind a device flag.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, status_ok, &
        FORTNUM_OK, FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED, &
        FORTNUM_CONVERGENCE_ERROR
    use fortml_device, only: FORTML_DEVICE_CPU
    use fortml_mlp, only: mlp_t
    use fortml_losses, only: poisson_nll_value, poisson_nll_vjp, poisson_nll_hvp
    use fortopt_objective, only: objective_t
    use fortopt_lbfgsb, only: lbfgsb_t, lbfgsb_options_t, lbfgsb_result_t
    implicit none
    private

    type, public :: mlp_poisson_training_objective_t
        !! Weighted Poisson NLL over a live MLP parameter vector.
        private
        type(mlp_t), pointer :: model => null()
        real(dp), allocatable :: features(:, :), targets(:, :), weights(:)
        real(dp) :: l2 = 0.0_dp
        logical :: optimize_l2 = .false.
    contains
        procedure, public :: initialize => mlp_poisson_objective_initialize
        procedure, public :: parameter_count => mlp_poisson_objective_parameter_count
        procedure, public :: parameters => mlp_poisson_objective_parameters
        procedure, public :: value_gradient => mlp_poisson_objective_value_gradient
        procedure, public :: jvp => mlp_poisson_objective_jvp
        procedure, public :: vjp => mlp_poisson_objective_vjp
        procedure, public :: hvp => mlp_poisson_objective_hvp
        procedure, public :: fortopt => mlp_poisson_objective_fortopt
    end type mlp_poisson_training_objective_t

    type, public :: mlp_poisson_lbfgsb_options_t
        !! Bounds and convergence controls for Poisson MLP L-BFGS-B.
        integer :: memory = 10
        integer :: max_iterations = 100
        integer :: max_line_search = 40
        real(dp) :: gradient_tolerance = 1.0e-8_dp
        real(dp) :: step_tolerance = 1.0e-12_dp
        real(dp) :: objective_tolerance = 1.0e-12_dp
        real(dp) :: lower_bound = -20.0_dp
        real(dp) :: upper_bound = 20.0_dp
        real(dp) :: l2 = 0.0_dp
        real(dp) :: l2_lower_bound = 0.0_dp
        real(dp) :: l2_upper_bound = 20.0_dp
        logical :: optimize_l2 = .false.
        integer :: device_kind = FORTML_DEVICE_CPU
    end type mlp_poisson_lbfgsb_options_t

    type, public :: mlp_poisson_lbfgsb_result_t
        !! Diagnostics returned by `mlp_poisson_optimize_lbfgsb`.
        logical :: converged = .false.
        integer :: iterations = 0
        integer :: line_search_evaluations = 0
        real(dp) :: objective = huge(1.0_dp)
        real(dp) :: gradient_norm = huge(1.0_dp)
        real(dp) :: l2 = 0.0_dp
    end type mlp_poisson_lbfgsb_result_t

    public :: mlp_poisson_loss_gradient
    public :: mlp_poisson_loss_hvp
    public :: mlp_poisson_optimize_lbfgsb

contains

    subroutine mlp_poisson_objective_initialize(self, model, x, targets, l2, &
            status, optimize_l2, sample_weight, device_kind)
        class(mlp_poisson_training_objective_t), intent(out) :: self
        type(mlp_t), target, intent(inout) :: model
        real(dp), intent(in) :: x(:, :), targets(:, :), l2
        type(fortnum_status_t), intent(out) :: status
        logical, intent(in), optional :: optimize_l2
        real(dp), intent(in), optional :: sample_weight(:)
        integer, intent(in), optional :: device_kind
        integer :: requested_device
        real(dp) :: weight_mass

        self%l2 = 0.0_dp
        self%optimize_l2 = .false.
        requested_device = FORTML_DEVICE_CPU
        if (present(device_kind)) requested_device = device_kind
        if (requested_device /= FORTML_DEVICE_CPU) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "MLP Poisson objective: resident device path is unavailable")
            return
        end if
        if (present(optimize_l2)) self%optimize_l2 = optimize_l2
        if (.not. allocated(model%layer_sizes)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP Poisson objective: model has no initialized topology")
            return
        end if
        if (size(model%layer_sizes) < 2) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP Poisson objective: model topology needs an input and output")
            return
        end if
        if (model%layer_sizes(1) /= size(x, 2) .or. &
                model%layer_sizes(size(model%layer_sizes)) /= 1 .or. size(x, 1) < 1 .or. &
                size(targets, 1) /= size(x, 1) .or. size(targets, 2) /= 1 .or. &
                any(.not. ieee_is_finite(x)) .or. any(.not. ieee_is_finite(targets)) .or. &
                any(targets < 0.0_dp) .or. .not. ieee_is_finite(l2) .or. l2 < 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP Poisson objective: model, data, or L2 value is invalid")
            return
        end if
        allocate(self%features, source=x)
        allocate(self%targets, source=targets)
        allocate(self%weights(size(x, 1)))
        self%weights = 1.0_dp
        if (present(sample_weight)) then
            if (size(sample_weight) /= size(x, 1) .or. &
                    any(.not. ieee_is_finite(sample_weight)) .or. &
                    any(sample_weight < 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP Poisson objective: sample weights are invalid")
                return
            end if
            self%weights = sample_weight
        end if
        weight_mass = sum(self%weights)
        if (.not. ieee_is_finite(weight_mass) .or. weight_mass <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP Poisson objective: effective weights have no positive mass")
            return
        end if
        self%model => model
        self%l2 = l2
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_poisson_objective_initialize

    integer function mlp_poisson_objective_parameter_count(self) result(count)
        class(mlp_poisson_training_objective_t), intent(in) :: self

        count = 0
        if (.not. associated(self%model)) return
        count = self%model%parameter_count()
        if (self%optimize_l2) count = count + 1
    end function mlp_poisson_objective_parameter_count

    function mlp_poisson_objective_parameters(self) result(parameters)
        class(mlp_poisson_training_objective_t), intent(in) :: self
        real(dp), allocatable :: parameters(:)
        integer :: n_model

        allocate(parameters(self%parameter_count()))
        parameters = 0.0_dp
        if (.not. associated(self%model)) return
        n_model = self%model%parameter_count()
        parameters(:n_model) = self%model%parameters()
        if (self%optimize_l2) parameters(n_model + 1) = self%l2
    end function mlp_poisson_objective_parameters

    subroutine mlp_poisson_objective_value_gradient(self, parameters, value, &
            gradient, status)
        class(mlp_poisson_training_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: l2
        integer :: n_model

        value = huge(1.0_dp)
        gradient = 0.0_dp
        if (.not. associated(self%model)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP Poisson objective: adapter is not initialized")
            return
        end if
        n_model = self%model%parameter_count()
        if (size(parameters) /= self%parameter_count() .or. &
                size(gradient) /= size(parameters) .or. &
                any(.not. ieee_is_finite(parameters))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP Poisson objective: parameter or gradient shape is invalid")
            return
        end if
        l2 = self%l2
        if (self%optimize_l2) then
            l2 = parameters(n_model + 1)
            if (.not. ieee_is_finite(l2) .or. l2 < 0.0_dp) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP Poisson objective: optimized L2 coefficient is invalid")
                return
            end if
        end if
        call self%model%set_parameters(parameters(:n_model), status)
        if (.not. status_ok(status)) return
        call mlp_poisson_loss_gradient(self%model, self%features, self%targets, &
            l2, value, gradient(:n_model), status, self%weights)
        if (.not. status_ok(status)) return
        if (self%optimize_l2) gradient(n_model + 1) = &
            0.5_dp*sum(parameters(:n_model)*parameters(:n_model))
        if (.not. ieee_is_finite(value) .or. any(.not. ieee_is_finite(gradient))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP Poisson objective: value or gradient is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_poisson_objective_value_gradient

    subroutine mlp_poisson_objective_jvp(self, parameters, direction, value, &
            tangent, status)
        class(mlp_poisson_training_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:), direction(:)
        real(dp), intent(out) :: value, tangent
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: gradient(:)

        value = huge(1.0_dp)
        tangent = 0.0_dp
        if (size(direction) /= size(parameters) .or. &
                any(.not. ieee_is_finite(direction))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP Poisson objective JVP: direction shape or values are invalid")
            return
        end if
        allocate(gradient(size(parameters)))
        call self%value_gradient(parameters, value, gradient, status)
        if (.not. status_ok(status)) return
        tangent = dot_product(gradient, direction)
        if (.not. ieee_is_finite(tangent)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP Poisson objective JVP: product is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_poisson_objective_jvp

    subroutine mlp_poisson_objective_vjp(self, parameters, output_bar, gradient, &
            status)
        class(mlp_poisson_training_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:), output_bar
        real(dp), intent(out) :: gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: value

        gradient = 0.0_dp
        if (.not. ieee_is_finite(output_bar)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP Poisson objective VJP: output cotangent is invalid")
            return
        end if
        call self%value_gradient(parameters, value, gradient, status)
        if (.not. status_ok(status)) return
        gradient = output_bar*gradient
        if (any(.not. ieee_is_finite(gradient))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP Poisson objective VJP: product is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_poisson_objective_vjp

    subroutine mlp_poisson_objective_hvp(self, parameters, direction, product, &
            status)
        class(mlp_poisson_training_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:), direction(:)
        real(dp), intent(out) :: product(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: theta_hvp(:)
        real(dp) :: l2, l2_direction
        integer :: n_model

        product = 0.0_dp
        if (.not. associated(self%model)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP Poisson objective HVP: adapter is not initialized")
            return
        end if
        n_model = self%model%parameter_count()
        if (size(parameters) /= self%parameter_count() .or. &
                size(direction) /= size(parameters) .or. &
                size(product) /= size(parameters) .or. &
                any(.not. ieee_is_finite(parameters)) .or. &
                any(.not. ieee_is_finite(direction))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP Poisson objective HVP: parameter or direction shape is invalid")
            return
        end if
        l2 = self%l2
        l2_direction = 0.0_dp
        if (self%optimize_l2) then
            l2 = parameters(n_model + 1)
            l2_direction = direction(n_model + 1)
            if (.not. ieee_is_finite(l2) .or. l2 < 0.0_dp) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP Poisson objective HVP: optimized L2 coefficient is invalid")
                return
            end if
        end if
        call self%model%set_parameters(parameters(:n_model), status)
        if (.not. status_ok(status)) return
        allocate(theta_hvp(n_model))
        call mlp_poisson_loss_hvp(self%model, self%features, self%targets, l2, &
            direction(:n_model), theta_hvp, status, self%weights)
        if (.not. status_ok(status)) return
        product(:n_model) = theta_hvp
        if (self%optimize_l2) then
            product(:n_model) = product(:n_model) + l2_direction*parameters(:n_model)
            product(n_model + 1) = dot_product(parameters(:n_model), direction(:n_model))
        end if
        if (any(.not. ieee_is_finite(product))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP Poisson objective HVP: product is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_poisson_objective_hvp

    subroutine mlp_poisson_objective_fortopt(self, objective, status)
        class(mlp_poisson_training_objective_t), target, intent(inout) :: self
        type(objective_t), intent(out) :: objective
        type(fortnum_status_t), intent(out) :: status

        if (.not. associated(self%model)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP Poisson objective: adapter is not initialized")
            return
        end if
        call objective%initialize_context(self%parameter_count(), self, &
            mlp_poisson_objective_context_callback, status)
    end subroutine mlp_poisson_objective_fortopt

    subroutine mlp_poisson_objective_context_callback(context, parameters, value, &
            gradient, status)
        class(*), intent(inout) :: context
        real(dp), intent(in) :: parameters(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status

        select type (adapter => context)
            type is (mlp_poisson_training_objective_t)
            call adapter%value_gradient(parameters, value, gradient, status)
        class default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP Poisson objective: context has the wrong type")
        end select
    end subroutine mlp_poisson_objective_context_callback

    subroutine mlp_poisson_optimize_lbfgsb(model, x, targets, options, result, &
            status, sample_weight)
        !! Optimize a Poisson MLP objective with FortOpt's L-BFGS-B.
        type(mlp_t), target, intent(inout) :: model
        real(dp), intent(in) :: x(:, :), targets(:, :)
        type(mlp_poisson_lbfgsb_options_t), intent(in) :: options
        type(mlp_poisson_lbfgsb_result_t), intent(out) :: result
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:)
        type(mlp_poisson_training_objective_t), target :: adapter
        type(objective_t) :: objective
        type(lbfgsb_t) :: optimizer
        type(lbfgsb_options_t) :: optimizer_options
        type(lbfgsb_result_t) :: optimizer_result
        real(dp), allocatable :: parameters(:), lower(:), upper(:), gradient(:)
        integer :: n_model, n_parameters

        result = mlp_poisson_lbfgsb_result_t()
        if (.not. valid_lbfgsb_options(options)) then
            if (options%device_kind /= FORTML_DEVICE_CPU) then
                call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                    "MLP Poisson L-BFGS-B: resident device path is unavailable")
            else
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP Poisson L-BFGS-B: options are invalid")
            end if
            return
        end if
        call adapter%initialize(model, x, targets, options%l2, status, &
            optimize_l2=options%optimize_l2, sample_weight=sample_weight, &
            device_kind=options%device_kind)
        if (.not. status_ok(status)) return
        n_model = model%parameter_count()
        n_parameters = adapter%parameter_count()
        parameters = adapter%parameters()
        allocate(lower(n_parameters), upper(n_parameters), gradient(n_parameters))
        lower(:n_model) = options%lower_bound
        upper(:n_model) = options%upper_bound
        if (options%optimize_l2) then
            lower(n_model + 1) = options%l2_lower_bound
            upper(n_model + 1) = options%l2_upper_bound
            parameters(n_model + 1) = min(max(options%l2, lower(n_model + 1)), &
                upper(n_model + 1))
        end if
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
        call model%set_parameters(parameters(:n_model), status)
        if (.not. status_ok(status)) return
        call adapter%value_gradient(parameters, result%objective, gradient, status)
        if (.not. status_ok(status)) return
        result%converged = optimizer_result%state%converged
        result%iterations = optimizer_result%state%iteration
        result%line_search_evaluations = optimizer_result%line_search_evaluations
        result%gradient_norm = sqrt(sum(gradient*gradient))
        result%l2 = options%l2
        if (options%optimize_l2) result%l2 = parameters(n_model + 1)
        if (.not. ieee_is_finite(result%objective) .or. &
                .not. ieee_is_finite(result%gradient_norm) .or. &
                .not. ieee_is_finite(result%l2)) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "MLP Poisson L-BFGS-B: result is not finite")
            return
        end if
        if (.not. result%converged) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "MLP Poisson L-BFGS-B: iteration limit reached")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_poisson_optimize_lbfgsb

    subroutine mlp_poisson_loss_gradient(model, x, targets, l2, value, gradient, &
            status, sample_weight)
        class(mlp_t), intent(in) :: model
        real(dp), intent(in) :: x(:, :), targets(:, :), l2
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:)
        real(dp), allocatable :: log_rate(:, :), log_rate_bar(:, :), x_bar(:, :), theta(:)

        value = 0.0_dp
        gradient = 0.0_dp
        if (.not. valid_loss_shapes(model, x, targets, l2, gradient, sample_weight)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP Poisson loss: model, data, penalty, or gradient shape is invalid")
            return
        end if
        allocate(log_rate(size(x, 1), 1), log_rate_bar(size(x, 1), 1), &
            x_bar(size(x, 1), size(x, 2)))
        call model%predict(x, log_rate, status)
        if (.not. status_ok(status)) return
        if (present(sample_weight)) then
            call poisson_nll_value(log_rate, targets, value, status, sample_weight=sample_weight)
            if (.not. status_ok(status)) return
            call poisson_nll_vjp(log_rate, targets, 1.0_dp, log_rate_bar, status, &
                sample_weight=sample_weight)
        else
            call poisson_nll_value(log_rate, targets, value, status)
            if (.not. status_ok(status)) return
            call poisson_nll_vjp(log_rate, targets, 1.0_dp, log_rate_bar, status)
        end if
        if (.not. status_ok(status)) return
        call model%vjp(x, log_rate_bar, gradient, x_bar, status)
        if (.not. status_ok(status)) return
        theta = model%parameters()
        value = value + 0.5_dp*l2*sum(theta*theta)
        gradient = gradient + l2*theta
        if (.not. ieee_is_finite(value) .or. any(.not. ieee_is_finite(gradient))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP Poisson loss: objective or gradient is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_poisson_loss_gradient

    subroutine mlp_poisson_loss_hvp(model, x, targets, l2, theta_dot, hvp, status, &
            sample_weight)
        class(mlp_t), intent(in) :: model
        real(dp), intent(in) :: x(:, :), targets(:, :), l2, theta_dot(:)
        real(dp), intent(out) :: hvp(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:)
        real(dp), allocatable :: log_rate(:, :), log_rate_bar(:, :), log_rate_dot(:, :)
        real(dp), allocatable :: log_rate_hvp(:, :), x_zero(:, :), x_bar(:, :), x_hvp(:, :)
        real(dp), allocatable :: direct_hvp(:), theta(:)

        hvp = 0.0_dp
        if (.not. valid_loss_shapes(model, x, targets, l2, hvp, sample_weight) .or. &
                size(theta_dot) /= model%parameter_count() .or. &
                any(.not. ieee_is_finite(theta_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP Poisson HVP: model, data, or direction shape is invalid")
            return
        end if
        allocate(log_rate(size(x, 1), 1), log_rate_bar(size(x, 1), 1), &
            log_rate_dot(size(x, 1), 1), log_rate_hvp(size(x, 1), 1), &
            x_zero(size(x, 1), size(x, 2)), x_bar(size(x, 1), size(x, 2)), &
            x_hvp(size(x, 1), size(x, 2)), direct_hvp(size(theta_dot)))
        x_zero = 0.0_dp
        call model%predict(x, log_rate, status)
        if (.not. status_ok(status)) return
        if (present(sample_weight)) then
            call poisson_nll_vjp(log_rate, targets, 1.0_dp, log_rate_bar, status, &
                sample_weight=sample_weight)
        else
            call poisson_nll_vjp(log_rate, targets, 1.0_dp, log_rate_bar, status)
        end if
        if (.not. status_ok(status)) return
        call model%jvp(x, theta_dot, x_zero, log_rate, log_rate_dot, status)
        if (.not. status_ok(status)) return
        if (present(sample_weight)) then
            call poisson_nll_hvp(log_rate, targets, log_rate_dot, log_rate_hvp, status, &
                sample_weight=sample_weight)
        else
            call poisson_nll_hvp(log_rate, targets, log_rate_dot, log_rate_hvp, status)
        end if
        if (.not. status_ok(status)) return
        call model%hvp(x, log_rate_bar, theta_dot, x_zero, direct_hvp, x_hvp, status)
        if (.not. status_ok(status)) return
        call model%vjp(x, log_rate_hvp, hvp, x_bar, status)
        if (.not. status_ok(status)) return
        hvp = hvp + direct_hvp
        theta = model%parameters()
        hvp = hvp + l2*theta_dot
        if (any(.not. ieee_is_finite(hvp))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP Poisson HVP: product is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_poisson_loss_hvp

    logical function valid_loss_shapes(model, x, targets, l2, vector, sample_weight) &
            result(valid)
        class(mlp_t), intent(in) :: model
        real(dp), intent(in) :: x(:, :), targets(:, :), l2, vector(:)
        real(dp), intent(in), optional :: sample_weight(:)

        valid = .false.
        if (.not. allocated(model%layer_sizes)) return
        if (size(model%layer_sizes) < 2) return
        valid = model%layer_sizes(1) == size(x, 2) .and. &
            model%layer_sizes(size(model%layer_sizes)) == 1 .and. size(x, 1) >= 1 .and. &
            size(targets, 1) == size(x, 1) .and. size(targets, 2) == 1 .and. &
            size(vector) == model%parameter_count() .and. all(ieee_is_finite(x)) .and. &
            all(ieee_is_finite(targets)) .and. all(targets >= 0.0_dp) .and. &
            ieee_is_finite(l2) .and. l2 >= 0.0_dp
        if (present(sample_weight)) then
            if (size(sample_weight) /= size(x, 1)) then
                valid = .false.
                return
            end if
            if (any(.not. ieee_is_finite(sample_weight)) .or. &
                    any(sample_weight < 0.0_dp) .or. sum(sample_weight) <= 0.0_dp) then
                valid = .false.
                return
            end if
        end if
    end function valid_loss_shapes

    logical function valid_lbfgsb_options(options) result(valid)
        type(mlp_poisson_lbfgsb_options_t), intent(in) :: options

        valid = options%device_kind == FORTML_DEVICE_CPU .and. options%memory >= 1 .and. &
            options%max_iterations >= 1 .and. options%max_line_search >= 1 .and. &
            ieee_is_finite(options%gradient_tolerance) .and. &
            ieee_is_finite(options%step_tolerance) .and. &
            ieee_is_finite(options%objective_tolerance) .and. &
            options%gradient_tolerance >= 0.0_dp .and. options%step_tolerance >= 0.0_dp .and. &
            options%objective_tolerance >= 0.0_dp .and. &
            ieee_is_finite(options%lower_bound) .and. ieee_is_finite(options%upper_bound) .and. &
            options%lower_bound < options%upper_bound .and. ieee_is_finite(options%l2) .and. &
            options%l2 >= 0.0_dp .and. ieee_is_finite(options%l2_lower_bound) .and. &
            ieee_is_finite(options%l2_upper_bound) .and. &
            options%l2_lower_bound >= 0.0_dp .and. options%l2_lower_bound < options%l2_upper_bound
    end function valid_lbfgsb_options

end module fortml_mlp_poisson
