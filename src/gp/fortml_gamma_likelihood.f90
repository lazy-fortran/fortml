module fortml_gamma_likelihood
    !! Gamma observation likelihood for positive GP targets and log-mean latents.
    !!
    !! The density uses ``mean=exp(latent)`` and ``shape=exp(log_shape)``.
    !! Free procedures differentiate the complete fixed-observation likelihood
    !! with respect to every latent and the transformed shape coordinate.  The
    !! object adapts the shape coordinate to FortOpt while holding the latent GP
    !! state fixed.  CPU evaluation is the reference path.  CUDA is refused
    !! until resident log-gamma, digamma, and trigamma kernels are linked.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, status_ok, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_CONVERGENCE_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    use fortopt_objective, only: objective_t
    use fortopt_lbfgsb, only: lbfgsb_t, lbfgsb_options_t, lbfgsb_result_t
    implicit none
    private

    integer, parameter, public :: GAMMA_LIKELIHOOD_N_PARAMETERS = 1

    type, public :: gamma_likelihood_lbfgsb_options_t
        integer :: memory = 6
        integer :: max_iterations = 100
        integer :: max_line_search = 40
        real(dp) :: gradient_tolerance = 1.0e-7_dp
        real(dp) :: step_tolerance = 1.0e-10_dp
        real(dp) :: objective_tolerance = 1.0e-10_dp
        real(dp) :: lower_log_shape = -10.0_dp
        real(dp) :: upper_log_shape = 10.0_dp
    end type gamma_likelihood_lbfgsb_options_t

    type, public :: gamma_likelihood_lbfgsb_result_t
        logical :: converged = .false.
        integer :: iterations = 0
        integer :: line_search_evaluations = 0
        real(dp) :: objective = huge(1.0_dp)
        real(dp) :: gradient_norm = huge(1.0_dp)
    end type gamma_likelihood_lbfgsb_result_t

    type, public :: gamma_likelihood_t
        private
        real(dp), allocatable :: observations(:), latents(:), weights(:)
        real(dp) :: log_shape = 0.0_dp
        integer :: device_kind = FORTML_DEVICE_CPU
    contains
        procedure, public :: initialize => gamma_likelihood_initialize
        procedure, public :: initialized => gamma_likelihood_initialized
        procedure, public :: device_supported => gamma_likelihood_device_supported
        procedure, public :: parameter_count => gamma_likelihood_parameter_count
        procedure, public :: parameters => gamma_likelihood_parameters
        procedure, public :: set_parameters => gamma_likelihood_set_parameters
        procedure, public :: value_gradient => gamma_likelihood_value_gradient
        procedure, public :: jvp => gamma_likelihood_objective_jvp
        procedure, public :: vjp => gamma_likelihood_objective_vjp
        procedure, public :: hvp => gamma_likelihood_objective_hvp
        procedure, public :: fortopt => gamma_likelihood_fortopt
        procedure, public :: optimize_lbfgsb => gamma_likelihood_optimize_lbfgsb
    end type gamma_likelihood_t

    public :: gamma_log_likelihood_value
    public :: gamma_log_likelihood_value_gradient
    public :: gamma_log_likelihood_jvp
    public :: gamma_log_likelihood_vjp
    public :: gamma_log_likelihood_hvp

contains

    subroutine gamma_likelihood_initialize(self, observations, latents, status, &
            log_shape, sample_weight, device_kind)
        class(gamma_likelihood_t), intent(out) :: self
        real(dp), intent(in) :: observations(:), latents(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: log_shape, sample_weight(:)
        integer, intent(in), optional :: device_kind
        real(dp) :: candidate

        self%device_kind = FORTML_DEVICE_CPU
        if (present(device_kind)) self%device_kind = device_kind
        if (self%device_kind == FORTML_DEVICE_CUDA) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "Gamma likelihood: resident CUDA special-function path is not linked")
            return
        end if
        if (self%device_kind /= FORTML_DEVICE_CPU) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Gamma likelihood: device kind is invalid")
            return
        end if
        if (.not. valid_data(observations, latents, status, sample_weight)) return
        candidate = 0.0_dp
        if (present(log_shape)) candidate = log_shape
        if (.not. valid_log_shape(candidate)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Gamma likelihood: transformed shape is invalid")
            return
        end if
        allocate(self%observations, source=observations)
        allocate(self%latents, source=latents)
        allocate(self%weights(size(observations)))
        self%weights = 1.0_dp
        if (present(sample_weight)) self%weights = sample_weight
        self%log_shape = candidate
        call status_set(status, FORTNUM_OK, "")
    end subroutine gamma_likelihood_initialize

    logical function gamma_likelihood_initialized(self) result(yes)
        class(gamma_likelihood_t), intent(in) :: self

        yes = allocated(self%observations)
        if (.not. yes) return
        yes = allocated(self%latents)
        if (.not. yes) return
        yes = allocated(self%weights)
        if (.not. yes) return
        yes = size(self%observations) > 0
        if (.not. yes) return
        yes = size(self%latents) == size(self%observations)
        if (.not. yes) return
        yes = size(self%weights) == size(self%observations)
    end function gamma_likelihood_initialized

    logical function gamma_likelihood_device_supported(self, device_kind) result(yes)
        class(gamma_likelihood_t), intent(in) :: self
        integer, intent(in), optional :: device_kind
        integer :: requested

        requested = self%device_kind
        if (present(device_kind)) requested = device_kind
        yes = requested == FORTML_DEVICE_CPU .and. self%initialized()
    end function gamma_likelihood_device_supported

    integer function gamma_likelihood_parameter_count(self) result(count)
        class(gamma_likelihood_t), intent(in) :: self

        count = 0
        if (self%initialized()) count = GAMMA_LIKELIHOOD_N_PARAMETERS
    end function gamma_likelihood_parameter_count

    function gamma_likelihood_parameters(self) result(parameters)
        class(gamma_likelihood_t), intent(in) :: self
        real(dp), allocatable :: parameters(:)

        allocate(parameters(0))
        if (.not. self%initialized()) return
        deallocate(parameters)
        allocate(parameters(1))
        parameters(1) = self%log_shape
    end function gamma_likelihood_parameters

    subroutine gamma_likelihood_set_parameters(self, parameters, status)
        class(gamma_likelihood_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:)
        type(fortnum_status_t), intent(out) :: status

        if (.not. self%initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Gamma likelihood: set before initialize")
            return
        end if
        if (size(parameters) /= 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Gamma likelihood: parameter shape is invalid")
            return
        end if
        if (.not. valid_log_shape(parameters(1))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Gamma likelihood: transformed shape is invalid")
            return
        end if
        self%log_shape = parameters(1)
        call status_set(status, FORTNUM_OK, "")
    end subroutine gamma_likelihood_set_parameters

    subroutine gamma_likelihood_value_gradient(self, parameters, value, gradient, &
            status)
        class(gamma_likelihood_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: latent_gradient(:)
        real(dp) :: log_value, shape_gradient

        value = huge(1.0_dp)
        gradient = 0.0_dp
        if (.not. self%initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Gamma likelihood objective: adapter is not initialized")
            return
        end if
        if (size(parameters) /= 1 .or. size(gradient) /= 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Gamma likelihood objective: parameter or gradient shape is invalid")
            return
        end if
        allocate(latent_gradient(size(self%latents)))
        call gamma_log_likelihood_value_gradient(self%observations, self%latents, &
            parameters(1), log_value, latent_gradient, shape_gradient, status, &
            self%weights)
        if (.not. status_ok(status)) return
        value = -log_value
        gradient(1) = -shape_gradient
        self%log_shape = parameters(1)
        call status_set(status, FORTNUM_OK, "")
    end subroutine gamma_likelihood_value_gradient

    subroutine gamma_likelihood_objective_jvp(self, parameters, direction, value, &
            tangent, status)
        class(gamma_likelihood_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:), direction(:)
        real(dp), intent(out) :: value, tangent
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: gradient(1)

        value = huge(1.0_dp)
        tangent = 0.0_dp
        if (size(direction) /= 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Gamma likelihood objective JVP: direction shape is invalid")
            return
        end if
        call self%value_gradient(parameters, value, gradient, status)
        if (.not. status_ok(status)) return
        tangent = gradient(1)*direction(1)
        call status_set(status, FORTNUM_OK, "")
    end subroutine gamma_likelihood_objective_jvp

    subroutine gamma_likelihood_objective_vjp(self, parameters, value_bar, &
            parameter_bar, status)
        class(gamma_likelihood_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:), value_bar
        real(dp), intent(out) :: parameter_bar(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: value

        parameter_bar = 0.0_dp
        if (size(parameter_bar) /= 1 .or. .not. ieee_is_finite(value_bar)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Gamma likelihood objective VJP: cotangent or shape is invalid")
            return
        end if
        call self%value_gradient(parameters, value, parameter_bar, status)
        if (.not. status_ok(status)) return
        parameter_bar = value_bar*parameter_bar
        call status_set(status, FORTNUM_OK, "")
    end subroutine gamma_likelihood_objective_vjp

    subroutine gamma_likelihood_objective_hvp(self, parameters, direction, &
            product, status)
        class(gamma_likelihood_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:), direction(:)
        real(dp), intent(out) :: product(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: latent_direction(:), latent_product(:)
        real(dp) :: shape_product

        product = 0.0_dp
        if (.not. self%initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Gamma likelihood objective HVP: adapter is not initialized")
            return
        end if
        if (size(parameters) /= 1 .or. size(direction) /= 1 .or. &
            size(product) /= 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Gamma likelihood objective HVP: product shape is invalid")
            return
        end if
        allocate(latent_direction(size(self%latents)))
        allocate(latent_product(size(self%latents)))
        latent_direction = 0.0_dp
        call gamma_log_likelihood_hvp(self%observations, self%latents, parameters(1), &
            latent_direction, direction(1), latent_product, shape_product, status, &
            self%weights)
        if (.not. status_ok(status)) return
        product(1) = -shape_product
        self%log_shape = parameters(1)
        call status_set(status, FORTNUM_OK, "")
    end subroutine gamma_likelihood_objective_hvp

    subroutine gamma_likelihood_fortopt(self, objective, status)
        class(gamma_likelihood_t), target, intent(inout) :: self
        type(objective_t), intent(out) :: objective
        type(fortnum_status_t), intent(out) :: status

        if (.not. self%initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Gamma likelihood objective: adapter is not initialized")
            return
        end if
        call objective%initialize_context(1, self, gamma_likelihood_context, status)
    end subroutine gamma_likelihood_fortopt

    subroutine gamma_likelihood_context(context_any, parameters, value, gradient, &
            status)
        class(*), intent(inout) :: context_any
        real(dp), intent(in) :: parameters(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status

        select type (adapter => context_any)
            type is (gamma_likelihood_t)
            call adapter%value_gradient(parameters, value, gradient, status)
        class default
            value = huge(1.0_dp)
            gradient = 0.0_dp
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Gamma likelihood objective: context has the wrong type")
        end select
    end subroutine gamma_likelihood_context

    subroutine gamma_likelihood_optimize_lbfgsb(self, options, result, status)
        class(gamma_likelihood_t), target, intent(inout) :: self
        type(gamma_likelihood_lbfgsb_options_t), intent(in) :: options
        type(gamma_likelihood_lbfgsb_result_t), intent(out) :: result
        type(fortnum_status_t), intent(out) :: status
        type(objective_t) :: objective
        type(lbfgsb_t) :: optimizer
        type(lbfgsb_options_t) :: optimizer_options
        type(lbfgsb_result_t) :: optimizer_result
        type(gamma_likelihood_lbfgsb_result_t) :: result_default
        real(dp) :: previous
        real(dp) :: parameters(1), lower(1), upper(1), gradient(1)

        result = result_default
        if (.not. self%initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Gamma likelihood L-BFGS-B: adapter is not initialized")
            return
        end if
        if (.not. valid_lbfgsb_options(options)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Gamma likelihood L-BFGS-B: options are invalid")
            return
        end if
        previous = self%log_shape
        lower(1) = options%lower_log_shape
        upper(1) = options%upper_log_shape
        parameters(1) = min(max(previous, lower(1)), upper(1))
        call self%fortopt(objective, status)
        if (.not. status_ok(status)) return
        optimizer_options%memory = options%memory
        optimizer_options%max_iterations = options%max_iterations
        optimizer_options%max_line_search = options%max_line_search
        optimizer_options%gradient_tolerance = options%gradient_tolerance
        optimizer_options%step_tolerance = options%step_tolerance
        optimizer_options%objective_tolerance = options%objective_tolerance
        call optimizer%minimize(objective, parameters, lower, upper, &
            optimizer_options, optimizer_result, status)
        if (.not. status_ok(status)) then
            self%log_shape = previous
            return
        end if
        call self%value_gradient(parameters, result%objective, gradient, status)
        if (.not. status_ok(status)) then
            self%log_shape = previous
            return
        end if
        result%converged = optimizer_result%state%converged
        result%iterations = optimizer_result%state%iteration
        result%line_search_evaluations = optimizer_result%line_search_evaluations
        result%gradient_norm = abs(gradient(1))
        if (.not. result%converged) then
            self%log_shape = previous
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "Gamma likelihood L-BFGS-B: iteration limit reached")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine gamma_likelihood_optimize_lbfgsb

    subroutine gamma_log_likelihood_value(observations, latents, log_shape, value, &
            status, sample_weight)
        real(dp), intent(in) :: observations(:), latents(:), log_shape
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:)
        real(dp), allocatable :: latent_gradient(:)
        real(dp) :: shape_gradient

        allocate(latent_gradient(size(latents)))
        call gamma_log_likelihood_value_gradient(observations, latents, log_shape, &
            value, latent_gradient, shape_gradient, status, sample_weight)
    end subroutine gamma_log_likelihood_value

    subroutine gamma_log_likelihood_value_gradient(observations, latents, log_shape, &
            value, latent_gradient, shape_gradient, status, sample_weight)
        real(dp), intent(in) :: observations(:), latents(:), log_shape
        real(dp), intent(out) :: value, latent_gradient(:), shape_gradient
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:)
        real(dp) :: shape, rate_ratio, weight, derivative_shape
        integer :: i

        value = 0.0_dp
        latent_gradient = 0.0_dp
        shape_gradient = 0.0_dp
        if (.not. valid_data(observations, latents, status, sample_weight)) return
        if (size(latent_gradient) /= size(latents) .or. &
            .not. valid_log_shape(log_shape)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Gamma likelihood: product shape or transformed shape is invalid")
            return
        end if
        shape = exp(log_shape)
        do i = 1, size(observations)
            weight = 1.0_dp
            if (present(sample_weight)) weight = sample_weight(i)
            rate_ratio = observations(i)*exp(-latents(i))
            if (.not. ieee_is_finite(rate_ratio)) then
                call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                    "Gamma likelihood: observation-to-mean ratio overflowed")
                return
            end if
            derivative_shape = log_shape + 1.0_dp - digamma_positive(shape) &
                + log(observations(i)) - latents(i) - rate_ratio
            value = value + weight*(shape*log_shape - log_gamma(shape) &
                + (shape - 1.0_dp)*log(observations(i)) - shape*latents(i) &
                - shape*rate_ratio)
            latent_gradient(i) = weight*shape*(rate_ratio - 1.0_dp)
            shape_gradient = shape_gradient + weight*shape*derivative_shape
        end do
        if (.not. ieee_is_finite(value) .or. &
            any(.not. ieee_is_finite(latent_gradient)) .or. &
            .not. ieee_is_finite(shape_gradient)) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "Gamma likelihood: value or gradient is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine gamma_log_likelihood_value_gradient

    subroutine gamma_log_likelihood_jvp(observations, latents, log_shape, &
            latent_direction, shape_direction, value, tangent, status, sample_weight)
        real(dp), intent(in) :: observations(:), latents(:), log_shape
        real(dp), intent(in) :: latent_direction(:), shape_direction
        real(dp), intent(out) :: value, tangent
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:)
        real(dp), allocatable :: latent_gradient(:)
        real(dp) :: shape_gradient

        value = 0.0_dp
        tangent = 0.0_dp
        if (size(latent_direction) /= size(latents) .or. &
            any(.not. ieee_is_finite(latent_direction)) .or. &
            .not. ieee_is_finite(shape_direction)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Gamma likelihood JVP: direction is invalid")
            return
        end if
        allocate(latent_gradient(size(latents)))
        call gamma_log_likelihood_value_gradient(observations, latents, log_shape, &
            value, latent_gradient, shape_gradient, status, sample_weight)
        if (.not. status_ok(status)) return
        tangent = dot_product(latent_gradient, latent_direction) &
            + shape_gradient*shape_direction
        call status_set(status, FORTNUM_OK, "")
    end subroutine gamma_log_likelihood_jvp

    subroutine gamma_log_likelihood_vjp(observations, latents, log_shape, value_bar, &
            latent_bar, shape_bar, status, sample_weight)
        real(dp), intent(in) :: observations(:), latents(:), log_shape, value_bar
        real(dp), intent(out) :: latent_bar(:), shape_bar
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:)
        real(dp) :: value

        latent_bar = 0.0_dp
        shape_bar = 0.0_dp
        if (.not. ieee_is_finite(value_bar)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Gamma likelihood VJP: cotangent is invalid")
            return
        end if
        call gamma_log_likelihood_value_gradient(observations, latents, log_shape, &
            value, latent_bar, shape_bar, status, sample_weight)
        if (.not. status_ok(status)) return
        latent_bar = value_bar*latent_bar
        shape_bar = value_bar*shape_bar
        call status_set(status, FORTNUM_OK, "")
    end subroutine gamma_log_likelihood_vjp

    subroutine gamma_log_likelihood_hvp(observations, latents, log_shape, &
            latent_direction, shape_direction, latent_product, shape_product, status, &
            sample_weight)
        real(dp), intent(in) :: observations(:), latents(:), log_shape
        real(dp), intent(in) :: latent_direction(:), shape_direction
        real(dp), intent(out) :: latent_product(:), shape_product
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:)
        real(dp) :: shape, rate_ratio, weight, derivative_shape
        real(dp) :: hessian_latent, hessian_mixed, hessian_shape
        integer :: i

        latent_product = 0.0_dp
        shape_product = 0.0_dp
        if (.not. valid_data(observations, latents, status, sample_weight)) return
        if (size(latent_direction) /= size(latents) .or. &
            size(latent_product) /= size(latents) .or. &
            any(.not. ieee_is_finite(latent_direction)) .or. &
            .not. ieee_is_finite(shape_direction) .or. &
            .not. valid_log_shape(log_shape)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Gamma likelihood HVP: direction or product is invalid")
            return
        end if
        shape = exp(log_shape)
        do i = 1, size(observations)
            weight = 1.0_dp
            if (present(sample_weight)) weight = sample_weight(i)
            rate_ratio = observations(i)*exp(-latents(i))
            if (.not. ieee_is_finite(rate_ratio)) then
                call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                    "Gamma likelihood HVP: observation-to-mean ratio overflowed")
                return
            end if
            derivative_shape = log_shape + 1.0_dp - digamma_positive(shape) &
                + log(observations(i)) - latents(i) - rate_ratio
            hessian_latent = -weight*shape*rate_ratio
            hessian_mixed = weight*shape*(rate_ratio - 1.0_dp)
            hessian_shape = weight*(shape*derivative_shape + shape &
                - shape*shape*trigamma_positive(shape))
            latent_product(i) = hessian_latent*latent_direction(i) &
                + hessian_mixed*shape_direction
            shape_product = shape_product + hessian_mixed*latent_direction(i) &
                + hessian_shape*shape_direction
        end do
        if (any(.not. ieee_is_finite(latent_product)) .or. &
            .not. ieee_is_finite(shape_product)) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "Gamma likelihood HVP: product is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine gamma_log_likelihood_hvp

    logical function valid_data(observations, latents, status, &
            sample_weight) result(valid)
        real(dp), intent(in) :: observations(:), latents(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:)

        valid = .false.
        if (size(observations) < 1 .or. size(latents) /= size(observations)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Gamma likelihood: observation and latent shapes disagree")
            return
        end if
        if (any(.not. ieee_is_finite(observations)) .or. any(observations <= 0.0_dp) &
            .or. any(.not. ieee_is_finite(latents))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Gamma likelihood: observations must be positive and inputs finite")
            return
        end if
        if (present(sample_weight)) then
            if (size(sample_weight) /= size(observations)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "Gamma likelihood: sample-weight shape is invalid")
                return
            end if
            if (any(.not. ieee_is_finite(sample_weight)) .or. &
                any(sample_weight < 0.0_dp) .or. sum(sample_weight) <= 0.0_dp) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "Gamma likelihood: sample weights need positive finite mass")
                return
            end if
        end if
        valid = .true.
        call status_set(status, FORTNUM_OK, "")
    end function valid_data

    pure logical function valid_log_shape(argument) result(valid)
        real(dp), intent(in) :: argument
        real(dp) :: shape

        valid = ieee_is_finite(argument)
        if (.not. valid) return
        valid = argument <= log(huge(1.0_dp))
        if (.not. valid) return
        valid = argument >= log(tiny(1.0_dp))
        if (.not. valid) return
        shape = exp(argument)
        valid = ieee_is_finite(shape) .and. shape > 0.0_dp
        if (.not. valid) return
        valid = ieee_is_finite(log_gamma(shape))
    end function valid_log_shape

    logical function valid_lbfgsb_options(options) result(valid)
        type(gamma_likelihood_lbfgsb_options_t), intent(in) :: options

        valid = options%memory > 0 .and. options%max_iterations > 0 &
            .and. options%max_line_search > 0
        if (.not. valid) return
        valid = ieee_is_finite(options%gradient_tolerance) &
            .and. ieee_is_finite(options%step_tolerance) &
            .and. ieee_is_finite(options%objective_tolerance) &
            .and. ieee_is_finite(options%lower_log_shape) &
            .and. ieee_is_finite(options%upper_log_shape)
        if (.not. valid) return
        valid = options%gradient_tolerance >= 0.0_dp &
            .and. options%step_tolerance >= 0.0_dp &
            .and. options%objective_tolerance >= 0.0_dp &
            .and. options%lower_log_shape < options%upper_log_shape
    end function valid_lbfgsb_options

    pure real(dp) function digamma_positive(argument) result(value)
        real(dp), intent(in) :: argument
        real(dp) :: x, inverse, inverse_square

        x = argument
        value = 0.0_dp
        do while (x < 8.0_dp)
            value = value - 1.0_dp/x
            x = x + 1.0_dp
        end do
        inverse = 1.0_dp/x
        inverse_square = inverse*inverse
        value = value + log(x) - 0.5_dp*inverse - inverse_square*(1.0_dp/12.0_dp &
            - inverse_square*(1.0_dp/120.0_dp - inverse_square*(1.0_dp/252.0_dp &
            - inverse_square*(1.0_dp/240.0_dp - inverse_square*5.0_dp/660.0_dp))))
    end function digamma_positive

    pure real(dp) function trigamma_positive(argument) result(value)
        real(dp), intent(in) :: argument
        real(dp) :: x, inverse, inverse_square

        x = argument
        value = 0.0_dp
        do while (x < 8.0_dp)
            value = value + 1.0_dp/(x*x)
            x = x + 1.0_dp
        end do
        inverse = 1.0_dp/x
        inverse_square = inverse*inverse
        value = value + inverse + 0.5_dp*inverse_square &
            + inverse_square*inverse/6.0_dp &
            - inverse_square**2*inverse/30.0_dp + inverse_square**3*inverse/42.0_dp &
            - inverse_square**4*inverse/30.0_dp + 5.0_dp*inverse_square**5/66.0_dp
    end function trigamma_positive

end module fortml_gamma_likelihood
