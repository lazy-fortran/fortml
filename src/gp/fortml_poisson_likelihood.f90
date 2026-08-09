module fortml_poisson_likelihood
    !! Poisson observation likelihood for count-valued GP targets.
    !!
    !! The latent coordinate is the log rate.  ``log_rate_offset`` is an
    !! optional scalar log exposure/intercept shared by all observations, so
    !! the rate for row ``i`` is ``exp(latents(i) + log_rate_offset)``.  The
    !! complete weighted log likelihood exposes analytic latent and offset
    !! gradients, JVP/VJP products, and an HVP.  The object adapter uses the
    !! same products as a FortOpt L-BFGS-B objective and commits a new offset
    !! only after a successful evaluation/fit.
    !!
    !! CPU is the reference path.  CUDA is refused before allocation until a
    !! resident exponential/reduction implementation is linked.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, status_ok, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_CONVERGENCE_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    use fortopt_objective, only: objective_t
    use fortopt_lbfgsb, only: lbfgsb_t, lbfgsb_options_t, lbfgsb_result_t
    implicit none
    private

    integer, parameter, public :: POISSON_LIKELIHOOD_N_PARAMETERS = 1
    real(dp), parameter :: MIN_LOG_RATE_OFFSET = -50.0_dp
    real(dp), parameter :: MAX_LOG_RATE_OFFSET = 50.0_dp

    type, public :: poisson_likelihood_lbfgsb_options_t
        integer :: memory = 6
        integer :: max_iterations = 100
        integer :: max_line_search = 40
        real(dp) :: gradient_tolerance = 1.0e-7_dp
        real(dp) :: step_tolerance = 1.0e-10_dp
        real(dp) :: objective_tolerance = 1.0e-10_dp
        real(dp) :: lower_log_rate_offset = -10.0_dp
        real(dp) :: upper_log_rate_offset = 10.0_dp
    end type poisson_likelihood_lbfgsb_options_t

    type, public :: poisson_likelihood_lbfgsb_result_t
        logical :: converged = .false.
        integer :: iterations = 0
        integer :: line_search_evaluations = 0
        real(dp) :: objective = huge(1.0_dp)
        real(dp) :: gradient_norm = huge(1.0_dp)
    end type poisson_likelihood_lbfgsb_result_t

    type, public :: poisson_likelihood_t
        private
        real(dp), allocatable :: observations(:), latents(:), weights(:)
        real(dp) :: log_rate_offset = 0.0_dp
        integer :: device_kind = FORTML_DEVICE_CPU
    contains
        procedure, public :: initialize => poisson_likelihood_initialize
        procedure, public :: initialized => poisson_likelihood_initialized
        procedure, public :: device_supported => poisson_likelihood_device_supported
        procedure, public :: parameter_count => poisson_likelihood_parameter_count
        procedure, public :: parameters => poisson_likelihood_parameters
        procedure, public :: set_parameters => poisson_likelihood_set_parameters
        procedure, public :: value_gradient => poisson_likelihood_value_gradient
        procedure, public :: jvp => poisson_likelihood_objective_jvp
        procedure, public :: vjp => poisson_likelihood_objective_vjp
        procedure, public :: hvp => poisson_likelihood_objective_hvp
        procedure, public :: fortopt => poisson_likelihood_fortopt
        procedure, public :: optimize_lbfgsb => poisson_likelihood_optimize_lbfgsb
    end type poisson_likelihood_t

    public :: poisson_log_likelihood_value
    public :: poisson_log_likelihood_value_gradient
    public :: poisson_log_likelihood_jvp
    public :: poisson_log_likelihood_vjp
    public :: poisson_log_likelihood_hvp

contains

    subroutine poisson_likelihood_initialize(self, observations, latents, status, &
            log_rate_offset, sample_weight, device_kind)
        class(poisson_likelihood_t), intent(out) :: self
        real(dp), intent(in) :: observations(:), latents(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: log_rate_offset, sample_weight(:)
        integer, intent(in), optional :: device_kind
        real(dp) :: candidate

        self%device_kind = FORTML_DEVICE_CPU
        if (present(device_kind)) self%device_kind = device_kind
        if (self%device_kind == FORTML_DEVICE_CUDA) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "Poisson likelihood: resident CUDA exponential reduction is not linked")
            return
        end if
        if (self%device_kind /= FORTML_DEVICE_CPU) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Poisson likelihood: device kind is invalid")
            return
        end if
        if (.not. valid_data(observations, latents, status, sample_weight)) return
        candidate = 0.0_dp
        if (present(log_rate_offset)) candidate = log_rate_offset
        if (.not. valid_log_rate_offset(candidate)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Poisson likelihood: log-rate offset is invalid")
            return
        end if
        allocate(self%observations, source=observations)
        allocate(self%latents, source=latents)
        allocate(self%weights(size(observations)))
        self%weights = 1.0_dp
        if (present(sample_weight)) self%weights = sample_weight
        self%log_rate_offset = candidate
        call status_set(status, FORTNUM_OK, "")
    end subroutine poisson_likelihood_initialize

    logical function poisson_likelihood_initialized(self) result(yes)
        class(poisson_likelihood_t), intent(in) :: self

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
    end function poisson_likelihood_initialized

    logical function poisson_likelihood_device_supported(self, device_kind) result(yes)
        class(poisson_likelihood_t), intent(in) :: self
        integer, intent(in), optional :: device_kind
        integer :: requested

        requested = self%device_kind
        if (present(device_kind)) requested = device_kind
        yes = requested == FORTML_DEVICE_CPU .and. self%initialized()
    end function poisson_likelihood_device_supported

    integer function poisson_likelihood_parameter_count(self) result(count)
        class(poisson_likelihood_t), intent(in) :: self

        count = 0
        if (self%initialized()) count = POISSON_LIKELIHOOD_N_PARAMETERS
    end function poisson_likelihood_parameter_count

    function poisson_likelihood_parameters(self) result(parameters)
        class(poisson_likelihood_t), intent(in) :: self
        real(dp), allocatable :: parameters(:)

        allocate(parameters(0))
        if (.not. self%initialized()) return
        deallocate(parameters)
        allocate(parameters(POISSON_LIKELIHOOD_N_PARAMETERS))
        parameters(1) = self%log_rate_offset
    end function poisson_likelihood_parameters

    subroutine poisson_likelihood_set_parameters(self, parameters, status)
        class(poisson_likelihood_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:)
        type(fortnum_status_t), intent(out) :: status

        if (.not. self%initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Poisson likelihood: set before initialize")
            return
        end if
        if (size(parameters) /= POISSON_LIKELIHOOD_N_PARAMETERS) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Poisson likelihood: parameter shape or offset is invalid")
            return
        end if
        if (.not. valid_log_rate_offset(parameters(1))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Poisson likelihood: parameter shape or offset is invalid")
            return
        end if
        self%log_rate_offset = parameters(1)
        call status_set(status, FORTNUM_OK, "")
    end subroutine poisson_likelihood_set_parameters

    subroutine poisson_likelihood_value_gradient(self, parameters, value, gradient, &
            status)
        class(poisson_likelihood_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: latent_gradient(:)
        real(dp) :: log_value, offset_gradient

        value = huge(1.0_dp)
        gradient = 0.0_dp
        if (.not. self%initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Poisson likelihood objective: adapter is not initialized")
            return
        end if
        if (size(parameters) /= POISSON_LIKELIHOOD_N_PARAMETERS .or. &
            size(gradient) /= POISSON_LIKELIHOOD_N_PARAMETERS) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Poisson likelihood objective: parameter or gradient shape is invalid")
            return
        end if
        allocate(latent_gradient(size(self%latents)))
        call poisson_log_likelihood_value_gradient(self%observations, self%latents, &
            parameters(1), log_value, latent_gradient, offset_gradient, status, &
            self%weights)
        if (.not. status_ok(status)) return
        value = -log_value
        gradient(1) = -offset_gradient
        self%log_rate_offset = parameters(1)
        call status_set(status, FORTNUM_OK, "")
    end subroutine poisson_likelihood_value_gradient

    subroutine poisson_likelihood_objective_jvp(self, parameters, direction, value, &
            tangent, status)
        class(poisson_likelihood_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:), direction(:)
        real(dp), intent(out) :: value, tangent
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: gradient(POISSON_LIKELIHOOD_N_PARAMETERS)

        value = huge(1.0_dp)
        tangent = 0.0_dp
        if (size(direction) /= POISSON_LIKELIHOOD_N_PARAMETERS) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Poisson likelihood objective JVP: direction shape is invalid")
            return
        end if
        call self%value_gradient(parameters, value, gradient, status)
        if (.not. status_ok(status)) return
        tangent = dot_product(gradient, direction)
        call status_set(status, FORTNUM_OK, "")
    end subroutine poisson_likelihood_objective_jvp

    subroutine poisson_likelihood_objective_vjp(self, parameters, value_bar, &
            parameter_bar, status)
        class(poisson_likelihood_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:), value_bar
        real(dp), intent(out) :: parameter_bar(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: value

        parameter_bar = 0.0_dp
        if (size(parameter_bar) /= POISSON_LIKELIHOOD_N_PARAMETERS .or. &
            .not. ieee_is_finite(value_bar)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Poisson likelihood objective VJP: cotangent or shape is invalid")
            return
        end if
        call self%value_gradient(parameters, value, parameter_bar, status)
        if (.not. status_ok(status)) return
        parameter_bar = value_bar*parameter_bar
        call status_set(status, FORTNUM_OK, "")
    end subroutine poisson_likelihood_objective_vjp

    subroutine poisson_likelihood_objective_hvp(self, parameters, direction, &
            product, status)
        class(poisson_likelihood_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:), direction(:)
        real(dp), intent(out) :: product(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: latent_direction(:), latent_product(:)
        real(dp) :: offset_product

        product = 0.0_dp
        if (.not. self%initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Poisson likelihood objective HVP: adapter is not initialized")
            return
        end if
        if (size(parameters) /= POISSON_LIKELIHOOD_N_PARAMETERS .or. &
            size(direction) /= POISSON_LIKELIHOOD_N_PARAMETERS .or. &
            size(product) /= POISSON_LIKELIHOOD_N_PARAMETERS) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Poisson likelihood objective HVP: product shape is invalid")
            return
        end if
        allocate(latent_direction(size(self%latents)), latent_product(size(self%latents)))
        latent_direction = 0.0_dp
        call poisson_log_likelihood_hvp(self%observations, self%latents, parameters(1), &
            latent_direction, direction(1), latent_product, offset_product, status, &
            self%weights)
        if (.not. status_ok(status)) return
        product(1) = -offset_product
        self%log_rate_offset = parameters(1)
        call status_set(status, FORTNUM_OK, "")
    end subroutine poisson_likelihood_objective_hvp

    subroutine poisson_likelihood_fortopt(self, objective, status)
        class(poisson_likelihood_t), target, intent(inout) :: self
        type(objective_t), intent(out) :: objective
        type(fortnum_status_t), intent(out) :: status

        if (.not. self%initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Poisson likelihood objective: adapter is not initialized")
            return
        end if
        call objective%initialize_context(POISSON_LIKELIHOOD_N_PARAMETERS, self, &
            poisson_likelihood_context, status)
    end subroutine poisson_likelihood_fortopt

    subroutine poisson_likelihood_context(context_any, parameters, value, gradient, &
            status)
        class(*), intent(inout) :: context_any
        real(dp), intent(in) :: parameters(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status

        select type (adapter => context_any)
            type is (poisson_likelihood_t)
            call adapter%value_gradient(parameters, value, gradient, status)
        class default
            value = huge(1.0_dp)
            gradient = 0.0_dp
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Poisson likelihood objective: context has the wrong type")
        end select
    end subroutine poisson_likelihood_context

    subroutine poisson_likelihood_optimize_lbfgsb(self, options, result, status)
        class(poisson_likelihood_t), target, intent(inout) :: self
        type(poisson_likelihood_lbfgsb_options_t), intent(in) :: options
        type(poisson_likelihood_lbfgsb_result_t), intent(out) :: result
        type(fortnum_status_t), intent(out) :: status
        type(objective_t) :: objective
        type(lbfgsb_t) :: optimizer
        type(lbfgsb_options_t) :: optimizer_options
        type(lbfgsb_result_t) :: optimizer_result
        type(poisson_likelihood_lbfgsb_result_t) :: result_default
        real(dp) :: previous
        real(dp) :: parameters(1), lower(1), upper(1), gradient(1)

        result = result_default
        if (.not. self%initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Poisson likelihood L-BFGS-B: adapter is not initialized")
            return
        end if
        if (.not. valid_lbfgsb_options(options)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Poisson likelihood L-BFGS-B: options are invalid")
            return
        end if
        previous = self%log_rate_offset
        lower(1) = options%lower_log_rate_offset
        upper(1) = options%upper_log_rate_offset
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
            self%log_rate_offset = previous
            return
        end if
        call self%value_gradient(parameters, result%objective, gradient, status)
        if (.not. status_ok(status)) then
            self%log_rate_offset = previous
            return
        end if
        result%converged = optimizer_result%state%converged
        result%iterations = optimizer_result%state%iteration
        result%line_search_evaluations = optimizer_result%line_search_evaluations
        result%gradient_norm = abs(gradient(1))
        if (.not. result%converged) then
            self%log_rate_offset = previous
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "Poisson likelihood L-BFGS-B: iteration limit reached")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine poisson_likelihood_optimize_lbfgsb

    subroutine poisson_log_likelihood_value(observations, latents, log_rate_offset, &
            value, status, sample_weight)
        real(dp), intent(in) :: observations(:), latents(:), log_rate_offset
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:)
        real(dp), allocatable :: latent_gradient(:)
        real(dp) :: offset_gradient

        allocate(latent_gradient(size(latents)))
        call poisson_log_likelihood_value_gradient(observations, latents, log_rate_offset, &
            value, latent_gradient, offset_gradient, status, sample_weight)
    end subroutine poisson_log_likelihood_value

    subroutine poisson_log_likelihood_value_gradient(observations, latents, &
            log_rate_offset, value, latent_gradient, offset_gradient, status, sample_weight)
        real(dp), intent(in) :: observations(:), latents(:), log_rate_offset
        real(dp), intent(out) :: value, latent_gradient(:), offset_gradient
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:)
        real(dp) :: eta, rate, weight, count
        integer :: i

        value = 0.0_dp
        latent_gradient = 0.0_dp
        offset_gradient = 0.0_dp
        if (.not. valid_data(observations, latents, status, sample_weight)) return
        if (size(latent_gradient) /= size(latents) .or. &
            .not. valid_log_rate_offset(log_rate_offset)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Poisson likelihood: product shape or offset is invalid")
            return
        end if
        do i = 1, size(observations)
            count = observations(i)
            weight = 1.0_dp
            if (present(sample_weight)) weight = sample_weight(i)
            eta = latents(i) + log_rate_offset
            rate = exp(eta)
            if (.not. ieee_is_finite(rate)) then
                call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                    "Poisson likelihood: rate overflowed")
                return
            end if
            value = value + weight*(count*eta - rate - log_gamma(count + 1.0_dp))
            latent_gradient(i) = weight*(count - rate)
            offset_gradient = offset_gradient + latent_gradient(i)
        end do
        if (.not. ieee_is_finite(value) .or. &
            any(.not. ieee_is_finite(latent_gradient)) .or. &
            .not. ieee_is_finite(offset_gradient)) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "Poisson likelihood: value or gradient is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine poisson_log_likelihood_value_gradient

    subroutine poisson_log_likelihood_jvp(observations, latents, log_rate_offset, &
            latent_direction, offset_direction, value, tangent, status, sample_weight)
        real(dp), intent(in) :: observations(:), latents(:), log_rate_offset
        real(dp), intent(in) :: latent_direction(:), offset_direction
        real(dp), intent(out) :: value, tangent
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:)
        real(dp), allocatable :: latent_gradient(:)
        real(dp) :: offset_gradient

        value = 0.0_dp
        tangent = 0.0_dp
        if (size(latent_direction) /= size(latents) .or. &
            any(.not. ieee_is_finite(latent_direction)) .or. &
            .not. ieee_is_finite(offset_direction)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Poisson likelihood JVP: direction is invalid")
            return
        end if
        allocate(latent_gradient(size(latents)))
        call poisson_log_likelihood_value_gradient(observations, latents, log_rate_offset, &
            value, latent_gradient, offset_gradient, status, sample_weight)
        if (.not. status_ok(status)) return
        tangent = dot_product(latent_gradient, latent_direction) + &
            offset_gradient*offset_direction
        call status_set(status, FORTNUM_OK, "")
    end subroutine poisson_log_likelihood_jvp

    subroutine poisson_log_likelihood_vjp(observations, latents, log_rate_offset, &
            value_bar, latent_bar, offset_bar, status, sample_weight)
        real(dp), intent(in) :: observations(:), latents(:), log_rate_offset, value_bar
        real(dp), intent(out) :: latent_bar(:), offset_bar
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:)
        real(dp) :: value

        latent_bar = 0.0_dp
        offset_bar = 0.0_dp
        if (.not. ieee_is_finite(value_bar)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Poisson likelihood VJP: cotangent is invalid")
            return
        end if
        call poisson_log_likelihood_value_gradient(observations, latents, log_rate_offset, &
            value, latent_bar, offset_bar, status, sample_weight)
        if (.not. status_ok(status)) return
        latent_bar = value_bar*latent_bar
        offset_bar = value_bar*offset_bar
        call status_set(status, FORTNUM_OK, "")
    end subroutine poisson_log_likelihood_vjp

    subroutine poisson_log_likelihood_hvp(observations, latents, log_rate_offset, &
            latent_direction, offset_direction, latent_product, offset_product, status, &
            sample_weight)
        real(dp), intent(in) :: observations(:), latents(:), log_rate_offset
        real(dp), intent(in) :: latent_direction(:), offset_direction
        real(dp), intent(out) :: latent_product(:), offset_product
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:)
        real(dp) :: eta, rate, weight, directional_eta
        integer :: i

        latent_product = 0.0_dp
        offset_product = 0.0_dp
        if (.not. valid_data(observations, latents, status, sample_weight)) return
        if (size(latent_direction) /= size(latents) .or. &
            size(latent_product) /= size(latents) .or. &
            any(.not. ieee_is_finite(latent_direction)) .or. &
            .not. ieee_is_finite(offset_direction) .or. &
            .not. valid_log_rate_offset(log_rate_offset)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Poisson likelihood HVP: direction or product is invalid")
            return
        end if
        do i = 1, size(observations)
            weight = 1.0_dp
            if (present(sample_weight)) weight = sample_weight(i)
            eta = latents(i) + log_rate_offset
            rate = exp(eta)
            if (.not. ieee_is_finite(rate)) then
                call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                    "Poisson likelihood HVP: rate overflowed")
                return
            end if
            directional_eta = latent_direction(i) + offset_direction
            latent_product(i) = -weight*rate*directional_eta
            offset_product = offset_product + latent_product(i)
        end do
        if (any(.not. ieee_is_finite(latent_product)) .or. &
            .not. ieee_is_finite(offset_product)) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "Poisson likelihood HVP: product is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine poisson_log_likelihood_hvp

    logical function valid_data(observations, latents, status, sample_weight) result(valid)
        real(dp), intent(in) :: observations(:), latents(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:)
        real(dp) :: nearest
        integer :: i

        valid = .false.
        if (size(observations) < 1 .or. size(latents) /= size(observations)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Poisson likelihood: observation and latent shapes disagree")
            return
        end if
        if (any(.not. ieee_is_finite(observations)) .or. &
            any(observations < 0.0_dp) .or. any(.not. ieee_is_finite(latents))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Poisson likelihood: counts must be nonnegative and inputs finite")
            return
        end if
        do i = 1, size(observations)
            nearest = floor(observations(i))
            if (abs(observations(i) - nearest) > 1.0e-12_dp) then
                valid = .false.
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "Poisson likelihood: observations must be integer counts")
                return
            end if
        end do
        if (present(sample_weight)) then
            if (size(sample_weight) /= size(observations)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "Poisson likelihood: sample-weight shape is invalid")
                return
            end if
            if (any(.not. ieee_is_finite(sample_weight)) .or. &
                any(sample_weight < 0.0_dp) .or. sum(sample_weight) <= 0.0_dp) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "Poisson likelihood: sample weights need positive finite mass")
                return
            end if
        end if
        valid = .true.
        call status_set(status, FORTNUM_OK, "")
    end function valid_data

    pure logical function valid_log_rate_offset(argument) result(valid)
        real(dp), intent(in) :: argument

        valid = ieee_is_finite(argument) .and. argument >= MIN_LOG_RATE_OFFSET &
            .and. argument <= MAX_LOG_RATE_OFFSET
    end function valid_log_rate_offset

    logical function valid_lbfgsb_options(options) result(valid)
        type(poisson_likelihood_lbfgsb_options_t), intent(in) :: options

        valid = options%memory > 0 .and. options%max_iterations > 0 &
            .and. options%max_line_search > 0
        if (.not. valid) return
        valid = ieee_is_finite(options%gradient_tolerance) &
            .and. ieee_is_finite(options%step_tolerance) &
            .and. ieee_is_finite(options%objective_tolerance) &
            .and. ieee_is_finite(options%lower_log_rate_offset) &
            .and. ieee_is_finite(options%upper_log_rate_offset)
        if (.not. valid) return
        valid = options%gradient_tolerance >= 0.0_dp
        if (.not. valid) return
        valid = options%step_tolerance >= 0.0_dp
        if (.not. valid) return
        valid = options%objective_tolerance >= 0.0_dp
        if (.not. valid) return
        valid = options%lower_log_rate_offset < options%upper_log_rate_offset
        if (.not. valid) return
        valid = valid_log_rate_offset(options%lower_log_rate_offset)
        if (.not. valid) return
        valid = valid_log_rate_offset(options%upper_log_rate_offset)
    end function valid_lbfgsb_options

end module fortml_poisson_likelihood
