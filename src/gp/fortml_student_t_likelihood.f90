module fortml_student_t_likelihood
    !! Stable Student-t observation likelihood products for GP latent states.
    !!
    !! The latent locations are fixed by the GP inference path.  The two
    !! likelihood coordinates are
    !!
    !!     theta = [ log(scale), log(nu) ].
    !!
    !! Exponentiating the coordinates keeps both scale and degrees of freedom
    !! positive at every optimizer iterate.  The free procedures return the
    !! summed log likelihood; the objective object returns its negative, as
    !! required by FortOpt's minimizers.  All products include the
    !! normalization constant, so this is a probability rather than a score.
    !!
    !! This is a fixed-latent-state contract: differentiating a GP mode or a
    !! covariance factorization belongs to the inference-specific adapters.
    !! CPU is the reference implementation.  CUDA requests are refused until
    !! a resident latent batch and special-function path are linked; no hidden
    !! host fallback is used.

    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_CONVERGENCE_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    use fortopt_objective, only: objective_t
    implicit none
    private

    integer, parameter, public :: STUDENT_T_LIKELIHOOD_N_PARAMETERS = 2

    type, public :: student_t_likelihood_t
        !! Fixed-latent Student-t objective over transformed likelihood state.
        private
        real(dp), allocatable :: observations(:), locations(:)
        real(dp) :: log_scale = 0.0_dp
        real(dp) :: log_nu = log(4.0_dp)
        integer :: device_kind = FORTML_DEVICE_CPU
    contains
        procedure, public :: initialize => student_t_likelihood_initialize
        procedure, public :: initialized => student_t_likelihood_initialized
        procedure, public :: device_supported => student_t_likelihood_device_supported
        procedure, public :: parameter_count => student_t_likelihood_parameter_count
        procedure, public :: parameters => student_t_likelihood_parameters
        procedure, public :: set_parameters => student_t_likelihood_set_parameters
        procedure, public :: value_gradient => student_t_likelihood_value_gradient
        procedure, public :: jvp => student_t_likelihood_jvp
        procedure, public :: vjp => student_t_likelihood_vjp
        procedure, public :: hvp => student_t_likelihood_hvp
        procedure, public :: fortopt => student_t_likelihood_fortopt
    end type student_t_likelihood_t

    public :: student_t_log_likelihood_value
    public :: student_t_log_likelihood_gradient
    public :: student_t_log_likelihood_jvp
    public :: student_t_log_likelihood_vjp
    public :: student_t_log_likelihood_hvp

contains

    subroutine student_t_likelihood_initialize(self, observations, locations, status, &
            parameters, device_kind)
        class(student_t_likelihood_t), intent(out) :: self
        real(dp), intent(in) :: observations(:), locations(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: parameters(:)
        integer, intent(in), optional :: device_kind
        real(dp) :: candidate(2)

        self%device_kind = FORTML_DEVICE_CPU
        if (present(device_kind)) self%device_kind = device_kind
        if (self%device_kind == FORTML_DEVICE_CUDA) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "Student-t likelihood: resident CUDA special-function path is not linked")
            return
        end if
        if (self%device_kind /= FORTML_DEVICE_CPU) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Student-t likelihood: device kind is invalid")
            return
        end if
        if (size(observations) < 1 .or. size(locations) /= size(observations)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Student-t likelihood: observation and latent shapes disagree")
            return
        end if
        if (any(.not. ieee_is_finite(observations)) .or. &
            any(.not. ieee_is_finite(locations))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Student-t likelihood: observations and latents must be finite")
            return
        end if
        candidate = [self%log_scale, self%log_nu]
        if (present(parameters)) then
            if (size(parameters) /= STUDENT_T_LIKELIHOOD_N_PARAMETERS) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "Student-t likelihood: parameter count must be two")
                return
            end if
            candidate = parameters
        end if
        if (any(.not. ieee_is_finite(candidate)) .or. &
            .not. valid_log_positive(candidate(1)) .or. &
            .not. valid_log_positive(candidate(2))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Student-t likelihood: transformed parameters must be finite")
            return
        end if
        allocate(self%observations, source=observations)
        allocate(self%locations, source=locations)
        self%log_scale = candidate(1)
        self%log_nu = candidate(2)
        call status_set(status, FORTNUM_OK, "")
    end subroutine student_t_likelihood_initialize

    logical function student_t_likelihood_initialized(self) result(yes)
        class(student_t_likelihood_t), intent(in) :: self

        yes = allocated(self%observations)
        if (.not. yes) return
        yes = allocated(self%locations)
        if (.not. yes) return
        yes = size(self%observations) > 0 .and. &
            size(self%locations) == size(self%observations)
    end function student_t_likelihood_initialized

    logical function student_t_likelihood_device_supported(self, device_kind) result(yes)
        class(student_t_likelihood_t), intent(in) :: self
        integer, intent(in), optional :: device_kind
        integer :: requested

        requested = self%device_kind
        if (present(device_kind)) requested = device_kind
        yes = requested == FORTML_DEVICE_CPU .and. self%initialized()
    end function student_t_likelihood_device_supported

    integer function student_t_likelihood_parameter_count(self) result(count)
        class(student_t_likelihood_t), intent(in) :: self

        count = 0
        if (self%initialized()) count = STUDENT_T_LIKELIHOOD_N_PARAMETERS
    end function student_t_likelihood_parameter_count

    function student_t_likelihood_parameters(self) result(parameters)
        class(student_t_likelihood_t), intent(in) :: self
        real(dp), allocatable :: parameters(:)

        allocate(parameters(0))
        if (.not. self%initialized()) return
        deallocate(parameters)
        allocate(parameters(STUDENT_T_LIKELIHOOD_N_PARAMETERS))
        parameters = [self%log_scale, self%log_nu]
    end function student_t_likelihood_parameters

    subroutine student_t_likelihood_set_parameters(self, parameters, status)
        class(student_t_likelihood_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:)
        type(fortnum_status_t), intent(out) :: status

        if (.not. self%initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Student-t likelihood: set before initialize")
            return
        end if
        if (size(parameters) /= STUDENT_T_LIKELIHOOD_N_PARAMETERS) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Student-t likelihood: parameter shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(parameters))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Student-t likelihood: transformed parameters must be finite")
            return
        end if
        if (.not. valid_log_positive(parameters(1)) .or. &
            .not. valid_log_positive(parameters(2))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Student-t likelihood: parameter shape or values are invalid")
            return
        end if
        ! No derived state depends on these coordinates, so commit only after
        ! all validation succeeds.  This is the transaction boundary exposed to
        ! FortOpt line searches.
        self%log_scale = parameters(1)
        self%log_nu = parameters(2)
        call status_set(status, FORTNUM_OK, "")
    end subroutine student_t_likelihood_set_parameters

    subroutine student_t_likelihood_value_gradient(self, parameters, value, gradient, &
            status)
        class(student_t_likelihood_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: log_value
        real(dp) :: log_gradient(2)

        value = huge(1.0_dp)
        gradient = 0.0_dp
        if (.not. self%initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Student-t likelihood objective: adapter is not initialized")
            return
        end if
        if (size(parameters) /= STUDENT_T_LIKELIHOOD_N_PARAMETERS .or. &
            size(gradient) /= size(parameters)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Student-t likelihood objective: parameter or gradient shape is invalid")
            return
        end if
        call self%set_parameters(parameters, status)
        if (status%code /= FORTNUM_OK) return
        call student_t_log_likelihood_value_gradient(self%observations, self%locations, &
            parameters, log_value, log_gradient, status)
        if (status%code /= FORTNUM_OK) return
        value = -log_value
        gradient = -log_gradient
        if (.not. ieee_is_finite(value) .or. any(.not. ieee_is_finite(gradient))) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "Student-t likelihood objective: result is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine student_t_likelihood_value_gradient

    subroutine student_t_likelihood_jvp(self, parameters, direction, value, tangent, &
            status)
        class(student_t_likelihood_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:), direction(:)
        real(dp), intent(out) :: value, tangent
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: gradient(2)

        value = huge(1.0_dp)
        tangent = 0.0_dp
        if (size(direction) /= size(parameters)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Student-t likelihood objective JVP: direction shape is invalid")
            return
        end if
        call self%value_gradient(parameters, value, gradient, status)
        if (status%code /= FORTNUM_OK) return
        tangent = dot_product(gradient, direction)
        call status_set(status, FORTNUM_OK, "")
    end subroutine student_t_likelihood_jvp

    subroutine student_t_likelihood_vjp(self, parameters, value_bar, parameter_bar, status)
        class(student_t_likelihood_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:), value_bar
        real(dp), intent(out) :: parameter_bar(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: value

        parameter_bar = 0.0_dp
        if (.not. ieee_is_finite(value_bar) .or. size(parameter_bar) /= size(parameters)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Student-t likelihood objective VJP: cotangent or shape is invalid")
            return
        end if
        call self%value_gradient(parameters, value, parameter_bar, status)
        if (status%code /= FORTNUM_OK) return
        parameter_bar = value_bar*parameter_bar
        call status_set(status, FORTNUM_OK, "")
    end subroutine student_t_likelihood_vjp

    subroutine student_t_likelihood_hvp(self, parameters, direction, product, status)
        class(student_t_likelihood_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:), direction(:)
        real(dp), intent(out) :: product(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: log_hessian(2, 2)

        product = 0.0_dp
        if (.not. self%initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Student-t likelihood objective HVP: adapter is not initialized")
            return
        end if
        if (size(parameters) /= STUDENT_T_LIKELIHOOD_N_PARAMETERS .or. &
            size(direction) /= size(parameters) .or. size(product) /= size(parameters)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Student-t likelihood objective HVP: direction or output shape is invalid")
            return
        end if
        call self%set_parameters(parameters, status)
        if (status%code /= FORTNUM_OK) return
        call student_t_log_likelihood_hessian(self%observations, self%locations, parameters, &
            log_hessian, status)
        if (status%code /= FORTNUM_OK) return
        product = -matmul(log_hessian, direction)
        if (any(.not. ieee_is_finite(product))) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "Student-t likelihood objective HVP: result is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine student_t_likelihood_hvp

    subroutine student_t_likelihood_fortopt(self, objective, status)
        class(student_t_likelihood_t), target, intent(inout) :: self
        type(objective_t), intent(out) :: objective
        type(fortnum_status_t), intent(out) :: status

        if (.not. self%initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Student-t likelihood objective: adapter is not initialized")
            return
        end if
        call objective%initialize_context(STUDENT_T_LIKELIHOOD_N_PARAMETERS, self, &
            student_t_likelihood_context, status)
    end subroutine student_t_likelihood_fortopt

    subroutine student_t_likelihood_context(context_any, parameters, value, gradient, &
            status)
        class(*), intent(inout) :: context_any
        real(dp), intent(in) :: parameters(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status

        select type (adapter => context_any)
            type is (student_t_likelihood_t)
            call adapter%value_gradient(parameters, value, gradient, status)
        class default
            value = huge(1.0_dp)
            gradient = 0.0_dp
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Student-t likelihood objective: context has the wrong type")
        end select
    end subroutine student_t_likelihood_context

    subroutine student_t_log_likelihood_value(observations, locations, parameters, &
            value, status)
        real(dp), intent(in) :: observations(:), locations(:), parameters(:)
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: gradient(2)

        call student_t_log_likelihood_value_gradient(observations, locations, parameters, &
            value, gradient, status)
    end subroutine student_t_log_likelihood_value

    subroutine student_t_log_likelihood_gradient(observations, locations, parameters, &
            gradient, status)
        real(dp), intent(in) :: observations(:), locations(:), parameters(:)
        real(dp), intent(out) :: gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: value

        gradient = 0.0_dp
        if (size(gradient) /= STUDENT_T_LIKELIHOOD_N_PARAMETERS) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Student-t likelihood gradient: output shape is invalid")
            return
        end if
        call student_t_log_likelihood_value_gradient(observations, locations, parameters, &
            value, gradient, status)
    end subroutine student_t_log_likelihood_gradient

    subroutine student_t_log_likelihood_jvp(observations, locations, parameters, direction, &
            value, value_dot, status)
        real(dp), intent(in) :: observations(:), locations(:), parameters(:), direction(:)
        real(dp), intent(out) :: value, value_dot
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: gradient(2)

        value = 0.0_dp
        value_dot = 0.0_dp
        if (size(direction) /= STUDENT_T_LIKELIHOOD_N_PARAMETERS) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Student-t likelihood JVP: direction shape is invalid")
            return
        end if
        call student_t_log_likelihood_value_gradient(observations, locations, parameters, &
            value, gradient, status)
        if (status%code /= FORTNUM_OK) return
        value_dot = dot_product(gradient, direction)
        call status_set(status, FORTNUM_OK, "")
    end subroutine student_t_log_likelihood_jvp

    subroutine student_t_log_likelihood_vjp(observations, locations, parameters, value_bar, &
            parameter_bar, status)
        real(dp), intent(in) :: observations(:), locations(:), parameters(:), value_bar
        real(dp), intent(out) :: parameter_bar(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: value

        parameter_bar = 0.0_dp
        if (size(parameter_bar) /= STUDENT_T_LIKELIHOOD_N_PARAMETERS .or. &
            .not. ieee_is_finite(value_bar)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Student-t likelihood VJP: cotangent or output shape is invalid")
            return
        end if
        call student_t_log_likelihood_value_gradient(observations, locations, parameters, &
            value, parameter_bar, status)
        if (status%code /= FORTNUM_OK) return
        parameter_bar = value_bar*parameter_bar
        call status_set(status, FORTNUM_OK, "")
    end subroutine student_t_log_likelihood_vjp

    subroutine student_t_log_likelihood_hvp(observations, locations, parameters, direction, &
            product, status)
        real(dp), intent(in) :: observations(:), locations(:), parameters(:), direction(:)
        real(dp), intent(out) :: product(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: hessian(2, 2)

        product = 0.0_dp
        if (size(direction) /= STUDENT_T_LIKELIHOOD_N_PARAMETERS .or. &
            size(product) /= STUDENT_T_LIKELIHOOD_N_PARAMETERS) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Student-t likelihood HVP: direction or output shape is invalid")
            return
        end if
        call student_t_log_likelihood_hessian(observations, locations, parameters, hessian, &
            status)
        if (status%code /= FORTNUM_OK) return
        product = matmul(hessian, direction)
        call status_set(status, FORTNUM_OK, "")
    end subroutine student_t_log_likelihood_hvp

    subroutine student_t_log_likelihood_value_gradient(observations, locations, parameters, &
            value, gradient, status)
        real(dp), intent(in) :: observations(:), locations(:), parameters(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: hessian(2, 2)

        call student_t_log_likelihood_evaluate(observations, locations, parameters, value, &
            gradient, hessian, status)
    end subroutine student_t_log_likelihood_value_gradient

    subroutine student_t_log_likelihood_hessian(observations, locations, parameters, hessian, &
            status)
        real(dp), intent(in) :: observations(:), locations(:), parameters(:)
        real(dp), intent(out) :: hessian(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: value, gradient(2)

        hessian = 0.0_dp
        if (size(hessian, 1) /= 2 .or. size(hessian, 2) /= 2) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Student-t likelihood Hessian: output shape is invalid")
            return
        end if
        call student_t_log_likelihood_evaluate(observations, locations, parameters, value, &
            gradient, hessian, status)
    end subroutine student_t_log_likelihood_hessian

    subroutine student_t_log_likelihood_evaluate(observations, locations, parameters, value, &
            gradient, hessian, status)
        real(dp), intent(in) :: observations(:), locations(:), parameters(:)
        real(dp), intent(out) :: value, gradient(:), hessian(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: scale, nu, residual, normalized, q, ratio, log_one_plus_q
        real(dp) :: digamma_difference, trigamma_difference, gradient_nu
        real(dp) :: hessian_aa, hessian_ab, hessian_bb
        integer :: i

        value = 0.0_dp
        gradient = 0.0_dp
        hessian = 0.0_dp
        if (size(observations) < 1 .or. size(locations) /= size(observations)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Student-t likelihood: observation and latent shapes disagree")
            return
        end if
        if (size(parameters) /= STUDENT_T_LIKELIHOOD_N_PARAMETERS .or. &
            size(gradient) /= STUDENT_T_LIKELIHOOD_N_PARAMETERS .or. &
            size(hessian, 1) /= 2 .or. size(hessian, 2) /= 2) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Student-t likelihood: parameter or product shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(observations)) .or. &
            any(.not. ieee_is_finite(locations)) .or. &
            any(.not. ieee_is_finite(parameters))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Student-t likelihood: inputs must be finite")
            return
        end if
        scale = exp(parameters(1))
        nu = exp(parameters(2))
        if (.not. ieee_is_finite(scale) .or. .not. ieee_is_finite(nu) .or. &
            scale <= 0.0_dp .or. nu <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Student-t likelihood: transformed scale or degrees of freedom are invalid")
            return
        end if
        digamma_difference = digamma_positive(0.5_dp*(nu + 1.0_dp)) - &
            digamma_positive(0.5_dp*nu)
        trigamma_difference = trigamma_positive(0.5_dp*(nu + 1.0_dp)) - &
            trigamma_positive(0.5_dp*nu)
        do i = 1, size(observations)
            residual = observations(i) - locations(i)
            normalized = residual/(scale*sqrt(nu))
            q = normalized*normalized
            if (.not. ieee_is_finite(q)) then
                call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                    "Student-t likelihood: residual scaling overflowed")
                return
            end if
            log_one_plus_q = log_one_plus_nonnegative(q)
            ratio = q/(1.0_dp + q)
            value = value + log_gamma(0.5_dp*(nu + 1.0_dp)) &
                - log_gamma(0.5_dp*nu) - 0.5_dp*log(nu*4.0_dp*atan(1.0_dp)) &
                - parameters(1) - 0.5_dp*(nu + 1.0_dp)*log_one_plus_q
            gradient(1) = gradient(1) - 1.0_dp + (nu + 1.0_dp)*ratio
            gradient_nu = 0.5_dp*digamma_difference - 0.5_dp/nu &
                - 0.5_dp*log_one_plus_q &
                + 0.5_dp*(nu + 1.0_dp)*ratio/nu
            gradient(2) = gradient(2) + nu*gradient_nu
            hessian_aa = -2.0_dp*(nu + 1.0_dp)*ratio*(1.0_dp - ratio)
            hessian_ab = nu*ratio - (nu + 1.0_dp)*ratio*(1.0_dp - ratio)
            hessian_bb = 0.5_dp*nu*digamma_difference &
                + 0.25_dp*nu*nu*trigamma_difference &
                - 0.5_dp*nu*log_one_plus_q + nu*ratio &
                - 0.5_dp*(nu + 1.0_dp)*ratio*(1.0_dp - ratio)
            hessian(1, 1) = hessian(1, 1) + hessian_aa
            hessian(1, 2) = hessian(1, 2) + hessian_ab
            hessian(2, 1) = hessian(2, 1) + hessian_ab
            hessian(2, 2) = hessian(2, 2) + hessian_bb
        end do
        if (.not. ieee_is_finite(value) .or. any(.not. ieee_is_finite(gradient)) .or. &
            any(.not. ieee_is_finite(hessian))) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "Student-t likelihood: product is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine student_t_log_likelihood_evaluate

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
        value = value + inverse + 0.5_dp*inverse_square + inverse_square*inverse/6.0_dp &
            - inverse_square**2*inverse/30.0_dp + inverse_square**3*inverse/42.0_dp &
            - inverse_square**4*inverse/30.0_dp + 5.0_dp*inverse_square**5/66.0_dp
    end function trigamma_positive

    pure real(dp) function log_one_plus_nonnegative(argument) result(value)
        !! Stable `log(1+x)` for the nonnegative quadratic ratio.
        real(dp), intent(in) :: argument
        real(dp) :: term

        if (argument < 1.0e-4_dp) then
            term = argument
            value = term - 0.5_dp*term*term + term*term*term/3.0_dp &
                - term**4/4.0_dp + term**5/5.0_dp
        else
            value = log(1.0_dp + argument)
        end if
    end function log_one_plus_nonnegative

    pure logical function valid_log_positive(argument) result(valid)
        real(dp), intent(in) :: argument

        valid = ieee_is_finite(argument)
        if (.not. valid) return
        valid = ieee_is_finite(exp(argument))
    end function valid_log_positive

end module fortml_student_t_likelihood
