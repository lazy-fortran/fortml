module fortml_physics_objective
    !! Composable weighted residual objectives for physics-informed models.
    !!
    !! A constraint is a residual map ``r(theta)`` and its first-order
    !! products.  The value is the normalized weighted squared residual,
    !! ``weight * dot_product(r,r) / (2*n_residuals)``.  Four named slots are
    !! available in the composite objective: data, differential-equation
    !! residual, boundary/initial, and conservation/invariant.  Keeping these
    !! slots separate makes PINN, physics-informed GP, and symplectic adapters
    !! observable without prescribing a model representation.
    !!
    !! The callbacks are deliberately explicit.  No finite-difference fallback
    !! is hidden behind a derivative method: callers provide a residual JVP
    !! and VJP.  Providers that can differentiate their VJP in a parameter
    !! direction may also provide the optional reverse-over-forward HVP
    !! callback; otherwise second-order products return a typed refusal.  A
    !! constraint is CPU/device agnostic; resident GPU callbacks can be
    !! supplied by an adapter without changing the objective reduction.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortopt_objective, only: objective_t
    implicit none
    private

    abstract interface
        subroutine physics_residual_proc(context, theta, residual, status)
            import :: dp, fortnum_status_t
            class(*), pointer, intent(in) :: context
            real(dp), intent(in) :: theta(:)
            real(dp), intent(out) :: residual(:)
            type(fortnum_status_t), intent(out) :: status
        end subroutine physics_residual_proc

        subroutine physics_residual_jvp_proc(context, theta, theta_dot, &
                residual, residual_dot, status)
            import :: dp, fortnum_status_t
            class(*), pointer, intent(in) :: context
            real(dp), intent(in) :: theta(:), theta_dot(:)
            real(dp), intent(out) :: residual(:), residual_dot(:)
            type(fortnum_status_t), intent(out) :: status
        end subroutine physics_residual_jvp_proc

        subroutine physics_residual_vjp_proc(context, theta, residual_bar, &
                theta_bar, status)
            import :: dp, fortnum_status_t
            class(*), pointer, intent(in) :: context
            real(dp), intent(in) :: theta(:), residual_bar(:)
            real(dp), intent(out) :: theta_bar(:)
            type(fortnum_status_t), intent(out) :: status
        end subroutine physics_residual_vjp_proc

        subroutine physics_residual_hvp_proc(context, theta, theta_dot, &
                residual_bar, residual_bar_dot, theta_hvp, status)
            !! Differentiate a residual VJP in ``theta_dot``.
            !!
            !! The callback returns
            !! ``d/dtheta [J(theta)^T residual_bar] theta_dot`` plus the
            !! contribution ``J(theta)^T residual_bar_dot``.  The physics
            !! constraint supplies the normalized residual and its JVP as
            !! ``residual_bar`` and ``residual_bar_dot`` respectively.  This
            !! reverse-over-forward contract is the exact Hessian-vector
            !! product of the weighted squared residual and does not form a
            !! Jacobian or Hessian.
            import :: dp, fortnum_status_t
            class(*), pointer, intent(in) :: context
            real(dp), intent(in) :: theta(:), theta_dot(:)
            real(dp), intent(in) :: residual_bar(:), residual_bar_dot(:)
            real(dp), intent(out) :: theta_hvp(:)
            type(fortnum_status_t), intent(out) :: status
        end subroutine physics_residual_hvp_proc

        subroutine physics_objective_context_proc(context, theta, value, &
                gradient, status)
            import :: dp, fortnum_status_t
            class(*), intent(inout) :: context
            real(dp), intent(in) :: theta(:)
            real(dp), intent(out) :: value, gradient(:)
            type(fortnum_status_t), intent(out) :: status
        end subroutine physics_objective_context_proc
    end interface

    type, public :: physics_constraint_t
        !! One normalized weighted residual term.
        private
        integer :: n_parameters = 0
        integer :: n_residuals = 0
        real(dp) :: weight = 0.0_dp
        logical :: ready = .false.
        class(*), pointer :: context => null()
        procedure(physics_residual_proc), pointer, nopass :: residual_proc => null()
        procedure(physics_residual_jvp_proc), pointer, nopass :: jvp_proc => null()
        procedure(physics_residual_vjp_proc), pointer, nopass :: vjp_proc => null()
        procedure(physics_residual_hvp_proc), pointer, nopass :: hvp_proc => null()
    contains
        procedure, public :: initialize => physics_constraint_initialize
        procedure, public :: initialized => physics_constraint_initialized
        procedure, public :: parameter_count => physics_constraint_parameter_count
        procedure, public :: residual_count => physics_constraint_residual_count
        procedure, public :: value => physics_constraint_value
        procedure, public :: gradient => physics_constraint_gradient
        procedure, public :: value_gradient => physics_constraint_value_gradient
        procedure, public :: jvp => physics_constraint_jvp
        procedure, public :: vjp => physics_constraint_vjp
        procedure, public :: hvp => physics_constraint_hvp
    end type physics_constraint_t

    type, public :: physics_objective_t
        !! Data + residual + boundary + conservation objective composition.
        private
        integer :: n_parameters = 0
        integer :: active_terms = 0
        logical :: ready = .false.
        type(physics_constraint_t) :: data
        type(physics_constraint_t) :: residual
        type(physics_constraint_t) :: boundary
        type(physics_constraint_t) :: conservation
    contains
        procedure, public :: initialize => physics_objective_initialize
        procedure, public :: initialized => physics_objective_initialized
        procedure, public :: parameter_count => physics_objective_parameter_count
        procedure, public :: term_values => physics_objective_term_values
        procedure, public :: term_gradients => physics_objective_term_gradients
        procedure, public :: term_hvps => physics_objective_term_hvps
        procedure, public :: value => physics_objective_value
        procedure, public :: gradient => physics_objective_gradient
        procedure, public :: value_gradient => physics_objective_value_gradient
        procedure, public :: jvp => physics_objective_jvp
        procedure, public :: vjp => physics_objective_vjp
        procedure, public :: hvp => physics_objective_hvp
        procedure, public :: as_objective => physics_objective_as_objective
    end type physics_objective_t

    public :: physics_residual_proc, physics_residual_jvp_proc
    public :: physics_residual_vjp_proc, physics_residual_hvp_proc
    public :: physics_objective_context_proc

contains

    subroutine physics_constraint_initialize(self, n_parameters, n_residuals, &
            weight, context, residual_proc, jvp_proc, vjp_proc, status, hvp_proc)
        class(physics_constraint_t), intent(out) :: self
        integer, intent(in) :: n_parameters, n_residuals
        real(dp), intent(in) :: weight
        class(*), target, intent(inout), optional :: context
        procedure(physics_residual_proc) :: residual_proc
        procedure(physics_residual_jvp_proc) :: jvp_proc
        procedure(physics_residual_vjp_proc) :: vjp_proc
        type(fortnum_status_t), intent(out) :: status
        procedure(physics_residual_hvp_proc), optional :: hvp_proc

        self%n_parameters = 0
        self%n_residuals = 0
        self%weight = 0.0_dp
        self%ready = .false.
        nullify(self%context, self%residual_proc, self%jvp_proc, self%vjp_proc, &
            self%hvp_proc)
        if (n_parameters < 1 .or. n_residuals < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "physics constraint: dimensions must be positive")
            return
        end if
        if (.not. ieee_is_finite(weight) .or. weight <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "physics constraint: weight must be finite and positive")
            return
        end if
        self%n_parameters = n_parameters
        self%n_residuals = n_residuals
        self%weight = weight
        if (present(context)) self%context => context
        self%residual_proc => residual_proc
        self%jvp_proc => jvp_proc
        self%vjp_proc => vjp_proc
        if (present(hvp_proc)) self%hvp_proc => hvp_proc
        self%ready = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine physics_constraint_initialize

    logical function physics_constraint_initialized(self) result(yes)
        class(physics_constraint_t), intent(in) :: self

        yes = self%ready
        if (.not. yes) return
        if (.not. associated(self%residual_proc)) yes = .false.
        if (.not. associated(self%jvp_proc)) yes = .false.
        if (.not. associated(self%vjp_proc)) yes = .false.
    end function physics_constraint_initialized

    integer function physics_constraint_parameter_count(self) result(count)
        class(physics_constraint_t), intent(in) :: self

        count = self%n_parameters
    end function physics_constraint_parameter_count

    integer function physics_constraint_residual_count(self) result(count)
        class(physics_constraint_t), intent(in) :: self

        count = self%n_residuals
    end function physics_constraint_residual_count

    subroutine physics_constraint_value(self, theta, value, status)
        class(physics_constraint_t), intent(in) :: self
        real(dp), intent(in) :: theta(:)
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status

        real(dp), allocatable :: residual(:)

        call validate_constraint_call(self, theta, status)
        if (status%code /= FORTNUM_OK) return
        allocate(residual(self%n_residuals))
        call self%residual_proc(self%context, theta, residual, status)
        if (status%code /= FORTNUM_OK) return
        if (any(.not. ieee_is_finite(residual))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "physics constraint: residual is not finite")
            return
        end if
        value = self%weight*dot_product(residual, residual)/ &
            (2.0_dp*real(self%n_residuals, dp))
        call status_set(status, FORTNUM_OK, "")
    end subroutine physics_constraint_value

    subroutine physics_constraint_value_gradient(self, theta, value, gradient, status)
        class(physics_constraint_t), intent(in) :: self
        real(dp), intent(in) :: theta(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status

        real(dp), allocatable :: residual(:), residual_bar(:)

        if (size(gradient) /= self%n_parameters) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "physics constraint: gradient shape is invalid")
            return
        end if
        call validate_constraint_call(self, theta, status)
        if (status%code /= FORTNUM_OK) return
        allocate(residual(self%n_residuals), residual_bar(self%n_residuals))
        call self%residual_proc(self%context, theta, residual, status)
        if (status%code /= FORTNUM_OK) return
        if (any(.not. ieee_is_finite(residual))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "physics constraint: residual is not finite")
            return
        end if
        value = self%weight*dot_product(residual, residual)/ &
            (2.0_dp*real(self%n_residuals, dp))
        residual_bar = self%weight*residual/real(self%n_residuals, dp)
        call self%vjp_proc(self%context, theta, residual_bar, gradient, status)
        if (status%code /= FORTNUM_OK) return
        if (any(.not. ieee_is_finite(gradient))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "physics constraint: VJP returned a non-finite gradient")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine physics_constraint_value_gradient

    subroutine physics_constraint_gradient(self, theta, gradient, status)
        class(physics_constraint_t), intent(in) :: self
        real(dp), intent(in) :: theta(:)
        real(dp), intent(out) :: gradient(:)
        type(fortnum_status_t), intent(out) :: status

        real(dp) :: value

        call self%value_gradient(theta, value, gradient, status)
    end subroutine physics_constraint_gradient

    subroutine physics_constraint_jvp(self, theta, theta_dot, value, value_dot, status)
        class(physics_constraint_t), intent(in) :: self
        real(dp), intent(in) :: theta(:), theta_dot(:)
        real(dp), intent(out) :: value, value_dot
        type(fortnum_status_t), intent(out) :: status

        real(dp), allocatable :: residual(:), residual_dot(:)

        call validate_constraint_call(self, theta, status)
        if (status%code /= FORTNUM_OK) return
        if (size(theta_dot) /= self%n_parameters) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "physics constraint: JVP direction shape is invalid")
            return
        end if
        allocate(residual(self%n_residuals), residual_dot(self%n_residuals))
        call self%jvp_proc(self%context, theta, theta_dot, residual, residual_dot, status)
        if (status%code /= FORTNUM_OK) return
        if (any(.not. ieee_is_finite(residual)) .or. &
            any(.not. ieee_is_finite(residual_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "physics constraint: JVP returned a non-finite residual")
            return
        end if
        value = self%weight*dot_product(residual, residual)/ &
            (2.0_dp*real(self%n_residuals, dp))
        value_dot = self%weight*dot_product(residual, residual_dot)/ &
            real(self%n_residuals, dp)
        call status_set(status, FORTNUM_OK, "")
    end subroutine physics_constraint_jvp

    subroutine physics_constraint_vjp(self, theta, value_bar, theta_bar, status)
        class(physics_constraint_t), intent(in) :: self
        real(dp), intent(in) :: theta(:), value_bar
        real(dp), intent(out) :: theta_bar(:)
        type(fortnum_status_t), intent(out) :: status

        real(dp), allocatable :: residual(:), residual_bar(:)

        if (size(theta_bar) /= self%n_parameters) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "physics constraint: VJP output shape is invalid")
            return
        end if
        if (.not. ieee_is_finite(value_bar)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "physics constraint: VJP cotangent is not finite")
            return
        end if
        call validate_constraint_call(self, theta, status)
        if (status%code /= FORTNUM_OK) return
        allocate(residual(self%n_residuals), residual_bar(self%n_residuals))
        call self%residual_proc(self%context, theta, residual, status)
        if (status%code /= FORTNUM_OK) return
        if (any(.not. ieee_is_finite(residual))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "physics constraint: residual is not finite")
            return
        end if
        residual_bar = value_bar*self%weight*residual/ &
            real(self%n_residuals, dp)
        call self%vjp_proc(self%context, theta, residual_bar, theta_bar, status)
        if (status%code /= FORTNUM_OK) return
        if (any(.not. ieee_is_finite(theta_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "physics constraint: VJP returned a non-finite cotangent")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine physics_constraint_vjp

    subroutine physics_constraint_hvp(self, theta, theta_dot, theta_hvp, status)
        class(physics_constraint_t), intent(in) :: self
        real(dp), intent(in) :: theta(:), theta_dot(:)
        real(dp), intent(out) :: theta_hvp(:)
        type(fortnum_status_t), intent(out) :: status

        real(dp), allocatable :: residual(:), residual_dot(:)
        real(dp), allocatable :: residual_bar(:), residual_bar_dot(:)

        theta_hvp = 0.0_dp
        if (size(theta_hvp) /= self%n_parameters) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "physics constraint: HVP shape is invalid")
            return
        end if
        if (size(theta_dot) /= self%n_parameters .or. &
                size(theta) /= self%n_parameters) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "physics constraint: HVP shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(theta_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "physics constraint: HVP direction is not finite")
            return
        end if
        call validate_constraint_call(self, theta, status)
        if (status%code /= FORTNUM_OK) return
        if (.not. associated(self%hvp_proc)) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "physics constraint: residual HVP is not provided")
            return
        end if
        allocate(residual(self%n_residuals), residual_dot(self%n_residuals))
        allocate(residual_bar(self%n_residuals), residual_bar_dot(self%n_residuals))
        call self%jvp_proc(self%context, theta, theta_dot, residual, residual_dot, status)
        if (status%code /= FORTNUM_OK) return
        if (any(.not. ieee_is_finite(residual)) .or. &
                any(.not. ieee_is_finite(residual_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "physics constraint: HVP JVP returned a non-finite residual")
            return
        end if
        residual_bar = self%weight*residual/real(self%n_residuals, dp)
        residual_bar_dot = self%weight*residual_dot/real(self%n_residuals, dp)
        call self%hvp_proc(self%context, theta, theta_dot, residual_bar, &
            residual_bar_dot, theta_hvp, status)
        if (status%code /= FORTNUM_OK) return
        if (any(.not. ieee_is_finite(theta_hvp))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "physics constraint: HVP returned a non-finite cotangent")
            return
        end if
    end subroutine physics_constraint_hvp

    subroutine physics_objective_initialize(self, n_parameters, data, residual, &
            boundary, conservation, status)
        class(physics_objective_t), intent(out) :: self
        integer, intent(in) :: n_parameters
        type(physics_constraint_t), intent(in), optional :: data, residual, boundary
        type(physics_constraint_t), intent(in), optional :: conservation
        type(fortnum_status_t), intent(out) :: status
        !! Default-initialized instances, standing in for empty
        !! structure constructors: nvfortran rejects `T()` outright,
        !! and a declared local carries the same default init.
        type(physics_constraint_t) :: physics_constraint_t_default

        self%n_parameters = 0
        self%active_terms = 0
        self%ready = .false.
        self%data = physics_constraint_t_default
        self%residual = physics_constraint_t_default
        self%boundary = physics_constraint_t_default
        self%conservation = physics_constraint_t_default
        if (n_parameters < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "physics objective: parameter count must be positive")
            return
        end if
        self%n_parameters = n_parameters
        if (present(data)) self%data = data
        if (present(residual)) self%residual = residual
        if (present(boundary)) self%boundary = boundary
        if (present(conservation)) self%conservation = conservation
        call validate_term(self%data, n_parameters, self%active_terms, status)
        if (status%code /= FORTNUM_OK) return
        call validate_term(self%residual, n_parameters, self%active_terms, status)
        if (status%code /= FORTNUM_OK) return
        call validate_term(self%boundary, n_parameters, self%active_terms, status)
        if (status%code /= FORTNUM_OK) return
        call validate_term(self%conservation, n_parameters, self%active_terms, status)
        if (status%code /= FORTNUM_OK) return
        if (self%active_terms < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "physics objective: at least one constraint is required")
            return
        end if
        self%ready = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine physics_objective_initialize

    logical function physics_objective_initialized(self) result(yes)
        class(physics_objective_t), intent(in) :: self

        yes = self%ready .and. self%active_terms > 0
    end function physics_objective_initialized

    integer function physics_objective_parameter_count(self) result(count)
        class(physics_objective_t), intent(in) :: self

        count = self%n_parameters
    end function physics_objective_parameter_count

    subroutine physics_objective_term_values(self, theta, values, status)
        !! Return the four named constraint contributions in a fixed order.
        !!
        !! ``values = [data, residual, boundary, conservation]``.  An
        !! inactive slot is reported as zero, so callers can use this method
        !! as a stable residual-balancing diagnostic without inspecting the
        !! private constraint components.  The entries sum to ``value(theta)``
        !! for every initialized objective.
        class(physics_objective_t), intent(in) :: self
        real(dp), intent(in) :: theta(:)
        real(dp), intent(out) :: values(:)
        type(fortnum_status_t), intent(out) :: status

        if (.not. self%initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "physics objective: object is not initialized")
            return
        end if
        if (size(theta) /= self%n_parameters .or. &
            any(.not. ieee_is_finite(theta)) .or. size(values) /= 4) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "physics objective: diagnostic shape or values are invalid")
            return
        end if
        call constraint_value_or_zero(self%data, theta, values(1), status)
        if (status%code /= FORTNUM_OK) return
        call constraint_value_or_zero(self%residual, theta, values(2), status)
        if (status%code /= FORTNUM_OK) return
        call constraint_value_or_zero(self%boundary, theta, values(3), status)
        if (status%code /= FORTNUM_OK) return
        call constraint_value_or_zero(self%conservation, theta, values(4), status)
    end subroutine physics_objective_term_values

    subroutine physics_objective_term_gradients(self, theta, gradients, status)
        !! Return one gradient column for each named objective term.
        !!
        !! Columns are ordered ``[data, residual, boundary, conservation]``;
        !! inactive terms are zero.  Unlike the aggregate gradient this
        !! diagnostic keeps residual-balancing contributions separately
        !! observable without exposing the objective's private slots.
        class(physics_objective_t), intent(in) :: self
        real(dp), intent(in) :: theta(:)
        real(dp), intent(out) :: gradients(:,:)
        type(fortnum_status_t), intent(out) :: status

        if (.not. self%initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "physics objective: object is not initialized")
            return
        end if
        if (size(theta) /= self%n_parameters .or. &
            any(.not. ieee_is_finite(theta)) .or. &
            size(gradients, 1) /= self%n_parameters .or. &
            size(gradients, 2) /= 4) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "physics objective: gradient diagnostic shape is invalid")
            return
        end if
        gradients = 0.0_dp
        call term_gradient_or_zero(self%data, theta, gradients(:, 1), status)
        if (status%code /= FORTNUM_OK) return
        call term_gradient_or_zero(self%residual, theta, gradients(:, 2), status)
        if (status%code /= FORTNUM_OK) return
        call term_gradient_or_zero(self%boundary, theta, gradients(:, 3), status)
        if (status%code /= FORTNUM_OK) return
        call term_gradient_or_zero(self%conservation, theta, gradients(:, 4), status)
    end subroutine physics_objective_term_gradients

    subroutine physics_objective_term_hvps(self, theta, theta_dot, hvps, status)
        !! Return one exact HVP column for each named objective term.
        !!
        !! The aggregate HVP has the same column sum.  If an active term has
        !! no reverse-over-forward callback the call returns that term's typed
        !! ``FORTNUM_NOT_IMPLEMENTED`` refusal and leaves all output zero.
        class(physics_objective_t), intent(in) :: self
        real(dp), intent(in) :: theta(:), theta_dot(:)
        real(dp), intent(out) :: hvps(:,:)
        type(fortnum_status_t), intent(out) :: status

        if (.not. self%initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "physics objective: object is not initialized")
            return
        end if
        if (size(theta) /= self%n_parameters .or. &
            size(theta_dot) /= self%n_parameters .or. &
            any(.not. ieee_is_finite(theta)) .or. &
            any(.not. ieee_is_finite(theta_dot)) .or. &
            size(hvps, 1) /= self%n_parameters .or. size(hvps, 2) /= 4) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "physics objective: HVP diagnostic shape is invalid")
            return
        end if
        hvps = 0.0_dp
        call term_hvp_or_zero(self%data, theta, theta_dot, hvps(:, 1), status)
        if (status%code /= FORTNUM_OK) return
        call term_hvp_or_zero(self%residual, theta, theta_dot, hvps(:, 2), status)
        if (status%code /= FORTNUM_OK) return
        call term_hvp_or_zero(self%boundary, theta, theta_dot, hvps(:, 3), status)
        if (status%code /= FORTNUM_OK) return
        call term_hvp_or_zero(self%conservation, theta, theta_dot, hvps(:, 4), status)
    end subroutine physics_objective_term_hvps

    subroutine physics_objective_value_gradient(self, theta, value, gradient, status)
        class(physics_objective_t), intent(in) :: self
        real(dp), intent(in) :: theta(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status

        real(dp) :: term_value
        real(dp), allocatable :: term_gradient(:)

        call validate_objective_call(self, theta, gradient, status)
        if (status%code /= FORTNUM_OK) return
        value = 0.0_dp
        gradient = 0.0_dp
        allocate(term_gradient(self%n_parameters))
        call accumulate_value_gradient(self%data, theta, value, gradient, &
            term_value, term_gradient, status)
        if (status%code /= FORTNUM_OK) return
        call accumulate_value_gradient(self%residual, theta, value, gradient, &
            term_value, term_gradient, status)
        if (status%code /= FORTNUM_OK) return
        call accumulate_value_gradient(self%boundary, theta, value, gradient, &
            term_value, term_gradient, status)
        if (status%code /= FORTNUM_OK) return
        call accumulate_value_gradient(self%conservation, theta, value, gradient, &
            term_value, term_gradient, status)
        if (status%code /= FORTNUM_OK) return
        call status_set(status, FORTNUM_OK, "")
    end subroutine physics_objective_value_gradient

    subroutine physics_objective_value(self, theta, value, status)
        class(physics_objective_t), intent(in) :: self
        real(dp), intent(in) :: theta(:)
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status

        real(dp), allocatable :: gradient(:)

        if (self%n_parameters > 0) then
            allocate(gradient(self%n_parameters))
        else
            allocate(gradient(0))
        end if
        call self%value_gradient(theta, value, gradient, status)
    end subroutine physics_objective_value

    subroutine physics_objective_gradient(self, theta, gradient, status)
        class(physics_objective_t), intent(in) :: self
        real(dp), intent(in) :: theta(:)
        real(dp), intent(out) :: gradient(:)
        type(fortnum_status_t), intent(out) :: status

        real(dp) :: value

        call self%value_gradient(theta, value, gradient, status)
    end subroutine physics_objective_gradient

    subroutine physics_objective_jvp(self, theta, theta_dot, value, value_dot, status)
        class(physics_objective_t), intent(in) :: self
        real(dp), intent(in) :: theta(:), theta_dot(:)
        real(dp), intent(out) :: value, value_dot
        type(fortnum_status_t), intent(out) :: status

        real(dp) :: term_value, term_dot
        integer :: n

        if (.not. self%initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "physics objective: object is not initialized")
            return
        end if
        n = self%n_parameters
        if (size(theta) /= n .or. size(theta_dot) /= n) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "physics objective: JVP shape is invalid")
            return
        end if
        value = 0.0_dp
        value_dot = 0.0_dp
        call accumulate_jvp(self%data, theta, theta_dot, value, value_dot, &
            term_value, term_dot, status)
        if (status%code /= FORTNUM_OK) return
        call accumulate_jvp(self%residual, theta, theta_dot, value, value_dot, &
            term_value, term_dot, status)
        if (status%code /= FORTNUM_OK) return
        call accumulate_jvp(self%boundary, theta, theta_dot, value, value_dot, &
            term_value, term_dot, status)
        if (status%code /= FORTNUM_OK) return
        call accumulate_jvp(self%conservation, theta, theta_dot, value, value_dot, &
            term_value, term_dot, status)
        if (status%code /= FORTNUM_OK) return
        call status_set(status, FORTNUM_OK, "")
    end subroutine physics_objective_jvp

    subroutine physics_objective_vjp(self, theta, value_bar, theta_bar, status)
        class(physics_objective_t), intent(in) :: self
        real(dp), intent(in) :: theta(:), value_bar
        real(dp), intent(out) :: theta_bar(:)
        type(fortnum_status_t), intent(out) :: status

        real(dp), allocatable :: term_bar(:)

        if (.not. self%initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "physics objective: object is not initialized")
            return
        end if
        if (size(theta) /= self%n_parameters .or. &
            size(theta_bar) /= self%n_parameters) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "physics objective: VJP shape is invalid")
            return
        end if
        if (.not. ieee_is_finite(value_bar)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "physics objective: VJP cotangent is not finite")
            return
        end if
        theta_bar = 0.0_dp
        allocate(term_bar(self%n_parameters))
        call accumulate_vjp(self%data, theta, value_bar, theta_bar, term_bar, status)
        if (status%code /= FORTNUM_OK) return
        call accumulate_vjp(self%residual, theta, value_bar, theta_bar, term_bar, status)
        if (status%code /= FORTNUM_OK) return
        call accumulate_vjp(self%boundary, theta, value_bar, theta_bar, term_bar, status)
        if (status%code /= FORTNUM_OK) return
        call accumulate_vjp(self%conservation, theta, value_bar, theta_bar, term_bar, status)
        if (status%code /= FORTNUM_OK) return
        call status_set(status, FORTNUM_OK, "")
    end subroutine physics_objective_vjp

    subroutine physics_objective_hvp(self, theta, theta_dot, theta_hvp, status)
        class(physics_objective_t), intent(in) :: self
        real(dp), intent(in) :: theta(:), theta_dot(:)
        real(dp), intent(out) :: theta_hvp(:)
        type(fortnum_status_t), intent(out) :: status

        theta_hvp = 0.0_dp
        if (.not. self%initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "physics objective: object is not initialized")
            return
        end if
        if (size(theta) /= self%n_parameters .or. &
            size(theta_dot) /= self%n_parameters .or. &
            size(theta_hvp) /= self%n_parameters) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "physics objective: HVP shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(theta_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "physics objective: HVP direction is not finite")
            return
        end if
        call accumulate_hvp(self%data, theta, theta_dot, theta_hvp, status)
        if (status%code /= FORTNUM_OK) return
        call accumulate_hvp(self%residual, theta, theta_dot, theta_hvp, status)
        if (status%code /= FORTNUM_OK) return
        call accumulate_hvp(self%boundary, theta, theta_dot, theta_hvp, status)
        if (status%code /= FORTNUM_OK) return
        call accumulate_hvp(self%conservation, theta, theta_dot, theta_hvp, status)
        if (status%code /= FORTNUM_OK) return
        call status_set(status, FORTNUM_OK, "")
    end subroutine physics_objective_hvp

    subroutine physics_objective_as_objective(self, objective, status)
        class(physics_objective_t), target, intent(inout) :: self
        type(objective_t), intent(out) :: objective
        type(fortnum_status_t), intent(out) :: status

        if (.not. self%initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "physics objective: object is not initialized")
            return
        end if
        call objective%initialize_context(self%n_parameters, self, &
            physics_objective_context, status)
    end subroutine physics_objective_as_objective

    subroutine validate_constraint_call(self, theta, status)
        class(physics_constraint_t), intent(in) :: self
        real(dp), intent(in) :: theta(:)
        type(fortnum_status_t), intent(out) :: status

        if (.not. self%initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "physics constraint: object is not initialized")
            return
        end if
        if (size(theta) /= self%n_parameters .or. &
            any(.not. ieee_is_finite(theta))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "physics constraint: parameter shape or values are invalid")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine validate_constraint_call

    subroutine validate_term(term, n_parameters, active_terms, status)
        type(physics_constraint_t), intent(in) :: term
        integer, intent(in) :: n_parameters
        integer, intent(inout) :: active_terms
        type(fortnum_status_t), intent(out) :: status

        if (.not. term%initialized()) then
            if (term%parameter_count() /= 0 .or. term%residual_count() /= 0) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "physics objective: malformed inactive constraint")
                return
            end if
            call status_set(status, FORTNUM_OK, "")
            return
        end if
        if (term%parameter_count() /= n_parameters) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "physics objective: constraint parameter count mismatch")
            return
        end if
        active_terms = active_terms + 1
        call status_set(status, FORTNUM_OK, "")
    end subroutine validate_term

    subroutine validate_objective_call(self, theta, gradient, status)
        class(physics_objective_t), intent(in) :: self
        real(dp), intent(in) :: theta(:)
        real(dp), intent(out) :: gradient(:)
        type(fortnum_status_t), intent(out) :: status

        if (.not. self%initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "physics objective: object is not initialized")
            return
        end if
        if (size(theta) /= self%n_parameters .or. &
            size(gradient) /= self%n_parameters .or. &
            any(.not. ieee_is_finite(theta))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "physics objective: parameter or gradient shape is invalid")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine validate_objective_call

    subroutine accumulate_value_gradient(term, theta, value, gradient, &
            term_value, term_gradient, status)
        type(physics_constraint_t), intent(in) :: term
        real(dp), intent(in) :: theta(:)
        real(dp), intent(inout) :: value, gradient(:)
        real(dp), intent(out) :: term_value, term_gradient(:)
        type(fortnum_status_t), intent(out) :: status

        if (.not. term%initialized()) then
            term_value = 0.0_dp
            term_gradient = 0.0_dp
            call status_set(status, FORTNUM_OK, "")
            return
        end if
        call term%value_gradient(theta, term_value, term_gradient, status)
        if (status%code /= FORTNUM_OK) return
        value = value + term_value
        gradient = gradient + term_gradient
    end subroutine accumulate_value_gradient

    subroutine constraint_value_or_zero(term, theta, value, status)
        type(physics_constraint_t), intent(in) :: term
        real(dp), intent(in) :: theta(:)
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status

        if (.not. term%initialized()) then
            value = 0.0_dp
            call status_set(status, FORTNUM_OK, "")
            return
        end if
        call term%value(theta, value, status)
    end subroutine constraint_value_or_zero

    subroutine term_gradient_or_zero(term, theta, gradient, status)
        type(physics_constraint_t), intent(in) :: term
        real(dp), intent(in) :: theta(:)
        real(dp), intent(out) :: gradient(:)
        type(fortnum_status_t), intent(out) :: status

        gradient = 0.0_dp
        if (.not. term%initialized()) then
            call status_set(status, FORTNUM_OK, "")
            return
        end if
        call term%gradient(theta, gradient, status)
    end subroutine term_gradient_or_zero

    subroutine term_hvp_or_zero(term, theta, theta_dot, hvp, status)
        type(physics_constraint_t), intent(in) :: term
        real(dp), intent(in) :: theta(:), theta_dot(:)
        real(dp), intent(out) :: hvp(:)
        type(fortnum_status_t), intent(out) :: status

        hvp = 0.0_dp
        if (.not. term%initialized()) then
            call status_set(status, FORTNUM_OK, "")
            return
        end if
        call term%hvp(theta, theta_dot, hvp, status)
    end subroutine term_hvp_or_zero

    subroutine accumulate_jvp(term, theta, theta_dot, value, value_dot, &
            term_value, term_dot, status)
        type(physics_constraint_t), intent(in) :: term
        real(dp), intent(in) :: theta(:), theta_dot(:)
        real(dp), intent(inout) :: value, value_dot
        real(dp), intent(out) :: term_value, term_dot
        type(fortnum_status_t), intent(out) :: status

        if (.not. term%initialized()) then
            term_value = 0.0_dp
            term_dot = 0.0_dp
            call status_set(status, FORTNUM_OK, "")
            return
        end if
        call term%jvp(theta, theta_dot, term_value, term_dot, status)
        if (status%code /= FORTNUM_OK) return
        value = value + term_value
        value_dot = value_dot + term_dot
    end subroutine accumulate_jvp

    subroutine accumulate_vjp(term, theta, value_bar, theta_bar, term_bar, status)
        type(physics_constraint_t), intent(in) :: term
        real(dp), intent(in) :: theta(:), value_bar
        real(dp), intent(inout) :: theta_bar(:)
        real(dp), intent(out) :: term_bar(:)
        type(fortnum_status_t), intent(out) :: status

        if (.not. term%initialized()) then
            term_bar = 0.0_dp
            call status_set(status, FORTNUM_OK, "")
            return
        end if
        call term%vjp(theta, value_bar, term_bar, status)
        if (status%code /= FORTNUM_OK) return
        theta_bar = theta_bar + term_bar
    end subroutine accumulate_vjp

    subroutine accumulate_hvp(term, theta, theta_dot, theta_hvp, status)
        type(physics_constraint_t), intent(in) :: term
        real(dp), intent(in) :: theta(:), theta_dot(:)
        real(dp), intent(inout) :: theta_hvp(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: term_hvp(:)

        if (.not. term%initialized()) then
            call status_set(status, FORTNUM_OK, "")
            return
        end if
        allocate(term_hvp(size(theta_hvp)))
        call term%hvp(theta, theta_dot, term_hvp, status)
        if (status%code /= FORTNUM_OK) return
        theta_hvp = theta_hvp + term_hvp
        call status_set(status, FORTNUM_OK, "")
    end subroutine accumulate_hvp

    subroutine physics_objective_context(context, theta, value, gradient, status)
        class(*), intent(inout) :: context
        real(dp), intent(in) :: theta(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status

        select type (context)
            type is (physics_objective_t)
            call context%value_gradient(theta, value, gradient, status)
        class default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "physics objective: context has unsupported type")
        end select
    end subroutine physics_objective_context

end module fortml_physics_objective
