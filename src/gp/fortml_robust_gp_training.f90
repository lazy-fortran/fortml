module fortml_robust_gp_training
    !! FortOpt and derivative products for the Poisson robust-GP path.
    !!
    !! The packed variable is the latent log-rate mode.  Keeping this as an
    !! explicit objective is useful for nested hyperparameter searches: the
    !! covariance factorization and Poisson likelihood are still owned by the
    !! fitted GP, while L-BFGS-B sees exact value/gradient products and the
    !! objective exposes JVP, VJP, and HVP products for outer differentiation.
    !! CUDA requests are typed refusals until a resident Laplace solve exists.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_CONVERGENCE_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    use fortml_robust_gp, only: robust_gp_t, FORTML_LIKELIHOOD_POISSON
    use fortopt_objective, only: objective_t
    use fortopt_lbfgsb, only: lbfgsb_t, lbfgsb_options_t, lbfgsb_result_t
    implicit none
    private

    type, public :: robust_gp_poisson_objective_t
        !! Fixed-covariance Poisson posterior objective over latent log rates.
        private
        type(robust_gp_t), pointer :: model => null()
        integer :: device_kind = FORTML_DEVICE_CPU
    contains
        procedure, public :: initialize => robust_poisson_objective_initialize
        procedure, public :: initialized => robust_poisson_objective_initialized
        procedure, public :: device_supported => robust_poisson_objective_device_supported
        procedure, public :: parameter_count => robust_poisson_objective_parameter_count
        procedure, public :: parameters => robust_poisson_objective_parameters
        procedure, public :: value_gradient => robust_poisson_objective_value_gradient
        procedure, public :: jvp => robust_poisson_objective_jvp
        procedure, public :: vjp => robust_poisson_objective_vjp
        procedure, public :: hvp => robust_poisson_objective_hvp
        procedure, public :: fortopt => robust_poisson_objective_fortopt
    end type robust_gp_poisson_objective_t

    type, public :: robust_gp_poisson_lbfgsb_options_t
        !! Bounds and convergence controls for latent-mode L-BFGS-B.
        integer :: memory = 10
        integer :: max_iterations = 100
        integer :: max_line_search = 40
        real(dp) :: gradient_tolerance = 1.0e-8_dp
        real(dp) :: step_tolerance = 1.0e-12_dp
        real(dp) :: objective_tolerance = 1.0e-12_dp
        real(dp) :: lower_bound = -20.0_dp
        real(dp) :: upper_bound = 20.0_dp
        integer :: device_kind = FORTML_DEVICE_CPU
    end type robust_gp_poisson_lbfgsb_options_t

    type, public :: robust_gp_poisson_lbfgsb_result_t
        !! Diagnostics returned by `robust_gp_poisson_optimize`.
        logical :: converged = .false.
        integer :: iterations = 0
        integer :: line_search_evaluations = 0
        real(dp) :: negative_log_posterior = huge(1.0_dp)
        real(dp) :: gradient_norm = huge(1.0_dp)
    end type robust_gp_poisson_lbfgsb_result_t

    public :: robust_gp_poisson_optimize

contains

    subroutine robust_poisson_objective_initialize(self, model, status, device_kind)
        class(robust_gp_poisson_objective_t), intent(out) :: self
        type(robust_gp_t), target, intent(inout) :: model
        type(fortnum_status_t), intent(out) :: status
        integer, intent(in), optional :: device_kind

        self%device_kind = FORTML_DEVICE_CPU
        if (present(device_kind)) self%device_kind = device_kind
        if (self%device_kind == FORTML_DEVICE_CUDA) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "robust Poisson objective: resident CUDA Laplace graph is not linked")
            return
        end if
        if (self%device_kind /= FORTML_DEVICE_CPU) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "robust Poisson objective: device kind is invalid")
            return
        end if
        if (.not. model%fitted .or. model%likelihood /= FORTML_LIKELIHOOD_POISSON) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "robust Poisson objective: fitted Poisson GP is required")
            return
        end if
        self%model => model
        call status_set(status, FORTNUM_OK, "")
    end subroutine robust_poisson_objective_initialize

    logical function robust_poisson_objective_initialized(self) result(yes)
        class(robust_gp_poisson_objective_t), intent(in) :: self

        yes = associated(self%model)
        if (.not. yes) return
        yes = self%model%fitted
        if (.not. yes) return
        yes = self%model%likelihood == FORTML_LIKELIHOOD_POISSON
    end function robust_poisson_objective_initialized

    logical function robust_poisson_objective_device_supported(self, device_kind) result(yes)
        class(robust_gp_poisson_objective_t), intent(in) :: self
        integer, intent(in), optional :: device_kind
        integer :: requested

        requested = self%device_kind
        if (present(device_kind)) requested = device_kind
        yes = requested == FORTML_DEVICE_CPU .and. self%initialized()
    end function robust_poisson_objective_device_supported

    integer function robust_poisson_objective_parameter_count(self) result(count)
        class(robust_gp_poisson_objective_t), intent(in) :: self

        count = 0
        if (self%initialized()) count = self%model%latent_parameter_count()
    end function robust_poisson_objective_parameter_count

    function robust_poisson_objective_parameters(self) result(parameters)
        class(robust_gp_poisson_objective_t), intent(in) :: self
        real(dp), allocatable :: parameters(:)

        if (.not. self%initialized()) then
            allocate(parameters(0))
            return
        end if
        parameters = self%model%latent_parameters()
    end function robust_poisson_objective_parameters

    subroutine robust_poisson_objective_value_gradient(self, parameters, value, gradient, &
            status)
        class(robust_gp_poisson_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: posterior

        value = huge(1.0_dp)
        gradient = 0.0_dp
        if (.not. self%initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "robust Poisson objective: adapter is not initialized")
            return
        end if
        if (size(parameters) /= self%parameter_count() .or. &
            size(gradient) /= size(parameters) .or. &
            any(.not. ieee_is_finite(parameters))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "robust Poisson objective: parameter or gradient shape is invalid")
            return
        end if
        call self%model%set_latent_parameters(parameters, status)
        if (status%code /= FORTNUM_OK) return
        call self%model%log_posterior(posterior, status)
        if (status%code /= FORTNUM_OK) return
        call self%model%log_posterior_gradient(gradient, status)
        if (status%code /= FORTNUM_OK) return
        value = -posterior
        gradient = -gradient
        if (.not. ieee_is_finite(value) .or. any(.not. ieee_is_finite(gradient))) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "robust Poisson objective: value or gradient is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine robust_poisson_objective_value_gradient

    subroutine robust_poisson_objective_jvp(self, parameters, direction, value, tangent, &
            status)
        class(robust_gp_poisson_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:), direction(:)
        real(dp), intent(out) :: value, tangent
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: gradient(:)

        value = huge(1.0_dp)
        tangent = 0.0_dp
        if (size(direction) /= size(parameters) .or. &
                any(.not. ieee_is_finite(direction))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "robust Poisson objective JVP: direction shape is invalid")
            return
        end if
        allocate(gradient(size(parameters)))
        call self%value_gradient(parameters, value, gradient, status)
        if (status%code /= FORTNUM_OK) return
        tangent = dot_product(gradient, direction)
        call status_set(status, FORTNUM_OK, "")
    end subroutine robust_poisson_objective_jvp

    subroutine robust_poisson_objective_vjp(self, parameters, value_bar, gradient, status)
        class(robust_gp_poisson_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:), value_bar
        real(dp), intent(out) :: gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: value

        gradient = 0.0_dp
        if (.not. ieee_is_finite(value_bar)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "robust Poisson objective VJP: cotangent is invalid")
            return
        end if
        call self%value_gradient(parameters, value, gradient, status)
        if (status%code /= FORTNUM_OK) return
        gradient = value_bar*gradient
        call status_set(status, FORTNUM_OK, "")
    end subroutine robust_poisson_objective_vjp

    subroutine robust_poisson_objective_hvp(self, parameters, direction, product, status)
        class(robust_gp_poisson_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:), direction(:)
        real(dp), intent(out) :: product(:)
        type(fortnum_status_t), intent(out) :: status

        product = 0.0_dp
        if (.not. self%initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "robust Poisson objective HVP: adapter is not initialized")
            return
        end if
        if (size(parameters) /= self%parameter_count() .or. &
            size(direction) /= size(parameters) .or. size(product) /= size(parameters) .or. &
            any(.not. ieee_is_finite(parameters)) .or. any(.not. ieee_is_finite(direction))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "robust Poisson objective HVP: direction or output shape is invalid")
            return
        end if
        call self%model%set_latent_parameters(parameters, status)
        if (status%code /= FORTNUM_OK) return
        call self%model%log_posterior_hvp(direction, product, status)
        if (status%code /= FORTNUM_OK) return
        product = -product
        call status_set(status, FORTNUM_OK, "")
    end subroutine robust_poisson_objective_hvp

    subroutine robust_poisson_objective_fortopt(self, objective, status)
        class(robust_gp_poisson_objective_t), target, intent(inout) :: self
        type(objective_t), intent(out) :: objective

        type(fortnum_status_t), intent(out) :: status

        if (.not. self%initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "robust Poisson objective: adapter is not initialized")
            return
        end if
        call objective%initialize_context(self%parameter_count(), self, &
            robust_poisson_objective_context, status)
    end subroutine robust_poisson_objective_fortopt

    subroutine robust_poisson_objective_context(context_any, parameters, value, gradient, &
            status)
        class(*), intent(inout) :: context_any
        real(dp), intent(in) :: parameters(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status

        select type (adapter => context_any)
            type is (robust_gp_poisson_objective_t)
            call adapter%value_gradient(parameters, value, gradient, status)
        class default
            value = huge(1.0_dp)
            gradient = 0.0_dp
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "robust Poisson objective: context has the wrong type")
        end select
    end subroutine robust_poisson_objective_context

    subroutine robust_gp_poisson_optimize(model, options, result, status, device)
        !! Minimize negative fixed-covariance Poisson log posterior with FortOpt.
        type(robust_gp_t), target, intent(inout) :: model
        type(robust_gp_poisson_lbfgsb_options_t), intent(in) :: options
        type(robust_gp_poisson_lbfgsb_result_t), intent(out) :: result
        type(fortnum_status_t), intent(out) :: status
        type(fortml_device_t), intent(in), optional :: device
        type(robust_gp_poisson_objective_t), target :: adapter
        type(objective_t) :: objective
        type(lbfgsb_t) :: optimizer
        type(lbfgsb_options_t) :: optimizer_options
        type(lbfgsb_result_t) :: optimizer_result
        real(dp), allocatable :: parameters(:), initial_parameters(:), lower(:), upper(:), gradient(:)
        type(fortnum_status_t) :: restore_status
        integer :: n
        type(robust_gp_poisson_lbfgsb_result_t) :: result_default

        result = result_default
        if (present(device)) then
            if (.not. device%selected .or. .not. device%available) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "robust Poisson training: selected device is unavailable")
                return
            end if
            select case (device%kind)
            case (FORTML_DEVICE_CPU)
                continue
            case (FORTML_DEVICE_CUDA)
                call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                    "robust Poisson training: resident CUDA optimizer is not linked")
                return
            case default
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "robust Poisson training: device kind is invalid")
                return
            end select
        end if
        if (.not. valid_options(options)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "robust Poisson training: options are invalid")
            return
        end if
        call adapter%initialize(model, status, options%device_kind)
        if (status%code /= FORTNUM_OK) return
        n = adapter%parameter_count()
        parameters = adapter%parameters()
        initial_parameters = parameters
        allocate(lower(n), upper(n), gradient(n))
        lower = options%lower_bound
        upper = options%upper_bound
        call adapter%fortopt(objective, status)
        if (status%code /= FORTNUM_OK) return
        call copy_options(options, optimizer_options)
        call optimizer%minimize(objective, parameters, lower, upper, optimizer_options, &
            optimizer_result, status)
        if (status%code /= FORTNUM_OK .and. status%code /= FORTNUM_CONVERGENCE_ERROR) then
            call model%set_latent_parameters(initial_parameters, restore_status)
            return
        end if
        result%iterations = optimizer_result%state%iteration
        result%line_search_evaluations = optimizer_result%line_search_evaluations
        result%converged = optimizer_result%state%converged
        call adapter%value_gradient(parameters, result%negative_log_posterior, gradient, status)
        if (status%code /= FORTNUM_OK) then
            call model%set_latent_parameters(initial_parameters, restore_status)
            return
        end if
        result%gradient_norm = sqrt(sum(gradient*gradient))
        if (.not. ieee_is_finite(result%negative_log_posterior) .or. &
            .not. ieee_is_finite(result%gradient_norm)) then
            call model%set_latent_parameters(initial_parameters, restore_status)
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "robust Poisson training: result is not finite")
            return
        end if
        if (.not. result%converged) then
            call model%set_latent_parameters(initial_parameters, restore_status)
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "robust Poisson training: iteration limit reached")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine robust_gp_poisson_optimize

    subroutine copy_options(options, target)
        type(robust_gp_poisson_lbfgsb_options_t), intent(in) :: options
        type(lbfgsb_options_t), intent(out) :: target

        target%memory = options%memory
        target%max_iterations = options%max_iterations
        target%max_line_search = options%max_line_search
        target%gradient_tolerance = options%gradient_tolerance
        target%step_tolerance = options%step_tolerance
        target%objective_tolerance = options%objective_tolerance
    end subroutine copy_options

    logical function valid_options(options) result(valid)
        type(robust_gp_poisson_lbfgsb_options_t), intent(in) :: options

        valid = options%memory >= 1 .and. options%max_iterations >= 1 .and. &
            options%max_line_search >= 1 .and. &
            ieee_is_finite(options%gradient_tolerance) .and. &
            ieee_is_finite(options%step_tolerance) .and. &
            ieee_is_finite(options%objective_tolerance) .and. &
            ieee_is_finite(options%lower_bound) .and. ieee_is_finite(options%upper_bound) .and. &
            options%gradient_tolerance >= 0.0_dp .and. options%step_tolerance >= 0.0_dp .and. &
            options%objective_tolerance >= 0.0_dp .and. options%lower_bound <= options%upper_bound
    end function valid_options

end module fortml_robust_gp_training
