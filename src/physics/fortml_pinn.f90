module fortml_pinn
    !! A bounded PINN training adapter over ``physics_objective_t``.
    !!
    !! The adapter deliberately does not prescribe a neural-network or GP
    !! representation.  A caller supplies a composed physics objective whose
    !! callbacks own the model, coordinates, collocation points, and units.
    !! This layer provides the stable training-facing value/JVP/VJP/HVP API,
    !! named residual diagnostics, and a FortOpt L-BFGS-B entry point.  It is
    !! CPU-only until a complete resident residual and derivative graph is
    !! linked; requesting CUDA is a typed refusal and never a host fallback.
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    use fortml_physics_objective, only: physics_objective_t
    use fortopt_objective, only: objective_t
    use fortopt_lbfgsb, only: lbfgsb_t, lbfgsb_options_t, lbfgsb_result_t
    implicit none
    private

    type, public :: pinn_training_adapter_t
        !! Training facade for a composed physics-informed objective.
        private
        type(physics_objective_t) :: objective
        integer :: device_kind = FORTML_DEVICE_CPU
        logical :: ready = .false.
    contains
        procedure, public :: initialize => pinn_initialize
        procedure, public :: initialized => pinn_initialized
        procedure, public :: parameter_count => pinn_parameter_count
        procedure, public :: device_supported => pinn_device_supported
        procedure, public :: select_device => pinn_select_device
        procedure, public :: term_values => pinn_term_values
        procedure, public :: term_gradients => pinn_term_gradients
        procedure, public :: term_hvps => pinn_term_hvps
        procedure, public :: value => pinn_value
        procedure, public :: gradient => pinn_gradient
        procedure, public :: value_gradient => pinn_value_gradient
        procedure, public :: jvp => pinn_jvp
        procedure, public :: vjp => pinn_vjp
        procedure, public :: hvp => pinn_hvp
        procedure, public :: as_objective => pinn_as_objective
        procedure, public :: fit_lbfgsb => pinn_fit_lbfgsb
    end type pinn_training_adapter_t

contains

    subroutine pinn_initialize(self, objective, status, device_kind)
        class(pinn_training_adapter_t), intent(out) :: self
        type(physics_objective_t), intent(in) :: objective
        type(fortnum_status_t), intent(out) :: status
        integer, intent(in), optional :: device_kind
        integer :: requested

        ! `self` is intent(out), so every component already holds its default
        ! initialization on entry; assigning an empty structure constructor
        ! here restated that and was redundant.
        !
        ! It also crashed nvfortran 26.5 outright -- fort1 terminated by signal
        ! 11, with no line number and at every optimization level -- which
        ! blocked the OpenACC build of the whole stack. Found by bisecting the
        ! module procedure by procedure and then statement by statement.
        self%device_kind = FORTML_DEVICE_CPU
        self%ready = .false.
        requested = FORTML_DEVICE_CPU
        if (present(device_kind)) requested = device_kind
        if (requested == FORTML_DEVICE_CUDA) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "PINN adapter: resident CUDA residual graph is not implemented")
            return
        end if
        if (requested /= FORTML_DEVICE_CPU) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "PINN adapter: device kind is invalid")
            return
        end if
        if (.not. objective%initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "PINN adapter: physics objective is not initialized")
            return
        end if
        self%objective = objective
        self%device_kind = requested
        self%ready = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine pinn_initialize

    logical function pinn_initialized(self) result(yes)
        class(pinn_training_adapter_t), intent(in) :: self

        yes = self%ready .and. self%objective%initialized()
    end function pinn_initialized

    integer function pinn_parameter_count(self) result(count)
        class(pinn_training_adapter_t), intent(in) :: self

        count = 0
        if (self%initialized()) count = self%objective%parameter_count()
    end function pinn_parameter_count

    logical function pinn_device_supported(self, device_kind) result(yes)
        class(pinn_training_adapter_t), intent(in) :: self
        integer, intent(in), optional :: device_kind
        integer :: requested

        requested = self%device_kind
        if (present(device_kind)) requested = device_kind
        yes = self%initialized() .and. requested == FORTML_DEVICE_CPU
    end function pinn_device_supported

    subroutine pinn_select_device(self, device_kind, status)
        class(pinn_training_adapter_t), intent(inout) :: self
        integer, intent(in) :: device_kind
        type(fortnum_status_t), intent(out) :: status

        if (.not. self%initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "PINN adapter: object is not initialized")
            return
        end if
        if (device_kind == FORTML_DEVICE_CUDA) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "PINN adapter: resident CUDA residual graph is not implemented")
            return
        end if
        if (device_kind /= FORTML_DEVICE_CPU) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "PINN adapter: device kind is invalid")
            return
        end if
        self%device_kind = device_kind
        call status_set(status, FORTNUM_OK, "")
    end subroutine pinn_select_device

    subroutine pinn_term_values(self, theta, values, status)
        class(pinn_training_adapter_t), intent(in) :: self
        real(dp), intent(in) :: theta(:)
        real(dp), intent(out) :: values(:)
        type(fortnum_status_t), intent(out) :: status

        if (.not. ready_or_error(self, status)) return
        call self%objective%term_values(theta, values, status)
    end subroutine pinn_term_values

    subroutine pinn_term_gradients(self, theta, gradients, status)
        class(pinn_training_adapter_t), intent(in) :: self
        real(dp), intent(in) :: theta(:)
        real(dp), intent(out) :: gradients(:,:)
        type(fortnum_status_t), intent(out) :: status

        if (.not. ready_or_error(self, status)) return
        call self%objective%term_gradients(theta, gradients, status)
    end subroutine pinn_term_gradients

    subroutine pinn_term_hvps(self, theta, theta_dot, hvps, status)
        class(pinn_training_adapter_t), intent(in) :: self
        real(dp), intent(in) :: theta(:), theta_dot(:)
        real(dp), intent(out) :: hvps(:,:)
        type(fortnum_status_t), intent(out) :: status

        if (.not. ready_or_error(self, status)) return
        call self%objective%term_hvps(theta, theta_dot, hvps, status)
    end subroutine pinn_term_hvps

    subroutine pinn_value(self, theta, value, status)
        class(pinn_training_adapter_t), intent(in) :: self
        real(dp), intent(in) :: theta(:)
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status

        if (.not. ready_or_error(self, status)) return
        call self%objective%value(theta, value, status)
    end subroutine pinn_value

    subroutine pinn_gradient(self, theta, gradient, status)
        class(pinn_training_adapter_t), intent(in) :: self
        real(dp), intent(in) :: theta(:)
        real(dp), intent(out) :: gradient(:)
        type(fortnum_status_t), intent(out) :: status

        if (.not. ready_or_error(self, status)) return
        call self%objective%gradient(theta, gradient, status)
    end subroutine pinn_gradient

    subroutine pinn_value_gradient(self, theta, value, gradient, status)
        class(pinn_training_adapter_t), intent(in) :: self
        real(dp), intent(in) :: theta(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status

        if (.not. ready_or_error(self, status)) return
        call self%objective%value_gradient(theta, value, gradient, status)
    end subroutine pinn_value_gradient

    subroutine pinn_jvp(self, theta, theta_dot, value, value_dot, status)
        class(pinn_training_adapter_t), intent(in) :: self
        real(dp), intent(in) :: theta(:), theta_dot(:)
        real(dp), intent(out) :: value, value_dot
        type(fortnum_status_t), intent(out) :: status

        if (.not. ready_or_error(self, status)) return
        call self%objective%jvp(theta, theta_dot, value, value_dot, status)
    end subroutine pinn_jvp

    subroutine pinn_vjp(self, theta, value_bar, theta_bar, status)
        class(pinn_training_adapter_t), intent(in) :: self
        real(dp), intent(in) :: theta(:), value_bar
        real(dp), intent(out) :: theta_bar(:)
        type(fortnum_status_t), intent(out) :: status

        if (.not. ready_or_error(self, status)) return
        call self%objective%vjp(theta, value_bar, theta_bar, status)
    end subroutine pinn_vjp

    subroutine pinn_hvp(self, theta, theta_dot, theta_hvp, status)
        class(pinn_training_adapter_t), intent(in) :: self
        real(dp), intent(in) :: theta(:), theta_dot(:)
        real(dp), intent(out) :: theta_hvp(:)
        type(fortnum_status_t), intent(out) :: status

        if (.not. ready_or_error(self, status)) return
        call self%objective%hvp(theta, theta_dot, theta_hvp, status)
    end subroutine pinn_hvp

    subroutine pinn_as_objective(self, objective, status)
        class(pinn_training_adapter_t), target, intent(inout) :: self
        type(objective_t), intent(out) :: objective
        type(fortnum_status_t), intent(out) :: status

        if (.not. ready_or_error(self, status)) return
        call self%objective%as_objective(objective, status)
    end subroutine pinn_as_objective

    subroutine pinn_fit_lbfgsb(self, parameters, lower, upper, options, result, status)
        class(pinn_training_adapter_t), target, intent(inout) :: self
        real(dp), intent(inout) :: parameters(:)
        real(dp), intent(in) :: lower(:), upper(:)
        type(lbfgsb_options_t), intent(in) :: options
        type(lbfgsb_result_t), intent(out) :: result
        type(fortnum_status_t), intent(out) :: status
        type(objective_t) :: objective
        type(lbfgsb_t) :: optimizer

        if (.not. ready_or_error(self, status)) return
        call self%objective%as_objective(objective, status)
        if (status%code /= FORTNUM_OK) return
        call optimizer%minimize(objective, parameters, lower, upper, options, &
            result, status)
    end subroutine pinn_fit_lbfgsb

    logical function ready_or_error(self, status) result(ready)
        class(pinn_training_adapter_t), intent(in) :: self
        type(fortnum_status_t), intent(out) :: status

        ready = self%initialized()
        if (.not. ready) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "PINN adapter: object is not initialized")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end function ready_or_error

end module fortml_pinn
