module fortml_radam
    !! Deterministic, validated Rectified Adam (RAdam) recurrence.
    !!
    !! The state is flat so it can be shared by the MLP trainer and by a
    !! future parameter-tree adapter.  The rectification factor follows the
    !! original RAdam convention: while the variance is not trustworthy
    !! (`rho_t <= 4`) the update uses the bias-corrected first moment only;
    !! after that point it uses the rectified Adam direction.  No finite
    !! differences or hidden host fallback are used by the device seam.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, &
        FORTML_DEVICE_CUDA
    implicit none
    private

    type, public :: radam_t
        !! Flat RAdam optimizer state.  Public arrays make exact checkpoint
        !! and resume possible without compiler-dependent serialization.
        real(dp), allocatable :: first_moment(:)
        real(dp), allocatable :: second_moment(:)
        real(dp) :: learning_rate = 1.0e-3_dp
        real(dp) :: beta1 = 0.9_dp
        real(dp) :: beta2 = 0.999_dp
        real(dp) :: epsilon = 1.0e-8_dp
        integer :: step_count = 0
    contains
        procedure, public :: initialize => radam_initialize
        procedure, public :: step => radam_step
        procedure, public :: step_device => radam_step_device
        procedure, public :: device_supported => radam_device_supported
    end type radam_t

    public :: radam_initialize
    public :: radam_step

contains

    subroutine radam_initialize(self, n, status, learning_rate, beta1, beta2, epsilon)
        class(radam_t), intent(out) :: self
        integer, intent(in) :: n
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: learning_rate, beta1, beta2, epsilon

        self%learning_rate = 1.0e-3_dp
        self%beta1 = 0.9_dp
        self%beta2 = 0.999_dp
        self%epsilon = 1.0e-8_dp
        if (present(learning_rate)) self%learning_rate = learning_rate
        if (present(beta1)) self%beta1 = beta1
        if (present(beta2)) self%beta2 = beta2
        if (present(epsilon)) self%epsilon = epsilon
        if (n < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "radam: invalid parameter dimension")
            return
        end if
        if (.not. ieee_is_finite(self%learning_rate)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "radam: learning rate is not finite")
            return
        end if
        if (self%learning_rate <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "radam: learning rate must be positive")
            return
        end if
        if (.not. ieee_is_finite(self%beta1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, "radam: beta1 is not finite")
            return
        end if
        if (self%beta1 < 0.0_dp .or. self%beta1 >= 1.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, "radam: beta1 must lie in [0,1)")
            return
        end if
        if (.not. ieee_is_finite(self%beta2)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, "radam: beta2 is not finite")
            return
        end if
        if (self%beta2 < 0.0_dp .or. self%beta2 >= 1.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, "radam: beta2 must lie in [0,1)")
            return
        end if
        if (.not. ieee_is_finite(self%epsilon)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, "radam: epsilon is not finite")
            return
        end if
        if (self%epsilon <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, "radam: epsilon must be positive")
            return
        end if
        allocate(self%first_moment(n), self%second_moment(n))
        self%first_moment = 0.0_dp
        self%second_moment = 0.0_dp
        self%step_count = 0
        call status_set(status, FORTNUM_OK, "")
    end subroutine radam_initialize

    subroutine radam_step(self, x, gradient, status)
        class(radam_t), intent(inout) :: self
        real(dp), intent(inout) :: x(:)
        real(dp), intent(in) :: gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: bias1, bias2, beta2_power, rho_inf, rho_t
        real(dp) :: rectification
        real(dp), allocatable :: first_corrected(:), second_corrected(:)

        if (.not. allocated(self%first_moment)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, "radam: optimizer is not initialized")
            return
        end if
        if (.not. allocated(self%second_moment)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, "radam: second-moment state is missing")
            return
        end if
        if (size(x) /= size(self%first_moment)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "radam: parameter and state dimensions do not match")
            return
        end if
        if (size(gradient) /= size(x)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "radam: gradient and parameter dimensions do not match")
            return
        end if
        if (.not. radam_hyperparameters_valid(self)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, "radam: invalid optimizer hyperparameters")
            return
        end if
        if (self%step_count < 0) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, "radam: negative step count")
            return
        end if
        if (self%step_count == huge(self%step_count)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, "radam: step count overflow")
            return
        end if
        if (any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "radam: parameter state is not finite")
            return
        end if
        if (any(.not. ieee_is_finite(gradient))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, "radam: gradient is not finite")
            return
        end if
        if (any(.not. ieee_is_finite(self%first_moment))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, "radam: first-moment state is not finite")
            return
        end if
        if (any(.not. ieee_is_finite(self%second_moment))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, "radam: second-moment state is not finite")
            return
        end if

        self%first_moment = self%beta1*self%first_moment + &
            (1.0_dp - self%beta1)*gradient
        self%second_moment = self%beta2*self%second_moment + &
            (1.0_dp - self%beta2)*gradient**2
        if (any(.not. ieee_is_finite(self%first_moment))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, "radam: first-moment overflow")
            return
        end if
        if (any(.not. ieee_is_finite(self%second_moment))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, "radam: second-moment overflow")
            return
        end if

        self%step_count = self%step_count + 1
        bias1 = 1.0_dp - self%beta1**self%step_count
        bias2 = 1.0_dp - self%beta2**self%step_count
        if (bias1 <= 0.0_dp .or. bias2 <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, "radam: invalid bias correction")
            return
        end if
        allocate(first_corrected(size(x)), second_corrected(size(x)))
        first_corrected = self%first_moment/bias1
        second_corrected = self%second_moment/bias2
        rho_inf = 2.0_dp/(1.0_dp - self%beta2) - 1.0_dp
        beta2_power = self%beta2**self%step_count
        rho_t = rho_inf - 2.0_dp*real(self%step_count, dp)*beta2_power/bias2
        if (rho_t > 4.0_dp) then
            rectification = sqrt((rho_t - 4.0_dp)*(rho_t - 2.0_dp)*rho_inf/ &
                ((rho_inf - 4.0_dp)*(rho_inf - 2.0_dp)*rho_t))
            x = x - self%learning_rate*rectification*first_corrected/ &
                (sqrt(second_corrected) + self%epsilon)
        else
            x = x - self%learning_rate*first_corrected
        end if
        if (any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, "radam: parameter update is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine radam_step

    subroutine radam_step_device(self, device, x, gradient, status)
        class(radam_t), intent(inout) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(inout) :: x(:)
        real(dp), intent(in) :: gradient(:)
        type(fortnum_status_t), intent(out) :: status

        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%step(x, gradient, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "radam: resident CUDA state is not linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, "radam: invalid device kind")
        end select
    end subroutine radam_step_device

    logical function radam_device_supported(self, device_kind) result(supported)
        class(radam_t), intent(in) :: self
        integer, intent(in) :: device_kind

        select case (device_kind)
        case (FORTML_DEVICE_CPU)
            supported = allocated(self%first_moment) .and. allocated(self%second_moment)
        case (FORTML_DEVICE_CUDA)
            supported = .false.
        case default
            supported = .false.
        end select
    end function radam_device_supported

    logical function radam_hyperparameters_valid(self) result(valid)
        class(radam_t), intent(in) :: self

        valid = ieee_is_finite(self%learning_rate)
        if (.not. valid) return
        valid = self%learning_rate > 0.0_dp
        if (.not. valid) return
        valid = ieee_is_finite(self%beta1)
        if (.not. valid) return
        valid = self%beta1 >= 0.0_dp .and. self%beta1 < 1.0_dp
        if (.not. valid) return
        valid = ieee_is_finite(self%beta2)
        if (.not. valid) return
        valid = self%beta2 >= 0.0_dp .and. self%beta2 < 1.0_dp
        if (.not. valid) return
        valid = ieee_is_finite(self%epsilon)
        if (.not. valid) return
        valid = self%epsilon > 0.0_dp
    end function radam_hyperparameters_valid

end module fortml_radam
