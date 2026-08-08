module fortml_lion
    !! Deterministic Lion recurrence for flat objective parameters.
    !!
    !! Lion keeps one exponential moving average and uses the sign of the
    !! beta1 interpolation for the decoupled update.  The state is explicit
    !! so generic trainers can checkpoint and resume it exactly.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR
    implicit none
    private

    type, public :: lion_t
        real(dp), allocatable :: momentum(:)
        real(dp) :: learning_rate = 1.0e-4_dp
        real(dp) :: beta1 = 0.9_dp
        real(dp) :: beta2 = 0.99_dp
        real(dp) :: weight_decay = 0.0_dp
        integer :: step_count = 0
    contains
        procedure, public :: initialize => lion_initialize
        procedure, public :: step => lion_step
    end type lion_t

    public :: lion_initialize
    public :: lion_step

contains

    subroutine lion_initialize(self, n, status, learning_rate, beta1, beta2, &
            weight_decay)
        class(lion_t), intent(out) :: self
        integer, intent(in) :: n
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: learning_rate, beta1, beta2, weight_decay

        self%learning_rate = 1.0e-4_dp
        self%beta1 = 0.9_dp
        self%beta2 = 0.99_dp
        self%weight_decay = 0.0_dp
        if (present(learning_rate)) self%learning_rate = learning_rate
        if (present(beta1)) self%beta1 = beta1
        if (present(beta2)) self%beta2 = beta2
        if (present(weight_decay)) self%weight_decay = weight_decay
        if (.not. valid_hyperparameters(self) .or. n < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "lion: invalid dimension or hyperparameter")
            return
        end if
        allocate(self%momentum(n))
        self%momentum = 0.0_dp
        self%step_count = 0
        call status_set(status, FORTNUM_OK, "")
    end subroutine lion_initialize

    subroutine lion_step(self, x, gradient, status)
        class(lion_t), intent(inout) :: self
        real(dp), intent(inout) :: x(:)
        real(dp), intent(in) :: gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: interpolated(:)
        integer :: i

        if (.not. allocated(self%momentum) .or. size(x) /= size(self%momentum) .or. &
            size(gradient) /= size(x) .or. .not. valid_hyperparameters(self) .or. &
            any(.not. ieee_is_finite(x)) .or. any(.not. ieee_is_finite(gradient)) .or. &
            any(.not. ieee_is_finite(self%momentum)) .or. self%step_count < 0) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "lion: state or gradient shape is invalid")
            return
        end if
        if (self%step_count == huge(self%step_count)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, "lion: step count overflow")
            return
        end if
        allocate(interpolated(size(x)))
        interpolated = self%beta1*self%momentum + (1.0_dp-self%beta1)*gradient
        do i = 1, size(x)
            if (interpolated(i) > 0.0_dp) then
                x(i) = x(i) - self%learning_rate*(1.0_dp + self%weight_decay*x(i))
            else if (interpolated(i) < 0.0_dp) then
                x(i) = x(i) - self%learning_rate*(-1.0_dp + self%weight_decay*x(i))
            else
                x(i) = x(i) - self%learning_rate*self%weight_decay*x(i)
            end if
        end do
        self%momentum = self%beta2*self%momentum + (1.0_dp-self%beta2)*gradient
        if (any(.not. ieee_is_finite(x)) .or. any(.not. ieee_is_finite(self%momentum))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "lion: update produced a non-finite state")
            return
        end if
        self%step_count = self%step_count + 1
        call status_set(status, FORTNUM_OK, "")
    end subroutine lion_step

    logical function valid_hyperparameters(self)
        class(lion_t), intent(in) :: self
        valid_hyperparameters = ieee_is_finite(self%learning_rate) .and. &
            self%learning_rate > 0.0_dp .and. ieee_is_finite(self%beta1) .and. &
            self%beta1 >= 0.0_dp .and. self%beta1 < 1.0_dp .and. &
            ieee_is_finite(self%beta2) .and. self%beta2 >= 0.0_dp .and. &
            self%beta2 < 1.0_dp .and. ieee_is_finite(self%weight_decay) .and. &
            self%weight_decay >= 0.0_dp
    end function valid_hyperparameters

end module fortml_lion
