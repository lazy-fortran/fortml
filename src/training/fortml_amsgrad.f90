module fortml_amsgrad
    !! Deterministic AMSGrad recurrence for flat objective parameters.
    !!
    !! AMSGrad keeps the elementwise maximum of the uncorrected second
    !! moment.  The maximum is bias-corrected at the update boundary, so the
    !! recurrence is compatible with the Adam convention used by FortOpt and
    !! the MLP trainer.  All state is public to make checkpoint/resume exact.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR
    implicit none
    private

    type, public :: amsgrad_t
        real(dp), allocatable :: first_moment(:)
        real(dp), allocatable :: second_moment(:)
        real(dp), allocatable :: max_second_moment(:)
        real(dp) :: learning_rate = 1.0e-3_dp
        real(dp) :: beta1 = 0.9_dp
        real(dp) :: beta2 = 0.999_dp
        real(dp) :: epsilon = 1.0e-8_dp
        integer :: step_count = 0
    contains
        procedure, public :: initialize => amsgrad_initialize
        procedure, public :: step => amsgrad_step
    end type amsgrad_t

    public :: amsgrad_initialize
    public :: amsgrad_step

contains

    subroutine amsgrad_initialize(self, n, status, learning_rate, beta1, beta2, epsilon)
        class(amsgrad_t), intent(out) :: self
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
        if (n < 1 .or. .not. ieee_is_finite(self%learning_rate) .or. &
                self%learning_rate <= 0.0_dp .or. .not. ieee_is_finite(self%beta1) .or. &
                self%beta1 < 0.0_dp .or. self%beta1 >= 1.0_dp .or. &
                .not. ieee_is_finite(self%beta2) .or. self%beta2 < 0.0_dp .or. &
                self%beta2 >= 1.0_dp .or. .not. ieee_is_finite(self%epsilon) .or. &
                self%epsilon <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "amsgrad: invalid dimension or hyperparameter")
            return
        end if
        allocate(self%first_moment(n), self%second_moment(n), self%max_second_moment(n))
        self%first_moment = 0.0_dp
        self%second_moment = 0.0_dp
        self%max_second_moment = 0.0_dp
        self%step_count = 0
        call status_set(status, FORTNUM_OK, "")
    end subroutine amsgrad_initialize

    subroutine amsgrad_step(self, x, gradient, status)
        class(amsgrad_t), intent(inout) :: self
        real(dp), intent(inout) :: x(:)
        real(dp), intent(in) :: gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: bias1, bias2

        if (.not. allocated(self%first_moment) .or. &
                .not. allocated(self%second_moment) .or. &
                .not. allocated(self%max_second_moment)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "amsgrad: optimizer is not initialized")
            return
        end if
        if (size(x) /= size(self%first_moment) .or. size(gradient) /= size(x)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "amsgrad: state and gradient dimensions do not match")
            return
        end if
        if (.not. ieee_is_finite(self%learning_rate) .or. self%learning_rate <= 0.0_dp .or. &
                .not. ieee_is_finite(self%beta1) .or. self%beta1 < 0.0_dp .or. &
                self%beta1 >= 1.0_dp .or. .not. ieee_is_finite(self%beta2) .or. &
                self%beta2 < 0.0_dp .or. self%beta2 >= 1.0_dp .or. &
                .not. ieee_is_finite(self%epsilon) .or. self%epsilon <= 0.0_dp .or. &
                any(.not. ieee_is_finite(x)) .or. any(.not. ieee_is_finite(gradient)) .or. &
                any(.not. ieee_is_finite(self%first_moment)) .or. &
                any(.not. ieee_is_finite(self%second_moment)) .or. &
                any(.not. ieee_is_finite(self%max_second_moment))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "amsgrad: invalid state or non-finite parameter/gradient")
            return
        end if
        if (self%step_count == huge(self%step_count)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, "amsgrad: step count overflow")
            return
        end if

        self%first_moment = self%beta1*self%first_moment + &
            (1.0_dp - self%beta1)*gradient
        self%second_moment = self%beta2*self%second_moment + &
            (1.0_dp - self%beta2)*gradient**2
        self%max_second_moment = max(self%max_second_moment, self%second_moment)
        if (any(.not. ieee_is_finite(self%first_moment)) .or. &
                any(.not. ieee_is_finite(self%second_moment)) .or. &
                any(.not. ieee_is_finite(self%max_second_moment))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, "amsgrad: moment state overflow")
            return
        end if
        self%step_count = self%step_count + 1
        bias1 = 1.0_dp - self%beta1**self%step_count
        bias2 = 1.0_dp - self%beta2**self%step_count
        x = x - self%learning_rate*(self%first_moment/bias1) / &
            (sqrt(self%max_second_moment/bias2) + self%epsilon)
        if (any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, "amsgrad: parameter update is non-finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine amsgrad_step

end module fortml_amsgrad
