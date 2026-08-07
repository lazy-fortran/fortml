module fortml_adafactor
    !! Deterministic vector Adafactor recurrence for flat objective parameters.
    !!
    !! The trainer API exposes a one-dimensional parameter vector, so this
    !! implementation stores the unfactored second moment.  It follows the
    !! Adafactor update contract (exponential squared-gradient state, update
    !! RMS clipping, and optional parameter scaling) without pretending that
    !! matrix-factorized state is available when parameter-layout metadata is
    !! absent.  All recurrence state is public so checkpoint/resume is exact.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR
    implicit none
    private

    type, public :: adafactor_t
        real(dp), allocatable :: second_moment(:)
        real(dp) :: learning_rate = 1.0e-3_dp
        real(dp) :: decay = 0.999_dp
        real(dp) :: epsilon = 1.0e-30_dp
        real(dp) :: clip_threshold = 1.0_dp
        logical :: relative_step = .false.
        logical :: scale_parameter = .false.
        integer :: step_count = 0
    contains
        procedure, public :: initialize => adafactor_initialize
        procedure, public :: step => adafactor_step
    end type adafactor_t

    public :: adafactor_initialize
    public :: adafactor_step

contains

    subroutine adafactor_initialize(self, n, status, learning_rate, decay, &
            epsilon, clip_threshold, relative_step, scale_parameter)
        class(adafactor_t), intent(out) :: self
        integer, intent(in) :: n
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: learning_rate, decay, epsilon, clip_threshold
        logical, intent(in), optional :: relative_step, scale_parameter

        self%learning_rate = 1.0e-3_dp
        self%decay = 0.999_dp
        self%epsilon = 1.0e-30_dp
        self%clip_threshold = 1.0_dp
        self%relative_step = .false.
        self%scale_parameter = .false.
        if (present(learning_rate)) self%learning_rate = learning_rate
        if (present(decay)) self%decay = decay
        if (present(epsilon)) self%epsilon = epsilon
        if (present(clip_threshold)) self%clip_threshold = clip_threshold
        if (present(relative_step)) self%relative_step = relative_step
        if (present(scale_parameter)) self%scale_parameter = scale_parameter
        if (n < 1 .or. .not. ieee_is_finite(self%learning_rate) .or. &
            self%learning_rate <= 0.0_dp .or. .not. ieee_is_finite(self%decay) .or. &
            self%decay < 0.0_dp .or. self%decay >= 1.0_dp .or. &
            .not. ieee_is_finite(self%epsilon) .or. self%epsilon <= 0.0_dp .or. &
            .not. ieee_is_finite(self%clip_threshold) .or. self%clip_threshold <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "adafactor: invalid dimension or hyperparameter")
            return
        end if
        allocate(self%second_moment(n))
        self%second_moment = 0.0_dp
        self%step_count = 0
        call status_set(status, FORTNUM_OK, "")
    end subroutine adafactor_initialize

    subroutine adafactor_step(self, x, gradient, status)
        class(adafactor_t), intent(inout) :: self
        real(dp), intent(inout) :: x(:)
        real(dp), intent(in) :: gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: update_rms, clip_scale, rate, parameter_rms

        if (.not. allocated(self%second_moment)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "adafactor: optimizer is not initialized")
            return
        end if
        if (size(x) /= size(self%second_moment) .or. size(gradient) /= size(x)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "adafactor: state and gradient dimensions do not match")
            return
        end if
        if (.not. ieee_is_finite(self%learning_rate) .or. self%learning_rate <= 0.0_dp .or. &
            .not. ieee_is_finite(self%decay) .or. self%decay < 0.0_dp .or. &
            self%decay >= 1.0_dp .or. .not. ieee_is_finite(self%epsilon) .or. &
            self%epsilon <= 0.0_dp .or. .not. ieee_is_finite(self%clip_threshold) .or. &
            self%clip_threshold <= 0.0_dp .or. any(.not. ieee_is_finite(x)) .or. &
            any(.not. ieee_is_finite(gradient)) .or. any(.not. ieee_is_finite(self%second_moment))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "adafactor: invalid state or non-finite parameter/gradient")
            return
        end if
        if (self%step_count == huge(self%step_count)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, "adafactor: step count overflow")
            return
        end if

        self%second_moment = self%decay*self%second_moment + &
            (1.0_dp - self%decay)*gradient**2
        if (any(.not. ieee_is_finite(self%second_moment))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "adafactor: second-moment state overflow")
            return
        end if
        update_rms = sqrt(sum(self%second_moment)/real(size(x), dp))
        clip_scale = max(1.0_dp, update_rms/self%clip_threshold)
        rate = self%learning_rate
        if (self%relative_step) rate = min(rate, 1.0_dp/sqrt(real(self%step_count + 1, dp)))
        if (self%scale_parameter) then
            parameter_rms = sqrt(sum(x**2)/real(size(x), dp))
            rate = rate*max(parameter_rms, self%epsilon)
        end if
        x = x - rate*gradient/clip_scale/(sqrt(self%second_moment) + self%epsilon)
        if (any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "adafactor: parameter update is non-finite")
            return
        end if
        self%step_count = self%step_count + 1
        call status_set(status, FORTNUM_OK, "")
    end subroutine adafactor_step

end module fortml_adafactor
