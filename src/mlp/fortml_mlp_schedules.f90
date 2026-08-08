module fortml_mlp_schedules
    !! Typed learning-rate schedules with analytic parameter products.
    !!
    !! The schedule is deliberately independent of the trainer's callback
    !! procedure pointer: a caller can evaluate it from a callback, a
    !! hypergradient recurrence, or a device lowering without hidden state.
    !! `rate_with_derivatives` returns derivatives with respect to the base
    !! learning rate, the minimum-rate fraction, and the decay factor.  The
    !! latter two derivatives are zero for schedule families that do not use
    !! those fields.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR
    use fortml_device, only: FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    implicit none
    private

    integer, parameter, public :: MLP_SCHEDULE_CONSTANT = 1
    integer, parameter, public :: MLP_SCHEDULE_LINEAR_WARMUP = 2
    integer, parameter, public :: MLP_SCHEDULE_COSINE_DECAY = 3
    integer, parameter, public :: MLP_SCHEDULE_WARMUP_COSINE = 4
    integer, parameter, public :: MLP_SCHEDULE_EXPONENTIAL_DECAY = 5
    integer, parameter, public :: MLP_SCHEDULE_ONE_CYCLE = 6

    type, public :: mlp_learning_rate_schedule_t
        !! Stateless schedule parameters.
        integer :: kind = MLP_SCHEDULE_CONSTANT
        integer :: warmup_updates = 0
        integer :: total_updates = 0
        real(dp) :: min_rate_fraction = 0.0_dp
        real(dp) :: decay_factor = 1.0_dp
        real(dp) :: peak_rate_fraction = 1.0_dp
        real(dp) :: final_rate_fraction = 0.1_dp
    contains
        procedure, public :: valid => schedule_valid
        procedure, public :: device_supported => schedule_device_supported
        procedure, public :: rate_with_derivatives => schedule_rate_with_derivatives
        procedure, public :: rate_with_full_derivatives => schedule_rate_with_full_derivatives
        procedure, public :: rate => schedule_rate
    end type mlp_learning_rate_schedule_t

    public :: make_mlp_schedule_constant
    public :: make_mlp_schedule_linear_warmup
    public :: make_mlp_schedule_cosine_decay
    public :: make_mlp_schedule_warmup_cosine
    public :: make_mlp_schedule_exponential_decay
    public :: make_mlp_schedule_one_cycle

contains

    function make_mlp_schedule_constant() result(schedule)
        type(mlp_learning_rate_schedule_t) :: schedule
        !! Default-initialized instances, standing in for empty
        !! structure constructors: nvfortran rejects `T()` outright,
        !! and a declared local carries the same default init.
        type(mlp_learning_rate_schedule_t) :: mlp_learning_rate_schedule_t_default

        schedule = mlp_learning_rate_schedule_t_default
    end function make_mlp_schedule_constant

    function make_mlp_schedule_linear_warmup(warmup_updates) result(schedule)
        integer, intent(in) :: warmup_updates
        type(mlp_learning_rate_schedule_t) :: schedule
        !! Default-initialized instances, standing in for empty
        !! structure constructors: nvfortran rejects `T()` outright,
        !! and a declared local carries the same default init.
        type(mlp_learning_rate_schedule_t) :: mlp_learning_rate_schedule_t_default

        schedule = mlp_learning_rate_schedule_t_default
        schedule%kind = MLP_SCHEDULE_LINEAR_WARMUP
        schedule%warmup_updates = warmup_updates
    end function make_mlp_schedule_linear_warmup

    function make_mlp_schedule_cosine_decay(total_updates, min_rate_fraction) result(schedule)
        integer, intent(in) :: total_updates
        real(dp), intent(in) :: min_rate_fraction
        type(mlp_learning_rate_schedule_t) :: schedule
        !! Default-initialized instances, standing in for empty
        !! structure constructors: nvfortran rejects `T()` outright,
        !! and a declared local carries the same default init.
        type(mlp_learning_rate_schedule_t) :: mlp_learning_rate_schedule_t_default

        schedule = mlp_learning_rate_schedule_t_default
        schedule%kind = MLP_SCHEDULE_COSINE_DECAY
        schedule%total_updates = total_updates
        schedule%min_rate_fraction = min_rate_fraction
    end function make_mlp_schedule_cosine_decay

    function make_mlp_schedule_warmup_cosine(warmup_updates, total_updates, &
            min_rate_fraction) result(schedule)
        integer, intent(in) :: warmup_updates, total_updates
        real(dp), intent(in) :: min_rate_fraction
        type(mlp_learning_rate_schedule_t) :: schedule
        !! Default-initialized instances, standing in for empty
        !! structure constructors: nvfortran rejects `T()` outright,
        !! and a declared local carries the same default init.
        type(mlp_learning_rate_schedule_t) :: mlp_learning_rate_schedule_t_default

        schedule = mlp_learning_rate_schedule_t_default
        schedule%kind = MLP_SCHEDULE_WARMUP_COSINE
        schedule%warmup_updates = warmup_updates
        schedule%total_updates = total_updates
        schedule%min_rate_fraction = min_rate_fraction
    end function make_mlp_schedule_warmup_cosine

    function make_mlp_schedule_exponential_decay(warmup_updates, decay_factor) result(schedule)
        integer, intent(in) :: warmup_updates
        real(dp), intent(in) :: decay_factor
        type(mlp_learning_rate_schedule_t) :: schedule
        !! Default-initialized instances, standing in for empty
        !! structure constructors: nvfortran rejects `T()` outright,
        !! and a declared local carries the same default init.
        type(mlp_learning_rate_schedule_t) :: mlp_learning_rate_schedule_t_default

        schedule = mlp_learning_rate_schedule_t_default
        schedule%kind = MLP_SCHEDULE_EXPONENTIAL_DECAY
        schedule%warmup_updates = warmup_updates
        schedule%decay_factor = decay_factor
    end function make_mlp_schedule_exponential_decay

    function make_mlp_schedule_one_cycle(warmup_updates, total_updates, &
            peak_rate_fraction, final_rate_fraction) result(schedule)
        !! Construct a stateless triangular/cosine one-cycle schedule.
        !!
        !! `base_rate` passed to `rate` is the initial rate.  The warm-up
        !! rises linearly to `base_rate*peak_rate_fraction`; the remaining
        !! updates follow a cosine decay to `base_rate*final_rate_fraction`.
        !! The continuous fractions have exact products from
        !! `rate_with_full_derivatives`.
        integer, intent(in) :: warmup_updates, total_updates
        real(dp), intent(in) :: peak_rate_fraction, final_rate_fraction
        type(mlp_learning_rate_schedule_t) :: schedule
        !! Default-initialized instances, standing in for empty
        !! structure constructors: nvfortran rejects `T()` outright,
        !! and a declared local carries the same default init.
        type(mlp_learning_rate_schedule_t) :: mlp_learning_rate_schedule_t_default

        schedule = mlp_learning_rate_schedule_t_default
        schedule%kind = MLP_SCHEDULE_ONE_CYCLE
        schedule%warmup_updates = warmup_updates
        schedule%total_updates = total_updates
        schedule%peak_rate_fraction = peak_rate_fraction
        schedule%final_rate_fraction = final_rate_fraction
    end function make_mlp_schedule_one_cycle

    logical function schedule_valid(self) result(valid)
        class(mlp_learning_rate_schedule_t), intent(in) :: self

        valid = self%kind >= MLP_SCHEDULE_CONSTANT .and. &
            self%kind <= MLP_SCHEDULE_ONE_CYCLE .and. &
            self%warmup_updates >= 0 .and. self%total_updates >= 0 .and. &
            ieee_is_finite(self%min_rate_fraction) .and. &
            ieee_is_finite(self%decay_factor) .and. &
            ieee_is_finite(self%peak_rate_fraction) .and. &
            ieee_is_finite(self%final_rate_fraction) .and. &
            self%min_rate_fraction >= 0.0_dp .and. &
            self%min_rate_fraction <= 1.0_dp .and. self%decay_factor > 0.0_dp .and. &
            self%decay_factor <= 1.0_dp .and. self%peak_rate_fraction > 0.0_dp .and. &
            self%final_rate_fraction > 0.0_dp
        if (.not. valid) return
        select case (self%kind)
        case (MLP_SCHEDULE_LINEAR_WARMUP)
            valid = self%warmup_updates >= 1
        case (MLP_SCHEDULE_COSINE_DECAY)
            valid = self%total_updates >= 1
        case (MLP_SCHEDULE_WARMUP_COSINE)
            valid = self%warmup_updates >= 1 .and. &
                self%total_updates > self%warmup_updates
        case (MLP_SCHEDULE_EXPONENTIAL_DECAY)
            ! `warmup_updates` is allowed to be zero.
            valid = self%decay_factor < 1.0_dp
        case (MLP_SCHEDULE_ONE_CYCLE)
            valid = self%warmup_updates >= 1 .and. &
                self%total_updates > self%warmup_updates .and. &
                self%peak_rate_fraction >= 1.0_dp .and. &
                self%final_rate_fraction <= self%peak_rate_fraction
        end select
    end function schedule_valid

    logical function schedule_device_supported(self, device_kind) result(supported)
        !! Report whether this scalar schedule has a resident device lowering.
        !!
        !! Schedules are stateless host products today.  They can be copied
        !! into a future native CUDA/OpenACC optimizer kernel, but no such
        !! kernel is linked by this release, so CUDA is explicitly refused.
        class(mlp_learning_rate_schedule_t), intent(in) :: self
        integer, intent(in) :: device_kind

        select case (device_kind)
        case (FORTML_DEVICE_CPU)
            supported = self%valid()
        case (FORTML_DEVICE_CUDA)
            supported = .false.
        case default
            supported = .false.
        end select
    end function schedule_device_supported

    subroutine schedule_rate(self, update, base_rate, rate, status)
        class(mlp_learning_rate_schedule_t), intent(in) :: self
        integer, intent(in) :: update
        real(dp), intent(in) :: base_rate
        real(dp), intent(out) :: rate
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: d_base_rate, d_min_rate_fraction, d_decay_factor

        call self%rate_with_derivatives(update, base_rate, rate, d_base_rate, &
            d_min_rate_fraction, d_decay_factor, status)
    end subroutine schedule_rate

    subroutine schedule_rate_with_derivatives(self, update, base_rate, rate, &
            d_base_rate, d_min_rate_fraction, d_decay_factor, status)
        class(mlp_learning_rate_schedule_t), intent(in) :: self
        integer, intent(in) :: update
        real(dp), intent(in) :: base_rate
        real(dp), intent(out) :: rate, d_base_rate, d_min_rate_fraction, d_decay_factor
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: d_peak_rate_fraction, d_final_rate_fraction

        call self%rate_with_full_derivatives(update, base_rate, rate, d_base_rate, &
            d_min_rate_fraction, d_decay_factor, d_peak_rate_fraction, &
            d_final_rate_fraction, status)
    end subroutine schedule_rate_with_derivatives

    subroutine schedule_rate_with_full_derivatives(self, update, base_rate, rate, &
            d_base_rate, d_min_rate_fraction, d_decay_factor, &
            d_peak_rate_fraction, d_final_rate_fraction, status)
        !! Return exact products for every continuous schedule parameter.
        !!
        !! The legacy three-product families return zero for the additional
        !! one-cycle products.  For `MLP_SCHEDULE_ONE_CYCLE`, the returned
        !! derivatives are with respect to `(base_rate, min_rate_fraction,
        !! decay_factor, peak_rate_fraction, final_rate_fraction)`; the
        !! minimum-rate and decay-factor slots are zero, preserving the
        !! common schedule product layout while exposing the full one-cycle
        !! tangent in the final two outputs.
        class(mlp_learning_rate_schedule_t), intent(in) :: self
        integer, intent(in) :: update
        real(dp), intent(in) :: base_rate
        real(dp), intent(out) :: rate, d_base_rate, d_min_rate_fraction, d_decay_factor
        real(dp), intent(out) :: d_peak_rate_fraction, d_final_rate_fraction
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: fraction, factor, d_factor_min, d_factor_decay
        real(dp) :: d_factor_peak, d_factor_final
        real(dp) :: phase, progress, exponent
        integer :: elapsed

        rate = 0.0_dp
        d_base_rate = 0.0_dp
        d_min_rate_fraction = 0.0_dp
        d_decay_factor = 0.0_dp
        d_peak_rate_fraction = 0.0_dp
        d_final_rate_fraction = 0.0_dp
        if (.not. self%valid() .or. update < 1 .or. &
            .not. ieee_is_finite(base_rate) .or. base_rate <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP schedule: invalid schedule, update, or base rate")
            return
        end if

        factor = 1.0_dp
        d_factor_min = 0.0_dp
        d_factor_decay = 0.0_dp
        d_factor_peak = 0.0_dp
        d_factor_final = 0.0_dp
        select case (self%kind)
        case (MLP_SCHEDULE_CONSTANT)
            continue
        case (MLP_SCHEDULE_LINEAR_WARMUP)
            fraction = min(1.0_dp, real(update, dp)/real(self%warmup_updates, dp))
            factor = fraction
        case (MLP_SCHEDULE_COSINE_DECAY)
            progress = min(1.0_dp, real(update, dp)/real(self%total_updates, dp))
            phase = acos(-1.0_dp)*progress
            factor = self%min_rate_fraction + (1.0_dp-self%min_rate_fraction)* &
                0.5_dp*(1.0_dp+cos(phase))
            d_factor_min = 0.5_dp*(1.0_dp-cos(phase))
        case (MLP_SCHEDULE_WARMUP_COSINE)
            if (update <= self%warmup_updates) then
                factor = real(update, dp)/real(self%warmup_updates, dp)
            else
                progress = min(1.0_dp, real(update-self%warmup_updates, dp)/ &
                    real(self%total_updates-self%warmup_updates, dp))
                phase = acos(-1.0_dp)*progress
                factor = self%min_rate_fraction + (1.0_dp-self%min_rate_fraction)* &
                    0.5_dp*(1.0_dp+cos(phase))
                d_factor_min = 0.5_dp*(1.0_dp-cos(phase))
            end if
        case (MLP_SCHEDULE_EXPONENTIAL_DECAY)
            elapsed = max(0, update-self%warmup_updates)
            exponent = real(elapsed, dp)
            factor = self%decay_factor**exponent
            if (elapsed > 0) d_factor_decay = exponent * &
                self%decay_factor**real(elapsed-1, dp)
        case (MLP_SCHEDULE_ONE_CYCLE)
            if (update <= self%warmup_updates) then
                progress = real(update, dp)/real(self%warmup_updates, dp)
                factor = 1.0_dp + (self%peak_rate_fraction-1.0_dp)*progress
                d_factor_peak = progress
            else
                progress = min(1.0_dp, real(update-self%warmup_updates, dp)/ &
                    real(self%total_updates-self%warmup_updates, dp))
                phase = acos(-1.0_dp)*progress
                fraction = 0.5_dp*(1.0_dp+cos(phase))
                factor = self%final_rate_fraction + &
                    (self%peak_rate_fraction-self%final_rate_fraction)*fraction
                d_factor_peak = fraction
                d_factor_final = 1.0_dp-fraction
            end if
        end select

        rate = base_rate*factor
        d_base_rate = factor
        d_min_rate_fraction = base_rate*d_factor_min
        d_decay_factor = base_rate*d_factor_decay
        d_peak_rate_fraction = base_rate*d_factor_peak
        d_final_rate_fraction = base_rate*d_factor_final
        if (.not. ieee_is_finite(rate) .or. .not. ieee_is_finite(d_base_rate) .or. &
            .not. ieee_is_finite(d_min_rate_fraction) .or. &
            .not. ieee_is_finite(d_decay_factor) .or. &
            .not. ieee_is_finite(d_peak_rate_fraction) .or. &
            .not. ieee_is_finite(d_final_rate_fraction) .or. rate <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP schedule: rate evaluation is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine schedule_rate_with_full_derivatives

end module fortml_mlp_schedules
