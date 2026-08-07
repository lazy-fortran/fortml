program test_mlp_schedules
    !! Independent formula and finite-difference checks for built-in schedules.
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_DOMAIN_ERROR
    use fortml_mlp_schedules, only: &
        mlp_learning_rate_schedule_t, make_mlp_schedule_constant, &
        make_mlp_schedule_linear_warmup, make_mlp_schedule_cosine_decay, &
        make_mlp_schedule_warmup_cosine, make_mlp_schedule_exponential_decay, &
        make_mlp_schedule_one_cycle, &
        MLP_SCHEDULE_CONSTANT, MLP_SCHEDULE_LINEAR_WARMUP, &
        MLP_SCHEDULE_COSINE_DECAY, MLP_SCHEDULE_WARMUP_COSINE, &
        MLP_SCHEDULE_EXPONENTIAL_DECAY, MLP_SCHEDULE_ONE_CYCLE
    implicit none

    type(mlp_learning_rate_schedule_t) :: schedule, invalid
    type(fortnum_status_t) :: status
    real(dp) :: rate, dbase, dmin, ddecay, dpeak, dfinal, plus, minus, h, base
    real(dp) :: expected, pi, progress
    integer :: failures

    failures = 0
    h = 1.0e-6_dp
    base = 0.2_dp
    pi = acos(-1.0_dp)

    schedule = make_mlp_schedule_constant()
    call check(schedule%kind == MLP_SCHEDULE_CONSTANT .and. schedule%valid(), &
        "constant schedule metadata", failures)
    call schedule%rate_with_derivatives(7, base, rate, dbase, dmin, ddecay, status)
    call check(status_ok(status) .and. abs(rate-base) < 1.0e-15_dp .and. &
        abs(dbase-1.0_dp) < 1.0e-15_dp .and. abs(dmin) < 1.0e-15_dp .and. &
        abs(ddecay) < 1.0e-15_dp, "constant schedule products", failures)

    schedule = make_mlp_schedule_linear_warmup(4)
    call schedule%rate_with_derivatives(2, base, rate, dbase, dmin, ddecay, status)
    call check(status_ok(status) .and. abs(rate-0.5_dp*base) < 1.0e-15_dp .and. &
        abs(dbase-0.5_dp) < 1.0e-15_dp, "linear warmup midpoint", failures)
    call schedule%rate(8, base, rate, status)
    call check(status_ok(status) .and. abs(rate-base) < 1.0e-15_dp, &
        "linear warmup plateau", failures)

    schedule = make_mlp_schedule_cosine_decay(10, 0.1_dp)
    progress = 0.5_dp
    expected = base*(0.1_dp+0.9_dp*0.5_dp*(1.0_dp+cos(pi*progress)))
    call schedule%rate_with_derivatives(5, base, rate, dbase, dmin, ddecay, status)
    call check(status_ok(status) .and. abs(rate-expected) < 1.0e-15_dp .and. &
        abs(dmin-base*0.5_dp) < 1.0e-14_dp, &
        "cosine midpoint and minimum-fraction derivative", failures)
    call schedule%rate(10, base, rate, status)
    call check(status_ok(status) .and. abs(rate-0.1_dp*base) < 1.0e-15_dp, &
        "cosine terminal rate", failures)
    ! Independent finite-difference checks for base rate and min fraction.
    call schedule%rate_with_derivatives(3, base, rate, dbase, dmin, ddecay, status)
    schedule%min_rate_fraction = schedule%min_rate_fraction+h
    call schedule%rate(3, base, plus, status)
    schedule%min_rate_fraction = schedule%min_rate_fraction-2.0_dp*h
    call schedule%rate(3, base, minus, status)
    schedule%min_rate_fraction = schedule%min_rate_fraction+h
    call check(abs(dmin-(plus-minus)/(2.0_dp*h)) < 2.0e-10_dp, &
        "cosine minimum-fraction finite difference", failures)
    call schedule%rate(3, base+h, plus, status)
    call schedule%rate(3, base-h, minus, status)
    call check(abs(dbase-(plus-minus)/(2.0_dp*h)) < 2.0e-10_dp, &
        "cosine base-rate finite difference", failures)

    schedule = make_mlp_schedule_warmup_cosine(2, 10, 0.2_dp)
    call schedule%rate(2, base, rate, status)
    call check(status_ok(status) .and. abs(rate-base) < 1.0e-15_dp, &
        "warmup-cosine transition", failures)
    call schedule%rate(10, base, rate, status)
    call check(status_ok(status) .and. abs(rate-0.2_dp*base) < 1.0e-15_dp, &
        "warmup-cosine terminal rate", failures)

    schedule = make_mlp_schedule_exponential_decay(2, 0.8_dp)
    call schedule%rate_with_derivatives(5, base, rate, dbase, dmin, ddecay, status)
    expected = base*0.8_dp**3
    call check(status_ok(status) .and. abs(rate-expected) < 1.0e-15_dp, &
        "exponential decay recurrence", failures)
    schedule%decay_factor = schedule%decay_factor+h
    call schedule%rate(5, base, plus, status)
    schedule%decay_factor = schedule%decay_factor-2.0_dp*h
    call schedule%rate(5, base, minus, status)
    schedule%decay_factor = schedule%decay_factor+h
    call check(abs(ddecay-(plus-minus)/(2.0_dp*h)) < 2.0e-9_dp, &
        "exponential decay-factor finite difference", failures)
    call schedule%rate(2, base, rate, status)
    call check(status_ok(status) .and. abs(rate-base) < 1.0e-15_dp, &
        "exponential warmup plateau", failures)

    schedule = make_mlp_schedule_one_cycle(2, 10, 4.0_dp, 0.1_dp)
    call check(schedule%kind == MLP_SCHEDULE_ONE_CYCLE .and. schedule%valid(), &
        "one-cycle schedule metadata", failures)
    call schedule%rate_with_full_derivatives(1, base, rate, dbase, dmin, ddecay, &
        dpeak, dfinal, status)
    call check(status_ok(status) .and. abs(rate-0.5_dp*base*(1.0_dp+4.0_dp)) < 1.0e-15_dp .and. &
        abs(dbase-2.5_dp) < 1.0e-15_dp .and. abs(dpeak-0.5_dp*base) < 1.0e-15_dp .and. &
        abs(dfinal) < 1.0e-15_dp .and. abs(dmin) < 1.0e-15_dp .and. &
        abs(ddecay) < 1.0e-15_dp, "one-cycle warmup products", failures)
    call schedule%rate_with_full_derivatives(6, base, rate, dbase, dmin, ddecay, &
        dpeak, dfinal, status)
    call check(status_ok(status) .and. rate < base*4.0_dp .and. rate > base*0.1_dp .and. &
        dpeak > 0.0_dp .and. dfinal > 0.0_dp, "one-cycle cosine products", failures)
    call schedule%rate_with_full_derivatives(6, base, rate, dbase, dmin, ddecay, &
        dpeak, dfinal, status)
    schedule%peak_rate_fraction = schedule%peak_rate_fraction+h
    call schedule%rate(6, base, plus, status)
    schedule%peak_rate_fraction = schedule%peak_rate_fraction-2.0_dp*h
    call schedule%rate(6, base, minus, status)
    schedule%peak_rate_fraction = schedule%peak_rate_fraction+h
    call check(abs(dpeak-(plus-minus)/(2.0_dp*h)) < 2.0e-10_dp, &
        "one-cycle peak-rate finite difference", failures)
    schedule%final_rate_fraction = schedule%final_rate_fraction+h
    call schedule%rate(6, base, plus, status)
    schedule%final_rate_fraction = schedule%final_rate_fraction-2.0_dp*h
    call schedule%rate(6, base, minus, status)
    schedule%final_rate_fraction = schedule%final_rate_fraction+h
    call check(abs(dfinal-(plus-minus)/(2.0_dp*h)) < 2.0e-10_dp, &
        "one-cycle final-rate finite difference", failures)
    call check(.not. schedule%device_supported(2), &
        "one-cycle CUDA capability refusal", failures)
    invalid = make_mlp_schedule_one_cycle(2, 10, 0.5_dp, 0.1_dp)
    call check(.not. invalid%valid(), "one-cycle peak below initial refusal", failures)

    invalid = mlp_learning_rate_schedule_t(kind=MLP_SCHEDULE_WARMUP_COSINE, &
        warmup_updates=4, total_updates=4, min_rate_fraction=0.1_dp)
    call check(.not. invalid%valid(), "invalid schedule metadata refusal", failures)
    call invalid%rate(1, base, rate, status)
    call check(status%code == FORTNUM_DOMAIN_ERROR, "invalid schedule evaluation refusal", failures)

    if (failures > 0) error stop 1
    write (*, '(a)') "PASS MLP schedule analytic products independent oracle"

contains

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: description
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (*, '(a)') "FAIL "//trim(description)
        end if
    end subroutine check

end program test_mlp_schedules
