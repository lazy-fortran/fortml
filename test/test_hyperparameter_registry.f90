program test_hyperparameter_registry
    !! Independent behavioral oracle for transform-aware optimizer vectors.
    use, intrinsic :: iso_fortran_env, only: dp => real64
    use fortml_hyperparameter_registry, only: hyperparameter_block_t, &
        hyperparameter_registry_t, HYPERPARAMETER_IDENTITY, HYPERPARAMETER_LOG, &
        HYPERPARAMETER_LOGIT
    use fortnum_status, only: fortnum_status_t, status_ok, status_set, &
        FORTNUM_OK, FORTNUM_DOMAIN_ERROR
    implicit none

    type :: holder_t
        real(dp) :: values(2)
    end type holder_t

    type(hyperparameter_block_t) :: identity, rate, probability, frozen
    type(hyperparameter_registry_t) :: registry
    type(fortnum_status_t) :: status
    type(holder_t), target :: holder
    type(hyperparameter_block_t) :: live
    real(dp) :: values(2), unconstrained(2), recovered(2), lower(2), upper(2)
    real(dp) :: wrong(1)
    real(dp) :: all_unconstrained(5), optimizer(4), optimizer_lower(4), optimizer_upper(4)
    real(dp) :: shifted(4)
    logical :: found
    integer :: first, last

    call identity%initialize_values("model.offset", [2.0_dp, -1.0_dp], status, &
        transform=HYPERPARAMETER_IDENTITY, lower=[-3.0_dp, -3.0_dp], &
        upper=[3.0_dp, 3.0_dp], provenance="unit-test", device="cpu", &
        hvp_available=.true.)
    call require(status, 1)
    if (.not. identity%initialized() .or. identity%size() /= 2) error stop 2
    if (identity%name() /= "model.offset" .or. identity%transform() /= HYPERPARAMETER_IDENTITY) error stop 3
    if (.not. identity%hvp_available() .or. identity%provenance() /= "unit-test" .or. &
        identity%device() /= "cpu") error stop 4
    call identity%get_unconstrained(unconstrained, status)
    call require(status, 5)
    if (maxval(abs(unconstrained-[2.0_dp, -1.0_dp])) > 1.0e-14_dp) error stop 6
    call identity%set_unconstrained([1.0_dp, -2.0_dp], status)
    call require(status, 7)
    call identity%get_physical(values, status)
    call require(status, 8)
    if (maxval(abs(values-[1.0_dp, -2.0_dp])) > 1.0e-14_dp) error stop 9
    call identity%set_physical([4.0_dp, 0.0_dp], status)
    if (status_ok(status)) error stop 10

    call rate%initialize_values("optimizer.learning_rate", [0.25_dp, 2.0_dp], status, &
        transform=HYPERPARAMETER_LOG, lower=[0.01_dp, 0.1_dp], &
        upper=[1.0_dp, 10.0_dp], provenance="schedule", device="cpu")
    call require(status, 11)
    call rate%get_unconstrained(unconstrained, status)
    call require(status, 12)
    if (maxval(abs(unconstrained-[log(0.25_dp), log(2.0_dp)])) > 1.0e-14_dp) error stop 13
    call rate%set_unconstrained(unconstrained, status)
    call require(status, 14)
    call rate%get_physical(recovered, status)
    call require(status, 15)
    if (maxval(abs(recovered-[0.25_dp, 2.0_dp])) > 1.0e-14_dp) error stop 16
    call rate%unconstrained_bounds(lower, upper, status)
    call require(status, 17)
    if (maxval(abs(lower-[log(0.01_dp), log(0.1_dp)])) > 1.0e-14_dp .or. &
        maxval(abs(upper-[log(1.0_dp), log(10.0_dp)])) > 1.0e-14_dp) error stop 18
    call rate%set_physical([0.0_dp, 1.0_dp], status)
    if (status_ok(status)) error stop 19

    call probability%initialize_values("likelihood.probability", [0.2_dp, 0.8_dp], status, &
        transform=HYPERPARAMETER_LOGIT, lower=[0.0_dp, -1.0_dp], &
        upper=[1.0_dp, 1.0_dp], provenance="likelihood", device="cpu")
    call require(status, 20)
    call probability%get_unconstrained(unconstrained, status)
    call require(status, 21)
    if (maxval(abs(unconstrained-[log(0.2_dp/0.8_dp), log(1.8_dp/0.2_dp)])) > 1.0e-14_dp) error stop 22
    call probability%set_unconstrained(unconstrained, status)
    call require(status, 23)
    call probability%get_physical(recovered, status)
    call require(status, 24)
    if (maxval(abs(recovered-[0.2_dp, 0.8_dp])) > 1.0e-14_dp) error stop 25
    call probability%set_physical([0.0_dp, 0.5_dp], status)
    if (status_ok(status)) error stop 26

    call frozen%initialize_values("fixed.seed", [7.0_dp], status, trainable=.false., &
        provenance="fixture", device="cpu")
    call require(status, 27)
    call registry%add(rate, status)
    call require(status, 28)
    call registry%add(probability, status)
    call require(status, 29)
    call registry%add(frozen, status)
    call require(status, 30)
    if (registry%parameter_count() /= 5 .or. registry%trainable_count() /= 4) error stop 31
    call registry%pack_unconstrained(all_unconstrained, status)
    call require(status, 32)
    call registry%pack_trainable(optimizer, status)
    call require(status, 33)
    if (maxval(abs(optimizer-all_unconstrained(:4))) > 1.0e-14_dp) error stop 34
    call registry%optimizer_bounds(optimizer_lower, optimizer_upper, status)
    call require(status, 35)
    if (any(optimizer_lower >= optimizer_upper)) error stop 36
    shifted = optimizer
    shifted(1) = optimizer_lower(1)-1.0_dp
    shifted(2) = optimizer_upper(2)+1.0_dp
    call registry%project(shifted, status)
    call require(status, 37)
    if (any(shifted < optimizer_lower) .or. any(shifted > optimizer_upper)) error stop 38
    call registry%unpack_trainable(optimizer, status)
    call require(status, 39)
    call registry%range("likelihood.probability", first, last, found, trainable_only=.true.)
    if (.not. found .or. first /= 3 .or. last /= 4) error stop 40
    call registry%range("fixed.seed", first, last, found, trainable_only=.true.)
    if (found) error stop 41
    call registry%add(rate, status)
    if (status_ok(status)) error stop 42

    holder%values = [3.0_dp, 4.0_dp]
    call live%initialize("live.callback", 2, holder, holder_get, holder_set, status, &
        transform=HYPERPARAMETER_LOG, lower=[1.0_dp, 1.0_dp], upper=[10.0_dp, 10.0_dp], &
        provenance="callback", device="cpu")
    call require(status, 46)
    call live%set_unconstrained([log(5.0_dp), log(6.0_dp)], status)
    call require(status, 47)
    if (maxval(abs(holder%values-[5.0_dp, 6.0_dp])) > 1.0e-14_dp) error stop 48

    ! Explicit invalid configurations and shape errors must refuse early.
    call identity%initialize_values("bad.logit", [0.5_dp], status, &
        transform=HYPERPARAMETER_LOGIT)
    if (status_ok(status)) error stop 43
    call identity%initialize_values("bad.shape", [0.5_dp], status, lower=[0.0_dp, 0.0_dp])
    if (status_ok(status)) error stop 44
    call rate%get_unconstrained(wrong, status)
    if (status_ok(status)) error stop 45

    write (*, '(a)') "PASS transform-aware hyperparameter registry independent oracle"

contains

    subroutine holder_get(context, values, status)
        class(*), pointer, intent(in) :: context
        real(dp), intent(out) :: values(:)
        type(fortnum_status_t), intent(out) :: status

        select type (object => context)
            type is (holder_t)
            if (size(values) /= 2) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, "bad getter shape")
                return
            end if
            values = object%values
            call status_set(status, FORTNUM_OK, "")
        class default
            call status_set(status, FORTNUM_DOMAIN_ERROR, "bad getter context")
        end select
    end subroutine holder_get

    subroutine holder_set(context, values, status)
        class(*), pointer, intent(inout) :: context
        real(dp), intent(in) :: values(:)
        type(fortnum_status_t), intent(out) :: status

        select type (object => context)
            type is (holder_t)
            if (size(values) /= 2) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, "bad setter shape")
                return
            end if
            object%values = values
            call status_set(status, FORTNUM_OK, "")
        class default
            call status_set(status, FORTNUM_DOMAIN_ERROR, "bad setter context")
        end select
    end subroutine holder_set

    subroutine require(status, code)
        type(fortnum_status_t), intent(in) :: status
        integer, intent(in) :: code

        if (.not. status_ok(status)) error stop code
    end subroutine require

end program test_hyperparameter_registry
