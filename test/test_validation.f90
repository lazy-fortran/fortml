program test_validation
    use, intrinsic :: iso_fortran_env, only: real64
    use fortml_validation, only: kfold_splitter_t, stratified_kfold_splitter_t, &
        time_series_splitter_t, estimator_score_metadata_t, &
        estimator_validation_metadata_t, FORTML_SCORE_INPUT_PROBABILITY, &
        FORTML_SCORE_LOG_LOSS
    use fortml_estimator_capabilities, only: estimator_capability_t, &
        make_regressor_capabilities
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer :: failures

    failures = 0
    call test_kfold(failures)
    call test_stratified(failures)
    call test_time_series(failures)
    call test_metadata(failures)
    call test_refusals(failures)
    if (failures > 0) error stop "validation tests failed"
    write (*, '(a)') "PASS validation independent behavioral oracles"

contains

    subroutine check(condition, label, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: label
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (*, '(a)') "  FAIL "//label
        end if
    end subroutine check

    subroutine test_kfold(failures)
        integer, intent(inout) :: failures
        type(kfold_splitter_t) :: splitter, repeated
        type(fortnum_status_t) :: status
        integer, allocatable :: train(:), test(:), train_again(:), test_again(:)
        logical :: has_split
        integer :: fold

        call splitter%initialize(7, 3, status)
        call check(status_ok(status), "kfold initialization", failures)
        do fold = 1, 3
            call splitter%next_split(train, test, has_split, status)
            call check(status_ok(status) .and. has_split, "kfold next split", failures)
            call check(size(train) + size(test) == 7, "kfold partition size", failures)
            call check(size(test) == merge(3, 2, fold == 1), &
                "kfold balanced test size", failures)
            select case (fold)
            case (1)
                call check(all(test == [1, 2, 3]), "kfold first oracle", failures)
            case (2)
                call check(all(test == [4, 5]), "kfold second oracle", failures)
            case (3)
                call check(all(test == [6, 7]), "kfold third oracle", failures)
            end select
        end do
        call splitter%next_split(train, test, has_split, status)
        call check(status_ok(status) .and. .not. has_split, "kfold exhaustion", failures)
        call splitter%reset()
        call repeated%initialize(7, 3, status, shuffle=.true., seed=91)
        call splitter%initialize(7, 3, status, shuffle=.true., seed=91)
        call splitter%next_split(train, test, has_split, status)
        call repeated%next_split(train_again, test_again, has_split, status)
        call check(all(test == test_again), "seeded kfold reproducibility", failures)
        call check(all(train == train_again), "seeded train reproducibility", failures)
    end subroutine test_kfold

    subroutine test_stratified(failures)
        integer, intent(inout) :: failures
        type(stratified_kfold_splitter_t) :: splitter
        type(fortnum_status_t) :: status
        integer, allocatable :: train(:), test(:)
        integer :: labels(8), counts(3), fold
        logical :: has_split

        labels = [10, 10, 20, 20, 20, 30, 30, 30]
        call splitter%initialize(labels, 3, status)
        call check(status_ok(status), "stratified initialization", failures)
        do fold = 1, 3
            call splitter%next_split(train, test, has_split, status)
            call check(status_ok(status) .and. has_split, &
                "stratified next split", failures)
            counts = [count(labels(test) == 10), count(labels(test) == 20), &
                count(labels(test) == 30)]
            call check(counts(1) <= 1 .and. counts(2) <= 1 .and. counts(3) <= 1, &
                "stratified per-class balance", failures)
            call check(size(train) + size(test) == size(labels), &
                "stratified partition size", failures)
        end do
    end subroutine test_stratified

    subroutine test_time_series(failures)
        integer, intent(inout) :: failures
        type(time_series_splitter_t) :: splitter, replay
        type(fortnum_status_t) :: status
        integer, allocatable :: train(:), test(:), train_again(:), test_again(:)
        logical :: has_split, replay_has_split
        integer :: fold

        call splitter%initialize(11, 3, status, test_size=2, gap=1, &
            max_train_size=4)
        call check(status_ok(status), "time-series initialization", failures)
        call check(splitter%sample_count() == 11 .and. &
            splitter%fold_count() == 3 .and. splitter%test_window() == 2 .and. &
            splitter%gap_size() == 1 .and. splitter%rolling(), &
            "time-series metadata", failures)
        do fold = 1, 3
            call splitter%next_split(train, test, has_split, status)
            call check(status_ok(status) .and. has_split, &
                "time-series next split", failures)
            select case (fold)
            case (1)
                call check(all(train == [1, 2, 3, 4]) .and. &
                    all(test == [6, 7]), "time-series first rolling window", failures)
            case (2)
                call check(all(train == [3, 4, 5, 6]) .and. &
                    all(test == [8, 9]), "time-series second rolling window", failures)
            case (3)
                call check(all(train == [5, 6, 7, 8]) .and. &
                    all(test == [10, 11]), "time-series third rolling window", failures)
            end select
            call check(maxval(train) < minval(test) - 1, &
                "time-series gap is excluded", failures)
        end do
        call splitter%next_split(train, test, has_split, status)
        call check(status_ok(status) .and. .not. has_split, &
            "time-series exhaustion", failures)
        call splitter%reset()
        call replay%initialize(11, 3, status, test_size=2, gap=1, &
            max_train_size=4)
        call splitter%next_split(train, test, has_split, status)
        call replay%next_split(train_again, test_again, replay_has_split, status)
        call check(has_split .and. replay_has_split .and. all(train == train_again) &
            .and. all(test == test_again), "time-series reset replay", failures)
    end subroutine test_time_series

    subroutine test_metadata(failures)
        integer, intent(inout) :: failures
        type(estimator_capability_t) :: capability
        type(estimator_score_metadata_t) :: score
        type(estimator_validation_metadata_t) :: metadata
        type(fortnum_status_t) :: status

        capability = make_regressor_capabilities("oracle", 2, 1, status)
        call check(status_ok(status), "regressor capability oracle", failures)
        call score%initialize("log-loss", FORTML_SCORE_INPUT_PROBABILITY, status, &
            kind=FORTML_SCORE_LOG_LOSS, higher_is_better=.false., &
            supports_sample_weight=.true., differentiable=.true.)
        call check(status_ok(status) .and. score%valid(), &
            "score metadata initialization", failures)
        call metadata%initialize("oracle", capability, score, status, &
            cloneable=.true., resettable=.true., parameter_count=3)
        call check(status_ok(status) .and. metadata%valid(), &
            "validation metadata initialization", failures)
        call check(metadata%can_clone() .and. metadata%can_reset(), &
            "clone/reset metadata", failures)
        call check(score%oriented_value(0.2_real64) > &
            score%oriented_value(0.4_real64), &
            "loss scorer maximize orientation", failures)
        call check(score%prefer(0.2_real64, 0.4_real64), &
            "loss scorer preference", failures)
    end subroutine test_metadata

    subroutine test_refusals(failures)
        integer, intent(inout) :: failures
        type(kfold_splitter_t) :: splitter
        type(stratified_kfold_splitter_t) :: stratified
        type(fortnum_status_t) :: status
        integer, allocatable :: train(:), test(:)
        logical :: has_split

        call splitter%initialize(3, 4, status)
        call check(.not. status_ok(status), "too many folds refusal", failures)
        call splitter%next_split(train, test, has_split, status)
        call check(.not. status_ok(status), "uninitialized split refusal", failures)
        call stratified%initialize([1, 1, 2], 4, status)
        call check(.not. status_ok(status), "invalid stratified fold refusal", failures)
    end subroutine test_refusals

end program test_validation
