program test_validation
    use fortml_validation, only: kfold_splitter_t, stratified_kfold_splitter_t
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer :: failures

    failures = 0
    call test_kfold(failures)
    call test_stratified(failures)
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
