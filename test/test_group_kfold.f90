program test_group_kfold
    use fortml_validation, only: group_kfold_splitter_t
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer :: failures

    failures = 0
    call test_group_isolation_and_balance(failures)
    call test_seed_replay_and_refusals(failures)
    if (failures > 0) error stop "group kfold tests failed"
    write (*, '(a)') "PASS group kfold independent behavioral oracles"

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

    subroutine test_group_isolation_and_balance(failures)
        integer, intent(inout) :: failures
        type(group_kfold_splitter_t) :: splitter
        type(fortnum_status_t) :: status
        integer, allocatable :: train(:), test(:)
        integer :: groups(10), expected(4), fold, group_id
        logical :: has_split

        ! Group sizes are 3,2,1,2,1,1.  The documented largest-first
        ! packing oracle gives test folds [1,2,3,10], [4,5,6], [7,8,9].
        groups = [1, 1, 1, 2, 2, 3, 4, 4, 5, 6]
        call splitter%initialize(groups, 3, status)
        call check(status_ok(status), "group initialization", failures)
        call check(splitter%sample_count() == 10, "group sample count", failures)
        call check(splitter%group_count() == 6, "group count", failures)
        call check(splitter%fold_count() == 3, "group fold count", failures)
        do fold = 1, 3
            call splitter%next_split(train, test, has_split, status)
            call check(status_ok(status) .and. has_split, &
                "group next split", failures)
            call check(size(train) + size(test) == 10, &
                "group partition size", failures)
            do group_id = 1, 6
                call check(count(groups(test) == group_id) == 0 .or. &
                    count(groups(test) == group_id) == count(groups == group_id), &
                    "group is not split", failures)
            end do
            select case (fold)
            case (1)
                expected = [1, 2, 3, 10]
            case (2)
                expected = [4, 5, 6, 0]
            case (3)
                expected = [7, 8, 9, 0]
            end select
            if (fold == 1) then
                call check(all(test == expected), "group largest-first oracle", failures)
            else
                call check(all(test == pack(expected, expected > 0)), &
                    "group balanced packing oracle", failures)
            end if
        end do
        call splitter%next_split(train, test, has_split, status)
        call check(status_ok(status) .and. .not. has_split, &
            "group exhaustion", failures)
    end subroutine test_group_isolation_and_balance

    subroutine test_seed_replay_and_refusals(failures)
        integer, intent(inout) :: failures
        type(group_kfold_splitter_t) :: splitter, repeated, invalid
        type(fortnum_status_t) :: status
        integer, allocatable :: train(:), test(:), train_again(:), test_again(:)
        integer :: groups(10)
        logical :: has_split

        groups = [1, 1, 1, 2, 2, 3, 4, 4, 5, 6]
        call splitter%initialize(groups, 3, status, shuffle=.true., seed=91)
        call repeated%initialize(groups, 3, status, shuffle=.true., seed=91)
        call splitter%next_split(train, test, has_split, status)
        call repeated%next_split(train_again, test_again, has_split, status)
        call check(all(test == test_again), "seeded group test replay", failures)
        call check(all(train == train_again), "seeded group train replay", failures)
        call splitter%reset()
        call splitter%next_split(train_again, test_again, has_split, status)
        call check(all(test == test_again), "group reset replay", failures)

        call invalid%initialize([1, 1, 2], 3, status)
        call check(.not. status_ok(status), "too many groups folds refusal", failures)
        call invalid%initialize(groups, 3, status, shuffle=.true., seed=0)
        call check(.not. status_ok(status), "nonpositive group seed refusal", failures)
        call invalid%next_split(train, test, has_split, status)
        call check(.not. status_ok(status), "uninitialized group refusal", failures)
    end subroutine test_seed_replay_and_refusals

end program test_group_kfold
