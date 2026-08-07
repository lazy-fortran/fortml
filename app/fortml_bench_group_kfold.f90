program fortml_bench_group_kfold
    use, intrinsic :: iso_fortran_env, only: real64
    use fortml_validation, only: group_kfold_splitter_t
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer, parameter :: n_samples = 10, n_splits = 3, repetitions = 512
    integer, parameter :: groups(n_samples) = [1, 1, 1, 2, 2, 3, 4, 4, 5, 6]
    type(group_kfold_splitter_t) :: splitter
    type(fortnum_status_t) :: status
    integer, allocatable :: train(:), test(:)
    integer :: fold, i, repetition, environment_status, io_status, unit
    logical :: has_split
    real(real64) :: started, finished
    character(len=1024) :: oracle_path

    call splitter%initialize(groups, n_splits, status)
    if (.not. status_ok(status)) error stop "group kfold benchmark initialization failed"
    call cpu_time(started)
    do repetition = 1, repetitions
        call splitter%reset()
        do fold = 1, n_splits
            call splitter%next_split(train, test, has_split, status)
            if (.not. status_ok(status) .or. .not. has_split) then
                error stop "group kfold benchmark split failed"
            end if
        end do
    end do
    call cpu_time(finished)
    write (*, '(a,",",es24.16)') "group_kfold_split", &
        (finished - started)/real(repetitions, real64)

    call get_environment_variable("FORTML_BENCH_GROUP_KFOLD_ORACLE", &
        oracle_path, status=environment_status)
    if (environment_status /= 0 .or. len_trim(oracle_path) == 0) stop
    open (newunit=unit, file=trim(oracle_path), status="replace", &
        action="write", iostat=io_status)
    if (io_status /= 0) error stop "group kfold benchmark oracle open failed"
    write (unit, '(a)') "quantity,fold,index,value"
    call splitter%reset()
    do fold = 1, n_splits
        call splitter%next_split(train, test, has_split, status)
        if (.not. status_ok(status) .or. .not. has_split) then
            close (unit)
            error stop "group kfold benchmark oracle split failed"
        end if
        do i = 1, size(test)
            write (unit, '(a,",",i0,",",i0,",",i0)') "test", fold, i, test(i)
        end do
        do i = 1, size(train)
            write (unit, '(a,",",i0,",",i0,",",i0)') "train", fold, i, train(i)
        end do
    end do
    close (unit)
end program fortml_bench_group_kfold
