program fortml_bench_time_series_split
    !! Release workload for chronological validation and scorer metadata.
    !! The companion Python benchmark independently derives every expected
    !! window from the same blocked-split contract before accepting timings.
    use, intrinsic :: iso_fortran_env, only: real64, int64
    use fortml_validation, only: time_series_splitter_t, &
        estimator_score_metadata_t, FORTML_SCORE_INPUT_PROBABILITY, &
        FORTML_SCORE_LOG_LOSS
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer, parameter :: n_samples = 29, n_splits = 4
    integer, parameter :: test_size = 3, gap = 2, max_train_size = 7
    integer, parameter :: repetitions = 2048
    type(time_series_splitter_t) :: splitter
    type(estimator_score_metadata_t) :: score
    type(fortnum_status_t) :: status
    integer, allocatable :: train(:), test(:)
    integer :: fold, repetition, i, environment_status, io_status, unit
    integer(int64) :: started, finished, clock_rate
    real(real64) :: elapsed
    logical :: has_split
    character(len=1024) :: oracle_path

    call splitter%initialize(n_samples, n_splits, status, test_size=test_size, &
        gap=gap, max_train_size=max_train_size)
    if (.not. status_ok(status)) error stop "time-series benchmark initialization failed"
    call score%initialize("log-loss", FORTML_SCORE_INPUT_PROBABILITY, status, &
        kind=FORTML_SCORE_LOG_LOSS, higher_is_better=.false., &
        supports_sample_weight=.true., differentiable=.true.)
    if (.not. status_ok(status)) error stop "score benchmark initialization failed"

    call system_clock(started, clock_rate)
    do repetition = 1, repetitions
        call splitter%reset()
        do fold = 1, n_splits
            call splitter%next_split(train, test, has_split, status)
            if (.not. status_ok(status) .or. .not. has_split) then
                error stop "time-series benchmark split failed"
            end if
        end do
    end do
    call system_clock(finished)
    elapsed = real(finished-started, real64)/real(clock_rate, real64) / &
        real(repetitions, real64)
    write (*, '(a,",",es24.16)') "time_series_split", elapsed
    write (*, '(a,",",es24.16)') "score_orientation", &
        score%oriented_value(0.2_real64)

    call get_environment_variable("FORTML_BENCH_TIME_SERIES_ORACLE", &
        oracle_path, status=environment_status)
    if (environment_status /= 0 .or. len_trim(oracle_path) == 0) stop
    open (newunit=unit, file=trim(oracle_path), status="replace", &
        action="write", iostat=io_status)
    if (io_status /= 0) error stop "time-series benchmark oracle open failed"
    write (unit, '(a)') "quantity,fold,index,value"
    call splitter%reset()
    do fold = 1, n_splits
        call splitter%next_split(train, test, has_split, status)
        if (.not. status_ok(status) .or. .not. has_split) then
            close (unit)
            error stop "time-series benchmark oracle split failed"
        end if
        do i = 1, size(train)
            write (unit, '(a,",",i0,",",i0,",",i0)') "train", fold, i, train(i)
        end do
        do i = 1, size(test)
            write (unit, '(a,",",i0,",",i0,",",i0)') "test", fold, i, test(i)
        end do
    end do
    close (unit)
end program fortml_bench_time_series_split
