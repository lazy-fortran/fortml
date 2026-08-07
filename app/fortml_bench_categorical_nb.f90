program fortml_bench_categorical_nb
    !! Release workload for weighted Categorical Naive Bayes.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortml_categorical_naive_bayes, only: categorical_naive_bayes_t
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none
    integer, parameter :: n_samples = 12, n_features = 2, n_classes = 2
    integer, parameter :: fit_repetitions = 64, predict_repetitions = 256
    integer :: x(n_samples, n_features), labels(n_samples), query(2, n_features)
    integer :: predicted(2), unit, env_status, i, j
    real(dp) :: probabilities(2, n_classes), fit_seconds, predict_seconds
    integer(int64) :: clock_start, clock_end, clock_rate
    character(len=1024) :: path
    type(categorical_naive_bayes_t) :: model
    type(fortnum_status_t) :: status

    x(:, 1) = [1, 1, 2, 2, 1, 2, 1, 2, 1, 2, 1, 2]
    x(:, 2) = [10, 20, 10, 20, 20, 10, 10, 20, 10, 20, 20, 10]
    labels = [-1, -1, -1, 4, 4, 4, -1, 4, -1, 4, 4, -1]
    query(1, :) = [1, 10]
    query(2, :) = [2, 20]
    call model%fit(x, labels, status, alpha=1.0_dp)
    if (.not. status_ok(status)) error stop "CategoricalNB benchmark fit failed"
    call model%predict_proba(query, probabilities, status)
    call model%predict(query, predicted, status)
    if (.not. status_ok(status)) error stop "CategoricalNB benchmark prediction failed"

    call get_environment_variable("FORTML_BENCH_CATEGORICAL_NB_ORACLE", path, &
        status=env_status)
    if (env_status == 0 .and. len_trim(path) > 0) then
        open (newunit=unit, file=trim(path), status="replace", action="write")
        write (unit, '(a)') "quantity,row,column,value"
        do i = 1, size(query, 1)
            write (unit, '(a,i0,a,i0,a,i0)') "prediction,", i, ",1,", predicted(i)
            do j = 1, n_classes
                write (unit, '(a,i0,a,i0,a,es26.17e3)') "probability,", i, ",", j, &
                    ",", probabilities(i, j)
            end do
        end do
        close (unit)
    end if
    if (oracle_only_requested()) stop

    call system_clock(clock_start, clock_rate)
    do i = 1, fit_repetitions
        call model%fit(x, labels, status, alpha=1.0_dp)
        if (.not. status_ok(status)) error stop "CategoricalNB timed fit failed"
    end do
    call system_clock(clock_end)
    fit_seconds = real(clock_end-clock_start, dp)/real(clock_rate, dp) &
        /real(fit_repetitions, dp)
    call system_clock(clock_start, clock_rate)
    do i = 1, predict_repetitions
        call model%predict_proba(query, probabilities, status)
        if (.not. status_ok(status)) error stop "CategoricalNB timed prediction failed"
    end do
    call system_clock(clock_end)
    predict_seconds = real(clock_end-clock_start, dp)/real(clock_rate, dp) &
        /real(predict_repetitions, dp)
    write (*, '(a,es24.16)') "categorical_nb_fit,", fit_seconds
    write (*, '(a,es24.16)') "categorical_nb_predict,", predict_seconds

contains

    logical function oracle_only_requested()
        character(len=16) :: value
        integer :: code
        call get_environment_variable("FORTML_BENCH_ORACLE_ONLY", value, status=code)
        oracle_only_requested = code == 0 .and. trim(value) == "1"
    end function oracle_only_requested

end program fortml_bench_categorical_nb
