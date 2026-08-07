program fortml_bench_hyperparameter_search
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortnum_status, only: fortnum_status_t, status_ok, status_set, FORTNUM_OK
    use fortopt_objective, only: objective_t
    use fortml_hyperparameter_search, only: hyperparameter_search_result_t, &
        hyperparameter_grid_search, hyperparameter_lbfgsb_search, &
        hyperparameter_lbfgsb_multistart_search, hyperparameter_random_search
    implicit none

    type(objective_t) :: objective
    type(hyperparameter_search_result_t) :: grid_result, optimizer_result, random_result
    type(hyperparameter_search_result_t) :: multistart_result
    type(fortnum_status_t) :: status
    real(dp) :: lower(3), upper(3), initial(3), elapsed
    integer(int64) :: start_clock, end_clock, clock_rate

    lower = [-2.0_dp, -2.0_dp, -2.0_dp]
    upper = [2.0_dp, 2.0_dp, 2.0_dp]
    initial = [0.0_dp, 0.0_dp, 0.0_dp]
    call objective%initialize(3, objective_callback, status)
    if (.not. status_ok(status)) error stop "objective initialization failed"
    call system_clock(start_clock, clock_rate)
    call hyperparameter_grid_search(objective, lower, upper, [5, 5, 5], &
        grid_result, status)
    call system_clock(end_clock)
    if (.not. status_ok(status)) error stop "grid search failed"
    elapsed = real(end_clock - start_clock, dp)/real(clock_rate, dp)
    write (*, '(a,i0,a,es24.16,a,es24.16)') "grid,", grid_result%evaluations, &
        ",", grid_result%best_value, ",", elapsed

    call system_clock(start_clock)
    call hyperparameter_lbfgsb_search(objective, initial, lower, upper, &
        optimizer_result, status)
    call system_clock(end_clock)
    if (.not. status_ok(status)) error stop "L-BFGS-B search failed"
    elapsed = real(end_clock - start_clock, dp)/real(clock_rate, dp)
    write (*, '(a,i0,a,es24.16,a,es24.16)') "lbfgsb,", optimizer_result%evaluations, &
        ",", optimizer_result%best_value, ",", elapsed

    call system_clock(start_clock, clock_rate)
    call hyperparameter_lbfgsb_multistart_search(objective, lower, upper, 8, &
        20260807_int64, multistart_result, status)
    call system_clock(end_clock)
    if (.not. status_ok(status)) error stop "multistart L-BFGS-B search failed"
    elapsed = real(end_clock - start_clock, dp)/real(clock_rate, dp)
    write (*, '(a,i0,a,i0,a,es24.16,a,es24.16)') "lbfgsb_multistart,", &
        multistart_result%start_count, ",", multistart_result%evaluations, ",", &
        multistart_result%best_value, ",", elapsed

    call system_clock(start_clock)
    call hyperparameter_random_search(objective, lower, upper, 128, 20260807_int64, &
        random_result, status)
    call system_clock(end_clock)
    if (.not. status_ok(status)) error stop "random search failed"
    elapsed = real(end_clock - start_clock, dp)/real(clock_rate, dp)
    write (*, '(a,i0,a,es24.16,a,es24.16)') "random,", random_result%evaluations, &
        ",", random_result%best_value, ",", elapsed

contains

    subroutine objective_callback(x, value, gradient, status)
        real(dp), intent(in) :: x(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status

        value = sum((x - [1.0_dp, -0.5_dp, 0.25_dp])**2)
        gradient = 2.0_dp*(x - [1.0_dp, -0.5_dp, 0.25_dp])
        call status_set(status, FORTNUM_OK, "")
    end subroutine objective_callback

end program fortml_bench_hyperparameter_search
