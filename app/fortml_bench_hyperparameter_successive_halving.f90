program fortml_bench_hyperparameter_successive_halving
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortnum_status, only: fortnum_status_t, status_ok, status_set, FORTNUM_OK
    use fortml_hyperparameter_search, only: &
        hyperparameter_resource_objective_t, hyperparameter_search_result_t, &
        hyperparameter_successive_halving_search, &
        hyperparameter_lbfgsb_resource_search
    implicit none

    type :: fixture_t
        real(dp) :: target(3) = [0.75_dp, -0.25_dp, 0.40_dp]
    end type fixture_t

    type(fixture_t), target :: fixture
    type(hyperparameter_resource_objective_t) :: objective
    type(hyperparameter_search_result_t) :: halving, refined
    type(fortnum_status_t) :: status
    real(dp) :: lower(3), upper(3), elapsed
    integer(int64) :: start_clock, end_clock, clock_rate

    lower = [-2.0_dp, -2.0_dp, -2.0_dp]
    upper = [2.0_dp, 2.0_dp, 2.0_dp]
    call objective%initialize_context(3, fixture, objective_callback, status)
    if (.not. status_ok(status)) error stop "resource objective initialization failed"
    call system_clock(start_clock, clock_rate)
    call hyperparameter_successive_halving_search(objective, lower, upper, 64, 1, 16, &
        2, 20260809_int64, halving, status)
    call system_clock(end_clock)
    if (.not. status_ok(status)) error stop "successive halving failed"
    elapsed = real(end_clock - start_clock, dp)/real(clock_rate, dp)
    write (*, '(a,i0,a,i0,a,i0,a,es24.16,a,es24.16)') &
        "successive_halving,", halving%candidate_count, ",", halving%rung_count, ",", &
        halving%evaluations, ",", halving%best_value, ",", elapsed

    call system_clock(start_clock)
    call hyperparameter_lbfgsb_resource_search(objective, halving%best_parameters, lower, &
        upper, 16, refined, status)
    call system_clock(end_clock)
    if (.not. status_ok(status)) error stop "resource L-BFGS-B failed"
    elapsed = real(end_clock - start_clock, dp)/real(clock_rate, dp)
    write (*, '(a,i0,a,es24.16,a,es24.16)') "lbfgsb_resource,", refined%evaluations, ",", &
        refined%best_value, ",", elapsed

contains

    subroutine objective_callback(context, parameters, resource, value, gradient, status)
        class(*), intent(inout) :: context
        real(dp), intent(in) :: parameters(:)
        integer, intent(in) :: resource
        real(dp), intent(out) :: value
        real(dp), intent(out) :: gradient(:)
        type(fortnum_status_t), intent(out) :: status

        select type (fixture => context)
            type is (fixture_t)
                value = sum((parameters - fixture%target)**2) + &
                    0.25_dp/real(resource, dp)
                gradient = 2.0_dp*(parameters - fixture%target)
                call status_set(status, FORTNUM_OK, "")
            class default
                value = huge(1.0_dp)
                gradient = 0.0_dp
                call status_set(status, FORTNUM_OK, "")
        end select
    end subroutine objective_callback

end program fortml_bench_hyperparameter_successive_halving
