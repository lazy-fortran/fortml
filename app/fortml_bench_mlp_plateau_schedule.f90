program fortml_bench_mlp_plateau_schedule
    !! Complete-array release workload for metric-aware plateau schedules.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_status, only: fortnum_status_t, status_ok
    use fortml_mlp_schedules, only: mlp_learning_rate_schedule_t, &
        make_mlp_schedule_plateau, MLP_SCHEDULE_METRIC_MINIMIZE, &
        MLP_SCHEDULE_METRIC_MAXIMIZE
    implicit none

    integer, parameter :: scenario_count = 2, observation_count = 6
    integer, parameter :: repetitions = 4096
    real(dp), parameter :: base_rate = 0.2_dp
    integer, parameter :: metric_modes(scenario_count) = [ &
        MLP_SCHEDULE_METRIC_MINIMIZE, MLP_SCHEDULE_METRIC_MAXIMIZE]
    real(dp), parameter :: initial_best(scenario_count) = [1.0_dp, 0.7_dp]
    real(dp), parameter :: metrics(observation_count, scenario_count) = reshape([ &
        1.1_dp, 1.0_dp, 0.99_dp, 0.90_dp, 0.90_dp, 0.90_dp, &
        0.8_dp, 0.7_dp, 0.7_dp, 0.85_dp, 0.85_dp, 0.85_dp], &
        [observation_count, scenario_count])
    type(mlp_learning_rate_schedule_t) :: schedules(scenario_count)
    type(fortnum_status_t) :: status
    real(dp) :: rate, best, next_best, d_base, d_metric, d_best, d_delta, d_factor
    real(dp) :: elapsed, sink
    integer :: bad, reductions, next_bad, next_reductions
    integer :: scenario, observation, repetition, oracle_unit, index
    integer(int64) :: clock_start, clock_end, clock_rate
    integer :: environment_status
    character(len=1024) :: oracle_path
    logical :: improved, reduced

    do scenario = 1, scenario_count
        schedules(scenario) = make_mlp_schedule_plateau(2, 0.05_dp, 0.5_dp, &
            metric_modes(scenario))
        if (.not. schedules(scenario)%valid()) error stop &
            "MLP plateau benchmark: invalid fixture"
    end do

    oracle_unit = -1
    call get_environment_variable("FORTML_BENCH_MLP_PLATEAU_ORACLE", oracle_path, &
        status=environment_status)
    if (environment_status == 0 .and. len_trim(oracle_path) > 0) then
        open (newunit=oracle_unit, file=trim(oracle_path), status="replace", &
            action="write")
        write (oracle_unit, '(a)') "quantity,index,value"
        do scenario = 1, scenario_count
            bad = 0
            reductions = 0
            best = initial_best(scenario)
            do observation = 1, observation_count
                call schedules(scenario)%rate_with_metric_derivatives(observation, &
                    base_rate, metrics(observation, scenario), best, bad, reductions, &
                    rate, next_best, next_bad, next_reductions, improved, &
                    reduced, d_base, d_metric, d_best, d_delta, d_factor, status)
                if (.not. status_ok(status)) error stop &
                    "MLP plateau benchmark: evaluation failed"
                index = 100*(scenario-1)+observation
                call write_value(oracle_unit, "rate", index, rate)
                call write_value(oracle_unit, "d_base_rate", index, d_base)
                call write_value(oracle_unit, "d_metric", index, d_metric)
                call write_value(oracle_unit, "d_best_metric", index, d_best)
                call write_value(oracle_unit, "d_min_delta", index, d_delta)
                call write_value(oracle_unit, "d_factor", index, d_factor)
                call write_value(oracle_unit, "next_best_metric", index, next_best)
                call write_value(oracle_unit, "next_bad_updates", index, real(next_bad, dp))
                call write_value(oracle_unit, "next_reductions", index, real(next_reductions, dp))
                call write_value(oracle_unit, "improved", index, merge(1.0_dp, 0.0_dp, improved))
                call write_value(oracle_unit, "reduced", index, merge(1.0_dp, 0.0_dp, reduced))
                best = next_best
                bad = next_bad
                reductions = next_reductions
            end do
        end do
        close (oracle_unit)
    end if
    if (oracle_only_requested()) stop

    sink = 0.0_dp
    call system_clock(clock_start, clock_rate)
    do repetition = 1, repetitions
        do scenario = 1, scenario_count
            bad = 0
            reductions = 0
            best = initial_best(scenario)
            do observation = 1, observation_count
                call schedules(scenario)%rate_with_metric(observation, base_rate, &
                    metrics(observation, scenario), best, bad, reductions, rate, &
                    next_best, next_bad, next_reductions, improved, reduced, status)
                if (.not. status_ok(status)) error stop &
                    "MLP plateau benchmark: timing evaluation failed"
                sink = sink+rate+real(next_bad+next_reductions, dp)
                best = next_best
                bad = next_bad
                reductions = next_reductions
            end do
        end do
    end do
    call system_clock(clock_end)
    elapsed = real(clock_end-clock_start, dp)/real(clock_rate, dp)/ &
        real(repetitions*scenario_count*observation_count, dp)
    write (*, '(a,es24.16)') "mlp_plateau_schedule_rate_with_metric,", elapsed
    if (.not. ieee_is_finite(sink)) error stop "MLP plateau benchmark: nonfinite sink"

contains

    subroutine write_value(unit, quantity, index, value)
        integer, intent(in) :: unit, index
        character(*), intent(in) :: quantity
        real(dp), intent(in) :: value

        write (unit, '(a,i0,a,es26.17e3)') trim(quantity)//",", index, ",", value
    end subroutine write_value

    logical function oracle_only_requested()
        character(len=16) :: environment_value
        integer :: environment_code

        call get_environment_variable("FORTML_BENCH_ORACLE_ONLY", environment_value, &
            status=environment_code)
        oracle_only_requested = environment_code == 0 .and. trim(environment_value) == "1"
    end function oracle_only_requested

end program fortml_bench_mlp_plateau_schedule
