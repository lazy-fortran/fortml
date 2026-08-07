program fortml_bench_mlp_schedules
    !! Complete-array release workload for built-in MLP schedules.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_status, only: fortnum_status_t, status_ok
    use fortml_mlp_schedules, only: &
        mlp_learning_rate_schedule_t, make_mlp_schedule_constant, &
        make_mlp_schedule_linear_warmup, make_mlp_schedule_cosine_decay, &
        make_mlp_schedule_warmup_cosine, make_mlp_schedule_exponential_decay
    implicit none

    integer, parameter :: schedule_count = 5, update_count = 5
    integer, parameter :: updates(update_count) = [1, 2, 5, 10, 12]
    integer, parameter :: repetitions = 4096
    real(dp), parameter :: base_rate = 0.2_dp
    type(mlp_learning_rate_schedule_t) :: schedules(schedule_count)
    type(fortnum_status_t) :: status
    real(dp) :: rate, d_base, d_min, d_decay, elapsed
    real(dp) :: sink
    integer(int64) :: clock_start, clock_end, clock_rate
    integer :: schedule_index, update_index, repetition, oracle_unit
    integer :: environment_status, index
    character(len=1024) :: oracle_path

    schedules(1) = make_mlp_schedule_constant()
    schedules(2) = make_mlp_schedule_linear_warmup(4)
    schedules(3) = make_mlp_schedule_cosine_decay(10, 0.1_dp)
    schedules(4) = make_mlp_schedule_warmup_cosine(2, 10, 0.2_dp)
    schedules(5) = make_mlp_schedule_exponential_decay(2, 0.8_dp)
    do schedule_index = 1, schedule_count
        if (.not. schedules(schedule_index)%valid()) then
            error stop "MLP schedule benchmark: invalid fixture"
        end if
    end do

    oracle_unit = -1
    call get_environment_variable("FORTML_BENCH_MLP_SCHEDULE_ORACLE", oracle_path, &
        status=environment_status)
    if (environment_status == 0 .and. len_trim(oracle_path) > 0) then
        open (newunit=oracle_unit, file=trim(oracle_path), status="replace", &
            action="write")
        write (oracle_unit, '(a)') "quantity,index,value"
        do schedule_index = 1, schedule_count
            do update_index = 1, update_count
                index = 100*(schedule_index-1) + updates(update_index)
                call schedules(schedule_index)%rate_with_derivatives( &
                    updates(update_index), base_rate, rate, d_base, d_min, d_decay, status)
                if (.not. status_ok(status)) error stop &
                    "MLP schedule benchmark: evaluation failed"
                write (oracle_unit, '(a,i0,a,es26.17e3)') "rate,", index, ",", rate
                write (oracle_unit, '(a,i0,a,es26.17e3)') "d_base_rate,", index, ",", d_base
                write (oracle_unit, '(a,i0,a,es26.17e3)') "d_min_rate_fraction,", index, ",", d_min
                write (oracle_unit, '(a,i0,a,es26.17e3)') "d_decay_factor,", index, ",", d_decay
            end do
        end do
        close (oracle_unit)
    end if
    if (oracle_only_requested()) stop

    sink = 0.0_dp
    call system_clock(clock_start, clock_rate)
    do repetition = 1, repetitions
        do schedule_index = 1, schedule_count
            do update_index = 1, update_count
                call schedules(schedule_index)%rate_with_derivatives( &
                    updates(update_index), base_rate, rate, d_base, d_min, d_decay, status)
                if (.not. status_ok(status)) error stop &
                    "MLP schedule benchmark: timing evaluation failed"
                sink = sink + rate + d_base + d_min + d_decay
            end do
        end do
    end do
    call system_clock(clock_end)
    elapsed = real(clock_end-clock_start, dp)/real(clock_rate, dp)/ &
        real(repetitions*schedule_count*update_count, dp)
    write (*, '(a,es24.16)') "mlp_schedule_rate_with_derivatives,", elapsed
    if (.not. ieee_is_finite(sink)) error stop "MLP schedule benchmark: nonfinite sink"

contains

    logical function oracle_only_requested()
        character(len=16) :: environment_value
        integer :: environment_code

        call get_environment_variable("FORTML_BENCH_ORACLE_ONLY", environment_value, &
            status=environment_code)
        oracle_only_requested = environment_code == 0 .and. trim(environment_value) == "1"
    end function oracle_only_requested

end program fortml_bench_mlp_schedules
