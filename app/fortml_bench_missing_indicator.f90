program fortml_bench_missing_indicator
    !! Release workload for the dense missing-indicator transformer.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use, intrinsic :: ieee_arithmetic, only: ieee_quiet_nan, ieee_value
    use fortml_missing_indicator, only: missing_indicator_t
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer, parameter :: n_samples = 64, n_features = 6, repetitions = 128
    real(dp) :: x(n_samples, n_features), x_dot(n_samples, n_features)
    real(dp), allocatable :: indicators(:, :), indicators_dot(:, :), x_bar(:, :)
    real(dp), allocatable :: indicator_bar(:, :)
    real(dp) :: transform_seconds, jvp_seconds, vjp_seconds
    integer(int64) :: clock_start, clock_end, clock_rate
    integer :: environment_status, oracle_unit, repetition
    character(len=1024) :: oracle_path
    type(fortnum_status_t) :: status
    type(missing_indicator_t) :: model

    call make_fixture(x, x_dot)
    oracle_unit = -1
    call get_environment_variable("FORTML_BENCH_MISSING_INDICATOR_ORACLE", &
        oracle_path, status=environment_status)
    if (environment_status == 0 .and. len_trim(oracle_path) > 0) then
        open (newunit=oracle_unit, file=trim(oracle_path), status="replace", &
            action="write")
        write (oracle_unit, '(a)') "mode,quantity,row,column,value"
    end if

    call benchmark_mode("all", oracle_unit)
    call benchmark_mode("missing-only", oracle_unit)
    if (oracle_unit /= -1) close (oracle_unit)

contains

    subroutine benchmark_mode(name, unit)
        character(*), intent(in) :: name
        integer, intent(in) :: unit
        integer :: n_outputs, i, j
        type(missing_indicator_t) :: indicator

        call indicator%fit(x, status, features=name)
        if (.not. status_ok(status)) error stop "missing indicator fit failed"
        n_outputs = indicator%output_count()
        allocate(indicators(n_samples, n_outputs), indicators_dot(n_samples, n_outputs), &
            indicator_bar(n_samples, n_outputs), x_bar(n_samples, n_features))
        call indicator%transform(x, indicators, status)
        if (.not. status_ok(status)) error stop "missing indicator transform failed"
        indicator_bar = 0.25_dp
        call indicator%transform_jvp(x, x_dot, indicators_dot, status)
        if (.not. status_ok(status)) error stop "missing indicator JVP failed"
        call indicator%transform_vjp(x, indicator_bar, x_bar, status)
        if (.not. status_ok(status)) error stop "missing indicator VJP failed"
        if (unit /= -1) then
            do j = 1, n_outputs
                do i = 1, n_samples
                    write (unit, '(a,a,a,i0,a,i0,a,es26.17e3)') trim(name), &
                        ",transform,", i, ",", j, ",", indicators(i, j)
                    write (unit, '(a,a,a,i0,a,i0,a,es26.17e3)') trim(name), &
                        ",jvp,", i, ",", j, ",", indicators_dot(i, j)
                end do
            end do
            do j = 1, n_features
                do i = 1, n_samples
                    write (unit, '(a,a,a,i0,a,i0,a,es26.17e3)') trim(name), &
                        ",vjp,", i, ",", j, ",", x_bar(i, j)
                end do
            end do
        end if

        call system_clock(clock_start, clock_rate)
        do i = 1, repetitions
            call indicator%transform(x, indicators, status)
            if (.not. status_ok(status)) error stop "missing indicator timing failed"
        end do
        call system_clock(clock_end)
        transform_seconds = real(clock_end - clock_start, dp)/real(clock_rate, dp) &
            /real(repetitions, dp)

        call system_clock(clock_start, clock_rate)
        do i = 1, repetitions
            call indicator%transform_jvp(x, x_dot, indicators_dot, status)
            if (.not. status_ok(status)) error stop "missing indicator JVP timing failed"
        end do
        call system_clock(clock_end)
        jvp_seconds = real(clock_end - clock_start, dp)/real(clock_rate, dp) &
            /real(repetitions, dp)

        call system_clock(clock_start, clock_rate)
        do i = 1, repetitions
            call indicator%transform_vjp(x, indicator_bar, x_bar, status)
            if (.not. status_ok(status)) error stop "missing indicator VJP timing failed"
        end do
        call system_clock(clock_end)
        vjp_seconds = real(clock_end - clock_start, dp)/real(clock_rate, dp) &
            /real(repetitions, dp)

        write (*, '(a,a,a,i0,a,i0,a,i0,a,3(es24.16,a))') "missing_indicator,", &
            trim(name), ",", n_samples, ",", n_features, ",", n_outputs, ",", &
            transform_seconds, ",", jvp_seconds, ",", vjp_seconds, ""
        deallocate(indicators, indicators_dot, indicator_bar, x_bar)
    end subroutine benchmark_mode

    subroutine make_fixture(values, tangents)
        real(dp), intent(out) :: values(:, :), tangents(:, :)
        real(dp) :: nan
        integer :: i, j

        nan = ieee_value(0.0_dp, ieee_quiet_nan)
        do j = 1, size(values, 2)
            do i = 1, size(values, 1)
                values(i, j) = sin(0.07_dp*real(i, dp) + 0.11_dp*real(j, dp))
                tangents(i, j) = cos(0.03_dp*real(i*j, dp))
            end do
        end do
        values(3::7, 2) = nan
        values(5::11, 5) = nan
    end subroutine make_fixture

end program fortml_bench_missing_indicator
