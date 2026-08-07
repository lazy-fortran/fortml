program fortml_bench_adagrad_training
    !! Release workload for the canonical FortOpt Adagrad recurrence.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortopt_adagrad, only: adagrad_t
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer, parameter :: n_parameters = 4096, steps = 128, repetitions = 16
    real(dp) :: parameter(n_parameters), target(n_parameters), gradient(n_parameters)
    real(dp) :: elapsed, final_norm
    integer(int64) :: clock_start, clock_end, clock_rate
    integer :: index, step, repetition
    type(adagrad_t) :: optimizer
    type(fortnum_status_t) :: status

    do index = 1, n_parameters
        parameter(index) = 0.1_dp*cos(0.003_dp*real(index, dp))
        target(index) = 0.25_dp*sin(0.0017_dp*real(index, dp))
    end do
    call optimizer%initialize(n_parameters, status, learning_rate=1.0e-2_dp, &
        epsilon=1.0e-8_dp)
    if (.not. status_ok(status)) error stop "Adagrad benchmark initialization failed"
    do step = 1, steps
        gradient = parameter - target
        call optimizer%step(parameter, gradient, status)
        if (.not. status_ok(status)) error stop "Adagrad benchmark reference failed"
    end do
    final_norm = sqrt(sum(parameter*parameter))

    call system_clock(clock_start, clock_rate)
    do repetition = 1, repetitions
        do index = 1, n_parameters
            parameter(index) = 0.1_dp*cos(0.003_dp*real(index, dp))
            target(index) = 0.25_dp*sin(0.0017_dp*real(index, dp))
        end do
        call optimizer%initialize(n_parameters, status, learning_rate=1.0e-2_dp, &
            epsilon=1.0e-8_dp)
        if (.not. status_ok(status)) error stop "Adagrad benchmark timing initialization failed"
        do step = 1, steps
            gradient = parameter - target
            call optimizer%step(parameter, gradient, status)
            if (.not. status_ok(status)) error stop "Adagrad benchmark timing failed"
        end do
    end do
    call system_clock(clock_end)
    elapsed = real(clock_end - clock_start, dp)/real(clock_rate, dp) &
        /real(repetitions, dp)
    write (*, '(a,i0,a,i0,a,es24.16,a,es24.16)') &
        "adagrad_training,", n_parameters, ",", steps, ",", final_norm, ",", elapsed
end program fortml_bench_adagrad_training
