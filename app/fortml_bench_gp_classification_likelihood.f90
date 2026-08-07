program fortml_bench_gp_classification_likelihood
    !! Correctness-gated timings for the binary GP likelihood primitive.
    !!
    !! Each record has the strict scalar release protocol
    !! ``gp_likelihood,<likelihood>,<operation>,<seconds>,<value>``.  The
    !! benchmark harness independently checks every scalar against NumPy;
    !! there is deliberately no GPU timing claim for this host primitive.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortml_gp_classification, only: &
        gp_classification_log_likelihood_value, &
        gp_classification_log_likelihood_jvp, &
        gp_classification_log_likelihood_vjp, &
        GP_LIKELIHOOD_LOGISTIC, GP_LIKELIHOOD_PROBIT
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer, parameter :: n_samples = 4096, repetitions = 48
    real(dp), parameter :: value_bar = 1.7_dp
    real(dp) :: eta(n_samples), eta_dot(n_samples), eta_bar(n_samples)
    integer(int64) :: clock_begin, clock_end, clock_rate
    real(dp) :: value, value_dot, seconds
    type(fortnum_status_t) :: status
    integer :: i

    do i = 1, n_samples
        eta(i) = 2.1_dp*sin(0.013_dp*real(i, dp)) + &
            0.7_dp*cos(0.031_dp*real(i, dp))
        eta_dot(i) = 0.3_dp*cos(0.017_dp*real(i, dp)) - &
            0.2_dp*sin(0.023_dp*real(i, dp))
    end do

    call benchmark_likelihood("logistic", GP_LIKELIHOOD_LOGISTIC, eta, eta_dot, &
        eta_bar, value, value_dot, seconds, status)
    if (.not. status_ok(status)) error stop "GP logistic likelihood benchmark failed"
    call benchmark_likelihood("probit", GP_LIKELIHOOD_PROBIT, eta, eta_dot, &
        eta_bar, value, value_dot, seconds, status)
    if (.not. status_ok(status)) error stop "GP probit likelihood benchmark failed"

contains

    subroutine benchmark_likelihood(name, likelihood, margins, margins_dot, &
            margins_bar, output_value, output_dot, elapsed, final_status)
        character(*), intent(in) :: name
        integer, intent(in) :: likelihood
        real(dp), intent(in) :: margins(:), margins_dot(:)
        real(dp), intent(out) :: margins_bar(:), output_value, output_dot, elapsed
        type(fortnum_status_t), intent(out) :: final_status
        real(dp) :: value_probe, value_dot_probe
        real(dp) :: vjp_norm
        integer :: repetition
        type(fortnum_status_t) :: status

        call gp_classification_log_likelihood_value(margins, likelihood, &
            output_value, status)
        if (.not. status_ok(status)) then
            final_status = status
            return
        end if
        call system_clock(clock_begin, clock_rate)
        do repetition = 1, repetitions
            call gp_classification_log_likelihood_value(margins, likelihood, &
                value_probe, status)
            if (.not. status_ok(status)) then
                final_status = status
                return
            end if
        end do
        call system_clock(clock_end)
        elapsed = real(clock_end - clock_begin, dp)/real(clock_rate, dp)/ &
            real(repetitions, dp)
        write (*, '(a,a,a,es24.16,a,es24.16)') &
            "gp_likelihood,", trim(name), ",value,", elapsed, ",", output_value

        call gp_classification_log_likelihood_jvp(margins, likelihood, margins_dot, &
            output_value, output_dot, status)
        if (.not. status_ok(status)) then
            final_status = status
            return
        end if
        call system_clock(clock_begin, clock_rate)
        do repetition = 1, repetitions
            call gp_classification_log_likelihood_jvp(margins, likelihood, margins_dot, &
                value_probe, value_dot_probe, status)
            if (.not. status_ok(status)) then
                final_status = status
                return
            end if
        end do
        call system_clock(clock_end)
        elapsed = real(clock_end - clock_begin, dp)/real(clock_rate, dp)/ &
            real(repetitions, dp)
        write (*, '(a,a,a,es24.16,a,es24.16)') &
            "gp_likelihood,", trim(name), ",jvp,", elapsed, ",", output_dot

        call gp_classification_log_likelihood_vjp(margins, likelihood, value_bar, &
            margins_bar, status)
        if (.not. status_ok(status)) then
            final_status = status
            return
        end if
        vjp_norm = sqrt(sum(margins_bar*margins_bar))
        call system_clock(clock_begin, clock_rate)
        do repetition = 1, repetitions
            call gp_classification_log_likelihood_vjp(margins, likelihood, value_bar, &
                margins_bar, status)
            if (.not. status_ok(status)) then
                final_status = status
                return
            end if
        end do
        call system_clock(clock_end)
        elapsed = real(clock_end - clock_begin, dp)/real(clock_rate, dp)/ &
            real(repetitions, dp)
        write (*, '(a,a,a,es24.16,a,es24.16)') &
            "gp_likelihood,", trim(name), ",vjp,", elapsed, ",", vjp_norm

        final_status = status
    end subroutine benchmark_likelihood

end program fortml_bench_gp_classification_likelihood
