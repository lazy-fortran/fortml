program fortml_bench_gp_ordinal_likelihood
    !! Release timing and checksum application for native ordinal likelihoods.
    use, intrinsic :: iso_fortran_env, only: dp => real64
    use fortnum_status, only: fortnum_status_t, status_ok
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    use fortml_gp_ordinal_classification, only: &
        GP_ORDINAL_LIKELIHOOD_LOGISTIC, GP_ORDINAL_LIKELIHOOD_PROBIT, &
        gp_ordinal_log_likelihood_value, gp_ordinal_log_likelihood_jvp, &
        gp_ordinal_log_likelihood_vjp, gp_ordinal_log_likelihood_hvp, &
        gp_ordinal_likelihood_device_supported, gp_ordinal_log_likelihood_value_device
    implicit none

    real(dp) :: eta(5), eta_dot(5), thresholds(2), thresholds_dot(2)
    real(dp) :: eta_bar(5), thresholds_bar(2), eta_hvp(5), thresholds_hvp(2)
    real(dp) :: value, value_dot, value_bar
    integer :: labels(5), likelihood, i, clock_start, clock_end, clock_rate
    integer :: repetitions
    type(fortnum_status_t) :: status
    type(fortml_device_t) :: cuda
    character(len=16) :: name
    real(dp) :: elapsed

    eta = [-1.2_dp, -0.15_dp, 0.35_dp, 1.1_dp, 0.2_dp]
    eta_dot = [0.17_dp, -0.11_dp, 0.08_dp, 0.13_dp, -0.09_dp]
    thresholds = [-0.45_dp, 0.8_dp]
    thresholds_dot = [0.06_dp, -0.04_dp]
    labels = [1, 2, 3, 2, 1]
    value_bar = 1.7_dp
    repetitions = 100000

    call system_clock(count_rate=clock_rate)
    do likelihood = GP_ORDINAL_LIKELIHOOD_LOGISTIC, GP_ORDINAL_LIKELIHOOD_PROBIT
        if (likelihood == GP_ORDINAL_LIKELIHOOD_LOGISTIC) then
            name = "logistic"
        else
            name = "probit"
        end if
        call system_clock(count=clock_start)
        do i = 1, repetitions
            call gp_ordinal_log_likelihood_value(eta, labels, thresholds, likelihood, &
                value, status)
        end do
        call system_clock(count=clock_end)
        elapsed = real(clock_end - clock_start, dp)/real(clock_rate, dp)/real(repetitions, dp)
        if (.not. status_ok(status)) error stop "ordinal likelihood value failed"
        write (*, '(a,a,a,es24.16,a,es24.16)') &
            "ordinal_likelihood_value,", trim(name), ",seconds,", elapsed, ",value,", value

        call system_clock(count=clock_start)
        do i = 1, repetitions
            call gp_ordinal_log_likelihood_jvp(eta, labels, thresholds, likelihood, eta_dot, &
                thresholds_dot, value, value_dot, status)
        end do
        call system_clock(count=clock_end)
        elapsed = real(clock_end - clock_start, dp)/real(clock_rate, dp)/real(repetitions, dp)
        if (.not. status_ok(status)) error stop "ordinal likelihood JVP failed"
        write (*, '(a,a,a,es24.16,a,es24.16)') &
            "ordinal_likelihood_jvp,", trim(name), ",seconds,", elapsed, ",value_dot,", value_dot

        call system_clock(count=clock_start)
        do i = 1, repetitions
            call gp_ordinal_log_likelihood_vjp(eta, labels, thresholds, likelihood, value_bar, &
                eta_bar, thresholds_bar, status)
        end do
        call system_clock(count=clock_end)
        elapsed = real(clock_end - clock_start, dp)/real(clock_rate, dp)/real(repetitions, dp)
        if (.not. status_ok(status)) error stop "ordinal likelihood VJP failed"
        write (*, '(a,a,a,es24.16,a,es24.16)') &
            "ordinal_likelihood_vjp,", trim(name), ",seconds,", elapsed, ",eta_bar_norm,", &
            sqrt(sum(eta_bar*eta_bar) + sum(thresholds_bar*thresholds_bar))

        call system_clock(count=clock_start)
        do i = 1, repetitions
            call gp_ordinal_log_likelihood_hvp(eta, labels, thresholds, likelihood, value_bar, &
                eta_dot, thresholds_dot, eta_hvp, thresholds_hvp, status)
        end do
        call system_clock(count=clock_end)
        elapsed = real(clock_end - clock_start, dp)/real(clock_rate, dp)/real(repetitions, dp)
        if (.not. status_ok(status)) error stop "ordinal likelihood HVP failed"
        write (*, '(a,a,a,es24.16,a,es24.16)') &
            "ordinal_likelihood_hvp,", trim(name), ",seconds,", elapsed, ",hvp_norm,", &
            sqrt(sum(eta_hvp*eta_hvp) + sum(thresholds_hvp*thresholds_hvp))
    end do
    write (*, '(a,l1)') "ordinal_likelihood_device,cpu,supported,", &
        gp_ordinal_likelihood_device_supported(FORTML_DEVICE_CPU)
    write (*, '(a,l1)') "ordinal_likelihood_device,cuda,supported,", &
        gp_ordinal_likelihood_device_supported(FORTML_DEVICE_CUDA)
    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call gp_ordinal_log_likelihood_value_device(cuda, eta, labels, thresholds, &
        GP_ORDINAL_LIKELIHOOD_LOGISTIC, value, status)
    write (*, '(a,i0)') "ordinal_likelihood_device,cuda,refused,", status%code
end program fortml_bench_gp_ordinal_likelihood
