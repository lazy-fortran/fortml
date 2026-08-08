program fortml_bench_contrastive_loss
    !! Release CPU timing app for pairwise contrastive loss products.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortnum_status, only: fortnum_status_t, status_ok
    use fortml_device, only: FORTML_DEVICE_CUDA
    use fortml_losses, only: contrastive_loss_value, contrastive_loss_jvp, &
        contrastive_loss_vjp, contrastive_loss_hvp, contrastive_loss_value_device
    implicit none

    integer, parameter :: n = 128, d = 16, repetitions = 512
    real(dp) :: embedding_a(n, d), embedding_b(n, d)
    real(dp) :: embedding_a_dot(n, d), embedding_b_dot(n, d)
    real(dp) :: embedding_a_bar(n, d), embedding_b_bar(n, d)
    real(dp) :: embedding_a_hvp(n, d), embedding_b_hvp(n, d)
    real(dp) :: weights(n), value, value_dot, checksum
    real(dp) :: seconds
    integer :: labels(n), i, j, repetition
    integer(int64) :: clock_start, clock_end, clock_rate
    type(fortnum_status_t) :: status

    do i = 1, n
        labels(i) = merge(1, 0, mod(i, 3) == 0)
        weights(i) = 0.5_dp + real(mod(i, 7), dp)/7.0_dp
        do j = 1, d
            embedding_a(i, j) = sin(0.017_dp*real(i*j, dp))
            embedding_b(i, j) = cos(0.013_dp*real(i*j + 2, dp))
            embedding_a_dot(i, j) = 0.03_dp*sin(0.011_dp*real(i + 2*j, dp))
            embedding_b_dot(i, j) = -0.02_dp*cos(0.009_dp*real(2*i + j, dp))
        end do
    end do

    call contrastive_loss_value(embedding_a, embedding_b, labels, 1.25_dp, value, &
        status, weights)
    if (.not. status_ok(status)) error stop "contrastive value warmup failed"
    call system_clock(clock_start, clock_rate)
    do repetition = 1, repetitions
        call contrastive_loss_value(embedding_a, embedding_b, labels, 1.25_dp, value, &
            status, weights)
    end do
    call system_clock(clock_end)
    seconds = real(clock_end - clock_start, dp)/real(clock_rate, dp) &
        /real(repetitions, dp)
    write (*, '(a,",",a,",",a,",",es24.16,",",es24.16)') &
        "contrastive_loss", "value", "cpu", seconds, value

    call contrastive_loss_jvp(embedding_a, embedding_b, labels, 1.25_dp, &
        embedding_a_dot, embedding_b_dot, value, value_dot, status, weights)
    if (.not. status_ok(status)) error stop "contrastive JVP warmup failed"
    call system_clock(clock_start, clock_rate)
    do repetition = 1, repetitions
        call contrastive_loss_jvp(embedding_a, embedding_b, labels, 1.25_dp, &
            embedding_a_dot, embedding_b_dot, value, value_dot, status, weights)
    end do
    call system_clock(clock_end)
    seconds = real(clock_end - clock_start, dp)/real(clock_rate, dp) &
        /real(repetitions, dp)
    write (*, '(a,",",a,",",a,",",es24.16,",",es24.16)') &
        "contrastive_loss", "jvp", "cpu", seconds, value_dot

    call contrastive_loss_vjp(embedding_a, embedding_b, labels, 1.25_dp, 0.7_dp, &
        embedding_a_bar, embedding_b_bar, status, weights)
    if (.not. status_ok(status)) error stop "contrastive VJP warmup failed"
    call system_clock(clock_start, clock_rate)
    do repetition = 1, repetitions
        call contrastive_loss_vjp(embedding_a, embedding_b, labels, 1.25_dp, 0.7_dp, &
            embedding_a_bar, embedding_b_bar, status, weights)
    end do
    call system_clock(clock_end)
    seconds = real(clock_end - clock_start, dp)/real(clock_rate, dp) &
        /real(repetitions, dp)
    write (*, '(a,",",a,",",a,",",es24.16,",",es24.16)') &
        "contrastive_loss", "vjp", "cpu", seconds, sum(embedding_a_bar)

    call contrastive_loss_hvp(embedding_a, embedding_b, labels, 1.25_dp, &
        embedding_a_dot, embedding_b_dot, embedding_a_hvp, embedding_b_hvp, status, &
        weights)
    if (.not. status_ok(status)) error stop "contrastive HVP warmup failed"
    call system_clock(clock_start, clock_rate)
    do repetition = 1, repetitions
        call contrastive_loss_hvp(embedding_a, embedding_b, labels, 1.25_dp, &
            embedding_a_dot, embedding_b_dot, embedding_a_hvp, embedding_b_hvp, &
            status, weights)
    end do
    call system_clock(clock_end)
    seconds = real(clock_end - clock_start, dp)/real(clock_rate, dp) &
        /real(repetitions, dp)
    checksum = sum(embedding_a_hvp) + sum(embedding_b_hvp)
    write (*, '(a,",",a,",",a,",",es24.16,",",es24.16)') &
        "contrastive_loss", "hvp", "cpu", seconds, checksum

    value = 0.0_dp
    call contrastive_loss_value_device(embedding_a, embedding_b, labels, 1.25_dp, &
        value, status, FORTML_DEVICE_CUDA, weights)
    if (status_ok(status)) error stop "contrastive CUDA request was accepted"
    write (*, '(a,",",a,",",a,",",a,",",i0)') "contrastive_loss", "value", &
        "cuda", "refused", status%code
end program fortml_bench_contrastive_loss
